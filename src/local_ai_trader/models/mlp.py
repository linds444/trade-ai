"""Small PyTorch classifier with training-only normalization and fixed epochs."""

from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
import uuid

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from local_ai_trader.data.storage import write_json, write_parquet
from local_ai_trader.features.build_features import FEATURE_COLUMNS
from local_ai_trader.models.baseline import CLASS_NAMES, encode_labels, feature_matrix
from local_ai_trader.models.evaluate import evaluate_probabilities
from local_ai_trader.models.train import file_digest, load_training_partitions
from local_ai_trader.settings import Settings, load_mlp_settings, positive_integer

LOGGER = logging.getLogger(__name__)
MODEL_FILENAME = "mlp_state.pt"


class MarketMLP(nn.Module):
    """Map ordered standardized features to three uncalibrated class logits."""

    def __init__(self, hidden_sizes: tuple[int, ...], dropout: float):
        super().__init__()
        layers = []
        width = len(FEATURE_COLUMNS)
        for size in hidden_sizes:
            layers.extend((nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)))
            width = size
        layers.append(nn.Linear(width, len(CLASS_NAMES)))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def training_device(name: str) -> torch.device:
    if name not in ("cpu", "cuda"):
        raise ValueError("MLP device must be cpu or cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use the verified CUDA Python environment or explicitly set mlp.device = 'cpu'")
    return torch.device(name)


def standardized_inputs(frame: pd.DataFrame, mean: np.ndarray, scale: np.ndarray) -> torch.Tensor:
    values = (feature_matrix(frame) - mean) / scale
    with np.errstate(over="ignore"):
        values = values.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("MLP standardized inputs must be finite float32 values")
    return torch.from_numpy(values)


def predict_model(model: MarketMLP, inputs: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    """Use evaluation mode and float64 softmax, without fitting or AMP inference."""
    positive_integer(batch_size, "batch_size")
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            logits = model(inputs[start:start + batch_size].to(device))
            if not torch.isfinite(logits).all():
                raise ValueError("MLP inference produced nonfinite logits")
            chunks.append(torch.softmax(logits.to(dtype=torch.float64), dim=1).cpu().numpy())
    return np.concatenate(chunks)


@dataclass
class FittedMLP:
    model: MarketMLP
    mean: np.ndarray
    scale: np.ndarray
    variance: np.ndarray
    device: torch.device
    batch_size: int
    history: list[dict]
    hardware: dict

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return predict_model(self.model, standardized_inputs(frame, self.mean, self.scale), self.device, self.batch_size)


def fit_mlp(train: pd.DataFrame, settings: Settings) -> FittedMLP:
    """Normalize and optimize on training rows only, in chronological batches."""
    config = settings.mlp
    values = feature_matrix(train)
    labels = encode_labels(train["target_class"])
    if not np.array_equal(np.unique(labels), np.arange(len(CLASS_NAMES))):
        raise ValueError("MLP needs all three training classes; use more history")
    device = training_device(config.device)
    amp_enabled = config.mixed_precision and device.type == "cuda"
    # Required for deterministic CUDA matrix multiplication, before GPU work.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(settings.random_seed)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(settings.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats(device)
    normalizer = StandardScaler().fit(values)
    mean, variance, scale = normalizer.mean_, normalizer.var_, normalizer.scale_
    inputs = standardized_inputs(train, mean, scale)
    targets = torch.from_numpy(labels)
    model = MarketMLP(config.hidden_sizes, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history = []
    for epoch in range(config.epochs):
        model.train()
        loss_sum = 0.0
        skipped_steps = 0
        for start in range(0, len(inputs), config.batch_size):
            batch = inputs[start:start + config.batch_size].to(device)
            truth = targets[start:start + config.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                loss = criterion(model(batch), truth)
            if not torch.isfinite(loss):
                raise ValueError("MLP training produced a nonfinite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm, error_if_nonfinite=not amp_enabled)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped_steps += int(scaler.get_scale() < scale_before)
            loss_sum += float(loss.detach().item()) * len(batch)
        history.append({"epoch": epoch + 1, "training_loss": loss_sum / len(inputs), "skipped_optimizer_steps": skipped_steps})
        LOGGER.info("Epoch %s/%s: training loss %.6f; skipped optimizer steps %s", epoch + 1, config.epochs, history[-1]["training_loss"], skipped_steps)
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("MLP weights are nonfinite; no model published")
    hardware = {"training_device": str(device), "mixed_precision": amp_enabled, "deterministic_algorithms": True}
    if device.type == "cuda":
        hardware.update({"gpu_name": torch.cuda.get_device_name(device), "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)})
    LOGGER.info("MLP training device: %s; mixed precision: %s", device, amp_enabled)
    return FittedMLP(model, mean, scale, variance, device, config.batch_size, history, hardware)


def predict_mlp_checkpoint(checkpoint: dict, frame: pd.DataFrame, run_directory: Path, device: str = "cpu") -> dict[str, np.ndarray]:
    """Load only tensors and restore the exact network and training statistics."""
    if (
        checkpoint.get("schema_version") != 1 or checkpoint.get("model_type") != "mlp"
        or checkpoint.get("feature_names") != list(FEATURE_COLUMNS)
        or checkpoint.get("class_names") != list(CLASS_NAMES)
        or checkpoint.get("calibration") != "none"
        or checkpoint.get("preprocessing") != "training_population_standardization"
        or checkpoint.get("probability_postprocessing") != "float64_softmax"
        or checkpoint.get("model_file") != MODEL_FILENAME
    ):
        raise ValueError("Incompatible MLP checkpoint schema, preprocessing or feature/class order")
    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float64)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float64)
    if mean.shape != (len(FEATURE_COLUMNS),) or scale.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Invalid MLP normalization parameters")
    config = load_mlp_settings(checkpoint["model_config"])
    target_device = training_device(device)
    torch.set_num_threads(config.cpu_threads)
    path = run_directory / MODEL_FILENAME
    if file_digest(path) != checkpoint["model_sha256"]:
        raise ValueError("MLP weights differ from the frozen checkpoint")
    model = MarketMLP(config.hidden_sizes, config.dropout)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    values = predict_model(model.to(target_device), standardized_inputs(frame, mean, scale), target_device, config.batch_size)
    if file_digest(path) != checkpoint["model_sha256"]:
        raise ValueError("MLP weights changed during inference")
    return {"mlp": values}


def train_mlp_experiment(directory: Path, settings: Settings) -> tuple[Path, dict]:
    """Publish a complete fixed-epoch experiment without reading test Parquet."""
    started = time.perf_counter()
    input_hashes = {name: file_digest(directory / name) for name in ("split.json", "train.parquet", "validation.parquet")}
    tables, split_report = load_training_partitions(directory, settings)
    fitted = fit_mlp(tables["train"], settings)
    timestamp = pd.Timestamp.now(tz="UTC")
    symbol = str(tables["train"]["symbol"].iloc[0])
    run_id = f"mlp_{symbol}_{timestamp.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    output = settings.models_dir / run_id
    staging = Path(tempfile.mkdtemp(prefix=".mlp-staging-", dir=settings.models_dir))
    try:
        torch.save({name: value.detach().cpu() for name, value in fitted.model.state_dict().items()}, staging / MODEL_FILENAME)
        checkpoint = {
            "schema_version": 1, "model_type": "mlp", "calibration": "none",
            "preprocessing": "training_population_standardization", "probability_postprocessing": "float64_softmax",
            "feature_names": list(FEATURE_COLUMNS), "class_names": list(CLASS_NAMES),
            "model_file": MODEL_FILENAME, "model_sha256": file_digest(staging / MODEL_FILENAME),
            "model_config": asdict(settings.mlp), "scaler_mean": fitted.mean.tolist(),
            "scaler_scale": fitted.scale.tolist(), "scaler_variance": fitted.variance.tolist(),
            "training_rows": len(tables["train"]),
        }
        metrics = {"test": {"evaluated": False}}
        for partition, frame in tables.items():
            values = fitted.predict(frame)
            restored = predict_mlp_checkpoint(checkpoint, frame, staging, device=str(fitted.device))["mlp"]
            np.testing.assert_allclose(restored, values, rtol=1e-7, atol=1e-8)
            metrics[partition] = {"mlp": evaluate_probabilities(frame["target_class"], values, settings.calibration_bins)}
            if partition == "validation":
                predicted = frame[["timestamp", "available_at", "target_available_at", "symbol", "exchange", "target_class", "target_return"]].copy()
                for index, label in enumerate(CLASS_NAMES):
                    predicted[f"p_{label}"] = values[:, index]
                predicted["predicted_class"] = np.array(CLASS_NAMES)[values.argmax(axis=1)]
                predicted["max_probability"] = values.max(axis=1)
                write_parquet(staging / "validation_mlp.parquet", predicted)
        parameters = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(settings).items()}
        experiment = {
            "schema_version": 1, "run_id": run_id, "created_at": timestamp.isoformat(),
            "model_types": ["pytorch_mlp"], "symbol": symbol, "exchange": str(tables["train"]["exchange"].iloc[0]),
            "settings": parameters, "feature_names": list(FEATURE_COLUMNS), "class_names": list(CLASS_NAMES),
            "source_split_directory": str(directory.resolve()), "split_metadata": split_report, "input_hashes": input_hashes,
            "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "torch": torch.__version__, "torch_cuda_build": torch.version.cuda},
            "random_seed": settings.random_seed, "elapsed_seconds": time.perf_counter() - started,
            "training_history": fitted.history, "hardware": fitted.hardware,
            "trainable_parameters": sum(parameter.numel() for parameter in fitted.model.parameters()),
            "validation_used_for_fitting": False, "early_stopping": False, "batch_order": "chronological",
            "test_data_opened": False, "calibration": "none", "metrics_path": "metrics.json",
            "checkpoint_path": "checkpoint.json", "prediction_files": {"mlp": "validation_mlp.parquet"},
        }
        checkpoint["training_metadata"] = experiment
        for name, digest in input_hashes.items():
            if file_digest(directory / name) != digest:
                raise ValueError("Training inputs changed during fitting; no experiment published")
        write_json(staging / "checkpoint.json", checkpoint)
        write_json(staging / "experiment.json", experiment)
        write_json(staging / "metrics.json", metrics)
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    LOGGER.info("Fitted on %s training rows; evaluated %s validation rows", len(tables["train"]), len(tables["validation"]))
    print("Validation metrics (probabilities are uncalibrated):")
    print(pd.DataFrame([{
        "model": "mlp", **{name: metrics["validation"]["mlp"][name] for name in ("accuracy", "f1_macro", "log_loss", "brier_score", "ece")},
    }]).to_string(index=False))
    LOGGER.info("Saved experiment and checkpoint to %s", output)
    return output, metrics
