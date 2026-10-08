"""Train a fixed CPU XGBoost baseline and restore its native JSON model."""

from dataclasses import asdict
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
import xgboost as xgb

from local_ai_trader.data.storage import write_json, write_parquet
from local_ai_trader.features.build_features import FEATURE_COLUMNS
from local_ai_trader.models.baseline import CLASS_NAMES, encode_labels, feature_matrix
from local_ai_trader.models.evaluate import evaluate_probabilities
from local_ai_trader.models.train import file_digest, load_training_partitions
from local_ai_trader.settings import Settings

LOGGER = logging.getLogger(__name__)
MODEL_FILENAME = "xgboost.json"


def model_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the shared allowlist and preserve feature names in the model."""
    return pd.DataFrame(feature_matrix(frame), columns=list(FEATURE_COLUMNS))


def fit_boosting(train: pd.DataFrame, settings: Settings) -> xgb.XGBClassifier:
    """Fit only training rows, without early stopping or a validation eval_set."""
    inputs = model_inputs(train)
    labels = encode_labels(train["target_class"])
    if not np.array_equal(np.unique(labels), np.arange(len(CLASS_NAMES))):
        raise ValueError("XGBoost needs all three training classes; use more history")
    model = xgb.XGBClassifier(
        **asdict(settings.boosting), random_state=settings.random_seed,
        objective="multi:softprob", num_class=len(CLASS_NAMES),
        tree_method="hist", device="cpu", eval_metric="mlogloss",
        subsample=1.0, colsample_bytree=1.0,
    )
    model.fit(inputs, labels)
    return model


def predict_boosting(model: xgb.XGBClassifier, frame: pd.DataFrame) -> np.ndarray:
    """Normalize float32 softmax rounding in float64; this is not calibration."""
    values = np.asarray(model.predict_proba(model_inputs(frame)), dtype=np.float64)
    if (
        values.shape != (len(frame), len(CLASS_NAMES)) or not np.isfinite(values).all()
        or (values < 0).any() or (values > 1).any()
        or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=5e-7)
    ):
        raise ValueError("XGBoost inference produced invalid probabilities")
    return values / values.sum(axis=1, keepdims=True)


def predict_boosting_checkpoint(checkpoint: dict, frame: pd.DataFrame, run_directory: Path) -> dict[str, np.ndarray]:
    """Restore saved native weights and verify their hash and input/output order."""
    if (
        checkpoint.get("schema_version") != 1 or checkpoint.get("model_type") != "xgboost"
        or checkpoint.get("feature_names") != list(FEATURE_COLUMNS)
        or checkpoint.get("class_names") != list(CLASS_NAMES)
        or checkpoint.get("calibration") != "none"
        or checkpoint.get("preprocessing") != "identity"
        or checkpoint.get("probability_postprocessing") != "float64_row_normalization"
        or checkpoint.get("model_file") != MODEL_FILENAME
    ):
        raise ValueError("Incompatible XGBoost checkpoint schema, preprocessing or feature/class order")
    path = run_directory / MODEL_FILENAME
    if file_digest(path) != checkpoint["model_sha256"]:
        raise ValueError("XGBoost model differs from the frozen checkpoint")
    model = xgb.XGBClassifier(n_jobs=1, device="cpu")
    model.load_model(path)
    booster = model.get_booster()
    learner = json.loads(booster.save_config())["learner"]
    if (
        booster.feature_names != list(FEATURE_COLUMNS)
        or booster.num_features() != len(FEATURE_COLUMNS)
        or learner["objective"]["name"] != "multi:softprob"
        or int(learner["learner_model_param"]["num_class"]) != len(CLASS_NAMES)
        or not np.array_equal(model.classes_, np.arange(len(CLASS_NAMES)))
    ):
        raise ValueError("Saved XGBoost model has incompatible features, objective or classes")
    values = predict_boosting(model, frame)
    if file_digest(path) != checkpoint["model_sha256"]:
        raise ValueError("XGBoost model changed during inference")
    return {"xgboost": values}


def train_boosting_experiment(directory: Path, settings: Settings) -> tuple[Path, dict]:
    """Read train/validation only; atomically publish weights, metrics and provenance."""
    started = time.perf_counter()
    input_hashes = {name: file_digest(directory / name) for name in ("split.json", "train.parquet", "validation.parquet")}
    tables, split_report = load_training_partitions(directory, settings)
    fitted = fit_boosting(tables["train"], settings)
    timestamp = pd.Timestamp.now(tz="UTC")
    symbol = str(tables["train"]["symbol"].iloc[0])
    run_id = f"boosting_{symbol}_{timestamp.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    output = settings.models_dir / run_id
    staging = Path(tempfile.mkdtemp(prefix=".boosting-staging-", dir=settings.models_dir))
    try:
        fitted.save_model(staging / MODEL_FILENAME)
        checkpoint = {
            "schema_version": 1, "model_type": "xgboost", "calibration": "none",
            "preprocessing": "identity", "feature_names": list(FEATURE_COLUMNS),
            "class_names": list(CLASS_NAMES), "model_file": MODEL_FILENAME,
            "model_sha256": file_digest(staging / MODEL_FILENAME),
            "probability_postprocessing": "float64_row_normalization",
        }
        metrics = {"test": {"evaluated": False}}
        for partition, frame in tables.items():
            values = predict_boosting(fitted, frame)
            restored = predict_boosting_checkpoint(checkpoint, frame, staging)["xgboost"]
            np.testing.assert_allclose(restored, values, rtol=1e-7, atol=1e-8)
            metrics[partition] = {"xgboost": evaluate_probabilities(frame["target_class"], values, settings.calibration_bins)}
            if partition == "validation":
                predicted = frame[["timestamp", "available_at", "target_available_at", "symbol", "exchange", "target_class", "target_return"]].copy()
                for index, label in enumerate(CLASS_NAMES):
                    predicted[f"p_{label}"] = values[:, index]
                predicted["predicted_class"] = np.array(CLASS_NAMES)[values.argmax(axis=1)]
                predicted["max_probability"] = values.max(axis=1)
                write_parquet(staging / "validation_xgboost.parquet", predicted)
        parameters = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(settings).items()}
        model_parameters = fitted.get_params()
        # XGBoost's default missing-value sentinel is NaN. Record it as text
        # while preserving strict JSON; feature validation rejects missing data.
        model_parameters["missing"] = "NaN"
        experiment = {
            "schema_version": 1, "run_id": run_id, "created_at": timestamp.isoformat(),
            "model_types": ["xgboost_hist_classifier"], "symbol": symbol,
            "exchange": str(tables["train"]["exchange"].iloc[0]),
            "settings": parameters, "model_parameters": model_parameters,
            "feature_names": list(FEATURE_COLUMNS), "class_names": list(CLASS_NAMES),
            "source_split_directory": str(directory.resolve()), "split_metadata": split_report,
            "input_hashes": input_hashes, "random_seed": settings.random_seed,
            "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "xgboost": xgb.__version__},
            "elapsed_seconds": time.perf_counter() - started, "test_data_opened": False,
            "validation_used_for_fitting": False, "early_stopping": False,
            "calibration": "none", "preprocessing": "identity", "device": "cpu",
            "metrics_path": "metrics.json", "checkpoint_path": "checkpoint.json",
            "prediction_files": {"xgboost": "validation_xgboost.parquet"},
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
        "model": "xgboost", **{name: metrics["validation"]["xgboost"][name] for name in ("accuracy", "f1_macro", "log_loss", "brier_score", "ece")},
    }]).to_string(index=False))
    LOGGER.info("Saved experiment and checkpoint to %s", output)
    return output, metrics
