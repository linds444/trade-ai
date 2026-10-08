"""Fit baseline models and record validation experiments without opening test data."""

from dataclasses import asdict
import hashlib
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

from local_ai_trader.data.storage import write_json, write_parquet
from local_ai_trader.data.validate import validate_candle_timeline
from local_ai_trader.features.build_features import FEATURE_COLUMNS
from local_ai_trader.models.baseline import CLASS_NAMES, encode_labels, feature_matrix, fit_baselines, predict_checkpoint
from local_ai_trader.models.evaluate import evaluate_probabilities
from local_ai_trader.settings import Settings

LOGGER = logging.getLogger(__name__)


def file_digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_training_partitions(directory: Path, settings: Settings) -> tuple[dict[str, pd.DataFrame], dict]:
    with (directory / "split.json").open(encoding="utf-8") as source:
        report = json.load(source)
    if not isinstance(report, dict) or report.get("schema_version") != 1 or report.get("feature_names") != list(FEATURE_COLUMNS):
        raise ValueError("Incompatible split metadata or model-input order")
    if report.get("purge_rule") != "target_available_at < next_partition_first_available_at":
        raise ValueError("Split metadata does not establish strict label purging")
    for name in ("train_fraction", "validation_fraction"):
        if report.get(name) != getattr(settings, name):
            raise ValueError(f"Split metadata and current settings disagree on {name}")
    feature_report = report["feature_metadata"]
    for name in ("candle_seconds", "momentum_steps", "rolling_window"):
        if feature_report.get(name) != getattr(settings, name):
            raise ValueError(f"Feature metadata and current settings disagree on {name}")
    for name in ("horizon_steps", "flat_return_threshold"):
        if feature_report["target_metadata"].get(name) != getattr(settings, name):
            raise ValueError(f"Target metadata and current settings disagree on {name}")
    tables = {}
    delay = pd.to_timedelta(settings.horizon_steps * settings.candle_seconds, unit="s")
    # The held-out test file is deliberately not opened, including for hashing.
    for name in ("train", "validation"):
        frame = validate_candle_timeline(pd.read_parquet(directory / f"{name}.parquet"), settings.candle_seconds)
        feature_matrix(frame)
        encode_labels(frame["target_class"])
        if not pd.api.types.is_numeric_dtype(frame["target_return"]) or not np.isfinite(frame["target_return"].to_numpy(dtype=float, na_value=np.nan)).all():
            raise ValueError("Target returns must be numeric and finite")
        if frame["target_available_at"].isna().any() or getattr(frame["target_available_at"].dtype, "tz", None) is None:
            raise ValueError("Target availability must contain timezone-aware datetimes")
        frame["target_available_at"] = frame["target_available_at"].dt.tz_convert("UTC")
        if not frame["target_available_at"].eq(frame["available_at"] + delay).all():
            raise ValueError("Label availability does not match the configured horizon")
        summary = report["partitions"][name]
        if summary["rows"] != len(frame):
            raise ValueError(f"{name} row count does not match split metadata")
        for key, actual in (
            ("first_prediction_time", frame["available_at"].iloc[0]),
            ("last_prediction_time", frame["available_at"].iloc[-1]),
            ("last_label_known_at", frame["target_available_at"].iloc[-1]),
        ):
            if pd.Timestamp(summary[key]) != actual:
                raise ValueError(f"{name} {key} does not match split metadata")
        if summary["class_counts"] != {label: int(frame["target_class"].eq(label).sum()) for label in CLASS_NAMES}:
            raise ValueError(f"{name} class counts do not match split metadata")
        tables[name] = frame
    train, validation = tables["train"], tables["validation"]
    if train["symbol"].iloc[0] != validation["symbol"].iloc[0] or train["exchange"].iloc[0] != validation["exchange"].iloc[0]:
        raise ValueError("Train and validation must use the same market")
    validation_start = validation["available_at"].iloc[0]
    test_start = pd.Timestamp(report["test_first_prediction_time"])
    if test_start.tzinfo is None or pd.Timestamp(report["validation_first_prediction_time"]) != validation_start:
        raise ValueError("Invalid partition boundary metadata")
    if not (train["target_available_at"] < validation_start).all() or not (validation["target_available_at"] < test_start).all():
        raise ValueError("Overlapping labels violate the split boundary")
    if validation["target_available_at"].max() > pd.Timestamp.now(tz="UTC"):
        raise ValueError("Validation outcomes are not yet available")
    return tables, report


def train_baseline_experiment(directory: Path, settings: Settings) -> tuple[Path, dict]:
    started = time.perf_counter()
    tables, split_report = load_training_partitions(directory, settings)
    fitted = fit_baselines(tables["train"], settings)
    checkpoint = fitted.checkpoint()
    metrics = {}
    predictions = {}
    for name, frame in tables.items():
        probabilities = fitted.predict(frame)
        restored = predict_checkpoint(checkpoint, frame)
        metrics[name] = {}
        for model in ("naive", "logistic"):
            np.testing.assert_allclose(restored[model], probabilities[model], rtol=1e-10, atol=1e-12)
            metrics[name][model] = evaluate_probabilities(frame["target_class"], probabilities[model], settings.calibration_bins)
            if name == "validation":
                result = frame[["timestamp", "available_at", "target_available_at", "symbol", "exchange", "target_class", "target_return"]].copy()
                for index, label in enumerate(CLASS_NAMES):
                    result[f"p_{label}"] = probabilities[model][:, index]
                result["predicted_class"] = np.array(CLASS_NAMES)[probabilities[model].argmax(axis=1)]
                result["max_probability"] = probabilities[model].max(axis=1)
                predictions[model] = result
    metrics["test"] = {"evaluated": False}
    symbol = str(tables["train"]["symbol"].iloc[0])
    timestamp = pd.Timestamp.now(tz="UTC")
    run_id = f"baseline_{symbol}_{timestamp.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    parameters = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(settings).items()}
    experiment = {
        "schema_version": 1, "run_id": run_id, "created_at": timestamp.isoformat(),
        "model_types": ["training_class_frequencies", "scaled_logistic_regression"],
        "symbol": symbol, "exchange": str(tables["train"]["exchange"].iloc[0]),
        "settings": parameters, "feature_names": list(FEATURE_COLUMNS), "class_names": list(CLASS_NAMES),
        "source_split_directory": str(directory.resolve()), "split_metadata": split_report,
        "input_hashes": {name: file_digest(directory / name) for name in ("split.json", "train.parquet", "validation.parquet")},
        "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "random_seed": settings.random_seed, "elapsed_seconds": time.perf_counter() - started,
        "test_data_opened": False, "calibration": "none", "metrics_path": "metrics.json",
        "checkpoint_path": "checkpoint.json", "prediction_files": {model: f"validation_{model}.parquet" for model in predictions},
    }
    checkpoint["training_metadata"] = experiment
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    output = settings.models_dir / run_id
    staging = Path(tempfile.mkdtemp(prefix=".baseline-staging-", dir=settings.models_dir))
    try:
        write_json(staging / "checkpoint.json", checkpoint)
        write_json(staging / "metrics.json", metrics)
        write_json(staging / "experiment.json", experiment)
        for model, frame in predictions.items():
            write_parquet(staging / f"validation_{model}.parquet", frame)
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    LOGGER.info("Fitted on %s training rows; evaluated %s validation rows", len(tables["train"]), len(tables["validation"]))
    print("Validation metrics (probabilities are uncalibrated):")
    print(pd.DataFrame([
        {"model": model, **{key: metrics["validation"][model][key] for key in ("accuracy", "f1_macro", "log_loss", "brier_score", "ece")}}
        for model in ("naive", "logistic")
    ]).to_string(index=False))
    LOGGER.info("Saved experiment and checkpoint to %s", output)
    return output, metrics
