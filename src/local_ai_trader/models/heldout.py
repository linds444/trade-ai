"""Evaluate existing JSON checkpoints on their original held-out test partition."""

import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from local_ai_trader.data.storage import write_json, write_parquet
from local_ai_trader.data.validate import validate_candle_timeline
from local_ai_trader.features.build_features import FEATURE_COLUMNS
from local_ai_trader.models.baseline import CLASS_NAMES, encode_labels, feature_matrix, predict_checkpoint
from local_ai_trader.models.evaluate import evaluate_probabilities
from local_ai_trader.models.train import file_digest
from local_ai_trader.settings import positive_integer

LOGGER = logging.getLogger(__name__)


def evaluate_heldout(run_directory: Path, split_directory: Path | None = None) -> tuple[Path, dict]:
    """Use saved weights/scaling only; publish one immutable test evaluation.

    Current settings.toml is intentionally not read: model parameters,
    preprocessing, label horizon and metric binning come from the frozen run.
    An optional split-directory override supports moving a saved bundle without
    relaxing the hashes identifying its training/validation inputs and manifest.
    """
    output = run_directory / "test_evaluation"
    if output.exists():
        raise ValueError("Test evaluation already saved; inspect test_evaluation/metrics.json")
    checkpoint_path = run_directory / "checkpoint.json"
    checkpoint_hash = file_digest(checkpoint_path)
    with checkpoint_path.open(encoding="utf-8") as source:
        checkpoint = json.load(source)
    metadata = checkpoint["training_metadata"]
    parameters = metadata["settings"]
    seconds = positive_integer(parameters["candle_seconds"], "candle_seconds")
    horizon = positive_integer(parameters["horizon_steps"], "horizon_steps")
    bins = positive_integer(parameters["calibration_bins"], "calibration_bins")
    directory = Path(metadata["source_split_directory"]) if split_directory is None else split_directory
    hashes = metadata["input_hashes"]
    for name in ("split.json", "train.parquet", "validation.parquet"):
        if file_digest(directory / name) != hashes[name]:
            raise ValueError(f"{name} differs from the frozen training experiment")
    with (directory / "split.json").open(encoding="utf-8") as source:
        split_report = json.load(source)
    if split_report.get("feature_names") != list(FEATURE_COLUMNS):
        raise ValueError("Incompatible test feature allowlist")
    test_path = directory / "test.parquet"
    test_hash = file_digest(test_path)
    frame = validate_candle_timeline(pd.read_parquet(test_path), seconds)
    feature_matrix(frame)
    encode_labels(frame["target_class"])
    if not pd.api.types.is_numeric_dtype(frame["target_return"]) or pd.api.types.is_bool_dtype(frame["target_return"]) or not np.isfinite(frame["target_return"].to_numpy(dtype=float, na_value=np.nan)).all():
        raise ValueError("Test returns must be numeric and finite")
    known_at = frame["target_available_at"]
    if known_at.isna().any() or getattr(known_at.dtype, "tz", None) is None:
        raise ValueError("Test target availability must contain timezone-aware datetimes")
    frame["target_available_at"] = known_at.dt.tz_convert("UTC")
    if not frame["target_available_at"].eq(frame["available_at"] + pd.to_timedelta(horizon * seconds, unit="s")).all():
        raise ValueError("Test labels do not match the frozen horizon")
    if frame["target_available_at"].max() > pd.Timestamp.now(tz="UTC"):
        raise ValueError("Test outcomes are not yet available")
    summary = split_report["partitions"]["test"]
    if len(frame) != summary["rows"]:
        raise ValueError("Test row count does not match the original split")
    for key, actual in (
        ("first_prediction_time", frame["available_at"].iloc[0]),
        ("last_prediction_time", frame["available_at"].iloc[-1]),
        ("last_label_known_at", frame["target_available_at"].iloc[-1]),
    ):
        if pd.Timestamp(summary[key]) != actual:
            raise ValueError(f"Test {key} does not match the original split")
    if pd.Timestamp(split_report["test_first_prediction_time"]) != frame["available_at"].iloc[0]:
        raise ValueError("Test start does not match the original split boundary")
    if summary["class_counts"] != {label: int(frame["target_class"].eq(label).sum()) for label in CLASS_NAMES}:
        raise ValueError("Test class counts do not match the original split")
    if frame["symbol"].iloc[0] != metadata["symbol"] or frame["exchange"].iloc[0] != metadata["exchange"]:
        raise ValueError("Test market does not match the frozen checkpoint")
    probabilities = predict_checkpoint(checkpoint, frame)
    metrics = {model: evaluate_probabilities(frame["target_class"], values, bins) for model, values in probabilities.items()}
    report = {
        "schema_version": 1, "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "partition": "test", "run_id": metadata["run_id"], "rows": len(frame),
        "checkpoint_sha256": checkpoint_hash, "split_manifest_sha256": hashes["split.json"],
        "test_parquet_sha256": test_hash,
        "split_directory": str(directory.resolve()), "refitted": False,
        "feature_names": list(FEATURE_COLUMNS), "class_names": list(CLASS_NAMES),
        "calibration_status": checkpoint["calibration"], "metrics": metrics,
    }
    if file_digest(checkpoint_path) != checkpoint_hash:
        raise ValueError("Checkpoint changed during evaluation; no results published")
    if file_digest(test_path) != test_hash:
        raise ValueError("Test dataset changed during evaluation; no results published")
    staging = Path(tempfile.mkdtemp(prefix=".test-staging-", dir=run_directory))
    try:
        write_json(staging / "metrics.json", report)
        for model, values in probabilities.items():
            predicted = frame[["timestamp", "available_at", "target_available_at", "symbol", "exchange", "target_class", "target_return"]].copy()
            for index, label in enumerate(CLASS_NAMES):
                predicted[f"p_{label}"] = values[:, index]
            predicted["predicted_class"] = np.array(CLASS_NAMES)[values.argmax(axis=1)]
            predicted["max_probability"] = values.max(axis=1)
            write_parquet(staging / f"{model}.parquet", predicted)
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    LOGGER.info("Evaluated %s held-out test rows using saved parameters; no fitting", len(frame))
    print("Held-out test metrics (probabilities are uncalibrated):")
    print(pd.DataFrame([
        {"model": model, **{key: metrics[model][key] for key in ("accuracy", "f1_macro", "log_loss", "brier_score", "ece")}}
        for model in ("naive", "logistic")
    ]).to_string(index=False))
    LOGGER.info("Saved held-out evaluation to %s", output)
    return output, report
