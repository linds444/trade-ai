"""Shared frozen prediction and partition validation for evaluation/calibration."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from local_ai_trader.data.validate import validate_candle_timeline
from local_ai_trader.features.build_features import FEATURE_COLUMNS
from local_ai_trader.models.baseline import CLASS_NAMES, encode_labels, feature_matrix, predict_checkpoint
from local_ai_trader.models.train import file_digest
from local_ai_trader.settings import positive_integer


@dataclass(frozen=True)
class FrozenPartition:
    checkpoint: dict
    checkpoint_hash: str
    directory: Path
    split_report: dict
    partition: str
    parquet_hash: str
    frame: pd.DataFrame
    bins: int


def load_frozen_partition(run_directory: Path, partition: str, split_directory: Path | None = None) -> FrozenPartition:
    """Load one audited partition using the saved horizon and metric settings."""
    if partition not in ("validation", "test"):
        raise ValueError("Frozen inference supports validation or test partitions")
    label = partition.title()
    path = run_directory / "checkpoint.json"
    checkpoint_hash = file_digest(path)
    with path.open(encoding="utf-8") as source:
        checkpoint = json.load(source)
    metadata = checkpoint["training_metadata"]
    parameters = metadata["settings"]
    seconds = positive_integer(parameters["candle_seconds"], "candle_seconds")
    horizon = positive_integer(parameters["horizon_steps"], "horizon_steps")
    bins = positive_integer(parameters["calibration_bins"], "calibration_bins")
    directory = Path(metadata["source_split_directory"]) if split_directory is None else split_directory
    for filename in ("split.json", "train.parquet", "validation.parquet"):
        if file_digest(directory / filename) != metadata["input_hashes"][filename]:
            raise ValueError(f"{filename} differs from the frozen training experiment")
    with (directory / "split.json").open(encoding="utf-8") as source:
        report = json.load(source)
    if report.get("feature_names") != list(FEATURE_COLUMNS):
        raise ValueError(f"Incompatible {partition} feature allowlist")
    parquet_path = directory / f"{partition}.parquet"
    parquet_hash = file_digest(parquet_path)
    if partition == "validation" and parquet_hash != metadata["input_hashes"]["validation.parquet"]:
        raise ValueError("Validation differs from the frozen training experiment")
    frame = validate_candle_timeline(pd.read_parquet(parquet_path), seconds)
    feature_matrix(frame)
    encode_labels(frame["target_class"])
    returns = frame["target_return"]
    if not pd.api.types.is_numeric_dtype(returns) or pd.api.types.is_bool_dtype(returns) or not np.isfinite(returns.to_numpy(dtype=float, na_value=np.nan)).all():
        raise ValueError(f"{label} returns must be numeric and finite")
    known_at = frame["target_available_at"]
    if known_at.isna().any() or getattr(known_at.dtype, "tz", None) is None:
        raise ValueError(f"{label} target availability must contain timezone-aware datetimes")
    frame["target_available_at"] = known_at.dt.tz_convert("UTC")
    if not frame["target_available_at"].eq(frame["available_at"] + pd.to_timedelta(horizon * seconds, unit="s")).all():
        raise ValueError(f"{label} labels do not match the frozen horizon")
    if frame["target_available_at"].max() > pd.Timestamp.now(tz="UTC"):
        raise ValueError(f"{label} outcomes are not yet available")
    summary = report["partitions"][partition]
    if len(frame) != summary["rows"]:
        raise ValueError(f"{label} row count does not match the original split")
    for key, actual in (
        ("first_prediction_time", frame["available_at"].iloc[0]),
        ("last_prediction_time", frame["available_at"].iloc[-1]),
        ("last_label_known_at", frame["target_available_at"].iloc[-1]),
    ):
        if pd.Timestamp(summary[key]) != actual:
            raise ValueError(f"{label} {key} does not match the original split")
    if pd.Timestamp(report[f"{partition}_first_prediction_time"]) != frame["available_at"].iloc[0]:
        raise ValueError(f"{label} start does not match the original split boundary")
    if partition == "validation" and not (frame["target_available_at"] < pd.Timestamp(report["test_first_prediction_time"])).all():
        raise ValueError("Validation labels overlap the reserved test partition")
    if summary["class_counts"] != {name: int(frame["target_class"].eq(name).sum()) for name in CLASS_NAMES}:
        raise ValueError(f"{label} class counts do not match the original split")
    if frame["symbol"].iloc[0] != metadata["symbol"] or frame["exchange"].iloc[0] != metadata["exchange"]:
        raise ValueError(f"{label} market does not match the frozen checkpoint")
    return FrozenPartition(checkpoint, checkpoint_hash, directory, report, partition, parquet_hash, frame, bins)


def predict_frozen_models(checkpoint: dict, frame: pd.DataFrame, run_directory: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Restore saved models on CPU; return probabilities and native artifact hashes."""
    kind = checkpoint.get("model_type")
    if kind == "xgboost":
        from local_ai_trader.models.boosting import MODEL_FILENAME, predict_boosting_checkpoint

        values = predict_boosting_checkpoint(checkpoint, frame, run_directory)
    elif kind == "mlp":
        from local_ai_trader.models.mlp import MODEL_FILENAME, predict_mlp_checkpoint

        values = predict_mlp_checkpoint(checkpoint, frame, run_directory)
    elif kind is None:
        return predict_checkpoint(checkpoint, frame), {}
    else:
        raise ValueError("Unsupported frozen checkpoint model type")
    return values, {MODEL_FILENAME: checkpoint["model_sha256"]}


def verify_frozen_inputs(snapshot: FrozenPartition, run_directory: Path, artifacts: dict[str, str]) -> None:
    """Reject changes between loading, prediction and publication."""
    if file_digest(run_directory / "checkpoint.json") != snapshot.checkpoint_hash:
        raise ValueError("Checkpoint changed during evaluation; no results published")
    if file_digest(snapshot.directory / f"{snapshot.partition}.parquet") != snapshot.parquet_hash:
        raise ValueError(f"{snapshot.partition.title()} dataset changed during evaluation; no results published")
    for filename in ("split.json", "train.parquet", "validation.parquet"):
        digest = snapshot.checkpoint["training_metadata"]["input_hashes"][filename]
        if file_digest(snapshot.directory / filename) != digest:
            raise ValueError("Original inputs changed during evaluation; no results published")
    for filename, digest in artifacts.items():
        if file_digest(run_directory / filename) != digest:
            raise ValueError("Model artifact changed during evaluation; no results published")
