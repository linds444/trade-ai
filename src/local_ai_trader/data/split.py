"""Chronological holdouts with strict label-availability boundaries."""

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from local_ai_trader.data.dataset import TARGET_COLUMNS
from local_ai_trader.data.storage import write_json, write_parquet
from local_ai_trader.data.validate import validate_candle_timeline
from local_ai_trader.features.build_features import FEATURE_COLUMNS
from local_ai_trader.settings import Settings, positive_integer, validate_split_fractions

LOGGER = logging.getLogger(__name__)


def validate_feature_dataset(features: pd.DataFrame, candle_seconds: int, horizon_steps: int) -> pd.DataFrame:
    """Check finite causal inputs and timed labels before any partitioning."""
    positive_integer(horizon_steps, "horizon_steps")
    required = set(FEATURE_COLUMNS) | set(TARGET_COLUMNS)
    if not required.issubset(features.columns):
        raise ValueError(f"Missing labelled feature columns: {sorted(required - set(features.columns))}")
    frame = validate_candle_timeline(features, candle_seconds)
    for column in (*FEATURE_COLUMNS, "target_return"):
        if not pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column]):
            raise ValueError(f"{column} must be numeric")
        if not np.isfinite(frame[column].to_numpy(dtype=np.float64, na_value=np.nan)).all():
            raise ValueError(f"{column} must contain finite values")
    if frame["target_class"].isna().any() or not frame["target_class"].isin(["up", "down", "flat"]).all():
        raise ValueError("Invalid target classes")
    label_times = frame["target_available_at"]
    if label_times.isna().any() or getattr(label_times.dtype, "tz", None) is None:
        raise ValueError("Target availability must contain timezone-aware datetimes")
    frame["target_available_at"] = label_times.dt.tz_convert("UTC")
    delay = pd.to_timedelta(horizon_steps * candle_seconds, unit="s")
    if not frame["target_available_at"].eq(frame["available_at"] + delay).all():
        raise ValueError("Target availability does not match the configured horizon")
    if frame["target_available_at"].max() > pd.Timestamp.now(tz="UTC"):
        raise ValueError("Target outcomes are not yet available")
    return frame


def chronological_split(
    features: pd.DataFrame, candle_seconds: int, horizon_steps: int,
    train_fraction: float, validation_fraction: float,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Partition by time, then drop labels known at or after the next split starts.

    Cut positions use floor(N * train_fraction) and
    floor(N * (train_fraction + validation_fraction)). No shuffling occurs.
    Validation/test may use past candle history, but their outcomes must not
    enter earlier training or model-selection periods.
    """
    train_fraction, validation_fraction = validate_split_fractions(train_fraction, validation_fraction)
    frame = validate_feature_dataset(features, candle_seconds, horizon_steps)
    train_end = int(len(frame) * train_fraction)
    validation_end = int(len(frame) * (train_fraction + validation_fraction))
    if not 0 < train_end < validation_end < len(frame):
        raise ValueError("Not enough data for three chronological partitions")
    validation_start = frame["available_at"].iloc[train_end]
    test_start = frame["available_at"].iloc[validation_end]
    raw = {
        "train": frame.iloc[:train_end],
        "validation": frame.iloc[train_end:validation_end],
        "test": frame.iloc[validation_end:],
    }
    boundaries = {"train": validation_start, "validation": test_start}
    partitions = {}
    summaries = {}
    for name, data in raw.items():
        keep = pd.Series(True, index=data.index)
        if name in boundaries:
            keep = data["target_available_at"] < boundaries[name]
        selected = data.loc[keep].copy().reset_index(drop=True)
        if selected.empty:
            raise ValueError(f"{name} is empty after label purging; use a longer dataset")
        partitions[name] = selected
        summaries[name] = {
            "raw_rows": len(data), "rows": len(selected), "purged_rows": int((~keep).sum()),
            "purged_prediction_times": [stamp.isoformat() for stamp in data.loc[~keep, "available_at"]],
            "first_prediction_time": selected["available_at"].iloc[0].isoformat(),
            "last_prediction_time": selected["available_at"].iloc[-1].isoformat(),
            "last_label_known_at": selected["target_available_at"].iloc[-1].isoformat(),
            "class_counts": {label: int(selected["target_class"].eq(label).sum()) for label in ("up", "down", "flat")},
        }
    report = {
        "source_rows": len(frame), "train_fraction": train_fraction,
        "validation_fraction": validation_fraction, "test_fraction": 1 - train_fraction - validation_fraction,
        "cut_positions": {"validation_start": train_end, "test_start": validation_end},
        "purge_rule": "target_available_at < next_partition_first_available_at",
        "validation_first_prediction_time": validation_start.isoformat(),
        "test_first_prediction_time": test_start.isoformat(), "partitions": summaries,
    }
    return partitions, report


def create_split_dataset(source: Path, settings: Settings) -> tuple[Path, dict]:
    """Publish all split files together from a staging directory on the same disk."""
    if not source.is_file():
        raise ValueError(f"Feature Parquet file not found: {source}")
    feature_report_path = source.with_suffix(".features.json")
    if not feature_report_path.is_file():
        raise ValueError("Feature metadata is missing; use output from the features command")
    with feature_report_path.open(encoding="utf-8") as report_file:
        feature_report = json.load(report_file)
    if not isinstance(feature_report, dict) or feature_report.get("feature_names") != list(FEATURE_COLUMNS):
        raise ValueError("Feature metadata has an incompatible model-input allowlist")
    for key, expected in (("candle_seconds", settings.candle_seconds), ("momentum_steps", settings.momentum_steps), ("rolling_window", settings.rolling_window)):
        if feature_report.get(key) != expected:
            raise ValueError(f"Feature metadata and current settings disagree on {key}")
    target_report = feature_report.get("target_metadata")
    if not isinstance(target_report, dict):
        raise ValueError("Feature metadata is missing target provenance")
    for key, expected in (("horizon_steps", settings.horizon_steps), ("flat_return_threshold", settings.flat_return_threshold)):
        if target_report.get(key) != expected:
            raise ValueError(f"Target metadata and current settings disagree on {key}")
    destination = settings.processed_dir / "splits" / source.stem
    if destination.exists():
        raise ValueError("Split output already exists; move the previous snapshot aside before a new run")
    frame = pd.read_parquet(source)
    if feature_report.get("feature_rows") != len(frame):
        raise ValueError("Feature metadata row count disagrees with the source file")
    partitions, split_report = chronological_split(
        frame, settings.candle_seconds, settings.horizon_steps,
        settings.train_fraction, settings.validation_fraction,
    )
    with source.open("rb") as source_file:
        digest = hashlib.file_digest(source_file, "sha256").hexdigest()
    report = {
        "schema_version": 1, "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_path": str(source.resolve()), "source_sha256": digest,
        "feature_names": list(FEATURE_COLUMNS), "future_only_columns": list(TARGET_COLUMNS),
        "feature_metadata": feature_report, **split_report,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".split-staging-", dir=destination.parent))
    try:
        for name, partition in partitions.items():
            write_parquet(staging / f"{name}.parquet", partition)
        write_json(staging / "split.json", report)
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    for name, summary in report["partitions"].items():
        LOGGER.info("%s: %s rows; purged %s overlapping labels", name, summary["rows"], summary["purged_rows"])
    LOGGER.info("Saved chronological splits to %s", destination)
    return destination, report
