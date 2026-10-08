"""Compute a small explicit feature set using only current and older candles."""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from local_ai_trader.data.dataset import TARGET_COLUMNS
from local_ai_trader.data.storage import write_json, write_parquet
from local_ai_trader.data.validate import validate_candle_timeline
from local_ai_trader.settings import Settings, positive_integer

LOGGER = logging.getLogger(__name__)
FEATURE_COLUMNS = (
    "feature_return_1", "feature_momentum", "feature_volatility",
    "feature_close_vs_mean", "feature_volume_ratio", "feature_range",
    "feature_body", "feature_close_location",
)
FEATURE_SCHEMA_VERSION = 1


def build_features(
    candles: pd.DataFrame, candle_seconds: int, momentum_steps: int, rolling_window: int,
) -> pd.DataFrame:
    """Compute trailing features; preserve other columns without using them.

    Current OHLCV is available at available_at. Rolling windows include that
    candle, never a future candle. No scaling or global statistics are fitted.
    Drop only the deterministic leading warmup, then reject any nonfinite value.
    """
    positive_integer(momentum_steps, "momentum_steps")
    positive_integer(rolling_window, "rolling_window")
    if rolling_window < 2:
        raise ValueError("rolling_window must be at least 2")
    if any(column.startswith("feature_") for column in candles.columns):
        raise ValueError("Input already contains features; use the original labelled dataset")
    frame = validate_candle_timeline(candles, candle_seconds)
    warmup = max(momentum_steps, rolling_window)
    if len(frame) <= warmup:
        raise ValueError("Not enough candles for feature warmup")
    for column in ("open", "high", "low", "volume"):
        if column not in frame.columns:
            raise ValueError(f"Missing candle column: {column}")
        if not pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column]):
            raise ValueError(f"{column} must be numeric")
        values = frame[column].to_numpy(dtype=np.float64, na_value=np.nan)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains nonfinite values")
        frame[column] = values
    if frame["volume"].lt(0).any() or frame[["open", "high", "low"]].le(0).any().any():
        raise ValueError("Prices must be positive and volume nonnegative")
    if (frame["low"] > frame[["open", "close"]].min(axis=1)).any() or (
        frame["high"] < frame[["open", "close"]].max(axis=1)
    ).any():
        raise ValueError("Inconsistent OHLC prices")
    close = frame["close"].astype("float64")
    trailing_close = close.rolling(rolling_window, min_periods=rolling_window).mean()
    trailing_volume = frame["volume"].rolling(rolling_window, min_periods=rolling_window).mean()
    width = frame["high"] - frame["low"]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        frame["feature_return_1"] = close / close.shift(1) - 1
        frame["feature_momentum"] = close / close.shift(momentum_steps) - 1
        frame["feature_volatility"] = np.log(close).diff().rolling(
            rolling_window, min_periods=rolling_window,
        ).std(ddof=0)
        frame["feature_close_vs_mean"] = close / trailing_close - 1
        # A completely inactive window has zero relative volume, not infinity.
        frame["feature_volume_ratio"] = frame["volume"] / trailing_volume.mask(trailing_volume.eq(0))
        frame.loc[trailing_volume.eq(0), "feature_volume_ratio"] = 0.0
        frame["feature_range"] = width / close
        frame["feature_body"] = (close - frame["open"]) / frame["open"]
        frame["feature_close_location"] = (close - frame["low"]) / width.mask(width.eq(0))
        frame.loc[width.eq(0), "feature_close_location"] = 0.5
    result = frame.iloc[warmup:].copy().reset_index(drop=True)
    if not np.isfinite(result[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)).all():
        raise ValueError("Feature calculation produced nonfinite values after warmup")
    return result


def create_feature_dataset(source: Path, settings: Settings) -> tuple[Path, dict]:
    """Create a labelled feature snapshot, with an explicit model-input allowlist."""
    if not source.is_file():
        raise ValueError(f"Labelled Parquet file not found: {source}")
    target_report_path = source.with_suffix(".targets.json")
    if not target_report_path.is_file():
        raise ValueError("Target metadata is missing; use output from the targets command")
    with target_report_path.open(encoding="utf-8") as report_file:
        target_report = json.load(report_file)
    for key, expected in (
        ("horizon_steps", settings.horizon_steps), ("candle_seconds", settings.candle_seconds),
        ("flat_return_threshold", settings.flat_return_threshold),
    ):
        if target_report.get(key) != expected:
            raise ValueError(f"Target metadata and current settings disagree on {key}")
    output = settings.processed_dir / "features" / f"{source.stem}_features.parquet"
    report_path = output.with_suffix(".features.json")
    if output.exists() or report_path.exists():
        raise ValueError("Feature output already exists; move the previous snapshot aside before a new run")
    labelled = pd.read_parquet(source)
    if not set(TARGET_COLUMNS).issubset(labelled.columns):
        raise ValueError("Input must contain all target columns")
    if labelled["target_class"].isna().any() or not labelled["target_class"].isin(["up", "down", "flat"]).all():
        raise ValueError("Invalid target classes")
    if not pd.api.types.is_numeric_dtype(labelled["target_return"]):
        raise ValueError("Target returns must be numeric")
    if not np.isfinite(labelled["target_return"].to_numpy(dtype=np.float64, na_value=np.nan)).all():
        raise ValueError("Target returns must be finite")
    if labelled["target_available_at"].isna().any() or getattr(labelled["target_available_at"].dtype, "tz", None) is None:
        raise ValueError("Target availability must contain timezone-aware datetimes")
    result = build_features(labelled, settings.candle_seconds, settings.momentum_steps, settings.rolling_window)
    result["target_available_at"] = result["target_available_at"].dt.tz_convert("UTC")
    label_delay = pd.to_timedelta(settings.horizon_steps * settings.candle_seconds, unit="s")
    if not result["target_available_at"].eq(result["available_at"] + label_delay).all():
        raise ValueError("Target availability does not match the configured horizon")
    if result["target_available_at"].max() > pd.Timestamp.now(tz="UTC"):
        raise ValueError("Target outcomes are not yet available")
    with source.open("rb") as source_file:
        digest = hashlib.file_digest(source_file, "sha256").hexdigest()
    report = {
        "schema_version": FEATURE_SCHEMA_VERSION, "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_path": str(source.resolve()), "source_sha256": digest,
        "target_metadata": target_report,
        "symbol": str(result["symbol"].iloc[0]), "exchange": str(result["exchange"].iloc[0]),
        "source_rows": len(labelled), "feature_rows": len(result),
        "warmup_rows_removed": max(settings.momentum_steps, settings.rolling_window),
        "candle_seconds": settings.candle_seconds, "momentum_steps": settings.momentum_steps,
        "rolling_window": settings.rolling_window, "volatility_ddof": 0,
        "feature_names": list(FEATURE_COLUMNS), "future_only_columns": list(TARGET_COLUMNS),
        "prediction_time": "available_at", "normalization": "none; train-only scaling is a later step",
        "zero_volume_window_ratio": 0.0, "zero_width_close_location": 0.5,
        "class_counts": {label: int(result["target_class"].eq(label).sum()) for label in ("up", "down", "flat")},
    }
    write_parquet(output, result)
    write_json(report_path, report)
    LOGGER.info("Saved %s feature rows to %s; removed %s warmup rows", len(result), output, report["warmup_rows_removed"])
    LOGGER.info("Model input features (%s): %s", len(FEATURE_COLUMNS), ", ".join(FEATURE_COLUMNS))
    return output, report
