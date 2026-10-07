"""Forward-return targets, kept explicitly separate from model inputs."""

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from local_ai_trader.data.storage import write_json, write_parquet
from local_ai_trader.data.validate import validate_candle_timeline
from local_ai_trader.settings import Settings, positive_integer

LOGGER = logging.getLogger(__name__)
TARGET_COLUMNS = ("target_return", "target_class", "target_available_at")
FLAT_BOUNDARY_TOLERANCE = 8 * np.finfo(np.float64).eps


def build_targets(
    candles: pd.DataFrame, horizon_steps: int, candle_seconds: int,
    flat_return_threshold: float,
) -> pd.DataFrame:
    """Label close[t+h]/close[t]-1, known at available_at[t+h].

    A prediction at available_at[t] has a horizon of h whole intervals. Never
    feed target_* columns into a prediction model. The final h rows are removed
    because their future closes are absent, rather than labelling them as flat.
    """
    positive_integer(horizon_steps, "horizon_steps")
    positive_integer(candle_seconds, "candle_seconds")
    if isinstance(flat_return_threshold, bool) or not isinstance(flat_return_threshold, (int, float)):
        raise ValueError("flat_return_threshold must be a numeric fraction")
    if not np.isfinite(flat_return_threshold) or not 0 <= flat_return_threshold < 1:
        raise ValueError("Invalid flat_return_threshold")
    if any(column.startswith("target_") for column in candles.columns):
        raise ValueError("Input already contains target columns; use the original candle dataset")
    if len(candles) <= horizon_steps:
        raise ValueError("Not enough candles for the requested horizon")
    frame = validate_candle_timeline(candles, candle_seconds)
    close = frame["close"].to_numpy(dtype=np.float64, na_value=np.nan)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        returns = close[horizon_steps:] / close[:-horizon_steps] - 1
    if not np.isfinite(returns).all():
        raise ValueError("Target calculation produced nonfinite returns")
    result = frame.iloc[:-horizon_steps].copy()
    result["target_return"] = returns
    # Only absorb floating-point roundoff at the inclusive flat boundaries.
    above = (returns > flat_return_threshold) & ~np.isclose(
        returns, flat_return_threshold, rtol=0, atol=FLAT_BOUNDARY_TOLERANCE,
    )
    below = (returns < -flat_return_threshold) & ~np.isclose(
        returns, -flat_return_threshold, rtol=0, atol=FLAT_BOUNDARY_TOLERANCE,
    )
    result["target_class"] = np.select([above, below], ["up", "down"], default="flat")
    result["target_available_at"] = frame["available_at"].iloc[horizon_steps:].reset_index(drop=True)
    return result


def create_target_dataset(source: Path, settings: Settings) -> tuple[Path, dict]:
    """Write a separate labelled snapshot plus reproducibility metadata."""
    if not source.is_file():
        raise ValueError(f"Candle Parquet file not found: {source}")
    directory = settings.processed_dir / "targets"
    output = directory / f"{source.stem}_h{settings.horizon_steps}.parquet"
    report_path = output.with_suffix(".targets.json")
    if output.exists() or report_path.exists():
        raise ValueError("Target output already exists; move the previous snapshot aside before a new run")
    candles = pd.read_parquet(source)
    result = build_targets(
        candles, settings.horizon_steps, settings.candle_seconds, settings.flat_return_threshold,
    )
    with source.open("rb") as input_file:
        source_digest = hashlib.file_digest(input_file, "sha256").hexdigest()
    report = {
        "schema_version": 1,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_path": str(source.resolve()), "source_sha256": source_digest,
        "symbol": str(result["symbol"].iloc[0]), "exchange": str(result["exchange"].iloc[0]),
        "source_rows": len(candles), "labelled_rows": len(result),
        "trailing_rows_removed": settings.horizon_steps,
        "candle_seconds": settings.candle_seconds,
        "horizon_steps": settings.horizon_steps,
        "horizon_minutes": settings.horizon_steps * settings.candle_seconds / 60,
        "flat_return_threshold": settings.flat_return_threshold,
        "flat_boundary_tolerance": FLAT_BOUNDARY_TOLERANCE,
        "formula": "close[t + horizon_steps] / close[t] - 1",
        "prediction_time": "available_at[t]",
        "label_known_at": "available_at[t + horizon_steps]",
        "future_only_columns": list(TARGET_COLUMNS),
        "class_counts": {label: int(result["target_class"].eq(label).sum()) for label in ("up", "down", "flat")},
        "first_prediction_time": result["available_at"].iloc[0].isoformat(),
        "last_prediction_time": result["available_at"].iloc[-1].isoformat(),
        "last_label_known_at": result["target_available_at"].iloc[-1].isoformat(),
    }
    write_parquet(output, result)
    write_json(report_path, report)
    LOGGER.info("Saved %s labelled rows to %s; removed %s trailing rows", len(result), output, settings.horizon_steps)
    LOGGER.info("Target class counts: %s", report["class_counts"])
    return output, report
