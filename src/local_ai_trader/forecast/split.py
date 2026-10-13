"""Chronological return-forecast partitions purged by execution label availability."""

import logging
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from local_ai_trader.data.storage import native_path, write_json, write_parquet
from local_ai_trader.features.build_features import FEATURE_COLUMNS
from local_ai_trader.forecast.features import load_execution_feature_bundle
from local_ai_trader.forecast.targets import CANDLE_COLUMNS, FUTURE_COLUMNS, validate_execution_candles, verify_input_hashes
from local_ai_trader.models.train import file_digest
from local_ai_trader.settings import Settings, positive_integer, validate_split_fractions

LOGGER = logging.getLogger(__name__)
LABEL_AVAILABLE_COLUMN = "execution_target_available_at"


def validate_execution_features(features: pd.DataFrame, candle_seconds: int, holding_bars: int, latency_bars: int) -> pd.DataFrame:
    """Require finite explicit inputs and correctly timed execution labels."""
    positive_integer(holding_bars, "holding_bars")
    positive_integer(latency_bars, "latency_bars")
    if not features.columns.is_unique or set(features.columns) != set(CANDLE_COLUMNS + FEATURE_COLUMNS + FUTURE_COLUMNS):
        raise ValueError("Require the execution feature schema without classifier labels or extra inputs")
    candles = validate_execution_candles(features[list(CANDLE_COLUMNS)], candle_seconds)
    frame = features.copy().reset_index(drop=True)
    frame[list(CANDLE_COLUMNS)] = candles
    for name in (*FEATURE_COLUMNS, "execution_entry_reference_open", "execution_exit_reference_open", "execution_target_return", "execution_target_net_return"):
        if not pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(frame[name]):
            raise ValueError(f"{name} must be numeric")
        values = frame[name].to_numpy(dtype=np.float64, na_value=np.nan)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite")
        if name.endswith("reference_open") and (values <= 0).any():
            raise ValueError("Execution reference prices must be positive")
        if name in ("execution_target_return", "execution_target_net_return") and (values < -1).any():
            raise ValueError("Execution returns cannot be below -100%")
    offsets = {"execution_entry_at": latency_bars, "execution_exit_at": latency_bars + holding_bars, LABEL_AVAILABLE_COLUMN: latency_bars + holding_bars + 1}
    for name, offset in offsets.items():
        times = frame[name]
        if times.isna().any() or getattr(times.dtype, "tz", None) is None:
            raise ValueError(f"{name} must contain nonmissing timezone-aware datetimes")
        frame[name] = times.dt.tz_convert("UTC")
        if not frame[name].eq(frame.available_at + pd.to_timedelta(offset * candle_seconds, unit="s")).all():
            raise ValueError(f"Execution label timing disagrees on {name}")
    if frame[LABEL_AVAILABLE_COLUMN].max() > pd.Timestamp.now(tz="UTC"):
        raise ValueError("Execution outcomes are not yet available")
    return frame


def chronological_execution_split(
    features: pd.DataFrame, candle_seconds: int, holding_bars: int, latency_bars: int,
    train_fraction: float, validation_fraction: float,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Preserve order, then require labels strictly before the next prediction.

    Default target availability is eight bars after the signal, so eight rows
    are purged from each earlier partition, including the equality boundary.
    The source already drops unknown trailing labels; test needs no extra purge.
    """
    train_fraction, validation_fraction = validate_split_fractions(train_fraction, validation_fraction)
    frame = validate_execution_features(features, candle_seconds, holding_bars, latency_bars)
    train_end = int(len(frame) * train_fraction)
    validation_end = int(len(frame) * (train_fraction + validation_fraction))
    if not 0 < train_end < validation_end < len(frame):
        raise ValueError("Not enough data for three chronological execution partitions")
    validation_start, test_start = frame.available_at.iloc[train_end], frame.available_at.iloc[validation_end]
    raw = {"train": frame.iloc[:train_end], "validation": frame.iloc[train_end:validation_end], "test": frame.iloc[validation_end:]}
    boundaries = {"train": validation_start, "validation": test_start}
    partitions, summaries = {}, {}
    for name, data in raw.items():
        keep = data[LABEL_AVAILABLE_COLUMN] < boundaries[name] if name in boundaries else pd.Series(True, index=data.index)
        selected = data.loc[keep].copy().reset_index(drop=True)
        if selected.empty:
            raise ValueError(f"{name} is empty after execution label purging; use a longer dataset")
        partitions[name] = selected
        summaries[name] = {
            "raw_rows": len(data), "rows": len(selected), "purged_rows": int((~keep).sum()),
            "purged_prediction_times": [stamp.isoformat() for stamp in data.loc[~keep, "available_at"]],
            "first_prediction_time": selected.available_at.iloc[0].isoformat(),
            "last_prediction_time": selected.available_at.iloc[-1].isoformat(),
            "last_label_known_at": selected[LABEL_AVAILABLE_COLUMN].iloc[-1].isoformat(),
        }
    return partitions, {
        "source_rows": len(frame), "train_fraction": train_fraction, "validation_fraction": validation_fraction,
        "test_fraction": 1 - train_fraction - validation_fraction,
        "cut_positions": {"validation_start": train_end, "test_start": validation_end},
        "purge_rule": "execution_target_available_at < next_partition_first_available_at",
        "target_known_offset_from_prediction_bars": latency_bars + holding_bars + 1,
        "validation_first_prediction_time": validation_start.isoformat(), "test_first_prediction_time": test_start.isoformat(),
        "partitions": summaries,
    }


def create_execution_split_dataset(directory: Path, settings: Settings) -> tuple[Path, dict]:
    """Audit source lineage and publish all three partitions atomically."""
    directory = native_path(directory.resolve())
    output = native_path(settings.processed_dir / "execution_splits" / directory.name)
    if output.exists():
        raise ValueError("Execution split bundle already exists; inspect its split.json")
    frame, metadata, inputs = load_execution_feature_bundle(directory)
    for name in ("candle_seconds", "momentum_steps", "rolling_window"):
        if metadata[name] != getattr(settings, name):
            raise ValueError(f"Execution feature metadata and current settings disagree on {name}")
    target = metadata["target_metadata"]
    if target["holding_bars"] != settings.horizon_steps or target["latency_bars"] != settings.backtest.latency_bars:
        raise ValueError("Execution target timing and current settings disagree")
    if metadata["symbol"] not in settings.symbols:
        raise ValueError("Execution split symbol is not enabled in the current market configuration")
    partitions, summaries = chronological_execution_split(frame, metadata["candle_seconds"], target["holding_bars"], target["latency_bars"], settings.train_fraction, settings.validation_fraction)
    report = {
        "schema_version": 1, "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "research_only": True, "target_kind": "execution_open_to_open",
        "feature_bundle_path": str(directory), "feature_metadata": metadata,
        "input_hashes": {str(path): digest for path, digest in inputs.items()},
        "symbol": metadata["symbol"], "exchange": metadata["exchange"],
        "candle_seconds": metadata["candle_seconds"], "holding_bars": target["holding_bars"], "latency_bars": target["latency_bars"],
        "feature_names": list(FEATURE_COLUMNS), "future_only_columns": list(FUTURE_COLUMNS),
        "regression_target_column": "execution_target_return", "target_available_column": LABEL_AVAILABLE_COLUMN,
        "normalization": "none; eventual scaling must fit training rows only",
        "models_fitted": False, "strategy_performance_evaluated": False,
        "reserved_test_status": "partition published but no model or strategy scored; freshness relative to prior research is not established",
        **summaries,
    }
    verify_input_hashes(inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".execution-split-staging-", dir=output.parent))
    try:
        for name, partition in partitions.items():
            path = staging / f"{name}.parquet"
            write_parquet(path, partition)
            report["partitions"][name]["sha256"] = file_digest(path)
        write_json(staging / "split.json", report)
        verify_input_hashes(inputs)
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    for name, summary in report["partitions"].items():
        LOGGER.info("%s: %s rows; purged %s overlapping execution labels", name, summary["rows"], summary["purged_rows"])
    LOGGER.info("Saved execution-aligned chronological splits to %s; no scaling or model fitting", output)
    return output, report
