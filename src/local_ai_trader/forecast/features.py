"""Causal, explicitly allowlisted features for execution-aligned returns."""

import logging
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from local_ai_trader.data.storage import native_path, write_json, write_parquet
from local_ai_trader.features.build_features import build_features, FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from local_ai_trader.forecast.targets import CANDLE_COLUMNS, FUTURE_COLUMNS, load_execution_target_bundle, verify_input_hashes
from local_ai_trader.models.train import file_digest
from local_ai_trader.settings import Settings

LOGGER = logging.getLogger(__name__)


def build_execution_features(labelled: pd.DataFrame, candle_seconds: int, momentum_steps: int, rolling_window: int) -> pd.DataFrame:
    """Give the feature calculation only completed causal OHLCV columns.

    Future label columns are attached after feature computation and deterministic
    warmup removal. Model inputs must select exactly FEATURE_COLUMNS, never all
    numeric columns. Scaling is left to the eventual training partition.
    """
    if not labelled.columns.is_unique or set(labelled.columns) != set(CANDLE_COLUMNS + FUTURE_COLUMNS):
        raise ValueError("Require execution labels without existing features or extra columns")
    features = build_features(labelled[list(CANDLE_COLUMNS)], candle_seconds, momentum_steps, rolling_window)
    warmup = max(momentum_steps, rolling_window)
    labels = labelled.iloc[warmup:][list(FUTURE_COLUMNS)].reset_index(drop=True)
    return pd.concat([features, labels], axis=1)


def create_execution_feature_dataset(directory: Path, settings: Settings) -> tuple[Path, dict]:
    """Audit the target bundle and atomically publish immutable feature data."""
    directory = native_path(directory.resolve())
    output = native_path(settings.processed_dir / "execution_features" / directory.name)
    if output.exists():
        raise ValueError("Execution feature bundle already exists; inspect its features.json")
    labelled, target_metadata, inputs = load_execution_target_bundle(directory)
    for name, expected in (
        ("candle_seconds", settings.candle_seconds), ("holding_bars", settings.horizon_steps),
        ("latency_bars", settings.backtest.latency_bars),
    ):
        if target_metadata[name] != expected:
            raise ValueError(f"Execution target metadata and current settings disagree on {name}")
    if target_metadata["symbol"] not in settings.symbols:
        raise ValueError("Execution feature symbol is not enabled in the current market configuration")
    result = build_execution_features(labelled, settings.candle_seconds, settings.momentum_steps, settings.rolling_window)
    report = {
        "schema_version": 1, "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "research_only": True, "target_kind": "execution_open_to_open",
        "target_bundle_path": str(directory), "target_metadata": target_metadata,
        "input_hashes": {str(path): digest for path, digest in inputs.items()},
        "symbol": target_metadata["symbol"], "exchange": target_metadata["exchange"],
        "source_rows": len(labelled), "feature_rows": len(result),
        "warmup_rows_removed": max(settings.momentum_steps, settings.rolling_window),
        "candle_seconds": settings.candle_seconds, "momentum_steps": settings.momentum_steps,
        "rolling_window": settings.rolling_window, "volatility_ddof": 0,
        "feature_names": list(FEATURE_COLUMNS), "future_only_columns": list(FUTURE_COLUMNS),
        "regression_target_column": "execution_target_return",
        "target_available_column": "execution_target_available_at",
        "target_known_offset_from_prediction_bars": target_metadata["target_known_offset_from_prediction_bars"],
        "prediction_time": "available_at", "normalization": "none; eventual scaling must fit training rows only",
        "zero_volume_window_ratio": 0.0, "zero_width_close_location": 0.5,
        "first_prediction_time": result.available_at.iloc[0].isoformat(),
        "last_prediction_time": result.available_at.iloc[-1].isoformat(),
        "last_label_known_at": result.execution_target_available_at.iloc[-1].isoformat(),
        "models_fitted": False, "strategy_performance_evaluated": False,
    }
    verify_input_hashes(inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".execution-features-staging-", dir=output.parent))
    try:
        write_parquet(staging / "features.parquet", result)
        report["features_sha256"] = file_digest(staging / "features.parquet")
        write_json(staging / "features.json", report)
        verify_input_hashes(inputs)
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    LOGGER.info("Saved %s execution feature rows to %s; removed %s warmup rows", len(result), output, report["warmup_rows_removed"])
    LOGGER.info("Model input features (%s): %s", len(FEATURE_COLUMNS), ", ".join(FEATURE_COLUMNS))
    LOGGER.info("Future execution columns are labels only; label known: %s bars after prediction", report["target_known_offset_from_prediction_bars"])
    return output, report
