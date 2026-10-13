"""Causal, explicitly allowlisted features for execution-aligned returns."""

import json
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
from local_ai_trader.settings import Settings, positive_integer

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


def feature_contract(result: pd.DataFrame, target_metadata: dict, candle_seconds: int, momentum_steps: int, rolling_window: int) -> dict:
    """Shared feature semantics for creation and downstream audit."""
    warmup = max(momentum_steps, rolling_window)
    return {
        "schema_version": 1, "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "research_only": True, "target_kind": "execution_open_to_open",
        "target_metadata": target_metadata,
        "symbol": target_metadata["symbol"], "exchange": target_metadata["exchange"],
        "source_rows": len(result) + warmup, "feature_rows": len(result),
        "warmup_rows_removed": warmup,
        "candle_seconds": candle_seconds, "momentum_steps": momentum_steps,
        "rolling_window": rolling_window, "volatility_ddof": 0,
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


def load_execution_feature_bundle(directory: Path) -> tuple[pd.DataFrame, dict, dict[Path, str]]:
    """Verify upstream targets, reconstruct causal features, and audit metadata.

    This is dataset preparation; do not invoke it inside a fitting/evaluation
    phase that must leave its held-out partition unopened.
    """
    directory = native_path(directory.resolve())
    metadata_path, features_path = directory / "features.json", directory / "features.parquet"
    if not metadata_path.is_file() or not features_path.is_file():
        raise ValueError("Execution feature bundle requires features.json and features.parquet")
    metadata_digest = file_digest(metadata_path)
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict) or type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
        raise ValueError("Unsupported execution feature metadata")
    for name in ("candle_seconds", "momentum_steps", "rolling_window"):
        positive_integer(metadata.get(name), name)
    target_path = metadata.get("target_bundle_path")
    if not isinstance(target_path, str) or not Path(target_path).is_absolute():
        raise ValueError("Execution feature metadata requires an absolute target bundle path")
    own_inputs = {metadata_path: metadata_digest, features_path: metadata.get("features_sha256")}
    verify_input_hashes(own_inputs)
    labelled, target_metadata, inputs = load_execution_target_bundle(Path(target_path))
    if metadata.get("input_hashes") != {str(path): digest for path, digest in inputs.items()}:
        raise ValueError("Execution feature upstream input hashes disagree")
    if metadata["candle_seconds"] != target_metadata["candle_seconds"]:
        raise ValueError("Execution feature candle interval disagrees with targets")
    expected = build_execution_features(labelled, metadata["candle_seconds"], metadata["momentum_steps"], metadata["rolling_window"])
    for name, value in feature_contract(expected, target_metadata, metadata["candle_seconds"], metadata["momentum_steps"], metadata["rolling_window"]).items():
        actual = metadata.get(name)
        if actual != value or (type(value) in (bool, int) and type(actual) is not type(value)):
            raise ValueError(f"Execution feature metadata disagrees on {name}")
    frame = pd.read_parquet(features_path)
    try:
        pd.testing.assert_frame_equal(frame, expected, check_exact=True)
    except AssertionError as error:
        raise ValueError("Execution features disagree with their causal candle source") from error
    inputs.update(own_inputs)
    verify_input_hashes(inputs)
    return frame, metadata, inputs


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
        **feature_contract(result, target_metadata, settings.candle_seconds, settings.momentum_steps, settings.rolling_window),
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(), "target_bundle_path": str(directory),
        "input_hashes": {str(path): digest for path, digest in inputs.items()},
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
