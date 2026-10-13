"""Historical reference-open returns for delayed-entry fixed-horizon forecasts."""

from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from local_ai_trader.data.storage import native_path, write_json, write_parquet
from local_ai_trader.data.validate import validate_candle_timeline
from local_ai_trader.decision.cost_aware import round_trip_factor
from local_ai_trader.models.train import file_digest
from local_ai_trader.settings import BacktestSettings, Settings, load_backtest_settings, positive_integer, validate_symbol

LOGGER = logging.getLogger(__name__)
CANDLE_COLUMNS = ("timestamp", "available_at", "open", "high", "low", "close", "volume", "symbol", "exchange")
FUTURE_COLUMNS = (
    "execution_entry_at", "execution_exit_at", "execution_entry_reference_open",
    "execution_exit_reference_open", "execution_target_return", "execution_target_net_return",
    "execution_target_available_at",
)


def validate_execution_candles(candles: pd.DataFrame, candle_seconds: int) -> pd.DataFrame:
    """Require original complete OHLCV candles, not features or existing labels."""
    if not candles.columns.is_unique or set(candles.columns) != set(CANDLE_COLUMNS):
        raise ValueError("Execution targets require original OHLCV candle columns without features or targets")
    frame = validate_candle_timeline(candles, candle_seconds)
    if not isinstance(frame.symbol.iloc[0], str):
        raise ValueError("Candle symbol must be a Coinbase product string")
    validate_symbol(frame.symbol.iloc[0])
    if frame.exchange.iloc[0] != "coinbase":
        raise ValueError("Execution targets currently support Coinbase candles")
    for name in ("open", "high", "low", "close", "volume"):
        if not pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(frame[name]):
            raise ValueError(f"Candle {name} must be numeric")
        values = frame[name].to_numpy(dtype=np.float64, na_value=np.nan)
        if not np.isfinite(values).all() or (values < 0).any() or (name != "volume" and (values == 0).any()):
            raise ValueError("Candle prices must be positive and finite; volume must be nonnegative and finite")
    if (frame.high < frame[["open", "close", "low"]].max(axis=1)).any() or (frame.low > frame[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Candle OHLC price relationships are inconsistent")
    return frame[list(CANDLE_COLUMNS)]


def build_execution_targets(
    candles: pd.DataFrame, candle_seconds: int, horizon_steps: int, costs: BacktestSettings,
) -> pd.DataFrame:
    """Use open[t+1+latency+h]/open[t+1+latency]-1, predicted at available_at[t].

    Target availability conservatively uses the exit candle's completion time,
    since this pipeline obtains completed OHLCV candles. Future opens/times and
    returns are labels only; none may be used as model inputs. Final rows lacking
    their exit candle are removed rather than imputed.
    """
    positive_integer(horizon_steps, "horizon_steps")
    costs = load_backtest_settings(asdict(costs))
    frame = validate_execution_candles(candles, candle_seconds)
    entry_offset = 1 + costs.latency_bars
    exit_offset = entry_offset + horizon_steps
    if len(frame) <= exit_offset:
        raise ValueError("Not enough complete candles for delayed entry and horizon exit")
    count = len(frame) - exit_offset
    entry = frame.iloc[entry_offset:entry_offset + count].reset_index(drop=True)
    exit = frame.iloc[exit_offset:].reset_index(drop=True)
    entry_prices = entry.open.to_numpy(dtype=np.float64)
    exit_prices = exit.open.to_numpy(dtype=np.float64)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        gross = exit_prices / entry_prices - 1
        net = (1 + gross) * round_trip_factor(costs) - 1
    if not np.isfinite(gross).all() or not np.isfinite(net).all():
        raise ValueError("Execution target calculation produced nonfinite returns")
    result = frame.iloc[:count].copy()
    result["execution_entry_at"] = entry.timestamp
    result["execution_exit_at"] = exit.timestamp
    result["execution_entry_reference_open"] = entry_prices
    result["execution_exit_reference_open"] = exit_prices
    result["execution_target_return"] = gross
    result["execution_target_net_return"] = net
    result["execution_target_available_at"] = exit.available_at
    return result


def target_contract(result: pd.DataFrame, source_rows: int, candle_seconds: int, holding_bars: int, costs: BacktestSettings) -> dict:
    """Shared timing and label semantics for creation and downstream audit."""
    entry_offset = costs.latency_bars + 1
    exit_offset = entry_offset + holding_bars
    return {
        "schema_version": 1,
        "target_kind": "execution_open_to_open", "research_only": True,
        "symbol": result.symbol.iloc[0], "exchange": result.exchange.iloc[0],
        "source_rows": source_rows, "labelled_rows": len(result), "trailing_rows_removed": exit_offset,
        "candle_seconds": candle_seconds, "holding_bars": holding_bars,
        "latency_bars": costs.latency_bars,
        "entry_row_offset": entry_offset, "exit_row_offset": exit_offset,
        "target_known_offset_from_prediction_bars": exit_offset,
        "formula": "open[t + 1 + latency_bars + holding_bars] / open[t + 1 + latency_bars] - 1",
        "net_formula": "(1 + execution_target_return) * round_trip_factor - 1",
        "round_trip_factor": round_trip_factor(costs),
        "costs": asdict(costs), "prediction_time": "available_at[t]",
        "label_known_at": "available_at[t + 1 + latency_bars + holding_bars]",
        "availability_assumption": "conservative_exit_candle_completion_for_complete_OHLCV_pipeline",
        "regression_target_column": "execution_target_return",
        "diagnostic_net_target_column": "execution_target_net_return",
        "causal_candle_columns": list(CANDLE_COLUMNS), "future_only_columns": list(FUTURE_COLUMNS),
        "first_prediction_time": result.available_at.iloc[0].isoformat(),
        "last_prediction_time": result.available_at.iloc[-1].isoformat(),
        "last_label_known_at": result.execution_target_available_at.iloc[-1].isoformat(),
        "models_fitted": False, "strategy_performance_evaluated": False,
        "limitations": "fixed-horizon price labels and diagnostic costs; no early policy/risk exit, capacity, partial-fill or intrabar-stop model",
    }


def verify_input_hashes(inputs: dict[Path, str]) -> None:
    """Detect mutation of the original candles, labels or metadata during use."""
    for path, digest in inputs.items():
        if not path.is_file() or file_digest(path) != digest:
            raise ValueError(f"Execution input changed or its hash disagrees: {path}")


def load_execution_target_bundle(directory: Path) -> tuple[pd.DataFrame, dict, dict[Path, str]]:
    """Audit hashes and reconstruct labels from their original complete candles.

    Reconstruction verifies timing, returns, row counts and the entire stored
    schema. It prepares historical supervised data, never fits a model or
    computes strategy performance.
    """
    directory = native_path(directory.resolve())
    metadata_path, labels_path = directory / "targets.json", directory / "labels.parquet"
    if not metadata_path.is_file() or not labels_path.is_file():
        raise ValueError("Execution target bundle requires targets.json and labels.parquet")
    metadata_digest = file_digest(metadata_path)
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("Unsupported execution target metadata")
    for name in ("candle_seconds", "holding_bars"):
        positive_integer(metadata.get(name), name)
    if not isinstance(metadata.get("costs"), dict) or not isinstance(metadata.get("source_path"), str):
        raise ValueError("Execution target costs and original source path are required")
    costs = load_backtest_settings(metadata["costs"])
    source = native_path(Path(metadata["source_path"]))
    if not source.is_absolute():
        raise ValueError("Execution target source path must be absolute")
    inputs = {source: metadata.get("source_sha256"), labels_path: metadata.get("labels_sha256"), metadata_path: metadata_digest}
    verify_input_hashes(inputs)
    candles = pd.read_parquet(source)
    expected = build_execution_targets(candles, metadata["candle_seconds"], metadata["holding_bars"], costs)
    for name, value in target_contract(expected, len(candles), metadata["candle_seconds"], metadata["holding_bars"], costs).items():
        actual = metadata.get(name)
        if actual != value or (type(value) in (bool, int) and type(actual) is not type(value)):
            raise ValueError(f"Execution target metadata disagrees on {name}")
    labelled = pd.read_parquet(labels_path)
    try:
        pd.testing.assert_frame_equal(labelled, expected, check_exact=True)
    except AssertionError as error:
        raise ValueError("Execution labels disagree with their original candle source") from error
    verify_input_hashes(inputs)
    return labelled, metadata, inputs


def create_execution_target_dataset(source: Path, settings: Settings) -> tuple[Path, dict]:
    """Atomically publish a source-bound, immutable regression-label bundle."""
    source = native_path(source.resolve())
    if not source.is_file():
        raise ValueError(f"Original candle Parquet not found: {source}")
    output = native_path(settings.processed_dir / "execution_targets" / f"{source.stem}_h{settings.horizon_steps}_l{settings.backtest.latency_bars}")
    if output.exists():
        raise ValueError("Execution target bundle already exists; inspect its targets.json")
    digest = file_digest(source)
    candles = pd.read_parquet(source)
    result = build_execution_targets(candles, settings.candle_seconds, settings.horizon_steps, settings.backtest)
    if result.symbol.iloc[0] not in settings.symbols:
        raise ValueError("Execution target symbol is not enabled in the current market configuration")
    exit_offset = settings.backtest.latency_bars + 1 + settings.horizon_steps
    report = {
        **target_contract(result, len(candles), settings.candle_seconds, settings.horizon_steps, settings.backtest),
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_path": str(source), "source_sha256": digest,
    }

    def verify_source():
        if file_digest(source) != digest:
            raise ValueError("Original candle source changed during target creation")

    verify_source()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".execution-targets-staging-", dir=output.parent))
    try:
        write_parquet(staging / "labels.parquet", result)
        report["labels_sha256"] = file_digest(staging / "labels.parquet")
        write_json(staging / "targets.json", report)
        verify_source()
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    LOGGER.info("Saved %s execution-aligned labels to %s; removed %s trailing rows", len(result), output, exit_offset)
    LOGGER.info("Entry delay: %s bars from prediction; holding: %s bars; label known: %s bars after prediction", settings.backtest.latency_bars, settings.horizon_steps, exit_offset)
    return output, report
