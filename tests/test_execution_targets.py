from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from local_ai_trader import cli
from local_ai_trader.backtest.engine import simulate
from local_ai_trader.decision.cost_aware import round_trip_factor
from local_ai_trader.data.dataset import build_targets
from local_ai_trader.forecast import targets
from local_ai_trader.models.train import file_digest
from local_ai_trader.settings import BacktestSettings, load_settings

ROOT = Path(__file__).resolve().parents[1]


def candles(count=30):
    times = pd.date_range("2026-01-01T00:00:00Z", periods=count, freq="5min")
    opens = 100 + np.arange(count, dtype=float) * 0.1
    return pd.DataFrame({
        "timestamp": times, "available_at": times + pd.to_timedelta(300, unit="s"),
        "open": opens, "high": opens + 0.04, "low": opens - 0.03,
        "close": opens + 0.02, "volume": 10.0, "symbol": "BTC-USD", "exchange": "coinbase",
    })


@pytest.fixture
def source(tmp_path):
    config = tmp_path / "config/settings.toml"
    config.parent.mkdir()
    config.write_text((ROOT / "config/settings.toml").read_text(), encoding="utf-8")
    settings = load_settings(config)
    settings.processed_dir.mkdir(parents=True)
    path = settings.processed_dir / "BTC_candles.parquet"
    candles().to_parquet(path, index=False)
    return path, settings, config


def test_default_alignment_uses_entry_2_exit_8_and_conservative_exit_candle_completion():
    frame = candles()
    original = frame.copy(deep=True)
    labelled = targets.build_execution_targets(frame, 300, 6, BacktestSettings())
    assert len(labelled) == 22
    first = labelled.iloc[0]
    assert first.available_at == pd.Timestamp("2026-01-01T00:05:00Z")
    assert first.execution_entry_at == pd.Timestamp("2026-01-01T00:10:00Z")
    assert first.execution_exit_at == pd.Timestamp("2026-01-01T00:40:00Z")
    assert first.execution_target_available_at == pd.Timestamp("2026-01-01T00:45:00Z")
    assert first.execution_target_return == pytest.approx(frame.open.iloc[8] / frame.open.iloc[2] - 1)
    assert first.execution_entry_reference_open == frame.open.iloc[2]
    assert first.execution_exit_reference_open == frame.open.iloc[8]
    assert labelled.execution_target_available_at.iloc[-1] == frame.available_at.iloc[-1]
    assert (labelled.execution_exit_at - labelled.execution_entry_at).eq(pd.to_timedelta(1800, unit="s")).all()
    assert (labelled.execution_target_available_at - labelled.available_at).eq(pd.to_timedelta(2400, unit="s")).all()
    pd.testing.assert_frame_equal(labelled[list(targets.CANDLE_COLUMNS)], original[list(targets.CANDLE_COLUMNS)].iloc[:22])
    pd.testing.assert_frame_equal(frame, original)


def test_net_target_reconciles_with_fixed_horizon_simulator_actual_cash_outlay():
    frame = candles(14)
    probabilities = np.tile([0.0, 0.0, 1.0], (len(frame), 1))
    probabilities[0] = [1, 0, 0]
    costs = BacktestSettings()
    simulation = simulate(frame, probabilities, costs, 300, 6)
    label = targets.build_execution_targets(frame, 300, 6, costs).iloc[0]
    entry = simulation.fills.iloc[0]
    trade = simulation.trades.iloc[0]
    assert trade.entry_at == label.execution_entry_at
    assert trade.exit_at == label.execution_exit_at
    assert trade.bars_held == 6
    assert label.execution_target_net_return == pytest.approx(trade.net_pnl / (entry.notional + entry.fee))


@pytest.mark.parametrize("latency,horizon", [(1, 1), (2, 6), (3, 12)])
def test_other_delays_and_horizons_preserve_timing_and_tail_counts(latency, horizon):
    frame = candles(50)
    costs = replace(BacktestSettings(), latency_bars=latency)
    result = targets.build_execution_targets(frame, 300, horizon, costs)
    offset = 1 + latency + horizon
    assert len(result) == 50 - offset
    assert result.execution_entry_at.iloc[0] == frame.timestamp.iloc[1 + latency]
    assert result.execution_exit_at.iloc[0] == frame.timestamp.iloc[offset]
    assert result.execution_target_available_at.iloc[0] == frame.available_at.iloc[offset]


def test_prices_after_an_exit_candle_cannot_change_that_samples_targets():
    frame = candles()
    original = targets.build_execution_targets(frame, 300, 6, BacktestSettings())
    changed = frame.copy()
    changed.loc[9:, ["open", "high", "low", "close"]] *= 1.5
    altered = targets.build_execution_targets(changed, 300, 6, BacktestSettings())
    pd.testing.assert_series_equal(original.iloc[0], altered.iloc[0])


def test_future_close_prices_do_not_substitute_for_reference_open_prices():
    frame = candles()
    original = targets.build_execution_targets(frame, 300, 6, BacktestSettings()).iloc[0]
    frame.loc[[2, 8], "close"] += 50
    frame.loc[[2, 8], "high"] += 50
    altered = targets.build_execution_targets(frame, 300, 6, BacktestSettings()).iloc[0]
    pd.testing.assert_series_equal(original, altered)


def test_cost_changes_affect_only_net_labels_and_allocation_does_not_change_unit_returns():
    frame = candles()
    base = targets.build_execution_targets(frame, 300, 6, BacktestSettings())
    free = targets.build_execution_targets(frame, 300, 6, replace(BacktestSettings(), fee_bps=0, spread_bps=0, slippage_bps=0))
    np.testing.assert_array_equal(base.execution_target_return, free.execution_target_return)
    np.testing.assert_allclose(free.execution_target_net_return, free.execution_target_return)
    assert (free.execution_target_net_return > base.execution_target_net_return).all()
    small = targets.build_execution_targets(frame, 300, 6, replace(BacktestSettings(), allocation_fraction=0.01))
    pd.testing.assert_frame_equal(base, small)


def test_143_day_source_produces_41176_regression_labels():
    result = targets.build_execution_targets(candles(41184), 300, 6, BacktestSettings())
    assert len(result) == 41176
    assert not result[list(targets.FUTURE_COLUMNS)].isna().any().any()


def test_source_timezone_is_normalized_to_utc():
    frame = candles()
    for name in ("timestamp", "available_at"):
        frame[name] = frame[name].dt.tz_convert("America/Los_Angeles")
    result = targets.build_execution_targets(frame, 300, 6, BacktestSettings())
    assert str(result.execution_entry_at.dt.tz) == "UTC"
    assert str(result.execution_target_available_at.dt.tz) == "UTC"


@pytest.mark.parametrize("mutation", [
    "gap", "duplicate", "unordered", "availability", "future", "naive_time", "mixed_market",
    "bad_open", "boolean_volume", "negative_volume", "inconsistent_high", "already_labelled", "too_short",
])
def test_invalid_source_cannot_generate_execution_targets(mutation):
    frame = candles()
    if mutation == "gap":
        frame = frame.drop(5)
    elif mutation == "duplicate":
        frame = pd.concat([frame.iloc[:5], frame.iloc[4:]])
    elif mutation == "unordered":
        frame = frame.iloc[::-1]
    elif mutation == "availability":
        frame.loc[2, "available_at"] += pd.to_timedelta(300, unit="s")
    elif mutation == "future":
        for name in ("timestamp", "available_at"):
            frame[name] += pd.to_timedelta(3650, unit="D")
    elif mutation == "naive_time":
        frame["timestamp"] = frame.timestamp.dt.tz_localize(None)
    elif mutation == "mixed_market":
        frame.loc[2, "symbol"] = "ETH-USD"
    elif mutation == "bad_open":
        frame.loc[2, "open"] = float("inf")
    elif mutation == "boolean_volume":
        frame["volume"] = True
    elif mutation == "negative_volume":
        frame.loc[2, "volume"] = -1
    elif mutation == "inconsistent_high":
        frame.loc[2, "high"] = 1
    elif mutation == "already_labelled":
        frame["target_return"] = 0.1
    else:
        frame = frame.iloc[:8]
    with pytest.raises(ValueError):
        targets.build_execution_targets(frame, 300, 6, BacktestSettings())


def test_return_ratio_overflow_is_rejected():
    frame = candles()
    for index, price in ((2, 1e-300), (8, 1e300)):
        frame.loc[index, ["open", "high", "low", "close"]] = price
    with pytest.raises(ValueError, match="nonfinite"):
        targets.build_execution_targets(frame, 300, 6, BacktestSettings())


def test_execution_labels_cannot_be_reused_as_original_classifier_candles():
    labelled = targets.build_execution_targets(candles(), 300, 6, BacktestSettings())
    with pytest.raises(ValueError, match="original candle"):
        build_targets(labelled, 6, 300, 0.001)


def test_published_bundle_records_timing_future_columns_and_both_file_hashes(source):
    path, settings, _ = source
    before = path.read_bytes()
    output, report = targets.create_execution_target_dataset(path, settings)
    restored = pd.read_parquet(output / "labels.parquet")
    assert len(restored) == 22
    assert report["source_sha256"] == file_digest(path)
    assert report["labels_sha256"] == file_digest(output / "labels.parquet")
    assert report["entry_row_offset"] == 2
    assert report["exit_row_offset"] == 8
    assert report["target_known_offset_from_prediction_bars"] == 8
    assert report["future_only_columns"] == list(targets.FUTURE_COLUMNS)
    assert report["regression_target_column"] == "execution_target_return"
    assert report["models_fitted"] is False
    assert report["strategy_performance_evaluated"] is False
    assert report["round_trip_factor"] == round_trip_factor(settings.backtest)
    assert json.loads((output / "targets.json").read_text())["labels_sha256"] == report["labels_sha256"]
    assert path.read_bytes() == before
    saved = (output / "labels.parquet").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        targets.create_execution_target_dataset(path, replace(settings, backtest=replace(settings.backtest, fee_bps=90)))
    assert (output / "labels.parquet").read_bytes() == saved


@pytest.mark.parametrize("writer", ["write_json", "write_parquet"])
def test_failed_publication_leaves_no_partial_bundle(source, monkeypatch, writer):
    path, settings, _ = source

    def fail(*args, **kwargs):
        raise OSError("Target disk failure")

    monkeypatch.setattr(targets, writer, fail)
    with pytest.raises(OSError):
        targets.create_execution_target_dataset(path, settings)
    assert not list((settings.processed_dir / "execution_targets").iterdir())


def test_source_mutation_prevents_publication(source, monkeypatch):
    path, settings, _ = source
    original_write = targets.write_json

    def write(*args, **kwargs):
        original_write(*args, **kwargs)
        with path.open("ab") as stream:
            stream.write(b"changed")

    monkeypatch.setattr(targets, "write_json", write)
    with pytest.raises(ValueError, match="source changed"):
        targets.create_execution_target_dataset(path, settings)
    assert not list((settings.processed_dir / "execution_targets").iterdir())


def test_cli_uses_original_candles_and_reports_expected_counts(source):
    path, settings, config = source
    assert cli.main(["execution-targets", str(path), "--config", str(config)]) == 0
    outputs = list((settings.processed_dir / "execution_targets").iterdir())
    assert len(outputs) == 1
    assert len(pd.read_parquet(outputs[0] / "labels.parquet")) == 22
