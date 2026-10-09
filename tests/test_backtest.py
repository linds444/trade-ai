from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from local_ai_trader import cli
from local_ai_trader.backtest import run as backtest_run
from local_ai_trader.backtest.engine import simulate
from local_ai_trader.calibration.run import create_calibration
from local_ai_trader.data.dataset import create_target_dataset
from local_ai_trader.data.split import create_split_dataset
from local_ai_trader.decision.policy import decide
from local_ai_trader.features.build_features import create_feature_dataset
from local_ai_trader.models import inference
from local_ai_trader.models.train import train_baseline_experiment
from local_ai_trader.risk.risk_gate import authorize_entry
from local_ai_trader.settings import BacktestSettings, load_backtest_settings, load_settings

ROOT = Path(__file__).resolve().parents[1]
NO_COST = BacktestSettings(fee_bps=0, spread_bps=0, slippage_bps=0)


def market(opens):
    opens = np.asarray(opens, dtype=float)
    timestamps = pd.date_range("2025-01-01", periods=len(opens), freq="5min", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps, "available_at": timestamps + pd.to_timedelta(300, unit="s"),
        "open": opens, "high": opens * 1.001, "low": opens * 0.999,
        "close": opens, "volume": 10.0, "symbol": "BTC-USD", "exchange": "coinbase",
    })


def one_entry(count):
    scores = np.tile([0.1, 0.1, 0.8], (count, 1))
    scores[0] = [0.8, 0.1, 0.1]
    return scores


def test_delayed_fill_uses_future_open_and_holds_six_bars():
    frame = market([100, 150] + [200] * 10)
    result = simulate(frame, one_entry(len(frame)), NO_COST, 300, 6)
    buy, sell = result.fills.iloc[0], result.fills.iloc[1]
    assert buy.signal_at == frame.available_at.iloc[0]
    assert buy.filled_at == buy.signal_at + pd.to_timedelta(300, unit="s")
    assert buy.filled_at == frame.timestamp.iloc[2]
    assert buy.reference_price == 200
    assert buy.quantity == 5  # 10% of $10,000, using the execution price.
    assert sell.filled_at == frame.timestamp.iloc[8]
    assert result.trades.iloc[0].bars_held == 6
    assert result.trades.iloc[0].exit_reason == "horizon"
    assert result.metrics["final_equity"] == 10000


def test_cash_pnl_reconciles_with_two_sided_fees_spread_and_slippage():
    frame = market([100] * 12)
    config = BacktestSettings()
    result = simulate(frame, one_entry(len(frame)), config, 300, 6)
    quantity = 1000 / (100.1 * 1.006)
    expected_fees = quantity * (100.1 + 99.9) * 0.006
    expected_impact = quantity * 0.2
    expected_loss = expected_fees + expected_impact
    assert result.fills.iloc[0].cash_after == pytest.approx(9000)
    assert result.metrics["fees_paid"] == pytest.approx(expected_fees)
    assert result.metrics["spread_and_slippage_cost"] == pytest.approx(expected_impact)
    assert result.metrics["final_equity"] == pytest.approx(10000 - expected_loss)
    assert result.trades.net_pnl.sum() == pytest.approx(-expected_loss)
    assert result.metrics["gross_pnl_before_costs"] == 0
    assert result.metrics["profit_factor"] == 0
    assert (result.equity.exposure_fraction <= config.max_exposure_fraction).all()


def test_price_profit_and_cost_attribution_use_actual_quantity():
    frame = market([100] * 8 + [110] * 4)
    result = simulate(frame, one_entry(len(frame)), NO_COST, 300, 6)
    assert result.metrics["final_equity"] == 10100
    assert result.trades.iloc[0].gross_pnl == 100
    assert result.metrics["net_return"] == pytest.approx(0.01)
    assert result.metrics["win_rate"] == 1
    assert result.metrics["profit_factor"] is None  # No losing trades.
    assert result.metrics["turnover"] == pytest.approx(0.21)


def test_sell_signal_exits_with_same_latency_without_opening_a_short():
    frame = market([100] * 12)
    scores = one_entry(len(frame))
    scores[3:] = [0.1, 0.8, 0.1]
    result = simulate(frame, scores, NO_COST, 300, 6)
    assert len(result.trades) == 1
    assert result.trades.iloc[0].bars_held == 3
    assert result.trades.iloc[0].exit_at == frame.timestamp.iloc[5]
    assert result.trades.iloc[0].exit_reason == "policy_sell"
    assert (result.equity.quantity >= 0).all()
    assert (result.equity.cash >= 0).all()


def test_continuous_entry_signals_never_create_overlapping_positions():
    frame = market([100] * 25)
    scores = np.tile([0.8, 0.1, 0.1], (len(frame), 1))
    result = simulate(frame, scores, NO_COST, 300, 6)
    assert result.fills.side.tolist() == ["BUY", "SELL"] * len(result.trades)
    assert (result.trades.entry_at.iloc[1:].reset_index(drop=True) > result.trades.exit_at.iloc[:-1].reset_index(drop=True)).all()
    assert result.equity.quantity.iloc[-1] == 0
    assert result.metrics["maximum_observed_exposure_fraction"] <= 0.1


def test_terminal_exit_is_predefined_and_charges_costs():
    frame = market([100] * 6)
    result = simulate(frame, one_entry(len(frame)), BacktestSettings(), 300, 6)
    assert result.trades.iloc[0].exit_reason == "period_end"
    assert result.trades.iloc[0].exit_at == frame.timestamp.iloc[-1]
    assert result.trades.iloc[0].bars_held == 3
    assert result.metrics["terminally_truncated_trades"] == 1
    assert result.fills.iloc[-1].fee > 0
    assert result.decisions["at"].max() <= frame.timestamp.iloc[-1]


def test_drawdown_blocks_reentry_but_does_not_block_exit():
    frame = market([100] * 3 + [50] * 17)
    config = replace(NO_COST, allocation_fraction=0.5, max_exposure_fraction=0.5, max_drawdown_fraction=0.05)
    scores = np.tile([0.8, 0.1, 0.1], (len(frame), 1))
    result = simulate(frame, scores, config, 300, 6)
    assert result.metrics["number_of_trades"] == 1
    assert result.metrics["blocked_entries"] > 0
    assert result.metrics["final_equity"] == 7500
    assert result.metrics["max_drawdown"] == pytest.approx(0.25)
    assert "drawdown_limit" in result.decisions.reason.tolist()
    assert result.trades.iloc[0].exit_reason == "horizon"


def test_exposure_gate_rejects_oversized_policy_orders():
    frame = market([100] * 12)
    result = simulate(frame, one_entry(len(frame)), replace(NO_COST, allocation_fraction=0.5), 300, 6)
    assert result.metrics["number_of_trades"] == 0
    assert result.metrics["blocked_entries"] == 1
    assert "exposure_limit" in result.decisions.reason.tolist()
    assert result.metrics["final_equity"] == 10000


def test_risk_gate_rejects_insufficient_cash_and_invalid_orders():
    assert authorize_entry(100, 100, 100, 2, 100, 100, NO_COST).reason == "insufficient_cash"
    assert not authorize_entry(100, 100, 100, float("nan"), 100, 100, NO_COST).allowed


def test_policy_thresholds_require_direction_and_margin():
    config = BacktestSettings()
    assert decide(0.55, 0.40, False, config).action == "BUY"
    assert decide(0.55, 0.41, False, config).action == "AVOID"
    assert decide(0.4, 0.55, True, config).action == "SELL"
    assert decide(0.6, 0.2, True, config).action == "HOLD"
    assert decide(0.2, 0.6, False, config).action == "AVOID"


def test_cash_and_buy_hold_benchmarks_use_matching_allocation_and_costs():
    frame = market([100] * 3 + [110] * 9)
    scores = one_entry(len(frame))
    cash = simulate(frame, scores, NO_COST, 300, 6, "cash")
    passive = simulate(frame, scores, NO_COST, 300, 6, "buy_and_hold")
    assert cash.metrics["net_return"] == 0
    assert cash.metrics["max_drawdown"] == 0
    assert cash.metrics["sharpe"] is None
    assert cash.metrics["win_rate"] is None
    assert passive.metrics["final_equity"] == 10100
    assert passive.trades.iloc[0].entry_at == frame.timestamp.iloc[2]
    assert passive.trades.iloc[0].exit_at == frame.timestamp.iloc[-1]
    assert passive.metrics["number_of_trades"] == 1
    charged = simulate(frame, scores, BacktestSettings(), 300, 6, "buy_and_hold")
    assert charged.metrics["final_equity"] < passive.metrics["final_equity"]


def test_future_predictions_and_prices_cannot_change_prior_cash_path():
    frame = market([100] * 20)
    scores = one_entry(len(frame))
    original = simulate(frame, scores, NO_COST, 300, 6)
    altered = frame.copy()
    altered.loc[10:, ["open", "close"]] = 200
    changed_scores = scores.copy()
    changed_scores[10:] = [0.8, 0.1, 0.1]
    altered["target_return"] = 999.0  # This column must never enter decisions.
    changed = simulate(altered, changed_scores, NO_COST, 300, 6)
    pd.testing.assert_frame_equal(original.equity.iloc[:10], changed.equity.iloc[:10])
    pd.testing.assert_frame_equal(original.fills, changed.fills.iloc[:2].reset_index(drop=True))


@pytest.mark.parametrize("mutation", ["gap", "zero_open", "nan_open", "bad_probabilities", "wrong_length"])
def test_invalid_execution_data_fails(mutation):
    frame, scores = market([100] * 12), one_entry(12)
    if mutation == "gap":
        frame = frame.drop(5)
        scores = scores[:-1]
    elif mutation == "zero_open":
        frame.loc[2, "open"] = 0
    elif mutation == "nan_open":
        frame.loc[2, "open"] = float("nan")
    elif mutation == "bad_probabilities":
        scores[2] = [0.8, 0.8, 0.8]
    else:
        scores = scores[:-1]
    with pytest.raises(ValueError):
        simulate(frame, scores, NO_COST, 300, 6)


@pytest.mark.parametrize("name,value", [
    ("initial_cash", 0), ("allocation_fraction", 2), ("fee_bps", -1),
    ("slippage_bps", 10000), ("spread_bps", float("nan")),
    ("latency_bars", 0), ("latency_bars", True), ("entry_probability", True),
    ("max_drawdown_fraction", 0), ("minimum_direction_margin", -0.1),
])
def test_invalid_backtest_settings_fail(name, value):
    with pytest.raises(ValueError):
        load_backtest_settings({name: value})


@pytest.fixture
def saved_experiment(tmp_path):
    config = tmp_path / "config/settings.toml"
    config.parent.mkdir()
    config.write_text((ROOT / "config/settings.toml").read_text(), encoding="utf-8")
    settings = load_settings(config)
    prices = 100 * (1 + 0.006 * np.sin(np.arange(300) * 0.21))
    source = tmp_path / "candles.parquet"
    market(prices).to_parquet(source, index=False)
    targets = create_target_dataset(source, settings)[0]
    features = create_feature_dataset(targets, settings)[0]
    splits = create_split_dataset(features, settings)[0]
    run = train_baseline_experiment(splits, settings)[0]
    create_calibration(run, settings.calibration)
    return run, splits, settings, config


def test_validation_only_publication_reads_no_test_or_forward_label_columns(saved_experiment, monkeypatch):
    run, splits, settings, _ = saved_experiment
    (splits / "test.parquet").write_bytes(b"DO NOT OPEN OR HASH")
    (run / "test_evaluation").mkdir()
    (run / "test_evaluation/metrics.json").write_text("Never read known test results")
    loaded, hashed = [], []
    original_read, original_digest = pd.read_parquet, inference.file_digest
    original_simulate = backtest_run.simulate

    def read(path, *args, **kwargs):
        loaded.append(Path(path).name)
        return original_read(path, *args, **kwargs)

    def digest(path):
        hashed.append(Path(path).name)
        return original_digest(path)

    def checked_simulate(candles, *args, **kwargs):
        assert set(candles.columns) == {"timestamp", "available_at", "open", "close", "symbol", "exchange"}
        return original_simulate(candles, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read)
    monkeypatch.setattr(inference, "file_digest", digest)
    monkeypatch.setattr(backtest_run, "simulate", checked_simulate)
    output, report = backtest_run.create_backtest(run, "logistic", "temperature", settings.backtest)
    assert loaded == ["validation.parquet"]
    assert "test.parquet" not in hashed
    assert report["test_data_opened"] is False
    assert report["classification_test_evaluation_already_exists"] is True
    assert report["model_or_calibration_refitted"] is False
    assert (output / "backtest.json").exists()
    assert len(list(output.glob("*.parquet"))) == 12
    assert set(report["metrics"]) == {"policy", "cash", "buy_and_hold"}
    assert pd.Timestamp(report["simulation_end"]) < pd.Timestamp(json.loads((splits / "split.json").read_text())["test_first_prediction_time"])


def test_raw_variant_is_separate_and_identical_settings_cannot_overwrite(saved_experiment):
    run, _, settings, _ = saved_experiment
    first, _ = backtest_run.create_backtest(run, "logistic", "temperature", settings.backtest)
    with pytest.raises(ValueError, match="already saved"):
        backtest_run.create_backtest(run, "logistic", "temperature", settings.backtest)
    second, report = backtest_run.create_backtest(run, "logistic", "raw", settings.backtest)
    assert first != second
    assert report["variant"] == "raw"


def test_active_trade_ledgers_round_trip_through_parquet(saved_experiment):
    run, _, settings, _ = saved_experiment
    config = replace(settings.backtest, entry_probability=0.01, minimum_direction_margin=0.0, exit_probability=0.99)
    output, report = backtest_run.create_backtest(run, "logistic", "temperature", config)
    assert report["metrics"]["policy"]["number_of_trades"] > 0
    equity = pd.read_parquet(output / "policy_equity.parquet")
    trades = pd.read_parquet(output / "policy_trades.parquet")
    fills = pd.read_parquet(output / "policy_fills.parquet")
    decisions = pd.read_parquet(output / "policy_decisions.parquet")
    assert equity.quantity.iloc[-1] == 0
    assert trades.net_pnl.sum() == pytest.approx(equity.cash.iloc[-1] - config.initial_cash)
    assert fills.side.tolist() == ["BUY", "SELL"] * len(trades)
    assert fills.fee.sum() == pytest.approx(report["metrics"]["policy"]["fees_paid"])
    assert decisions.action.eq("RISK_CHECK").any()


def test_missing_calibration_fails_before_loading_data(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No dataset should be opened")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises(ValueError, match="Saved calibration required"):
        backtest_run.create_backtest(tmp_path, "mlp", "temperature", BacktestSettings())


def test_failed_backtest_write_leaves_no_partial_bundle(saved_experiment, monkeypatch):
    run, _, settings, _ = saved_experiment

    def fail(*args, **kwargs):
        raise OSError("Simulated full disk")

    monkeypatch.setattr(backtest_run, "write_parquet", fail)
    with pytest.raises(OSError):
        backtest_run.create_backtest(run, "logistic", "temperature", settings.backtest)
    assert not list((run / "backtests").iterdir())


@pytest.mark.parametrize("filename", ["checkpoint.json", "calibration/calibration.json"])
def test_changed_frozen_inputs_prevent_publication(saved_experiment, monkeypatch, filename):
    run, _, settings, _ = saved_experiment
    original = backtest_run.write_parquet

    def changed(*args, **kwargs):
        original(*args, **kwargs)
        with (run / filename).open("ab") as stream:
            stream.write(b" ")

    monkeypatch.setattr(backtest_run, "write_parquet", changed)
    with pytest.raises(ValueError, match="changed during"):
        backtest_run.create_backtest(run, "logistic", "temperature", settings.backtest)
    assert not list((run / "backtests").iterdir())


def test_backtest_cli_reports_validation_research(saved_experiment, capsys):
    run, _, _, config = saved_experiment
    assert cli.main(["backtest", str(run), "--model", "logistic", "--config", str(config)]) == 0
    assert "not unseen strategy performance" in capsys.readouterr().out
