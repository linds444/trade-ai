from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from local_ai_trader.backtest.engine import simulate
from local_ai_trader.decision.cost_aware import (
    DecisionSettings, ProbabilityPrediction, ReturnForecast, decide_cost_aware,
    load_decision_settings, net_return_after_costs, round_trip_factor, validate_decision_settings,
)
from local_ai_trader.settings import BacktestSettings

TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def signal():
    return ProbabilityPrediction("BTC-USD", TIME, (0.8, 0.1, 0.1))


@pytest.fixture
def forecast():
    return ReturnForecast(0.03, "BTC-USD", TIME, 300, 1, 6)


def decision(signal, forecast=None, *, reference=None, holding=False, policy=None, costs=None):
    return decide_cost_aware(signal, holding, policy or DecisionSettings(), costs or BacktestSettings(), 300, 6, forecast, reference)


def test_cost_factor_matches_delayed_fill_simulator_cash_outlay_and_exit_proceeds():
    times = pd.date_range("2026-01-01T00:00:00Z", periods=14, freq="5min")
    opens = np.r_[np.full(8, 100.0), np.full(6, 102.0)]
    frame = pd.DataFrame({"timestamp": times, "available_at": times + pd.to_timedelta(300, unit="s"), "open": opens, "close": opens, "symbol": "BTC-USD", "exchange": "coinbase"})
    probabilities = np.tile([0.0, 0.0, 1.0], (len(times), 1))
    probabilities[0] = [1, 0, 0]
    costs = BacktestSettings()
    simulated = simulate(frame, probabilities, costs, 300, 6)
    assert len(simulated.trades) == 1
    assert simulated.trades.entry_at.iloc[0] == times[2]
    assert simulated.trades.exit_at.iloc[0] == times[8]
    entry, exit = simulated.fills.iloc[0], simulated.fills.iloc[1]
    entry_outlay = entry.notional + entry.fee
    exit_proceeds = exit.notional - exit.fee
    expected = net_return_after_costs(0.02, costs)
    assert expected == pytest.approx(exit_proceeds / entry_outlay - 1)
    assert expected == pytest.approx(simulated.trades.net_pnl.iloc[0] / entry_outlay)
    assert 1 / round_trip_factor(costs) - 1 == pytest.approx((1.001 * 1.006) / (0.999 * 0.994) - 1)
    assert net_return_after_costs(0, costs) < 0


def test_no_costs_preserve_fractional_return():
    costs = replace(BacktestSettings(), fee_bps=0, spread_bps=0, slippage_bps=0)
    assert round_trip_factor(costs) == 1
    assert net_return_after_costs(0.025, costs) == pytest.approx(0.025)


def test_missing_return_forecast_blocks_entry_even_with_certain_up_probabilities(signal):
    certain = replace(signal, probabilities=(1.0, 0.0, 0.0))
    result = decision(certain, reference=certain)
    assert (result.action, result.reason) == ("AVOID", "expected_return_unavailable")
    assert result.expected_net_return is None
    assert result.model_disagreement_normalized == 0
    assert result.risk_check_required is False


def test_buy_is_only_candidate_for_separate_risk_gate(signal, forecast):
    result = decision(signal, forecast, reference=signal)
    assert result.action == "BUY"
    assert result.reason == "entry_conditions_met_pending_risk"
    assert result.risk_check_required is True
    assert result.expected_net_return > DecisionSettings().minimum_expected_net_return


def test_cost_margin_boundary_is_strict_and_is_per_invested_cash(signal, forecast):
    initial = decision(signal, forecast, reference=signal)
    at_boundary = decision(signal, replace(forecast, expected_gross_return=initial.required_gross_return), reference=signal)
    assert at_boundary.action == "AVOID"
    assert at_boundary.reason == "expected_return_does_not_cover_costs_and_margin"
    above = decision(signal, replace(forecast, expected_gross_return=initial.required_gross_return + 1e-6), reference=signal)
    assert above.action == "BUY"


@pytest.mark.parametrize("holding,probabilities,action", [
    (True, (0.1, 0.8, 0.1), "SELL"), (True, (0.8, 0.1, 0.1), "HOLD"),
])
def test_entry_forecast_entropy_and_comparison_filters_cannot_veto_exits(signal, holding, probabilities, action):
    signal = replace(signal, probabilities=probabilities)
    result = decision(signal, holding=holding, reference="invalid comparison", policy=replace(DecisionSettings(), maximum_entropy=0))
    assert result.action == action
    assert result.risk_check_required is False


@pytest.mark.parametrize("field,value", [
    ("basis", "close_to_close"), ("symbol", "ETH-USD"),
    ("prediction_time", TIME + timedelta(minutes=5)),
    ("candle_seconds", 60), ("latency_bars", 2), ("holding_bars", 12),
])
def test_forecast_requires_matching_market_time_execution_basis_and_horizon(signal, forecast, field, value):
    result = decision(signal, replace(forecast, **{field: value}), reference=signal)
    assert result.action == "AVOID"
    assert result.reason == "return_forecast_alignment_mismatch"
    assert result.expected_net_return is None


@pytest.mark.parametrize("field,value", [
    ("expected_gross_return", float("nan")), ("expected_gross_return", float("inf")),
    ("expected_gross_return", True), ("expected_gross_return", -1.01),
    ("prediction_time", TIME.replace(tzinfo=None)), ("latency_bars", True), ("holding_bars", 0),
])
def test_invalid_return_forecast_cannot_approve_buy(signal, forecast, field, value):
    result = decision(signal, replace(forecast, **{field: value}), reference=signal)
    assert (result.action, result.reason) == ("AVOID", "invalid_return_forecast")


def test_reference_model_is_required_by_default_and_alignment_is_checked(signal, forecast):
    assert decision(signal, forecast).reason == "model_comparison_unavailable"
    assert decision(signal, forecast, reference=replace(signal, symbol="ETH-USD")).reason == "model_comparison_alignment_mismatch"
    assert decision(signal, forecast, reference=replace(signal, prediction_time=TIME + timedelta(minutes=5))).reason == "model_comparison_alignment_mismatch"
    assert decision(signal, forecast, reference=replace(signal, probabilities=(float("nan"), 0, 1))).reason == "invalid_model_comparison"
    assert decision(signal, forecast, policy=replace(DecisionSettings(), require_model_comparison=False)).action == "BUY"


def test_uncertainty_limits_and_direction_rules_have_distinct_reasons(signal, forecast):
    assert decision(signal, forecast, reference=signal, policy=replace(DecisionSettings(), maximum_entropy=0.1)).reason == "entropy_limit"
    opposing = replace(signal, probabilities=(0.1, 0.8, 0.1))
    assert decision(signal, forecast, reference=opposing).reason == "model_disagreement_limit"
    bearish = decision(opposing, forecast, reference=opposing)
    assert bearish.reason == "entry_probability_or_margin_not_met"


def test_same_utc_instant_with_different_timezone_is_accepted(signal, forecast):
    other_zone = timezone(timedelta(hours=-8))
    forecast = replace(forecast, prediction_time=TIME.astimezone(other_zone))
    assert decision(signal, forecast, reference=signal).action == "BUY"


@pytest.mark.parametrize("field,value", [
    ("maximum_entropy", float("nan")), ("maximum_entropy", 1.01),
    ("minimum_expected_net_return", -0.01), ("entry_probability", 0),
    ("require_model_comparison", 1), ("unknown_option", 0.1),
])
def test_bad_policy_configuration_fails(field, value):
    with pytest.raises(ValueError):
        validate_decision_settings({field: value})


def test_config_loads_separate_policy_and_preserves_defaults_for_older_configs(tmp_path):
    assert load_decision_settings(ROOT / "config/settings.toml") == DecisionSettings()
    path = tmp_path / "old.toml"
    path.write_text("[market]\nexchange = 'coinbase'\n", encoding="utf-8")
    assert load_decision_settings(path) == DecisionSettings()


def test_prediction_requires_aware_aligned_time_and_valid_probability_distribution(signal, forecast):
    for invalid in (replace(signal, prediction_time=TIME.replace(tzinfo=None)), replace(signal, prediction_time=TIME + timedelta(seconds=1)), replace(signal, probabilities=(0.2, 0.2, 0.2))):
        with pytest.raises(ValueError):
            decision(invalid, forecast, reference=signal)
