"""Spot-long decision candidates requiring an execution-aligned return forecast."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from numbers import Real
from pathlib import Path
import tomllib

import numpy as np

from local_ai_trader.settings import BacktestSettings, load_backtest_settings, positive_integer, validate_symbol
from local_ai_trader.uncertainty.scores import model_disagreement, uncertainty_scores


@dataclass(frozen=True)
class DecisionSettings:
    entry_probability: float = 0.55
    exit_probability: float = 0.55
    minimum_direction_margin: float = 0.15
    maximum_entropy: float = 0.90
    maximum_model_disagreement: float = 0.10
    require_model_comparison: bool = True
    minimum_expected_net_return: float = 0.002


def validate_decision_settings(values: dict) -> DecisionSettings:
    """Validate configurable engineering defaults; reject misspelled settings."""
    defaults = DecisionSettings()
    if not isinstance(values, dict) or set(values) - set(defaults.__dataclass_fields__):
        raise ValueError("Unsupported decision policy setting")
    parameters = {name: values.get(name, getattr(defaults, name)) for name in defaults.__dataclass_fields__}
    if type(parameters["require_model_comparison"]) is not bool:
        raise ValueError("decision.require_model_comparison must be boolean")
    for name, value in parameters.items():
        if name == "require_model_comparison":
            continue
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"decision.{name} must be a finite fraction in [0, 1]")
    if parameters["entry_probability"] == 0 or parameters["exit_probability"] == 0:
        raise ValueError("Decision entry and exit probabilities must be positive")
    return DecisionSettings(**parameters)


def load_decision_settings(path: Path) -> DecisionSettings:
    """Read the separate future-policy table, leaving frozen research Settings intact."""
    with path.open("rb") as stream:
        values = tomllib.load(stream)
    return validate_decision_settings(values.get("decision", {}))


@dataclass(frozen=True)
class ProbabilityPrediction:
    symbol: str
    prediction_time: datetime
    probabilities: tuple[float, float, float]


@dataclass(frozen=True)
class ReturnForecast:
    """Fractional mean reference-price return from delayed entry open to exit open.

    Classifier close-to-close targets have a different timing basis and cannot
    serve as this forecast. All metadata must match the current prediction.
    """
    expected_gross_return: float
    symbol: str
    prediction_time: datetime
    candle_seconds: int
    latency_bars: int
    holding_bars: int
    basis: str = "execution_open_to_open"


@dataclass(frozen=True)
class CostAwareDecision:
    action: str
    reason: str
    expected_net_return: float | None
    break_even_gross_return: float
    required_gross_return: float
    entropy_normalized: float
    top_class_probability: float
    top_class_margin: float
    model_disagreement_normalized: float | None
    risk_check_required: bool


def round_trip_factor(costs: BacktestSettings) -> float:
    """Ratio of exit proceeds to entry outlay for an unchanged reference price.

    Entry price = open*(1+impact), exit price = open*(1-impact).
    Entry outlay includes fees; exit proceeds deduct fees on actual fill notional.
    Return is per unit of cash invested, not a whole-account return.
    """
    costs = load_backtest_settings(asdict(costs))
    fee = costs.fee_bps / 10000
    impact = (costs.spread_bps / 2 + costs.slippage_bps) / 10000
    return (1 - impact) * (1 - fee) / ((1 + impact) * (1 + fee))


def net_return_after_costs(gross_return: float, costs: BacktestSettings) -> float:
    """Convert a fractional fixed-horizon gross mean to a modeled net mean."""
    if isinstance(gross_return, bool) or not isinstance(gross_return, Real) or not math.isfinite(gross_return) or gross_return < -1:
        raise ValueError("Expected gross return must be a finite fraction at least -1")
    return (1 + gross_return) * round_trip_factor(costs) - 1


def utc_prediction_time(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Prediction time must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def prediction_scores(prediction: ProbabilityPrediction, candle_seconds: int):
    if not isinstance(prediction, ProbabilityPrediction):
        raise ValueError("Require a structured probability prediction")
    validate_symbol(prediction.symbol)
    time = utc_prediction_time(prediction.prediction_time)
    if time.microsecond or int(time.timestamp()) % candle_seconds:
        raise ValueError("Prediction time must be aligned to the candle interval")
    matrix = np.asarray([prediction.probabilities])
    return matrix, uncertainty_scores(matrix)


def decide_cost_aware(
    prediction: ProbabilityPrediction, holding: bool, policy: DecisionSettings,
    costs: BacktestSettings, candle_seconds: int, horizon_steps: int,
    forecast: ReturnForecast | None = None, reference: ProbabilityPrediction | None = None,
) -> CostAwareDecision:
    """Propose BUY/SELL/HOLD/AVOID; a BUY still requires independent risk approval.

    Entry forecasting and uncertainty filters never veto a valid SELL signal.
    Fixed-horizon return estimates do not model an earlier policy/risk exit.
    Freshness, account state and execution authorization belong to the paper
    coordinator and risk gate, rather than this historical-compatible policy.
    """
    policy = validate_decision_settings(asdict(policy))
    costs = load_backtest_settings(asdict(costs))
    positive_integer(candle_seconds, "candle_seconds")
    positive_integer(horizon_steps, "horizon_steps")
    if type(holding) is not bool:
        raise ValueError("holding must be boolean")
    probabilities, scores = prediction_scores(prediction, candle_seconds)
    factor = round_trip_factor(costs)
    break_even = 1 / factor - 1
    required = (1 + policy.minimum_expected_net_return) / factor - 1
    divergence, expected_net = None, None

    def result(action, reason):
        return CostAwareDecision(
            action, reason, expected_net, break_even, required,
            float(scores["entropy_normalized"][0]), float(scores["top_class_probability"][0]),
            float(scores["top_class_margin"][0]), divergence, action == "BUY",
        )

    up, down, _ = probabilities[0]
    if holding:
        if down >= policy.exit_probability and down - up >= policy.minimum_direction_margin:
            return result("SELL", "down_probability_and_margin")
        return result("HOLD", "position_open")
    comparison_issue = None
    if reference is not None:
        try:
            comparison, _ = prediction_scores(reference, candle_seconds)
            if reference.symbol != prediction.symbol or utc_prediction_time(reference.prediction_time) != utc_prediction_time(prediction.prediction_time):
                comparison_issue = "model_comparison_alignment_mismatch"
            else:
                divergence = float(model_disagreement(probabilities, comparison)[0])
        except ValueError:
            comparison_issue = "invalid_model_comparison"
    elif policy.require_model_comparison:
        comparison_issue = "model_comparison_unavailable"
    if forecast is None:
        return result("AVOID", "expected_return_unavailable")
    if not isinstance(forecast, ReturnForecast):
        return result("AVOID", "invalid_return_forecast")
    try:
        for name in ("candle_seconds", "latency_bars", "holding_bars"):
            positive_integer(getattr(forecast, name), name)
        forecast_time = utc_prediction_time(forecast.prediction_time)
        expected_net = net_return_after_costs(forecast.expected_gross_return, costs)
    except ValueError:
        return result("AVOID", "invalid_return_forecast")
    if (
        forecast.symbol != prediction.symbol or forecast_time != utc_prediction_time(prediction.prediction_time)
        or forecast.basis != "execution_open_to_open" or forecast.candle_seconds != candle_seconds
        or forecast.latency_bars != costs.latency_bars or forecast.holding_bars != horizon_steps
    ):
        expected_net = None
        return result("AVOID", "return_forecast_alignment_mismatch")
    if forecast.expected_gross_return <= required:
        return result("AVOID", "expected_return_does_not_cover_costs_and_margin")
    if comparison_issue is not None:
        return result("AVOID", comparison_issue)
    if scores["entropy_normalized"][0] > policy.maximum_entropy:
        return result("AVOID", "entropy_limit")
    if divergence is not None and divergence > policy.maximum_model_disagreement:
        return result("AVOID", "model_disagreement_limit")
    if up < policy.entry_probability or up - down < policy.minimum_direction_margin:
        return result("AVOID", "entry_probability_or_margin_not_met")
    return result("BUY", "entry_conditions_met_pending_risk")
