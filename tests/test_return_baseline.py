from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from local_ai_trader.decision.cost_aware import DecisionSettings, ProbabilityPrediction, ReturnForecast, decide_cost_aware
from local_ai_trader.features.build_features import FEATURE_COLUMNS
from local_ai_trader.forecast import baseline
from local_ai_trader.settings import BacktestSettings, load_settings
from test_execution_split import feature_frame

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fitted():
    settings = load_settings(ROOT / "config/settings.toml")
    frame = feature_frame(300)
    rng = np.random.default_rng(42)
    inputs = rng.normal(size=(len(frame), len(FEATURE_COLUMNS)))
    inputs[:, -1] = 3.0
    coefficients = np.array([0.005, -0.003, 0.002, 0, 0, 0, 0, 0])
    frame[list(FEATURE_COLUMNS)] = inputs
    frame["execution_target_return"] = 0.01 + inputs @ coefficients
    model = baseline.fit_return_baselines(frame, settings, 1e-6)
    return model, frame, settings


def later_frame(frame):
    later = frame.iloc[:20].copy()
    for name in ("timestamp", "available_at", "execution_entry_at", "execution_exit_at", "execution_target_available_at"):
        later[name] += pd.to_timedelta(14, unit="D")
    return later


def test_ridge_learns_a_known_linear_mean_and_training_control(fitted):
    model, frame, _ = fitted
    predictions = model.predict(frame)
    np.testing.assert_allclose(predictions["ridge"], frame.execution_target_return, atol=1e-8)
    np.testing.assert_allclose(predictions["mean"], frame.execution_target_return.mean())
    checkpoint = model.checkpoint()
    assert checkpoint["training_rows"] == len(frame)
    assert checkpoint["feature_names"] == list(FEATURE_COLUMNS)
    assert checkpoint["target_column"] == "execution_target_return"
    assert checkpoint["scaler_variance"][-1] == 0
    assert checkpoint["scaler_scale"][-1] == 1
    assert checkpoint["ridge_coefficients"][-1] == pytest.approx(0)
    inputs = frame[list(FEATURE_COLUMNS)].to_numpy()
    np.testing.assert_allclose(checkpoint["scaler_mean"], inputs.mean(axis=0))
    np.testing.assert_allclose(checkpoint["scaler_variance"], inputs.var(axis=0, ddof=0))


def test_json_checkpoint_predictions_match_native_pipeline_on_later_data(fitted):
    model, frame, _ = fitted
    later = later_frame(frame)
    later[list(FEATURE_COLUMNS)] *= 3
    checkpoint = json.loads(json.dumps(model.checkpoint()))
    native = model.predict(later)
    restored = baseline.predict_return_checkpoint(checkpoint, later)
    for name in baseline.MODEL_NAMES:
        np.testing.assert_allclose(restored[name], native[name], rtol=1e-10, atol=1e-12)


def test_future_labels_and_extra_numeric_hints_are_not_inference_inputs(fitted):
    model, frame, _ = fitted
    later = later_frame(frame)
    before = baseline.predict_return_checkpoint(model.checkpoint(), later)
    for name in ("execution_target_return", "execution_target_net_return", "execution_entry_reference_open", "execution_exit_reference_open"):
        later[name] = "not a feature"
    later["future_profit_hint"] = 1e10
    after = baseline.predict_return_checkpoint(model.checkpoint(), later)
    for name in baseline.MODEL_NAMES:
        np.testing.assert_array_equal(before[name], after[name])


def test_forecast_packets_match_market_time_delay_and_holding_basis(fitted):
    model, frame, _ = fitted
    later = later_frame(frame)
    forecasts = baseline.checkpoint_return_forecasts(model.checkpoint(), later)
    assert len(forecasts) == len(later)
    assert all(isinstance(value, ReturnForecast) for value in forecasts)
    first = forecasts[0]
    assert first.symbol == "BTC-USD"
    assert first.prediction_time == later.available_at.iloc[0].to_pydatetime()
    assert (first.candle_seconds, first.latency_bars, first.holding_bars, first.basis) == (300, 1, 6, "execution_open_to_open")
    assert first.expected_gross_return == pytest.approx(model.predict(later)["ridge"][0])
    # A trained forecast now reaches the economic gate, still without orders.
    probabilities = ProbabilityPrediction(first.symbol, first.prediction_time, (0.9, 0.05, 0.05))
    result = decide_cost_aware(probabilities, False, DecisionSettings(), BacktestSettings(), 300, 6, forecast=first, reference=probabilities)
    assert result.reason != "expected_return_unavailable"
    assert result.expected_net_return is not None


def test_forecasts_at_or_before_training_label_availability_are_refused(fitted):
    model, frame, _ = fitted
    with pytest.raises(ValueError, match="training label availability"):
        baseline.checkpoint_return_forecasts(model.checkpoint(), frame)
    later = later_frame(frame)
    checkpoint = model.checkpoint()
    checkpoint["training_last_label_known_at"] = later.available_at.iloc[0].isoformat()
    with pytest.raises(ValueError, match="training label availability"):
        baseline.checkpoint_return_forecasts(checkpoint, later)


def test_invalid_ridge_forecasts_are_reported_but_never_clipped_into_packets(fitted):
    model, frame, _ = fitted
    checkpoint = model.checkpoint()
    checkpoint["ridge_coefficients"] = [0.0] * 8
    checkpoint["ridge_intercept"] = -1.1
    later = later_frame(frame)
    values = baseline.predict_return_checkpoint(checkpoint, later)["ridge"]
    assert (values == -1.1).all()
    metrics = baseline.evaluate_return_predictions(later.execution_target_return, values)
    assert metrics["invalid_forecast_rows"] == len(later)
    with pytest.raises(ValueError, match="below -100%"):
        baseline.checkpoint_return_forecasts(checkpoint, later)


@pytest.mark.parametrize("key,value", [
    ("target_kind", "close_to_close"), ("target_column", "execution_target_net_return"),
    ("feature_names", ["execution_target_return"]), ("scaler_scale", [-1.0] * 8),
    ("scaler_mean", [0.0] * 7), ("scaler_variance", [-1.0] * 8),
    ("ridge_coefficients", [np.nan] * 8), ("ridge_intercept", "0.1"),
    ("training_mean_return", -1.1), ("holding_bars", True), ("symbol", None), ("schema_version", True),
])
def test_checkpoint_sanity_checks_refuse_wrong_schema_and_parameters(fitted, key, value):
    model, frame, _ = fitted
    checkpoint = deepcopy(model.checkpoint())
    checkpoint[key] = value
    with pytest.raises(ValueError):
        baseline.predict_return_checkpoint(checkpoint, later_frame(frame))


@pytest.mark.parametrize("mutation", ["wrong_market", "boolean_feature", "complex_feature", "duplicate_feature", "nonfinite_feature"])
def test_inference_requires_finite_real_features_and_matching_market(fitted, mutation):
    model, frame, _ = fitted
    later = later_frame(frame)
    if mutation == "wrong_market":
        later["symbol"] = "ETH-USD"
    elif mutation == "boolean_feature":
        later[FEATURE_COLUMNS[0]] = True
    elif mutation == "complex_feature":
        later[FEATURE_COLUMNS[0]] = 1 + 2j
    elif mutation == "duplicate_feature":
        later = pd.concat([later, later[[FEATURE_COLUMNS[0]]]], axis=1)
    else:
        later.loc[later.index[0], FEATURE_COLUMNS[0]] = np.inf
    with pytest.raises(ValueError):
        baseline.predict_return_checkpoint(model.checkpoint(), later)


def test_return_metrics_have_analytic_error_values_and_constant_target_convention():
    metrics = baseline.evaluate_return_predictions([-0.01, 0.01, 0.03], [-0.015, 0.02, 0.025])
    assert metrics["mae"] == pytest.approx(0.02 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.00015 / 3))
    assert metrics["r2"] == pytest.approx(0.8125)
    assert metrics["mean_error"] == pytest.approx(0)
    assert metrics["return_unit"] == "fractional_reference_price_return"
    constant = baseline.evaluate_return_predictions([0.003] * 100, [0.002] * 100)
    assert constant["r2"] is None
    assert constant["mae"] == pytest.approx(0.001)
    assert baseline.evaluate_return_predictions([0.01], [0.0])["r2"] is None


@pytest.mark.parametrize("outcomes,predictions", [([], []), ([0, 1], [0]), ([np.nan], [0]), ([0], [np.inf]), ([True], [0]), ([0], [1 + 2j]), ([-1.1], [0])])
def test_invalid_metric_inputs_are_refused(outcomes, predictions):
    with pytest.raises(ValueError):
        baseline.evaluate_return_predictions(outcomes, predictions)


@pytest.mark.parametrize("value", [0, -1, True, np.nan, np.inf, "1.0"])
def test_invalid_ridge_parameters_are_refused(value):
    with pytest.raises(ValueError, match="ridge_alpha"):
        baseline.validate_ridge_alpha(value)


def test_return_config_defaults_and_typo_detection_do_not_change_classifier_settings(tmp_path):
    config = tmp_path / "config/settings.toml"
    config.parent.mkdir()
    original = (ROOT / "config/settings.toml").read_text()
    config.write_text(original)
    assert baseline.load_return_ridge_alpha(config) == 1.0
    settings = load_settings(config)
    config.write_text(original.replace("ridge_alpha = 1.0", "ridge_alpha = 2.0"))
    assert baseline.load_return_ridge_alpha(config) == 2.0
    assert load_settings(config) == settings
    config.write_text(original.replace("[return_forecast]\n", "[return_forecast]\nridge_alhpa = 2.0\n"))
    with pytest.raises(ValueError, match="Unsupported"):
        baseline.load_return_ridge_alpha(config)
