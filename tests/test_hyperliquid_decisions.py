from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from local_ai_trader.cli import main
from local_ai_trader.hyperliquid import decisions, feed
from local_ai_trader.data.storage import native_path, write_json
from local_ai_trader.paper.perpetual import modeled_entry_net_return
from local_ai_trader.risk.entry_policy import EntryPolicy

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 1, 5, 0, 5, 10, tzinfo=timezone.utc)


def prediction(probability=0.8, gross=0.01):
    return {"primary_model": "xgboost", "probabilities": {"xgboost": np.array([[probability, (1-probability)/2, (1-probability)/2]])},
            "return_forecasts": {"ridge": np.array([gross]), "mean": np.array([0.0])}, "calibration_id": "sha256:fake-test-model",
            "confidence": None, "reliability_validation_id": None}


def asset():
    return {"book": {"bid": 99.99, "ask": 100.01}, "taker_fee_bps_assumption": 4.5,
            "hourly_funding_rate": 0.0001, "oracle_price": 101}


@pytest.fixture
def scanner(tmp_path, monkeypatch):
    config = native_path(tmp_path / "config/hyperliquid.toml")
    config.parent.mkdir(parents=True)
    config.write_bytes((ROOT / "config/hyperliquid.toml").read_bytes())
    settings = feed.load_hyperliquid_settings(config)
    client = feed.HyperliquidPublicClient(settings, clock=lambda: NOW)
    def info(payload):
        if payload["type"] == "metaAndAssetCtxs":
            coins = [coin for coin in settings.coins if feed.BY_COIN[coin].dex == payload.get("dex", "")]
            return [{"collateralToken": 0, "universe": [{"name": coin, "szDecimals": 4, "maxLeverage": 20,
                    **({"deployerFeeScale": "1", "growthMode": "enabled"} if ":" in coin else {})} for coin in coins]},
                    [{"markPx": "100", "oraclePx": "101", "funding": "0.0001"} for _ in coins]]
        if payload["type"] == "l2Book":
            return {"coin": payload["coin"], "time": int(NOW.timestamp()*1000), "levels": [[{"px": "99.99", "sz": "1", "n": 1}], [{"px": "100.01", "sz": "1", "n": 1}]]}
        req = payload["req"]
        return [{"s": req["coin"], "i": "5m", "t": stamp, "T": stamp+299999, "o": "100", "c": "100", "h": "101", "l": "99", "v": "10", "n": 1}
                for stamp in range(req["startTime"], req["endTime"], 300000)]
    monkeypatch.setattr(client, "info", info)
    model_root = native_path(tmp_path / "saved-models")
    protocol = {"momentum_steps": 6, "rolling_window": 12, "candle_seconds": 300, "latency_bars": 1, "holding_bars": 6, "target_known_offset_from_prediction_bars": 8}
    write_json(model_root / "models.json", {"coins": list(settings.coins), "protocol": protocol, "markets": {coin: {"status": "trained"} for coin in settings.coins}})
    monkeypatch.setattr(decisions, "load_saved_market", lambda root, coin: ({"protocol": protocol}, root, "sha256:fake-test-model"))
    def predict(root, coin, frame):
        assert len(frame) in (1, 2)
        assert frame.available_at.iloc[0].to_pydatetime() == NOW.replace(second=0)-timedelta(minutes=5)
        if len(frame) == 2:
            assert frame.available_at.iloc[-1].to_pydatetime() == NOW.replace(second=0)
        assert not any(name.startswith("execution_") for name in frame.columns)
        result = prediction()
        result["primary_model"] = "mlp" if coin == "BTC" else "xgboost"
        result["probabilities"] = {result["primary_model"]: np.repeat(result["probabilities"]["xgboost"], len(frame), axis=0)}
        result["return_forecasts"] = {name: np.repeat(values, len(frame)) for name, values in result["return_forecasts"].items()}
        return result
    monkeypatch.setattr(decisions, "predict_saved_market", predict)
    return client, config, model_root


def test_all_eight_delayed_previews_are_reviewed_without_accounts_or_fills(scanner):
    client, config, models = scanner
    output, report = decisions.create_decision_preview(client, config, models)
    assert len(report["decisions"]) == 8
    for row in report["decisions"].values():
        assert row["action"] == "BUY"
        assert row["signal_gate"]["status"] == "REVIEW"
        assert row["signal_gate"]["reason"] == "validated_confidence_unavailable"
        assert row["account_risk_authorized"] is False
        assert row["confidence"] is None
    assert report["decisions"]["xyz:HOOD"]["taker_fee_bps"] == pytest.approx(0.9)
    assert report["decisions"]["BTC"]["taker_fee_bps"] == 4.5
    assert report["decisions"]["kPEPE"]["underlying_units_per_contract"] == 1000
    assert (output / "decisions.json").exists()
    assert report["models_fitted"] is False and report["test_opened"] is False
    assert report["paper_account_created"] is False and report["forward_strategy_started"] is False
    assert not (output.parent.parent / "paper").exists()


@pytest.mark.parametrize("probability,gross,reason", [(0.59, 0.03, "up_probability_below_minimum"), (0.8, 0.0001, "forecast_does_not_cover_costs_and_margin")])
def test_low_probability_and_cost_failure_are_avoid(probability, gross, reason):
    result = decisions.decide_market(asset(), prediction(probability, gross), EntryPolicy())
    assert result["action"] == "AVOID" and result["decision_reason"] == reason


def test_flat_probability_is_never_used_as_buy_probability():
    evidence = prediction()
    evidence["probabilities"]["xgboost"] = np.array([[0.1, 0.05, 0.85]])
    result = decisions.decide_market(asset(), evidence, EntryPolicy())
    assert result["calibrated_up_probability"] == 0.1
    assert result["action"] == "AVOID"


def test_negative_funding_cannot_credit_hoped_for_income_and_oracle_is_used():
    data = asset()
    expected = 0.0001*101/(100.01*1.0001)
    result = decisions.decide_market(data, prediction(), EntryPolicy())
    assert result["current_funding_cost_budget"] == pytest.approx(expected)
    data["hourly_funding_rate"] = -0.001
    assert decisions.decide_market(data, prediction(), EntryPolicy())["current_funding_cost_budget"] == 0


def test_shared_net_return_matches_analytic_cash_flows_and_margin_equality():
    quote = {"bid": 100, "ask": 101, "fee_bps": 10}
    entry, exit = 101*1.0001, 100*0.9999*1.05
    expected = (exit-entry-entry*0.001-exit*0.001)/entry-0.0002
    assert modeled_entry_net_return(quote, 0.05, 1, 0.0002) == pytest.approx(expected)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_return_forecasts_are_refused(value):
    with pytest.raises(ValueError):
        decisions.decide_market(asset(), prediction(gross=value), EntryPolicy())


def test_primary_unavailability_cannot_select_another_family():
    evidence = prediction()
    evidence["primary_model"] = "mlp"
    result = decisions.decide_market(asset(), evidence, EntryPolicy())
    assert result["action"] == "AVOID" and result["decision_reason"] == "primary_model_unavailable"


@pytest.mark.parametrize("changed", ["momentum_steps", "latency_bars", "holding_bars"])
def test_feature_and_timing_mismatches_are_refused(scanner, changed):
    client, config, models = scanner
    import json
    report = json.loads((models / "models.json").read_text())
    report["protocol"][changed] += 1
    write_json(models / "models.json", report)
    with pytest.raises(ValueError, match="Scanner"):
        decisions.create_decision_preview(client, config, models)


@pytest.mark.parametrize("seconds", [30, 300])
def test_quotes_expiring_or_boundary_advancing_during_inference_refuse_publication(scanner, monkeypatch, seconds):
    client, config, models = scanner
    original = decisions.predict_saved_market
    def slow(*args):
        result = original(*args)
        client.clock = lambda: NOW+timedelta(seconds=seconds)
        return result
    monkeypatch.setattr(decisions, "predict_saved_market", slow)
    with pytest.raises(ValueError):
        decisions.create_decision_preview(client, config, models)
    assert not (client.settings.observations.parent / "decisions").exists()


def test_changed_model_manifest_refuses_publication(scanner, monkeypatch):
    client, config, models = scanner
    original = decisions.predict_saved_market
    def changed(*args):
        result = original(*args)
        with (models / "models.json").open("a") as stream:
            stream.write("\n")
        return result
    monkeypatch.setattr(decisions, "predict_saved_market", changed)
    with pytest.raises(ValueError, match="manifest changed"):
        decisions.create_decision_preview(client, config, models)


def test_failed_publication_cleans_staging(scanner, monkeypatch):
    client, config, models = scanner
    original = decisions.write_json
    def failed(path, value):
        if path.name == "decisions.json":
            raise OSError("injected write failure")
        return original(path, value)
    monkeypatch.setattr(decisions, "write_json", failed)
    with pytest.raises(OSError, match="injected"):
        decisions.create_decision_preview(client, config, models)
    assert list((client.settings.observations.parent / "decisions").iterdir()) == []


def test_cli_publishes_preview_without_execution(scanner, monkeypatch, capsys):
    client, config, models = scanner
    monkeypatch.setattr(decisions, "HyperliquidPublicClient", lambda settings: client)
    assert main(["hyperliquid-decisions", "--models", str(models), "--config", str(config)]) == 0
    output = capsys.readouterr().out
    assert "independent account risk not requested; no fills" in output
    assert "Saved live decision preview to" in output


def test_forward_tracking_records_newest_signal_and_resolves_after_restart_and_40_minutes(scanner):
    from local_ai_trader.hyperliquid.forward import track_preview
    client, config, models = scanner
    output, report = decisions.create_decision_preview(client, config, models, track_outcomes=True)
    assert len(report["pending_forward_predictions"]) == 8
    for coin, value in report["pending_forward_predictions"].items():
        assert datetime.fromisoformat(value["signal_at"]) == NOW.replace(second=0)
        assert datetime.fromisoformat(report["decisions"][coin]["signal_at"]) == NOW.replace(second=0)-timedelta(minutes=5)
    path, summary = track_preview(client, output, report)
    assert summary["recorded"] == 8 and summary["pending"] == 8 and summary["resolved"] == 0
    del client._forward_books  # Emulate a fresh process reopening its book.
    client.clock = lambda: NOW+timedelta(minutes=40)
    _, summary = track_preview(client, output, report)
    assert summary["recorded"] == 8 and summary["resolved"] == 8 and summary["pending"] == 0
    assert summary["observed_accuracy"] == 0
    assert summary["up_probability_ge_60_rows"] == 8 and summary["up_probability_ge_60_hits"] == 0
    assert summary["confidence"] is None and summary["paper_profitability_trial_started"] is False
    _, repeated = track_preview(client, output, report)
    assert repeated == summary


def test_late_tracker_does_not_backfill_unrecorded_predictions(scanner):
    from local_ai_trader.hyperliquid.forward import track_preview
    client, config, models = scanner
    output, report = decisions.create_decision_preview(client, config, models, track_outcomes=True)
    client.clock = lambda: NOW+timedelta(minutes=6)
    _, summary = track_preview(client, output, report)
    assert summary["recorded"] == summary["resolved"] == 0


def test_forward_tracker_rejects_changed_preview(scanner):
    from local_ai_trader.hyperliquid.forward import track_preview
    client, config, models = scanner
    output, report = decisions.create_decision_preview(client, config, models, track_outcomes=True)
    changed = deepcopy(report)
    changed["pending_forward_predictions"]["BTC"]["ridge_return"] = 0.1
    write_json(output / "decisions.json", changed)
    with pytest.raises(ValueError, match="preview changed"):
        track_preview(client, output, report)
