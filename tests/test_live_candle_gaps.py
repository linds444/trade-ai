import json
from datetime import timedelta

import pytest

from local_ai_trader.hyperliquid import decisions, feed
from local_ai_trader.hyperliquid.forward import track_preview
from local_ai_trader.hyperliquid.paper_monitor import monitor_preview

from test_hyperliquid_decisions import scanner, NOW


def gap(client, monkeypatch, *, index=3, empty=False, all_markets=False):
    original = client.info
    def missing(payload):
        result = original(payload)
        if payload["type"] == "candleSnapshot" and (all_markets or payload["req"]["coin"] == "xyz:HOOD"):
            if empty:
                return []
            return result[:index]+result[index+1:] if index >= 0 else result[:index]
        return result
    monkeypatch.setattr(client, "info", missing)
    return original


@pytest.mark.parametrize("index", [0, 3, -1])
def test_gap_skips_affected_market_without_features_inference_or_new_forward_prediction(scanner, monkeypatch, caplog, index):
    client, config, models = scanner
    gap(client, monkeypatch, index=index)
    original_predict = decisions.predict_saved_market
    inferred = []
    def predict(root, coin, frame):
        inferred.append(coin)
        return original_predict(root, coin, frame)
    monkeypatch.setattr(decisions, "predict_saved_market", predict)
    path, report = decisions.create_decision_preview(client, config, models, track_outcomes=True)
    assert set(inferred) == set(client.settings.coins)-{"xyz:HOOD"}
    row = report["decisions"]["xyz:HOOD"]
    assert row["action"] == "AVOID" and row["decision_reason"] == "missing_completed_candles"
    assert row["features"] is None and row["signal_at"] is None and row["expected_gross_return"] is None
    assert row["calibrated_up_probability"] is None and row["confidence"] is None
    assert row["data_quality"]["missing_count"] == 1
    assert len(row["data_quality"]["missing_timestamps_utc"]) == 1
    assert "xyz:HOOD" not in report["pending_forward_predictions"]
    assert len(report["pending_forward_predictions"]) == 7
    assert "Other complete markets continue; no candle filling" in caplog.text
    archived = json.loads((path / "raw_market.json").read_text())
    assert len(archived["xyz:HOOD_candles"]) == 23
    assert all(r["features"] is not None for coin, r in report["decisions"].items() if coin != "xyz:HOOD")
    asset = report["market_snapshot"]["assets"]["xyz:HOOD"]
    assert asset["market_state"]["momentum"] is None and asset["feature_rows"] == 0
    assert asset["market_state"]["price"] > 0  # independent fresh quotes remain available


def test_strict_data_validation_default_still_refuses_the_same_gap(scanner, monkeypatch):
    client, config, models = scanner
    gap(client, monkeypatch)
    with pytest.raises(ValueError, match="xyz:HOOD: 1 missing completed candles"):
        client.snapshot()
    with pytest.raises(ValueError, match="missing completed candles"):
        feed.save_hyperliquid_snapshot(client, config)


def test_empty_market_history_has_no_invented_features_but_other_markets_continue(scanner, monkeypatch):
    client, config, models = scanner
    gap(client, monkeypatch, empty=True)
    path, report = decisions.create_decision_preview(client, config, models, track_outcomes=True)
    row = report["decisions"]["xyz:HOOD"]
    assert row["data_quality"]["missing_count"] == 24
    assert row["features"] is None and row["expected_gross_return"] is None
    assert len(report["pending_forward_predictions"]) == 7
    assert len(json.loads((path / "raw_market.json").read_text())["xyz:HOOD_candles"]) == 0


def test_all_market_gaps_can_maintain_paused_account_without_forecasts_or_fills(scanner, monkeypatch):
    client, config, models = scanner
    gap(client, monkeypatch, all_markets=True)
    monkeypatch.setattr(decisions, "predict_saved_market", lambda *args: pytest.fail("No incomplete market may infer"))
    path, report = decisions.create_decision_preview(client, config, models, track_outcomes=True)
    assert report["pending_forward_predictions"] == {}
    assert all(row["action"] == "AVOID" for row in report["decisions"].values())
    book = monitor_preview(client, path, report)
    assert book["account"]["entries_enabled"] is False
    assert book["account"]["fill_count"] == 0 and book["profitability_trial_started"] is False
    _, observed = track_preview(client, path, report)
    assert observed["recorded"] == observed["resolved"] == 0


def test_gap_and_recovery_preserve_existing_paper_and_forward_books(scanner, monkeypatch):
    client, config, models = scanner
    original = gap(client, monkeypatch)
    path, report = decisions.create_decision_preview(client, config, models, track_outcomes=True)
    first = monitor_preview(client, path, report)
    forecast_path, first_counts = track_preview(client, path, report)
    assert first_counts["recorded"] == 7
    assert first["entry_reviews"]["xyz:HOOD"]["execution_status"] == "block"
    monkeypatch.setattr(client, "info", original)
    path, report = decisions.create_decision_preview(client, config, models, track_outcomes=True)
    assert report["decisions"]["xyz:HOOD"]["features"] is not None
    assert report["market_snapshot"]["assets"]["xyz:HOOD"]["signal_ready"] is True
    second = monitor_preview(client, path, report)
    second_path, second_counts = track_preview(client, path, report)
    assert second["account_path"] == first["account_path"] and second_path == forecast_path
    assert second["account"]["wallet_balance"] == 10000 and second["account"]["fill_count"] == 0
    assert second_counts["recorded"] == 8 and second_counts["pending"] == 8


def test_existing_affected_outcome_stays_pending_while_other_due_outcomes_resolve(scanner, monkeypatch):
    client, config, models = scanner
    path, report = decisions.create_decision_preview(client, config, models, track_outcomes=True)
    _, initial = track_preview(client, path, report)
    assert initial["recorded"] == 8
    gap(client, monkeypatch)
    path, report = decisions.create_decision_preview(client, config, models, track_outcomes=True)
    client.clock = lambda: NOW+timedelta(minutes=40)
    _, later = track_preview(client, path, report)
    assert later["recorded"] == 8 and later["pending"] == 1 and later["resolved"] == 7
    assert later["confidence"] is None and later["reliability_validated"] is False


def test_malformed_candles_are_not_treated_as_ordinary_missing_intervals(scanner, monkeypatch):
    client, config, models = scanner
    original = client.info
    def malformed(payload):
        result = original(payload)
        if payload["type"] == "candleSnapshot" and payload["req"]["coin"] == "xyz:HOOD":
            result[0]["h"] = "1"
        return result
    monkeypatch.setattr(client, "info", malformed)
    with pytest.raises(ValueError, match="invalid OHLC"):
        decisions.create_decision_preview(client, config, models)
    assert not (client.settings.observations.parent / "decisions").exists()


def test_stale_book_still_refuses_even_for_market_with_candle_gap(scanner, monkeypatch):
    client, config, models = scanner
    gap(client, monkeypatch)
    original = client.info
    def stale(payload):
        result = original(payload)
        if payload["type"] == "l2Book" and payload["coin"] == "xyz:HOOD":
            result["time"] -= 31000
        return result
    monkeypatch.setattr(client, "info", stale)
    with pytest.raises(ValueError, match="stale"):
        decisions.create_decision_preview(client, config, models)
    assert not (client.settings.observations.parent / "decisions").exists()
