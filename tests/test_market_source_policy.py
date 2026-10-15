from dataclasses import asdict, replace
from pathlib import Path

import pytest

from local_ai_trader.hyperliquid import feed
from local_ai_trader.hyperliquid.markets import BY_COIN, MARKETS
from local_ai_trader.markets.source import generate_offline_states, live_market_state
from local_ai_trader.risk.entry_policy import EntryPolicy, SignalEvidence, evaluate_entry_signal
from test_hyperliquid_feed import NOW, book, candles, config

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = SignalEvidence("BUY", 0.60, "calibration-artifact", 0.70, "reliability-validation-artifact")


def test_default_roster_and_policy_match_requested_eight_markets():
    settings = feed.load_hyperliquid_settings(ROOT / "config/hyperliquid.toml")
    assert settings.coins == tuple(m.coin for m in MARKETS)
    assert [m.symbol for m in MARKETS] == ["HOOD", "NVDA", "SOL/USD", "ETH/USD", "TSLA", "BTC/USD", "PEPE/USD", "DOGE/USD"]
    assert settings.entry_policy == EntryPolicy(0.60, 0.70, 5)


def test_live_pepe_display_converts_price_but_never_notional_volume_or_features():
    state = live_market_state(BY_COIN["kPEPE"], {"mark_price": 0.01, "previous_day_price": 0.008, "day_notional_volume": 12000},
                              {"mid": 0.01, "exchange_time": NOW.isoformat()}, {"feature_momentum": 0.005})
    assert state.price == pytest.approx(0.00001)
    assert state.contract_price == 0.01
    assert state.underlying_units_per_contract == 1000
    assert state.volume_24h_notional == 12000
    assert state.change_24h == pytest.approx(0.25)
    assert state.momentum == 0.005
    assert state.source_mode == "observed"
    assert state.news is None and state.regime == "unavailable"


def test_offline_and_live_have_same_shape_but_generated_data_is_explicit():
    generated = generate_offline_states(NOW, seed=42)
    assert generated == generate_offline_states(NOW, seed=42)
    assert generated != generate_offline_states(NOW, seed=43)
    assert len(generated) == 8
    assert all(x.source_mode == "synthetic" and x.regime_method == "generated_test_input" for x in generated)
    live = live_market_state(MARKETS[0], {"mark_price": 100}, {"mid": 100, "exchange_time": NOW.isoformat()}, {"feature_momentum": 0})
    assert set(asdict(live)) == set(asdict(generated[0]))
    assert live.change_24h is None and live.volume_24h_notional is None


def expanded_config(config):
    config.write_text((ROOT / "config/hyperliquid.toml").read_text(encoding="utf-8"), encoding="utf-8")
    return config


def mock_eight_client(config, monkeypatch, *, collateral=0):
    settings = feed.load_hyperliquid_settings(expanded_config(config))
    client = feed.HyperliquidPublicClient(settings, clock=lambda: NOW)
    calls = []
    def info(payload):
        calls.append(payload)
        if payload["type"] == "metaAndAssetCtxs":
            selected = [m for m in MARKETS if m.dex == payload.get("dex", "")]
            return [{"collateralToken": collateral, "universe": [
                {"name": m.coin, "szDecimals": 3, "maxLeverage": 10,
                 **({"deployerFeeScale": "1.0", "growthMode": "enabled"} if m.dex else {})} for m in selected]},
                [{"markPx": "100.5", "oraclePx": "100.7", "funding": "0.0000125", "dayNtlVlm": "5000", "prevDayPx": "99"} for _ in selected]]
        if payload["type"] == "l2Book":
            return book(payload["coin"])
        req = payload["req"]
        return candles(req["coin"], req["startTime"] // 1000, req["endTime"] // 1000)
    monkeypatch.setattr(client, "info", info)
    return client, calls


def test_eight_market_snapshot_routes_exact_dex_and_contracts_and_preserves_units(config, monkeypatch):
    client, calls = mock_eight_client(config, monkeypatch)
    report, raw, frames, features = client.snapshot()
    assert set(report["assets"]) == set(BY_COIN)
    assert len(calls) == 18
    assert {x.get("dex", "") for x in calls if x["type"] == "metaAndAssetCtxs"} == {"", "xyz"}
    assert set(raw) >= {"contexts", "xyz_contexts"}
    assert frames["xyz:HOOD"].symbol.eq("xyz:HOOD-USDC").all()
    assert frames["kPEPE"].close.iloc[0] == 100  # No display-unit conversion in model inputs.
    assert report["assets"]["kPEPE"]["market_state"]["price"] == pytest.approx(0.1005)
    assert report["assets"]["BTC"]["taker_fee_bps_assumption"] == 4.5
    assert report["assets"]["xyz:HOOD"]["taker_fee_bps_assumption"] == pytest.approx(0.9)
    assert report["entry_policy"] == asdict(EntryPolicy())
    assert report["signal_gate_connected"] is False
    assert report["forward_strategy_started"] is False
    assert report["paper_fills_created"] is False


@pytest.mark.parametrize("collateral", [360, None, False, "0"])
def test_different_or_unverified_dex_collateral_refuses_entire_observation(config, monkeypatch, collateral):
    client, _ = mock_eight_client(config, monkeypatch, collateral=collateral)
    with pytest.raises(ValueError, match="USDC"):
        client.snapshot()


def test_eight_market_files_are_windows_safe_with_original_contracts_in_reports(config, monkeypatch):
    client, _ = mock_eight_client(config, monkeypatch)
    output, report = feed.save_hyperliquid_snapshot(client, config)
    assert len(list(output.glob("*_candles.parquet"))) == 8
    assert (output / "xyz_HOOD_candles.parquet").is_file()
    assert all(":" not in name for name in report["artifact_hashes"])
    assert report["assets"]["xyz:HOOD"]["contract_identifier"] == "xyz:HOOD"


@pytest.mark.parametrize("changes", [
    {"growthMode": "unknown"}, {"deployerFeeScale": "nan"},
    {"deployerFeeScale": "-1"}, {"deployerFeeScale": "4"},
])
def test_hip3_unknown_fee_settings_refuse_observation(changes):
    metadata = [{"universe": [{"name": "xyz:HOOD", "szDecimals": 3, "maxLeverage": 10,
                              "growthMode": "enabled", "deployerFeeScale": "1", **changes}]},
                [{"markPx": "100", "oraclePx": "100", "funding": "0"}]]
    with pytest.raises(ValueError):
        feed.parse_contexts(metadata, ("xyz:HOOD",), NOW)


def test_one_bad_market_never_publishes_partial_universe(config, monkeypatch):
    client, _ = mock_eight_client(config, monkeypatch)
    original = client.info
    def info(payload):
        result = original(payload)
        if payload["type"] == "l2Book" and payload["coin"] == "xyz:NVDA":
            result["time"] -= 31000
        return result
    monkeypatch.setattr(client, "info", info)
    with pytest.raises(ValueError, match="stale"):
        feed.save_hyperliquid_snapshot(client, config)
    assert not client.settings.observations.exists()


@pytest.mark.parametrize("payload", [
    {"type": "metaAndAssetCtxs", "dex": "cash"},
    {"type": "metaAndAssetCtxs", "dex": "xyz", "signature": "bad"},
    {"type": "metaAndAssetCtxs", "dex": "xyz", "user": "wallet"},
])
def test_unconfigured_or_nonpublic_dex_requests_never_reach_http(config, monkeypatch, payload):
    settings = feed.load_hyperliquid_settings(expanded_config(config))
    def forbidden(*args, **kwargs):
        pytest.fail("No HTTP request expected")
    monkeypatch.setattr(feed, "urlopen", forbidden)
    with pytest.raises(ValueError):
        feed.HyperliquidPublicClient(settings).info(payload)


def test_exact_thresholds_allow_only_next_account_risk_check():
    result = evaluate_entry_signal(EVIDENCE, 4)
    assert result.status == "PASS" and result.eligible_for_account_risk_check
    assert result.reason == "signal_policy_passed"


@pytest.mark.parametrize("changes,status,reason", [
    ({"action": "HOLD"}, "BLOCK", "action_not_entry"),
    ({"action": "AVOID"}, "BLOCK", "action_not_entry"),
    ({"action": "SELL"}, "BLOCK", "action_not_entry"),
    ({"action": "SHORT"}, "BLOCK", "invalid_action"),
    ({"probability": 0.599999}, "BLOCK", "probability_below_minimum"),
    ({"probability": None}, "REVIEW", "calibrated_probability_unavailable"),
    ({"probability": float("nan")}, "REVIEW", "calibrated_probability_unavailable"),
    ({"probability": True}, "REVIEW", "calibrated_probability_unavailable"),
    ({"calibration_id": " "}, "REVIEW", "calibrated_probability_unavailable"),
    ({"confidence": None}, "REVIEW", "validated_confidence_unavailable"),
    ({"confidence": float("inf")}, "REVIEW", "validated_confidence_unavailable"),
    ({"confidence": 1.1}, "REVIEW", "validated_confidence_unavailable"),
    ({"reliability_validation_id": None}, "REVIEW", "validated_confidence_unavailable"),
    ({"confidence": 0.699999}, "REVIEW", "confidence_below_minimum"),
])
def test_signal_gate_names_each_skip_rule(changes, status, reason):
    result = evaluate_entry_signal(replace(EVIDENCE, **changes), 0)
    assert (result.status, result.reason) == (status, reason)
    assert result.eligible_for_account_risk_check is False


@pytest.mark.parametrize("occupied", [5, 6])
def test_five_position_cap_counts_pending_reservations(occupied):
    result = evaluate_entry_signal(EVIDENCE, occupied)
    assert result.status == "BLOCK" and result.reason == "position_count_limit"


@pytest.mark.parametrize("occupied", [-1, False, 2.5])
def test_invalid_account_count_is_refused(occupied):
    with pytest.raises(ValueError):
        evaluate_entry_signal(EVIDENCE, occupied)


@pytest.mark.parametrize("kwargs", [{"minimum_probability": float("nan")}, {"minimum_confidence": True}, {"maximum_positions": 0}, {"maximum_positions": 5.5}])
def test_invalid_policy_is_refused(kwargs):
    with pytest.raises(ValueError):
        EntryPolicy(**kwargs)
