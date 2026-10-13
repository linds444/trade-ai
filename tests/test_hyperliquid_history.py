from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from local_ai_trader.cli import main
from local_ai_trader.features.build_features import FEATURE_COLUMNS
from local_ai_trader.hyperliquid import feed, history
from local_ai_trader.models.train import file_digest

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 1, 5, 0, 5, 10, tzinfo=timezone.utc)
END = int(NOW.timestamp()) // 300 * 300


def candles(coin, start, end):
    return [{"s": coin, "i": "5m", "t": stamp * 1000, "T": (stamp + 300) * 1000 - 1,
             "o": str(100 + index / 100), "c": str(100.05 + index / 100),
             "h": str(100.1 + index / 100), "l": str(99.9 + index / 100), "v": "10", "n": 3}
            for index, stamp in enumerate(range(start, end, 300))]


def contexts(coins):
    return [{"collateralToken": 0, "universe": [
        {"name": coin, "szDecimals": 4, "maxLeverage": 20,
         **({"deployerFeeScale": "1", "growthMode": "enabled"} if ":" in coin else {})}
        for coin in coins]}, [{"markPx": "100", "oraclePx": "100", "funding": "0.00001"} for _ in coins]]


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "config/hyperliquid.toml"
    path.parent.mkdir()
    text = (ROOT / "config/hyperliquid.toml").read_text(encoding="utf-8")
    path.write_text(text.replace('coins = ["xyz:HOOD", "xyz:NVDA", "SOL", "ETH", "xyz:TSLA", "BTC", "kPEPE", "DOGE"]', 'coins = ["BTC", "ETH"]'), encoding="utf-8")
    return path


def fake_client(config, monkeypatch, transform=None):
    settings = feed.load_hyperliquid_settings(config)
    client = feed.HyperliquidPublicClient(settings, clock=lambda: NOW)
    requests = []
    def info(payload):
        requests.append(payload)
        if payload["type"] == "metaAndAssetCtxs":
            dex = payload.get("dex", "")
            coins = [coin for coin in settings.coins if feed.BY_COIN[coin].dex == dex]
            return contexts(coins)
        assert payload["type"] == "candleSnapshot"
        req = payload["req"]
        result = candles(req["coin"], req["startTime"] // 1000, req["endTime"] // 1000)
        # Real API commonly returns the forming candle at the inclusive end.
        result += candles(req["coin"], req["endTime"] // 1000, req["endTime"] // 1000 + 300)
        return transform(result) if transform else result
    monkeypatch.setattr(client, "info", info)
    return client, requests


def raw_for(settings, bars=500):
    start = END - bars * 300
    dexes = {feed.BY_COIN[coin].dex for coin in settings.coins}
    return {"start_inclusive": start, "end_exclusive": END, "captured_at": NOW.isoformat(),
            "candles": {coin: candles(coin, start, END) for coin in settings.coins},
            "candles_received_at": {coin: NOW.isoformat() for coin in settings.coins},
            "contexts": {dex: contexts([coin for coin in settings.coins if feed.BY_COIN[coin].dex == dex]) for dex in dexes},
            "contexts_received_at": {dex: NOW.isoformat() for dex in dexes}}


def test_prepares_all_eight_native_markets_without_orders_or_new_cost_assumptions(config, monkeypatch):
    config.write_text((ROOT / "config/hyperliquid.toml").read_text(encoding="utf-8"), encoding="utf-8")
    client, requests = fake_client(config, monkeypatch)
    output, report = history.create_history_bundle(client, config, bars=500)
    assert history.load_history_bundle(output) == report
    assert len(report["assets"]) == 8
    assert report["protocol"]["costs_in_labels"] is False
    assert report["protocol"]["funding_in_labels"] is False
    assert report["protocol"]["feature_names"] == list(FEATURE_COLUMNS)
    assert all(a["status"] == "ready" for a in report["assets"].values())
    assert {p["type"] for p in requests} == {"candleSnapshot", "metaAndAssetCtxs"}
    pepe = pd.read_parquet(output / "kPEPE/candles.parquet")
    assert pepe.open.iloc[0] == 100
    assert report["assets"]["kPEPE"]["underlying_units_per_contract"] == 1000
    assert (output / "xyz_HOOD/train.parquet").exists()
    assert not any(":" in name for name in report["artifact_hashes"])
    for asset in report["assets"].values():
        assert asset["quality"]["cleaned_rows"] == 500
        assert asset["quality"]["out_of_range_rows_removed"] == 1
        assert asset["research_rows"] == 480
        assert asset["partitions"]["train"]["rows"] == 280
        assert asset["partitions"]["validation"]["rows"] == 88
        assert asset["partitions"]["test"]["rows"] == 96


def test_label_times_and_returns_match_delayed_entry_and_six_bar_holding(config):
    settings = feed.load_hyperliquid_settings(config)
    raw = raw_for(settings)
    assets, frames = history.derive_history(raw, settings, history.research_protocol(settings, 500))
    rows = frames["BTC/features.parquet"]
    start = datetime.fromtimestamp(raw["start_inclusive"], timezone.utc)
    first = rows.iloc[0]
    assert first.timestamp.to_pydatetime() == start + timedelta(minutes=60)
    assert first.available_at.to_pydatetime() == start + timedelta(minutes=65)
    assert first.execution_entry_at.to_pydatetime() == first.available_at.to_pydatetime() + timedelta(minutes=5)
    assert first.execution_exit_at.to_pydatetime() == first.execution_entry_at.to_pydatetime() + timedelta(minutes=30)
    assert first.execution_target_available_at.to_pydatetime() == first.available_at.to_pydatetime() + timedelta(minutes=40)
    assert first.execution_target_return == pytest.approx(100.20 / 100.14 - 1)
    assert first.target_class == "flat"
    assert rows.execution_target_available_at.max() <= pd.Timestamp(NOW)
    assert assets["BTC"]["discarded_window_rows"] == 20


def test_future_prices_change_labels_but_do_not_change_current_features(config):
    settings = feed.load_hyperliquid_settings(config)
    raw = raw_for(settings)
    _, original = history.derive_history(raw, settings, history.research_protocol(settings, 500))
    for candle in raw["candles"]["BTC"][20:]:
        for name in ("o", "h", "l", "c"):
            candle[name] = str(float(candle[name]) * 2)
    _, changed = history.derive_history(raw, settings, history.research_protocol(settings, 500))
    a, b = original["BTC/features.parquet"], changed["BTC/features.parquet"]
    np.testing.assert_array_equal(a.loc[0, list(FEATURE_COLUMNS)], b.loc[0, list(FEATURE_COLUMNS)])
    assert a.execution_target_return.iloc[1] != b.execution_target_return.iloc[1]


def test_gaps_reset_features_and_never_create_labels_across_missing_intervals(config):
    settings = feed.load_hyperliquid_settings(config)
    raw = raw_for(settings)
    missing = raw["candles"]["BTC"].pop(250)["t"]
    assets, frames = history.derive_history(raw, settings, history.research_protocol(settings, 500))
    asset, rows = assets["BTC"], frames["BTC/features.parquet"]
    assert asset["quality"]["missing_count"] == 1
    assert asset["research_rows"] == 459
    assert len(asset["segments"]) == 2
    missing_at = pd.Timestamp(missing, unit="ms", tz="UTC")
    assert not ((rows.timestamp < missing_at) & (rows.execution_target_available_at > missing_at)).any()
    later = rows.loc[rows.segment_id == 2].iloc[0]
    assert later.timestamp.to_pydatetime() == missing_at.to_pydatetime() + timedelta(minutes=65)
    assert later.feature_return_1 == pytest.approx(102.68 / 102.67 - 1)


def test_short_segments_are_excluded_and_unavailable_markets_are_named(config, monkeypatch):
    def short_segments(result):
        return [c for i, c in enumerate(result) if i % 15 != 0]
    client, _ = fake_client(config, monkeypatch, short_segments)
    output, report = history.create_history_bundle(client, config, bars=500)
    for asset in report["assets"].values():
        assert asset["status"] == "insufficient_history"
        assert asset["research_rows"] == 0
    assert not (output / "BTC/train.parquet").exists()
    assert not (output / "BTC/features.parquet").exists()
    assert history.load_history_bundle(output) == report


def test_empty_market_is_archived_as_insufficient_without_inventing_candles(config, monkeypatch):
    client, _ = fake_client(config, monkeypatch, lambda result: [] if result[0]["s"] == "BTC" else result)
    output, report = history.create_history_bundle(client, config, bars=500)
    assert report["assets"]["BTC"]["quality"]["missing_count"] == 500
    assert report["assets"]["BTC"]["status"] == "insufficient_history"
    assert pd.read_parquet(output / "BTC/candles.parquet").empty
    assert report["assets"]["ETH"]["status"] == "ready"


def test_purge_includes_equality_and_does_not_read_future_labels_into_earlier_partition(config):
    settings = feed.load_hyperliquid_settings(config)
    _, frames = history.derive_history(raw_for(settings), settings, history.research_protocol(settings, 500))
    partitions, report = history.split_research_rows(frames["BTC/features.parquet"])
    assert report["status"] == "ready"
    assert partitions["train"].execution_target_available_at.max() < partitions["validation"].available_at.min()
    assert partitions["validation"].execution_target_available_at.max() < partitions["test"].available_at.min()
    assert report["partitions"]["train"]["purged_rows"] == 8
    assert report["partitions"]["validation"]["purged_rows"] == 8


@pytest.mark.parametrize("bars", [True, 127, 5000, -1, 300.0])
def test_bad_requested_history_sizes_are_refused(config, bars):
    with pytest.raises(ValueError, match="128–4999"):
        history.research_protocol(feed.load_hyperliquid_settings(config), bars)


@pytest.mark.parametrize("change", [
    lambda r: r["candles"]["BTC"][0].update(s="ETH"),
    lambda r: r["candles"]["BTC"][0].update(n=True),
    lambda r: r["candles"]["BTC"][0].update(T=0),
    lambda r: r["candles"]["BTC"][0].update(o="nan"),
    lambda r: r["candles"]["BTC"][0].update(h="1"),
    lambda r: r["contexts"][""][0].update(collateralToken=360),
    lambda r: r["contexts"][""][0]["universe"][0].update(isDelisted=True),
    lambda r: r["candles_received_at"].update(BTC="2026-01-05T00:05:11+00:00"),
    lambda r: r["contexts_received_at"].update({"": "2026-01-04T00:05:11+00:00"}),
    lambda r: r.update(end_exclusive=END+300),
])
def test_invalid_source_or_provenance_is_refused(config, change):
    settings = feed.load_hyperliquid_settings(config)
    raw = raw_for(settings)
    change(raw)
    with pytest.raises(ValueError):
        history.derive_history(raw, settings, history.research_protocol(settings, 500))


def test_identical_duplicates_are_audited_and_conflicting_duplicates_refused(config):
    settings = feed.load_hyperliquid_settings(config)
    raw = raw_for(settings)
    raw["candles"]["BTC"].append(deepcopy(raw["candles"]["BTC"][0]))
    assets, _ = history.derive_history(raw, settings, history.research_protocol(settings, 500))
    assert assets["BTC"]["quality"]["duplicate_rows_removed"] == 1
    raw["candles"]["BTC"][-1]["v"] = "11"
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        history.derive_history(raw, settings, history.research_protocol(settings, 500))


@pytest.mark.parametrize("artifact", ["input_config.toml", "raw.json", "BTC/train.parquet", "BTC/test.parquet"])
def test_artifact_tampering_is_refused(config, monkeypatch, artifact):
    client, _ = fake_client(config, monkeypatch)
    output, _ = history.create_history_bundle(client, config, bars=500)
    with (output / artifact).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        history.load_history_bundle(output)


@pytest.mark.parametrize("change", [
    lambda r: r["protocol"].update(latency_bars=2),
    lambda r: r["assets"]["BTC"].update(research_rows=9999),
    lambda r: r.update(models_fitted=True),
    lambda r: r["artifact_hashes"].update({"../outside.json": "x"}),
])
def test_summary_or_protocol_tampering_is_refused(config, monkeypatch, change):
    client, _ = fake_client(config, monkeypatch)
    output, report = history.create_history_bundle(client, config, bars=500)
    change(report)
    feed.write_json(output / "history.json", report)
    with pytest.raises(ValueError):
        history.load_history_bundle(output)


def test_changed_features_with_recomputed_hash_still_fail_raw_reconstruction(config, monkeypatch):
    client, _ = fake_client(config, monkeypatch)
    output, report = history.create_history_bundle(client, config, bars=500)
    path = output / "BTC/train.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "feature_return_1"] = 999
    feed.write_parquet(path, frame)
    report["artifact_hashes"]["BTC/train.parquet"] = file_digest(path)
    feed.write_json(output / "history.json", report)
    with pytest.raises(ValueError, match="rows disagree"):
        history.load_history_bundle(output)


def test_config_mutation_and_write_failure_publish_no_bundle(config, monkeypatch):
    client, _ = fake_client(config, monkeypatch)
    original = history.load_history_bundle
    def audit_and_change(path):
        result = original(path)
        config.write_text(config.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
        return result
    monkeypatch.setattr(history, "load_history_bundle", audit_and_change)
    with pytest.raises(ValueError, match="configuration changed"):
        history.create_history_bundle(client, config, bars=500)
    parent = client.settings.observations.parent / "research"
    assert list(parent.iterdir()) == []


def test_partial_write_failure_cleans_staging_and_leaves_existing_bundle_intact(config, monkeypatch):
    client, _ = fake_client(config, monkeypatch)
    output, report = history.create_history_bundle(client, config, bars=500)
    saved = {name: file_digest(output / name) for name in report["artifact_hashes"]}
    original = history.write_parquet
    calls = 0
    def fail_midway(path, frame):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected write failure")
        original(path, frame)
    monkeypatch.setattr(history, "write_parquet", fail_midway)
    with pytest.raises(OSError, match="injected"):
        history.create_history_bundle(client, config, bars=500)
    assert list(output.parent.iterdir()) == [output]
    assert saved == {name: file_digest(output / name) for name in report["artifact_hashes"]}


def test_minimum_history_is_archived_but_not_presented_as_trainable(config, monkeypatch):
    client, _ = fake_client(config, monkeypatch)
    output, report = history.create_history_bundle(client, config, bars=128)
    assert report["assets"]["BTC"]["research_rows"] == 108
    assert report["assets"]["BTC"]["status"] == "insufficient_history"
    assert (output / "BTC/features.parquet").is_file()
    assert not (output / "BTC/train.parquet").exists()


def test_gross_class_labels_and_threshold_are_independent_of_cost_inputs(config):
    settings = feed.load_hyperliquid_settings(config)
    raw = raw_for(settings)
    for index in (20, 21):
        candle = raw["candles"]["BTC"][index]
        candle.update(o="120", h="121")
    for index in (22, 23):
        candle = raw["candles"]["BTC"][index]
        candle.update(o="80", l="79")
    _, frames = history.derive_history(raw, settings, history.research_protocol(settings, 500))
    rows = frames["BTC/features.parquet"]
    assert rows.target_class.iloc[0] == "up"
    assert rows.target_class.iloc[2] == "down"
    assert rows.target_class.iloc[4] == "flat"


def test_deep_native_paths_without_windows_registry_changes(config, monkeypatch):
    path = feed.native_path(config.parent.parent / ("long-root-" + "x"*70) / ("nested-" + "y"*70) / ("leaf-" + "z"*70) / "config/hyperliquid.toml")
    path.parent.mkdir(parents=True)
    path.write_bytes(config.read_bytes())
    client, _ = fake_client(path, monkeypatch)
    output, report = history.create_history_bundle(client, path, bars=500)
    assert len(str(output / "BTC/train.parquet")) > 310
    assert history.load_history_bundle(output) == report


def test_cli_prints_market_counts_and_preparation_only_status(config, monkeypatch, capsys):
    client, _ = fake_client(config, monkeypatch)
    monkeypatch.setattr(history, "HyperliquidPublicClient", lambda settings: client)
    assert main(["hyperliquid-history", "--config", str(config), "--bars", "500"]) == 0
    output = capsys.readouterr().out
    assert "2/2 markets have prepared partitions" in output
    assert "test partitions unscored; reliability unvalidated" in output
    assert "Saved Hyperliquid research to" in output


def test_live_feed_still_refuses_missing_candles_even_though_research_handles_gaps():
    payload = candles("BTC", END-500*300, END)
    payload.pop(200)
    _, quality = feed.clean_hyperliquid_candles(payload, "BTC", END-500*300, END)
    assert quality.missing_count == 1
    with pytest.raises(ValueError, match="missing completed candles"):
        feed.parse_candles(payload, "BTC", END-500*300, END)
