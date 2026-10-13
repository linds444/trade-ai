from copy import deepcopy
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import pandas as pd
import pytest

from local_ai_trader.cli import main
from local_ai_trader.features.build_features import FEATURE_COLUMNS
from local_ai_trader.hyperliquid import feed
from local_ai_trader.models.train import file_digest

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 1, 5, 0, 5, 10, tzinfo=timezone.utc)
END = int(NOW.timestamp()) // 300 * 300
START = END - 24 * 300


def book(coin="BTC", now=NOW):
    return {"coin": coin, "time": int(now.timestamp() * 1000), "levels": [
        [{"px": "100", "sz": "1.2", "n": 2}, {"px": "99", "sz": "2", "n": 1}],
        [{"px": "101", "sz": "1.1", "n": 2}, {"px": "102", "sz": "3", "n": 1}],
    ]}


def contexts():
    return [{"collateralToken": 0, "universe": [{"name": "BTC", "szDecimals": 5, "maxLeverage": 40}, {"name": "ETH", "szDecimals": 4, "maxLeverage": 25}]},
            [{"markPx": "100.5", "oraclePx": "100.7", "funding": "0.0000125"}, {"markPx": "101.5", "oraclePx": "101.7", "funding": "-0.0001"}]]


def candles(coin="BTC", start=START, end=END):
    return [{"s": coin, "i": "5m", "t": stamp * 1000, "T": (stamp + 300) * 1000 - 1,
             "o": "100", "c": str(100 + (index % 4) * 0.1), "h": "101", "l": "99", "v": "10", "n": 12}
            for index, stamp in enumerate(range(start, end, 300))]


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "config/hyperliquid.toml"
    path.parent.mkdir()
    text = (ROOT / "config/hyperliquid.toml").read_text(encoding="utf-8")
    path.write_text(text.replace('coins = ["xyz:HOOD", "xyz:NVDA", "SOL", "ETH", "xyz:TSLA", "BTC", "kPEPE", "DOGE"]', 'coins = ["BTC", "ETH"]'), encoding="utf-8")
    return path


@pytest.fixture
def client(config, monkeypatch):
    client = feed.HyperliquidPublicClient(feed.load_hyperliquid_settings(config), clock=lambda: NOW)
    def info(payload):
        if payload["type"] == "metaAndAssetCtxs":
            return contexts()
        if payload["type"] == "l2Book":
            return book(payload["coin"])
        request = payload["req"]
        return candles(request["coin"], request["startTime"] // 1000, request["endTime"] // 1000)
    monkeypatch.setattr(client, "info", info)
    return client


def test_config_binds_confirmed_perpetual_setup_and_never_enables_live(config):
    settings = feed.load_hyperliquid_settings(config)
    assert settings.coins == ("BTC", "ETH")
    assert settings.minimum_forward_days == 30
    assert settings.taker_fee_bps == 4.5
    assert settings.history_bars == 24
    assert settings.observations == feed.native_path(config.parent.parent / "data/hyperliquid/live").resolve()


@pytest.mark.parametrize("old,new", [
    ('live_execution_enabled = false', 'live_execution_enabled = true'),
    ('leverage = 1', 'leverage = 2'),
    ('leverage = 1', 'leverage = true'),
    ('long_only = true', 'long_only = false'),
    ('instrument = "perpetual"', 'instrument = "spot"'),
    ('collateral_currency = "USDC"', 'collateral_currency = "USD"'),
    ('coins = ["BTC", "ETH"]', 'coins = ["BTC", "BTC"]'),
    ('minimum_forward_days = 30', 'minimum_forward_days = 29'),
    ('history_bars = 24', 'history_bars = 12'),
    ('maximum_book_age_seconds = 30', 'maximum_book_age_seconds = 31'),
    ('poll_seconds = 10', 'poll_seconds = 0'),
    ('taker_fee_bps = 4.5', 'taker_fee_bps = nan'),
])
def test_invalid_or_unconfirmed_contract_configuration_is_refused(config, old, new):
    config.write_text(config.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError):
        feed.load_hyperliquid_settings(config)


def test_book_uses_exchange_time_and_actual_spread():
    result = feed.parse_book(book(), "BTC", NOW, 30)
    assert result["mid"] == 100.5
    assert result["bid_size"] == 1.2
    assert result["spread_bps"] == pytest.approx(1 / 100.5 * 10000)
    assert result["age_seconds"] == 0
    assert datetime.fromisoformat(result["exchange_time"]) == NOW


@pytest.mark.parametrize("mutation", [
    lambda data: data.update(coin="ETH"),
    lambda data: data.update(time=int((NOW - timedelta(seconds=30)).timestamp() * 1000)),
    lambda data: data.update(time=int((NOW + timedelta(milliseconds=1)).timestamp() * 1000)),
    lambda data: data.update(time=True),
    lambda data: data.update(levels=[[], []]),
    lambda data: data["levels"][0][0].update(px="101"),
    lambda data: data["levels"][0][0].update(sz="0"),
    lambda data: data["levels"][0][0].update(px="NaN"),
    lambda data: data["levels"][0][0].update(n=0),
    lambda data: data["levels"][0].reverse(),
])
def test_invalid_future_stale_or_crossed_books_are_refused(mutation):
    payload = book()
    mutation(payload)
    with pytest.raises(ValueError):
        feed.parse_book(payload, "BTC", NOW, 30)


def test_contexts_capture_funding_sign_and_do_not_invent_exchange_timestamp():
    result = feed.parse_contexts(contexts(), ("BTC", "ETH"), NOW)
    assert result["BTC"]["hourly_funding_rate"] == 0.0000125
    assert result["ETH"]["hourly_funding_rate"] == -0.0001
    assert result["BTC"]["context_exchange_time"] is None
    assert result["BTC"]["context_received_at"] == NOW.isoformat()
    assert result["BTC"]["exchange_max_leverage"] == 40


@pytest.mark.parametrize("mutation", [
    lambda data: data[0]["universe"].pop(),
    lambda data: data[0]["universe"][0].update(name="SOL"),
    lambda data: data[0]["universe"][0].update(szDecimals=True),
    lambda data: data[0]["universe"][0].update(isDelisted=True),
    lambda data: data[1][0].update(funding="0.05"),
    lambda data: data[1][0].update(funding="NaN"),
    lambda data: data[1][0].update(oraclePx="0"),
])
def test_missing_delisted_or_invalid_contract_contexts_are_refused(mutation):
    payload = contexts()
    mutation(payload)
    with pytest.raises(ValueError):
        feed.parse_contexts(payload, ("BTC", "ETH"), NOW)


def test_unfinished_candle_is_excluded_and_never_changes_features(client, monkeypatch):
    before, _, frames, features = client.snapshot()
    original = client.info
    def with_future(payload):
        data = original(payload)
        if payload["type"] == "candleSnapshot":
            request = payload["req"]
            future = candles(request["coin"], END, END + 300)[0]
            future.update(o="99999", c="99999", h="99999", l="99999", v="99999")
            return data + [future]
        return data
    monkeypatch.setattr(client, "info", with_future)
    after, _, changed_frames, changed_features = client.snapshot()
    for coin in ("BTC", "ETH"):
        pd.testing.assert_frame_equal(changed_frames[coin], frames[coin])
        pd.testing.assert_frame_equal(changed_features[coin], features[coin])
        assert after["assets"][coin]["quality"]["out_of_range_rows_removed"] == 1
        assert after["assets"][coin]["latest_features"] == before["assets"][coin]["latest_features"]
        assert frames[coin].available_at.max() == pd.Timestamp(END, unit="s", tz="UTC")


@pytest.mark.parametrize("mutation", [
    lambda rows: rows.pop(),
    lambda rows: rows[0].update(s="ETH"),
    lambda rows: rows[0].update(i="1m"),
    lambda rows: rows[0].update(T=rows[0]["T"] + 1),
    lambda rows: rows[0].update(t=rows[0]["t"] + 1),
    lambda rows: rows[0].update(c="NaN"),
    lambda rows: rows[0].update(v="-1"),
])
def test_candle_gaps_market_timing_and_bad_ohlcv_are_refused(mutation):
    payload = candles()
    mutation(payload)
    with pytest.raises(ValueError):
        feed.parse_candles(payload, "BTC", START, END)


def test_identical_duplicate_candles_are_audited_and_conflicts_rejected():
    payload = candles()
    payload.append(deepcopy(payload[0]))
    _, quality = feed.parse_candles(payload, "BTC", START, END)
    assert quality.duplicate_rows_removed == 1
    payload[-1]["v"] = "100"
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        feed.parse_candles(payload, "BTC", START, END)


def test_zero_trade_zero_volume_candle_is_valid_without_imputation():
    payload = candles()
    payload[0].update(n=0, v="0")
    frame, quality = feed.parse_candles(payload, "BTC", START, END)
    assert frame.volume.iloc[0] == 0
    assert quality.missing_count == 0


def test_snapshot_contains_venue_bound_causal_inputs_and_no_fake_paper_performance(client):
    report, raw, frames, features = client.snapshot()
    assert report["exchange"] == "hyperliquid"
    assert report["instrument"] == "perpetual"
    assert report["collateral_currency"] == "USDC"
    assert report["configured_leverage"] == 1
    assert report["live_execution_enabled"] is False
    assert report["forward_strategy_started"] is False
    assert report["orders_created"] is report["paper_fills_created"] is False
    for coin in ("BTC", "ETH"):
        assert frames[coin].symbol.eq(f"{coin}-USDC").all()
        assert features[coin].exchange.eq("hyperliquid").all()
        assert len(frames[coin]) == 24
        assert len(features[coin]) == 12
        assert set(report["assets"][coin]["latest_features"]) == set(FEATURE_COLUMNS)
        assert not any(name.startswith(("target_", "execution_")) for name in features[coin].columns)


def test_saved_snapshot_hashes_raw_candles_and_features(config, client):
    output, report = feed.save_hyperliquid_snapshot(client, config)
    assert json.loads((output / "snapshot.json").read_text(encoding="utf-8")) == report
    assert report["input_config_sha256"] == file_digest(config)
    assert set(report["artifact_hashes"]) == {"raw.json", "BTC_candles.parquet", "ETH_candles.parquet", "BTC_features.parquet", "ETH_features.parquet"}
    for name, digest in report["artifact_hashes"].items():
        assert file_digest(output / name) == digest
    second, _ = feed.save_hyperliquid_snapshot(client, config)
    assert second != output


def test_failed_publish_removes_staging_and_publishes_nothing(config, client, monkeypatch):
    def fail(*args):
        raise OSError("simulated publish failure")
    monkeypatch.setattr(feed.os, "rename", fail)
    with pytest.raises(OSError):
        feed.save_hyperliquid_snapshot(client, config)
    assert list(client.settings.observations.iterdir()) == []


def test_changed_config_refuses_publication(config, client, monkeypatch):
    original = client.snapshot
    def changed():
        result = original()
        config.write_text(config.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
        return result
    monkeypatch.setattr(client, "snapshot", changed)
    with pytest.raises(ValueError, match="changed"):
        feed.save_hyperliquid_snapshot(client, config)
    assert list(client.settings.observations.iterdir()) == []


def test_stale_book_before_publish_is_refused(config, client, monkeypatch):
    original = client.snapshot
    def became_stale():
        result = original()
        monkeypatch.setattr(client, "clock", lambda: NOW + timedelta(seconds=30))
        return result
    monkeypatch.setattr(client, "snapshot", became_stale)
    with pytest.raises(ValueError, match="stale"):
        feed.save_hyperliquid_snapshot(client, config)
    assert list(client.settings.observations.iterdir()) == []


@pytest.mark.parametrize("kind", ["order", "cancel", "updateLeverage", "withdraw3", "approveAgent", "userFees"])
def test_order_wallet_and_unrequested_private_types_never_reach_http(config, monkeypatch, kind):
    def forbidden(*args, **kwargs):
        pytest.fail("Non-public request reached HTTP")
    monkeypatch.setattr(feed, "urlopen", forbidden)
    client = feed.HyperliquidPublicClient(feed.load_hyperliquid_settings(config))
    with pytest.raises(ValueError, match="allowlisted"):
        client.info({"type": kind})


def test_unexpected_signature_or_wallet_fields_do_not_reach_http(config, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Request with unexpected fields reached HTTP")
    monkeypatch.setattr(feed, "urlopen", forbidden)
    client = feed.HyperliquidPublicClient(feed.load_hyperliquid_settings(config))
    with pytest.raises(ValueError, match="Unexpected fields"):
        client.info({"type": "l2Book", "coin": "BTC", "signature": "unrequested"})


def test_failed_network_retries_are_bounded(config, monkeypatch):
    calls, sleeps = [], []
    def fail(*args, **kwargs):
        calls.append(1)
        raise URLError("temporary failure")
    monkeypatch.setattr(feed, "urlopen", fail)
    monkeypatch.setattr(feed.time, "sleep", sleeps.append)
    client = feed.HyperliquidPublicClient(feed.load_hyperliquid_settings(config))
    with pytest.raises(RuntimeError, match="retries exhausted"):
        client.info({"type": "l2Book", "coin": "BTC"})
    assert len(calls) == 3
    assert sleeps == [1, 2]


def test_watch_archives_once_per_closed_window_and_stops_cleanly(config, client, monkeypatch, capsys):
    monkeypatch.setattr(feed, "HyperliquidPublicClient", lambda settings: client)
    pauses = []
    def sleep(seconds):
        pauses.append(seconds)
        if len(pauses) == 3:
            raise KeyboardInterrupt()
    monkeypatch.setattr(feed.time, "sleep", sleep)
    feed.run_hyperliquid_feed(config, watch=True)
    assert len(list(client.settings.observations.iterdir())) == 1
    assert pauses == [10, 10, 10]
    assert "public feed stopped" in capsys.readouterr().out


def test_http_request_uses_only_info_endpoint_and_no_credentials(config, monkeypatch):
    captured = []
    def open_url(request, **kwargs):
        captured.append(request)
        return io.BytesIO(json.dumps(book()).encode())
    monkeypatch.setattr(feed, "urlopen", open_url)
    client = feed.HyperliquidPublicClient(feed.load_hyperliquid_settings(config))
    assert client.info({"type": "l2Book", "coin": "BTC"}) == book()
    request = captured[0]
    assert request.full_url == "https://api.hyperliquid.xyz/info"
    assert request.get_method() == "POST"
    assert json.loads(request.data) == {"type": "l2Book", "coin": "BTC"}
    assert not any(name.lower() in ("authorization", "api-key") for name, _ in request.header_items())


def test_transient_failure_retries_and_permanent_http_failure_does_not(config, monkeypatch):
    calls = []
    def open_url(request, **kwargs):
        calls.append(request)
        if len(calls) == 1:
            raise HTTPError(feed.INFO_URL, 429, "rate limited", {}, None)
        return io.BytesIO(json.dumps(book()).encode())
    monkeypatch.setattr(feed, "urlopen", open_url)
    sleeps = []
    monkeypatch.setattr(feed.time, "sleep", sleeps.append)
    client = feed.HyperliquidPublicClient(feed.load_hyperliquid_settings(config))
    client.info({"type": "l2Book", "coin": "BTC"})
    assert len(calls) == 2
    assert sleeps == [1]
    def denied(*args, **kwargs):
        raise HTTPError(feed.INFO_URL, 403, "forbidden", {}, None)
    monkeypatch.setattr(feed, "urlopen", denied)
    with pytest.raises(RuntimeError, match="HTTP 403"):
        client.info({"type": "l2Book", "coin": "BTC"})


def test_long_native_windows_observation_paths(config, client):
    root = config.parent.parent
    while len(str(root)) < 300:
        root /= "hyperliquid-deep-observation-directory"
    text = config.read_text(encoding="utf-8")
    # TOML literal strings preserve Windows backslashes without escape parsing.
    config.write_text(text.replace('observations = "data/hyperliquid/live"', f"observations = '{root}'"), encoding="utf-8")
    client.settings = feed.load_hyperliquid_settings(config)
    output, _ = feed.save_hyperliquid_snapshot(client, config)
    assert (output / "BTC_features.parquet").is_file()
    assert (output / "snapshot.json").is_file()


def test_hyperliquid_cli_uses_separate_config_and_data_only_route(config, client, monkeypatch, capsys):
    monkeypatch.setattr(feed, "HyperliquidPublicClient", lambda settings: client)
    assert main(["hyperliquid-feed", "--config", str(config)]) == 0
    output = capsys.readouterr().out
    assert "PASS: live books are fresh" in output
    assert "forward strategy month has not started" in output
    assert len(list(client.settings.observations.iterdir())) == 1
