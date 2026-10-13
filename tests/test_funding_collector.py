from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3

import pytest

from local_ai_trader.cli import main
from local_ai_trader.hyperliquid import funding as module
from local_ai_trader.hyperliquid.feed import HyperliquidPublicClient, load_hyperliquid_settings
from local_ai_trader.paper.ledger import _hash
from local_ai_trader.paper.perpetual import PerpetualPaperAccount
from test_hyperliquid_feed import config, contexts, book
from test_perpetual_paper import entry, quote

HOUR = datetime(2026, 1, 5, 1, 0, tzinfo=timezone.utc)
AT = HOUR + timedelta(seconds=10)


@pytest.fixture
def collector(config, monkeypatch):
    current = [AT]
    client = HyperliquidPublicClient(load_hyperliquid_settings(config), clock=lambda: current[0])
    calls = []
    def info(payload):
        calls.append(deepcopy(payload))
        if payload["type"] == "fundingHistory":
            return [{"coin": payload["coin"], "time": stamp + 80, "fundingRate": "0.0001", "premium": "-0.0001"}
                    for stamp in range(payload["startTime"], payload["endTime"] + 1, 3600000) if stamp + 80 <= payload["endTime"]]
        if payload["type"] == "metaAndAssetCtxs":
            return contexts()
        if payload["type"] == "l2Book":
            return book(payload["coin"], current[0])
        pytest.fail("Unexpected request")
    monkeypatch.setattr(client, "info", info)
    return module.FundingCollector(client, config), current, calls


def row(**changes):
    return {"coin": "BTC", "time": int(HOUR.timestamp() * 1000) + 80,
            "fundingRate": "-0.0001", "premium": "0.0001", **changes}


def test_actual_published_milliseconds_and_signed_rate_are_preserved():
    points, duplicates = module.parse_funding_history([row(), row()], "BTC", int(HOUR.timestamp() * 1000), int(AT.timestamp() * 1000))
    assert duplicates == 1
    assert points[0]["hour"] == module._iso(HOUR)
    assert points[0]["exchange_event_at"] == module._iso(HOUR + timedelta(milliseconds=80))
    assert points[0]["rate"] == -0.0001


@pytest.mark.parametrize("changes", [
    {"coin": "ETH"}, {"time": True}, {"time": int(HOUR.timestamp() * 1000) - 1},
    {"time": int((HOUR + timedelta(seconds=30)).timestamp() * 1000)},
    {"fundingRate": "nan"}, {"fundingRate": "0.0401"}, {"premium": "inf"},
])
def test_invalid_history_is_refused(changes):
    with pytest.raises(ValueError):
        module.parse_funding_history([row(**changes)], "BTC", int(HOUR.timestamp() * 1000), int(AT.timestamp() * 1000))


def test_conflicting_same_hour_rows_are_not_averaged():
    with pytest.raises(ValueError, match="Conflicting"):
        module.parse_funding_history([row(), row(fundingRate="0.0001")], "BTC", int(HOUR.timestamp() * 1000), int(AT.timestamp() * 1000))


def test_capture_verifies_config_and_raw_evidence_and_pairs_oracle_not_mark(collector):
    recorder, _, calls = collector
    output, _, near = recorder.capture()
    observation, _ = module.load_funding_capture(output)
    assert near
    assert len(observation["rates"]) == 48
    assert all(p["modeled_proxy"] and p["exchange_time"] is None for p in observation["oracles"])
    ready, missing = module.pair_recorded_funding(recorder.root)
    btc = ready["BTC|" + module._iso(HOUR)]
    assert btc["oracle_price"] == 100.7  # context mark is 100.5
    assert btc["rate_observed_at"] == module._iso(AT)
    assert btc["source_mode"] == "observed"
    assert btc["rate_source_id"].startswith("sha256:")
    assert len(ready) == 2 and len(missing) == 46
    assert all(p["type"] in ("fundingHistory", "metaAndAssetCtxs", "l2Book") for p in calls)
    report = json.loads((output / "capture.json").read_text(encoding="utf-8"))
    assert report["forward_strategy_started"] is report["orders_created"] is report["paper_fills_created"] is False


def test_mid_hour_current_oracle_never_fills_missing_historical_oracles(collector):
    recorder, current, _ = collector
    current[0] = HOUR + timedelta(minutes=15)
    recorder.capture()
    ready, missing = module.pair_recorded_funding(recorder.root)
    assert not ready and len(missing) == 48


def test_missing_settled_rate_is_reported_without_using_current_context_rate(collector, monkeypatch):
    recorder, _, _ = collector
    original = recorder.client.info
    def info(payload):
        result = original(payload)
        if payload["type"] == "fundingHistory" and payload["coin"] == "BTC":
            return result[:-1]
        return result
    monkeypatch.setattr(recorder.client, "info", info)
    output, _, _ = recorder.capture()
    observation, _ = module.load_funding_capture(output)
    assert observation["assets"]["BTC"]["missing_rate_hours"] == [module._iso(HOUR)]
    ready, _ = module.pair_recorded_funding(recorder.root)
    assert "BTC|" + module._iso(HOUR) not in ready
    assert ready["ETH|" + module._iso(HOUR)]["rate"] == 0.0001  # current context rate is -0.0001


def test_oracle_exactly_thirty_seconds_from_actual_settlement_remains_unavailable(collector):
    recorder, current, _ = collector
    current[0] = HOUR + timedelta(seconds=30, milliseconds=80)
    recorder.capture()
    ready, _ = module.pair_recorded_funding(recorder.root)
    assert not ready


def test_pre_hour_oracle_is_retained_and_nearest_evidence_is_selected(collector):
    recorder, current, _ = collector
    current[0] = HOUR - timedelta(seconds=20)
    recorder.capture()
    current[0] = HOUR + timedelta(seconds=10)
    recorder.capture()
    ready, _ = module.pair_recorded_funding(recorder.root)
    assert ready["BTC|" + module._iso(HOUR)]["oracle_at"] == module._iso(current[0])


@pytest.mark.parametrize("filename", ["raw.json", "input_config.toml", "capture.json"])
def test_changed_artifacts_or_cooked_oracle_refuse(collector, filename):
    recorder, _, _ = collector
    output, _, _ = recorder.capture()
    path = output / filename
    if filename == "capture.json":
        report = json.loads(path.read_text(encoding="utf-8"))
        report["observation"]["oracles"][0]["oracle_price"] = 999
        path.write_text(json.dumps(report), encoding="utf-8")
    else:
        path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError):
        module.load_funding_capture(output)


def test_changed_settled_rate_never_publishes_conflicting_capture(collector, monkeypatch):
    recorder, current, _ = collector
    recorder.capture()
    current[0] += timedelta(seconds=61)
    original = recorder.client.info
    def changed(payload):
        result = original(payload)
        if payload["type"] == "fundingHistory":
            result[-1]["fundingRate"] = "0.0002"
        return result
    monkeypatch.setattr(recorder.client, "info", changed)
    with pytest.raises(ValueError, match="conflict"):
        recorder.capture()
    assert len(list(recorder.root.glob("capture_*"))) == 1
    assert not list(recorder.root.glob(".funding-staging-*"))


def test_duplicate_captures_do_not_change_settled_inputs_or_rate_availability(collector):
    recorder, current, _ = collector
    recorder.capture()
    before, _ = module.pair_recorded_funding(recorder.root)
    current[0] += timedelta(seconds=61)
    recorder.capture()
    after, _ = module.pair_recorded_funding(recorder.root)
    assert after == before


def test_history_response_order_and_numeric_format_do_not_trigger_changed_rate_signature(collector, monkeypatch):
    recorder, current, _ = collector
    _, before, _ = recorder.capture()
    current[0] += timedelta(seconds=61)
    original = recorder.client.info
    def info(payload):
        result = original(payload)
        if payload["type"] == "fundingHistory":
            for point in result:
                point["fundingRate"] = "1e-4"
            result.reverse()
        return result
    monkeypatch.setattr(recorder.client, "info", info)
    _, after, _ = recorder.capture()
    assert before == after


def test_books_that_expire_before_publication_leave_no_capture(collector, monkeypatch):
    recorder, current, _ = collector
    original = module.write_json
    def write(path, data):
        original(path, data)
        if path.name == "capture.json":
            current[0] += timedelta(seconds=30)
    monkeypatch.setattr(module, "write_json", write)
    with pytest.raises(ValueError, match="stale"):
        recorder.capture()
    assert not list(recorder.root.glob("capture_*"))
    assert not list(recorder.root.glob(".funding-staging-*"))


def test_configuration_change_during_capture_never_publishes(collector, monkeypatch):
    recorder, _, _ = collector
    original = recorder.client.info
    changed = []
    def info(payload):
        result = original(payload)
        if payload["type"] == "l2Book" and not changed:
            changed.append(True)
            with recorder.config.open("a", encoding="utf-8") as f:
                f.write("\n# changed\n")
        return result
    monkeypatch.setattr(recorder.client, "info", info)
    with pytest.raises(ValueError, match="configuration changed"):
        recorder.capture()
    assert not list(recorder.root.glob("capture_*"))


def test_history_cache_refreshes_at_hour_change_and_after_sixty_seconds(collector):
    recorder, current, calls = collector
    recorder.capture()
    current[0] += timedelta(seconds=10)
    recorder.capture()
    assert len([p for p in calls if p["type"] == "fundingHistory"]) == 2
    current[0] += timedelta(seconds=51)
    recorder.capture()
    assert len([p for p in calls if p["type"] == "fundingHistory"]) == 4
    current[0] = HOUR + timedelta(hours=1, seconds=1)
    recorder.capture()
    assert len([p for p in calls if p["type"] == "fundingHistory"]) == 6


def test_recorded_funding_adapter_settles_observed_account_and_replays_without_fills(collector, tmp_path):
    recorder, current, _ = collector
    recorder.capture()
    opened = HOUR - timedelta(minutes=5) + timedelta(seconds=10)
    account = PerpetualPaperAccount.create(tmp_path / "observed.sqlite", now=opened, source_mode="observed")
    packet = entry(opened)
    packet["quote"]["source_mode"] = "observed"
    account.execute("entry", "entry", packet, now=opened)
    mark = quote(current[0], source_mode="observed")
    account.execute("mark", "mark", {"quotes": [mark]}, now=current[0])
    results, report = module.apply_recorded_funding(account.path, recorder.root, now=current[0])
    assert len(results) == 1 and results[0]["cash_flow"] == pytest.approx(-9 * 100.7 * 0.0001)
    assert report["missing_funding"] == [] and report["fill_count"] == 1
    assert module.apply_recorded_funding(account.path, recorder.root, now=current[0])[0] == []


def test_recorded_funding_cannot_enter_synthetic_account(collector, tmp_path):
    recorder, _, _ = collector
    recorder.capture()
    account = PerpetualPaperAccount.create(tmp_path / "synthetic.sqlite", now=AT)
    with pytest.raises(ValueError, match="observed account"):
        module.apply_recorded_funding(account.path, recorder.root, now=AT)


def test_publication_lock_serializes_writers_and_releases_after_exception(collector):
    recorder, _, _ = collector
    recorder.root.mkdir(parents=True)
    with module._publication_lock(recorder.root):
        with pytest.raises(ValueError, match="Another"):
            with module._publication_lock(recorder.root):
                pytest.fail("Second publisher acquired the lock")
    with module._publication_lock(recorder.root):
        pass


def test_watch_avoids_duplicate_mid_hour_bundles_and_stops_cleanly(collector, monkeypatch, capsys):
    recorder, current, _ = collector
    current[0] = HOUR + timedelta(minutes=15)
    monkeypatch.setattr(module, "FundingCollector", lambda *args: recorder)
    ticks = []
    def sleep(seconds):
        ticks.append(seconds)
        current[0] += timedelta(seconds=seconds)
        if len(ticks) == 8:
            raise KeyboardInterrupt
    monkeypatch.setattr(module.time, "sleep", sleep)
    module.run_funding_collector(recorder.config, watch=True)
    assert len(list(recorder.root.glob("capture_*"))) == 1
    assert "collector stopped" in capsys.readouterr().out


def test_watch_retains_near_hour_observations_without_repeating_history_requests(collector, monkeypatch):
    recorder, current, calls = collector
    monkeypatch.setattr(module, "FundingCollector", lambda *args: recorder)
    ticks = []
    def sleep(seconds):
        ticks.append(seconds)
        current[0] += timedelta(seconds=seconds)
        if len(ticks) == 3:
            raise KeyboardInterrupt
    monkeypatch.setattr(module.time, "sleep", sleep)
    module.run_funding_collector(recorder.config, watch=True)
    assert len(list(recorder.root.glob("capture_*"))) == 2
    assert len([p for p in calls if p["type"] == "fundingHistory"]) == 2


def test_cli_and_long_native_capture_paths(collector, monkeypatch, capsys):
    recorder, _, _ = collector
    root = recorder.root
    while len(str(root)) < 310:
        root /= "funding-deep-native-path"
    recorder.root = module.native_path(root).resolve()
    monkeypatch.setattr(module, "FundingCollector", lambda *args: recorder)
    assert main(["hyperliquid-funding", "--config", str(recorder.config)]) == 0
    assert len(list(recorder.root.glob("capture_*"))) == 1
    assert "PASS: settled rates" in capsys.readouterr().out
