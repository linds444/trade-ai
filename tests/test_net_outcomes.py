"""Invented source evidence; no outcome in this file is strategy performance."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3

import pytest

from local_ai_trader.cli import main
from local_ai_trader.data.storage import write_json
from local_ai_trader.hyperliquid import net_outcomes as module
from local_ai_trader.hyperliquid.forward import ForwardForecastBook
from local_ai_trader.hyperliquid.guardian import PaperGuardian
from local_ai_trader.hyperliquid.funding import FundingCollector, pair_recorded_funding
from local_ai_trader.hyperliquid.markets import BY_COIN
from local_ai_trader.models.train import file_digest
from local_ai_trader.paper.ledger import _hash, _iso
from test_forward_forecasts import forecast
from test_hyperliquid_decisions import scanner, NOW
from test_paper_guardian import source


@pytest.fixture
def observed(source):
    client, config, state = source
    models = config.parent.parent/"saved-models"
    manifest = json.loads((models/"models.json").read_text(encoding="utf-8"))
    manifest.update(test_opened=False, live_execution_enabled=False,
                    source_config_sha256=file_digest(config),
                    artifact_hashes={BY_COIN[c].filename+"/checkpoint.json": "invented-model" for c in client.settings.coins})
    write_json(models/"models.json", manifest)
    guardian = PaperGuardian(config, client=client)
    guardian.tick()
    contract = {"mode": "prospective_forecast_observations", "coins": list(client.settings.coins),
                "storage_schema": 2, "entry_delay_bars": 1, "holding_bars": 6, "label_known_delay_bars": 8, "class_threshold": .002,
                "source_mode": "observed", "class_names": ["up", "down", "flat"],
                "model_manifest_sha256": file_digest(models/"models.json"), "config_sha256": file_digest(config)}
    book = ForwardForecastBook(config.parent.parent/"data/hyperliquid/forward/test/observations.sqlite", contract)
    observer = module.NetOutcomeObserver(config, models, book.path, guardian.root, clock=client.clock)
    return observer, guardian, book, state


def enroll(observed, *, coin="SOL", probability=.8, gross=.02, signal=None):
    observer, guardian, book, state = observed
    signal = signal or module._time(observer.contract["first_signal_at"])
    state["at"] = signal+timedelta(seconds=2)
    record = forecast(coin)
    record.update(signal_at=signal.isoformat(), issued_at=(signal+timedelta(seconds=1)).isoformat(),
                  published_at=(signal+timedelta(seconds=2)).isoformat(), calibration_id="sha256:invented-model",
                  probabilities=[probability, (1-probability)/2, (1-probability)/2], ridge_return=gross)
    book.record([record], state["at"])
    observer.tick()
    return coin+"|"+_iso(signal), signal


def enter(observed, signal, delay=10):
    observer, guardian, _, state = observed
    state["at"] = signal+timedelta(minutes=5, seconds=delay)
    guardian.tick()
    return observer.tick()


def exit_at_horizon(observed, key, price=102, delay=0):
    observer, guardian, _, state = observed
    state["at"] = module._time(observer.records[key]["entry"]["at"])+timedelta(minutes=30, seconds=delay)
    state["prices"][observer.records[key]["forecast"]["forecast"]["coin"]] = price
    guardian.tick()
    return observer.tick()


def reopen(observed):
    observer, guardian, book, _ = observed
    return module.NetOutcomeObserver(observer.config, observer.models, book.path, guardian.root, clock=observer.clock)


def test_prospective_costs_reconcile_and_restart_never_creates_paper_fills(observed):
    observer, guardian, book, _ = observed
    before_account, before_forward = file_digest(guardian.account.path), file_digest(book.path)
    key, signal = enroll(observed)
    forward_after_enrollment = file_digest(book.path)
    assert observer.summary()["registered"] == 1
    assert enter(observed, signal)["entered_benchmarks"] == 1
    result = exit_at_horizon(observed, key)
    outcome = observer.records[key]["outcome"]
    assert outcome["success"] and outcome["cost_screen_candidate"]
    assert outcome["funding_cash_flow"] == 0
    assert outcome["mid_price_pnl"]-outcome["spread_cost"]-outcome["slippage_and_rounding_cost"]-outcome["fees"] == pytest.approx(outcome["net_pnl"])
    assert outcome["net_pnl"] < outcome["mid_price_pnl"]
    assert result["resolved"] == result["cost_screen_candidates_resolved"] == 1
    assert result["confidence"] is None and not result["reliability_validated"]
    assert not result["profitability_trial_started"] and not result["paper_fills_created"]
    version, head = observer.version, observer.head
    assert reopen(observed).head == head
    observer.tick()
    assert observer.version == version
    assert file_digest(guardian.account.path) == before_account
    assert file_digest(book.path) == forward_after_enrollment != before_forward


def test_old_forecasts_and_late_registration_cannot_be_backfilled(observed):
    observer, _, book, state = observed
    signal = module._time(observer.contract["first_signal_at"])-timedelta(minutes=5)
    record = forecast("SOL")
    record.update(signal_at=signal.isoformat(), issued_at=(signal+timedelta(seconds=1)).isoformat(),
                  published_at=(signal+timedelta(seconds=2)).isoformat(), calibration_id="sha256:invented-model")
    book.record([record], signal+timedelta(seconds=2))
    state["at"] = signal+timedelta(minutes=6)
    assert observer.tick()["registered"] == 0


@pytest.mark.parametrize("delay", [30, 35, 3600])
def test_missing_entry_window_is_unavailable_not_a_retroactive_fill(observed, delay):
    observer, _, _, _ = observed
    key, signal = enroll(observed)
    result = enter(observed, signal, delay=delay)
    assert result["unavailable"] == 1 and result["resolved"] == 0
    assert "entry" not in observer.records[key]
    assert observer.records[key]["unavailable"]["reason"] == "missed_entry_quote_window"


@pytest.mark.parametrize("problem", ["stale", "future", "unavailable", "changed_rules", "pre_deadline"])
def test_unusable_quotes_do_not_assign_entries(observed, problem):
    observer, guardian, _, state = observed
    key, signal = enroll(observed)
    if problem == "stale":
        state["stale"].add("SOL")
    elif problem == "unavailable":
        state["failed"].add("SOL")
    elif problem == "changed_rules":
        state["changed"].add("SOL")
    elif problem == "future":
        state["at"] = signal+timedelta(minutes=5, seconds=20)
        guardian.tick()
        state["at"] -= timedelta(seconds=10)
        observer.tick()
        assert "entry" not in observer.records[key]
        return
    elif problem == "pre_deadline":
        state["at"] = signal+timedelta(minutes=5)-timedelta(seconds=1)
        guardian.tick()
        state["at"] += timedelta(seconds=11)
        observer.tick()
        assert "entry" not in observer.records[key]
        return
    enter(observed, signal)
    assert "entry" not in observer.records[key]


def test_restart_after_entering_preserves_quantity_and_requires_timely_exit(observed):
    observer, guardian, book, state = observed
    key, signal = enroll(observed)
    enter(observed, signal)
    original = deepcopy(observer.records[key]["entry"])
    resumed = reopen(observed)
    assert resumed.records[key]["entry"] == original
    state["at"] = module._time(original["at"])+timedelta(minutes=30, seconds=31)
    guardian.tick()
    resumed.tick()
    assert resumed.records[key]["unavailable"]["reason"] == "missed_exit_quote_window"
    assert "outcome" not in resumed.records[key]


def test_constant_market_loses_costs_and_low_probability_cannot_certify_reliability(observed):
    observer, _, _, _ = observed
    key, signal = enroll(observed, probability=.4)
    enter(observed, signal)
    exit_at_horizon(observed, key, price=100)
    result = observer.records[key]["outcome"]
    assert result["net_pnl"] < 0 and not result["success"]
    assert not result["probability_candidate"] and not result["cost_screen_candidate"]
    assert observer.summary()["cost_screen_candidates_resolved"] == 0


def test_correct_up_direction_can_still_fail_net_profit_success(observed):
    observer, _, _, _ = observed
    key, signal = enroll(observed)
    enter(observed, signal)
    exit_at_horizon(observed, key, price=100.05)
    result = observer.records[key]["outcome"]
    assert result["mid_price_pnl"] > 0
    assert result["net_pnl"] < 0 and not result["success"]


def test_entry_after_boundary_event_does_not_pay_that_events_funding(observed):
    observer, _, _, _ = observed
    key, signal = enroll(observed)
    enter(observed, signal)
    exit_at_horizon(observed, key)
    state = observer.records[key]
    entry, exit = deepcopy(state["entry"]), deepcopy(state["exit"])
    start = datetime(2026, 1, 5, 1, 0, 10, tzinfo=timezone.utc)
    entry["at"], exit["at"] = _iso(start), _iso(start+timedelta(minutes=30))
    point = {"exchange_event_at": _iso(start.replace(second=0)+timedelta(milliseconds=80)),
             "oracle_price": 100, "rate": .01}
    result = module.calculate_outcome(state["forecast"]["forecast"], entry, exit, [point], .6, start+timedelta(minutes=31))
    assert result["funding_cash_flow"] == 0
    point["exchange_event_at"] = _iso(start+timedelta(seconds=1))
    assert module.calculate_outcome(state["forecast"]["forecast"], entry, exit, [point], .6, start+timedelta(minutes=31))["funding_cash_flow"] < 0
    point["rate"] = -.01
    assert module.calculate_outcome(state["forecast"]["forecast"], entry, exit, [point], .6, start+timedelta(minutes=31))["funding_cash_flow"] > 0


def test_prospective_registration_expired_during_commit_rolls_back(observed, monkeypatch):
    observer, _, book, state = observed
    signal = module._time(observer.contract["first_signal_at"])
    record = forecast("SOL")
    record.update(signal_at=signal.isoformat(), issued_at=(signal+timedelta(seconds=1)).isoformat(),
                  published_at=(signal+timedelta(seconds=2)).isoformat(), calibration_id="sha256:invented-model")
    state["at"] = signal+timedelta(seconds=2)
    original = observer._validate
    def delayed(*args):
        original(*args)
        state["at"] = signal+timedelta(minutes=5)
    monkeypatch.setattr(observer, "_validate", delayed)
    with pytest.raises(ValueError, match="expired before commit"):
        observer._append("SOL|"+_iso(signal), "forecast", {"forecast": record, "enrolled_at": _iso(signal+timedelta(seconds=2))})
    assert not observer.records and observer.version == 0
    assert reopen(observed).version == 0


def test_frozen_quote_tampering_is_rejected_during_restart(observed):
    observer, guardian, _, _ = observed
    key, signal = enroll(observed)
    enter(observed, signal)
    proof = observer.records[key]["entry"]["proof"]
    path = guardian.root/proof["capture"]
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="archive changed"):
        reopen(observed)


def test_reading_live_forward_subset_keeps_global_deletion_checks(observed):
    observer, _, book, _ = observed
    key, signal = enroll(observed)
    contract, selected = module.read_forecasts(book.path, signal_since=signal+timedelta(minutes=5))
    assert selected == []
    with sqlite3.connect(str(book.path)) as db:
        db.execute("DELETE FROM forecasts")
    with pytest.raises(ValueError, match="counters changed"):
        module.read_forecasts(book.path, signal_since=signal+timedelta(minutes=5))


def test_native_contract_quantity_and_cost_failure_are_preserved(observed):
    observer, _, _, _ = observed
    key, signal = enroll(observed, coin="kPEPE", gross=0)
    enter(observed, signal)
    exit_at_horizon(observed, key)
    assert observer.records[key]["entry"]["quantity"] > 1
    assert observer.records[key]["outcome"]["probability_candidate"]
    assert not observer.records[key]["outcome"]["cost_screen_candidate"]


def test_insufficient_full_size_liquidity_remains_unavailable(observed):
    observer, _, _, state = observed
    key, signal = enroll(observed)
    state["sizes"]["SOL"] = 1  # best bid; ask remains twenty contracts
    enter(observed, signal)
    result = exit_at_horizon(observed, key)
    assert result["unavailable"] == 1 and result["resolved"] == 0
    assert observer.records[key]["unavailable"]["reason"] == "insufficient_full_size_exit_liquidity"


def test_top_of_hour_funding_waits_for_real_operands_and_posts_once(observed, monkeypatch):
    observer, guardian, book, state = observed
    signal = module._time(observer.contract["first_signal_at"])+timedelta(minutes=20)
    key, _ = enroll(observed, coin="BTC", signal=signal)  # 00:30 signal, entry 00:35, exit 01:05
    enter(observed, signal)
    result = exit_at_horizon(observed, key)
    assert result["pending_funding"] == 1 and result["resolved"] == 0
    client = guardian.client
    original = client.info
    def info(request):
        if request["type"] == "fundingHistory":
            return [{"coin": request["coin"], "time": stamp+80, "fundingRate": ".0001", "premium": "0"}
                    for stamp in range(request["startTime"], request["endTime"]+1, 3600000) if stamp+80 <= request["endTime"]]
        return original(request)
    monkeypatch.setattr(client, "info", info)
    recorder = FundingCollector(client, observer.config)
    state["at"] = signal.replace(hour=1, minute=0, second=10)
    capture, _, _ = recorder.capture()  # invented proxy observed near the actual event
    state["at"] = signal.replace(hour=1, minute=6)
    result = observer.tick()
    outcome = observer.records[key]["outcome"]
    assert result["resolved"] == 1 and result["pending_funding"] == 0
    expected = -observer.records[key]["entry"]["quantity"]*101*.0001
    assert outcome["funding_cash_flow"] == pytest.approx(expected)
    assert outcome["funding_oracle_estimated"]
    assert outcome["funding"][0]["source_captures"] == [str(capture)]
    version = observer.version
    observer.tick()
    assert observer.version == version and reopen(observed).head == observer.head


@pytest.mark.parametrize("entry_second,exit_second,expected", [(0, 0, 1), (29, 0, 1), (30, 0, 0)])
def test_first_thirty_seconds_require_settlement_evidence(entry_second, exit_second, expected):
    start = datetime(2026, 1, 5, 1, 0, entry_second, tzinfo=timezone.utc)
    assert len(module.funding_keys("BTC", start, start+timedelta(minutes=30))) == expected


@pytest.mark.parametrize("mutation", ["delete", "change", "configuration", "guardian", "source_book"])
def test_tampering_or_rebinding_refuses_without_reset(observed, mutation):
    observer, guardian, book, _ = observed
    key, signal = enroll(observed)
    if mutation in ("delete", "change"):
        with sqlite3.connect(str(observer.path)) as db:
            db.execute("DELETE FROM events" if mutation == "delete" else "UPDATE events SET payload='{}'")
    elif mutation == "configuration":
        observer.config.write_text(observer.config.read_text()+"\n# changed\n")
    elif mutation == "guardian":
        (guardian.root/"run.json").write_text("{}")
    else:
        with sqlite3.connect(str(book.path)) as db:
            db.execute("UPDATE contract SET payload='{}'")
    with pytest.raises((ValueError, KeyError)):
        reopen(observed)
    assert observer.path.exists()


def test_observer_cli_is_separate_and_prints_pretrial_status(observed, capsys):
    observer, guardian, book, _ = observed
    # The CLI's wall clock is intentionally injected; no historical registration.
    from unittest.mock import patch
    with patch.object(module, "NetOutcomeObserver", return_value=observer):
        assert main(["hyperliquid-net-outcomes", "--config", str(observer.config), "--models", str(observer.models),
                     "--book", str(book.path), "--guardian", str(guardian.root)]) == 0
    output = capsys.readouterr().out
    assert "hypothetical" in output and "strategy month not started" in output
