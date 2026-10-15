"""Continuous paper maintenance with fully invented API and model evidence."""
from copy import deepcopy
from dataclasses import asdict
from datetime import timedelta
from decimal import Decimal
import json
from pathlib import Path

import pytest

from local_ai_trader.cli import main
from local_ai_trader.data.storage import write_json, native_path
from local_ai_trader.hyperliquid import guardian as module
from local_ai_trader.hyperliquid.guardian import PaperGuardian, capture_quotes, derive_quotes, load_capture
from local_ai_trader.hyperliquid.funding import _publication_lock, apply_recorded_funding
from local_ai_trader.models.train import file_digest
from local_ai_trader.paper.ledger import _hash
from local_ai_trader.paper.perpetual import PerpetualPaperAccount
from local_ai_trader.risk.entry_policy import SignalEvidence
from test_hyperliquid_decisions import scanner, NOW


@pytest.fixture
def source(scanner, monkeypatch):
    client, config, _ = scanner
    original = client.info
    state = {"at": NOW, "prices": {}, "sizes": {}, "failed": set(), "stale": set(), "changed": set(), "requests": []}
    client.clock = lambda: state["at"]
    def info(request):
        state["requests"].append(deepcopy(request))
        assert request["type"] in ("metaAndAssetCtxs", "l2Book"), "Guardian requested inference/history data"
        key = request.get("coin", request.get("dex", "native"))
        if key in state["failed"]:
            raise RuntimeError("invented API outage")
        result = deepcopy(original(request))
        if request["type"] == "metaAndAssetCtxs":
            for asset, context in zip(result[0]["universe"], result[1]):
                maximum = 25 if asset["name"] == "ETH" else 20
                asset.update(szDecimals=2, maxLeverage=maximum, marginTableId=maximum)
                if asset["name"] in state["changed"]:
                    asset["onlyIsolated"] = True
                context["markPx"] = str(state["prices"].get(asset["name"], 100.))
        else:
            coin = request["coin"]
            at = state["at"]-timedelta(seconds=30) if coin in state["stale"] else state["at"]
            result["time"] = int(at.timestamp()*1000)
            price = Decimal(str(state["prices"].get(coin, 100.)))
            delta = Decimal(".01") if price >= 100 else Decimal(".001")
            result["levels"][0][0].update(px=str(price-delta), sz=str(state["sizes"].get(coin, 20)))
            result["levels"][1][0].update(px=str(price+delta), sz="20")
        return result
    monkeypatch.setattr(client, "info", info)
    return client, config, state


def make(source):
    client, config, state = source
    return PaperGuardian(config, client=client), state


def seed(book, at=NOW, coin="SOL", quantity=9, identifier="test-entry"):
    """Only controlled test code can add this invented, qualified evidence."""
    book.account.execute("test-enable", "pause", {"enabled": True}, now=at)
    quote = {"coin": coin, "at": at, "bid": 99.99, "ask": 100.01, "mark": 100.,
             "size_decimals": 2, "fee_bps": 4.5, "source_mode": "observed"}
    result = book.account.execute(identifier, "entry", {
        "quote": quote, "quantity": quantity, "ask_size": 20,
        "signal_at": at.replace(minute=at.minute//5*5, second=0, microsecond=0)-timedelta(minutes=5),
        "evidence": asdict(SignalEvidence("BUY", .8, "invented", .8, "invented")),
        "expected_gross_return": .03, "expected_funding_cost_fraction": .001}, now=at)
    assert result["status"] == "filled"
    return result


def charge(book, at, rate=.02):
    hour = at.replace(minute=0, second=0, microsecond=0)
    book.account.execute("test-funding", "funding", {"coin": "SOL", "hour": hour,
        "rate": rate, "oracle_price": 100., "oracle_at": hour,
        "rate_source_id": "invented", "oracle_source_id": "invented", "source_mode": "observed"}, now=at)


def test_empty_guardian_has_native_rules_and_never_invokes_entry_or_model(source, monkeypatch):
    book, state = make(source)
    original = book.account.execute
    def execute(identifier, kind, inputs, **kwargs):
        assert kind != "entry"
        return original(identifier, kind, inputs, **kwargs)
    monkeypatch.setattr(book.account, "execute", execute)
    result = book.tick()
    assert result["healthy"] and result["position_count"] == 0
    assert result["account"]["fill_count"] == result["account"]["version"] == 0
    assert not result["account"]["entries_enabled"]
    assert not result["reliability_validated"] and not result["profitability_trial_started"]
    assert book.account.contract()["schema_version"] == 2
    assert set(r["type"] for r in state["requests"]) == {"metaAndAssetCtxs", "l2Book"}
    assert (book.root/"latest.json").is_file()
    assert load_capture(Path(result["source_capture"]), book.settings, result["source_sha256"])["markets"]["SOL"]["status"] == "READY"


def test_flat_ticks_do_not_bloat_the_account_journal(source):
    book, state = make(source)
    for _ in range(3):
        state["at"] += timedelta(seconds=10)
        book.tick()
    assert book.account.audit()["version"] == 0
    assert len(list((book.root/"captures").rglob("*.json.gz"))) == 3


def test_restart_reopens_identical_account_without_reset(source):
    book, state = make(source)
    seed(book)
    state["at"] += timedelta(seconds=10)
    book.tick()
    before = file_digest(book.account.path)
    reopened = PaperGuardian(source[1], client=source[0])
    assert reopened.root == book.root and file_digest(reopened.account.path) == before
    assert reopened.account.audit()["fill_count"] == 1


def test_missing_existing_account_refuses_reset(source):
    book, _ = make(source)
    book.account.path.unlink()
    with pytest.raises(ValueError, match="refusing to reset"):
        PaperGuardian(source[1], client=source[0])


def test_partial_api_outage_does_not_prevent_other_market_horizon_exit(source):
    book, state = make(source)
    seed(book)
    state["failed"].add("xyz:HOOD")
    state["at"] += timedelta(minutes=30)
    result = book.tick()
    assert result["markets"]["xyz:HOOD"]["status"] == "UNAVAILABLE"
    assert result["markets"]["SOL"]["status"] == "READY"
    assert result["account"]["fill_count"] == 2 and result["position_count"] == 0
    assert not result["account"]["performance_provisional"]  # no funding boundary during this holding
    assert not result["healthy"]


def test_native_context_failure_keeps_xyz_marks_available(source):
    book, state = make(source)
    state["failed"].add("native")
    result = book.tick()
    assert result["markets"]["BTC"]["status"] == "UNAVAILABLE"
    assert result["markets"]["xyz:HOOD"]["status"] == "READY"


def test_stale_book_never_fills_and_recovers_on_next_fresh_tick(source):
    book, state = make(source)
    seed(book)
    state["at"] += timedelta(minutes=30)
    state["stale"].add("SOL")
    result = book.tick()
    assert result["position_count"] == 1 and result["account"]["fill_count"] == 1
    assert "stale" in result["markets"]["SOL"]["reason"]
    state["stale"].clear()
    state["at"] += timedelta(seconds=10)
    result = book.tick()
    assert result["position_count"] == 0 and result["account"]["fill_count"] == 2


def test_rules_change_is_unavailable_without_rebinding_and_other_markets_continue(source):
    book, state = make(source)
    state["changed"].add("SOL")
    result = book.tick()
    assert "Venue rules changed" in result["markets"]["SOL"]["reason"]
    assert result["markets"]["ETH"]["status"] == "READY"
    assert book.account.contract()["isolated_venue_rules"]["SOL"]["only_isolated"] is False


def test_marks_between_scanner_boundaries_do_not_exit_early(source):
    book, state = make(source)
    seed(book)
    state["at"] += timedelta(seconds=10)
    state["prices"]["SOL"] = 101
    result = book.tick()
    assert result["position_count"] == 1 and result["account"]["fill_count"] == 1
    assert book.account.audit()["positions"]["SOL"]["mark_price"] == 101
    assert not result["actions"]


def stress(source, *, backstop=False):
    client, config, state = source
    state["at"] = NOW.replace(minute=55, second=10)
    book = PaperGuardian(config, client=client)
    seed(book, at=state["at"])
    state["at"] = state["at"].replace(minute=0, second=10)+timedelta(hours=1)
    charge(book, state["at"])
    state["prices"]["SOL"] = 1.9 if backstop else 2.05
    if backstop:
        state["sizes"]["SOL"] = .01
    return book, state


def test_fresh_mark_breach_closes_before_horizon(source):
    book, state = stress(source)
    result = book.tick()
    assert result["position_count"] == 0 and result["account"]["fill_count"] == 2
    assert result["actions"][0]["reason"] == "isolated_liquidation"
    assert not result["account"]["entries_enabled"]
    assert book.account.audit()["reconciled"]


def test_backstop_transfer_uses_observed_liquidity_and_survives_restart(source):
    book, state = stress(source, backstop=True)
    free = book.account.status()["available_collateral"]
    result = book.tick()
    assert result["account"]["backstop_transfers"] == 1 and result["account"]["fill_count"] == 1
    assert result["account"]["available_collateral"] == pytest.approx(free)
    assert PaperGuardian(source[1], client=source[0]).account.audit()["backstop_transfers"] == 1


def test_horizon_blocked_by_depth_retries_with_new_capture_instead_of_replaying_block(source):
    book, state = make(source)
    seed(book)
    state["at"] += timedelta(minutes=30)
    state["sizes"]["SOL"] = 1
    assert book.tick()["actions"][0]["reason"] == "insufficient_full_size_bid_liquidity"
    assert book.account.status()["positions"]
    state["at"] += timedelta(seconds=10)
    state["sizes"]["SOL"] = 20
    assert book.tick()["actions"][0]["status"] == "filled"
    assert book.account.audit()["fill_count"] == 2


def test_crash_after_sell_before_receipt_recovery_does_not_double_sell(source, monkeypatch):
    book, state = make(source)
    seed(book)
    state["at"] += timedelta(minutes=30)
    original = module.write_json
    def fail_receipt(path, value):
        if str(path).endswith(".receipt.json"):
            raise OSError("invented crash after ledger commit")
        return original(path, value)
    monkeypatch.setattr(module, "write_json", fail_receipt)
    with pytest.raises(OSError, match="invented crash"):
        book.tick()
    assert book.account.audit()["fill_count"] == 2
    assert (book.root/"pending.json").is_file()
    monkeypatch.setattr(module, "write_json", original)
    state["at"] += timedelta(seconds=10)
    reopened = PaperGuardian(source[1], client=source[0])
    result = reopened.tick()
    assert result["account"]["fill_count"] == 2 and not (book.root/"pending.json").exists()


def test_expired_pending_capture_is_not_used_to_fill_after_restart(source, monkeypatch):
    book, state = make(source)
    seed(book)
    state["at"] += timedelta(minutes=30)
    original = book._maintain
    monkeypatch.setattr(book, "_maintain", lambda saved: (_ for _ in ()).throw(OSError("invented crash before mutation")))
    with pytest.raises(OSError):
        book.tick()
    assert book.account.status()["fill_count"] == 1
    state["at"] += timedelta(minutes=2)
    monkeypatch.setattr(book, "_maintain", original)
    result = PaperGuardian(source[1], client=source[0]).tick()
    assert result["account"]["fill_count"] == 2
    trade = book.account.audit()["trades"][0]
    assert trade["exit_at"] == state["at"].isoformat(timespec="microseconds")


def test_pending_source_tampering_refuses_before_mutation(source, monkeypatch):
    book, state = make(source)
    monkeypatch.setattr(book, "_maintain", lambda saved: (_ for _ in ()).throw(OSError("invented crash")))
    with pytest.raises(OSError):
        book.tick()
    pending = json.loads((book.root/"pending.json").read_text())
    (book.root/pending["capture"]).write_bytes(b"changed")
    version = book.account.status()["version"]
    with pytest.raises(ValueError, match="source archive changed"):
        PaperGuardian(source[1], client=source[0]).tick()
    assert book.account.audit()["version"] == version


def test_config_change_does_not_create_replacement_book_and_running_guardian_keeps_exit(source):
    book, state = make(source)
    seed(book)
    source[1].write_bytes(source[1].read_bytes()+b"\n# edited\n")
    state["at"] += timedelta(minutes=30)
    result = book.tick()
    assert result["config_changed"] and not result["healthy"]
    assert result["account"]["fill_count"] == 2
    with pytest.raises(ValueError, match="config changed"):
        PaperGuardian(source[1], client=source[0])
    assert len(list(book.root.parent.glob("guardian_*"))) == 1


def test_publication_lock_rejects_second_watcher(source):
    book, state = make(source)
    with _publication_lock(book.root, label="paper guardian"):
        with pytest.raises(ValueError, match="Another paper guardian"):
            with _publication_lock(book.root, label="paper guardian"):
                pass


def test_cached_funding_account_requires_matching_path(source, tmp_path):
    book, state = make(source)
    with pytest.raises(ValueError, match="account/path disagree"):
        apply_recorded_funding(tmp_path/"wrong.sqlite", tmp_path, account=book.account)


def test_cli_wires_one_shot_and_watch_modes(monkeypatch):
    calls=[]
    monkeypatch.setattr(module, "run_paper_guardian", lambda config, watch=False: calls.append((config, watch)))
    assert main(["hyperliquid-paper", "--config", "config/hyperliquid.toml", "--watch"]) == 0
    assert calls == [(Path("config/hyperliquid.toml"), True)]


def test_funding_posted_after_fresh_mark_can_trigger_immediate_liquidation(source, monkeypatch):
    client, config, state = source
    state["at"] = NOW.replace(minute=55, second=10)
    book = PaperGuardian(config, client=client)
    seed(book, at=state["at"])
    state["at"] = state["at"].replace(minute=0, second=10)+timedelta(hours=1)
    state["prices"]["SOL"] = 2.05
    calls=[]
    def posted(path, archive, *, clock, account):
        assert account is book.account
        calls.append(path)
        charge(book, clock())
        return [{"status": "posted"}], account.status()
    monkeypatch.setattr(module, "apply_recorded_funding", posted)
    result = book.tick()
    assert calls == [book.account.path]
    assert result["account"]["fill_count"] == 2 and result["position_count"] == 0
    assert result["actions"][0]["reason"] == "isolated_liquidation"
    assert not result["account"]["missing_funding"]


def test_slow_funding_io_cannot_allow_stale_liquidation(source, monkeypatch):
    client, config, state = source
    state["at"] = NOW.replace(minute=55, second=10)
    book = PaperGuardian(config, client=client)
    seed(book, at=state["at"])
    state["at"] = state["at"].replace(minute=0, second=10)+timedelta(hours=1)
    state["prices"]["SOL"] = 2.05
    def slow(path, archive, *, clock, account):
        charge(book, clock())
        state["at"] += timedelta(seconds=31)
        return [{"status": "posted"}], account.status()
    monkeypatch.setattr(module, "apply_recorded_funding", slow)
    result = book.tick()
    assert result["account"]["fill_count"] == 1 and result["position_count"] == 1
    assert "stale" in result["errors"]["liquidation:SOL"]
    assert not result["healthy"]
    assert book.tick()["account"]["fill_count"] == 2


def test_whole_run_deletion_is_detected_by_identity_witness(source):
    import shutil
    book, state = make(source)
    shutil.rmtree(book.root)
    with pytest.raises(ValueError, match="Previously created guardian book is missing"):
        PaperGuardian(source[1], client=source[0])


def test_missing_funding_is_provisional_but_never_traps_a_fresh_horizon_exit(source):
    client, config, state = source
    state["at"] = NOW.replace(minute=55, second=10)
    book = PaperGuardian(config, client=client)
    seed(book, at=state["at"])
    state["at"] += timedelta(minutes=30)
    result = book.tick()
    assert result["account"]["fill_count"] == 2 and result["position_count"] == 0
    assert len(result["account"]["missing_funding"]) == 1 and result["account"]["performance_provisional"]


def test_watch_process_ownership_is_checked_before_startup_audit(source, monkeypatch):
    book, state = make(source)
    config = native_path(source[1]).resolve()
    identity = _hash({"purpose": "hyperliquid_isolated_guardian_v1", "config_path": str(config)})[:16]
    lock_root = book.root.parent/(".guardian-service-"+identity)
    lock_root.mkdir(exist_ok=True)
    monkeypatch.setattr(module, "PaperGuardian", lambda *a, **k: pytest.fail("Second watcher reached startup"))
    with _publication_lock(lock_root, label="paper guardian"):
        with pytest.raises(ValueError, match="Another paper guardian"):
            module.run_paper_guardian(config, watch=True)


def test_runtime_contract_change_is_detected_before_marks_or_exits(source):
    import sqlite3
    book, state = make(source)
    with sqlite3.connect(str(book.account.path)) as db:
        contract = json.loads(db.execute("SELECT contract FROM account").fetchone()[0])
        contract["funding_time_basis"] = "utc_hour"
        db.execute("UPDATE account SET contract=?,contract_hash=?", (json.dumps(contract), _hash(contract)))
    with pytest.raises(ValueError, match="account/run binding changed"):
        book.tick()
    assert book.account.status()["version"] == 0


def test_new_public_client_uses_one_bounded_attempt_without_changing_source_config(source, monkeypatch):
    seen=[]
    def factory(settings):
        seen.append(settings)
        return source[0]
    monkeypatch.setattr(module, "HyperliquidPublicClient", factory)
    book = PaperGuardian(source[1])
    assert seen[0].max_attempts == 1 and seen[0].timeout_seconds == 5 and seen[0].poll_seconds <= 10
    assert book.settings.max_attempts == 3 and book.settings.timeout_seconds == 10


def test_auditable_bad_quotes_reconstruct_at_the_original_capture_time(source):
    client, config, state = source
    state["stale"].add("BTC")
    payload = capture_quotes(client)
    assert payload["markets"]["BTC"]["status"] == "UNAVAILABLE"
    assert derive_quotes(payload, client.settings, state["at"]) == payload["markets"]


def test_watch_loop_polls_and_releases_ownership_on_ctrl_c(source, monkeypatch, capsys):
    book, state = make(source)
    monkeypatch.setattr(module, "PaperGuardian", lambda config: book)
    sleeps=[]
    def stop(seconds):
        sleeps.append(seconds)
        raise KeyboardInterrupt
    monkeypatch.setattr(module.time, "sleep", stop)
    module.run_paper_guardian(source[1], watch=True)
    assert len(sleeps) == 1 and 0 < sleeps[0] <= 10
    assert "paper guardian stopped" in capsys.readouterr().out
    # A normal second start succeeds after Ctrl+C releases both process locks.
    assert module.run_paper_guardian(source[1], watch=False)["account"]["fill_count"] == 0


def test_known_liquidation_does_not_wait_for_other_positions_funding_archive_io(source, monkeypatch):
    client, config, state = source
    state["at"] = NOW.replace(minute=55, second=10)
    book = PaperGuardian(config, client=client)
    seed(book, at=state["at"])
    seed(book, at=state["at"], coin="ETH", identifier="test-eth")
    state["at"] = state["at"].replace(minute=0, second=10)+timedelta(hours=1)
    charge(book, state["at"])
    state["prices"]["SOL"] = 2.05
    before_io = state["at"]
    calls=[]
    def slow(path, archive, *, clock, account):
        calls.append(path)
        assert "SOL" not in account.status()["positions"]
        assert account.status()["fill_count"] == 3
        state["at"] += timedelta(seconds=31)
        return [], account.status()
    monkeypatch.setattr(module, "apply_recorded_funding", slow)
    result = book.tick()
    assert calls == [book.account.path]
    assert result["account"]["fill_count"] == 3 and result["position_count"] == 1
    assert result["actions"][0]["reason"] == "isolated_liquidation"
    assert book.account.audit()["trades"][0]["exit_at"] == before_io.isoformat(timespec="microseconds")


def test_output_path_edit_cannot_silently_start_an_empty_replacement_book(source):
    book, state = make(source)
    config = source[1]
    config.write_text(config.read_text().replace('observations = "data/hyperliquid/live"', 'observations = "data/elsewhere/live"'))
    with pytest.raises(ValueError, match="config changed"):
        PaperGuardian(config, client=None)
    assert book.account.audit()["wallet_balance"] == 10000
    assert not (config.parent.parent/"data"/"elsewhere"/"paper").exists()


@pytest.mark.parametrize("damaged", [b"", b"{", b"\xff"])
def test_malformed_pending_is_retained_and_latest_rebuilt_without_reset(source, damaged):
    book, state = make(source)
    first = book.tick()
    archive = Path(first["source_capture"])
    original_archive = archive.read_bytes()
    before = book.account.path.read_bytes()
    (book.root/"pending.json").write_bytes(damaged)
    (book.root/"latest.json").unlink()
    state["at"] += timedelta(seconds=10)
    result = book.tick()
    assert result["healthy"] and result["account"]["version"] == 0
    assert book.account.path.read_bytes() == before
    assert archive.read_bytes() == original_archive
    backups = list((book.root/"recovery").glob("*.invalid.json"))
    assert len(backups) == 1 and backups[0].read_bytes() == damaged
    record = json.loads(next((book.root/"recovery").glob("*.recovery.json")).read_text())
    assert record["original_sha256"] == file_digest(backups[0])
    assert record["ledger_version"] == 0
    assert json.loads((book.root/"latest.json").read_text())["source_capture"] != str(archive)
    assert not (book.root/"pending.json").exists()


def test_malformed_pending_after_committed_sell_never_repeats_sell(source, monkeypatch):
    book, state = make(source)
    seed(book)
    state["at"] += timedelta(minutes=30)
    original = module.write_json
    def fail_receipt(path, value):
        if str(path).endswith(".receipt.json"):
            raise OSError("interrupted receipt")
        return original(path, value)
    monkeypatch.setattr(module, "write_json", fail_receipt)
    with pytest.raises(OSError, match="interrupted receipt"):
        book.tick()
    before = book.account.audit()
    assert before["fill_count"] == 2 and not before["positions"]
    (book.root/"pending.json").write_bytes(b"")
    monkeypatch.setattr(module, "write_json", original)
    state["at"] += timedelta(seconds=10)
    result = PaperGuardian(source[1], client=source[0]).tick()
    assert result["account"]["fill_count"] == 2
    assert result["account"]["wallet_balance"] == before["wallet_balance"]
    assert book.account.audit()["version"] == before["version"]


def test_malformed_pending_never_selects_an_orphaned_old_quote_for_exit(source, monkeypatch):
    book, state = make(source)
    seed(book)
    state["at"] += timedelta(minutes=30)
    original = book._maintain
    def fail_before_mutation(saved):
        raise OSError("interrupted before mutation")
    monkeypatch.setattr(book, "_maintain", fail_before_mutation)
    with pytest.raises(OSError):
        book.tick()
    old = json.loads((book.root/"pending.json").read_text())
    old_path = book.root/old["capture"]
    old_bytes = old_path.read_bytes()
    (book.root/"pending.json").write_bytes(b"{")
    state["at"] += timedelta(minutes=2)
    state["prices"]["SOL"] = 101.
    monkeypatch.setattr(book, "_maintain", original)
    result = book.tick()
    assert result["account"]["fill_count"] == 2
    trade = book.account.audit()["trades"][0]
    assert trade["exit_at"] == state["at"].isoformat(timespec="microseconds")
    assert trade["exit_fill_price"] > 100
    assert old_path.read_bytes() == old_bytes


def test_malformed_pending_requires_successful_full_ledger_audit(source, monkeypatch):
    book, state = make(source)
    pending = book.root/"pending.json"
    pending.write_bytes(b"")
    requests = len(state["requests"])
    def fail_audit():
        raise ValueError("invented journal mismatch")
    monkeypatch.setattr(book.account, "audit", fail_audit)
    with pytest.raises(ValueError, match="journal mismatch"):
        book.tick()
    assert pending.read_bytes() == b""
    assert not (book.root/"recovery").exists()
    assert len(state["requests"]) == requests


@pytest.mark.parametrize("value", [{}, None, []])
def test_json_valid_but_invalid_pending_structure_is_not_quarantined(source, value):
    book, state = make(source)
    pending = book.root/"pending.json"
    pending.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="invalid structure"):
        book.tick()
    assert pending.exists() and not (book.root/"recovery").exists()


def test_recovery_record_failure_preserves_original_marker_and_ledger(source, monkeypatch):
    book, state = make(source)
    pending = book.root/"pending.json"
    pending.write_bytes(b"")
    before = book.account.path.read_bytes()
    original = module.write_json
    def fail_recovery(path, value):
        if path.parent == book.root/"recovery":
            raise OSError("recovery record disk failure")
        return original(path, value)
    monkeypatch.setattr(module, "write_json", fail_recovery)
    with pytest.raises(OSError, match="recovery record disk failure"):
        book.tick()
    assert pending.read_bytes() == b""
    assert book.account.path.read_bytes() == before
