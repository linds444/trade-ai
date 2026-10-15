from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta
import json
from pathlib import Path

import pytest

from local_ai_trader.cli import main
from local_ai_trader.data.storage import native_path, write_json
from local_ai_trader.hyperliquid import decisions, paper_monitor
from local_ai_trader.models.train import file_digest
from local_ai_trader.paper.perpetual import PerpetualPaperAccount
from local_ai_trader.risk.entry_policy import SignalEvidence

from test_hyperliquid_decisions import scanner, NOW


def monitor(scanner):
    client, config, models = scanner
    path, report = decisions.create_decision_preview(client, config, models)
    result = paper_monitor.monitor_preview(client, path, report)
    return client, path, report, result


def revised(path, report, client, at, price=102):
    """Invented controlled-clock observations, never an external feed assertion."""
    raw = json.loads((path / "raw_market.json").read_text())
    for key, payload in raw.items():
        if key.endswith("_book"):
            payload["time"] = int(at.timestamp()*1000)
            payload["levels"][0][0]["px"] = str(price-0.01)
            payload["levels"][1][0]["px"] = str(price+0.01)
        elif key == "contexts" or key.endswith("_contexts"):
            for context in payload[1]:
                context["markPx"] = str(price)
    result = deepcopy(report)
    result["created_at"] = at.isoformat()
    for coin, asset in result["market_snapshot"]["assets"].items():
        asset.update(context_received_at=at.isoformat(), mark_price=price)
        asset["book"].update(exchange_time=at.isoformat(), bid=price-0.01, ask=price+0.01)
        result["decisions"][coin]["signal_at"] = (at.replace(second=0, minute=at.minute//5*5)-timedelta(minutes=5)).isoformat()
    output = path.parent / ("preview_test_" + at.strftime("%H%M%S"))
    write_json(output / "raw_market.json", raw)
    (output / "input_config.toml").write_bytes((path / "input_config.toml").read_bytes())
    result["artifact_hashes"]["raw_market.json"] = file_digest(output / "raw_market.json")
    write_json(output / "decisions.json", result)
    client.clock = lambda: at
    return output, result


def seed_holding(result, at):
    """Test-only approved synthetic evidence inserted into an isolated observed-mode fixture."""
    account = PerpetualPaperAccount(Path(result["account_path"]))
    account.execute("test-unpause", "pause", {"enabled": True}, now=at)
    quote = {"coin": "BTC", "at": at, "bid": 99.99, "ask": 100.01, "mark": 100,
             "fee_bps": 4.5, "size_decimals": 4, "source_mode": "observed"}
    inputs = {"quote": quote, "quantity": 9,
              "signal_at": at.replace(second=0, minute=at.minute//5*5)-timedelta(minutes=5),
              "evidence": asdict(SignalEvidence("BUY", 0.8, "test-calibration", 0.8, "test-reliability")),
              "expected_gross_return": 0.03, "expected_funding_cost_fraction": 0.0001}
    assert account.execute("test-holding", "entry", inputs, now=at)["status"] == "filled"
    return account


def test_observed_book_records_all_gate_skips_without_fills_or_trial(scanner):
    client, path, report, result = monitor(scanner)
    assert result["account"]["wallet_balance"] == 10000
    assert result["account"]["fill_count"] == 0
    assert result["account"]["source_mode"] == "observed"
    assert result["account"]["entries_enabled"] is False
    assert result["profitability_trial_started"] is False and result["reliability_validated"] is False
    assert len(result["entry_reviews"]) == 8
    assert all(row["execution_reason"] == "validated_confidence_unavailable" for row in result["entry_reviews"].values())
    account = PerpetualPaperAccount(Path(result["account_path"]))
    assert account.contract()["run_binding"]["model_manifest_sha256"] == report["model_manifest_sha256"]
    assert account.status() == account.audit()
    with account._transaction() as db:
        event = json.loads(db.execute("SELECT event FROM events").fetchone()[0])
    assert event["inputs"]["source_preview_sha256"] == result["source_preview_sha256"]
    assert event["inputs"]["source_decisions"] == report["decisions"]
    pepe = next(q for q in event["inputs"]["quotes"] if q["coin"] == "kPEPE")
    assert pepe["mark"] == 100  # native contract units are never divided by 1000 in accounting


def test_passing_signal_cannot_unpause_monitor_or_trigger_entry(scanner):
    client, config, models = scanner
    path, report = decisions.create_decision_preview(client, config, models)
    for row in report["decisions"].values():
        row.update(confidence=0.9, reliability_validation_id="test-only-reliability")
    write_json(path / "decisions.json", report)
    result = paper_monitor.monitor_preview(client, path, report)
    assert all(r["signal_gate"]["status"] == "PASS" and r["execution_reason"] == "paper_monitor_entries_paused" for r in result["entry_reviews"].values())
    assert result["account"]["fill_count"] == 0 and result["account"]["entries_enabled"] is False


def test_restart_duplicate_preserves_first_receipt_cash_and_journal(scanner):
    client, path, report, first = monitor(scanner)
    del client._paper_monitors
    again = paper_monitor.monitor_preview(client, path, report)
    assert again == first
    assert PerpetualPaperAccount(Path(first["account_path"])).audit()["version"] == 1


@pytest.mark.parametrize("age", [30, -0.001])
def test_expired_or_future_quote_does_not_create_account(scanner, age):
    client, config, models = scanner
    path, report = decisions.create_decision_preview(client, config, models)
    client.clock = lambda: NOW+timedelta(seconds=age)
    with pytest.raises(ValueError):
        paper_monitor.monitor_preview(client, path, report)
    assert not (client.settings.observations.parent / "paper").exists()


@pytest.mark.parametrize("field", ["mark_price", "size_decimals", "taker_fee_bps_assumption"])
def test_report_cannot_substitute_prices_precision_or_costs_for_raw_evidence(scanner, field):
    client, config, models = scanner
    path, report = decisions.create_decision_preview(client, config, models)
    report["market_snapshot"]["assets"]["BTC"][field] += 1
    write_json(path / "decisions.json", report)
    with pytest.raises(ValueError, match="raw evidence"):
        paper_monitor.monitor_preview(client, path, report)
    assert not (client.settings.observations.parent / "paper").exists()


def test_changed_raw_artifact_refuses_before_account_write(scanner):
    client, config, models = scanner
    path, report = decisions.create_decision_preview(client, config, models)
    with (path / "raw_market.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="artifact changed"):
        paper_monitor.monitor_preview(client, path, report)


def test_missing_account_is_not_silently_recreated(scanner):
    client, path, report, result = monitor(scanner)
    Path(result["account_path"]).unlink()
    with pytest.raises(ValueError, match="refusing to reset"):
        paper_monitor.monitor_preview(client, path, report)
    assert not Path(result["account_path"]).exists()


def test_changed_run_manifest_and_receipt_are_detected(scanner):
    client, path, report, result = monitor(scanner)
    root = Path(result["account_path"]).parent
    receipt = root / "observations" / (path.name+".json")
    damaged = json.loads(receipt.read_text())
    damaged["account"]["wallet_balance"] += 100
    write_json(receipt, damaged)
    with pytest.raises(ValueError, match="receipt changed"):
        paper_monitor.monitor_preview(client, path, report)
    assert PerpetualPaperAccount(Path(result["account_path"])).audit()["version"] == 1
    run = json.loads((root / "run.json").read_text())
    run["entries_enabled"] = True
    write_json(root / "run.json", run)
    with pytest.raises(ValueError, match="binding changed"):
        paper_monitor.monitor_preview(client, path, report)


def test_crash_between_mark_commit_and_receipt_retries_without_double_record(scanner, monkeypatch):
    client, config, models = scanner
    path, report = decisions.create_decision_preview(client, config, models)
    original = paper_monitor.write_json
    def fail_receipt(target, payload):
        if target.parent.name == "observations":
            raise OSError("test crash after mark")
        original(target, payload)
    monkeypatch.setattr(paper_monitor, "write_json", fail_receipt)
    with pytest.raises(OSError, match="test crash"):
        paper_monitor.monitor_preview(client, path, report)
    del client._paper_monitors
    monkeypatch.setattr(paper_monitor, "write_json", original)
    result = paper_monitor.monitor_preview(client, path, report)
    assert result["mark_result"]["replayed"] is True
    assert result["account"]["version"] == 1 and result["account"]["wallet_balance"] == 10000


def test_due_holding_closes_after_restart_before_broken_funding_archive(scanner, monkeypatch):
    client, path, report, first = monitor(scanner)
    entered = NOW.replace(minute=55)
    seed_holding(first, entered)
    at = entered+timedelta(minutes=31)
    path, report = revised(path, report, client, at)
    del client._paper_monitors
    def broken(account_path, archive, **kwargs):
        assert PerpetualPaperAccount(account_path).status()["positions"] == {}
        raise ValueError("test missing oracle evidence")
    monkeypatch.setattr(paper_monitor, "apply_recorded_funding", broken)
    result = paper_monitor.monitor_preview(client, path, report)
    assert result["exit_results"][0]["status"] == "filled"
    assert result["account"]["fill_count"] == 2 and result["account"]["positions"] == {}
    assert result["account"]["entries_enabled"] is False
    assert result["account"]["performance_provisional"] is True
    assert result["funding_error"] == "test missing oracle evidence"
    assert datetime.fromisoformat(result["account"]["trades"][0]["exit_at"]) == at
    assert result["account"]["trades"][0]["exit_fill_price"] == pytest.approx(101.99*0.9999)
    assert paper_monitor.monitor_preview(client, path, report)["account"]["fill_count"] == 2


def test_due_holding_posts_only_verified_adapter_funding_after_exit(scanner, monkeypatch):
    client, path, report, first = monitor(scanner)
    entered = NOW.replace(minute=55)
    seed_holding(first, entered)
    at = entered+timedelta(minutes=31)
    path, report = revised(path, report, client, at)
    hour = entered.replace(minute=0, second=0)+timedelta(hours=1)
    def settled(account_path, archive, **kwargs):
        account = PerpetualPaperAccount(account_path)
        event = {"coin": "BTC", "hour": hour, "exchange_event_at": hour+timedelta(milliseconds=30),
                 "rate": 0.0001, "oracle_price": 101, "oracle_at": hour,
                 "rate_observed_at": hour+timedelta(seconds=1), "rate_source_id": "test-settled-rate",
                 "oracle_source_id": "test-observed-oracle", "source_mode": "observed"}
        return [account.execute("test-verified-funding", "funding", event, clock=kwargs["clock"])], account.status()
    monkeypatch.setattr(paper_monitor, "apply_recorded_funding", settled)
    result = paper_monitor.monitor_preview(client, path, report)
    assert result["funding_results"][0]["cash_flow"] == pytest.approx(-0.0909)
    assert result["account"]["performance_provisional"] is False
    assert result["account"]["wallet_balance"]-10000 == pytest.approx(result["account"]["realized_net_pnl"])


def test_running_clock_detects_expiration_after_persist_and_rolls_back(scanner, monkeypatch):
    client, path, report, first = monitor(scanner)
    account = PerpetualPaperAccount(Path(first["account_path"]))
    original = account._persist
    current = [NOW]
    def delay(*args):
        original(*args)
        current[0] += timedelta(seconds=5)
    monkeypatch.setattr(account, "_persist", delay)
    with pytest.raises(ValueError, match="expired before commit"):
        account.execute("test-delayed-pause", "pause", {"enabled": True}, clock=lambda: current[0])
    assert account.audit()["version"] == 1 and account.audit()["entries_enabled"] is False


def test_quote_expiring_inside_transaction_rolls_back_even_before_operation_deadline(scanner):
    client, path, report, first = monitor(scanner)
    account = PerpetualPaperAccount(Path(first["account_path"]))
    quote = {"coin": "BTC", "at": NOW, "bid": 99.99, "ask": 100.01, "mark": 100,
             "fee_bps": 4.5, "size_decimals": 4, "source_mode": "observed"}
    times = iter([NOW+timedelta(seconds=28), NOW+timedelta(seconds=30)])
    with pytest.raises(ValueError, match="stale"):
        account.execute("test-quote-expiry", "mark", {"quotes": [quote]}, clock=lambda: next(times))
    assert account.audit()["version"] == 1


def test_crash_after_exit_before_receipt_does_not_refill_on_restart(scanner, monkeypatch):
    client, path, report, first = monitor(scanner)
    entered = NOW.replace(minute=55)
    seed_holding(first, entered)
    path, report = revised(path, report, client, entered+timedelta(minutes=31))
    original = paper_monitor.write_json
    def failed(target, payload):
        if target.parent.name == "observations":
            raise OSError("test crash after exit")
        return original(target, payload)
    monkeypatch.setattr(paper_monitor, "write_json", failed)
    with pytest.raises(OSError, match="crash after exit"):
        paper_monitor.monitor_preview(client, path, report)
    account = PerpetualPaperAccount(Path(first["account_path"]))
    assert account.audit()["fill_count"] == 2
    version = account.status()["version"]
    del client._paper_monitors
    monkeypatch.setattr(paper_monitor, "write_json", original)
    result = paper_monitor.monitor_preview(client, path, report)
    assert result["mark_result"]["replayed"] is True
    assert result["account"]["fill_count"] == 2 and result["account"]["version"] == version


def test_cli_monitor_runs_before_outcome_resolution_and_preserves_both_books(scanner, monkeypatch, capsys):
    client, config, models = scanner
    monkeypatch.setattr(decisions, "HyperliquidPublicClient", lambda settings: client)
    assert main(["hyperliquid-decisions", "--models", str(models), "--config", str(config), "--track-outcomes", "--paper-monitor"]) == 0
    output = capsys.readouterr().out
    assert output.index("Observed paper monitor") < output.index("Forward forecast observations")
    assert "8 recorded; 8 pending; 0 resolved" in output
    assert "profitability month has not started" in output
    assert "fills 0" in output


def test_monitor_account_and_receipt_support_deep_native_windows_paths(scanner, tmp_path):
    client, path, report, first = monitor(scanner)
    # Native path coverage for the SQLite/account methods, also exercised in Windows CI.
    deep = native_path(tmp_path / ("nested_"*18) / ("more_"*20) / "account.sqlite")
    account = PerpetualPaperAccount.create(deep, now=NOW, source_mode="observed", entries_enabled=False,
        run_binding={"purpose": "test-long-path"})
    assert len(str(deep)) > 260
    assert account.contract()["run_binding"] == {"purpose": "test-long-path"}
    assert PerpetualPaperAccount(deep).status()["wallet_balance"] == 10000
