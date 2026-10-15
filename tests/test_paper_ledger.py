from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import multiprocessing
import os
from pathlib import Path
import sqlite3

import pytest

from local_ai_trader.cli import main
from local_ai_trader.decision.cost_aware import DecisionSettings, ProbabilityPrediction, ReturnForecast, decide_cost_aware, net_return_after_costs
from local_ai_trader.paper.demo import paper_demo, paper_status
from local_ai_trader.paper.ledger import MarketQuote, PaperLedger
from local_ai_trader.risk.paper import PaperEntryRequest, PaperRiskSettings
from local_ai_trader.settings import BacktestSettings

NOW = datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]


def packet(symbol="BTC-USD", now=NOW, *, price=100.0, costs=None, signal_at=None):
    costs = costs or BacktestSettings()
    boundary = datetime.fromtimestamp(int(now.timestamp()) // 300 * 300, timezone.utc)
    signal_at = signal_at or boundary - timedelta(minutes=5)
    signal = ProbabilityPrediction(symbol, signal_at, (0.8, 0.1, 0.1))
    forecast = ReturnForecast(0.03, symbol, signal_at, 300, 1, 6)
    decision = decide_cost_aware(signal, False, DecisionSettings(), costs, 300, 6, forecast, signal)
    return PaperEntryRequest(symbol, signal_at, now, price, decision)


@pytest.fixture
def ledger(tmp_path):
    return PaperLedger.create(tmp_path / "paper/account.sqlite", PaperRiskSettings(), BacktestSettings(), now=NOW, source_mode="synthetic")


def test_entry_restart_horizon_exit_and_cash_pnl_reconcile(ledger):
    entry = ledger.execute_entry("entry", packet(), now=NOW)
    assert entry.status == "filled"
    assert entry.account_version == 1
    assert entry.authorization["account_version"] == 0
    report = ledger.audit()
    assert report["cash"] == pytest.approx(9000)
    assert report["equity"] - report["initial_cash"] == pytest.approx(report["unrealized_net_pnl"])
    assert report["unrealized_net_pnl"] == pytest.approx(-entry.fill["fee"] - entry.fill["price_impact_cost"])
    assert report["realized_net_pnl"] == 0
    due = NOW + timedelta(minutes=30)
    restarted = PaperLedger(ledger.path)
    exit = restarted.exit_due_positions((MarketQuote("BTC-USD", 102, due),), now=due)[0]
    assert exit.status == "filled"
    assert exit.reason == "horizon"
    report = PaperLedger(ledger.path).audit()
    assert report["version"] == report["fill_count"] == 2
    assert len(report["trades"]) == 1
    assert report["positions"] == []
    assert report["unrealized_net_pnl"] == 0
    expected = 1000 * net_return_after_costs(0.02, BacktestSettings())
    assert report["realized_net_pnl"] == pytest.approx(expected)
    assert report["cash"] == report["equity"] == pytest.approx(10000 + expected)
    trade = report["trades"][0]
    assert trade["net_pnl"] == pytest.approx(trade["gross_pnl"] - trade["fees"] - trade["price_impact_cost"])
    assert restarted.exit_due_positions((MarketQuote("BTC-USD", 102, due),), now=due) == []


def test_duplicate_buy_and_sell_replay_after_restart_without_more_fills(ledger):
    entry = ledger.execute_entry("entry", packet(), now=NOW)
    duplicate = PaperLedger(ledger.path).execute_entry("entry", packet(), now=NOW)
    assert duplicate.replayed
    assert replace(duplicate, replayed=False) == entry
    quote = MarketQuote("BTC-USD", 100, NOW + timedelta(minutes=5))
    exit = ledger.execute_exit("exit", quote, now=quote.at)
    replay = PaperLedger(ledger.path).execute_exit("exit", quote, now=quote.at)
    assert replay.replayed
    assert replace(replay, replayed=False) == exit
    assert ledger.audit()["fill_count"] == 2
    assert ledger.audit()["version"] == 2


def test_operation_id_collision_with_changed_inputs_is_refused(ledger):
    ledger.execute_entry("entry", packet(), now=NOW)
    before = ledger.audit()
    with pytest.raises(ValueError, match="different inputs"):
        ledger.execute_entry("entry", packet(price=101), now=NOW)
    assert ledger.audit() == before


def test_blocked_operation_is_also_idempotent_and_saved(ledger):
    stale = replace(packet(), signal_at=NOW - timedelta(minutes=10))
    denied = ledger.execute_entry("old", stale, now=NOW)
    assert (denied.status, denied.reason) == ("blocked", "stale_signal")
    assert PaperLedger(ledger.path).execute_entry("old", stale, now=NOW).replayed
    assert ledger.audit()["version"] == 1
    assert ledger.audit()["fill_count"] == 0


def test_stale_worker_version_cannot_fill_against_changed_account(ledger):
    ledger.set_entries_enabled("pause", False, now=NOW)
    result = ledger.execute_entry("old-worker", packet(), expected_version=0, now=NOW)
    assert result.reason == "account_version_changed"
    assert ledger.audit()["fill_count"] == 0


def test_existing_portfolio_is_remarked_and_sized_from_whole_equity(ledger):
    first = ledger.execute_entry("btc", packet(), now=NOW)
    now = NOW + timedelta(minutes=5)
    result = ledger.execute_entry("eth", packet("ETH-USD", now), (MarketQuote("BTC-USD", 100, now),), now=now)
    assert result.status == "filled"
    equity = 9000 + first.fill["quantity"] * 100
    assert -result.fill["cash_delta"] == pytest.approx(equity * 0.1)
    report = ledger.audit()
    assert len(report["positions"]) == 2
    assert report["equity"] == pytest.approx(report["cash"] + sum(p["quantity"] * p["mark_price"] for p in report["positions"]))


def test_missing_or_stale_other_market_mark_blocks_entry(ledger):
    ledger.execute_entry("btc", packet(), now=NOW)
    now = NOW + timedelta(minutes=5)
    result = ledger.execute_entry("eth", packet("ETH-USD", now), now=now)
    assert result.reason == "stale_position_mark"
    assert ledger.audit()["fill_count"] == 1


def test_observed_daily_loss_blocks_new_entry_without_resetting_the_baseline(ledger):
    ledger.execute_entry("btc", packet(), now=NOW)
    now = NOW + timedelta(minutes=5)
    result = ledger.execute_entry("eth", packet("ETH-USD", now), (MarketQuote("BTC-USD", 70, now),), now=now)
    assert result.reason == "daily_loss_limit"
    report = ledger.audit()
    assert report["day_start_equity"] == 10000
    assert report["fill_count"] == 1


def test_observed_portfolio_drawdown_blocks_new_entries(ledger):
    ledger.execute_entry("btc", packet(), now=NOW)
    ledger.execute_entry("eth", packet("ETH-USD"), now=NOW)
    now = NOW + timedelta(minutes=5)
    quotes = (MarketQuote("BTC-USD", 40, now), MarketQuote("ETH-USD", 40, now))
    result = ledger.execute_entry("new", packet(now=now), quotes, now=now)
    assert result.reason == "drawdown_limit"
    assert ledger.audit()["peak_equity"] == 10000
    assert ledger.audit()["fill_count"] == 2


def test_pause_and_resume_survive_restart(ledger):
    ledger.set_entries_enabled("pause", False, now=NOW)
    restarted = PaperLedger(ledger.path)
    assert restarted.execute_entry("paused-buy", packet(), now=NOW).reason == "new_entries_paused"
    restarted.set_entries_enabled("resume", True, now=NOW)
    assert PaperLedger(ledger.path).execute_entry("enabled-buy", packet(), now=NOW).status == "filled"


@pytest.mark.parametrize("exit_reason", ["policy_sell", "risk", "manual", "horizon"])
def test_entry_pause_and_loss_limits_never_veto_fresh_reducing_exit(ledger, exit_reason):
    ledger.execute_entry("btc", packet(), now=NOW)
    ledger.set_entries_enabled("pause", False, now=NOW)
    now = NOW + timedelta(minutes=30)
    quote = MarketQuote("BTC-USD", 50, now)
    result = ledger.execute_exit("exit", quote, reason=exit_reason, now=now)
    assert result.status == "filled"
    assert ledger.audit()["positions"] == []
    assert ledger.audit()["realized_net_pnl"] < -400
    assert ledger.execute_entry("new", packet(now=now), now=now).reason == "new_entries_paused"


def test_exit_does_not_require_fresh_marks_for_unrelated_holdings(ledger):
    ledger.execute_entry("btc", packet(), now=NOW)
    ledger.execute_entry("eth", packet("ETH-USD"), now=NOW)
    now = NOW + timedelta(minutes=10)
    result = ledger.execute_exit("exit-btc", MarketQuote("BTC-USD", 100, now), now=now)
    assert result.status == "filled"
    assert ledger.audit()["positions"][0]["symbol"] == "ETH-USD"


def test_exit_checks_freshness_due_time_and_position_identity(ledger):
    ledger.execute_entry("btc", packet(), now=NOW)
    assert ledger.execute_exit("early", MarketQuote("BTC-USD", 100, NOW), reason="horizon", now=NOW).reason == "exit_not_due"
    now = NOW + timedelta(minutes=5)
    assert ledger.execute_exit("stale", MarketQuote("BTC-USD", 100, now - timedelta(seconds=30)), now=now).reason == "stale_exit_quote"
    assert ledger.execute_exit("wrong-position", MarketQuote("BTC-USD", 100, now), position_id="old-entry", now=now).reason == "position_changed"
    assert ledger.audit()["fill_count"] == 1
    assert ledger.execute_exit("absent", MarketQuote("ETH-USD", 100, now), now=now).reason == "no_position"


def test_horizon_exit_requires_quote_from_the_due_time_or_later(ledger):
    ledger.execute_entry("btc", packet(), now=NOW)
    due = NOW + timedelta(minutes=30)
    result = ledger.execute_exit("early-quote", MarketQuote("BTC-USD", 100, due - timedelta(seconds=1)), reason="horizon", now=due)
    assert result.reason == "quote_precedes_horizon_exit"
    assert ledger.audit()["fill_count"] == 1


def test_horizon_operation_ids_support_long_entry_ids(ledger):
    ledger.execute_entry("b" * 200, packet(), now=NOW)
    due = NOW + timedelta(minutes=30)
    assert ledger.exit_due_positions((MarketQuote("BTC-USD", 102, due),), now=due)[0].status == "filled"


def test_duplicate_horizon_quotes_are_refused_before_any_exit(ledger):
    ledger.execute_entry("btc", packet(), now=NOW)
    due = NOW + timedelta(minutes=30)
    quotes = (MarketQuote("BTC-USD", 102, due), MarketQuote("BTC-USD", 103, due))
    with pytest.raises(ValueError, match="duplicate"):
        ledger.exit_due_positions(quotes, now=due)
    assert ledger.audit()["fill_count"] == 1


def test_real_quote_timestamp_is_not_refreshed_to_fill_time(ledger):
    now = NOW + timedelta(seconds=20)
    entry = ledger.execute_entry("btc", packet(), now=now)
    assert entry.status == "filled"
    position = ledger.audit()["positions"][0]
    assert datetime.fromisoformat(position["marked_at"]) == NOW
    assert datetime.fromisoformat(position["entry_at"]) == now
    assert datetime.fromisoformat(position["exit_due_at"]) == now + timedelta(minutes=30)


def test_clock_and_mark_regressions_roll_back(ledger):
    ledger.execute_entry("btc", packet(), now=NOW)
    before = ledger.audit()
    with pytest.raises(ValueError, match="clock"):
        ledger.set_entries_enabled("pause", False, now=NOW - timedelta(seconds=1))
    with pytest.raises(ValueError, match="mark"):
        ledger.execute_entry("eth", packet("ETH-USD"), (MarketQuote("BTC-USD", 100, NOW - timedelta(seconds=1)),), now=NOW)
    assert ledger.audit() == before


@pytest.mark.parametrize("quote", [MarketQuote("BTC-USD", -1, NOW), MarketQuote("BTC-USD", float("nan"), NOW), MarketQuote("BTC-USD", 100, NOW + timedelta(seconds=1))])
def test_invalid_market_observations_roll_back(ledger, quote):
    before = ledger.audit()
    with pytest.raises(ValueError):
        ledger.execute_entry("entry", packet(), (quote,), now=NOW)
    assert ledger.audit() == before


def test_interrupted_save_rolls_back_cash_holding_fill_journal_and_version(ledger, monkeypatch):
    original = ledger._save
    before = ledger.audit()
    def fail_after_save(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("simulated crash before commit")
    monkeypatch.setattr(ledger, "_save", fail_after_save)
    with pytest.raises(RuntimeError, match="simulated crash"):
        ledger.execute_entry("btc", packet(), now=NOW)
    assert PaperLedger(ledger.path).audit() == before
    monkeypatch.setattr(ledger, "_save", original)
    assert ledger.execute_entry("btc", packet(), now=NOW).status == "filled"


def test_interrupted_exit_retains_position_and_entry_costs(ledger, monkeypatch):
    ledger.execute_entry("btc", packet(), now=NOW)
    before = ledger.audit()
    original = ledger._save
    def fail_after_save(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("simulated exit crash")
    monkeypatch.setattr(ledger, "_save", fail_after_save)
    now = NOW + timedelta(minutes=30)
    with pytest.raises(RuntimeError):
        ledger.execute_exit("exit", MarketQuote("BTC-USD", 102, now), now=now)
    assert PaperLedger(ledger.path).audit() == before


@pytest.mark.parametrize("stage", ["before_fill", "before_commit"])
def test_expiring_real_time_approval_cannot_commit(ledger, monkeypatch, stage):
    import local_ai_trader.paper.ledger as module
    stamps = iter([NOW, NOW + timedelta(seconds=6)] if stage == "before_fill" else [NOW, NOW, NOW + timedelta(seconds=6)])
    class FakeClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return next(stamps)
    monkeypatch.setattr(module, "datetime", FakeClock)
    if stage == "before_fill":
        result = ledger.execute_entry("btc", packet())
        assert result.reason == "approval_expired_or_inputs_changed"
    else:
        with pytest.raises(ValueError, match="expired before commit"):
            ledger.execute_entry("btc", packet())
    assert ledger.audit()["fill_count"] == 0
    assert ledger.audit()["positions"] == []


def test_cash_and_inventory_reconcile_after_multiple_round_trips(ledger):
    for index, price in enumerate((102, 98, 100)):
        now = NOW + timedelta(minutes=index * 40)
        assert ledger.execute_entry(f"buy-{index}", packet(now=now), now=now).status == "filled"
        exit_at = now + timedelta(minutes=30)
        assert ledger.execute_exit(f"sell-{index}", MarketQuote("BTC-USD", price, exit_at), now=exit_at).status == "filled"
    report = PaperLedger(ledger.path).audit()
    assert report["fill_count"] == 6
    assert len(report["trades"]) == 3
    assert report["cash"] - 10000 == pytest.approx(sum(t["net_pnl"] for t in report["trades"]))


def test_overnight_hold_requires_known_midnight_baseline_and_exit_still_works(tmp_path):
    start = NOW.replace(hour=23, minute=55)
    ledger = PaperLedger.create(tmp_path / "overnight.sqlite", PaperRiskSettings(), BacktestSettings(), now=start)
    entry = ledger.execute_entry("btc", packet(now=start), now=start)
    next_day = start + timedelta(minutes=10)
    marks = (MarketQuote("BTC-USD", 100, next_day),)
    blocked = ledger.execute_entry("eth", packet("ETH-USD", next_day), marks, now=next_day)
    assert blocked.reason == "utc_day_baseline_unavailable"
    baseline = (MarketQuote("BTC-USD", 101, next_day.replace(hour=0, minute=0)),)
    filled = ledger.execute_entry("eth-new", packet("ETH-USD", next_day), marks, day_quotes=baseline, now=next_day)
    assert filled.status == "filled"
    assert ledger.audit()["day_start_equity"] == pytest.approx(9000 + entry.fill["quantity"] * 101)


def test_intraday_exit_cannot_erase_unknown_overnight_daily_loss_baseline(tmp_path):
    start = NOW.replace(hour=23, minute=55)
    ledger = PaperLedger.create(tmp_path / "overnight.sqlite", PaperRiskSettings(), BacktestSettings(), now=start)
    ledger.execute_entry("btc", packet(now=start), now=start)
    next_day = start + timedelta(minutes=10)
    assert ledger.execute_exit("exit", MarketQuote("BTC-USD", 99, next_day), now=next_day).status == "filled"
    assert ledger.execute_entry("retry", packet(now=next_day), now=next_day).reason == "utc_day_baseline_unavailable"
    baseline = (MarketQuote("BTC-USD", 100, next_day.replace(hour=0, minute=0)),)
    assert ledger.execute_entry("with-baseline", packet(now=next_day), day_quotes=baseline, now=next_day).status == "filled"


def test_flat_account_midnight_baseline_reconstructed_from_cash_history(ledger):
    ledger.execute_entry("btc", packet(), now=NOW)
    exit_at = NOW + timedelta(minutes=30)
    ledger.execute_exit("exit", MarketQuote("BTC-USD", 99, exit_at), now=exit_at)
    cash = ledger.audit()["cash"]
    next_day = NOW + timedelta(days=1)
    assert ledger.execute_entry("next-day", packet(now=next_day), now=next_day).status == "filled"
    assert ledger.audit()["day_start_equity"] == cash


@pytest.mark.parametrize("sql", [
    "UPDATE account SET cash=cash+1",
    "UPDATE account SET peak_equity=peak_equity-1",
    "UPDATE account SET day_start_equity=day_start_equity-1",
    "UPDATE account SET entries_enabled=0",
    "UPDATE positions SET quantity=quantity+1",
    "UPDATE positions SET mark_price=mark_price+1",
    "UPDATE fills SET fee=fee+1",
    "UPDATE fills SET cash_delta=cash_delta+1",
    "UPDATE operations SET result='{}'",
    "DELETE FROM operations",
])
def test_corrupted_account_inventory_costs_or_journal_are_refused(ledger, sql):
    ledger.execute_entry("btc", packet(), now=NOW)
    with sqlite3.connect(str(ledger.path)) as db:
        db.execute(sql)
    with pytest.raises(ValueError):
        PaperLedger(ledger.path)


def test_corrupted_closed_trade_is_refused(ledger):
    ledger.execute_entry("btc", packet(), now=NOW)
    now = NOW + timedelta(minutes=30)
    ledger.execute_exit("exit", MarketQuote("BTC-USD", 102, now), now=now)
    with sqlite3.connect(str(ledger.path)) as db:
        db.execute("UPDATE trades SET net_pnl=net_pnl+1")
    with pytest.raises(ValueError, match="PnL"):
        PaperLedger(ledger.path)


def worker(path, operation_id):
    ledger = PaperLedger(Path(path))
    result = ledger.execute_entry(operation_id, packet(), now=NOW)
    return result.status, result.reason, result.replayed


def crash_worker(path):
    ledger = PaperLedger(Path(path))
    original = ledger._save
    def die_before_commit(*args, **kwargs):
        original(*args, **kwargs)
        os._exit(77)
    ledger._save = die_before_commit
    ledger.execute_entry("interrupted", packet(), now=NOW)


def test_abrupt_process_death_before_commit_recovers_without_a_partial_fill(ledger):
    before = ledger.audit()
    process = multiprocessing.get_context("spawn").Process(target=crash_worker, args=(str(ledger.path),))
    process.start()
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
        pytest.fail("Crash-recovery worker did not finish")
    assert process.exitcode == 77
    assert PaperLedger(ledger.path).audit() == before
    assert ledger.execute_entry("interrupted", packet(), now=NOW).status == "filled"


def test_failed_creation_closes_database_and_removes_incomplete_file(tmp_path, monkeypatch):
    import local_ai_trader.paper.ledger as module
    path = tmp_path / "bad.sqlite"
    monkeypatch.setattr(module, "SCHEMA", "invalid SQL")
    with pytest.raises(sqlite3.Error):
        PaperLedger.create(path, PaperRiskSettings(), BacktestSettings(), now=NOW)
    assert not path.exists()


def test_concurrent_threads_cannot_duplicate_a_market_position(ledger):
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda name: worker(str(ledger.path), name), ("a", "b")))
    assert sorted(status for status, _, _ in results) == ["blocked", "filled"]
    assert ledger.audit()["fill_count"] == 1
    assert len(ledger.audit()["positions"]) == 1


def test_concurrent_process_delivery_of_same_operation_commits_once(ledger):
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = [executor.submit(worker, str(ledger.path), "same") for _ in range(2)]
        results = [future.result(timeout=30) for future in futures]
    assert all(status == "filled" for status, _, _ in results)
    assert sorted(replayed for _, _, replayed in results) == [False, True]
    assert ledger.audit()["version"] == ledger.audit()["fill_count"] == 1


def test_creation_never_overwrites_existing_account_and_missing_account_fails(ledger, tmp_path):
    before = ledger.path.read_bytes()
    with pytest.raises(FileExistsError):
        PaperLedger.create(ledger.path, PaperRiskSettings(), BacktestSettings(), now=NOW)
    assert ledger.path.read_bytes() == before
    with pytest.raises(ValueError, match="not found"):
        PaperLedger(tmp_path / "absent.sqlite")


def test_long_native_windows_account_path_works_without_system_opt_in(tmp_path):
    root = tmp_path
    while len(str(root)) < 300:
        root /= "paper-account-deep-directory"
    ledger = PaperLedger.create(root / "account.sqlite", PaperRiskSettings(), BacktestSettings(), now=NOW)
    assert ledger.execute_entry("btc", packet(), now=NOW).status == "filled"
    assert PaperLedger(ledger.path).audit()["reconciled"]
    now = NOW + timedelta(minutes=30)
    assert ledger.execute_exit("exit", MarketQuote("BTC-USD", 102, now), now=now).status == "filled"


def temporary_config(tmp_path):
    path = tmp_path / "config/settings.toml"
    path.parent.mkdir()
    path.write_text((ROOT / "config/settings.toml").read_text(encoding="utf-8"), encoding="utf-8")
    return path


def test_persisted_synthetic_demo_and_read_only_status(tmp_path, capsys):
    path = paper_demo(temporary_config(tmp_path), now=NOW)
    assert "PASS: persisted cash, inventory and trade PnL reconcile" in capsys.readouterr().out
    before = path.read_bytes()
    report = paper_status(path)
    assert report["source_mode"] == "synthetic"
    assert report["fill_count"] == 2
    assert path.read_bytes() == before
    assert len(report["trades"]) == 1


def test_paper_demo_and_status_cli(tmp_path, capsys):
    config = temporary_config(tmp_path)
    assert main(["paper-demo", "--config", str(config)]) == 0
    account = next((tmp_path / "data/paper").rglob("account.sqlite"))
    assert main(["paper-status", str(account)]) == 0
    output = capsys.readouterr().out
    assert "Synthetic persisted paper workflow" in output
    assert "Paper account (synthetic inputs" in output


def test_cli_missing_or_invalid_account_returns_failure(tmp_path):
    assert main(["paper-status", str(tmp_path / "absent.sqlite")]) == 1
    path = tmp_path / "invalid.sqlite"
    path.write_bytes(b"invalid sqlite contents")
    assert main(["paper-status", str(path)]) == 1
