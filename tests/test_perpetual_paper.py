from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3

import pytest

from local_ai_trader.cli import main
from local_ai_trader.paper.perpetual import PerpetualPaperAccount
from local_ai_trader.paper import perpetual
from local_ai_trader.paper.perpetual_demo import perpetual_demo, perpetual_status
from local_ai_trader.risk.entry_policy import EntryPolicy, SignalEvidence

NOW = datetime(2026, 1, 1, 12, 55, 10, tzinfo=timezone.utc)
HOUR = NOW.replace(minute=0, second=0) + timedelta(hours=1)
ROOT = Path(__file__).resolve().parents[1]


def quote(at=NOW, price=100, coin="BTC", **changes):
    return {"coin": coin, "at": at, "bid": price - 0.01, "ask": price + 0.01,
            "mark": price, "fee_bps": 4.5, "size_decimals": 3, "source_mode": "synthetic", **changes}


def entry(at=NOW, coin="BTC", quantity=9, **changes):
    return {"quote": quote(at, coin=coin), "quantity": quantity,
            "signal_at": at.replace(minute=at.minute // 5 * 5, second=0, microsecond=0) - timedelta(minutes=5),
            "evidence": asdict(SignalEvidence("BUY", 0.8, "synthetic-calibration", 0.8, "synthetic-reliability")),
            "expected_gross_return": 0.03, "expected_funding_cost_fraction": 0.000101, **changes}


def funding(coin="BTC", rate=0.0001, hour=HOUR, **changes):
    return {"coin": coin, "hour": hour, "rate": rate, "oracle_price": 101., "oracle_at": hour,
            "rate_source_id": "synthetic-settled-rate", "oracle_source_id": "synthetic-oracle", "source_mode": "synthetic", **changes}


@pytest.fixture
def account(tmp_path):
    return PerpetualPaperAccount.create(tmp_path / "perpetual/account.sqlite", now=NOW)


def process_entry(path):
    return PerpetualPaperAccount(Path(path)).execute("entry", "entry", entry(), now=NOW)


def crash_after_write(path):
    account = PerpetualPaperAccount(Path(path))
    original = account._persist
    def crash(*args):
        original(*args)
        os._exit(79)
    account._persist = crash
    account.execute("entry", "entry", entry(), now=NOW)


def close(account, at=NOW + timedelta(minutes=30), price=102, reason="horizon", operation="exit"):
    return account.execute(operation, "exit", {"quote": quote(at, price), "position_id": "entry", "reason": reason}, now=at)


def test_wallet_retains_principal_while_one_times_initial_margin_is_reserved(account):
    result = account.execute("entry", "entry", entry(), now=NOW)
    report = account.audit()
    assert result["status"] == "filled"
    fill = result["fill"]
    assert report["wallet_balance"] == pytest.approx(10000 - fill["fee"])
    assert report["reserved_initial_margin"] == pytest.approx(9 * fill["price"])
    assert report["equity"] == pytest.approx(10000 - fill["fee"] + 9 * (100 - fill["price"]))
    assert report["available_collateral"] == pytest.approx(report["equity"] - report["reserved_initial_margin"])
    assert report["configured_leverage"] == 1 and report["currency"] == "USDC"


@pytest.mark.parametrize("rate", [0.0001, -0.0001, 0])
def test_hourly_funding_uses_oracle_not_mark_and_positive_rate_charges_longs(account, rate):
    account.execute("entry", "entry", entry(), now=NOW)
    account.execute("mark", "mark", {"quotes": [quote(HOUR, 200)]}, now=HOUR)
    result = account.execute("funding", "funding", funding(rate=rate), now=HOUR)
    expected = -9 * 101 * rate
    assert result["cash_flow"] == pytest.approx(expected)
    assert account.audit()["positions"]["BTC"]["funding_cash_flow"] == pytest.approx(expected)
    assert account.audit()["missing_funding"] == []


def test_restart_duplicates_and_horizon_exit_reconcile_all_costs_and_margin(account):
    original = account.execute("entry", "entry", entry(), now=NOW)
    assert PerpetualPaperAccount(account.path).execute("entry", "entry", entry(), now=HOUR)["replayed"]
    settled = account.execute("funding", "funding", funding(), now=HOUR)
    assert PerpetualPaperAccount(account.path).execute("funding", "funding", funding(), now=HOUR + timedelta(seconds=1))["replayed"]
    exit_result = close(PerpetualPaperAccount(account.path))
    report = account.audit()
    trade = report["trades"][0]
    expected = trade["gross_pnl"] - trade["price_impact_cost"] - trade["fees"] + settled["cash_flow"]
    assert trade["net_pnl"] == pytest.approx(expected)
    assert report["wallet_balance"] == report["equity"] == pytest.approx(10000 + expected)
    assert report["realized_net_pnl"] == pytest.approx(expected)
    assert report["reserved_initial_margin"] == 0 and report["positions"] == {}
    assert report["fill_count"] == 2 and report["version"] == 3
    assert close(PerpetualPaperAccount(account.path))["replayed"]
    assert original["fill"]["side"] == "BUY" and exit_result["fill"]["side"] == "SELL"


def test_unpaid_funding_blocks_new_entries_but_exits_and_later_settlement_work(account):
    account.execute("entry", "entry", entry(), now=NOW)
    closed = close(account)
    assert closed["status"] == "filled"
    provisional = account.audit()
    assert provisional["performance_provisional"] and len(provisional["missing_funding"]) == 1
    at = NOW + timedelta(minutes=31)
    result = account.execute("new", "entry", entry(at, coin="ETH"), now=at)
    assert result["reason"] == "missing_funding_settlement"
    account.execute("late-funding", "funding", funding(), now=at)
    final = account.audit()
    assert not final["performance_provisional"]
    assert final["trades"][0]["funding_cash_flow"] == pytest.approx(-0.0909)
    assert final["wallet_balance"] - 10000 == pytest.approx(final["realized_net_pnl"])


def test_funding_does_not_apply_before_entry_or_after_exit_and_cannot_double_charge(account):
    account.execute("entry", "entry", entry(), now=NOW)
    close(account, at=NOW + timedelta(minutes=1), reason="manual")
    result = account.execute("funding", "funding", funding(), now=HOUR)
    assert result["cash_flow"] == 0
    assert account.execute("same-hour", "funding", funding(), now=HOUR)["reason"] == "funding_already_posted"
    with pytest.raises(ValueError, match="Changed funding"):
        account.execute("changed-rate", "funding", funding(rate=0.001), now=HOUR)
    assert account.audit()["trades"][0]["funding_cash_flow"] == 0


@pytest.mark.parametrize("changes", [
    {"rate": 0.041}, {"rate": float("nan")}, {"oracle_price": 0},
    {"hour": HOUR + timedelta(minutes=1)}, {"hour": HOUR + timedelta(hours=1)},
    {"oracle_at": HOUR + timedelta(seconds=30)}, {"oracle_at": HOUR - timedelta(seconds=30)},
    {"rate_source_id": ""}, {"oracle_source_id": ""}, {"coin": "UNKNOWN"},
])
def test_invalid_or_unproven_funding_never_changes_account(account, changes):
    before = account.audit()
    with pytest.raises(ValueError):
        account.execute("bad-funding", "funding", funding(**changes), now=HOUR)
    assert account.audit() == before


@pytest.mark.parametrize("changes", [
    {"quantity": -1}, {"quantity": True}, {"quantity": 0.0001},
    {"quantity": 1e308}, {"expected_gross_return": float("inf")}, {"expected_gross_return": -1},
])
def test_invalid_quantity_or_forecast_refuses_without_committing(account, changes):
    with pytest.raises(ValueError):
        account.execute("bad-entry", "entry", entry(**changes), now=NOW)
    assert account.audit()["version"] == 0


@pytest.mark.parametrize("changes", [
    {"at": NOW + timedelta(milliseconds=1)}, {"at": NOW - timedelta(seconds=30)},
    {"bid": 101}, {"source_mode": "observed"}, {"fee_bps": True}, {"mark": float("nan")},
])
def test_quote_timestamp_mode_and_book_validation(account, changes):
    inputs = entry()
    inputs["quote"].update(changes)
    with pytest.raises(ValueError):
        account.execute("bad", "entry", inputs, now=NOW)
    assert account.audit()["fill_count"] == 0


@pytest.mark.parametrize("field,value,reason", [
    ("confidence", None, "validated_confidence_unavailable"),
    ("confidence", 0.69, "confidence_below_minimum"),
    ("probability", 0.59, "probability_below_minimum"),
    ("action", "AVOID", "action_not_entry"),
])
def test_independent_policy_routes_or_blocks_without_fills(account, field, value, reason):
    inputs = entry()
    inputs["evidence"][field] = value
    result = account.execute("candidate", "entry", inputs, now=NOW)
    assert result["reason"] == reason
    assert result["status"] in ("review", "blocked")
    assert account.execute("candidate", "entry", inputs, now=HOUR)["replayed"]
    assert account.audit()["fill_count"] == 0


def test_positive_probability_does_not_override_cost_or_exposure(account):
    assert account.execute("no-edge", "entry", entry(expected_gross_return=0.0001), now=NOW)["reason"] == "forecast_does_not_cover_costs_and_margin"
    assert account.execute("oversized", "entry", entry(quantity=11), now=NOW)["reason"] == "position_exposure_limit"
    for index, coin in enumerate(("BTC", "ETH", "SOL", "DOGE", "kPEPE")):
        assert account.execute(str(index), "entry", entry(coin=coin, quantity=1), now=NOW)["status"] == "filled"
    assert account.execute("sixth", "entry", entry(coin="xyz:HOOD", quantity=1), now=NOW)["reason"] == "position_count_limit"
    assert len(account.audit()["positions"]) == 5


def test_total_exposure_and_other_position_freshness_are_independent(account):
    account.execute("entry", "entry", entry(), now=NOW)
    account.execute("eth", "entry", entry(coin="ETH"), now=NOW)
    assert account.execute("sol", "entry", entry(coin="SOL", quantity=3), now=NOW)["reason"] == "total_exposure_limit"
    at = NOW + timedelta(minutes=1)
    assert account.execute("stale", "entry", entry(at, coin="SOL", quantity=1), now=at)["reason"] == "stale_position_mark"


def test_pause_loss_limits_and_missing_day_baseline_do_not_veto_exits(account):
    account.execute("entry", "entry", entry(), now=NOW)
    account.execute("pause", "pause", {"enabled": False}, now=NOW)
    assert account.execute("paused", "entry", entry(coin="ETH"), now=NOW)["reason"] == "new_entries_paused"
    assert close(account, at=NOW + timedelta(minutes=1), price=10, reason="risk")["status"] == "filled"
    assert account.audit()["realized_net_pnl"] < -800


def test_overnight_missing_baseline_cannot_be_reset_by_intraday_exit(tmp_path):
    at = NOW.replace(hour=23)
    account = PerpetualPaperAccount.create(tmp_path / "overnight.sqlite", now=at)
    account.execute("entry", "entry", entry(at), now=at)
    exit_at = at + timedelta(minutes=30)
    assert close(account, at=exit_at)["status"] == "filled"
    account.execute("fund", "funding", funding(hour=exit_at.replace(hour=0, minute=0, second=0)), now=exit_at)
    result = account.execute("next", "entry", entry(exit_at, coin="ETH"), now=exit_at)
    assert result["reason"] == "daily_equity_unavailable"
    assert account.audit()["day_start_equity"] is None


def test_exit_identity_and_deadline_are_checked(account):
    account.execute("entry", "entry", entry(), now=NOW)
    assert close(account, at=NOW, operation="early")["reason"] == "exit_not_due"
    assert account.execute("wrong", "exit", {"quote": quote(), "position_id": "old", "reason": "manual"}, now=NOW)["reason"] == "position_changed"
    assert account.audit()["fill_count"] == 1


def test_quantity_in_kpepe_contracts_is_not_multiplied_twice(account):
    account.execute("pepe", "entry", entry(coin="kPEPE", quantity=2), now=NOW)
    result = account.execute("fund", "funding", funding(coin="kPEPE"), now=HOUR)
    assert result["cash_flow"] == pytest.approx(-2 * 101 * 0.0001)


def test_operation_id_changed_inputs_and_backwards_time_refuse(account):
    account.execute("entry", "entry", entry(), now=NOW)
    with pytest.raises(ValueError, match="different inputs"):
        account.execute("entry", "entry", entry(quantity=8), now=NOW)
    with pytest.raises(ValueError, match="backwards"):
        account.execute("past", "pause", {"enabled": False}, now=NOW - timedelta(seconds=1))


def test_partial_persist_failure_rolls_back_journal_and_state(account, monkeypatch):
    before = account.audit()
    original = account._persist
    def fail(*args):
        original(*args)
        raise RuntimeError("Injected write failure")
    monkeypatch.setattr(account, "_persist", fail)
    with pytest.raises(RuntimeError):
        account.execute("entry", "entry", entry(), now=NOW)
    assert account.audit() == before


def test_abrupt_worker_death_before_commit_recovers_without_fill(account):
    process = multiprocessing.get_context("spawn").Process(target=crash_after_write, args=(str(account.path),))
    process.start()
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail("Crash worker did not exit")
    assert process.exitcode == 79
    recovered = PerpetualPaperAccount(account.path)
    assert recovered.audit()["fill_count"] == recovered.audit()["version"] == 0
    assert recovered.execute("entry", "entry", entry(), now=NOW)["status"] == "filled"


def test_expired_execution_rolls_back_state_and_fill_before_commit(account, monkeypatch):
    class Clock(datetime):
        calls = 0
        @classmethod
        def now(cls, tz=None):
            cls.calls += 1
            return NOW if cls.calls == 1 else NOW + timedelta(seconds=6)
    monkeypatch.setattr(perpetual, "datetime", Clock)
    with pytest.raises(ValueError, match="expired"):
        account.execute("entry", "entry", entry())
    assert account.audit()["fill_count"] == account.audit()["version"] == 0


@pytest.mark.parametrize("price,reason", [(50, "daily_loss_limit"), (200, "existing_position_exposure_limit")])
def test_current_portfolio_losses_and_existing_exposure_block_entries(account, price, reason):
    account.execute("entry", "entry", entry(), now=NOW)
    account.execute("mark", "mark", {"quotes": [quote(NOW, price)]}, now=NOW)
    result = account.execute("new", "entry", entry(coin="ETH", quantity=1), now=NOW)
    assert result["reason"] == reason
    assert account.audit()["fill_count"] == 1


def test_observed_oracle_funding_is_explicitly_an_estimate(tmp_path):
    account = PerpetualPaperAccount.create(tmp_path / "observed.sqlite", now=NOW, source_mode="observed")
    account.execute("fund", "funding", funding(source_mode="observed", rate_source_id="funding-history-row", oracle_source_id="observed-oracle"), now=HOUR)
    assert account.audit()["funding_oracle_estimated"] is True


def test_missing_or_expensive_funding_budget_cannot_be_bypassed_by_positive_forecast(account):
    assert account.execute("unknown", "entry", entry(expected_funding_cost_fraction=None), now=NOW)["reason"] == "funding_cost_estimate_unavailable"
    assert account.execute("costly", "entry", entry(expected_funding_cost_fraction=0.05), now=NOW)["reason"] == "forecast_does_not_cover_costs_and_margin"
    with pytest.raises(ValueError):
        account.execute("credit", "entry", entry(expected_funding_cost_fraction=-0.1), now=NOW)
    assert account.audit()["fill_count"] == 0


def test_observed_account_rejects_synthetic_funding_source(tmp_path):
    account = PerpetualPaperAccount.create(tmp_path / "observed.sqlite", now=NOW, source_mode="observed")
    with pytest.raises(ValueError, match="source mode"):
        account.execute("fund", "funding", funding(), now=HOUR)


@pytest.mark.parametrize("workers", ["thread", "process"])
def test_concurrent_delivery_produces_exactly_one_fill_and_no_double_margin(account, workers):
    executor = ThreadPoolExecutor(max_workers=2) if workers == "thread" else ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"))
    with executor as pool:
        results = list(pool.map(process_entry, [str(account.path)] * 2))
    assert sum(not x["replayed"] for x in results) == 1
    assert account.audit()["fill_count"] == 1 and account.audit()["version"] == 1


@pytest.mark.parametrize("table,column", [("account", "state"), ("account", "contract"), ("events", "result")])
def test_audit_refuses_changed_saved_data(account, table, column):
    account.execute("entry", "entry", entry(), now=NOW)
    with sqlite3.connect(str(account.path)) as db:
        db.execute(f"UPDATE {table} SET {column}=?", ('{}',))
    with pytest.raises((ValueError, KeyError)):
        PerpetualPaperAccount(account.path)


def test_creation_never_overwrites_and_policy_cannot_relax_minimums(account, tmp_path):
    with pytest.raises(FileExistsError):
        PerpetualPaperAccount.create(account.path, now=NOW)
    for policy in (EntryPolicy(0.59, 0.7, 5), EntryPolicy(0.6, 0.69, 5), EntryPolicy(0.6, 0.7, 6)):
        with pytest.raises(ValueError):
            PerpetualPaperAccount.create(tmp_path / "bad.sqlite", policy=policy, now=NOW)
    assert not (tmp_path / "bad.sqlite").exists()


def test_cli_demo_is_synthetic_and_reconciles_restart_with_funding(tmp_path, capsys):
    config = tmp_path / "config/hyperliquid.toml"
    config.parent.mkdir()
    config.write_text((ROOT / "config/hyperliquid.toml").read_text(encoding="utf-8"), encoding="utf-8")
    assert main(["perpetual-paper-demo", "--config", str(config)]) == 0
    path = next((tmp_path / "data/hyperliquid/paper").glob("*/account.sqlite"))
    report = perpetual_status(path)
    assert report["source_mode"] == "synthetic"
    assert report["fill_count"] == 2 and not report["performance_provisional"]
    assert "invented prices, forecasts and funding" in capsys.readouterr().out


def test_deep_native_windows_perpetual_path(tmp_path):
    path = tmp_path
    while len(str(path)) < 310:
        path /= "perpetual-paper-deep-directory"
    account = PerpetualPaperAccount.create(path / "account.sqlite", now=NOW)
    assert account.execute("entry", "entry", entry(), now=NOW)["status"] == "filled"
    assert PerpetualPaperAccount(account.path).audit()["fill_count"] == 1
