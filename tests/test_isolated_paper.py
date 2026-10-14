"""Economic boundaries for the separately versioned isolated paper model."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import timedelta
from decimal import Decimal

import pytest

from local_ai_trader.hyperliquid.venue import VenueRules, validate_price
from local_ai_trader.paper.perpetual import PerpetualPaperAccount
from local_ai_trader.paper.isolated import decode_rules, encode_rules
from local_ai_trader.risk.entry_policy import SignalEvidence
from test_perpetual_paper import NOW, HOUR, funding


def rules(coin="SOL", size_decimals=2, maximum=20):
    return VenueRules(coin, size_decimals, maximum, maximum, ((Decimal(0), maximum),))


def quote(at=NOW, price=100, coin="SOL", **changes):
    return {"coin": coin, "at": at, "bid": float(Decimal(str(price)) - (Decimal("0.01") if price >= 100 else Decimal("0.001"))),
            "ask": float(Decimal(str(price)) + (Decimal("0.01") if price >= 100 else Decimal("0.001"))), "mark": price,
            "fee_bps": 4.5, "size_decimals": 2, "source_mode": "synthetic", **changes}


def entry(at=NOW, coin="SOL", **changes):
    return {"quote": quote(at, coin=coin), "quantity": 9., "ask_size": 20.,
            "signal_at": at.replace(minute=at.minute // 5 * 5, second=0, microsecond=0) - timedelta(minutes=5),
            "evidence": asdict(SignalEvidence("BUY", .8, "invented", .8, "invented")),
            "expected_gross_return": .03, "expected_funding_cost_fraction": .001, **changes}


@pytest.fixture
def account(tmp_path):
    return PerpetualPaperAccount.create(tmp_path/"isolated.sqlite", now=NOW, coins=("SOL", "ETH"),
                                       venue_rules={c: rules(c) for c in ("SOL", "ETH")})


def open_position(account, **changes):
    result = account.execute("entry", "entry", entry(**changes), now=NOW)
    assert result["status"] == "filled"
    return result


def charge(account, rate=.02):
    return account.execute("funding", "funding", funding(coin="SOL", rate=rate, oracle_price=100.), now=HOUR)


def liquidate(account, price=2.05, bid_size=20, at=HOUR, **changes):
    return account.execute("liquidation", "liquidation", {
        "quote": quote(at, price), "position_id": "entry", "bid_size": bid_size, **changes}, now=at)


def test_isolated_model_and_rule_contract_survive_restart(account):
    assert account.contract()["schema_version"] == 2
    assert account.contract()["isolated_model"] == "isolated_full_size_or_backstop_v1"
    assert decode_rules(encode_rules(rules())) == rules()
    opened = open_position(account)
    validate_price(opened["fill"]["price"], rules())
    report = PerpetualPaperAccount(account.path).audit()
    assert report["margin_mode"] == "isolated"
    assert report["reconciled"] and report["partial_liquidation_supported"] is False
    assert report["available_collateral"] == pytest.approx(report["wallet_balance"] - report["reserved_isolated_collateral"])


@pytest.mark.parametrize("rate", [.02, -.02, 0])
def test_funding_changes_escrow_but_cannot_spend_other_free_cash(account, rate):
    open_position(account)
    before = account.status()
    posted = charge(account, rate)
    after = account.audit()
    flow = -9 * 100 * rate
    assert posted["cash_flow"] == pytest.approx(flow)
    assert after["wallet_balance"] - before["wallet_balance"] == pytest.approx(flow)
    assert after["reserved_isolated_collateral"] - before["reserved_isolated_collateral"] == pytest.approx(flow)
    assert after["available_collateral"] == pytest.approx(before["available_collateral"])
    assert after["reserved_initial_margin"] == before["reserved_initial_margin"]
    root = after["position_risks"]["SOL"]["static_liquidation_mark"]
    assert (root > 0) if rate > 0 else (root == 0)


def test_one_times_margin_covers_mark_even_when_mark_exceeds_ask(account):
    opened = open_position(account, quote=quote(mark=105))
    assert opened["fill"]["reserved_margin"] == pytest.approx(9 * 105)


@pytest.mark.parametrize("changes,match", [
    ({"quantity": .01}, "minimum notional"),
    ({"quantity": 9.001}, "precision"),
    ({"quote": quote(size_decimals=3)}, "precision"),
    ({"quote": quote(ask=100.00123)}, "limits"),
])
def test_native_entry_constraints_reject_invalid_inputs_without_writing(account, changes, match):
    with pytest.raises(ValueError, match=match):
        account.execute("entry", "entry", entry(**changes), now=NOW)
    assert account.audit()["version"] == 0


@pytest.mark.parametrize("ask_size", [None, 0, 8.99])
def test_entry_requires_full_observed_ask_liquidity(account, ask_size):
    inputs = entry()
    if ask_size is None:
        del inputs["ask_size"]
    else:
        inputs["ask_size"] = ask_size
    result = account.execute("entry", "entry", inputs, now=NOW)
    assert result["reason"] == "insufficient_full_size_ask_liquidity"
    assert account.audit()["positions"] == {}


def test_unvalidated_reliability_still_routes_to_review(account):
    result = account.execute("entry", "entry", entry(evidence=asdict(SignalEvidence("BUY", .8, "invented", None, None))), now=NOW)
    assert result["status"] == "review"
    assert not account.audit()["positions"]


def test_liquidation_closes_against_book_and_replays_without_second_sell(account):
    open_position(account)
    charge(account)
    result = liquidate(account)
    assert result["status"] == "filled" and result["reason"] == "isolated_liquidation"
    validate_price(result["fill"]["price"], rules())
    report = PerpetualPaperAccount(account.path).audit()
    assert not report["positions"] and not report["entries_enabled"]
    assert report["fill_count"] == 2 and report["backstop_transfers"] == 0
    assert report["equity"] == pytest.approx(10000 + report["realized_net_pnl"])
    assert liquidate(PerpetualPaperAccount(account.path))["replayed"]
    assert account.status()["fill_count"] == 2


def test_insufficient_liquidity_retains_position_without_fabricated_partial_fill(account):
    open_position(account)
    charge(account)
    result = liquidate(account, bid_size=8)
    report = account.audit()
    assert result["reason"] == "insufficient_full_size_bid_liquidity"
    assert report["positions"]["SOL"]["quantity"] == 9
    assert report["fill_count"] == 1 and not report["trades"]
    assert not report["entries_enabled"]
    assert report["position_risks"]["SOL"]["maintenance_breached"]


def test_backstop_forfeits_only_affected_escrow_and_is_not_a_sell_fill(account):
    opened = open_position(account)
    account.execute("eth-entry", "entry", entry(coin="ETH"), now=NOW)
    before = account.status()
    eth = before["positions"]["ETH"]
    charge(account)
    result = liquidate(account, price=1.9, bid_size=0)
    after = PerpetualPaperAccount(account.path).audit()
    assert result["status"] == "transferred" and "fill" not in result
    assert result["trade"]["exit_fill_price"] is None
    assert result["trade"]["backstop_collateral_transfer"] == pytest.approx(opened["fill"]["reserved_margin"] - 18)
    assert after["available_collateral"] == pytest.approx(before["available_collateral"])
    assert after["positions"]["ETH"] == eth
    assert after["realized_net_pnl"] == pytest.approx(-opened["fill"]["reserved_margin"] - opened["fill"]["fee"])
    assert after["fill_count"] == 2 and after["backstop_transfers"] == 1
    assert liquidate(PerpetualPaperAccount(account.path), price=1.9, bid_size=0)["replayed"]


def test_book_attempt_precedes_backstop_when_fill_preserves_isolated_collateral(account):
    open_position(account)
    charge(account)
    # A mark below backstop can differ from an executable book. Use mark 1.9,
    # bid 2.1: the full exit retains collateral instead of forcing a transfer.
    result = liquidate(account, price=1.9, quote=quote(HOUR, 1.9, bid=2.1, ask=2.101))
    assert result["status"] == "filled" and account.audit()["backstop_transfers"] == 0


def test_bankrupt_book_price_cannot_debit_other_isolated_or_free_collateral(account):
    open_position(account)
    charge(account)
    free = account.status()["available_collateral"]
    result = liquidate(account, price=1.9)
    assert result["status"] == "transferred"
    assert account.audit()["available_collateral"] == pytest.approx(free)


def test_late_funding_changes_backstop_transfer_not_free_cash_or_fixed_loss(account):
    open_position(account)
    charge(account)
    liquidate(account, price=1.9, bid_size=0, at=HOUR + timedelta(hours=1))
    before = account.audit()
    assert before["missing_funding"]
    hour = HOUR + timedelta(hours=1)
    account.execute("late", "funding", funding(coin="SOL", rate=.001, oracle_price=2, hour=hour), now=hour)
    after = PerpetualPaperAccount(account.path).audit()
    assert after["wallet_balance"] == before["wallet_balance"]
    assert after["realized_net_pnl"] == before["realized_net_pnl"]
    assert after["trades"][0]["backstop_collateral_transfer"] == pytest.approx(before["trades"][0]["backstop_collateral_transfer"] - .018)
    assert after["missing_funding"] == []


def test_missing_funding_does_not_prevent_regular_risk_exit(account):
    open_position(account)
    result = account.execute("close", "exit", {"quote": quote(HOUR), "position_id": "entry", "reason": "risk", "bid_size": 20}, now=HOUR)
    assert result["status"] == "filled"
    report = account.audit()
    assert report["performance_provisional"] and not report["positions"]
    account.execute("late", "funding", funding(coin="SOL"), now=HOUR)
    final = account.audit()
    assert final["wallet_balance"] == pytest.approx(10000 + final["realized_net_pnl"])
    assert not final["performance_provisional"]


def test_dust_exit_is_not_trapped_by_opening_notional_minimum(account):
    open_position(account)
    result = account.execute("dust", "exit", {"quote": quote(price=.5), "position_id": "entry", "reason": "risk", "bid_size": 20}, now=NOW)
    assert result["status"] == "filled" and not account.audit()["positions"]


@pytest.mark.parametrize("seconds", [-1, 30])
def test_liquidation_requires_fresh_not_future_quotes(account, seconds):
    open_position(account)
    charge(account)
    version = account.status()["version"]
    with pytest.raises(ValueError, match="future-dated or stale"):
        liquidate(account, quote=quote(HOUR - timedelta(seconds=seconds), 2.05))
    assert account.audit()["version"] == version


def test_liquidation_commit_freshness_failure_rolls_back(account):
    open_position(account)
    charge(account)
    times = iter([HOUR + timedelta(seconds=29), HOUR + timedelta(seconds=31)])
    with pytest.raises(ValueError, match="stale"):
        account.execute("liquidation", "liquidation", {"quote": quote(HOUR, 2.05), "position_id": "entry", "bid_size": 20}, clock=lambda: next(times))
    assert account.audit()["fill_count"] == 1


def test_liquidation_never_closes_a_healthy_or_replaced_position(account):
    open_position(account)
    assert liquidate(account, price=100, at=NOW)["reason"] == "maintenance_not_breached"
    result = account.execute("wrong", "liquidation", {"quote": quote(), "position_id": "changed", "bid_size": 20}, now=NOW)
    assert result["reason"] == "position_changed"
    assert account.audit()["fill_count"] == 1


def test_mark_breach_pauses_entries_and_cannot_be_cleared_while_unsafe(account):
    open_position(account)
    charge(account)
    account.execute("mark", "mark", {"quotes": [quote(HOUR, 2.05)]}, now=HOUR)
    assert not account.status()["entries_enabled"]
    account.execute("enable", "pause", {"enabled": True}, now=HOUR)
    assert not account.audit()["entries_enabled"]


def test_concurrent_liquidations_produce_one_sell(account):
    open_position(account)
    charge(account)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: liquidate(PerpetualPaperAccount(account.path)), range(2)))
    assert sorted(r["replayed"] for r in results) == [False, True]
    assert account.audit()["fill_count"] == 2


def test_legacy_account_cannot_be_reinterpreted_as_isolated(tmp_path):
    account = PerpetualPaperAccount.create(tmp_path/"legacy.sqlite", now=NOW)
    assert account.contract()["schema_version"] == 1
    with pytest.raises(ValueError, match="isolated venue-bound"):
        account.execute("liquidation", "liquidation", {"quote": {**quote(), "coin": "BTC"}, "position_id": "entry", "bid_size": 20}, now=NOW)
    assert account.audit()["liquidation_simulated"] is False


def test_absent_depth_is_unknown_and_cannot_trigger_backstop(account):
    open_position(account)
    charge(account)
    result = account.execute("unknown-depth", "liquidation", {"quote": quote(HOUR, 1.9), "position_id": "entry"}, now=HOUR)
    report = account.audit()
    assert result["reason"] == "bid_liquidity_unavailable"
    assert report["backstop_transfers"] == 0 and report["fill_count"] == 1
    assert report["liquidation_pending"] == ["SOL"] and report["performance_provisional"]


def test_backstop_threshold_is_strict_and_uses_isolated_equity():
    from local_ai_trader.paper.isolated import position_risk
    position = {"quantity": 1, "mark_price": 9, "entry_fill_price": 100, "isolated_collateral": 94}
    risk = position_risk(position, rules(maximum=1))
    assert risk["isolated_equity"] == 3 and risk["maintenance_margin"] == 4.5
    assert risk["maintenance_breached"] and not risk["below_backstop_threshold"]
    position["isolated_collateral"] = 93.99
    assert position_risk(position, rules(maximum=1))["below_backstop_threshold"]


def test_negative_escrow_does_not_charge_free_cash_when_transferred(account):
    open_position(account)
    for i in range(26):
        at = HOUR + timedelta(hours=i)
        account.execute(f"fund:{i}", "funding", funding(coin="SOL", rate=.04, oracle_price=100, hour=at), now=at)
    free = account.status()["available_collateral"]
    assert account.status()["positions"]["SOL"]["isolated_collateral"] < 0
    liquidate(account, price=1.9, bid_size=0, at=HOUR + timedelta(hours=25))
    report = PerpetualPaperAccount(account.path).audit()
    assert report["available_collateral"] == pytest.approx(free)
    assert not report["positions"] and report["reconciled"]


def test_large_entries_are_refused_instead_of_claiming_partial_liquidation_support(tmp_path):
    account = PerpetualPaperAccount.create(tmp_path/"large.sqlite", now=NOW, coins=("SOL",), initial_collateral=2000000,
                                         venue_rules={"SOL": rules()})
    with pytest.raises(ValueError, match="100000"):
        account.execute("large", "entry", entry(quantity=1001, ask_size=2000), now=NOW)
    assert account.audit()["version"] == 0


def test_rounding_below_minimum_tick_does_not_fabricate_zero_price_exit(account):
    open_position(account)
    charge(account)
    result = liquidate(account, quote=quote(HOUR, .0001, bid=.0001, ask=.0002))
    assert result["status"] == "transferred" and "fill" not in result
    assert account.audit()["reconciled"]


def test_healthy_position_cannot_be_labeled_liquidated_through_normal_exit(account):
    open_position(account)
    with pytest.raises(ValueError, match="exit reason"):
        account.execute("fake-liquidation", "exit", {"quote": quote(), "position_id": "entry", "reason": "isolated_liquidation", "bid_size": 20}, now=NOW)
    assert account.audit()["fill_count"] == 1


def test_liquidation_size_cap_cannot_trap_healthy_reducing_exit_after_price_increases(tmp_path):
    account = PerpetualPaperAccount.create(tmp_path/"large.sqlite", now=NOW, coins=("SOL",), initial_collateral=2000000,
                                         venue_rules={"SOL": rules()})
    assert account.execute("entry", "entry", entry(quantity=999, ask_size=2000), now=NOW)["status"] == "filled"
    result = account.execute("exit", "exit", {"quote": quote(price=110), "position_id": "entry", "reason": "risk", "bid_size": 2000}, now=NOW)
    assert result["status"] == "filled" and not account.audit()["positions"]
