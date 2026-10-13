from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
import pytest

from local_ai_trader.backtest.engine import simulate
from local_ai_trader.cli import main
from local_ai_trader.decision.cost_aware import DecisionSettings, ProbabilityPrediction, ReturnForecast, decide_cost_aware
from local_ai_trader.risk.demo import paper_risk_demo
from local_ai_trader.risk.paper import (
    MarkedPosition, PendingEntry, PaperAccountSnapshot, PaperEntryRequest,
    PaperRiskSettings, authorize_paper_entry, load_paper_risk_settings,
    paper_authorization_is_current, validate_paper_risk_settings,
)
from local_ai_trader.settings import BacktestSettings, load_settings, settings_from_config

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)
COSTS = BacktestSettings()


@pytest.fixture
def request_packet():
    signal_at = NOW - timedelta(minutes=5)
    signal = ProbabilityPrediction("BTC-USD", signal_at, (0.8, 0.1, 0.1))
    forecast = ReturnForecast(0.03, signal.symbol, signal_at, 300, 1, 6)
    candidate = decide_cost_aware(signal, False, DecisionSettings(), COSTS, 300, 6, forecast, signal)
    assert candidate.action == "BUY"
    return PaperEntryRequest(signal.symbol, signal_at, NOW, 100.0, candidate)


@pytest.fixture
def account():
    return PaperAccountSnapshot(7, NOW, 10000.0, 10000.0, 10000.0, NOW.replace(hour=0, minute=0))


def gate(request, account, *, config=None, costs=None, now=NOW):
    return authorize_paper_entry(request, account, config or PaperRiskSettings(), costs or COSTS, now)


def blocked(result, reason):
    assert result.allowed is False
    assert result.reason == reason
    assert result.authorization is None


def test_entry_cash_fees_impact_and_marked_equity_reconcile(request_packet, account):
    before = asdict(account)
    result = gate(request_packet, account)
    assert result.allowed
    approval = result.authorization
    expected_qty = 1000 / (100 * 1.001 * 1.006)
    assert approval.quantity == pytest.approx(expected_qty)
    assert approval.fill_price == pytest.approx(100.1)
    assert approval.cash_outlay == pytest.approx(1000)
    assert approval.fee == pytest.approx(expected_qty * 100.1 * 0.006)
    assert approval.price_impact_cost == pytest.approx(expected_qty * 0.1)
    assert approval.projected_equity == pytest.approx(10000 - approval.fee - approval.price_impact_cost)
    assert approval.projected_equity == pytest.approx(account.cash - approval.cash_outlay + expected_qty * 100)
    assert approval.projected_total_exposure_fraction == pytest.approx(expected_qty * 100 / approval.projected_equity)
    assert approval.projected_position_exposure_fraction == approval.projected_total_exposure_fraction
    assert approval.account_version == 7
    assert approval.mode == "paper"
    assert asdict(account) == before


def test_full_portfolio_equity_is_used_instead_of_only_cash(request_packet, account):
    account = replace(account, cash=9100, positions=(MarkedPosition("ETH-USD", 9, 100, NOW),))
    approval = gate(request_packet, account).authorization
    assert approval is not None
    assert approval.cash_outlay == pytest.approx(1000)
    assert approval.projected_equity == pytest.approx(9100 - 1000 + 900 + approval.quantity * 100)
    assert approval.projected_total_exposure_fraction == pytest.approx((900 + approval.quantity * 100) / approval.projected_equity)


def test_zero_costs_allow_exact_exposure_boundary(request_packet, account):
    free = replace(COSTS, fee_bps=0, slippage_bps=0, spread_bps=0)
    approval = gate(request_packet, account, costs=free).authorization
    assert approval.quantity == 10
    assert approval.fee == approval.price_impact_cost == 0
    assert approval.projected_equity == 10000
    assert approval.projected_position_exposure_fraction == 0.1


@pytest.mark.parametrize("fee,spread,slippage", [(60, 10, 5), (0, 0, 0), (150, 20, 15)])
def test_paper_sizing_matches_existing_delayed_fill_accounting(request_packet, account, fee, spread, slippage):
    costs = replace(COSTS, fee_bps=fee, spread_bps=spread, slippage_bps=slippage)
    times = pd.date_range("2026-01-01T00:00:00Z", periods=14, freq="5min")
    available = pd.date_range("2026-01-01T00:05:00Z", periods=14, freq="5min")
    frame = pd.DataFrame({"timestamp": times, "available_at": available, "open": 100.0, "close": 100.0, "symbol": "BTC-USD", "exchange": "coinbase"})
    probabilities = np.tile([0.0, 0.0, 1.0], (len(times), 1))
    probabilities[0] = [1, 0, 0]
    fill = simulate(frame, probabilities, costs, 300, 6).fills.iloc[0]
    approval = gate(request_packet, account, costs=costs).authorization
    assert approval is not None
    assert fill.filled_at == approval.issued_at
    assert approval.quantity == pytest.approx(fill.quantity)
    assert approval.fill_price == pytest.approx(fill.fill_price)
    assert approval.fee == pytest.approx(fill.fee)
    assert approval.cash_outlay == pytest.approx(fill.notional + fill.fee)


@pytest.mark.parametrize("action", ["AVOID", "HOLD", "SELL"])
def test_only_buy_candidates_reach_entry_authorization(request_packet, account, action):
    decision = replace(request_packet.decision, action=action, risk_check_required=False)
    blocked(gate(replace(request_packet, decision=decision), account), "decision_not_buy")


@pytest.mark.parametrize("net_return,risk_flag", [(None, True), (0, True), (-0.01, True), (float("nan"), True), (0.03, False)])
def test_malformed_buy_cannot_authorize(request_packet, account, net_return, risk_flag):
    packet = replace(request_packet, decision=replace(request_packet.decision, expected_net_return=net_return, risk_check_required=risk_flag))
    blocked(gate(packet, account), "invalid_buy_candidate")


@pytest.mark.parametrize("field,value,reason", [
    ("symbol", "DOGE-USD", "market_not_allowed"),
    ("reference_price", 0, "invalid_reference_price"),
    ("reference_price", float("inf"), "invalid_reference_price"),
    ("reference_price", True, "invalid_reference_price"),
    ("candle_seconds", 60, "execution_contract_mismatch"),
    ("holding_bars", 12, "execution_contract_mismatch"),
    ("latency_bars", True, "execution_contract_mismatch"),
    ("latency_bars", 2, "execution_contract_mismatch"),
    ("signal_at", NOW - timedelta(minutes=10), "stale_signal"),
    ("signal_at", NOW + timedelta(minutes=5), "future_signal"),
    ("signal_at", NOW - timedelta(minutes=5, seconds=1), "signal_not_interval_aligned"),
    ("signal_at", NOW.replace(tzinfo=None), "invalid_account_or_entry_data"),
    ("signal_at", NOW, "execution_latency_not_elapsed"),
    ("quote_at", NOW + timedelta(microseconds=1), "future_quote"),
    ("quote_at", NOW - timedelta(seconds=30), "stale_quote"),
    ("quote_at", NOW - timedelta(microseconds=1), "quote_precedes_execution"),
])
def test_market_timing_and_invalid_quote_refusal(request_packet, account, field, value, reason):
    blocked(gate(replace(request_packet, **{field: value}), account), reason)


def test_old_historical_buy_is_blocked_despite_fresh_quote_and_account(request_packet, account):
    now = NOW + timedelta(days=100)
    snapshot = replace(account, as_of=now, day_start_at=now.replace(hour=0, minute=0))
    blocked(gate(replace(request_packet, quote_at=now), snapshot, now=now), "stale_signal")


def test_timezone_offsets_represent_same_utc_instants(request_packet, account):
    zone = timezone(timedelta(hours=-8))
    packet = replace(request_packet, signal_at=request_packet.signal_at.astimezone(zone), quote_at=NOW.astimezone(zone))
    snapshot = replace(account, as_of=NOW.astimezone(zone), day_start_at=account.day_start_at.astimezone(zone))
    assert gate(packet, snapshot, now=NOW.astimezone(zone)) == gate(request_packet, account)


@pytest.mark.parametrize("field,value,reason", [
    ("mode", "live", "invalid_paper_account"),
    ("currency", "USDT", "invalid_paper_account"),
    ("version", True, "invalid_paper_account"),
    ("version", -1, "invalid_paper_account"),
    ("cash", float("nan"), "invalid_paper_account"),
    ("cash", -1, "invalid_paper_account"),
    ("peak_equity", 0, "invalid_paper_account"),
    ("as_of", NOW - timedelta(seconds=30), "stale_account"),
    ("as_of", NOW + timedelta(seconds=1), "future_account"),
    ("day_start_at", NOW - timedelta(days=1), "utc_day_baseline_unavailable"),
    ("day_start_equity", 10001, "inconsistent_day_baseline"),
    ("peak_equity", 9999, "inconsistent_day_baseline"),
    ("positions", [], "invalid_portfolio"),
    ("pending_entries", [], "invalid_portfolio"),
])
def test_account_validation_fails_closed(request_packet, account, field, value, reason):
    blocked(gate(request_packet, replace(account, **{field: value})), reason)


@pytest.mark.parametrize("entry,reason", [
    (MarkedPosition("ETH-USD", 1, 100, NOW - timedelta(seconds=30)), "stale_position_mark"),
    (MarkedPosition("ETH-USD", 1, 100, NOW + timedelta(seconds=1)), "mark_after_account_snapshot"),
    (MarkedPosition("ETH-USD", 0, 100, NOW), "invalid_portfolio"),
    (MarkedPosition("ETH-USD", 1, float("inf"), NOW), "invalid_portfolio"),
    (MarkedPosition("OTHER-USD", 1, 100, NOW), "invalid_portfolio"),
    (PendingEntry("ETH-USD", 100, 99, NOW - timedelta(seconds=30)), "stale_pending_entry_mark"),
    (PendingEntry("ETH-USD", -1, 100, NOW), "invalid_portfolio"),
    (PendingEntry("ETH-USD", 10100, 10100, NOW), "inconsistent_account_equity_or_reservations"),
])
def test_existing_holdings_and_pending_entries_require_valid_fresh_marks(request_packet, account, entry, reason):
    if isinstance(entry, MarkedPosition):
        snapshot = replace(account, cash=9900, positions=(entry,))
    else:
        snapshot = replace(account, pending_entries=(entry,))
    blocked(gate(request_packet, snapshot), reason)


def test_duplicate_portfolio_symbols_are_not_double_counted(request_packet, account):
    position = MarkedPosition("ETH-USD", 1, 100, NOW)
    pending = PendingEntry("ETH-USD", 100, 99, NOW)
    blocked(gate(request_packet, replace(account, cash=9900, positions=(position,), pending_entries=(pending,))), "invalid_portfolio")


def test_existing_and_pending_same_market_block_new_entry(request_packet, account):
    holding = replace(account, cash=9900, positions=(MarkedPosition("BTC-USD", 1, 100, NOW),))
    blocked(gate(request_packet, holding), "position_already_open")
    pending = replace(account, pending_entries=(PendingEntry("BTC-USD", 100, 99, NOW),))
    blocked(gate(request_packet, pending), "entry_already_pending")


def test_reservations_reduce_spendable_cash_and_count_toward_position_limit(request_packet, account):
    pending = replace(account, pending_entries=(PendingEntry("ETH-USD", 9500, 9490, NOW),))
    blocked(gate(request_packet, pending), "insufficient_unreserved_cash")
    smaller = replace(account, pending_entries=(PendingEntry("ETH-USD", 100, 99, NOW),))
    blocked(gate(request_packet, smaller, config=replace(PaperRiskSettings(), max_open_positions=1)), "position_count_limit")


@pytest.mark.parametrize("marked,expected_loss", [(900, 100), (1100, 0)])
def test_pending_costs_lower_projected_equity_but_potential_gains_never_inflate_it(request_packet, account, marked, expected_loss):
    pending = replace(account, pending_entries=(PendingEntry("ETH-USD", 1000, marked, NOW),))
    config = replace(PaperRiskSettings(), max_position_exposure_fraction=0.2, max_total_exposure_fraction=0.3)
    approval = gate(request_packet, pending, config=config).authorization
    assert approval is not None
    assert approval.cash_outlay == pytest.approx(1000)
    assert approval.projected_equity == pytest.approx(10000 - expected_loss - approval.fee - approval.price_impact_cost)
    assert approval.projected_total_exposure_fraction == pytest.approx((marked + approval.quantity * 100) / approval.projected_equity)


@pytest.mark.parametrize("cash,peak,day_start,reason", [
    (9000, 10000, 9000, "drawdown_limit"),
    (8999, 10000, 8999, "drawdown_limit"),
    (9800, 10000, 10000, "daily_loss_limit"),
    (9799, 10000, 10000, "daily_loss_limit"),
    (9801, 10000, 10000, "projected_daily_loss_limit"),
    (9001, 10000, 9001, "projected_drawdown_limit"),
])
def test_current_and_post_cost_loss_boundaries(request_packet, account, cash, peak, day_start, reason):
    blocked(gate(request_packet, replace(account, cash=cash, peak_equity=peak, day_start_equity=day_start)), reason)


def test_pending_entry_losses_are_included_in_daily_limit(request_packet, account):
    snapshot = replace(account, pending_entries=(PendingEntry("ETH-USD", 1100, 900, NOW),))
    blocked(gate(request_packet, snapshot), "projected_daily_loss_limit")


def test_total_exposure_checks_all_assets_after_costs(request_packet, account):
    snapshot = replace(account, cash=8500, positions=(MarkedPosition("ETH-USD", 15, 100, NOW),))
    config = replace(PaperRiskSettings(), max_position_exposure_fraction=0.2)
    blocked(gate(request_packet, snapshot, config=config), "total_exposure_limit")


def test_existing_position_exceeding_individual_cap_blocks_additional_entries(request_packet, account):
    snapshot = replace(account, cash=8800, positions=(MarkedPosition("ETH-USD", 12, 100, NOW),))
    blocked(gate(request_packet, snapshot), "existing_position_exposure_limit")


def test_oversized_new_position_and_small_accounts_are_refused(request_packet, account):
    blocked(gate(request_packet, account, config=replace(PaperRiskSettings(), allocation_fraction=0.11)), "position_exposure_limit")
    snapshot = replace(account, cash=99, peak_equity=99, day_start_equity=99)
    blocked(gate(request_packet, snapshot), "below_minimum_entry_cash")


def test_new_entry_pause_is_independent_of_decision(request_packet, account):
    blocked(gate(request_packet, account, config=replace(PaperRiskSettings(), new_entries_enabled=False)), "new_entries_paused")


@pytest.mark.parametrize("limiting_source", ["ttl", "quote", "account", "mark", "signal", "utc_day"])
def test_expiry_never_outlives_input_freshness_or_utc_day(request_packet, account, limiting_source):
    config = PaperRiskSettings()
    packet, snapshot, now = request_packet, account, NOW
    seconds = 5
    if limiting_source == "quote":
        now = NOW + timedelta(seconds=29)
        snapshot = replace(account, as_of=now)
        seconds = 1
    elif limiting_source == "account":
        now = NOW + timedelta(seconds=29)
        packet = replace(packet, quote_at=now)
        seconds = 1
    elif limiting_source == "mark":
        snapshot = replace(account, cash=9900, positions=(MarkedPosition("ETH-USD", 1, 100, NOW - timedelta(seconds=29)),))
        seconds = 1
    elif limiting_source == "signal":
        now = NOW + timedelta(seconds=299)
        packet, snapshot = replace(packet, quote_at=now), replace(account, as_of=now)
        seconds = 1
    elif limiting_source == "utc_day":
        now = NOW.replace(hour=23, minute=59, second=59)
        config = replace(config, max_signal_age_bars=3)
        packet = replace(packet, signal_at=NOW.replace(hour=23, minute=50), quote_at=now)
        snapshot = replace(account, as_of=now)
        seconds = 1
    approval = gate(packet, snapshot, config=config, now=now).authorization
    assert approval is not None
    assert approval.expires_at == now + timedelta(seconds=seconds)


def test_midnight_requires_actual_current_day_baseline(request_packet, account):
    now = NOW.replace(hour=0, minute=0) + timedelta(days=1)
    packet = replace(request_packet, signal_at=now - timedelta(minutes=5), quote_at=now)
    old = replace(account, as_of=now)
    blocked(gate(packet, old, now=now), "utc_day_baseline_unavailable")
    inconsistent = replace(account, as_of=now - timedelta(seconds=1), day_start_at=now)
    blocked(gate(packet, inconsistent, now=now), "inconsistent_day_baseline")
    assert gate(packet, replace(account, as_of=now, day_start_at=now), now=now).allowed


def test_authorization_revalidation_rejects_expiry_account_changes_and_tampered_sizing(request_packet, account):
    config = PaperRiskSettings()
    approval = gate(request_packet, account).authorization
    assert paper_authorization_is_current(approval, request_packet, account, config, COSTS, NOW)
    assert paper_authorization_is_current(approval, request_packet, account, config, COSTS, NOW + timedelta(seconds=4))
    assert not paper_authorization_is_current(approval, request_packet, account, config, COSTS, approval.expires_at)
    assert not paper_authorization_is_current(approval, request_packet, replace(account, version=8), config, COSTS, NOW)
    assert not paper_authorization_is_current(approval, request_packet, replace(account, cash=9999), config, COSTS, NOW)
    assert not paper_authorization_is_current(replace(approval, quantity=approval.quantity * 2), request_packet, account, config, COSTS, NOW)
    assert not paper_authorization_is_current(replace(approval, expires_at=NOW + timedelta(seconds=60)), request_packet, account, config, COSTS, NOW)
    assert not paper_authorization_is_current(approval, replace(request_packet, reference_price=101), account, config, COSTS, NOW)
    assert not paper_authorization_is_current(approval, request_packet, account, replace(config, allocation_fraction=0.09), COSTS, NOW)
    assert not paper_authorization_is_current(approval, request_packet, account, config, replace(COSTS, fee_bps=61), NOW)
    assert not paper_authorization_is_current(replace(approval, issued_at=NOW + timedelta(seconds=1)), request_packet, account, config, COSTS, NOW)
    updated = replace(account, as_of=NOW + timedelta(seconds=1))
    assert not paper_authorization_is_current(approval, request_packet, updated, config, COSTS, NOW + timedelta(seconds=1))


@pytest.mark.parametrize("field,value", [
    ("allocation_fraction", float("nan")), ("allocation_fraction", True),
    ("max_total_exposure_fraction", 1.1), ("max_total_exposure_fraction", 0.05),
    ("max_daily_loss_fraction", 0), ("max_open_positions", True),
    ("authorization_ttl_seconds", 0), ("max_quote_age_seconds", -1),
    ("new_entries_enabled", 1), ("minimum_entry_cash", float("inf")),
    ("allowed_symbols", ("BTC-USD", "BTC-USD")),
    ("allowed_symbols", ("BTC-EUR",)),
])
def test_invalid_risk_configuration_is_refused(field, value):
    with pytest.raises(ValueError):
        validate_paper_risk_settings(replace(PaperRiskSettings(), **{field: value}))


def test_separate_risk_table_preserves_frozen_settings_and_older_configs(tmp_path):
    path = ROOT / "config/settings.toml"
    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    assert load_paper_risk_settings(path) == PaperRiskSettings()
    with_table = settings_from_config(parsed, ROOT)
    parsed.pop("paper_risk")
    assert with_table == settings_from_config(parsed, ROOT)
    # Use a complete old config with the separate table omitted.
    text = path.read_text(encoding="utf-8")
    start, end = text.index("[paper_risk]"), text.index("[download]")
    old = tmp_path / "old.toml"
    old.write_text(text[:start] + text[end:], encoding="utf-8")
    assert load_paper_risk_settings(old) == PaperRiskSettings()
    assert load_settings(old).symbols == PaperRiskSettings().allowed_symbols


@pytest.mark.parametrize("extra", ["unknown_option = 1", "allowed_symbols = ['ETH-USD']", "max_signal_age_bars = 1"])
def test_risk_loader_rejects_unknown_market_overrides_and_impossible_freshness(tmp_path, extra):
    text = (ROOT / "config/settings.toml").read_text(encoding="utf-8")
    start, end = text.index("[paper_risk]"), text.index("[download]")
    path = tmp_path / "config.toml"
    path.write_text(text[:start] + "[paper_risk]\n" + extra + "\n\n" + text[end:], encoding="utf-8")
    with pytest.raises(ValueError):
        load_paper_risk_settings(path)


@pytest.mark.parametrize("now", [NOW, NOW + timedelta(seconds=123, microseconds=456)])
def test_synthetic_demo_checks_limits_without_creating_a_ledger_or_orders(capsys, now):
    results = dict(paper_risk_demo(ROOT / "config/settings.toml", now))
    assert results["fresh synthetic candidate"].allowed
    assert sum(result.allowed for result in results.values()) == 1
    assert results["cash already reserved"].reason == "insufficient_unreserved_cash"
    assert results["daily loss boundary"].reason == "daily_loss_limit"
    assert results["stale holding mark"].reason == "stale_position_mark"
    assert results["stale signal"].reason == "stale_signal"
    assert results["fresh synthetic candidate"].authorization.expires_at > now
    assert "invented test inputs; no orders or saved account" in capsys.readouterr().out


def test_paper_risk_demo_cli(capsys):
    assert main(["paper-risk-demo", "--config", str(ROOT / "config/settings.toml")]) == 0
    output = capsys.readouterr().out
    assert "Synthetic paper-risk checks" in output
    assert "paper_entry_approved" in output
