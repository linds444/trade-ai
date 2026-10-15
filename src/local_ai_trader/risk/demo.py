"""Synthetic checks of the independent paper entry gate; never place orders."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from local_ai_trader.decision.cost_aware import ProbabilityPrediction, ReturnForecast, decide_cost_aware, load_decision_settings
from local_ai_trader.risk.paper import MarkedPosition, PendingEntry, PaperAccountSnapshot, PaperEntryRequest, authorize_paper_entry, load_paper_risk_settings
from local_ai_trader.settings import load_settings


def paper_risk_demo(path: Path, now: datetime | None = None):
    """Exercise fresh, stale and limited synthetic accounts with configured costs.

    The positive return and probabilities are test inputs, not model outputs.
    No account, approval or prediction is saved and no paper fill is committed.
    """
    settings, risk, policy = load_settings(path), load_paper_risk_settings(path), load_decision_settings(path)
    now = datetime.now(timezone.utc) if now is None else now
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Demo time must include a timezone")
    now = now.astimezone(timezone.utc)
    boundary = datetime.fromtimestamp(int(now.timestamp()) // risk.candle_seconds * risk.candle_seconds, timezone.utc)
    signal_at = boundary - timedelta(seconds=settings.backtest.latency_bars * risk.candle_seconds)
    signal = ProbabilityPrediction(risk.allowed_symbols[0], signal_at, (0.8, 0.1, 0.1))
    forecast = ReturnForecast(0.03, signal.symbol, signal_at, risk.candle_seconds, settings.backtest.latency_bars, risk.holding_bars)
    candidate = decide_cost_aware(signal, False, policy, settings.backtest, risk.candle_seconds, risk.holding_bars, forecast, signal)
    avoid = decide_cost_aware(signal, False, policy, settings.backtest, risk.candle_seconds, risk.holding_bars, None, signal)
    request = PaperEntryRequest(signal.symbol, signal_at, now, 100.0, candidate, risk.candle_seconds, risk.holding_bars, settings.backtest.latency_bars)
    account = PaperAccountSnapshot(0, now, 10000.0, 10000.0, 10000.0, now.replace(hour=0, minute=0, second=0, microsecond=0))
    cases = [
        ("fresh synthetic candidate", request, account, risk),
        ("AVOID decision", replace(request, decision=avoid), account, risk),
        ("stale signal", replace(request, signal_at=boundary - timedelta(seconds=risk.max_signal_age_bars * risk.candle_seconds)), account, risk),
        ("stale quote", replace(request, quote_at=now - timedelta(seconds=risk.max_quote_age_seconds)), account, risk),
        ("future quote", replace(request, quote_at=now + timedelta(seconds=1)), account, risk),
        ("stale account", request, replace(account, as_of=now - timedelta(seconds=risk.max_account_age_seconds)), risk),
        ("entries paused", request, account, replace(risk, new_entries_enabled=False)),
        ("drawdown boundary", request, replace(account, cash=10000 * (1 - risk.max_drawdown_fraction), day_start_equity=10000 * (1 - risk.max_drawdown_fraction)), risk),
        ("daily loss boundary", request, replace(account, cash=10000 * (1 - risk.max_daily_loss_fraction)), risk),
        ("duplicate pending entry", request, replace(account, pending_entries=(PendingEntry(signal.symbol, 1000, 990, now),)), risk),
    ]
    if len(risk.allowed_symbols) > 1:
        other = next(symbol for symbol in risk.allowed_symbols if symbol != signal.symbol)
        cases.extend([
            ("cash already reserved", request, replace(account, pending_entries=(PendingEntry(other, 9500, 9490, now),)), risk),
            ("stale holding mark", request, replace(account, cash=9900, positions=(MarkedPosition(other, 1, 100, now - timedelta(seconds=risk.max_account_age_seconds)),)), risk),
            ("position count limit", request, replace(account, cash=9900, positions=(MarkedPosition(other, 1, 100, now),)), replace(risk, max_open_positions=1)),
        ])
    results = [(name, authorize_paper_entry(entry, snapshot, limits, settings.backtest, now)) for name, entry, snapshot, limits in cases]
    print("Synthetic paper-risk checks (invented test inputs; no orders or saved account):")
    for name, result in results:
        print(f"{name:27} {'ALLOW' if result.allowed else 'BLOCK':5} {result.reason}")
    approval = results[0][1].authorization
    if approval is not None:
        print(f"Synthetic approval: cash outlay ${approval.cash_outlay:.2f}; expires {approval.expires_at.isoformat()}; account version {approval.account_version}")
    return results
