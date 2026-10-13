"""Persisted synthetic workflow demonstrating restart and ledger reconciliation."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from local_ai_trader.decision.cost_aware import ProbabilityPrediction, ReturnForecast, decide_cost_aware, load_decision_settings
from local_ai_trader.paper.ledger import MarketQuote, PaperLedger
from local_ai_trader.risk.paper import PaperEntryRequest, load_paper_risk_settings
from local_ai_trader.settings import load_settings


def paper_demo(path: Path, *, now: datetime | None = None):
    settings, risk, policy = load_settings(path), load_paper_risk_settings(path), load_decision_settings(path)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Synthetic paper time requires a timezone")
    now = datetime.fromtimestamp(int(now.timestamp()) // risk.candle_seconds * risk.candle_seconds, timezone.utc)
    symbol = risk.allowed_symbols[0]
    signal_at = now - timedelta(seconds=settings.backtest.latency_bars * risk.candle_seconds)
    signal = ProbabilityPrediction(symbol, signal_at, (0.8, 0.1, 0.1))
    forecast = ReturnForecast(0.03, symbol, signal_at, risk.candle_seconds, settings.backtest.latency_bars, risk.holding_bars)
    candidate = decide_cost_aware(signal, False, policy, settings.backtest, risk.candle_seconds, risk.holding_bars, forecast, signal)
    request = PaperEntryRequest(symbol, signal_at, now, 100.0, candidate, risk.candle_seconds, risk.holding_bars, settings.backtest.latency_bars)
    account_path = settings.processed_dir.parent / "paper" / f"synthetic_{now.strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}" / "account.sqlite"
    ledger = PaperLedger.create(account_path, risk, settings.backtest, now=now, source_mode="synthetic")
    entry = ledger.execute_entry("synthetic-entry", request, now=now)
    print("Synthetic persisted paper workflow (invented prices/forecasts; not strategy performance):")
    print(f"Entry: {entry.status}; reason: {entry.reason}")
    if entry.status != "filled":
        print(f"Account saved to {ledger.path}; configured policy/risk refused the synthetic entry")
        return ledger.path
    duplicate = ledger.execute_entry("synthetic-entry", request, now=now)
    restarted = PaperLedger(ledger.path)
    due = now + timedelta(seconds=risk.holding_bars * risk.candle_seconds)
    exits = restarted.exit_due_positions((MarketQuote(symbol, 102.0, due),), now=due)
    report = PaperLedger(ledger.path).audit()
    if len(exits) != 1 or exits[0].status != "filled" or not duplicate.replayed or report["fill_count"] != 2 or report["positions"] or not report["reconciled"]:
        raise RuntimeError("Synthetic paper workflow failed reconciliation")
    print("Duplicate entry: replayed original result; one BUY fill")
    print("Restart: reopened account; holding deadline produced one SELL fill")
    print(f"Cash / final equity: ${report['cash']:.6f}; realized net PnL: ${report['realized_net_pnl']:.6f}")
    print(f"Fills: {report['fill_count']}; closed trades: {len(report['trades'])}; open positions: {len(report['positions'])}")
    print("PASS: persisted cash, inventory and trade PnL reconcile")
    print(f"Synthetic account saved to {ledger.path}")
    return ledger.path


def paper_status(path: Path):
    report = PaperLedger(path).audit()
    print(f"Paper account ({report['source_mode']} inputs; last marked values):")
    print(f"Version: {report['version']}; as of: {report['as_of']}")
    print(f"Cash: ${report['cash']:.6f}; marked equity: ${report['equity']:.6f}")
    print(f"Realized / unrealized net PnL: ${report['realized_net_pnl']:.6f} / ${report['unrealized_net_pnl']:.6f}")
    print(f"Fills: {report['fill_count']}; closed trades: {len(report['trades'])}; open positions: {len(report['positions'])}")
    print("PASS: persisted cash, inventory and trade PnL reconcile")
    return report
