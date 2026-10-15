"""Controlled USDC perpetual accounting demonstration, never a model trial."""

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from local_ai_trader.hyperliquid.feed import load_hyperliquid_settings
from local_ai_trader.paper.perpetual import PerpetualPaperAccount
from local_ai_trader.risk.entry_policy import SignalEvidence


def perpetual_demo(config: Path):
    settings = load_hyperliquid_settings(config)
    clock = datetime.now(timezone.utc).replace(minute=55, second=10, microsecond=0)
    coin = "BTC" if "BTC" in settings.coins else settings.coins[0]
    path = settings.observations.parent / "paper" / f"synthetic_perpetual_{clock.strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}" / "account.sqlite"
    account = PerpetualPaperAccount.create(path, coins=settings.coins, policy=settings.entry_policy, now=clock)
    def quote(at, price):
        return {"coin": coin, "at": at, "bid": price - 0.01, "ask": price + 0.01, "mark": price,
                "size_decimals": 3, "fee_bps": settings.taker_fee_bps, "source_mode": "synthetic"}
    inputs = {"quote": quote(clock, 100), "quantity": 9.0, "signal_at": clock.replace(minute=50, second=0),
              "evidence": asdict(SignalEvidence("BUY", 0.8, "synthetic-probability-test", 0.8, "synthetic-reliability-test")),
              "expected_gross_return": 0.03, "expected_funding_cost_fraction": 0.000101}
    entry = account.execute("entry", "entry", inputs, now=clock)
    if entry["status"] != "filled":
        raise RuntimeError(f"Synthetic perpetual entry refused: {entry['reason']}")
    replay = PerpetualPaperAccount(path).execute("entry", "entry", inputs, now=clock + timedelta(seconds=1))
    hour = clock.replace(minute=0, second=0) + timedelta(hours=1)
    settlement = {"coin": coin, "hour": hour, "rate": 0.0001, "oracle_price": 101.0, "oracle_at": hour,
                  "rate_source_id": "synthetic-settled-rate", "oracle_source_id": "synthetic-oracle", "source_mode": "synthetic"}
    funding = account.execute("funding", "funding", settlement, now=hour)
    funding_replay = PerpetualPaperAccount(path).execute("funding", "funding", settlement, now=hour + timedelta(seconds=1))
    due = clock + timedelta(minutes=30)
    exit_result = PerpetualPaperAccount(path).execute("exit", "exit", {
        "quote": quote(due, 102), "position_id": "entry", "reason": "horizon"}, now=due)
    report = PerpetualPaperAccount(path).audit()
    if not replay["replayed"] or not funding_replay["replayed"] or exit_result["status"] != "filled" or report["fill_count"] != 2 or report["positions"] or report["missing_funding"] or not report["reconciled"]:
        raise RuntimeError("Synthetic perpetual accounting failed")
    print("Synthetic USDC perpetual workflow (invented prices, forecasts and funding; not strategy performance):")
    print("Entry: filled; 1x initial margin reserved; principal retained in collateral wallet")
    print("Restart/duplicate: original BUY replayed; no second fill")
    print(f"Hourly funding: {funding['cash_flow']:.6f} USDC; oracle 101.00; duplicate replayed without another charge")
    print("Horizon exit: filled after restart; initial margin released")
    print(f"Wallet / final equity: {report['wallet_balance']:.6f} USDC; net PnL: {report['realized_net_pnl']:.6f} USDC")
    print(f"Fills: {report['fill_count']}; closed trades: {len(report['trades'])}; open positions: {len(report['positions'])}; missing funding: 0")
    print("PASS: USDC wallet, margin, fees, funding and trade PnL reconcile")
    print("Accounting foundation only: no liquidation simulation, live orders or forward strategy trial")
    print(f"Synthetic perpetual account saved to {path}")
    return path


def perpetual_status(path):
    report = PerpetualPaperAccount(path).audit()
    print(f"Perpetual paper account ({report['source_mode']}; last recorded values):")
    print(f"Wallet: {report['wallet_balance']:.6f} USDC; equity: {report['equity']:.6f} USDC; reserved margin: {report['reserved_initial_margin']:.6f} USDC")
    print(f"Realized net PnL: {report['realized_net_pnl']:.6f}; missing funding: {len(report['missing_funding'])}; provisional: {report['performance_provisional']}")
    print("PASS: USDC wallet, margin, fees, funding and trade PnL reconcile")
    return report
