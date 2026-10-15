"""Synthetic stress verification using archived, audited native venue rules."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from uuid import uuid4

from local_ai_trader.data.storage import native_path
from local_ai_trader.hyperliquid.feed import load_hyperliquid_settings
from local_ai_trader.hyperliquid.markets import BY_COIN
from local_ai_trader.hyperliquid.venue import load_venue_check, parse_rules, round_price, number
from local_ai_trader.models.train import file_digest
from local_ai_trader.paper.perpetual import PerpetualPaperAccount
from local_ai_trader.risk.entry_policy import SignalEvidence


def isolated_demo(config: Path, venue_check: Path):
    config, venue_check = native_path(config).resolve(), native_path(venue_check).resolve()
    settings = load_hyperliquid_settings(config)
    checked = load_venue_check(venue_check)
    if checked["input_config_sha256"] != file_digest(config):
        raise ValueError("Venue-check config disagrees with current synthetic demonstration")
    if not {"SOL", "ETH"} <= set(settings.coins) or any(row["status"] != "READY" for row in checked["markets"].values()):
        raise ValueError("Synthetic isolated stress verification requires READY venue rules including SOL and ETH")
    raw = json.loads((venue_check/"raw.json").read_text(encoding="utf-8"))["responses"]
    rules = {coin: parse_rules(raw[f"{BY_COIN[coin].dex}_contexts" if BY_COIN[coin].dex else "contexts"][0], coin) for coin in settings.coins}
    # Fixed invented prices below are constructed for SOL 20x and ETH 25x
    # metadata. Refuse changed tiers rather than silently pass a different test.
    if rules["SOL"].max_leverage != 20 or rules["ETH"].max_leverage != 25:
        raise ValueError("Venue margin rules changed; review the synthetic stress scenarios")
    clock = datetime.now(timezone.utc).replace(minute=55, second=10, microsecond=0)
    folder = settings.observations.parent / "paper" / f"synthetic_isolated_{clock.strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    account = PerpetualPaperAccount.create(folder/"account.sqlite", coins=settings.coins, policy=settings.entry_policy,
                                          now=clock, venue_rules=rules, run_binding={
                                              "purpose": "synthetic_isolated_verification", "profitability_trial": False,
                                              "venue_check_sha256": file_digest(venue_check/"venue_check.json")})
    def quote(coin, at, mark):
        delta = number(".01" if mark >= 100 else ".001")
        return {"coin": coin, "at": at, "mark": mark,
                "bid": float(round_price(number(mark)-delta, rules[coin], side="SELL")),
                "ask": float(round_price(number(mark)+delta, rules[coin], side="BUY")),
                "size_decimals": rules[coin].size_decimals, "fee_bps": 4.5, "source_mode": "synthetic"}
    for coin in ("SOL", "ETH"):
        result = account.execute(f"entry:{coin}", "entry", {
            "quote": quote(coin, clock, 100), "quantity": 9., "ask_size": 20.,
            "signal_at": clock.replace(minute=50, second=0),
            "evidence": asdict(SignalEvidence("BUY", .8, "invented-calibration", .8, "invented-reliability")),
            "expected_gross_return": .03, "expected_funding_cost_fraction": .02}, now=clock)
        if result["status"] != "filled":
            raise RuntimeError(f"Synthetic isolated entry refused: {result['reason']}")
    hour = clock.replace(minute=0, second=0) + timedelta(hours=1)
    free_before = account.status()["available_collateral"]
    for coin in ("SOL", "ETH"):
        account.execute(f"funding:{coin}", "funding", {
            "coin": coin, "hour": hour, "rate": .02, "oracle_price": 100., "oracle_at": hour,
            "source_mode": "synthetic", "rate_source_id": "invented-extreme-rate", "oracle_source_id": "invented-oracle"}, now=hour)
    funded = account.audit()
    if abs(funded["available_collateral"] - free_before) > 1e-8:
        raise RuntimeError("Synthetic funding consumed unrelated free collateral")
    unaffected = deepcopy(funded["positions"]["ETH"])
    book_inputs = {"quote": quote("SOL", hour, 2.05), "position_id": "entry:SOL", "bid_size": 20.}
    closed = PerpetualPaperAccount(account.path).execute("liquidation:SOL", "liquidation", book_inputs, now=hour)
    if closed["status"] != "filled" or account.audit()["positions"]["ETH"] != unaffected:
        raise RuntimeError("Synthetic book liquidation failed or affected another position")
    before_transfer = account.status()["available_collateral"]
    backstop_inputs = {"quote": quote("ETH", hour, 1.9), "position_id": "entry:ETH", "bid_size": 0.}
    transferred = PerpetualPaperAccount(account.path).execute("liquidation:ETH", "liquidation", backstop_inputs, now=hour)
    replay = PerpetualPaperAccount(account.path).execute("liquidation:ETH", "liquidation", backstop_inputs, now=hour)
    report = account.audit()
    if (transferred["status"] != "transferred" or not replay["replayed"] or report["positions"]
            or report["fill_count"] != 3 or report["backstop_transfers"] != 1 or report["missing_funding"]
            or abs(report["available_collateral"] - before_transfer) > 1e-8 or not report["reconciled"]):
        raise RuntimeError("Synthetic isolated liquidation accounting failed")
    print("Synthetic isolated-margin stress checks (invented prices, funding and reliability; not strategy performance):")
    print("Venue rules: audited archived metadata; native ticks, lot sizes and opening minimum enforced")
    print("Entries: two 1x isolated positions; funding changes their escrow, not unrelated free cash")
    print("Book liquidation: one full-size SELL; the other position is unchanged")
    print("Backstop: affected position and escrow transferred; no fictitious SELL or clearance fee")
    print("Restart/duplicate: original transfer replayed; no second collateral loss")
    print(f"Fills: {report['fill_count']}; backstop transfers: {report['backstop_transfers']}; positions: 0; missing funding: 0")
    print("PASS: isolated collateral, fees, funding and realized losses reconcile after restart")
    print("Bounded simulator: no partial liquidations; opening notional capped at 100000 USDC; live guardian and reliability qualification still pending")
    print("Synthetic checks only: existing monitor unchanged; profitability month has not started; live execution disabled")
    print(f"Synthetic isolated account saved to {account.path}")
    return account.path
