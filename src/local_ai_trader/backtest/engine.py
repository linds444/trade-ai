"""One-position spot simulator with delayed open fills and explicit cash accounting."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from local_ai_trader.backtest.metrics import performance_metrics
from local_ai_trader.data.validate import validate_candle_timeline
from local_ai_trader.decision.policy import decide
from local_ai_trader.risk.risk_gate import authorize_entry
from local_ai_trader.settings import BacktestSettings, load_backtest_settings, positive_integer

FILL_COLUMNS = ["signal_at", "filled_at", "side", "reason", "quantity", "reference_price", "fill_price", "notional", "fee", "price_impact_cost", "cash_after"]
TRADE_COLUMNS = ["entry_at", "exit_at", "exit_reason", "bars_held", "quantity", "entry_reference_price", "exit_reference_price", "gross_pnl", "fees", "price_impact_cost", "net_pnl"]


@dataclass(frozen=True)
class Simulation:
    equity: pd.DataFrame
    fills: pd.DataFrame
    trades: pd.DataFrame
    decisions: pd.DataFrame
    metrics: dict


def simulate(
    candles: pd.DataFrame, probabilities: np.ndarray, config: BacktestSettings,
    candle_seconds: int, horizon_steps: int, strategy: str = "policy",
) -> Simulation:
    """Signals at candle close fill only after latency_bars full subsequent bars.

    The policy uses fixed-entry sizing, no rebalancing, no shorts and no overlapping
    positions. Horizon exits occur at entry-open + horizon_steps bars; earlier
    SELL signals use the same latency as BUY. All positions are liquidated at
    the final open, a boundary fixed before simulation. Cash earns no interest.
    Equity and risk are marked only at opens, not hypothetical intrabar prices.
    """
    config = load_backtest_settings(asdict(config))
    positive_integer(horizon_steps, "horizon_steps")
    if strategy not in ("policy", "buy_and_hold", "cash"):
        raise ValueError("Unknown backtest strategy")
    frame = validate_candle_timeline(candles, candle_seconds)
    if len(frame) <= config.latency_bars + 2:
        raise ValueError("Not enough candles for delayed entry and terminal liquidation")
    if "open" not in frame or not pd.api.types.is_numeric_dtype(frame["open"]) or pd.api.types.is_bool_dtype(frame["open"]):
        raise ValueError("Open prices must be numeric")
    opens = frame["open"].to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(opens).all() or not (opens > 0).all():
        raise ValueError("Open prices must be positive and finite")
    scores = np.asarray(probabilities, dtype=float)
    if scores.shape != (len(frame), 3) or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any() or not np.allclose(scores.sum(axis=1), 1, rtol=0, atol=1e-10):
        raise ValueError("Require aligned finite normalized up/down/flat probabilities")
    cash, quantity, peak = float(config.initial_cash), 0.0, float(config.initial_cash)
    impact = (config.spread_bps / 2 + config.slippage_bps) / 10000
    fee_rate = config.fee_bps / 10000
    pending = None
    entry = None
    rows, fills, trades, decisions = [], [], [], []
    blocked_entries = 0

    for index, bar in enumerate(frame.itertuples(index=False)):
        reference = opens[index]
        marked_equity = cash + quantity * reference
        peak = max(peak, marked_equity)
        terminal = index == len(frame) - 1
        due = pending if pending is not None and pending["index"] == index else None
        if due is not None:
            pending = None

        # Exits precede entries. Horizon/terminal orders were fixed in advance.
        exit_reason = None
        if entry is not None:
            if terminal:
                exit_reason = "period_end"
            elif strategy == "policy" and index >= entry["index"] + horizon_steps:
                exit_reason = "horizon"
            elif due is not None and due["side"] == "SELL":
                exit_reason = "policy_sell"
        if exit_reason is not None:
            fill_price = reference * (1 - impact)
            notional = quantity * fill_price
            fee = notional * fee_rate
            price_cost = quantity * (reference - fill_price)
            cash += notional - fee
            signal_at = due["signal_at"] if exit_reason == "policy_sell" else entry["signal_at"]
            fills.append(dict(zip(FILL_COLUMNS, [signal_at, bar.timestamp, "SELL", exit_reason, quantity, reference, fill_price, notional, fee, price_cost, cash])))
            gross = quantity * (reference - entry["reference"])
            total_fees = entry["fee"] + fee
            total_impact = entry["price_cost"] + price_cost
            trades.append(dict(zip(TRADE_COLUMNS, [entry["filled_at"], bar.timestamp, exit_reason, index - entry["index"], quantity, entry["reference"], reference, gross, total_fees, total_impact, gross - total_fees - total_impact])))
            quantity, entry, pending = 0.0, None, None

        if due is not None and due["side"] == "BUY" and quantity == 0 and not terminal:
            fill_price = reference * (1 + impact)
            proposed_quantity = config.allocation_fraction * cash / (fill_price * (1 + fee_rate))
            risk = authorize_entry(cash, cash, peak, proposed_quantity, reference, fill_price, config)
            decisions.append({"at": bar.timestamp, "action": "RISK_CHECK", "reason": risk.reason, "approved": risk.allowed})
            if risk.allowed:
                quantity = proposed_quantity
                notional = quantity * fill_price
                fee = notional * fee_rate
                price_cost = quantity * (fill_price - reference)
                cash -= notional + fee
                entry = {"index": index, "filled_at": bar.timestamp, "signal_at": due["signal_at"], "reference": reference, "fee": fee, "price_cost": price_cost}
                fills.append(dict(zip(FILL_COLUMNS, [due["signal_at"], bar.timestamp, "BUY", "policy_entry" if strategy == "policy" else "benchmark_entry", quantity, reference, fill_price, notional, fee, price_cost, cash])))
            else:
                blocked_entries += 1

        marked_equity = cash + quantity * reference
        peak = max(peak, marked_equity)
        if not np.isfinite(marked_equity) or marked_equity <= 0 or cash < -1e-8 * config.initial_cash:
            raise ValueError("Invalid simulation cash or equity; no results published")
        rows.append({"at": bar.timestamp, "reference_price": reference, "cash": cash, "quantity": quantity, "equity": marked_equity, "exposure_fraction": quantity * reference / marked_equity})

        if terminal:
            continue

        # Current probabilities are known only at this candle's close.
        if strategy == "policy":
            decision = decide(scores[index, 0], scores[index, 1], quantity > 0, config)
            action, reason = decision.action, decision.reason
        elif strategy == "buy_and_hold" and index == 0:
            action, reason = "BUY", "benchmark_initial_allocation"
        else:
            action, reason = "HOLD", "benchmark"
        execution_index = index + 1 + config.latency_bars
        queued = pending is None and action in ("BUY", "SELL") and execution_index < len(frame) - 1
        decisions.append({"at": bar.available_at, "action": action, "reason": reason, "queued": queued, "p_up": scores[index, 0], "p_down": scores[index, 1], "p_flat": scores[index, 2]})
        if queued:
            pending = {"index": execution_index, "side": action, "signal_at": bar.available_at}

    equity = pd.DataFrame(rows)
    fill_frame, trade_frame = pd.DataFrame(fills, columns=FILL_COLUMNS), pd.DataFrame(trades, columns=TRADE_COLUMNS)
    metrics = performance_metrics(equity, trade_frame, fill_frame, config.initial_cash, candle_seconds)
    metrics["blocked_entries"] = blocked_entries
    metrics["terminally_truncated_trades"] = int(trade_frame["exit_reason"].eq("period_end").sum()) if strategy == "policy" else 0
    if not np.isclose(metrics["final_equity"] - config.initial_cash, trade_frame["net_pnl"].sum(), rtol=1e-10, atol=1e-8 * config.initial_cash):
        raise ValueError("Trade PnL does not reconcile with final cash")
    return Simulation(equity, fill_frame, trade_frame, pd.DataFrame(decisions), metrics)
