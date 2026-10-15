"""Accounting and descriptive statistics for a marked equity path."""

import numpy as np
import pandas as pd


def performance_metrics(
    equity: pd.DataFrame, trades: pd.DataFrame, fills: pd.DataFrame,
    initial_cash: float, candle_seconds: int,
) -> dict:
    """Annualization uses 365-day crypto years and zero risk-free return."""
    values = np.r_[initial_cash, equity["equity"].to_numpy(dtype=float)]
    # The first open is the initial-cash point; each subsequent open is one bar.
    returns = values[2:] / values[1:-1] - 1
    annual_periods = 365 * 86400 / candle_seconds
    deviation = returns.std(ddof=1) if len(returns) > 1 else 0.0
    downside = np.sqrt(np.mean(np.minimum(returns, 0) ** 2))
    pnl = trades["net_pnl"].to_numpy(dtype=float)
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {
        "initial_cash": initial_cash, "final_equity": float(values[-1]),
        "net_return": float(values[-1] / initial_cash - 1),
        "gross_pnl_before_costs": float(trades["gross_pnl"].sum()),
        "fees_paid": float(fills["fee"].sum()),
        "spread_and_slippage_cost": float(fills["price_impact_cost"].sum()),
        "max_drawdown": float(np.max(1 - values / np.maximum.accumulate(values))),
        "sharpe": float(returns.mean() / deviation * np.sqrt(annual_periods)) if deviation > 1e-15 else None,
        "sortino": float(returns.mean() / downside * np.sqrt(annual_periods)) if downside > 1e-15 else None,
        "win_rate": float(np.mean(pnl > 0)) if len(pnl) else None,
        "profit_factor": float(wins / losses) if losses > 0 else None,
        "number_of_trades": len(trades), "number_of_fills": len(fills),
        "turnover": float(fills["notional"].sum() / initial_cash),
        "time_in_market_fraction": float(equity["quantity"].gt(0).mean()),
        "average_exposure_fraction": float(equity["exposure_fraction"].mean()),
        "maximum_observed_exposure_fraction": float(equity["exposure_fraction"].max()),
        "annualization_periods": annual_periods,
    }
