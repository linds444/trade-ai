"""Entry authorization at execution time; closing a position is always allowed."""

from dataclasses import dataclass
import math

from local_ai_trader.settings import BacktestSettings


@dataclass(frozen=True)
class RiskResult:
    allowed: bool
    reason: str


def authorize_entry(
    cash: float, equity: float, peak_equity: float, quantity: float,
    mark_price: float, fill_price: float, config: BacktestSettings,
) -> RiskResult:
    """Account for entry costs when testing cash, drawdown and marked exposure."""
    values = (cash, equity, peak_equity, quantity, mark_price, fill_price)
    if not all(math.isfinite(value) and value > 0 for value in values):
        return RiskResult(False, "invalid_order_or_equity")
    if 1 - equity / peak_equity >= config.max_drawdown_fraction:
        return RiskResult(False, "drawdown_limit")
    cost = quantity * fill_price * (1 + config.fee_bps / 10000)
    if cost > cash + 1e-10 * equity:
        return RiskResult(False, "insufficient_cash")
    after_equity = cash - cost + quantity * mark_price
    if after_equity <= 0 or quantity * mark_price / after_equity > config.max_exposure_fraction + 1e-12:
        return RiskResult(False, "exposure_limit")
    return RiskResult(True, "approved")
