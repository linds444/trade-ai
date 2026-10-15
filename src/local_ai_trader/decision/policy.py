"""Provisional long-only research policy; probabilities are not expected returns."""

from dataclasses import dataclass

from local_ai_trader.settings import BacktestSettings


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str


def decide(p_up: float, p_down: float, holding: bool, config: BacktestSettings) -> Decision:
    """BUY opens a spot position; SELL closes it. No shorting or invented confidence."""
    if holding:
        if p_down >= config.exit_probability and p_down - p_up >= config.minimum_direction_margin:
            return Decision("SELL", "down_probability_and_margin")
        return Decision("HOLD", "position_open")
    if p_up >= config.entry_probability and p_up - p_down >= config.minimum_direction_margin:
        return Decision("BUY", "up_probability_and_margin")
    return Decision("AVOID", "entry_threshold_not_met")
