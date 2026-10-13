"""A common market state without invented live news or model confidence."""

from dataclasses import dataclass
from datetime import datetime
import random

from local_ai_trader.hyperliquid.markets import MARKETS
from local_ai_trader.risk.paper import _utc


@dataclass(frozen=True)
class MarketState:
    symbol: str
    contract: str
    source_mode: str
    observed_at: str
    price: float
    change_24h: float | None
    volume_24h_notional: float | None
    momentum: float
    news: tuple[str, ...] | None
    regime: str
    regime_method: str
    underlying_units_per_contract: int
    contract_price: float
    collateral_currency: str = "USDC"


def live_market_state(market, context, book, features):
    """Display price per underlying; feature and execution units stay contracts.

    Context 24h volume is quote notional and must not be multiplied by kPEPE's
    quantity conversion. News/regime inference is not part of this price feed.
    """
    previous = context.get("previous_day_price")
    return MarketState(
        symbol=market.symbol, contract=market.coin, source_mode="observed",
        observed_at=book["exchange_time"],
        price=book["mid"] / market.underlying_units_per_contract,
        change_24h=None if previous is None else context["mark_price"] / previous - 1,
        volume_24h_notional=context.get("day_notional_volume"),
        momentum=features["feature_momentum"], news=None,
        regime="unavailable", regime_method="no_regime_model_connected",
        underlying_units_per_contract=market.underlying_units_per_contract,
        contract_price=book["mid"],
    )


def generate_offline_states(at: datetime, *, seed: int = 0):
    """Seeded test universe, explicitly synthetic, with no claimed calibration."""
    stamp = _utc(at).isoformat()
    rng = random.Random(seed)
    reference = (80., 180., 150., 2500., 350., 83000., 0.00001, 0.2)
    return tuple(MarketState(
        symbol=market.symbol, contract=market.coin, source_mode="synthetic",
        observed_at=stamp, price=price, change_24h=rng.uniform(-0.1, 0.1),
        volume_24h_notional=rng.uniform(1e6, 1e9), momentum=rng.uniform(-0.02, 0.02),
        news=("Invented offline test input",), regime="synthetic",
        regime_method="generated_test_input",
        underlying_units_per_contract=market.underlying_units_per_contract,
        contract_price=price * market.underlying_units_per_contract,
    ) for market, price in zip(MARKETS, reference))
