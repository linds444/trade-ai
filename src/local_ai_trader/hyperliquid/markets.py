"""Explicit display symbols, exchange contracts and contract quantity units."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Market:
    symbol: str
    coin: str
    underlying_units_per_contract: int = 1

    @property
    def dex(self):
        return self.coin.split(":")[0] if ":" in self.coin else ""

    @property
    def filename(self):
        # Colons are illegal Windows filename characters; the API identifier
        # itself is preserved in every report, request and candle dataset.
        return self.coin.replace(":", "_")


MARKETS = (
    Market("HOOD", "xyz:HOOD"), Market("NVDA", "xyz:NVDA"),
    Market("SOL/USD", "SOL"), Market("ETH/USD", "ETH"),
    Market("TSLA", "xyz:TSLA"), Market("BTC/USD", "BTC"),
    Market("PEPE/USD", "kPEPE", 1000), Market("DOGE/USD", "DOGE"),
)
BY_COIN = {market.coin: market for market in MARKETS}
