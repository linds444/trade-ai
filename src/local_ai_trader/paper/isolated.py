"""Bounded isolated-margin paper mechanics; no exchange execution.

Full-size book exits require sufficient observed best-bid size. Partial book
fills and liquidations above 100,000 USDC are unsupported. A failed book attempt
below two thirds of maintenance transfers only that position's escrow in this
simplified backstop model. This is not a replica of the exchange matching engine.
"""

from dataclasses import asdict

from local_ai_trader.hyperliquid.venue import (
    VenueRules, isolated_long_liquidation_price, maintenance_margin, number,
    round_price, validate_price,
)

MAX_NOTIONAL = 100000
MODEL = "isolated_full_size_or_backstop_v1"


def encode_rules(rules):
    if not isinstance(rules, VenueRules):
        raise ValueError("Require validated venue rules")
    result = asdict(rules)
    result["tiers"] = [[str(bound), maximum] for bound, maximum in rules.tiers]
    return result


def decode_rules(value):
    value = dict(value)
    value["tiers"] = tuple((number(bound), maximum) for bound, maximum in value["tiers"])
    return VenueRules(**value)


def rules_for(contract, coin):
    return decode_rules(contract["isolated_venue_rules"][coin])


def fill_price(quote, contract, side):
    price = number(quote["ask" if side == "BUY" else "bid"])
    slip = number(contract["slippage_bps"]) / 10000
    return float(round_price(price * (1 + slip if side == "BUY" else 1 - slip), rules_for(contract, quote["coin"]), side=side))


def validate_quote(quote, contract):
    rules = rules_for(contract, quote["coin"])
    if quote["size_decimals"] != rules.size_decimals:
        raise ValueError("Quote size precision disagrees with bound venue rules")
    validate_price(quote["bid"], rules)
    validate_price(quote["ask"], rules)


def position_risk(position, rules):
    quantity = number(position["quantity"])
    mark, entry = number(position["mark_price"]), number(position["entry_fill_price"])
    collateral = number(position["isolated_collateral"])
    equity = collateral + quantity * (mark - entry)
    maintenance = maintenance_margin(quantity * mark, rules)
    # After an extreme funding debit, equity can already be below zero. The
    # static root helper accepts nonnegative collateral only; do not invent one.
    root = isolated_long_liquidation_price(quantity, entry, collateral, rules) if collateral >= 0 else None
    return {"isolated_collateral": float(collateral), "isolated_equity": float(equity),
            "maintenance_margin": float(maintenance), "maintenance_breached": equity <= maintenance,
            "below_backstop_threshold": 3 * equity < 2 * maintenance,
            "static_liquidation_mark": float(root) if root is not None else None}


def book_exit_supported(position, quote, contract, inputs):
    if "bid_size" not in inputs:
        return False, "bid_liquidity_unavailable"
    size = number(inputs["bid_size"])
    if size < 0:
        raise ValueError("Bid liquidity cannot be negative")
    quantity = number(position["quantity"])
    if quantity * number(quote["mark"]) > MAX_NOTIONAL and position_risk(position, rules_for(contract, quote["coin"]))["maintenance_breached"]:
        return False, "partial_liquidation_model_unavailable"
    if size < quantity:
        return False, "insufficient_full_size_bid_liquidity"
    try:
        price = fill_price(quote, contract, "SELL")
    except ValueError:
        return False, "modeled_exit_price_unavailable"
    residual = (number(position["isolated_collateral"]) + quantity * (number(price) - number(position["entry_fill_price"]))
                - quantity * number(price) * number(quote["fee_bps"]) / 10000)
    if residual < 0:
        return False, "book_exit_would_exceed_isolated_collateral"
    return True, None
