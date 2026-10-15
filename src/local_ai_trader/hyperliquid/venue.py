"""Public venue constraints and isolated-long margin mathematics; no execution.

Decimal calculations preserve contract units. These checks do not authorize a
signal or simulate liquidation fills, partial liquidation or a backstop vault.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR, localcontext
import json
import os
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

import pandas as pd

from local_ai_trader.data.storage import native_path, write_json
from local_ai_trader.hyperliquid.feed import HyperliquidPublicClient, load_hyperliquid_settings, parse_book, parse_contexts
from local_ai_trader.hyperliquid.markets import BY_COIN
from local_ai_trader.models.train import file_digest
from local_ai_trader.risk.paper import _utc

DOCUMENTS = {name: "https://hyperliquid.gitbook.io/hyperliquid-docs/" + path for name, path in {
    "precision": "for-developers/api/tick-and-lot-size", "minimum_notional": "for-developers/api/error-responses",
    "margin": "trading/margining", "tiers": "trading/margin-tiers", "liquidations": "trading/liquidations"}.items()}


def number(value, *, positive=False):
    if type(value) not in (int, float, str, Decimal) or len(str(value)) > 100:
        raise ValueError("Require bounded finite decimal venue inputs")
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("Invalid decimal venue input") from error
    if not result.is_finite() or result.copy_abs() > Decimal("1e30") or (result and result.copy_abs() < Decimal("1e-30")) or (positive and result <= 0):
        raise ValueError("Invalid or out-of-range decimal venue input")
    return result


def _integer(value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("Invalid integer venue metadata")
    return value


def _text(value):
    with localcontext() as context:
        context.prec = 100
        return "0" if not value else format(value.normalize(), "f")


@dataclass(frozen=True)
class VenueRules:
    coin: str
    size_decimals: int
    max_leverage: int
    margin_table_id: int
    tiers: tuple[tuple[Decimal, int], ...]
    only_isolated: bool = False
    strict_isolated: bool = False

    def __post_init__(self):
        if self.coin not in BY_COIN:
            raise ValueError("Unregistered venue contract")
        _integer(self.size_decimals, 0, 8)
        _integer(self.max_leverage, 1, 1000)
        _integer(self.margin_table_id, 1, 1000000)
        if type(self.only_isolated) is not bool or type(self.strict_isolated) is not bool or self.strict_isolated and not self.only_isolated:
            raise ValueError("Invalid venue margin mode")
        if type(self.tiers) is not tuple or not self.tiers or len(self.tiers) > 100 or any(type(tier) is not tuple or len(tier) != 2 for tier in self.tiers) or self.tiers[0][0] != 0 or self.tiers[0][1] != self.max_leverage:
            raise ValueError("Margin tiers do not establish the asset maximum leverage")
        previous, leverage = Decimal(-1), self.max_leverage
        for bound, maximum in self.tiers:
            if not isinstance(bound, Decimal) or number(bound) < 0 or bound <= previous or _integer(maximum, 1, 1000) > leverage:
                raise ValueError("Margin tiers must increase in notional and not increase leverage")
            previous, leverage = bound, maximum

    @property
    def size_step(self):
        return Decimal(1).scaleb(-self.size_decimals)


def parse_rules(metadata, coin):
    if not isinstance(metadata, dict) or type(metadata.get("collateralToken")) is not int or metadata["collateralToken"] != 0 or not isinstance(metadata.get("universe"), list):
        raise ValueError("Venue metadata must verify USDC collateral and a universe")
    matches = [asset for asset in metadata["universe"] if isinstance(asset, dict) and asset.get("name") == coin]
    if len(matches) != 1 or matches[0].get("isDelisted", False) is not False:
        raise ValueError("Venue contract is missing, duplicated or delisted")
    asset = matches[0]
    table_id = _integer(asset.get("marginTableId"), 1, 1000000)
    tables = metadata.get("marginTables", [])
    if not isinstance(tables, list) or any(not isinstance(item, list) or len(item) != 2 or type(item[0]) is not int for item in tables):
        raise ValueError("Malformed venue margin table registry")
    table = [item[1] for item in tables if item[0] == table_id]
    if len(table) > 1:
        raise ValueError("Duplicate venue margin table")
    if table:
        if not isinstance(table[0], dict) or not isinstance(table[0].get("marginTiers"), list):
            raise ValueError("Malformed venue margin tiers")
        if any(not isinstance(tier, dict) or not {"lowerBound", "maxLeverage"} <= set(tier) for tier in table[0]["marginTiers"]):
            raise ValueError("Incomplete venue margin tier")
        tiers = tuple((number(tier["lowerBound"]), tier["maxLeverage"]) for tier in table[0]["marginTiers"])
        if table_id < 50 and tiers != ((Decimal(0), table_id),):
            raise ValueError("Single-tier margin ID disagrees with documented semantics")
    elif table_id < 50:
        # Official margin-tier specification defines IDs 1..49 as single-tier.
        tiers = ((Decimal(0), table_id),)
    else:
        raise ValueError("Explicit tiered margin table unavailable; no leverage fallback")
    mode = asset.get("marginMode")
    if mode not in (None, "cross", "isolated", "noCross", "strictIsolated") or type(asset.get("onlyIsolated", False)) is not bool:
        raise ValueError("Unsupported venue margin mode")
    return VenueRules(coin, asset.get("szDecimals"), asset.get("maxLeverage"), table_id, tiers,
                      asset.get("onlyIsolated", False) or mode in ("isolated", "noCross", "strictIsolated"), mode == "strictIsolated")


def validate_price(price, rules):
    price = number(price, positive=True)
    # Integer prices are always allowed, including >5 significant figures.
    if price != price.to_integral_value():
        with localcontext() as context:
            context.prec = 100
            normalized = price.normalize()
        if -normalized.as_tuple().exponent > max(0, 6-rules.size_decimals) or len(normalized.as_tuple().digits) > 5:
            raise ValueError("Price violates perpetual decimal/significant-figure limits")
    return price


def round_price(price, rules, *, side):
    price = number(price, positive=True)
    if side not in ("BUY", "SELL"):
        raise ValueError("Require BUY or SELL rounding direction")
    places = max(0, min(6-rules.size_decimals, 4-price.adjusted()))
    with localcontext() as context:
        context.prec = 60
        result = price.quantize(Decimal(1).scaleb(-places), rounding=ROUND_CEILING if side == "BUY" else ROUND_FLOOR)
    return validate_price(result, rules)


def floor_size(quantity, rules):
    quantity = number(quantity)
    if quantity < 0:
        raise ValueError("Long-only size cannot be negative")
    with localcontext() as context:
        context.prec = 60
        return quantity.quantize(rules.size_step, rounding=ROUND_FLOOR)


def maintenance_margin(notional, rules):
    notional = number(notional)
    if notional < 0:
        raise ValueError("Maintenance notional must be nonnegative")
    with localcontext() as context:
        context.prec = 60
        deduction, previous_rate = Decimal(0), Decimal(0)
        for index, (lower, maximum) in enumerate(rules.tiers):
            rate = Decimal(1)/(2*maximum)
            deduction += lower*(rate-previous_rate)
            if index == len(rules.tiers)-1 or notional < rules.tiers[index+1][0]:
                return notional*rate-deduction
            previous_rate = rate


def isolated_long_liquidation_price(quantity, entry_price, collateral_after_costs, rules):
    """Static mark threshold; caller supplies isolated collateral after fees/funding.

    Solve equity(mark) = maintenance(quantity*mark), selecting the tier at the
    root, not the current mark. Zero means no positive root under these inputs.
    It does not model future funding, liquidation fills, cross-margin or vaults.
    """
    quantity, entry_price = number(quantity, positive=True), number(entry_price, positive=True)
    collateral = number(collateral_after_costs)
    if collateral < 0:
        raise ValueError("Isolated collateral must be nonnegative")
    with localcontext() as context:
        context.prec = 60
        deduction, previous_rate = Decimal(0), Decimal(0)
        for index, (lower, maximum) in enumerate(rules.tiers):
            rate = Decimal(1)/(2*maximum)
            deduction += lower*(rate-previous_rate)
            price = (quantity*entry_price-collateral-deduction)/(quantity*(1-rate))
            notional = price*quantity
            if price <= 0 and index == 0:
                return Decimal(0)
            if price > 0 and lower <= notional and (index == len(rules.tiers)-1 or notional < rules.tiers[index+1][0]):
                return price
            previous_rate = rate
    raise ValueError("Unable to locate a consistent liquidation tier")


def validate_entry_order(quantity, price, mark, rules, *, side="BUY", leverage=1, margin_mode="isolated"):
    quantity, price, mark = number(quantity, positive=True), validate_price(price, rules), number(mark, positive=True)
    if side != "BUY" or type(leverage) is not int or leverage != 1:
        raise ValueError("Paper entry plan requires 1x long-only")
    if margin_mode not in ("isolated", "cross") or rules.only_isolated and margin_mode != "isolated":
        raise ValueError("Venue requires a supported margin mode; this asset may require isolated")
    if floor_size(quantity, rules) != quantity:
        raise ValueError("Size violates native contract precision")
    with localcontext() as context:
        context.prec = 60
        if quantity*price < 10:
            raise ValueError("Opening order is below documented 10 USDC minimum notional")
        return {"entry_notional": _text(quantity*price), "initial_margin_at_mark_1x": _text(quantity*mark),
                "maintenance_at_mark": _text(maintenance_margin(quantity*mark, rules)), "account_risk_authorized": False}


def assess_metadata(raw, assets, settings, at):
    """Pure, auditable per-market rules report; absent evidence never grants risk."""
    results = {}
    for coin in settings.coins:
        try:
            asset = assets[coin]
            dex = BY_COIN[coin].dex
            metadata = raw[f"{dex}_contexts" if dex else "contexts"][0]
            rules = parse_rules(metadata, coin)
            if rules.size_decimals != asset["size_decimals"]:
                raise ValueError("Venue size precision disagrees with parsed market context")
            book = parse_book(raw[f"{coin}_book"], coin, at, settings.maximum_book_age_seconds)
            validate_price(book["bid"], rules)
            validate_price(book["ask"], rules)
            age = (at-_utc(datetime.fromisoformat(asset["context_received_at"]))).total_seconds()
            if not 0 <= age < settings.maximum_book_age_seconds:
                raise ValueError("Venue mark context is future-dated or stale")
            with localcontext() as context:
                context.prec = 60
                first_rate = Decimal(1)/(2*rules.max_leverage)
            results[coin] = {"status": "READY", "size_step_contracts": _text(rules.size_step),
                "minimum_entry_notional_usdc": "10", "exchange_max_leverage": rules.max_leverage,
                "planned_leverage": 1, "planned_margin_mode": "isolated", "isolated_required": rules.only_isolated,
                "strict_isolated": rules.strict_isolated, "maximum_price_decimals": max(0, 6-rules.size_decimals),
                "fractional_price_significant_digits": 5, "integer_price_exception": True,
                "margin_table_id": rules.margin_table_id,
                "margin_tiers": [{"lower_bound_usdc": _text(lower), "max_leverage": maximum} for lower, maximum in rules.tiers],
                "first_tier_maintenance_rate": _text(first_rate),
                "underlying_units_per_contract": BY_COIN[coin].underlying_units_per_contract,
                "account_risk_authorized": False}
        except (ValueError, KeyError, TypeError, IndexError) as error:
            results[coin] = {"status": "UNAVAILABLE", "reason": str(error), "account_risk_authorized": False}
    return {"schema_version": 1, "mode": "public_venue_constraints", "markets": results,
            "documentation": DOCUMENTS, "entries_enabled": False, "account_risk_authorized": False,
            "liquidation_fill_simulation_connected": False, "cross_margin_simulation_supported": False}


def capture_venue_check(config_path, *, client=None):
    config_path = native_path(Path(config_path)).resolve()
    settings = load_hyperliquid_settings(config_path)
    digest = file_digest(config_path)
    client = client or HyperliquidPublicClient(settings)
    if client.settings != settings:
        raise ValueError("Venue-check client/config disagree")
    raw, assets, receipts = {}, {}, {}
    dex_coins = {dex: tuple(c for c in settings.coins if BY_COIN[c].dex == dex) for dex in dict.fromkeys(BY_COIN[c].dex for c in settings.coins)}
    with ThreadPoolExecutor(max_workers=len(dex_coins)) as pool:
        jobs = {dex: pool.submit(client._contexts, dex) for dex in dex_coins}
        for dex, job in jobs.items():
            key = f"{dex}_contexts" if dex else "contexts"
            raw[key], at = job.result()
            receipts[key] = at.isoformat()
            assets.update(parse_contexts(raw[key], dex_coins[dex], at))
    with ThreadPoolExecutor(max_workers=min(8, len(settings.coins))) as pool:
        jobs = {coin: pool.submit(client.info, {"type": "l2Book", "coin": coin}) for coin in settings.coins}
        for coin, job in jobs.items():
            raw[f"{coin}_book"] = job.result()
    at = _utc(client.clock())
    report = assess_metadata(raw, assets, settings, at)
    parent = settings.observations.parent / "venue_checks"
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / f"check_{at.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:8]}"
    staging = Path(tempfile.mkdtemp(prefix=".venue-staging-", dir=parent))
    try:
        write_json(staging / "raw.json", {"responses": raw, "context_receipts": receipts})
        (staging / "input_config.toml").write_bytes(config_path.read_bytes())
        report.update(captured_at=at.isoformat(), source_mode="observed", input_config_sha256=digest,
                      artifact_hashes={name: file_digest(staging/name) for name in ("raw.json", "input_config.toml")})
        write_json(staging / "venue_check.json", report)
        if file_digest(config_path) != digest or assess_metadata(raw, assets, settings, _utc(client.clock())) != {key: value for key, value in report.items() if key not in ("captured_at", "source_mode", "input_config_sha256", "artifact_hashes")}:
            raise ValueError("Venue evidence/config expired or changed before publication")
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print("Hyperliquid venue constraints (public mainnet; native contract units):")
    rows = [{"symbol": BY_COIN[coin].symbol, "status": row["status"], "size_step": row.get("size_step_contracts", "-"),
             "max_venue_leverage": row.get("exchange_max_leverage", "-"), "margin_plan": row.get("planned_margin_mode", "-"),
             "isolated_required": row.get("isolated_required", "-"), "reason": row.get("reason", "")} for coin, row in report["markets"].items()]
    print(pd.DataFrame(rows).to_string(index=False))
    ready = sum(row["status"] == "READY" for row in report["markets"].values())
    print(f"Venue metadata ready: {ready}/{len(rows)}; planned 1x, long only, isolated")
    print("Constraints only: no entries, account changes or strategy trial; liquidation fills and independent reliability remain unconnected")
    print(f"Saved venue check to {output}")
    return output, report


def load_venue_check(path):
    path = native_path(Path(path)).resolve()
    report = json.loads((path/"venue_check.json").read_text(encoding="utf-8"))
    for name in ("raw.json", "input_config.toml"):
        if file_digest(path/name) != report["artifact_hashes"][name]:
            raise ValueError("Venue-check artifact changed")
    if report["source_mode"] != "observed" or report["input_config_sha256"] != file_digest(path/"input_config.toml"):
        raise ValueError("Venue-check source/config changed")
    settings = load_hyperliquid_settings(path/"input_config.toml")
    payload = json.loads((path/"raw.json").read_text(encoding="utf-8"))
    raw, assets = payload["responses"], {}
    for dex in dict.fromkeys(BY_COIN[c].dex for c in settings.coins):
        key = f"{dex}_contexts" if dex else "contexts"
        coins = tuple(c for c in settings.coins if BY_COIN[c].dex == dex)
        assets.update(parse_contexts(raw[key], coins, _utc(datetime.fromisoformat(payload["context_receipts"][key]))))
    expected = assess_metadata(raw, assets, settings, _utc(datetime.fromisoformat(report["captured_at"])))
    if expected != {key: value for key, value in report.items() if key not in ("captured_at", "source_mode", "input_config_sha256", "artifact_hashes")}:
        raise ValueError("Venue-check derived rules changed")
    return report
