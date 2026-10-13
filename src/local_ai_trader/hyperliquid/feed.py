"""Read public mainnet information, validate it and save causal observations."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import tomllib
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import pandas as pd

from local_ai_trader.data.clean import clean_candles
from local_ai_trader.data.storage import native_path, write_json, write_parquet
from local_ai_trader.features.build_features import FEATURE_COLUMNS, build_features
from local_ai_trader.hyperliquid.markets import BY_COIN
from local_ai_trader.markets.source import live_market_state
from local_ai_trader.models.train import file_digest
from local_ai_trader.risk.entry_policy import EntryPolicy
from local_ai_trader.risk.paper import _utc
from local_ai_trader.settings import positive_integer

LOGGER = logging.getLogger(__name__)
INFO_URL = "https://api.hyperliquid.xyz/info"
READ_TYPES = {"metaAndAssetCtxs", "l2Book", "candleSnapshot", "fundingHistory"}
SECONDS = 300


@dataclass(frozen=True)
class HyperliquidSettings:
    coins: tuple[str, ...] = ("BTC", "ETH")
    history_bars: int = 24
    maximum_book_age_seconds: int = 30
    poll_seconds: int = 10
    timeout_seconds: int = 10
    max_attempts: int = 3
    momentum_steps: int = 6
    rolling_window: int = 12
    taker_fee_bps: float = 4.5
    minimum_forward_days: int = 30
    observations: Path = Path("data/hyperliquid/live")
    entry_policy: EntryPolicy = EntryPolicy()


def load_hyperliquid_settings(path: Path) -> HyperliquidSettings:
    path = native_path(path).resolve()
    with path.open("rb") as stream:
        values = tomllib.load(stream)
    expected = {
        "market": {"exchange", "instrument", "coins", "collateral_currency", "leverage", "long_only", "candle_interval", "prediction_horizon_minutes"},
        "features": {"momentum_steps", "rolling_window"},
        "feed": {"history_bars", "maximum_book_age_seconds", "poll_seconds", "timeout_seconds", "max_attempts"},
        "paper": {"taker_fee_bps", "minimum_forward_days", "live_execution_enabled"},
        "gate": {"minimum_probability", "minimum_confidence", "maximum_positions"},
        "paths": {"observations"},
    }
    if set(values) != set(expected) or any(not isinstance(values[name], dict) or set(values[name]) != keys for name, keys in expected.items()):
        raise ValueError("Require the complete Hyperliquid configuration without unknown settings")
    market = values["market"]
    for key, expected_value in {"exchange": "hyperliquid", "instrument": "perpetual", "collateral_currency": "USDC", "candle_interval": "5m"}.items():
        if market[key] != expected_value:
            raise ValueError(f"Unsupported Hyperliquid market.{key}")
    if type(market["leverage"]) is not int or market["leverage"] != 1 or market["long_only"] is not True or type(market["prediction_horizon_minutes"]) is not int or market["prediction_horizon_minutes"] != 30:
        raise ValueError("Hyperliquid paper setup requires 1×, long-only, 30-minute predictions")
    coins = market["coins"]
    if not isinstance(coins, list) or not coins or not all(isinstance(coin, str) and coin in BY_COIN for coin in coins) or len(set(coins)) != len(coins):
        raise ValueError("Hyperliquid coins must be unique registered perpetual identifiers")
    if values["paper"]["live_execution_enabled"] is not False:
        raise ValueError("Live execution is unavailable in the paper data integration")
    parameters = {**values["features"], **values["feed"], **{name: values["paper"][name] for name in ("taker_fee_bps", "minimum_forward_days")}}
    for name in parameters:
        if name != "taker_fee_bps":
            positive_integer(parameters[name], f"hyperliquid.{name}")
    if not 2 <= parameters["rolling_window"] < parameters["history_bars"] <= 5000 or parameters["momentum_steps"] >= parameters["history_bars"]:
        raise ValueError("Hyperliquid history must cover feature warmup and stay within 5000 bars")
    if parameters["poll_seconds"] > 60 or parameters["timeout_seconds"] > 30 or parameters["max_attempts"] > 5 or parameters["maximum_book_age_seconds"] > 30:
        raise ValueError("Hyperliquid polling/freshness/retry limits are too large")
    fee = parameters["taker_fee_bps"]
    if isinstance(fee, bool) or not isinstance(fee, (int, float)) or not math.isfinite(fee) or not 0 <= fee < 10000 or parameters["minimum_forward_days"] < 30:
        raise ValueError("Require a finite fee assumption and at least 30 forward paper days")
    output = values["paths"]["observations"]
    if not isinstance(output, str) or not output.strip():
        raise ValueError("Hyperliquid observation path is required")
    output = Path(output)
    if not output.is_absolute():
        output = path.parent.parent / output
    return HyperliquidSettings(coins=tuple(coins), observations=native_path(output).resolve(), entry_policy=EntryPolicy(**values["gate"]), **parameters)


def _float(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("Invalid Hyperliquid numeric field")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError("Nonfinite or nonpositive Hyperliquid field")
    return value


def _millisecond(value):
    if type(value) is not int or value < 0:
        raise ValueError("Hyperliquid time must be an integer millisecond timestamp")
    return datetime.fromtimestamp(value / 1000, timezone.utc)


class BookTimestampError(ValueError):
    """A rejected exchange timestamp, with enough detail to diagnose the cause."""

    def __init__(self, coin, stamp, now, max_age):
        self.coin = coin
        self.age_seconds = (now - stamp).total_seconds()
        reason = "future-dated" if self.age_seconds < 0 else "stale"
        hint = ("Check Windows automatic time synchronization; the local clock may be behind the exchange."
                if self.age_seconds < 0 else "The exchange response is old or acquisition took too long.")
        super().__init__(f"Hyperliquid {coin} order book is {reason}: age={self.age_seconds:.6f}s; "
                         f"exchange_utc={stamp.isoformat()}; local_utc={now.isoformat()}; "
                         f"require 0 <= age < {max_age}s. {hint}")


def parse_book(payload: dict, coin: str, now: datetime, max_age: int) -> dict:
    if not isinstance(payload, dict) or payload.get("coin") != coin or not isinstance(payload.get("levels"), list) or len(payload["levels"]) != 2 or not all(isinstance(side, list) and side for side in payload["levels"]):
        raise ValueError("Hyperliquid order book is missing, empty or for another market")
    stamp = _millisecond(payload.get("time"))
    age = (_utc(now) - stamp).total_seconds()
    if age < 0 or age >= max_age:
        raise BookTimestampError(coin, stamp, _utc(now), max_age)
    prices, sizes = [], []
    for index, side in enumerate(payload["levels"]):
        side_prices = []
        for level in side:
            if not isinstance(level, dict):
                raise ValueError("Invalid Hyperliquid book level")
            side_prices.append(_float(level.get("px"), positive=True))
            _float(level.get("sz"), positive=True)
            positive_integer(level.get("n"), "book order count")
        if any((a <= b if index == 0 else a >= b) for a, b in zip(side_prices, side_prices[1:])):
            raise ValueError("Hyperliquid book levels are not ordered")
        prices.append(side_prices[0])
        sizes.append(_float(side[0]["sz"], positive=True))
    bid, ask = prices
    if bid >= ask:
        raise ValueError("Hyperliquid order book is crossed or locked")
    mid = bid + (ask - bid) / 2
    return {"exchange_time": stamp.isoformat(), "age_seconds": age, "bid": bid, "ask": ask,
            "bid_size": sizes[0], "ask_size": sizes[1], "mid": mid, "spread_bps": (ask - bid) / mid * 10000}


def parse_contexts(payload: list, coins: tuple[str, ...], received_at: datetime) -> dict:
    if not isinstance(payload, list) or len(payload) != 2 or not isinstance(payload[0], dict) or not isinstance(payload[0].get("universe"), list) or not isinstance(payload[1], list) or len(payload[0]["universe"]) != len(payload[1]):
        raise ValueError("Hyperliquid perpetual metadata/context shape disagrees")
    found = {}
    for asset, context in zip(payload[0]["universe"], payload[1]):
        if not isinstance(asset, dict):
            raise ValueError("Invalid Hyperliquid asset context")
        coin = asset.get("name")
        if coin not in coins:
            continue
        if not isinstance(context, dict):
            raise ValueError("Invalid configured Hyperliquid asset context")
        if coin in found or asset.get("isDelisted", False) is not False or type(asset.get("szDecimals")) is not int or not 0 <= asset["szDecimals"] <= 8 or type(asset.get("maxLeverage")) is not int or asset["maxLeverage"] < 1 or type(asset.get("onlyIsolated", False)) is not bool:
            raise ValueError("Invalid or duplicate Hyperliquid perpetual contract")
        funding = _float(context.get("funding"))
        if abs(funding) > 0.04:
            raise ValueError("Hyperliquid funding rate exceeds its documented hourly cap")
        found[coin] = {"mark_price": _float(context.get("markPx"), positive=True), "oracle_price": _float(context.get("oraclePx"), positive=True),
                       "hourly_funding_rate": funding, "size_decimals": asset["szDecimals"], "exchange_max_leverage": asset["maxLeverage"],
                       "only_isolated": asset.get("onlyIsolated", False), "context_received_at": _utc(received_at).isoformat(),
                       "context_exchange_time": None}
        if "prevDayPx" in context:
            found[coin]["previous_day_price"] = _float(context["prevDayPx"], positive=True)
        if "dayNtlVlm" in context:
            volume = _float(context["dayNtlVlm"])
            if volume < 0:
                raise ValueError("Negative daily notional volume")
            found[coin]["day_notional_volume"] = volume
        if BY_COIN[coin].dex:
            scale = _float(asset.get("deployerFeeScale"))
            growth = asset.get("growthMode")
            if not 0 <= scale <= 3 or growth not in ("enabled", "disabled"):
                raise ValueError("Unknown HIP-3 deployer fee configuration")
            found[coin].update(deployer_fee_scale=scale, growth_mode=growth)
    if set(found) != set(coins):
        raise ValueError(f"Configured perpetual metadata is missing: {sorted(set(coins) - set(found))}")
    return found


def parse_candles(payload: list, coin: str, start: int, end: int):
    if not isinstance(payload, list):
        raise ValueError("Hyperliquid candles must be a list")
    rows = []
    for candle in payload:
        if not isinstance(candle, dict) or candle.get("s") != coin or candle.get("i") != "5m":
            raise ValueError("Hyperliquid candle market/interval disagrees")
        opened, closed = candle.get("t"), candle.get("T")
        _millisecond(opened)
        _millisecond(closed)
        if opened % (SECONDS * 1000) or closed != opened + SECONDS * 1000 - 1:
            raise ValueError("Hyperliquid candle open/close times disagree")
        if type(candle.get("n")) is not int or candle["n"] < 0:
            raise ValueError("Candle trade count must be a nonnegative integer")
        rows.append([opened // 1000, *(_float(candle.get(name)) for name in ("l", "h", "o", "c", "v"))])
    frame, quality = clean_candles(rows, start, end, SECONDS)
    if quality.missing_count:
        raise ValueError(f"Hyperliquid {coin}: {quality.missing_count} missing completed candles")
    frame["symbol"], frame["exchange"] = f"{coin}-USDC", "hyperliquid"
    return frame, quality


class HyperliquidPublicClient:
    def __init__(self, settings: HyperliquidSettings, clock=None):
        self.settings = settings
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def info(self, payload: dict):
        # No exchange/signing/wallet method exists; forbid action types here too.
        if not isinstance(payload, dict) or payload.get("type") not in READ_TYPES:
            raise ValueError("Only allowlisted public Hyperliquid information requests are supported")
        kind = payload["type"]
        expected = {"metaAndAssetCtxs": {"type", "dex"} if "dex" in payload else {"type"}, "l2Book": {"type", "coin"}, "candleSnapshot": {"type", "req"}}
        if kind in expected and set(payload) != expected[kind]:
            raise ValueError("Unexpected fields in public Hyperliquid request")
        if kind == "metaAndAssetCtxs" and "dex" in payload and payload["dex"] not in {BY_COIN[coin].dex for coin in self.settings.coins if BY_COIN[coin].dex}:
            raise ValueError("Unconfigured Hyperliquid perpetual DEX")
        if kind == "fundingHistory" and not {"type", "coin", "startTime"} <= set(payload) <= {"type", "coin", "startTime", "endTime"}:
            raise ValueError("Unexpected funding history fields")
        if kind in ("l2Book", "fundingHistory") and payload["coin"] not in self.settings.coins:
            raise ValueError("Unconfigured Hyperliquid perpetual coin")
        if kind == "candleSnapshot":
            request_fields = payload["req"]
            if not isinstance(request_fields, dict) or set(request_fields) != {"coin", "interval", "startTime", "endTime"} or request_fields["coin"] not in self.settings.coins or request_fields["interval"] != "5m":
                raise ValueError("Invalid public candle request")
            start, end = request_fields["startTime"], request_fields["endTime"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end or start % (SECONDS * 1000) or end % (SECONDS * 1000):
                raise ValueError("Candle bounds must be aligned integer milliseconds")
        if kind == "fundingHistory":
            start = payload["startTime"]
            end = payload.get("endTime")
            if type(start) is not int or start < 0 or (end is not None and (type(end) is not int or end <= start)):
                raise ValueError("Invalid funding history millisecond bounds")
        request = Request(INFO_URL, data=json.dumps(payload, allow_nan=False).encode(), method="POST",
                          headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "local-ai-trader/0.1"})
        for attempt in range(self.settings.max_attempts):
            try:
                with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                    return json.load(response)
            except HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"Hyperliquid public API HTTP {error.code}") from error
                last_error = error
            except (URLError, TimeoutError) as error:
                last_error = error
            if attempt + 1 < self.settings.max_attempts:
                time.sleep(2 ** attempt)
        raise RuntimeError("Hyperliquid public API retries exhausted") from last_error

    def _contexts(self, dex=""):
        payload = self.info({"type": "metaAndAssetCtxs", **({"dex": dex} if dex else {})})
        return payload, _utc(self.clock())

    def snapshot(self):
        for boundary_attempt in range(2):
            now = _utc(self.clock())
            end = int(now.timestamp()) // SECONDS * SECONDS
            start = end - self.settings.history_bars * SECONDS
            raw = {}
            dex_coins = {}
            for coin in self.settings.coins:
                dex_coins.setdefault(BY_COIN[coin].dex, []).append(coin)
            context_receipts = {}
            # Fetch and process history before requesting executable quotes.
            # A slow/retried candle request must not age an already received book.
            with ThreadPoolExecutor(max_workers=min(8, len(dex_coins) + len(self.settings.coins))) as pool:
                jobs = {(f"{dex}_contexts" if dex else "contexts"): pool.submit(self._contexts, dex) for dex in dex_coins}
                for coin in self.settings.coins:
                    jobs[f"{coin}_candles"] = pool.submit(self.info, {"type": "candleSnapshot", "req": {"coin": coin, "interval": "5m", "startTime": start * 1000, "endTime": end * 1000}})
                for name, job in jobs.items():
                    if name == "contexts" or name.endswith("_contexts"):
                        raw[name], context_receipts[name] = job.result()
                    else:
                        raw[name] = job.result()
            now = _utc(self.clock())
            if int(now.timestamp()) // SECONDS * SECONDS != end:
                if boundary_attempt == 0:
                    continue
                raise ValueError("Candle boundary advanced while fetching Hyperliquid data")
            contexts = {}
            for dex, coins in dex_coins.items():
                key = f"{dex}_contexts" if dex else "contexts"
                metadata = raw[key]
                # Token index zero is USDC on Hyperliquid. Refuse a venue
                # collateral change instead of treating a USDH account as USDC.
                if not isinstance(metadata, list) or not metadata or not isinstance(metadata[0], dict) or type(metadata[0].get("collateralToken")) is not int or metadata[0]["collateralToken"] != 0:
                    raise ValueError("Configured perpetual DEX collateral is not verified USDC")
                contexts.update(parse_contexts(metadata, tuple(coins), context_receipts[key]))
            assets, frames, feature_frames = {}, {}, {}
            for coin in self.settings.coins:
                frame, quality = parse_candles(raw[f"{coin}_candles"], coin, start, end)
                features = build_features(frame, SECONDS, self.settings.momentum_steps, self.settings.rolling_window)
                if frame.available_at.iloc[-1] != pd.Timestamp(end, unit="s", tz="UTC"):
                    raise ValueError("Latest Hyperliquid completed candle is stale")
                assets[coin] = {**contexts[coin], "quality": asdict(quality), "last_prediction_time": features.available_at.iloc[-1].isoformat(),
                                "feature_rows": len(features), "latest_features": {name: float(features[name].iloc[-1]) for name in FEATURE_COLUMNS}}
                frames[coin], feature_frames[coin] = frame, features
            with ThreadPoolExecutor(max_workers=min(8, len(self.settings.coins))) as pool:
                jobs = {coin: pool.submit(self.info, {"type": "l2Book", "coin": coin}) for coin in self.settings.coins}
                for coin, job in jobs.items():
                    raw[f"{coin}_book"] = job.result()
            now = _utc(self.clock())
            if int(now.timestamp()) // SECONDS * SECONDS != end:
                if boundary_attempt == 0:
                    continue
                raise ValueError("Candle boundary advanced while fetching Hyperliquid books")
            try:
                books = {coin: parse_book(raw[f"{coin}_book"], coin, now, self.settings.maximum_book_age_seconds) for coin in self.settings.coins}
            except BookTimestampError as error:
                # Refresh once on stale responses, preserving exchange times.
                # Never mask a clock that places fresh exchange data in the future.
                if error.age_seconds >= 0 and boundary_attempt == 0:
                    LOGGER.warning("%s Retrying observation once with new exchange responses.", error)
                    continue
                raise
            for coin in self.settings.coins:
                book = books[coin]
                assets[coin]["book"] = book
                market = BY_COIN[coin]
                assets[coin]["market_state"] = asdict(live_market_state(market, contexts[coin], book, assets[coin]["latest_features"]))
                assets[coin]["contract_identifier"] = coin
                assets[coin]["underlying_units_per_contract"] = market.underlying_units_per_contract
                if market.dex:
                    scale = contexts[coin]["deployer_fee_scale"]
                    multiplier = (1 + scale) if scale < 1 else 2 * scale
                    multiplier *= 0.1 if contexts[coin]["growth_mode"] == "enabled" else 1
                else:
                    multiplier = 1
                assets[coin]["taker_fee_bps_assumption"] = self.settings.taker_fee_bps * multiplier
            report = {"schema_version": 2, "mode": "public_mainnet_data_only", "exchange": "hyperliquid", "instrument": "perpetual", "collateral_currency": "USDC",
                      "configured_leverage": 1, "long_only": True, "created_at": now.isoformat(), "candle_seconds": SECONDS, "prediction_horizon_minutes": 30,
                      "assets": assets, "features": list(FEATURE_COLUMNS), "taker_fee_bps_assumption": self.settings.taker_fee_bps,
                      "minimum_forward_paper_days": self.settings.minimum_forward_days, "forward_strategy_started": False,
                      "live_execution_enabled": False, "orders_created": False, "paper_fills_created": False,
                      "entry_policy": asdict(self.settings.entry_policy), "signal_gate_connected": False,
                      "fee_note": "Base-tier assumption without account discounts; HIP-3 deployer and growth multipliers applied per contract",
                      "context_time_note": "Asset context has no exchange timestamp; reception is recorded separately from exchange-timestamped books",
                      "funding_note": "Current hourly rate; not guaranteed future funding. Payments use position size * oracle price * actual hourly rate."}
            return report, raw, frames, feature_frames


def save_hyperliquid_snapshot(client: HyperliquidPublicClient, config_path: Path):
    config_path = native_path(config_path).resolve()
    config_hash = file_digest(config_path)
    if load_hyperliquid_settings(config_path) != client.settings:
        raise ValueError("Hyperliquid client and observation config disagree")
    report, raw, candles, features = client.snapshot()
    output = client.settings.observations / f"snapshot_{datetime.fromisoformat(report['created_at']).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:8]}"
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".hyperliquid-staging-", dir=output.parent))
    try:
        write_json(staging / "raw.json", raw)
        for coin in client.settings.coins:
            filename = BY_COIN[coin].filename
            write_parquet(staging / f"{filename}_candles.parquet", candles[coin])
            write_parquet(staging / f"{filename}_features.parquet", features[coin])
        report["input_config_sha256"] = config_hash
        report["artifact_hashes"] = {path.name: file_digest(path) for path in staging.iterdir() if path.is_file()}
        write_json(staging / "snapshot.json", report)
        final_now = _utc(client.clock())
        if int(final_now.timestamp()) // SECONDS * SECONDS != int(datetime.fromisoformat(report["created_at"]).timestamp()) // SECONDS * SECONDS:
            raise ValueError("Candle boundary advanced before publishing Hyperliquid observation")
        for coin in client.settings.coins:
            parse_book(raw[f"{coin}_book"], coin, final_now, client.settings.maximum_book_age_seconds)
        if file_digest(config_path) != config_hash:
            raise ValueError("Hyperliquid configuration changed during observation")
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"Hyperliquid mainnet feed: {'/'.join(client.settings.coins)} perpetuals; USDC; configured 1×, long only")
    for coin, asset in report["assets"].items():
        book = asset["book"]
        state = asset["market_state"]
        print(f"{state['symbol']} ({coin}): display price {state['price']:.10g}; contract bid {book['bid']:.10g}; ask {book['ask']:.10g}; hourly funding {100 * asset['hourly_funding_rate']:.6f}%; taker assumption {asset['taker_fee_bps_assumption']:.3f} bps")
        print(f"  Completed candles: {asset['quality']['cleaned_rows']}; missing: 0; feature rows: {asset['feature_rows']}; latest candle available: {asset['last_prediction_time']}")
    print("PASS: live books are fresh; completed candles and causal features validated")
    print("Data-only setup: no paper fills; forward strategy month has not started; live execution disabled")
    print(f"Saved Hyperliquid observation to {output}")
    return output, report


def run_hyperliquid_feed(config_path: Path, *, watch=False):
    settings = load_hyperliquid_settings(config_path)
    client = HyperliquidPublicClient(settings)
    last_boundary = None
    try:
        while True:
            if load_hyperliquid_settings(config_path) != settings:
                raise ValueError("Hyperliquid feed config changed; restart to apply it")
            boundary = int(client.clock().timestamp()) // SECONDS
            if watch and boundary == last_boundary:
                time.sleep(settings.poll_seconds)
                continue
            try:
                _, report = save_hyperliquid_snapshot(client, config_path)
                last_boundary = int(datetime.fromisoformat(report["created_at"]).timestamp()) // SECONDS
            except (ValueError, RuntimeError, OSError) as error:
                if not watch:
                    raise
                LOGGER.error("Hyperliquid observation refused: %s", error)
            if not watch:
                return
            time.sleep(settings.poll_seconds)
    except KeyboardInterrupt:
        print("Hyperliquid public feed stopped")
