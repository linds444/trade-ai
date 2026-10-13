"""Public settled rates and auditable near-settlement oracle proxies.

Historical funding responses contain no oracle prices. We preserve missing
operands; a current oracle never substitutes for an unobserved historical one.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
from uuid import uuid4

from local_ai_trader.data.storage import native_path, write_json
from local_ai_trader.hyperliquid.feed import HyperliquidPublicClient, _float, _millisecond, load_hyperliquid_settings, parse_book, parse_contexts
from local_ai_trader.hyperliquid.markets import BY_COIN
from local_ai_trader.models.train import file_digest
from local_ai_trader.paper.ledger import _hash, _iso
from local_ai_trader.paper.perpetual import PerpetualPaperAccount
from local_ai_trader.risk.paper import _utc

LOGGER = logging.getLogger(__name__)


def _hour(at):
    return _utc(at).replace(minute=0, second=0, microsecond=0)


def _stamp(value):
    return _utc(datetime.fromisoformat(value))


def parse_funding_history(payload, coin, start_ms, end_ms):
    if not isinstance(payload, list):
        raise ValueError("Funding history must be a list")
    found = {}
    duplicates = 0
    for row in payload:
        if not isinstance(row, dict) or row.get("coin") != coin:
            raise ValueError(f"Funding history market disagrees: {coin}")
        at = _millisecond(row.get("time"))
        hour = _hour(at)
        if not start_ms <= row["time"] <= end_ms or not 0 <= (at - hour).total_seconds() < 30:
            raise ValueError("Funding event time is outside the request or hourly settlement window")
        rate, premium = _float(row.get("fundingRate")), _float(row.get("premium"))
        if abs(rate) > 0.04:
            raise ValueError("Settled funding exceeds the documented hourly cap")
        point = {"coin": coin, "hour": _iso(hour), "exchange_event_at": _iso(at), "rate": rate, "premium": premium}
        key = point["hour"]
        if key in found:
            if found[key] != point:
                raise ValueError(f"Conflicting funding settlement: {coin} {key}")
            duplicates += 1
        found[key] = point
    return [found[key] for key in sorted(found)], duplicates


def derive_observation(raw, coins, maximum_book_age_seconds=30):
    now = _stamp(raw["captured_at"])
    query = raw["history_request"]
    if set(query) != {"startTime", "endTime"} or type(query["startTime"]) is not int or type(query["endTime"]) is not int or not 0 <= query["startTime"] < query["endTime"] <= int(now.timestamp() * 1000):
        raise ValueError("Invalid archived funding request bounds")
    if set(raw["history"]) != set(coins) or set(raw["books"]) != set(coins) or set(raw["history_received_at"]) != set(coins):
        raise ValueError("Funding observation requires all configured contracts")
    dex_coins = {}
    for coin in coins:
        if coin not in BY_COIN:
            raise ValueError("Unregistered funding contract")
        dex_coins.setdefault(BY_COIN[coin].dex, []).append(coin)
    if set(raw["contexts"]) != set(dex_coins) or set(raw["contexts_received_at"]) != set(dex_coins):
        raise ValueError("Funding DEX metadata disagrees")
    contexts = {}
    for dex, group in dex_coins.items():
        received = _stamp(raw["contexts_received_at"][dex])
        if received > now:
            raise ValueError("Future oracle reception timestamp")
        data = raw["contexts"][dex]
        contexts.update(parse_contexts(data, tuple(group), received))
        if type(data[0].get("collateralToken")) is not int or data[0]["collateralToken"] != 0:
            raise ValueError("Funding source collateral must be USDC")
    rates, oracles, summaries = [], [], {}
    first_hour = _hour(now) - timedelta(hours=23)
    expected = {_iso(first_hour + timedelta(hours=index)) for index in range(24)}
    for coin in coins:
        receipt = _stamp(raw["history_received_at"][coin])
        if not _millisecond(query["endTime"]) <= receipt <= now:
            raise ValueError("Invalid funding history receipt time")
        points, duplicates = parse_funding_history(raw["history"][coin], coin, query["startTime"], query["endTime"])
        for point in points:
            point["available_at"] = _iso(receipt)
        rates.extend(points)
        book = parse_book(raw["books"][coin], coin, now, maximum_book_age_seconds)
        ctx = contexts[coin]
        oracles.append({"coin": coin, "oracle_price": ctx["oracle_price"], "at": _iso(_stamp(ctx["context_received_at"])),
                        "exchange_time": None, "corroborating_book_time": book["exchange_time"], "modeled_proxy": True})
        summaries[coin] = {"rates": len(points), "duplicate_rows_removed": duplicates,
                           "missing_rate_hours": sorted(expected - {p["hour"] for p in points})}
    return {"rates": rates, "oracles": oracles, "assets": summaries}


def load_funding_capture(path):
    path = native_path(path).resolve()
    with (path / "capture.json").open(encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("schema_version") not in (1, 2) or report.get("mode") != "observed_funding_data_only" or report.get("exchange") != "hyperliquid" or report.get("currency") != "USDC" or set(report.get("artifact_hashes", {})) != {"raw.json", "input_config.toml"}:
        raise ValueError("Unsupported funding capture")
    for filename, digest in report["artifact_hashes"].items():
        if file_digest(path / filename) != digest:
            raise ValueError("Funding capture artifact changed")
    if report["input_config_sha256"] != report["artifact_hashes"]["input_config.toml"]:
        raise ValueError("Funding configuration checksum disagrees")
    settings = load_hyperliquid_settings(path / "input_config.toml")
    if list(settings.coins) != report["coins"]:
        raise ValueError("Funding configuration contracts disagree")
    with (path / "raw.json").open(encoding="utf-8") as stream:
        raw = json.load(stream)
    observation = derive_observation(raw, settings.coins, settings.maximum_book_age_seconds)
    expected = observation
    if report["schema_version"] == 1:
        expected = {**observation, "rates": [{k: v for k, v in p.items() if k != "available_at"} for p in observation["rates"]]}
    if expected != report["observation"] or raw["captured_at"] != report["created_at"]:
        raise ValueError("Funding capture summary differs from raw evidence")
    source_id = "sha256:" + report["artifact_hashes"]["raw.json"]
    return observation, source_id


def pair_recorded_funding(root, required_keys=None, *, extra_paths=()):
    root = native_path(root).resolve()
    rates, oracles = {}, {}
    paths = sorted(root.glob("capture_*")) if root.exists() else []
    if required_keys:
        hours = [_stamp(key.split("|", 1)[1]) for key in required_keys]
        lower, upper = min(hours) - timedelta(minutes=1), max(hours) + timedelta(hours=25)
        def in_window(path):
            try:
                at = datetime.strptime(path.name.split("_")[1], "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc)
            except (ValueError, IndexError):
                raise ValueError("Invalid funding capture directory name")
            return lower <= at <= upper
        paths = [p for p in paths if in_window(p)]
    paths.extend(extra_paths)
    if paths:
        for path in paths:
            observation, source_id = load_funding_capture(path)
            for point in observation["rates"]:
                key = point["coin"] + "|" + point["hour"]
                if required_keys is not None and key not in required_keys:
                    continue
                previous = rates.get(key)
                economics = {k: v for k, v in point.items() if k != "available_at"}
                if previous is not None and {k: v for k, v in previous[0].items() if k != "available_at"} != economics:
                    raise ValueError("Archived funding rates conflict across captures")
                if previous is None or (point["available_at"], source_id) < (previous[0]["available_at"], previous[1]):
                    rates[key] = (point, source_id)
            for point in observation["oracles"]:
                oracles.setdefault(point["coin"], []).append((point, source_id))
    ready, unavailable = {}, []
    for key, (point, source_id) in sorted(rates.items()):
        event = _stamp(point["exchange_event_at"])
        candidates = [(abs((_stamp(p["at"]) - event).total_seconds()), p["at"], sid, p)
                      for p, sid in oracles.get(point["coin"], []) if abs((_stamp(p["at"]) - event).total_seconds()) < 30]
        if not candidates:
            unavailable.append(key)
            continue
        _, _, oracle_id, oracle = min(candidates, key=lambda item: item[:3])
        ready[key] = {"coin": point["coin"], "hour": point["hour"], "exchange_event_at": point["exchange_event_at"],
                      "rate": point["rate"], "oracle_price": oracle["oracle_price"], "oracle_at": oracle["at"],
                      "rate_source_id": source_id, "oracle_source_id": oracle_id, "source_mode": "observed"}
        ready[key]["rate_observed_at"] = point["available_at"]
    return ready, unavailable


def apply_recorded_funding(account_path, archive_root, *, now=None, clock=None):
    """Future coordinator adapter; this function never opens or closes positions."""
    account = PerpetualPaperAccount(account_path)
    report = account.audit()
    if report["source_mode"] != "observed" or report["funding_time_basis"] != "exchange_event":
        raise ValueError("Recorded funding requires an observed account with actual event timing")
    if not report["missing_funding"]:
        return [], report
    ready, _ = pair_recorded_funding(archive_root, set(report["missing_funding"]))
    results = []
    for key in report["missing_funding"]:
        if key in ready:
            results.append(account.execute("recorded-funding:" + key, "funding", ready[key], now=now, clock=clock))
    return results, account.audit()


@contextmanager
def _publication_lock(root, *, label="funding collector"):
    # OS locks are released on process death; the marker file is retained so
    # unlocking cannot race with a new publisher opening another inode.
    with (root / ".publication.lock").open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError(f"Another {label} is publishing; retry") from error
        try:
            yield
        finally:
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class FundingCollector:
    def __init__(self, client, config):
        self.client = client
        self.config = native_path(config).resolve()
        self.root = client.settings.observations.parent / "funding"
        self.cache = None

    def capture(self, *, publish=True):
        cfg_hash = file_digest(self.config)
        if load_hyperliquid_settings(self.config) != self.client.settings:
            raise ValueError("Funding client/config disagree")
        now = _utc(self.client.clock())
        if self.cache is None or not 0 <= (now - self.cache[0]).total_seconds() < 60 or _hour(now) != _hour(self.cache[0]):
            query = {"startTime": int((_hour(now) - timedelta(hours=23)).timestamp() * 1000), "endTime": int(now.timestamp() * 1000)}
            history, receipts = {}, {}
            def fetch(coin):
                payload = self.client.info({"type": "fundingHistory", "coin": coin, **query})
                return payload, _iso(self.client.clock())
            with ThreadPoolExecutor(max_workers=min(8, len(self.client.settings.coins))) as pool:
                jobs = {coin: pool.submit(fetch, coin) for coin in self.client.settings.coins}
                for coin, job in jobs.items():
                    history[coin], receipts[coin] = job.result()
            self.cache = (now, query, history, receipts)
        _, query, history, receipts = self.cache
        contexts, context_receipts, books = {}, {}, {}
        dexes = {BY_COIN[coin].dex for coin in self.client.settings.coins}
        with ThreadPoolExecutor(max_workers=min(8, len(dexes))) as pool:
            jobs = {dex: pool.submit(self.client._contexts, dex) for dex in dexes}
            for dex, job in jobs.items():
                contexts[dex], at = job.result()
                context_receipts[dex] = _iso(at)
        with ThreadPoolExecutor(max_workers=min(8, len(self.client.settings.coins))) as pool:
            jobs = {coin: pool.submit(self.client.info, {"type": "l2Book", "coin": coin}) for coin in self.client.settings.coins}
            for coin, job in jobs.items():
                books[coin] = job.result()
        captured = _utc(self.client.clock())
        raw = {"history_request": query, "history": history, "history_received_at": receipts, "contexts": contexts,
               "contexts_received_at": context_receipts, "books": books, "captured_at": _iso(captured)}
        observation = derive_observation(raw, self.client.settings.coins, self.client.settings.maximum_book_age_seconds)
        fingerprint = _hash([{k: v for k, v in point.items() if k != "available_at"} for point in observation["rates"]])
        near_hour = any(min(abs((_stamp(p["at"]) - _hour(_stamp(p["at"]))).total_seconds()),
                            abs((_stamp(p["at"]) - (_hour(_stamp(p["at"])) + timedelta(hours=1))).total_seconds())) < 30 for p in observation["oracles"])
        if not publish:
            return None, fingerprint, near_hour
        output = self.root / f"capture_{captured.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:8]}"
        self.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".funding-staging-", dir=self.root))
        try:
            write_json(staging / "raw.json", raw)
            shutil.copyfile(self.config, staging / "input_config.toml")
            report = {"schema_version": 2, "mode": "observed_funding_data_only", "exchange": "hyperliquid", "currency": "USDC",
                      "created_at": raw["captured_at"], "coins": list(self.client.settings.coins), "input_config_sha256": cfg_hash,
                      "artifact_hashes": {name: file_digest(staging / name) for name in ("raw.json", "input_config.toml")},
                      "observation": observation, "orders_created": False, "paper_fills_created": False, "forward_strategy_started": False}
            if report["artifact_hashes"]["input_config.toml"] != cfg_hash:
                raise ValueError("Funding configuration changed during capture")
            write_json(staging / "capture.json", report)
            expected_keys = {coin + "|" + _iso(_hour(captured) - timedelta(hours=index))
                             for coin in self.client.settings.coins for index in range(24)}
            with _publication_lock(self.root):
                ready, unavailable = pair_recorded_funding(self.root, expected_keys, extra_paths=(staging,))
                final_now = _utc(self.client.clock())
                for coin in self.client.settings.coins:
                    parse_book(books[coin], coin, final_now, self.client.settings.maximum_book_age_seconds)
                if file_digest(self.config) != cfg_hash or report["artifact_hashes"]["input_config.toml"] != cfg_hash:
                    raise ValueError("Funding configuration changed during capture")
                os.rename(staging, output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        print("Hyperliquid settled-funding collector (public mainnet; oracle proxies explicitly estimated):")
        for coin, summary in observation["assets"].items():
            print(f"{coin}: settled rates {summary['rates']}; missing rate hours {len(summary['missing_rate_hours'])}")
        print(f"Last 24 hours: accounting-ready recorded settlements {len(ready)}; rates lacking a near-settlement oracle {len(unavailable)}")
        print("PASS: settled rates and observed oracle evidence saved and audited; missing historical oracles remain unavailable")
        print("Data collector only: no paper fills or forward strategy trial")
        print(f"Saved funding capture to {output}")
        return output, fingerprint, near_hour


def run_funding_collector(config, *, watch=False):
    settings = load_hyperliquid_settings(config)
    collector = FundingCollector(HyperliquidPublicClient(settings), config)
    last_signature, last_capture = None, None
    try:
        while True:
            if load_hyperliquid_settings(config) != settings:
                raise ValueError("Funding configuration changed; restart to apply it")
            now = _utc(collector.client.clock())
            seconds = (now - _hour(now)).total_seconds()
            near = seconds < 30 or seconds > 3570
            due = last_capture is None or near or (now - last_capture).total_seconds() >= 60
            if due:
                try:
                    # Away from the hour, avoid duplicate full-response bundles.
                    # The dry capture populates cache; publish uses that same
                    # settled history with a freshly obtained context/book.
                    if watch and last_signature is not None and not near:
                        _, signature, _ = collector.capture(publish=False)
                        if signature == last_signature:
                            last_capture = now
                            time.sleep(settings.poll_seconds)
                            continue
                    _, last_signature, _ = collector.capture()
                    last_capture = now
                except (ValueError, RuntimeError, OSError) as error:
                    if not watch:
                        raise
                    LOGGER.error("Funding capture refused: %s", error)
            if not watch:
                return
            time.sleep(settings.poll_seconds)
    except KeyboardInterrupt:
        print("Hyperliquid funding collector stopped")
