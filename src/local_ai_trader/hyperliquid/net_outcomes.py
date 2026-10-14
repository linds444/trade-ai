"""Prospective hypothetical trade outcomes from archived public quotes; no fills.

Each forecast is a separate 1,000-USDC cost benchmark, not a portfolio. No
reliability certificate, liquidation path simulation or strategy P&L is produced.
"""

from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import logging
import math
from pathlib import Path
import sqlite3
import time

from local_ai_trader.data.storage import native_path, write_json
from local_ai_trader.hyperliquid.diagnostics import read_forecasts
from local_ai_trader.hyperliquid.feed import load_hyperliquid_settings, parse_contexts
from local_ai_trader.hyperliquid.forward import _check_prediction, _time
from local_ai_trader.hyperliquid.funding import _publication_lock, load_funding_capture, pair_recorded_funding
from local_ai_trader.hyperliquid.guardian import derive_quotes, load_capture
from local_ai_trader.hyperliquid.markets import BY_COIN
from local_ai_trader.hyperliquid.venue import floor_size, number, round_price
from local_ai_trader.models.train import file_digest
from local_ai_trader.paper.isolated import decode_rules
from local_ai_trader.paper.ledger import _hash, _iso, _json
from local_ai_trader.paper.perpetual import RISK_LIMITS
from local_ai_trader.risk.paper import _utc

LOGGER = logging.getLogger(__name__)
PROTOCOL = {"version": 1, "budget_usdc": 1000, "slippage_bps_per_side": 1,
            "entry_delay_seconds": 300, "holding_seconds": 1800, "quote_window_seconds": 30,
            "minimum_net_margin": RISK_LIMITS["minimum_net_margin"],
            "selection": "first timely valid guardian sample observed after deadline; no retrospective quote substitution",
            "funding": "actual settled event while held; near-event observed oracle proxy explicitly estimated",
            "success": "strictly positive net PnL after spread, native rounding/slippage, fees and settled funding",
            "scope": "independent hypothetical forecast benchmarks; no portfolio, liquidation-path or reliability certification"}
SCHEMA = """
CREATE TABLE contract(id INTEGER PRIMARY KEY CHECK(id=1),payload TEXT NOT NULL,digest TEXT NOT NULL,
 version INTEGER NOT NULL,head TEXT NOT NULL);
CREATE TABLE events(sequence INTEGER PRIMARY KEY,key TEXT NOT NULL,kind TEXT NOT NULL,
 payload TEXT NOT NULL,previous TEXT NOT NULL,digest TEXT NOT NULL,UNIQUE(key,kind));
"""


def funding_keys(coin, entry_at, exit_at):
    """First-30-second entries need the event to establish whether funding is owed."""
    hour = entry_at.replace(minute=0, second=0, microsecond=0)
    result = set()
    while hour <= exit_at:
        if entry_at < hour + timedelta(seconds=30):
            result.add(coin + "|" + _iso(hour))
        hour += timedelta(hours=1)
    return result


def validate_funding(point, now):
    """Reconstruct frozen settlement operands from their original archived sources."""
    sources = {}
    for name in point["source_captures"]:
        observation, identity = load_funding_capture(Path(name))
        sources[identity] = observation
    event, hour = _time(point["exchange_event_at"]), _time(point["hour"])
    if (point["source_mode"] != "observed" or hour != hour.replace(minute=0, second=0, microsecond=0)
            or not 0 <= (event-hour).total_seconds() < 30
            or not event <= _time(point["rate_observed_at"]) <= now
            or not _time(point["oracle_at"]) <= now
            or abs((_time(point["oracle_at"])-event).total_seconds()) >= 30):
        raise ValueError("Invalid net-outcome funding timing")
    rates = sources[point["rate_source_id"]]["rates"]
    oracles = sources[point["oracle_source_id"]]["oracles"]
    if not any(all(row[name] == point[name] for name in ("coin", "hour", "exchange_event_at", "rate"))
               and row["available_at"] == point["rate_observed_at"] for row in rates):
        raise ValueError("Net-outcome settled rate differs from archived source")
    if not any(row["coin"] == point["coin"] and row["oracle_price"] == point["oracle_price"]
               and row["at"] == point["oracle_at"] for row in oracles):
        raise ValueError("Net-outcome oracle differs from archived source")


def calculate_outcome(record, entry, exit, funding, minimum_probability, resolved_at):
    quantity = entry["quantity"]
    a, b = entry["quote"], exit["quote"]
    entry_mid, exit_mid = (a["bid"]+a["ask"])/2, (b["bid"]+b["ask"])/2
    mid_pnl = quantity * (exit_mid-entry_mid)
    spread = quantity * ((a["ask"]-entry_mid)+(exit_mid-b["bid"]))
    impact = quantity * ((entry["fill_price"]-a["ask"])+(b["bid"]-exit["fill_price"]))
    fees = quantity * (entry["fill_price"]*a["fee_bps"]+exit["fill_price"]*b["fee_bps"])/10000
    flow = math.fsum(-quantity*p["oracle_price"]*p["rate"] for p in funding
                     if _time(entry["at"]) < _time(p["exchange_event_at"]) <= _time(exit["at"]))
    gross = quantity * (exit["fill_price"]-entry["fill_price"])
    net = math.fsum((gross, -fees, flow))
    if not math.isclose(net, math.fsum((mid_pnl, -spread, -impact, -fees, flow)), rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError("Net-outcome cost components do not reconcile")
    return {"resolved_at": _iso(resolved_at), "mid_price_pnl": mid_pnl, "spread_cost": spread, "slippage_and_rounding_cost": impact,
            "fill_price_gross_pnl": gross, "fees": fees, "funding_cash_flow": flow,
            "net_pnl": net, "net_return": net/(quantity*entry["fill_price"]), "success": net > 0,
            "probability_candidate": record["probabilities"][0] >= minimum_probability,
            "cost_screen_candidate": record["probabilities"][0] >= minimum_probability and entry["forecast_net_return"] > RISK_LIMITS["minimum_net_margin"],
            "funding_oracle_estimated": bool(funding), "funding": funding}


class NetOutcomeObserver:
    def __init__(self, config, models, book, guardian, *, clock=None):
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.config, self.models, self.book, self.guardian = [native_path(Path(p)).resolve() for p in (config, models, book, guardian)]
        self.settings = load_hyperliquid_settings(self.config)
        self.source_contract, _ = read_forecasts(self.book, signal_since=_utc(self.clock()))
        manifest = json.loads((self.models/"models.json").read_text(encoding="utf-8"))
        self.manifest = manifest
        self.run = json.loads((self.guardian/"run.json").read_text(encoding="utf-8"))
        if (self.source_contract.get("mode") != "prospective_forecast_observations"
                or self.source_contract.get("model_manifest_sha256") != file_digest(self.models/"models.json")
                or self.source_contract.get("config_sha256") != file_digest(self.config)
                or self.source_contract.get("coins") != list(self.settings.coins)
                or any(self.source_contract.get(k) != v for k, v in (("storage_schema", 2), ("entry_delay_bars", 1),
                       ("holding_bars", 6), ("label_known_delay_bars", 8), ("class_threshold", .002)))
                or manifest.get("coins") != list(self.settings.coins)
                or manifest.get("source_config_sha256") != file_digest(self.config)
                or manifest.get("test_opened") is not False or manifest.get("live_execution_enabled") is not False
                or self.run.get("mode") != "hyperliquid_isolated_guardian"
                or self.run.get("source_mode") != "observed" or self.run.get("entries_enabled") is not False
                or self.run.get("profitability_trial_started") is not False
                or self.run.get("input_config_sha256") != file_digest(self.config)
                or file_digest(self.guardian/"input_config.toml") != file_digest(self.config)
                or self.run.get("coins") != list(self.settings.coins)):
            raise ValueError("Net observer requires matching frozen observed models, forward book and paused guardian")
        bootstrap = load_capture(self.guardian/"bootstrap.json.gz", self.settings, self.run["bootstrap_sha256"])
        self.rules = {coin: row["venue_rules"] for coin, row in bootstrap["markets"].items() if row["status"] == "READY"}
        if set(self.rules) != set(self.settings.coins):
            raise ValueError("Net observer bootstrap lacks market rules")
        binding = {"mode": "prospective_hypothetical_net_outcomes", "schema_version": 1, "source_mode": "observed",
                   "protocol": PROTOCOL, "config_sha256": file_digest(self.config), "source_contract_sha256": _hash(self.source_contract),
                   "guardian_run_sha256": _hash(self.run), "models": str(self.models), "book": str(self.book),
                   "guardian": str(self.guardian), "minimum_probability": self.settings.entry_policy.minimum_probability,
                   "minimum_confidence": self.settings.entry_policy.minimum_confidence}
        parent = self.settings.observations.parent/"net_outcomes"
        self.root = parent/_hash(binding)[:16]
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root/"observations.sqlite"
        with _publication_lock(self.root, label="net-outcome observer"):
            if not self.path.exists():
                if (self.root/"run.json").exists():
                    raise ValueError("Net-outcome journal is missing; refusing to reset")
                now = _utc(self.clock())
                contract = {**binding, "created_at": _iso(now),
                            "first_signal_at": _iso(datetime.fromtimestamp((int(now.timestamp())//300+1)*300, timezone.utc))}
                with closing(sqlite3.connect(str(self.path))) as db:
                    db.executescript(SCHEMA)
                    db.execute("INSERT INTO contract VALUES(1,?,?,0,?)", (_json(contract), _hash(contract), _hash(contract)))
                    db.commit()
                write_json(self.root/"run.json", contract)
            with closing(sqlite3.connect(str(self.path))) as db:
                payload, digest, self.version, self.head = db.execute("SELECT payload,digest,version,head FROM contract").fetchone()
                self.contract = json.loads(payload)
                if digest != _hash(self.contract) or {k: self.contract.get(k) for k in binding} != binding:
                    raise ValueError("Net-outcome run binding changed")
                if json.loads((self.root/"run.json").read_text(encoding="utf-8")) != self.contract:
                    raise ValueError("Net-outcome run witness changed")
                events = db.execute("SELECT sequence,key,kind,payload,previous,digest FROM events ORDER BY sequence").fetchall()
            self.records = {}
            head = _hash(self.contract)
            for expected_sequence, (sequence, key, kind, payload, previous, digest) in enumerate(events, 1):
                value = json.loads(payload)
                expected = _hash({"sequence": sequence, "key": key, "kind": kind, "payload": value, "previous": head})
                if previous != head or digest != expected or sequence != expected_sequence:
                    raise ValueError("Net-outcome event journal changed")
                self._validate(key, kind, value)
                self.records.setdefault(key, {})[kind] = value
                head = digest
            if len(events) != self.version or head != self.head:
                raise ValueError("Net-outcome journal rows were removed or changed")

    def _capture(self, proof):
        path = (self.guardian/Path(proof["capture"])).resolve()
        if not path.is_relative_to(self.guardian/"captures"):
            raise ValueError("Net-outcome quote evidence is outside the guardian run")
        payload = load_capture(path, self.settings, proof["sha256"])
        at, observed = _time(payload["captured_at"]), _time(proof["observed_at"])
        if not 0 <= (observed-at).total_seconds() < PROTOCOL["quote_window_seconds"]:
            raise ValueError("Net-outcome quote was stale or future-dated when observed")
        return payload, derive_quotes(payload, self.settings, at, self.rules), at

    def _execution(self, record, proof, entry=None):
        payload, markets, at = self._capture(proof)
        coin = record["coin"]
        due = _time(record["signal_at"])+timedelta(seconds=300) if entry is None else _time(entry["at"])+timedelta(seconds=1800)
        if not due <= at <= _time(proof["observed_at"]) < due+timedelta(seconds=30):
            raise ValueError("Net-outcome quote missed its prospective execution window")
        market = markets[coin]
        if market["status"] != "READY":
            return None
        quote = market["quote"]
        if _time(quote["at"]) < due:
            return None
        rules, side = decode_rules(self.rules[coin]), "BUY" if entry is None else "SELL"
        price = float(round_price(number(quote["ask"] if entry is None else quote["bid"]) * number(1.0001 if entry is None else .9999), rules, side=side))
        result = {"at": _iso(at), "proof": proof, "quote": quote, "fill_price": price}
        if entry is not None:
            if market["bid_size"] < entry["quantity"]:
                return {"unavailable": "insufficient_full_size_exit_liquidity"}
            return result
        quantity = float(floor_size(number(PROTOCOL["budget_usdc"])/(number(price)*(1+number(quote["fee_bps"])/10000)), rules))
        if quantity <= 0 or quantity*price < 10 or market["ask_size"] < quantity:
            return {"unavailable": "entry_size_or_full_size_liquidity_unavailable"}
        dex = BY_COIN[coin].dex
        key = f"{dex}_contexts" if dex else "contexts"
        context = parse_contexts(payload["responses"][key], (coin,), _time(payload["context_receipts"][key]))[coin]
        budget = max(0., context["hourly_funding_rate"])*context["oracle_price"]/price
        estimated_exit = float(round_price(number(quote["bid"])*number(.9999)*(1+number(record["ridge_return"])), rules, side="SELL"))
        modeled_net = (estimated_exit*(1-quote["fee_bps"]/10000)-price*(1+quote["fee_bps"]/10000))/price-budget
        return {**result, "quantity": quantity, "forecast_net_return": modeled_net, "funding_cost_budget": budget}

    def _validate(self, key, kind, value):
        state = self.records.get(key, {})
        if kind in state or "outcome" in state or "unavailable" in state:
            raise ValueError("Duplicate or terminal net-outcome observation changed")
        if kind == "forecast":
            record, enrolled = value["forecast"], _time(value["enrolled_at"])
            signal = _check_prediction(record, self.source_contract)
            expected_id = "sha256:"+self.manifest["artifact_hashes"][BY_COIN[record["coin"]].filename+"/checkpoint.json"]
            if (key != record["coin"]+"|"+_iso(signal) or record["calibration_id"] != expected_id
                    or not _time(self.contract["first_signal_at"]) <= signal <= _time(record["published_at"]) <= enrolled < signal+timedelta(minutes=5)):
                raise ValueError("Net-outcome forecast was not registered prospectively")
        elif "forecast" not in state:
            raise ValueError("Net outcome lacks a prospective forecast")
        elif kind in ("entry", "exit"):
            if (kind == "entry" and "entry" in state) or (kind == "exit" and "entry" not in state):
                raise ValueError("Net-outcome execution sequence disagrees")
            expected = self._execution(state["forecast"]["forecast"], value["proof"], state.get("entry") if kind == "exit" else None)
            if expected != value:
                raise ValueError("Net-outcome execution differs from archived source")
        elif kind == "outcome":
            if "exit" not in state:
                raise ValueError("Net outcome lacks an observed exit")
            forecast, entry, exit = state["forecast"]["forecast"], state["entry"], state["exit"]
            required = funding_keys(forecast["coin"], _time(entry["at"]), _time(exit["at"]))
            points = value["funding"]
            resolved_at = _time(value["resolved_at"])
            if not _time(exit["at"]) <= resolved_at <= _utc(self.clock()):
                raise ValueError("Invalid net-outcome resolution time")
            if {p["coin"]+"|"+p["hour"] for p in points} != required or len(points) != len(required):
                raise ValueError("Net outcome has incomplete funding operands")
            for point in points:
                validate_funding(point, resolved_at)
            if value != calculate_outcome(forecast, entry, exit, points, self.contract["minimum_probability"], resolved_at):
                raise ValueError("Net outcome differs from observed costs and funding")
        elif kind != "unavailable" or not value.get("reason"):
            raise ValueError("Unsupported net-outcome event")

    def _append(self, key, kind, value):
        self._validate(key, kind, value)
        sequence = self.version+1
        digest = _hash({"sequence": sequence, "key": key, "kind": kind, "payload": value, "previous": self.head})
        with closing(sqlite3.connect(str(self.path))) as db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT digest,version,head FROM contract").fetchone() != (_hash(self.contract), self.version, self.head):
                raise ValueError("Net-outcome observer journal changed concurrently")
            db.execute("INSERT INTO events VALUES(?,?,?,?,?,?)", (sequence, key, kind, _json(value), self.head, digest))
            db.execute("UPDATE contract SET version=?,head=?", (sequence, digest))
            if kind in ("forecast", "entry", "exit"):
                forecast = value["forecast"] if kind == "forecast" else self.records[key]["forecast"]["forecast"]
                due = _time(forecast["signal_at"])+timedelta(minutes=5)
                if kind == "exit":
                    due = _time(self.records[key]["entry"]["at"])+timedelta(minutes=30)
                deadline = due if kind == "forecast" else due+timedelta(seconds=30)
                if _utc(self.clock()) >= deadline:
                    raise ValueError("Net-outcome prospective event expired before commit")
            db.commit()
        self.records.setdefault(key, {})[kind] = value
        self.version, self.head = sequence, digest

    def tick(self):
        now = _utc(self.clock())
        if file_digest(self.config) != self.contract["config_sha256"] or file_digest(self.models/"models.json") != self.source_contract["model_manifest_sha256"] or _hash(json.loads((self.guardian/"run.json").read_text(encoding="utf-8"))) != self.contract["guardian_run_sha256"]:
            raise ValueError("Net-outcome source bindings changed")
        contract, forecasts = read_forecasts(self.book, signal_since=max(_time(self.contract["first_signal_at"]), now-timedelta(minutes=5)))
        if contract != self.source_contract:
            raise ValueError("Net-outcome forward source contract changed")
        for record in forecasts:
            now = _utc(self.clock())
            signal = _time(record["signal_at"])
            key = record["coin"]+"|"+_iso(signal)
            if key not in self.records and _time(record["published_at"]) <= now < signal+timedelta(minutes=5):
                self._append(key, "forecast", {"forecast": record, "enrolled_at": _iso(now)})
        proof, quote_error = None, None
        try:
            receipt = json.loads((self.guardian/"latest.json").read_text(encoding="utf-8"))
            checksum = receipt.pop("receipt_sha256")
            if checksum != _hash(receipt) or receipt["run_binding_sha256"] != self.contract["guardian_run_sha256"]:
                raise ValueError("Guardian latest receipt changed")
            path = native_path(Path(receipt["source_capture"])).resolve()
            proof = {"capture": str(path.relative_to(self.guardian)), "sha256": receipt["source_sha256"], "observed_at": _iso(_utc(self.clock()))}
            self._capture(proof)
        except (ValueError, OSError, KeyError) as error:
            proof, quote_error = None, str(error)
        for key, state in list(self.records.items()):
            now = _utc(self.clock())
            if "outcome" in state or "unavailable" in state:
                continue
            record = state["forecast"]["forecast"]
            if "exit" not in state:
                kind = "exit" if "entry" in state else "entry"
                due = _time(state["entry"]["at"])+timedelta(minutes=30) if kind == "exit" else _time(record["signal_at"])+timedelta(minutes=5)
                if now >= due+timedelta(seconds=30):
                    self._append(key, "unavailable", {"reason": "missed_"+kind+"_quote_window", "at": _iso(now)})
                    continue
                if now < due or proof is None:
                    continue
                try:
                    execution = self._execution(record, proof, state.get("entry"))
                except ValueError:
                    continue  # A pre-deadline/stale sample cannot stand in for an execution.
                if execution is None:
                    continue
                if "unavailable" in execution:
                    self._append(key, "unavailable", {"reason": execution["unavailable"], "proof": proof, "at": _iso(now)})
                    continue
                self._append(key, kind, execution)
        required = set().union(*(funding_keys(s["forecast"]["forecast"]["coin"], _time(s["entry"]["at"]), _time(s["exit"]["at"]))
                                for s in self.records.values() if "exit" in s and "outcome" not in s and "unavailable" not in s))
        ready, _ = pair_recorded_funding(self.settings.observations.parent/"funding", required, include_provenance=True) if required else ({}, [])
        for key, state in list(self.records.items()):
            if "exit" not in state or "outcome" in state or "unavailable" in state:
                continue
            record, entry, exit = state["forecast"]["forecast"], state["entry"], state["exit"]
            keys = funding_keys(record["coin"], _time(entry["at"]), _time(exit["at"]))
            if keys <= ready.keys():
                result = calculate_outcome(record, entry, exit, [ready[k] for k in sorted(keys)], self.contract["minimum_probability"], _utc(self.clock()))
                self._append(key, "outcome", result)
        summary = self.summary()
        summary.update(observed_at=_iso(now), quote_error=quote_error)
        write_json(self.root/"latest.json", summary)
        return summary

    def summary(self):
        states = list(self.records.values())
        outcomes = [s["outcome"] for s in states if "outcome" in s]
        candidates = [p for p in outcomes if p["cost_screen_candidate"]]
        return {"registered": len(states), "entered_benchmarks": sum("entry" in s for s in states),
                "resolved": len(outcomes), "unavailable": sum("unavailable" in s for s in states),
                "pending_quotes": sum("exit" not in s and "unavailable" not in s for s in states),
                "pending_funding": sum("exit" in s and "outcome" not in s for s in states),
                "probability_candidates_registered": sum(s["forecast"]["forecast"]["probabilities"][0] >= self.contract["minimum_probability"] for s in states),
                "cost_screen_candidates_resolved": len(candidates), "cost_screen_candidate_successes": sum(p["success"] for p in candidates),
                "confidence": None, "reliability_validated": False, "profitability_trial_started": False,
                "paper_fills_created": False, "live_execution_enabled": False}


def run_net_outcomes(config, models, book, guardian, *, watch=False):
    # A separate observer process never owns or writes a paper account.
    observer = NetOutcomeObserver(config, models, book, guardian)
    print("Prospective net-outcome observer: archived public quotes; hypothetical 1000-USDC benchmarks; no orders")
    print(f"First eligible future signal: {observer.contract['first_signal_at']}")
    print(f"Saved net-outcome journal to {observer.path}")
    try:
        with _publication_lock(observer.root, label="net-outcome observer"):
            last_print, previous = 0., None
            while True:
                started = time.monotonic()
                try:
                    result = observer.tick()
                    signature = _hash({k: v for k, v in result.items() if k != "observed_at"})
                    if not watch or signature != previous or started-last_print >= 60:
                        print(f"Registered {result['registered']}; resolved {result['resolved']}; pending quotes {result['pending_quotes']}; pending funding {result['pending_funding']}; unavailable {result['unavailable']}")
                        print(f"Cost-screen candidates: {result['cost_screen_candidates_resolved']} resolved; net-positive {result['cost_screen_candidate_successes']}; confidence unavailable")
                        if result["quote_error"]:
                            print(f"Quote evidence unavailable: {result['quote_error']}")
                        print("Overlapping hypothetical outcomes are not independent reliability samples or paper P&L; strategy month not started")
                        last_print, previous = started, signature
                    if not watch:
                        return result
                except (ValueError, OSError, KeyError, sqlite3.Error) as error:
                    if not watch:
                        raise
                    LOGGER.error("Net-outcome observation refused: %s", error)
                time.sleep(max(.1, 5-(time.monotonic()-started)))
    except KeyboardInterrupt:
        print("Hyperliquid net-outcome observer stopped")
