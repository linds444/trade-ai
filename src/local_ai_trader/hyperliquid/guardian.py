"""Continuous, public-quote isolated paper maintenance, independent of inference.

This pretrial guardian never opens positions or validates model reliability.
It maintains a separate, entry-paused schema-2 book. Source failures stay
unavailable, and only fresh source-backed quotes can produce reducing fills.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime
import gzip
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
from uuid import uuid4

from local_ai_trader.data.storage import native_path, write_json
from local_ai_trader.hyperliquid.feed import HyperliquidPublicClient, load_hyperliquid_settings, parse_book, parse_contexts
from local_ai_trader.hyperliquid.funding import _publication_lock, apply_recorded_funding
from local_ai_trader.hyperliquid.markets import BY_COIN
from local_ai_trader.hyperliquid.venue import parse_rules, validate_price
from local_ai_trader.models.train import file_digest
from local_ai_trader.paper.isolated import MODEL, encode_rules, position_risk, rules_for
from local_ai_trader.paper.ledger import _hash, _json
from local_ai_trader.paper.perpetual import PerpetualPaperAccount, RISK_LIMITS
from local_ai_trader.risk.paper import _utc

LOGGER = logging.getLogger(__name__)
ERRORS = (ValueError, RuntimeError, OSError, KeyError, TypeError)
SLIPPAGE_BPS = 1.0


def derive_quotes(payload, settings, at, bound_rules=None):
    """Reconstruct each market independently from captured public evidence."""
    markets = {}
    for coin in settings.coins:
        try:
            dex = BY_COIN[coin].dex
            key = f"{dex}_contexts" if dex else "contexts"
            for source in (key, f"{coin}_book"):
                if source in payload["errors"]:
                    raise ValueError(payload["errors"][source])
            raw = payload["responses"][key]
            received = _utc(datetime.fromisoformat(payload["context_receipts"][key]))
            if not 0 <= (at-received).total_seconds() < settings.maximum_book_age_seconds:
                raise ValueError("Guardian mark context is future-dated or stale")
            context = parse_contexts(raw, (coin,), received)[coin]
            rules = parse_rules(raw[0], coin)
            encoded = encode_rules(rules)
            if bound_rules is not None and encoded != bound_rules[coin]:
                raise ValueError("Venue rules changed from the frozen isolated account; review required")
            book = parse_book(payload["responses"][f"{coin}_book"], coin, at, settings.maximum_book_age_seconds)
            validate_price(book["bid"], rules)
            validate_price(book["ask"], rules)
            multiplier = 1.0
            if dex:
                scale = context["deployer_fee_scale"]
                multiplier = (1+scale) if scale < 1 else 2*scale
                multiplier *= .1 if context["growth_mode"] == "enabled" else 1
            quote = {"coin": coin, "at": min(book["exchange_time"], received.isoformat()),
                     "bid": book["bid"], "ask": book["ask"], "mark": context["mark_price"],
                     "fee_bps": settings.taker_fee_bps * multiplier, "size_decimals": rules.size_decimals, "source_mode": "observed"}
            markets[coin] = {"status": "READY", "quote": quote, "bid_size": book["bid_size"],
                             "ask_size": book["ask_size"], "venue_rules": encoded}
        except ERRORS as error:
            markets[coin] = {"status": "UNAVAILABLE", "reason": str(error)[:800]}
    return markets


def capture_quotes(client):
    """No candles or model reads; one concurrent batch of bounded public reads."""
    requests = {}
    for dex in dict.fromkeys(BY_COIN[c].dex for c in client.settings.coins):
        requests[f"{dex}_contexts" if dex else "contexts"] = {"type": "metaAndAssetCtxs", **({"dex": dex} if dex else {})}
    requests.update({f"{coin}_book": {"type": "l2Book", "coin": coin} for coin in client.settings.coins})
    payload = {"schema_version": 1, "mode": "hyperliquid_guardian_quotes", "source_mode": "observed",
               "responses": {}, "context_receipts": {}, "errors": {}, "live_execution_enabled": False}
    def fetch(request):
        response = client.info(request)
        return response, _utc(client.clock()).isoformat()
    with ThreadPoolExecutor(max_workers=len(requests)) as pool:
        jobs = {key: pool.submit(fetch, request) for key, request in requests.items()}
        for key, job in jobs.items():
            try:
                response, received = job.result()
                payload["responses"][key] = response
                if key.endswith("contexts"):
                    payload["context_receipts"][key] = received
            except ERRORS as error:
                payload["errors"][key] = str(error)[:800]
    at = _utc(client.clock())
    payload["captured_at"] = at.isoformat()
    payload["markets"] = derive_quotes(payload, client.settings, at)
    return payload


def _write_capture(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(gzip.compress(_json(payload).encode("utf-8"), mtime=0))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_capture(path, settings, digest):
    path = native_path(Path(path)).resolve()
    if file_digest(path) != digest:
        raise ValueError("Guardian source archive changed")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema_version") != 1 or payload.get("mode") != "hyperliquid_guardian_quotes" or payload.get("source_mode") != "observed" or payload.get("live_execution_enabled") is not False:
        raise ValueError("Unsupported guardian source archive")
    expected = derive_quotes(payload, settings, _utc(datetime.fromisoformat(payload["captured_at"])))
    if expected != payload["markets"]:
        raise ValueError("Guardian archived quotes disagree with source evidence")
    return payload


class PaperGuardian:
    def __init__(self, config, *, client=None):
        self.config = native_path(Path(config)).resolve()
        self.settings = load_hyperliquid_settings(self.config)
        # A failing asset must not hold the batch through retries. Retry on the
        # next tick; each HTTP request has at most a five-second timeout.
        runtime = replace(self.settings, max_attempts=1, timeout_seconds=min(5, self.settings.timeout_seconds), poll_seconds=min(10, self.settings.poll_seconds))
        if client is not None and client.settings not in (self.settings, runtime):
            raise ValueError("Guardian client/config disagree")
        self.client = client or HyperliquidPublicClient(runtime)
        # Account location must survive edits to the observation-output path.
        # Otherwise a changed config could silently create an empty book elsewhere.
        parent = native_path(self.config.parent.parent/"data"/"hyperliquid"/"paper").resolve()
        parent.mkdir(parents=True, exist_ok=True)
        identity = _hash({"purpose": "hyperliquid_isolated_guardian_v1", "config_path": str(self.config)})[:16]
        self.root = parent/f"guardian_{identity}"
        witness = parent/(".guardian-identity-"+identity+".json")
        with _publication_lock(parent, label="paper guardian bootstrap"):
            if witness.exists() and not self.root.exists():
                raise ValueError("Previously created guardian book is missing; refusing to reset")
            if not self.root.exists():
                self._create(parent)
            if not (self.root/"account.sqlite").is_file():
                raise ValueError("Guardian account is missing; refusing to reset the book")
            self.account = PerpetualPaperAccount(self.root/"account.sqlite")  # full replay once at startup
            self.contract_hash = _hash(self.account.contract())
            self.binding = self.account.contract().get("run_binding")
            self._validate_run()
            if file_digest(self.config) != self.binding["input_config_sha256"]:
                raise ValueError("Guardian config changed; existing book was retained, not reset")
            bootstrap = load_capture(self.root/"bootstrap.json.gz", self.settings, self.binding["bootstrap_sha256"])
            self.bound_rules = {coin: row["venue_rules"] for coin, row in bootstrap["markets"].items() if row["status"] == "READY"}
            if self.account.contract()["isolated_venue_rules"] != self.bound_rules:
                raise ValueError("Guardian bootstrap rules disagree with the isolated ledger")
            expected = {"root": self.root.name, "run_binding_sha256": _hash(self.binding)}
            if witness.exists() and json.loads(witness.read_text(encoding="utf-8")) != expected:
                raise ValueError("Guardian identity witness changed")
            if not witness.exists():
                write_json(witness, expected)

    def _create(self, parent):
        digest = file_digest(self.config)
        initial = capture_quotes(self.client)
        if any(row["status"] != "READY" for row in initial["markets"].values()):
            reasons = {c: r["reason"] for c, r in initial["markets"].items() if r["status"] != "READY"}
            raise ValueError(f"Guardian bootstrap requires all configured venue rules: {reasons}")
        staging = Path(tempfile.mkdtemp(prefix=".guardian-staging-", dir=parent))
        try:
            _write_capture(staging/"bootstrap.json.gz", initial)
            (staging/"input_config.toml").write_bytes(self.config.read_bytes())
            binding = {"schema_version": 1, "mode": "hyperliquid_isolated_guardian", "source_mode": "observed",
                       "input_config_sha256": digest, "coins": list(self.settings.coins), "initial_collateral": 10000.0,
                       "entry_policy": asdict(self.settings.entry_policy), "risk": dict(RISK_LIMITS),
                       "slippage_bps": SLIPPAGE_BPS, "isolated_model": MODEL, "entries_enabled": False,
                       "profitability_trial_started": False, "bootstrap_sha256": file_digest(staging/"bootstrap.json.gz")}
            rules = {coin: parse_rules(initial["responses"][f"{BY_COIN[coin].dex}_contexts" if BY_COIN[coin].dex else "contexts"][0], coin) for coin in self.settings.coins}
            PerpetualPaperAccount.create(staging/"account.sqlite", coins=self.settings.coins,
                policy=self.settings.entry_policy, slippage_bps=SLIPPAGE_BPS, source_mode="observed",
                now=_utc(self.client.clock()), run_binding=binding, entries_enabled=False, venue_rules=rules)
            write_json(staging/"run.json", binding)
            if file_digest(self.config) != digest or file_digest(staging/"input_config.toml") != digest:
                raise ValueError("Guardian configuration changed during bootstrap")
            if any(row["status"] != "READY" for row in derive_quotes(initial, self.settings, _utc(self.client.clock())).values()):
                raise ValueError("Guardian bootstrap evidence expired before publication")
            os.rename(staging, self.root)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def _validate_run(self):
        binding = json.loads((self.root/"run.json").read_text(encoding="utf-8"))
        contract = self.account.contract()
        if (_hash(contract) != self.contract_hash or not isinstance(self.binding, dict) or binding != self.binding or contract.get("run_binding") != binding
                or binding.get("mode") != "hyperliquid_isolated_guardian" or binding.get("source_mode") != "observed"
                or binding.get("entries_enabled") is not False or binding.get("profitability_trial_started") is not False
                or binding.get("input_config_sha256") != file_digest(self.root/"input_config.toml")
                or binding.get("coins") != list(self.settings.coins) or binding.get("entry_policy") != asdict(self.settings.entry_policy)
                or binding.get("risk") != RISK_LIMITS or binding.get("isolated_model") != MODEL
                or contract.get("schema_version") != 2 or contract.get("source_mode") != "observed"
                or contract.get("initial_entries_enabled") is not False or contract["coins"] != binding["coins"]
                or contract["entry_policy"] != binding["entry_policy"] or contract["slippage_bps"] != SLIPPAGE_BPS
                or contract["initial_collateral"] != 10000 or binding.get("initial_collateral") != 10000):
            raise ValueError("Guardian account/run binding changed")

    def tick(self):
        """Caller holds the run lock; finish pending publication, then poll afresh."""
        self._validate_run()
        pending = self.root/"pending.json"
        if pending.exists():
            saved = self._pending_marker(pending)
            if saved is not None:
                self._maintain(saved)
                pending.unlink()
        payload = capture_quotes(self.client)
        at = _utc(datetime.fromisoformat(payload["captured_at"]))
        name = f"tick_{at.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:8]}"
        path = self.root/"captures"/at.strftime("%Y%m%d")/(name+".json.gz")
        _write_capture(path, payload)
        saved = {"capture": str(path.relative_to(self.root)), "sha256": file_digest(path)}
        write_json(pending, saved)  # durable source precedes ledger mutations
        result = self._maintain(saved)
        pending.unlink()
        return result

    def _pending_marker(self, pending):
        """Keep malformed queue bytes; the audited ledger remains financial truth."""
        raw = pending.read_bytes()
        try:
            saved = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            # A lost pointer cannot establish the original source hash/operation
            # ID. Never guess an archive or execute its quotes retrospectively.
            # Committed operations already live in the journal; remaining
            # maintenance can use a new source capture after a full replay audit.
            status = self.account.audit()
            at = _utc(self.client.clock())
            recovery = self.root/"recovery"
            recovery.mkdir(exist_ok=True)
            backup = recovery/f"pending_{at.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:8]}.invalid.json"
            if pending.read_bytes() != raw:
                raise ValueError("Guardian pending marker changed during recovery")
            record = {"mode": "guardian_malformed_pending_recovery", "observed_at": at.isoformat(),
                      "original": "pending.json", "backup": str(backup.relative_to(self.root)),
                      "original_sha256": file_digest(pending), "reason": str(error),
                      "run_binding_sha256": _hash(self.binding), "ledger_version": status["version"],
                      "action": "retain malformed queue; resume with freshly acquired quotes"}
            # Persist the recovery record before moving the original bytes.
            write_json(backup.with_suffix(".recovery.json"), record)
            os.replace(pending, backup)
            LOGGER.warning("Recovered malformed guardian pending.json; original retained at %s; "
                           "ledger audited at version %s; polling fresh quotes", backup, status["version"])
            return None
        if (not isinstance(saved, dict) or set(saved) != {"capture", "sha256"}
                or not all(isinstance(saved[name], str) and saved[name] for name in ("capture", "sha256"))):
            raise ValueError(f"Guardian pending marker has invalid structure: {pending}")
        return saved

    def _maintain(self, saved):
        relative = Path(saved["capture"])
        path = (self.root/relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(self.root/"captures"):
            raise ValueError("Guardian pending capture is outside this run")
        payload = load_capture(path, self.settings, saved["sha256"])
        receipt = path.with_suffix("").with_suffix(".receipt.json")
        if receipt.exists():
            result = json.loads(receipt.read_text(encoding="utf-8"))
            checksum = result.pop("receipt_sha256", None)
            if _hash(result) != checksum or result["source_sha256"] != saved["sha256"] or result["run_binding_sha256"] != _hash(self.binding):
                raise ValueError("Guardian receipt changed")
            result["receipt_sha256"] = checksum
            write_json(self.root/"latest.json", result)
            return result
        source_sha = saved["sha256"]
        key = _hash(saved)
        errors, actions, funding = {}, [], []
        # Never enable entries, even if another process changes the switch.
        if self.account.status()["entries_enabled"]:
            self.account.execute("guardian-pause:"+key, "pause", {"enabled": False}, clock=self.client.clock)
        markets = derive_quotes(payload, self.settings, _utc(self.client.clock()), self.bound_rules)
        held = self.account.status()["positions"]
        quotes = [r["quote"] for c, r in markets.items() if r["status"] == "READY" and c in held]
        if quotes:
            try:
                self.account.execute("guardian-mark:"+key, "mark", {"quotes": quotes, "source_sha256": source_sha, "source_capture": str(path)}, clock=self.client.clock)
            except ERRORS as error:
                errors["mark"] = str(error)
        # Due exits and known maintenance breaches must not wait for funding IO.
        self._exits(markets, key, source_sha, actions, errors, horizon=True)
        self._exits(markets, key+":before-funding", source_sha, actions, errors, horizon=False)
        if self.account.status()["missing_funding"]:
            try:
                funding, _ = apply_recorded_funding(self.account.path, self.settings.observations.parent/"funding",
                                                    clock=self.client.clock, account=self.account)
            except ERRORS as error:
                errors["funding"] = str(error)
        # Settled funding can itself breach maintenance. Every fill still
        # validates quote freshness at execution and immediately before commit.
        self._exits(markets, key, source_sha, actions, errors, horizon=False)
        current = derive_quotes(payload, self.settings, _utc(self.client.clock()), self.bound_rules)
        status = self.account.status()
        summary_keys = ("wallet_balance", "equity", "available_collateral", "reserved_isolated_collateral", "realized_net_pnl",
                        "unrealized_net_pnl", "fill_count", "backstop_transfers", "missing_funding", "performance_provisional", "version", "entries_enabled", "position_risks", "liquidation_pending")
        result = {"schema_version": 1, "mode": "hyperliquid_paper_guardian_tick", "source_mode": "observed",
                  "observed_at": _utc(self.client.clock()).isoformat(), "source_capture": str(path), "source_sha256": source_sha,
                  "run_binding_sha256": _hash(self.binding), "account_path": str(self.account.path),
                  "markets": {c: {k: r[k] for k in ("status", "reason") if k in r} for c, r in current.items()},
                  "actions": actions, "funding_results": funding, "errors": errors,
                  "account": {k: status[k] for k in summary_keys}, "position_count": len(status["positions"]),
                  "config_changed": file_digest(self.config) != self.binding["input_config_sha256"] if self.config.is_file() else True,
                  "profitability_trial_started": False, "reliability_validated": False, "live_execution_enabled": False}
        result["healthy"] = all(r["status"] == "READY" for r in current.values()) and not errors and not status["missing_funding"] and not status["liquidation_pending"] and not result["config_changed"]
        result["receipt_sha256"] = _hash(result)
        write_json(receipt, result)
        write_json(self.root/"latest.json", result)
        return result

    def _exits(self, markets, key, source_sha, actions, errors, *, horizon):
        now = _utc(self.client.clock())
        for coin, position in self.account.status()["positions"].items():
            row = markets[coin]
            if row["status"] != "READY":
                continue
            due = _utc(datetime.fromisoformat(position["exit_due_at"]))
            if horizon:
                if now < due or _utc(datetime.fromisoformat(row["quote"]["at"])) < due:
                    continue
                kind, inputs = "exit", {"reason": "horizon"}
            else:
                if not position_risk(position, rules_for(self.account.contract(), coin))["maintenance_breached"]:
                    continue
                kind, inputs = "liquidation", {}
            inputs.update(quote=row["quote"], bid_size=row["bid_size"], position_id=position["position_id"], source_sha256=source_sha)
            operation = f"guardian-{kind}:"+_hash({"position_id": position["position_id"], "capture": key})
            try:
                actions.append(self.account.execute(operation, kind, inputs, clock=self.client.clock))
            except ERRORS as error:
                errors[f"{kind}:{coin}"] = str(error)


def run_paper_guardian(config, *, watch=False):
    # Acquire ownership before startup replay or any API request, so accidentally
    # launching a second watcher cannot delay the active guardian's transactions.
    config = native_path(Path(config)).resolve()
    settings = load_hyperliquid_settings(config)
    identity = _hash({"purpose": "hyperliquid_isolated_guardian_v1", "config_path": str(config)})[:16]
    lock_root = config.parent.parent/"data"/"hyperliquid"/"paper"/(".guardian-service-"+identity)
    lock_root.mkdir(parents=True, exist_ok=True)
    with _publication_lock(lock_root, label="paper guardian"):
        return _run_paper_guardian(config, watch=watch)


def _run_paper_guardian(config, *, watch=False):
    guardian = None
    try:
        while guardian is None:
            try:
                guardian = PaperGuardian(config)
            except ERRORS as error:
                if not watch:
                    raise
                LOGGER.error("Paper guardian bootstrap refused: %s", error)
                time.sleep(10)
        print("Hyperliquid continuous paper guardian: public mainnet quotes; 1x isolated, long only; entries paused")
        print("Quote polling is independent of candles and model inference; one HTTP attempt, at most 5s per request")
        print(f"Saved isolated guardian account to {guardian.account.path}")
        last_print, last_signature = 0., None
        with _publication_lock(guardian.root, label="paper guardian"):
            while True:
                started = time.monotonic()
                try:
                    result = guardian.tick()
                    signature = _hash({k: result[k] for k in ("healthy", "markets", "account", "position_count", "config_changed", "errors")})
                    if not watch or signature != last_signature or started-last_print >= 60:
                        state = result["account"]
                        ready = sum(r["status"] == "READY" for r in result["markets"].values())
                        print(f"{'READY' if result['healthy'] else 'DEGRADED'} {ready}/{len(result['markets'])} markets; equity {state['equity']:.6f} USDC; positions {result['position_count']}; fills {state['fill_count']}; backstops {state['backstop_transfers']}; missing funding {len(state['missing_funding'])}")
                        for coin, row in result["markets"].items():
                            if row["status"] != "READY":
                                print(f"  {coin}: UNAVAILABLE; {row['reason']}")
                        if result["errors"]:
                            print(f"  Maintenance errors: {result['errors']}")
                        print("Entries paused: reliability unvalidated; profitability month has not started; live execution disabled")
                        print(f"Latest guardian receipt: {guardian.root/'latest.json'}")
                        last_signature, last_print = signature, started
                except ERRORS as error:
                    if not watch:
                        raise
                    LOGGER.error("Paper guardian tick refused; no stale execution: %s", error)
                if not watch:
                    return result
                time.sleep(max(.1, min(10, guardian.settings.poll_seconds)-(time.monotonic()-started)))
    except KeyboardInterrupt:
        print("Hyperliquid paper guardian stopped")
