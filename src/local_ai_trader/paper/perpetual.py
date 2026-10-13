"""Restartable USDC linear-perpetual accounting; no exchange or signing client.

This is a conservative 1x collateral reservation model, not Hyperliquid's
liquidation engine. Trusted model/source adapters supply validated evidence.
Funding history supplies settled rates; observed oracle proxies are explicitly
estimates. Missing settlements make performance provisional and block entries.
"""

from contextlib import closing, contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import json
import sqlite3

from local_ai_trader.data.storage import native_path
from local_ai_trader.hyperliquid.markets import BY_COIN
from local_ai_trader.paper.ledger import _hash, _iso, _json
from local_ai_trader.risk.entry_policy import EntryPolicy, SignalEvidence, evaluate_entry_signal
from local_ai_trader.risk.paper import _utc

RISK_LIMITS = {"position_fraction": 0.10, "total_fraction": 0.20,
               "drawdown_fraction": 0.10, "daily_loss_fraction": 0.02,
               "minimum_net_margin": 0.002}


def _number(value, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError("Require finite numeric perpetual inputs")
    return float(value)


def _time(value):
    return _utc(datetime.fromisoformat(value))


def _hour(value):
    return value.replace(minute=0, second=0, microsecond=0)


def _initial(contract):
    return {"wallet_balance": contract["initial_collateral"], "peak_equity": contract["initial_collateral"],
            "day_start_equity": contract["initial_collateral"], "day": contract["created_at"][:10],
            "as_of": contract["created_at"], "version": 0, "entries_enabled": True,
            "positions": {}, "trades": [], "settlements": {}}


def _gaps(state, now):
    missing = set()
    for position in list(state["positions"].values()) + state["trades"]:
        end = min(now, _time(position.get("exit_at", _iso(now))))
        hour = _hour(_time(position["entry_at"])) + timedelta(hours=1)
        while hour <= end:
            key = position["coin"] + "|" + _iso(hour)
            if key not in state["settlements"]:
                missing.add(key)
            hour += timedelta(hours=1)
    return sorted(missing)


def _totals(state):
    positions = state["positions"].values()
    unrealized = math.fsum(p["quantity"] * (p["mark_price"] - p["entry_fill_price"]) for p in positions)
    margin = math.fsum(p["initial_margin"] for p in positions)
    equity = state["wallet_balance"] + unrealized
    available = min(state["wallet_balance"], equity) - margin
    return equity, margin, available


def _quote(quote, contract, now):
    if not isinstance(quote, dict) or set(quote) != {"coin", "at", "bid", "ask", "mark", "fee_bps", "size_decimals", "source_mode"}:
        raise ValueError("Require a complete perpetual quote in contract units")
    if quote["coin"] not in contract["coins"] or quote["source_mode"] != contract["source_mode"]:
        raise ValueError("Perpetual quote market/source mode disagrees with account")
    age = (now - _time(quote["at"])).total_seconds()
    if not 0 <= age < 30:
        raise ValueError("Perpetual quote is future-dated or stale")
    for name in ("bid", "ask", "mark"):
        _number(quote[name], positive=True)
    if quote["bid"] >= quote["ask"] or not 0 <= _number(quote["fee_bps"]) < 10000:
        raise ValueError("Invalid perpetual spread/fee")
    if type(quote["size_decimals"]) is not int or not 0 <= quote["size_decimals"] <= 8:
        raise ValueError("Invalid perpetual quantity precision")
    return quote


def _report(state, contract):
    equity, margin, available = _totals(state)
    realized = math.fsum(t["net_pnl"] for t in state["trades"])
    open_net = math.fsum(p["quantity"] * (p["mark_price"] - p["entry_fill_price"]) - p["entry_fee"] + p["funding_cash_flow"] for p in state["positions"].values())
    if not math.isclose(equity - contract["initial_collateral"], realized + open_net, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError("Perpetual equity does not reconcile")
    missing = _gaps(state, _time(state["as_of"]))
    return {**deepcopy(state), "mode": "paper_perpetual", "currency": "USDC", "source_mode": contract["source_mode"],
            "configured_leverage": 1, "equity": equity, "reserved_initial_margin": margin,
            "available_collateral": available, "realized_net_pnl": realized, "unrealized_net_pnl": open_net,
            "missing_funding": missing, "performance_provisional": bool(missing), "reconciled": True,
            "funding_oracle_estimated": contract["source_mode"] == "observed" and bool(state["settlements"]),
            "fill_count": len(state["trades"]) * 2 + len(state["positions"]),
            "live_execution_enabled": False, "liquidation_simulated": False}


def _reduce(state, contract, event):
    state = deepcopy(state)
    now, inputs, kind = _time(event["at"]), event["inputs"], event["kind"]
    risk = contract["risk"]
    if now < _time(state["as_of"]):
        raise ValueError("Perpetual account time cannot move backwards")
    if state["day"] != _iso(now)[:10]:
        state["day_start_equity"] = state["wallet_balance"] if not state["positions"] and not _gaps(state, now) else None
        state["day"] = _iso(now)[:10]
    state["as_of"] = _iso(now)
    result = {"status": "blocked", "reason": "unsupported_operation"}
    if kind in ("entry", "exit"):
        quote = _quote(inputs["quote"], contract, now)
        coin, mid = quote["coin"], quote["bid"] + (quote["ask"] - quote["bid"]) / 2
    if kind == "mark":
        quotes = inputs["quotes"]
        if len({q["coin"] for q in quotes}) != len(quotes):
            raise ValueError("Duplicate perpetual marks")
        for quote in quotes:
            _quote(quote, contract, now)
            if quote["coin"] in state["positions"]:
                state["positions"][quote["coin"]].update(mark_price=quote["mark"], marked_at=quote["at"])
        result = {"status": "marked", "reason": "observed_marks_recorded"}
    elif kind == "pause":
        if type(inputs["enabled"]) is not bool:
            raise ValueError("Entries-enabled flag must be boolean")
        state["entries_enabled"] = inputs["enabled"]
        result = {"status": "updated", "reason": "entry_switch_recorded"}
    elif kind == "funding":
        if inputs.get("source_mode") != contract["source_mode"]:
            raise ValueError("Funding source mode disagrees with account")
        coin, hour = inputs["coin"], _time(inputs["hour"])
        rate, oracle = _number(inputs["rate"]), _number(inputs["oracle_price"], positive=True)
        if coin not in contract["coins"] or hour != _hour(hour) or hour > now or abs(rate) > 0.04:
            raise ValueError("Invalid settled hourly funding")
        if abs((_time(inputs["oracle_at"]) - hour).total_seconds()) >= 30 or _time(inputs["oracle_at"]) > now:
            raise ValueError("Require an observed oracle near the funding settlement, not a current mark")
        if not all(isinstance(inputs.get(key), str) and inputs[key].strip() for key in ("rate_source_id", "oracle_source_id")):
            raise ValueError("Funding requires settled-rate and oracle provenance")
        key = coin + "|" + _iso(hour)
        if key in state["settlements"]:
            if state["settlements"][key] != inputs:
                raise ValueError("Changed funding settlement")
            result = {"status": "duplicate", "reason": "funding_already_posted", "cash_flow": 0.0}
        else:
            amount = 0.0
            for position in list(state["positions"].values()) + state["trades"]:
                if position["coin"] == coin and _time(position["entry_at"]) < hour <= _time(position.get("exit_at", _iso(now))):
                    flow = -position["quantity"] * oracle * rate
                    position["funding_cash_flow"] += flow
                    amount += flow
                    if "exit_at" in position:
                        position["net_pnl"] += flow
            state["wallet_balance"] += amount
            state["settlements"][key] = inputs
            result = {"status": "posted", "reason": "settled_hourly_funding", "cash_flow": amount}
    elif kind == "entry":
        evidence = SignalEvidence(**inputs["evidence"])
        gate = evaluate_entry_signal(evidence, len(state["positions"]), EntryPolicy(**contract["entry_policy"]))
        quantity = _number(inputs["quantity"], positive=True)
        scaled = quantity * 10 ** quote["size_decimals"]
        if not math.isfinite(scaled) or not math.isclose(scaled, round(scaled), abs_tol=1e-7, rel_tol=0):
            raise ValueError("Quantity violates contract size precision")
        signal = _time(inputs["signal_at"])
        age = (now - signal).total_seconds()
        expected = _number(inputs["expected_gross_return"])
        if expected <= -1:
            raise ValueError("Gross long return forecast must exceed -100%")
        fill_price = quote["ask"] * (1 + contract["slippage_bps"] / 10000)
        fee = quantity * fill_price * quote["fee_bps"] / 10000
        margin = quantity * fill_price
        impact = quantity * (fill_price - mid)
        equity, reserved, available = _totals(state)
        projected = equity - fee + quantity * (quote["mark"] - fill_price)
        exit_estimate = quote["bid"] * (1 - contract["slippage_bps"] / 10000) * (1 + expected)
        modeled_net = (exit_estimate * (1 - quote["fee_bps"] / 10000) - fill_price * (1 + quote["fee_bps"] / 10000)) / fill_price
        funding_estimate = inputs.get("expected_funding_cost_fraction")
        if funding_estimate is not None:
            funding_estimate = _number(funding_estimate)
            if funding_estimate < 0:
                raise ValueError("Funding cost budget must be nonnegative; do not credit hoped-for funding")
            modeled_net -= funding_estimate
        for value in (margin, fee, impact, projected, modeled_net):
            _number(value)
        exposure = math.fsum(p["quantity"] * p["mark_price"] for p in state["positions"].values())
        reason = None
        if not gate.eligible_for_account_risk_check:
            result = {"status": "review" if gate.status == "REVIEW" else "blocked", "reason": gate.reason}
        elif funding_estimate is None:
            result = {"status": "review", "reason": "funding_cost_estimate_unavailable"}
        else:
            checks = [
                (not state["entries_enabled"], "new_entries_paused"),
                (coin in state["positions"], "position_already_open"),
                (signal.timestamp() % 300 != 0 or not 300 <= age < 600 or _time(quote["at"]) < signal + timedelta(minutes=5), "invalid_or_stale_signal"),
                (bool(_gaps(state, now)), "missing_funding_settlement"),
                (any(not 0 <= (now - _time(p["marked_at"])).total_seconds() < 30 for p in state["positions"].values()), "stale_position_mark"),
                (state["day_start_equity"] is None, "daily_equity_unavailable"),
                (equity <= 0 or projected <= 0, "nonpositive_equity"),
                (min(equity, projected) <= state["peak_equity"] * (1 - risk["drawdown_fraction"]), "drawdown_limit"),
                (state["day_start_equity"] is not None and min(equity, projected) <= state["day_start_equity"] * (1 - risk["daily_loss_fraction"]), "daily_loss_limit"),
                (margin + fee > available, "insufficient_collateral"),
                (margin + fee > equity * risk["position_fraction"] or quantity * quote["mark"] > projected * risk["position_fraction"], "position_exposure_limit"),
                (any(p["quantity"] * p["mark_price"] > projected * risk["position_fraction"] for p in state["positions"].values()), "existing_position_exposure_limit"),
                (exposure + quantity * quote["mark"] > projected * risk["total_fraction"], "total_exposure_limit"),
                (modeled_net <= risk["minimum_net_margin"], "forecast_does_not_cover_costs_and_margin"),
            ]
            reason = next((name for failed, name in checks if failed), None)
            result = {"status": "blocked", "reason": reason} if reason else {"status": "filled", "reason": "perpetual_paper_entry_approved"}
            if reason is None:
                position = {"coin": coin, "position_id": event["operation_id"], "quantity": quantity,
                            "underlying_units_per_contract": BY_COIN[coin].underlying_units_per_contract,
                            "entry_at": _iso(now), "exit_due_at": _iso(now + timedelta(minutes=30)),
                            "entry_reference_price": mid, "entry_fill_price": fill_price, "entry_fee": fee,
                            "entry_impact_cost": impact, "initial_margin": margin, "mark_price": quote["mark"],
                            "marked_at": quote["at"], "funding_cash_flow": 0.0}
                state["wallet_balance"] -= fee
                state["positions"][coin] = position
                result["fill"] = {"side": "BUY", "coin": coin, "quantity": quantity, "price": fill_price, "fee": fee, "reserved_margin": margin}
    elif kind == "exit":
        position = state["positions"].get(coin)
        reason = inputs["reason"]
        if reason not in ("horizon", "risk", "policy_sell", "manual"):
            raise ValueError("Invalid perpetual exit reason")
        if position is None:
            result = {"status": "blocked", "reason": "position_unavailable"}
        elif inputs["position_id"] != position["position_id"]:
            result = {"status": "blocked", "reason": "position_changed"}
        elif reason == "horizon" and (_time(quote["at"]) < _time(position["exit_due_at"]) or now < _time(position["exit_due_at"])):
            result = {"status": "blocked", "reason": "exit_not_due"}
        else:
            fill_price = quote["bid"] * (1 - contract["slippage_bps"] / 10000)
            fee = position["quantity"] * fill_price * quote["fee_bps"] / 10000
            actual_gross = position["quantity"] * (fill_price - position["entry_fill_price"])
            state["wallet_balance"] += actual_gross - fee
            trade = {**position, "exit_at": _iso(now), "exit_fill_price": fill_price,
                     "gross_pnl": position["quantity"] * (mid - position["entry_reference_price"]),
                     "fees": position["entry_fee"] + fee,
                     "price_impact_cost": position["entry_impact_cost"] + position["quantity"] * (mid - fill_price),
                     "net_pnl": actual_gross - position["entry_fee"] - fee + position["funding_cash_flow"]}
            state["trades"].append(trade)
            del state["positions"][coin]
            result = {"status": "filled", "reason": reason, "trade": deepcopy(trade),
                      "fill": {"side": "SELL", "coin": coin, "quantity": position["quantity"], "price": fill_price, "fee": fee}}
    else:
        if kind not in ("mark", "funding", "pause"):
            raise ValueError("Unsupported perpetual operation")
    equity, _, _ = _totals(state)
    if not all(math.isfinite(value) for value in (equity, state["wallet_balance"])):
        raise ValueError("Nonfinite perpetual balances")
    state["peak_equity"] = max(state["peak_equity"], equity)
    state["version"] += 1
    result["account_version"] = state["version"]
    _report(state, contract)
    return state, result


SCHEMA = """
CREATE TABLE account(id INTEGER PRIMARY KEY CHECK(id=1), contract TEXT NOT NULL,
 contract_hash TEXT NOT NULL, state TEXT NOT NULL, state_hash TEXT NOT NULL);
CREATE TABLE events(sequence INTEGER PRIMARY KEY, operation_id TEXT UNIQUE NOT NULL,
 inputs_hash TEXT NOT NULL, event TEXT NOT NULL, result TEXT NOT NULL,
 state_hash TEXT NOT NULL, previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL);
"""


class PerpetualPaperAccount:
    """Serialized immediate paper operations with idempotency and replay audits."""

    def __init__(self, path: Path):
        self.path = native_path(path).resolve()
        if not self.path.is_file():
            raise ValueError("Perpetual account does not exist")
        self.audit()

    @classmethod
    def create(cls, path, *, coins=tuple(BY_COIN), initial_collateral=10000.0,
               policy=EntryPolicy(), slippage_bps=1.0, source_mode="synthetic", now=None):
        initial_collateral = _number(initial_collateral, positive=True)
        slippage_bps = _number(slippage_bps)
        if not 0 <= slippage_bps < 10000 or source_mode not in ("synthetic", "observed"):
            raise ValueError("Invalid perpetual paper contract")
        if not coins or len(set(coins)) != len(coins) or any(coin not in BY_COIN for coin in coins):
            raise ValueError("Require registered perpetual identifiers")
        policy = EntryPolicy(**asdict(policy))
        if policy.maximum_positions > 5 or policy.minimum_probability < 0.6 or policy.minimum_confidence < 0.7:
            raise ValueError("Perpetual policy cannot relax the configured minimums")
        now = _utc(now or datetime.now(timezone.utc))
        contract = {"schema_version": 1, "mode": "paper_perpetual", "currency": "USDC", "leverage": 1,
                    "coins": list(coins), "initial_collateral": initial_collateral, "slippage_bps": slippage_bps,
                    "source_mode": source_mode, "created_at": _iso(now), "entry_policy": asdict(policy), "risk": dict(RISK_LIMITS)}
        path = native_path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb"):
            pass
        try:
            with closing(sqlite3.connect(str(path))) as db:
                db.executescript(SCHEMA)
                with db:
                    state = _initial(contract)
                    db.execute("INSERT INTO account VALUES(1,?,?,?,?)", (_json(contract), _hash(contract), _json(state), _hash(state)))
            return cls(path)
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    @contextmanager
    def _transaction(self, write=False):
        db = sqlite3.connect(str(self.path), timeout=10)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _read(self, db):
        row = db.execute("SELECT * FROM account WHERE id=1").fetchone()
        if row is None:
            raise ValueError("Perpetual account metadata is missing")
        contract, state = json.loads(row["contract"]), json.loads(row["state"])
        if _hash(contract) != row["contract_hash"] or _hash(state) != row["state_hash"] or contract.get("mode") != "paper_perpetual" or contract.get("currency") != "USDC" or type(contract.get("schema_version")) is not int or contract["schema_version"] != 1 or type(contract.get("leverage")) is not int or contract["leverage"] != 1:
            raise ValueError("Perpetual account contract/state changed")
        if contract.get("source_mode") not in ("observed", "synthetic") or contract.get("risk") != RISK_LIMITS:
            raise ValueError("Unsupported perpetual source/risk contract")
        head = db.execute("SELECT * FROM events ORDER BY sequence DESC LIMIT 1").fetchone()
        if head is not None:
            self._event_integrity(head)
        if (head is None and state != _initial(contract)) or (head is not None and (head["sequence"] != state["version"] or head["state_hash"] != row["state_hash"])):
            raise ValueError("Perpetual journal head disagrees with account")
        return contract, state, head

    @staticmethod
    def _event_integrity(row):
        event, result = json.loads(row["event"]), json.loads(row["result"])
        digest = _hash({"event": event, "result": result, "state_hash": row["state_hash"], "previous_hash": row["previous_hash"]})
        if row["event_hash"] != digest or row["operation_id"] != event["operation_id"] or row["sequence"] != result["account_version"] or row["inputs_hash"] != _hash({"kind": event["kind"], "inputs": event["inputs"]}):
            raise ValueError("Perpetual event checksum changed")

    def audit(self):
        with self._transaction() as db:
            contract, persisted, _ = self._read(db)
            state, previous = _initial(contract), _hash(contract)
            for row in db.execute("SELECT * FROM events ORDER BY sequence"):
                event, saved = json.loads(row["event"]), json.loads(row["result"])
                state, result = _reduce(state, contract, event)
                digest = _hash({"event": event, "result": result, "state_hash": _hash(state), "previous_hash": previous})
                if row["sequence"] != state["version"] or event["operation_id"] != row["operation_id"] or row["inputs_hash"] != _hash({"kind": event["kind"], "inputs": event["inputs"]}) or saved != result or row["previous_hash"] != previous or row["event_hash"] != digest or row["state_hash"] != _hash(state):
                    raise ValueError("Perpetual journal does not reconcile")
                previous = digest
            if state != persisted:
                raise ValueError("Perpetual balances do not match replay")
            return _report(state, contract)

    def _persist(self, db, state, event, result, previous, inputs_hash):
        state_hash = _hash(state)
        digest = _hash({"event": event, "result": result, "state_hash": state_hash, "previous_hash": previous})
        db.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,?)", (state["version"], event["operation_id"], inputs_hash, _json(event), _json(result), state_hash, previous, digest))
        db.execute("UPDATE account SET state=?,state_hash=? WHERE id=1", (_json(state), state_hash))

    def execute(self, operation_id, kind, inputs, *, now=None):
        if not isinstance(operation_id, str) or not operation_id.strip() or len(operation_id) > 200:
            raise ValueError("Require a bounded stable operation identifier")
        inputs = json.loads(_json(inputs))
        inputs_hash = _hash({"kind": kind, "inputs": inputs})
        with self._transaction(write=True) as db:
            contract, state, head = self._read(db)
            old = db.execute("SELECT * FROM events WHERE operation_id=?", (operation_id,)).fetchone()
            if old is not None:
                self._event_integrity(old)
                if old["inputs_hash"] != inputs_hash:
                    raise ValueError("Operation identifier reused with different inputs")
                return {**json.loads(old["result"]), "replayed": True}
            at = _utc(now or datetime.now(timezone.utc))
            event = {"operation_id": operation_id, "kind": kind, "inputs": inputs, "at": _iso(at)}
            state, result = _reduce(state, contract, event)
            previous = _hash(contract) if head is None else head["event_hash"]
            self._persist(db, state, event, result, previous, inputs_hash)
            commit_at = at if now is not None else datetime.now(timezone.utc)
            if not 0 <= (commit_at - at).total_seconds() < 5:
                raise ValueError("Perpetual operation expired before commit")
            for quote in ([inputs["quote"]] if kind in ("entry", "exit") else inputs.get("quotes", [])):
                _quote(quote, contract, commit_at)
            return {**result, "replayed": False}
