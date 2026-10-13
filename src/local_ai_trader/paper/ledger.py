"""Transactional spot-long paper fills with restart-safe operation identifiers."""

from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from local_ai_trader.data.storage import native_path
from local_ai_trader.risk.paper import (
    MarkedPosition, PaperAccountSnapshot, PaperEntryRequest, PaperRiskSettings,
    _number, _utc, authorize_paper_entry, paper_authorization_is_current,
    validate_paper_risk_settings,
)
from local_ai_trader.settings import BacktestSettings, load_backtest_settings


@dataclass(frozen=True)
class MarketQuote:
    symbol: str
    price: float
    at: datetime


@dataclass(frozen=True)
class PaperOperation:
    operation_id: str
    status: str
    reason: str
    account_version: int
    fill: dict | None = None
    trade: dict | None = None
    authorization: dict | None = None
    account_snapshot: dict | None = None
    replayed: bool = False


def _iso(value):
    return _utc(value).isoformat(timespec="microseconds")


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, default=_iso, separators=(",", ":"))


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _equal(actual, expected):
    return _number(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8)


SCHEMA = """
CREATE TABLE account (
 id INTEGER PRIMARY KEY CHECK (id=1), contract TEXT NOT NULL, contract_hash TEXT NOT NULL,
 cash REAL NOT NULL, peak_equity REAL NOT NULL, day_start_equity REAL NOT NULL,
 day_start_at TEXT NOT NULL, as_of TEXT NOT NULL, version INTEGER NOT NULL,
 entries_enabled INTEGER NOT NULL CHECK(entries_enabled IN (0,1))
);
CREATE TABLE operations (
 operation_id TEXT PRIMARY KEY, payload TEXT NOT NULL, payload_hash TEXT NOT NULL,
 result TEXT NOT NULL, result_hash TEXT NOT NULL
);
CREATE TABLE positions (
 symbol TEXT PRIMARY KEY, quantity REAL NOT NULL, entry_id TEXT UNIQUE NOT NULL,
 entry_at TEXT NOT NULL, exit_due_at TEXT NOT NULL,
 mark_price REAL NOT NULL, marked_at TEXT NOT NULL
);
CREATE TABLE fills (
 sequence INTEGER PRIMARY KEY, operation_id TEXT UNIQUE NOT NULL REFERENCES operations(operation_id),
 position_id TEXT NOT NULL, side TEXT NOT NULL CHECK(side IN ('BUY','SELL')),
 symbol TEXT NOT NULL, filled_at TEXT NOT NULL, signal_at TEXT NOT NULL,
 reference_price REAL NOT NULL, fill_price REAL NOT NULL, quantity REAL NOT NULL,
 notional REAL NOT NULL, fee REAL NOT NULL, price_impact_cost REAL NOT NULL,
 cash_delta REAL NOT NULL, reason TEXT NOT NULL
);
CREATE TABLE trades (
 entry_id TEXT PRIMARY KEY, exit_id TEXT UNIQUE NOT NULL, symbol TEXT NOT NULL,
 entry_at TEXT NOT NULL, exit_at TEXT NOT NULL, quantity REAL NOT NULL,
 gross_pnl REAL NOT NULL, fees REAL NOT NULL, price_impact_cost REAL NOT NULL, net_pnl REAL NOT NULL
);
CREATE INDEX fills_time ON fills(filled_at);
"""


class PaperLedger:
    """One USD simulated account; every mutation is an immediate SQL transaction.

    Costs and risk defaults are frozen at creation. An account has no queued or
    broker orders: a successful entry is sized and committed inside one lock.
    All operation IDs, including refusals, are idempotent; a new attempt needs a
    new ID. Quotes are observed inputs supplied by the future market-data loop.
    """

    def __init__(self, path: Path):
        self.path = native_path(path).resolve()
        if not self.path.is_file():
            raise ValueError(f"Paper account not found: {path}")
        self.audit()

    @classmethod
    def create(cls, path: Path, risk: PaperRiskSettings, costs: BacktestSettings, *, now: datetime | None = None, source_mode: str = "observed"):
        risk = validate_paper_risk_settings(risk)
        costs = load_backtest_settings(asdict(costs))
        if source_mode not in ("observed", "synthetic"):
            raise ValueError("Paper source_mode must be observed or synthetic")
        if risk.max_signal_age_bars <= costs.latency_bars:
            raise ValueError("Paper signal freshness must exceed execution latency")
        now = _utc(now or datetime.now(timezone.utc))
        path = native_path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents overwriting an existing account.
        with path.open("xb"):
            pass
        contract = {"schema_version": 1, "mode": "paper", "currency": "USD", "source_mode": source_mode,
                    "created_at": _iso(now), "initial_cash": costs.initial_cash, "risk": asdict(risk), "costs": asdict(costs)}
        try:
            with closing(sqlite3.connect(str(path))) as db:
                with db:
                    db.executescript(SCHEMA)
                    db.execute("INSERT INTO account VALUES(1,?,?,?,?,?,?,?,?,?)", (
                        _json(contract), _hash(contract), costs.initial_cash, costs.initial_cash, costs.initial_cash,
                        _iso(now.replace(hour=0, minute=0, second=0, microsecond=0)), _iso(now), 0, 1))
            return cls(path)
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    @contextmanager
    def _transaction(self, *, write=False):
        db = sqlite3.connect(str(self.path), timeout=10)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _account(self, db):
        row = db.execute("SELECT * FROM account WHERE id=1").fetchone()
        if row is None:
            raise ValueError("Paper account metadata is missing")
        state = dict(row)
        contract = json.loads(state["contract"])
        if contract.get("schema_version") != 1 or contract.get("mode") != "paper" or contract.get("currency") != "USD" or contract.get("source_mode") not in ("synthetic", "observed") or _hash(contract) != state["contract_hash"]:
            raise ValueError("Invalid or changed paper account contract")
        parameters = dict(contract["risk"])
        parameters["allowed_symbols"] = tuple(parameters["allowed_symbols"])
        risk = validate_paper_risk_settings(PaperRiskSettings(**parameters))
        costs = load_backtest_settings(contract["costs"])
        if not _equal(contract["initial_cash"], costs.initial_cash) or type(state["version"]) is not int or state["version"] < 0 or state["entries_enabled"] not in (0, 1):
            raise ValueError("Invalid paper account balances or version")
        for key in ("cash", "peak_equity", "day_start_equity"):
            if not _number(state[key], positive=key != "cash"):
                raise ValueError("Invalid paper account balance")
        for key in ("as_of", "day_start_at"):
            _utc(datetime.fromisoformat(state[key]))
        return state, contract, risk, costs

    def _audit(self, db):
        """Reconstruct cash, inventory and every closed trade from fill history."""
        state, contract, risk, costs = self._account(db)
        impact = (costs.spread_bps / 2 + costs.slippage_bps) / 10000
        fee_rate = costs.fee_bps / 10000
        cash, held, closed = contract["initial_cash"], {}, {}
        previous = contract["created_at"]
        operations, payloads = {}, {}
        for row in db.execute("SELECT * FROM operations"):
            payload, result = json.loads(row["payload"]), json.loads(row["result"])
            if _hash(payload) != row["payload_hash"] or _hash(result) != row["result_hash"] or result["operation_id"] != row["operation_id"]:
                raise ValueError("Paper operation journal changed")
            operations[row["operation_id"]] = result
            payloads[row["operation_id"]] = payload
        if state["version"] != len(operations):
            raise ValueError("Paper account version does not match journal")
        if sorted(result["account_version"] for result in operations.values()) != list(range(1, state["version"] + 1)):
            raise ValueError("Paper journal versions are inconsistent")
        for row in db.execute("SELECT * FROM fills ORDER BY sequence"):
            fill = dict(row)
            if fill["symbol"] not in risk.allowed_symbols or fill["filled_at"] < previous or fill["filled_at"] > state["as_of"] or not _number(fill["quantity"], positive=True) or not _number(fill["reference_price"], positive=True):
                raise ValueError("Invalid paper fill timeline or inventory")
            previous = fill["filled_at"]
            _utc(datetime.fromisoformat(previous))
            direction = 1 if fill["side"] == "BUY" else -1
            price = fill["reference_price"] * (1 + direction * impact)
            notional, fee = fill["quantity"] * price, fill["quantity"] * price * fee_rate
            price_cost = fill["quantity"] * abs(price - fill["reference_price"])
            cash_delta = -notional - fee if direction == 1 else notional - fee
            if not all(_equal(fill[key], value) for key, value in (("fill_price", price), ("notional", notional), ("fee", fee), ("price_impact_cost", price_cost))) or not math.isclose(fill["cash_delta"], cash_delta, rel_tol=1e-10, abs_tol=1e-8):
                raise ValueError("Paper fill costs do not reconcile")
            result = operations.get(fill["operation_id"])
            if result is None or result["status"] != "filled" or result["fill"] != {key: value for key, value in fill.items() if key != "sequence"}:
                raise ValueError("Paper fill does not match operation journal")
            if direction == 1:
                approval = result.get("authorization")
                if not isinstance(approval, dict) or approval.get("mode") != "paper" or approval.get("account_version") != result["account_version"] - 1 or any(approval.get(key) != fill[key] for key in ("quantity", "fill_price", "fee", "price_impact_cost", "reference_price")):
                    raise ValueError("Paper entry approval does not match fill")
                request = payloads[fill["operation_id"]]["request"]
                signal_at, quote_at, filled_at = (datetime.fromisoformat(stamp) for stamp in (request["signal_at"], request["quote_at"], fill["filled_at"]))
                issued, expiry = (datetime.fromisoformat(approval[key]) for key in ("issued_at", "expires_at"))
                if request["symbol"] != fill["symbol"] or request["reference_price"] != fill["reference_price"] or _iso(signal_at) != fill["signal_at"] or not issued <= filled_at < expiry or expiry > issued + timedelta(seconds=risk.authorization_ttl_seconds) or not 0 <= (filled_at - quote_at).total_seconds() < risk.max_quote_age_seconds or not costs.latency_bars * risk.candle_seconds <= (filled_at - signal_at).total_seconds() < risk.max_signal_age_bars * risk.candle_seconds:
                    raise ValueError("Paper entry timing or approval expired")
            cash += cash_delta
            if cash < -1e-8:
                raise ValueError("Paper cash was overspent")
            if direction == 1:
                if fill["symbol"] in held or fill["position_id"] != fill["operation_id"]:
                    raise ValueError("Duplicate paper entry")
                held[fill["symbol"]] = fill
            else:
                entry = held.pop(fill["symbol"], None)
                if entry is None or entry["position_id"] != fill["position_id"] or entry["quantity"] != fill["quantity"]:
                    raise ValueError("Paper exit inventory does not reconcile")
                gross = fill["quantity"] * (fill["reference_price"] - entry["reference_price"])
                fees, price_impact = entry["fee"] + fee, entry["price_impact_cost"] + price_cost
                closed[entry["position_id"]] = {
                    "entry_id": entry["operation_id"], "exit_id": fill["operation_id"], "symbol": fill["symbol"],
                    "entry_at": entry["filled_at"], "exit_at": fill["filled_at"], "quantity": fill["quantity"],
                    "gross_pnl": gross, "fees": fees, "price_impact_cost": price_impact, "net_pnl": gross - fees - price_impact,
                }
        if not _equal(state["cash"], cash):
            raise ValueError("Paper cash does not reconcile with fills")
        positions = [dict(row) for row in db.execute("SELECT * FROM positions ORDER BY symbol")]
        if set(held) != {position["symbol"] for position in positions}:
            raise ValueError("Paper holdings do not reconcile with fills")
        for position in positions:
            entry = held[position["symbol"]]
            due = datetime.fromisoformat(entry["filled_at"]) + timedelta(seconds=risk.holding_bars * risk.candle_seconds)
            if position["entry_id"] != entry["operation_id"] or position["quantity"] != entry["quantity"] or position["entry_at"] != entry["filled_at"] or position["exit_due_at"] != _iso(due) or not _number(position["mark_price"], positive=True) or not entry["signal_at"] <= position["marked_at"] <= state["as_of"]:
                raise ValueError("Paper position metadata does not reconcile")
        trades = [dict(row) for row in db.execute("SELECT * FROM trades ORDER BY exit_at, entry_id")]
        if set(closed) != {trade["entry_id"] for trade in trades}:
            raise ValueError("Paper trade ledger does not reconcile")
        for trade in trades:
            expected = closed[trade["entry_id"]]
            if any(not math.isclose(value, expected[key], rel_tol=1e-10, abs_tol=1e-8) if isinstance(value, float) else value != expected[key] for key, value in trade.items()):
                raise ValueError("Paper trade PnL does not reconcile")
        if sum(result["status"] == "filled" for result in operations.values()) != db.execute("SELECT count(*) FROM fills").fetchone()[0]:
            raise ValueError("Paper operation is missing a fill")
        if operations:
            latest = max(operations.values(), key=lambda result: result["account_version"])
            if latest.get("account_snapshot") != self._state_record(state, positions):
                raise ValueError("Paper account or marks changed outside the journal")
        elif state["cash"] != contract["initial_cash"] or state["peak_equity"] != contract["initial_cash"] or state["day_start_equity"] != contract["initial_cash"] or state["as_of"] != contract["created_at"] or state["entries_enabled"] != 1 or state["day_start_at"] != _iso(datetime.fromisoformat(contract["created_at"]).replace(hour=0, minute=0, second=0, microsecond=0)):
            raise ValueError("Initial paper state changed")
        return state, contract, risk, costs, positions, trades

    def audit(self):
        with self._transaction() as db:
            state, contract, risk, costs, positions, trades = self._audit(db)
            equity = state["cash"] + math.fsum(p["quantity"] * p["mark_price"] for p in positions)
            net = math.fsum(trade["net_pnl"] for trade in trades)
            entry_costs = db.execute("SELECT coalesce(sum(fee+price_impact_cost),0) FROM fills WHERE side='BUY' AND position_id IN (SELECT entry_id FROM positions)").fetchone()[0]
            unrealized = math.fsum(p["quantity"] * p["mark_price"] - db.execute("SELECT -cash_delta FROM fills WHERE operation_id=?", (p["entry_id"],)).fetchone()[0] for p in positions)
            if not math.isclose(equity - contract["initial_cash"], net + unrealized, rel_tol=1e-10, abs_tol=1e-8):
                raise ValueError("Paper equity and PnL do not reconcile")
            return {"mode": "paper", "source_mode": contract["source_mode"], "currency": "USD", "version": state["version"],
                    "as_of": state["as_of"], "initial_cash": contract["initial_cash"], "cash": state["cash"], "equity": equity,
                    "realized_net_pnl": net, "unrealized_net_pnl": unrealized, "open_entry_costs": entry_costs,
                    "peak_equity": state["peak_equity"], "day_start_at": state["day_start_at"], "day_start_equity": state["day_start_equity"],
                    "entries_enabled": bool(state["entries_enabled"] and risk.new_entries_enabled), "positions": positions, "trades": trades,
                    "fill_count": db.execute("SELECT count(*) FROM fills").fetchone()[0], "reconciled": True}

    def _quotes(self, quotes, risk, now, *, day=None):
        if not isinstance(quotes, tuple):
            raise ValueError("Require a tuple of timestamped paper quotes")
        found = {}
        for quote in quotes:
            if not isinstance(quote, MarketQuote) or quote.symbol not in risk.allowed_symbols or quote.symbol in found or not _number(quote.price, positive=True):
                raise ValueError("Invalid or duplicate paper quote")
            stamp = _utc(quote.at)
            if stamp > now or (day is not None and stamp != day):
                raise ValueError("Invalid paper quote time")
            found[quote.symbol] = quote
        return found

    def _refresh(self, db, state, contract, risk, quotes, day_quotes, now):
        if _iso(now) < state["as_of"]:
            raise ValueError("Paper clock cannot move backwards")
        marks = self._quotes(quotes, risk, now)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        opening = self._quotes(day_quotes, risk, now, day=midnight)
        for position in db.execute("SELECT * FROM positions").fetchall():
            quote = marks.get(position["symbol"])
            if quote is not None:
                if _iso(quote.at) < position["marked_at"]:
                    raise ValueError("Paper position mark cannot move backwards")
                db.execute("UPDATE positions SET mark_price=?,marked_at=? WHERE symbol=?", (quote.price, _iso(quote.at), quote.symbol))
        if state["day_start_at"] != _iso(midnight):
            # Reconstruct midnight inventory/cash even after an intraday exit.
            cash, holdings = contract["initial_cash"], {}
            for fill in db.execute("SELECT * FROM fills WHERE filled_at<? ORDER BY sequence", (_iso(midnight),)):
                cash += fill["cash_delta"]
                if fill["side"] == "BUY":
                    holdings[fill["symbol"]] = fill["quantity"]
                else:
                    holdings.pop(fill["symbol"])
            if set(holdings) <= set(opening):
                state["day_start_equity"] = cash + math.fsum(quantity * opening[symbol].price for symbol, quantity in holdings.items())
                state["day_start_at"] = _iso(midnight)
                state["peak_equity"] = max(state["peak_equity"], state["day_start_equity"])
        positions = [dict(row) for row in db.execute("SELECT * FROM positions ORDER BY symbol")]
        if all(0 <= (now - datetime.fromisoformat(p["marked_at"])).total_seconds() < risk.max_account_age_seconds for p in positions):
            state["peak_equity"] = max(state["peak_equity"], state["cash"] + math.fsum(p["quantity"] * p["mark_price"] for p in positions))
        state["as_of"] = _iso(now)
        snapshot = PaperAccountSnapshot(state["version"], now, state["cash"], state["peak_equity"], state["day_start_equity"], datetime.fromisoformat(state["day_start_at"]),
            tuple(MarkedPosition(p["symbol"], p["quantity"], p["mark_price"], datetime.fromisoformat(p["marked_at"])) for p in positions))
        return snapshot, positions

    def _replay(self, db, operation_id, payload):
        if not isinstance(operation_id, str) or not operation_id.strip() or len(operation_id) > 200:
            raise ValueError("Require a nonempty operation ID of at most 200 characters")
        row = db.execute("SELECT * FROM operations WHERE operation_id=?", (operation_id,)).fetchone()
        if row is None:
            return None
        if row["payload_hash"] != _hash(payload):
            raise ValueError("Operation ID already used with different inputs")
        return replace(PaperOperation(**json.loads(row["result"])), replayed=True)

    def _state_record(self, state, positions):
        return {key: state[key] for key in ("cash", "peak_equity", "day_start_equity", "day_start_at", "as_of", "version", "entries_enabled")} | {"positions": positions}

    def _save(self, db, state, operation_id, payload, status, reason, fill=None, trade=None, authorization=None):
        positions = [dict(row) for row in db.execute("SELECT * FROM positions ORDER BY symbol")]
        snapshot = self._state_record(state | {"version": state["version"] + 1}, positions)
        result = PaperOperation(operation_id, status, reason, state["version"] + 1, fill, trade, authorization, snapshot)
        result_data = json.loads(_json(asdict(result)))
        result = PaperOperation(**result_data)
        db.execute("INSERT INTO operations VALUES(?,?,?,?,?)", (operation_id, _json(payload), _hash(payload), _json(result_data), _hash(result_data)))
        if fill:
            keys = tuple(fill)
            db.execute(f"INSERT INTO fills({','.join(keys)}) VALUES({','.join('?' for _ in keys)})", tuple(fill.values()))
        if trade:
            keys = tuple(trade)
            db.execute(f"INSERT INTO trades({','.join(keys)}) VALUES({','.join('?' for _ in keys)})", tuple(trade.values()))
        updated = db.execute("UPDATE account SET cash=?,peak_equity=?,day_start_equity=?,day_start_at=?,as_of=?,version=?,entries_enabled=? WHERE id=1 AND version=?", (
            state["cash"], state["peak_equity"], state["day_start_equity"], state["day_start_at"], state["as_of"], result.account_version, state["entries_enabled"], state["version"]))
        if updated.rowcount != 1:
            raise RuntimeError("Paper account version changed before commit")
        self._audit(db)
        return result

    def execute_entry(self, operation_id: str, request: PaperEntryRequest, quotes: tuple[MarketQuote, ...] = (), *, day_quotes: tuple[MarketQuote, ...] = (), expected_version: int | None = None, now: datetime | None = None):
        """Refresh all holdings, authorize inside the lock, then commit one BUY.

        No outside approval is accepted. now is a deterministic replay/test clock;
        the real-time loop leaves it unset, including the final expiry recheck.
        """
        if not isinstance(request, PaperEntryRequest):
            raise ValueError("Require a structured paper entry request")
        payload = {"kind": "entry", "request": asdict(request), "quotes": [asdict(q) for q in quotes], "day_quotes": [asdict(q) for q in day_quotes], "expected_version": expected_version, "clock": None if now is None else _iso(now)}
        with self._transaction(write=True) as db:
            state, contract, risk, costs, _, _ = self._audit(db)
            replay = self._replay(db, operation_id, payload)
            if replay:
                return replay
            clock = _utc(now or datetime.now(timezone.utc))
            snapshot, _ = self._refresh(db, state, contract, risk, quotes, day_quotes, clock)
            if expected_version is not None and (type(expected_version) is not int or expected_version != state["version"]):
                return self._save(db, state, operation_id, payload, "blocked", "account_version_changed")
            active_risk = replace(risk, new_entries_enabled=risk.new_entries_enabled and bool(state["entries_enabled"]))
            result = authorize_paper_entry(request, snapshot, active_risk, costs, clock)
            if not result.allowed:
                return self._save(db, state, operation_id, payload, "blocked", result.reason)
            fill_at = _utc(now or datetime.now(timezone.utc))
            if not paper_authorization_is_current(result.authorization, request, snapshot, active_risk, costs, fill_at):
                return self._save(db, state, operation_id, payload, "blocked", "approval_expired_or_inputs_changed")
            approval = result.authorization
            fill = {"operation_id": operation_id, "position_id": operation_id, "side": "BUY", "symbol": request.symbol,
                    "filled_at": _iso(fill_at), "signal_at": _iso(request.signal_at), "reference_price": approval.reference_price,
                    "fill_price": approval.fill_price, "quantity": approval.quantity, "notional": approval.quantity * approval.fill_price,
                    "fee": approval.fee, "price_impact_cost": approval.price_impact_cost, "cash_delta": -approval.cash_outlay, "reason": "paper_entry_approved"}
            due = fill_at + timedelta(seconds=risk.holding_bars * risk.candle_seconds)
            db.execute("INSERT INTO positions VALUES(?,?,?,?,?,?,?)", (request.symbol, approval.quantity, operation_id, _iso(fill_at), _iso(due), approval.reference_price, _iso(request.quote_at)))
            state["cash"] = max(0.0, state["cash"] - approval.cash_outlay)
            state["as_of"] = _iso(fill_at)
            saved = self._save(db, state, operation_id, payload, "filled", "paper_entry_approved", fill, authorization=asdict(approval))
            if not paper_authorization_is_current(approval, request, snapshot, active_risk, costs, _utc(now or datetime.now(timezone.utc))):
                raise ValueError("Paper approval expired before commit; transaction rolled back")
            return saved

    def execute_exit(self, operation_id: str, quote: MarketQuote, *, reason: str = "policy_sell", position_id: str | None = None, now: datetime | None = None):
        """Close a full position with fresh prices; entry limits never block exits."""
        if reason not in ("policy_sell", "horizon", "risk", "manual") or not isinstance(quote, MarketQuote):
            raise ValueError("Require a valid paper exit reason and quote")
        payload = {"kind": "exit", "quote": asdict(quote), "reason": reason, "position_id": position_id, "clock": None if now is None else _iso(now)}
        with self._transaction(write=True) as db:
            state, contract, risk, costs, _, _ = self._audit(db)
            replay = self._replay(db, operation_id, payload)
            if replay:
                return replay
            clock = _utc(now or datetime.now(timezone.utc))
            self._refresh(db, state, contract, risk, (quote,), (), clock)
            position = db.execute("SELECT * FROM positions WHERE symbol=?", (quote.symbol,)).fetchone()
            if position is None:
                return self._save(db, state, operation_id, payload, "blocked", "no_position")
            if position_id is not None and position_id != position["entry_id"]:
                return self._save(db, state, operation_id, payload, "blocked", "position_changed")
            if (clock - _utc(quote.at)).total_seconds() >= risk.max_quote_age_seconds:
                return self._save(db, state, operation_id, payload, "blocked", "stale_exit_quote")
            if reason == "horizon" and _iso(clock) < position["exit_due_at"]:
                return self._save(db, state, operation_id, payload, "blocked", "exit_not_due")
            if reason == "horizon" and _iso(quote.at) < position["exit_due_at"]:
                return self._save(db, state, operation_id, payload, "blocked", "quote_precedes_horizon_exit")
            impact = (costs.spread_bps / 2 + costs.slippage_bps) / 10000
            quantity, fill_price = position["quantity"], quote.price * (1 - impact)
            notional, fee = quantity * fill_price, quantity * fill_price * costs.fee_bps / 10000
            price_cost = quantity * (quote.price - fill_price)
            entry = db.execute("SELECT * FROM fills WHERE operation_id=?", (position["entry_id"],)).fetchone()
            gross = quantity * (quote.price - entry["reference_price"])
            fees, impact_cost = entry["fee"] + fee, entry["price_impact_cost"] + price_cost
            trade = {"entry_id": entry["operation_id"], "exit_id": operation_id, "symbol": quote.symbol, "entry_at": position["entry_at"], "exit_at": _iso(clock), "quantity": quantity,
                     "gross_pnl": gross, "fees": fees, "price_impact_cost": impact_cost, "net_pnl": gross - fees - impact_cost}
            fill = {"operation_id": operation_id, "position_id": entry["operation_id"], "side": "SELL", "symbol": quote.symbol,
                    "filled_at": _iso(clock), "signal_at": _iso(quote.at), "reference_price": quote.price, "fill_price": fill_price,
                    "quantity": quantity, "notional": notional, "fee": fee, "price_impact_cost": price_cost, "cash_delta": notional - fee, "reason": reason}
            state["cash"] += notional - fee
            db.execute("DELETE FROM positions WHERE symbol=?", (quote.symbol,))
            self._refresh(db, state, contract, risk, (), (), clock)
            return self._save(db, state, operation_id, payload, "filled", reason, fill, trade)

    def set_entries_enabled(self, operation_id: str, enabled: bool, *, now: datetime | None = None):
        if type(enabled) is not bool:
            raise ValueError("Entry pause must be boolean")
        payload = {"kind": "entries_enabled", "enabled": enabled, "clock": None if now is None else _iso(now)}
        with self._transaction(write=True) as db:
            state, contract, risk, _, _, _ = self._audit(db)
            replay = self._replay(db, operation_id, payload)
            if replay:
                return replay
            clock = _utc(now or datetime.now(timezone.utc))
            self._refresh(db, state, contract, risk, (), (), clock)
            state["entries_enabled"] = int(enabled)
            return self._save(db, state, operation_id, payload, "updated", "entries_enabled" if enabled else "entries_paused")

    def exit_due_positions(self, quotes: tuple[MarketQuote, ...], *, now: datetime | None = None):
        """Exit due positions using deterministic IDs; safe to retry after restart."""
        clock = _utc(now or datetime.now(timezone.utc))
        report = self.audit()
        with self._transaction() as db:
            _, _, risk, _ = self._account(db)
        marks = self._quotes(quotes, risk, clock)
        return [self.execute_exit("horizon:" + hashlib.sha256(p["entry_id"].encode()).hexdigest(), marks[p["symbol"]], reason="horizon", position_id=p["entry_id"], now=now)
                for p in report["positions"] if p["exit_due_at"] <= _iso(clock) and p["symbol"] in marks]
