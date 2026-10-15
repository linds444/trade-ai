"""Prospective forecast observations; no account, fitting or reliability approval."""

from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import sqlite3

import numpy as np

from local_ai_trader.data.storage import native_path
from local_ai_trader.features.build_features import FEATURE_COLUMNS
from local_ai_trader.hyperliquid.feed import parse_candles
from local_ai_trader.hyperliquid.funding import _publication_lock
from local_ai_trader.hyperliquid.markets import BY_COIN
from local_ai_trader.models.baseline import CLASS_NAMES
from local_ai_trader.models.train import file_digest
from local_ai_trader.paper.ledger import _hash, _iso, _json
from local_ai_trader.risk.paper import _utc

LOGGER = logging.getLogger(__name__)
SCHEMA = """
CREATE TABLE contract(id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL, digest TEXT NOT NULL,
 recorded_count INTEGER NOT NULL, resolved_count INTEGER NOT NULL, counters_digest TEXT NOT NULL);
CREATE TABLE forecasts(key TEXT PRIMARY KEY, coin TEXT NOT NULL, known_at TEXT NOT NULL, payload TEXT NOT NULL, digest TEXT NOT NULL);
CREATE TABLE outcomes(key TEXT PRIMARY KEY REFERENCES forecasts(key), payload TEXT NOT NULL, digest TEXT NOT NULL,
 correct INTEGER NOT NULL, up_candidate INTEGER NOT NULL, up_hit INTEGER NOT NULL,
 ridge_error REAL NOT NULL, mean_error REAL NOT NULL);
CREATE INDEX forecast_due ON forecasts(known_at);
"""


def _time(value):
    return _utc(datetime.fromisoformat(value))


def _check_prediction(record, contract):
    coin, signal, issued = record["coin"], _time(record["signal_at"]), _time(record["issued_at"])
    if coin not in contract["coins"] or signal.timestamp() % 300 or not signal <= issued < signal+timedelta(minutes=5):
        raise ValueError("Forward prediction must be issued after its signal and before future entry")
    if record["primary_model"] != ("mlp" if coin == "BTC" else "xgboost") or not isinstance(record["calibration_id"], str) or not record["calibration_id"].startswith("sha256:"):
        raise ValueError("Forward prediction lacks frozen primary provenance")
    probabilities = np.asarray(record["probabilities"])
    if probabilities.shape != (3,) or probabilities.dtype.kind not in "fiu" or not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any() or not np.isclose(probabilities.sum(), 1, rtol=0, atol=1e-8):
        raise ValueError("Forward probabilities must use normalized up/down/flat order")
    for name in ("ridge_return", "mean_return"):
        value = record[name]
        if type(value) not in (int, float) or not np.isfinite(value) or value <= -1:
            raise ValueError("Invalid forward point return forecast")
    if set(record["features"]) != set(FEATURE_COLUMNS):
        raise ValueError("Forward prediction requires exactly the causal feature allowlist")
    features = np.asarray([record["features"][name] for name in FEATURE_COLUMNS])
    if features.dtype.kind not in "fiu" or not np.isfinite(features).all():
        raise ValueError("Forward input features must be finite real values")
    published = _time(record["published_at"])
    if not issued <= published < signal+timedelta(minutes=5):
        raise ValueError("Forward prediction was published after future entry")
    if not record.get("preview_sha256") or not record.get("preview_path"):
        raise ValueError("Forward prediction requires archived source evidence")
    return signal


def _economics(record):
    return {key: value for key, value in record.items() if key not in ("issued_at", "published_at", "preview_path", "preview_sha256")}


def resolve_outcome(record, candles, collected_at):
    signal = _time(record["signal_at"])
    known = signal+timedelta(minutes=40)
    now = _utc(collected_at)
    if now < known:
        raise ValueError("Forward outcome is not yet known")
    start, end = int((signal+timedelta(minutes=5)).timestamp()), int(known.timestamp())
    frame, quality = parse_candles(candles, record["coin"], start, end)
    gross = float(frame.open.iloc[-1]/frame.open.iloc[0]-1)
    actual = "up" if gross > 0.002 else "down" if gross < -0.002 else "flat"
    predicted = CLASS_NAMES[int(np.argmax(record["probabilities"]))]
    return {"coin": record["coin"], "signal_at": record["signal_at"], "collected_at": _iso(now),
            "entry_at": _iso(signal+timedelta(minutes=5)), "exit_at": _iso(signal+timedelta(minutes=35)),
            "label_available_at": _iso(known), "gross_return": gross, "actual_class": actual,
            "predicted_class": predicted, "correct": predicted == actual,
            "up_candidate": record["probabilities"][0] >= 0.6, "up_hit": actual == "up",
            "ridge_absolute_error": abs(record["ridge_return"]-gross), "mean_absolute_error": abs(record["mean_return"]-gross),
            "raw_candles": candles, "missing_candles": quality.missing_count,
            "interpretation": "overlapping forecast outcomes; no realized profit or confidence certification"}


class ForwardForecastBook:
    """Immediate SQLite transactions, immutable forecasts and idempotent outcomes."""

    def __init__(self, path, contract):
        self.path, self.contract = native_path(Path(path)).resolve(), contract
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _publication_lock(self.path.parent):
            if not self.path.exists():
                try:
                    with closing(sqlite3.connect(str(self.path))) as db:
                        db.executescript(SCHEMA)
                        with db:
                            db.execute("INSERT INTO contract VALUES(1,?,?,0,0,?)", (_json(contract), _hash(contract), _hash({"recorded": 0, "resolved": 0})))
                except BaseException:
                    self.path.unlink(missing_ok=True)
                    raise
            self.audit()

    @contextmanager
    def transaction(self):
        with closing(sqlite3.connect(str(self.path), timeout=10)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute("SELECT * FROM contract WHERE id=1").fetchone()
                if row is None or row["payload"] != _json(self.contract) or row["digest"] != _hash(self.contract):
                    raise ValueError("Forward observation contract changed")
                counts = {"recorded": db.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0],
                          "resolved": db.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]}
                if row["recorded_count"] != counts["recorded"] or row["resolved_count"] != counts["resolved"] or row["counters_digest"] != _hash(counts):
                    raise ValueError("Forward observation rows or counters were removed/changed")
                yield db
                counts = {"recorded": db.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0],
                          "resolved": db.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]}
                db.execute("UPDATE contract SET recorded_count=?,resolved_count=?,counters_digest=? WHERE id=1", (counts["recorded"], counts["resolved"], _hash(counts)))
                db.commit()
            except BaseException:
                db.rollback()
                raise

    def record(self, records, now):
        now = _utc(now)
        inserted = 0
        with self.transaction() as db:
            for record in records:
                signal = _check_prediction(record, self.contract)
                key = record["coin"]+"|"+_iso(signal)
                existing = db.execute("SELECT * FROM forecasts WHERE key=?", (key,)).fetchone()
                if existing:
                    saved = json.loads(existing["payload"])
                    if existing["digest"] != _hash(saved) or _economics(saved) != _economics(record):
                        raise ValueError("Changed forward prediction for the same frozen signal")
                    continue
                if not _time(record["published_at"]) <= now < signal+timedelta(minutes=5):
                    # Recovery cannot retrofit unrecorded predictions after entry.
                    continue
                db.execute("INSERT INTO forecasts VALUES(?,?,?,?,?)", (key, record["coin"], _iso(signal+timedelta(minutes=40)), _json(record), _hash(record)))
                inserted += 1
        return inserted

    def pending(self, now):
        with self.transaction() as db:
            rows = db.execute("SELECT f.* FROM forecasts f LEFT JOIN outcomes o ON f.key=o.key WHERE o.key IS NULL AND f.known_at<=? ORDER BY f.known_at,f.key", (_iso(_utc(now)),)).fetchall()
        result = []
        for row in rows:
            value = json.loads(row["payload"])
            if row["digest"] != _hash(value):
                raise ValueError("Forward prediction changed")
            result.append((row["key"], value))
        return result

    def settle(self, key, outcome):
        with self.transaction() as db:
            row = db.execute("SELECT * FROM forecasts WHERE key=?", (key,)).fetchone()
            if row is None:
                raise ValueError("Outcome lacks a recorded forward prediction")
            record = json.loads(row["payload"])
            if row["digest"] != _hash(record) or resolve_outcome(record, outcome["raw_candles"], _time(outcome["collected_at"])) != outcome:
                raise ValueError("Forward outcome disagrees with recorded prediction and raw candles")
            previous = db.execute("SELECT * FROM outcomes WHERE key=?", (key,)).fetchone()
            if previous:
                saved = json.loads(previous["payload"])
                if previous["digest"] != _hash(saved) or {k: v for k, v in saved.items() if k != "collected_at"} != {k: v for k, v in outcome.items() if k != "collected_at"}:
                    raise ValueError("Changed resolved forward outcome")
                return False
            db.execute("INSERT INTO outcomes VALUES(?,?,?,?,?,?,?,?)", (key, _json(outcome), _hash(outcome),
                       int(outcome["correct"]), int(outcome["up_candidate"]), int(outcome["up_hit"]),
                       outcome["ridge_absolute_error"], outcome["mean_absolute_error"]))
            return True

    def audit(self):
        with self.transaction() as db:
            records = {row["key"]: row for row in db.execute("SELECT * FROM forecasts")}
            outcomes = list(db.execute("SELECT * FROM outcomes"))
        for key, row in records.items():
            record = json.loads(row["payload"])
            signal = _check_prediction(record, self.contract)
            if row["digest"] != _hash(record) or key != record["coin"]+"|"+_iso(signal) or row["known_at"] != _iso(signal+timedelta(minutes=40)) or row["coin"] != record["coin"]:
                raise ValueError("Forward prediction audit failed")
        for row in outcomes:
            outcome = json.loads(row["payload"])
            if row["key"] not in records or row["digest"] != _hash(outcome) or resolve_outcome(json.loads(records[row["key"]]["payload"]), outcome["raw_candles"], _time(outcome["collected_at"])) != outcome:
                raise ValueError("Forward outcome audit failed")
            if any(row[column] != outcome[key] for column, key in (("correct", "correct"), ("up_candidate", "up_candidate"), ("up_hit", "up_hit"), ("ridge_error", "ridge_absolute_error"), ("mean_error", "mean_absolute_error"))):
                raise ValueError("Forward diagnostic counters disagree with audited outcomes")
        return self.summary()

    def summary(self):
        with self.transaction() as db:
            total = db.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
            values = db.execute("SELECT COUNT(*),AVG(correct),SUM(up_candidate),SUM(up_candidate*up_hit),AVG(ridge_error),AVG(mean_error) FROM outcomes").fetchone()
        return {"recorded": total, "resolved": values[0], "pending": total-values[0],
                "observed_accuracy": values[1], "up_probability_ge_60_rows": values[2] or 0,
                "up_probability_ge_60_hits": values[3] or 0, "ridge_mae": values[4], "mean_mae": values[5],
                "confidence": None, "reliability_validated": False, "paper_profitability_trial_started": False}


def track_preview(client, preview_path, report):
    """Save before future entry, then resolve already-due forecasts without fitting."""
    contract = {"schema_version": 1, "storage_schema": 2, "mode": "prospective_forecast_observations", "source_mode": "observed",
                "coins": list(client.settings.coins), "model_manifest_sha256": report["model_manifest_sha256"],
                "config_sha256": report["input_config_sha256"], "class_names": list(CLASS_NAMES),
                "entry_delay_bars": 1, "holding_bars": 6, "label_known_delay_bars": 8, "class_threshold": 0.002}
    name = _hash(contract)[:16]
    path = client.settings.observations.parent / "forward" / name / "observations.sqlite"
    cache = getattr(client, "_forward_books", {})
    if name not in cache:
        cache[name] = ForwardForecastBook(path, contract)
        client._forward_books = cache
    book = cache[name]
    published_at = _utc(client.clock())
    digest = file_digest(preview_path / "decisions.json")
    with (preview_path / "decisions.json").open(encoding="utf-8") as stream:
        if json.load(stream) != report:
            raise ValueError("Forward preview changed before recording")
    records = [{**value, "published_at": _iso(published_at), "preview_path": str(preview_path.resolve()), "preview_sha256": digest}
               for value in report.get("pending_forward_predictions", {}).values()]
    # A delayed tracker cannot turn old, already-entered observations into new
    # forward predictions. Its original immutable preview remains available.
    records = [r for r in records if published_at < _time(r["signal_at"])+timedelta(minutes=5)]
    book.record(records, published_at)
    with (preview_path / "raw_market.json").open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if file_digest(preview_path / "raw_market.json") != report["artifact_hashes"]["raw_market.json"]:
        raise ValueError("Forward outcome source changed")
    for key, record in book.pending(_utc(client.clock())):
        signal = _time(record["signal_at"])
        start, end = int((signal+timedelta(minutes=5)).timestamp()), int((signal+timedelta(minutes=40)).timestamp())
        try:
            source = raw[f"{record['coin']}_candles"]
            try:
                parse_candles(source, record["coin"], start, end)
            except ValueError:
                source = client.info({"type": "candleSnapshot", "req": {"coin": record["coin"], "interval": "5m", "startTime": start*1000, "endTime": end*1000}})
            outcome = resolve_outcome(record, source, _utc(client.clock()))
            book.settle(key, outcome)
        except (ValueError, RuntimeError, OSError) as error:
            LOGGER.warning("Forward outcome remains pending for %s: %s", key, error)
    summary = book.summary()
    print(f"Forward forecast observations: {summary['recorded']} recorded; {summary['pending']} pending; {summary['resolved']} resolved")
    if summary["resolved"]:
        print(f"Descriptive accuracy: {summary['observed_accuracy']:.4f}; up>=60% hits: {summary['up_probability_ge_60_hits']}/{summary['up_probability_ge_60_rows']}; Ridge MAE: {summary['ridge_mae']:.6f}")
    print("Forward outcomes are forecast diagnostics; overlapping rows are not independent samples, reliability confidence or paper P&L")
    print(f"Saved forward observation book to {path}")
    return path, summary
