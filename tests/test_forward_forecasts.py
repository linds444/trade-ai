from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from local_ai_trader.data.storage import native_path
from local_ai_trader.hyperliquid.forward import ForwardForecastBook, resolve_outcome
from local_ai_trader.features.build_features import FEATURE_COLUMNS

SIGNAL = datetime(2026, 1, 5, 0, 5, tzinfo=timezone.utc)
CONTRACT = {"schema_version": 1, "coins": ["BTC", "ETH"], "mode": "prospective_forecast_observations"}


def forecast(coin="BTC"):
    return {"coin": coin, "signal_at": SIGNAL.isoformat(), "issued_at": (SIGNAL+timedelta(seconds=1)).isoformat(),
            "published_at": (SIGNAL+timedelta(seconds=2)).isoformat(), "primary_model": "mlp" if coin == "BTC" else "xgboost",
            "calibration_id": "sha256:frozen-test-model", "probabilities": [0.8, 0.1, 0.1], "ridge_return": 0.01,
            "mean_return": 0.0, "features": {name: 0.0 for name in FEATURE_COLUMNS}, "preview_path": "test-preview", "preview_sha256": "test-source-hash"}


def candles(coin="BTC", missing=False):
    start = int((SIGNAL+timedelta(minutes=5)).timestamp())*1000
    rows = [{"s": coin, "i": "5m", "t": start+i*300000, "T": start+(i+1)*300000-1,
             "o": str(100+i/3), "c": str(100+i/3), "h": "103", "l": "99", "v": "10", "n": 1} for i in range(7)]
    if missing:
        rows.pop(3)
    return rows


@pytest.fixture
def book(tmp_path):
    return ForwardForecastBook(tmp_path / "observations.sqlite", CONTRACT)


def test_prediction_logged_before_future_entry_and_resolved_only_after_exit_completion(book):
    record = forecast()
    assert book.record([record], SIGNAL+timedelta(seconds=2)) == 1
    assert book.pending(SIGNAL+timedelta(minutes=39, seconds=59)) == []
    key, saved = book.pending(SIGNAL+timedelta(minutes=40))[0]
    result = resolve_outcome(saved, candles(), SIGNAL+timedelta(minutes=40))
    assert result["gross_return"] == pytest.approx(0.02)
    assert result["actual_class"] == "up" and result["correct"] is True
    assert book.settle(key, result) is True
    summary = book.audit()
    assert summary["recorded"] == 1 and summary["resolved"] == 1 and summary["pending"] == 0
    assert summary["observed_accuracy"] == 1
    assert summary["ridge_mae"] == pytest.approx(0.01)
    assert summary["confidence"] is None and summary["reliability_validated"] is False
    assert summary["paper_profitability_trial_started"] is False


def test_early_outcome_is_refused_even_when_fake_future_candles_are_supplied():
    with pytest.raises(ValueError, match="not yet known"):
        resolve_outcome(forecast(), candles(), SIGNAL+timedelta(minutes=39))


def test_late_tracker_cannot_retrofit_old_predictions(book):
    assert book.record([forecast()], SIGNAL+timedelta(minutes=5)) == 0
    assert book.summary()["recorded"] == 0


@pytest.mark.parametrize("field,value", [
    ("issued_at", (SIGNAL+timedelta(minutes=5)).isoformat()),
    ("issued_at", (SIGNAL-timedelta(seconds=1)).isoformat()),
    ("published_at", (SIGNAL+timedelta(minutes=5)).isoformat()),
    ("primary_model", "xgboost"), ("probabilities", [0.8, 0.8, 0.1]),
    ("ridge_return", float("nan")), ("coin", "UNKNOWN"),
])
def test_invalid_or_post_entry_forecasts_are_refused(book, field, value):
    record = forecast()
    record[field] = value
    with pytest.raises(ValueError):
        book.record([record], SIGNAL+timedelta(seconds=2))
    assert book.summary()["recorded"] == 0


def test_restart_duplicates_do_not_add_predictions_or_outcomes(book):
    record = forecast()
    book.record([record], SIGNAL+timedelta(seconds=2))
    restarted = ForwardForecastBook(book.path, CONTRACT)
    newer = {**record, "issued_at": (SIGNAL+timedelta(seconds=3)).isoformat(), "published_at": (SIGNAL+timedelta(seconds=4)).isoformat(), "preview_path": "new-preview"}
    assert restarted.record([newer], SIGNAL+timedelta(seconds=4)) == 0
    key, original = restarted.pending(SIGNAL+timedelta(minutes=40))[0]
    result = resolve_outcome(original, candles(), SIGNAL+timedelta(minutes=40))
    assert restarted.settle(key, result) is True
    assert ForwardForecastBook(book.path, CONTRACT).settle(key, result) is False
    assert restarted.summary()["resolved"] == 1


def test_changed_frozen_prediction_is_refused(book):
    book.record([forecast()], SIGNAL+timedelta(seconds=2))
    changed = forecast()
    changed["ridge_return"] = 0.02
    with pytest.raises(ValueError, match="Changed forward prediction"):
        book.record([changed], SIGNAL+timedelta(seconds=3))


def test_gap_never_resolves_a_future_return(book):
    book.record([forecast()], SIGNAL+timedelta(seconds=2))
    with pytest.raises(ValueError, match="missing completed candles"):
        resolve_outcome(forecast(), candles(missing=True), SIGNAL+timedelta(minutes=40))
    assert book.summary()["resolved"] == 0


def test_changed_settled_economics_is_refused(book):
    book.record([forecast()], SIGNAL+timedelta(seconds=2))
    key, record = book.pending(SIGNAL+timedelta(minutes=40))[0]
    original = resolve_outcome(record, candles(), SIGNAL+timedelta(minutes=40))
    book.settle(key, original)
    altered = candles()
    altered[-1]["o"] = "101"
    changed = resolve_outcome(record, altered, SIGNAL+timedelta(minutes=41))
    with pytest.raises(ValueError, match="Changed resolved"):
        book.settle(key, changed)


def test_unrecorded_and_manually_changed_outcomes_are_refused(book):
    result = resolve_outcome(forecast(), candles(), SIGNAL+timedelta(minutes=40))
    with pytest.raises(ValueError, match="lacks a recorded"):
        book.settle("not-recorded", result)
    book.record([forecast()], SIGNAL+timedelta(seconds=2))
    key, _ = book.pending(SIGNAL+timedelta(minutes=40))[0]
    result["gross_return"] = 10
    with pytest.raises(ValueError, match="disagrees"):
        book.settle(key, result)


def test_concurrent_record_delivery_commits_once(book):
    with ThreadPoolExecutor(max_workers=4) as pool:
        counts = list(pool.map(lambda _: book.record([forecast()], SIGNAL+timedelta(seconds=2)), range(4)))
    assert sum(counts) == 1
    assert book.audit()["recorded"] == 1


def test_batch_failure_rolls_back_prior_forecasts(book):
    bad = forecast("ETH")
    bad["probabilities"] = [1, 1, 1]
    with pytest.raises(ValueError):
        book.record([forecast(), bad], SIGNAL+timedelta(seconds=2))
    assert book.summary()["recorded"] == 0


@pytest.mark.parametrize("table,column,value", [("forecasts", "digest", "changed"), ("outcomes", "digest", "changed"), ("outcomes", "correct", 0)])
def test_corruption_of_records_or_diagnostic_counters_is_refused(book, table, column, value):
    book.record([forecast()], SIGNAL+timedelta(seconds=2))
    key, record = book.pending(SIGNAL+timedelta(minutes=40))[0]
    book.settle(key, resolve_outcome(record, candles(), SIGNAL+timedelta(minutes=40)))
    with sqlite3.connect(str(book.path)) as db:
        db.execute(f"UPDATE {table} SET {column}=?", (value,))
    with pytest.raises(ValueError):
        ForwardForecastBook(book.path, CONTRACT)


def test_changed_model_contract_cannot_reuse_observations(book):
    with pytest.raises(ValueError, match="contract changed"):
        ForwardForecastBook(book.path, {**CONTRACT, "coins": ["ETH"]})


def test_deleting_observations_cannot_silently_improve_diagnostics(book):
    book.record([forecast()], SIGNAL+timedelta(seconds=2))
    with sqlite3.connect(str(book.path)) as db:
        db.execute("DELETE FROM forecasts")
    with pytest.raises(ValueError, match="removed/changed"):
        book.summary()


def test_native_deep_windows_paths(tmp_path):
    path = native_path(tmp_path / ("nested-"+"x"*80) / ("deeper-"+"y"*80) / ("leaf-"+"z"*80) / "observations.sqlite")
    book = ForwardForecastBook(path, CONTRACT)
    book.record([forecast()], SIGNAL+timedelta(seconds=2))
    assert len(str(path)) > 310
    assert ForwardForecastBook(path, CONTRACT).audit()["recorded"] == 1
