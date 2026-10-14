from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import sqlite3

import pandas as pd
import pytest

from local_ai_trader.hyperliquid import diagnostics, train
from local_ai_trader.hyperliquid.forward import ForwardForecastBook
from local_ai_trader.models.train import file_digest
from local_ai_trader.paper.ledger import _hash
from test_forward_forecasts import forecast, SIGNAL
from test_hyperliquid_train import FAST, make_history


@pytest.fixture
def observed(tmp_path, monkeypatch):
    history = make_history(tmp_path, monkeypatch)
    models, manifest = train.train_hyperliquid_models(history, device="cpu", recipe=FAST)
    records = []
    for coin in ("BTC", "ETH"):
        checkpoint, root, identity = train.load_saved_market(models, coin)
        frame = pd.read_parquet(history / coin / "train.parquet").tail(3)
        raw = train.predict_classifiers(checkpoint, frame, root)[checkpoint["primary_model"]]
        probabilities = diagnostics.apply_temperature(raw, checkpoint["temperatures"][checkpoint["primary_model"]]["temperature"])
        returns = train.predict_returns(checkpoint["returns"], frame)
        for index, (_, row) in enumerate(frame.iterrows()):
            record = forecast(coin)
            signal = SIGNAL + timedelta(minutes=5 * index)
            record.update(signal_at=signal.isoformat(), issued_at=(signal+timedelta(seconds=1)).isoformat(),
                          published_at=(signal+timedelta(seconds=2)).isoformat(), calibration_id=identity,
                          features={name: float(row[name]) for name in diagnostics.FEATURE_COLUMNS},
                          probabilities=probabilities[index].tolist(), ridge_return=float(returns["ridge"][index]),
                          mean_return=float(returns["mean"][index]))
            records.append(record)
    contract = {"class_names": list(train.CLASS_NAMES), "source_mode": "observed", "coins": ["BTC", "ETH"],
                "config_sha256": manifest["source_config_sha256"], "model_manifest_sha256": file_digest(models / "models.json")}
    book = ForwardForecastBook(tmp_path / "forward/observations.sqlite", contract)
    for record in records:
        book.record([record], diagnostics._check_prediction(record, contract)+timedelta(seconds=2))
    return models, history, book


def test_replay_does_not_write_book_refit_or_open_test(observed, monkeypatch):
    models, history, book = observed
    before = book.path.read_bytes()
    original = Path.open
    def guarded(path, *args, **kwargs):
        if path.name == "test.parquet":
            pytest.fail("Diagnostic opened reserved test")
        return original(path, *args, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail("Diagnostic refitted a model")
    monkeypatch.setattr(Path, "open", guarded)
    monkeypatch.setattr(train, "_fit_market", forbidden)
    scores, inputs = diagnostics.diagnose(models, book.path)
    assert scores.rows.tolist() == [3, 3]
    assert scores.replay_max_abs_error.max() < 2e-6
    assert len(inputs) == 16
    assert (inputs.outside_train_range_pct == 0).all()
    assert book.path.read_bytes() == before


def test_snapshot_can_run_while_writer_holds_reserved_lock(observed):
    _, _, book = observed
    with sqlite3.connect(str(book.path)) as writer:
        writer.execute("BEGIN IMMEDIATE")
        _, records = diagnostics.read_forecasts(book.path)
        assert len(records) == 6
        writer.rollback()


@pytest.mark.parametrize("change", ["mapping", "model", "prediction", "training"])
def test_diagnostic_refuses_incompatible_or_changed_evidence(observed, change):
    models, history, book = observed
    if change == "training":
        path = history / "BTC/train.parquet"
        path.write_bytes(path.read_bytes() + b"changed")
    else:
        with sqlite3.connect(str(book.path)) as db:
            if change == "prediction":
                key, payload = db.execute("SELECT key,payload FROM forecasts LIMIT 1").fetchone()
                record = json.loads(payload)
                record["probabilities"] = [0.99, 0.005, 0.005]
                db.execute("UPDATE forecasts SET payload=?,digest=? WHERE key=?", (json.dumps(record), _hash(record), key))
            else:
                contract = deepcopy(book.contract)
                contract["class_names" if change == "mapping" else "model_manifest_sha256"] = ["flat", "down", "up"] if change == "mapping" else "other-models"
                db.execute("UPDATE contract SET payload=?,digest=? WHERE id=1", (json.dumps(contract), _hash(contract)))
    with pytest.raises(ValueError):
        diagnostics.diagnose(models, book.path)


def test_missing_observation_book_is_not_created(tmp_path):
    missing = tmp_path / "observations.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        diagnostics.read_forecasts(missing)
    assert not missing.exists()
