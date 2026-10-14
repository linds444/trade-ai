"""Read-only model replay and descriptive input comparisons; no qualification."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from local_ai_trader.calibration.temperature import apply_temperature
from local_ai_trader.data.storage import native_path
from local_ai_trader.features.build_features import FEATURE_COLUMNS
from local_ai_trader.hyperliquid.forward import _check_prediction
from local_ai_trader.hyperliquid.markets import BY_COIN
from local_ai_trader.hyperliquid.train import load_saved_market, predict_classifiers, predict_returns
from local_ai_trader.models.baseline import CLASS_NAMES, feature_matrix
from local_ai_trader.models.train import file_digest
from local_ai_trader.paper.ledger import _hash, _iso


def read_forecasts(path, *, signal_since=None):
    """Take a short consistent snapshot without invoking the writable book API."""
    path = native_path(Path(path)).resolve()
    # An extended Windows path's as_uri() treats '?' as a network authority.
    # Use the ordinary drive/UNC namespace for SQLite's read-only file URI.
    uri_path = str(path)
    if uri_path.startswith("\\\\?\\UNC\\"):
        uri_path = "\\\\" + uri_path[8:]
    elif uri_path.startswith("\\\\?\\"):
        uri_path = uri_path[4:]
    with closing(sqlite3.connect(Path(uri_path).as_uri() + "?mode=ro", uri=True, timeout=10)) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        contract_row = db.execute("SELECT payload,digest,recorded_count,resolved_count,counters_digest FROM contract WHERE id=1").fetchone()
        total = db.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
        if signal_since is None:
            rows = db.execute("SELECT key,coin,payload,digest FROM forecasts ORDER BY key").fetchall()
        else:
            from datetime import timedelta
            rows = db.execute("SELECT key,coin,payload,digest FROM forecasts WHERE known_at>=? ORDER BY key",
                              (_iso(signal_since + timedelta(minutes=40)),)).fetchall()
        resolved = db.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
    if contract_row is None:
        raise ValueError("Missing forward observation contract")
    contract = json.loads(contract_row[0])
    counts = {"recorded": total, "resolved": resolved}
    if contract_row[1] != _hash(contract) or tuple(contract_row[2:4]) != (total, resolved) or contract_row[4] != _hash(counts):
        raise ValueError("Forward contract or observation counters changed")
    if contract["class_names"] != list(CLASS_NAMES) or contract["source_mode"] != "observed":
        raise ValueError("Forward class mapping/source disagrees")
    records = []
    for key, coin, payload, digest in rows:
        record = json.loads(payload)
        signal = _check_prediction(record, contract)
        if digest != _hash(record) or coin != record["coin"] or key != coin + "|" + _iso(signal):
            raise ValueError("Forward forecast evidence changed")
        records.append(record)
    return contract, records


def diagnose(model_directory, book_path):
    """Replay frozen forecasts; read training and saved assessment only, never test."""
    model_directory = native_path(Path(model_directory)).resolve()
    manifest_path = model_directory / "models.json"
    manifest_digest = file_digest(manifest_path)
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract, records = read_forecasts(book_path)
    if contract["model_manifest_sha256"] != manifest_digest or contract["coins"] != report["coins"] or contract["config_sha256"] != report["source_config_sha256"]:
        raise ValueError("Forward observations belong to different models/configuration")
    history = native_path(Path(report["source_history_directory"]))
    if file_digest(history / "history.json") != report["history_manifest_sha256"]:
        raise ValueError("Original training history manifest changed")
    scores, inputs = [], []
    for coin in report["coins"]:
        selected = [record for record in records if record["coin"] == coin]
        if not selected:
            continue
        checkpoint, root, calibration_id = load_saved_market(model_directory, coin)
        if any(record["calibration_id"] != calibration_id or record["primary_model"] != checkpoint["primary_model"] for record in selected):
            raise ValueError("Forward forecast model provenance disagrees")
        primary = checkpoint["primary_model"]
        frame = pd.DataFrame([record["features"] for record in selected])
        raw = predict_classifiers(checkpoint, frame, root)[primary]
        temperature = checkpoint["temperatures"][primary]["temperature"]
        calibrated = apply_temperature(raw, temperature)
        saved = np.asarray([record["probabilities"] for record in selected])
        # CPU replay of GPU-trained MLPs may differ slightly in floating point.
        if not np.allclose(calibrated, saved, rtol=1e-5, atol=2e-6):
            raise ValueError(f"{coin}: saved probabilities do not replay from frozen features")
        returns = predict_returns(checkpoint["returns"], frame)
        for name in ("ridge", "mean"):
            if not np.allclose(returns[name], [record[f"{name}_return"] for record in selected], rtol=1e-9, atol=1e-12):
                raise ValueError(f"{coin}: saved {name} return forecasts do not replay")
        train_path = history / BY_COIN[coin].filename / "train.parquet"
        if report["input_hashes"].get(str(train_path)) != file_digest(train_path):
            raise ValueError(f"{coin}: original training inputs changed or moved")
        training = pd.read_parquet(train_path)
        assessment = pd.read_parquet(root / "assessment_predictions.parquet")
        reference, observed = feature_matrix(training), feature_matrix(frame)
        scores.append({"coin": coin, "rows": len(frame), "temperature": temperature,
                       "train_up_pct": 100 * training.target_class.eq("up").mean(),
                       "assessment_max_up_pct": 100 * assessment[f"{primary}_temperature_p_up"].max(),
                       "live_raw_max_up_pct": 100 * raw[:, 0].max(),
                       "live_scaled_max_up_pct": 100 * saved[:, 0].max(),
                       "raw_60_plus": int((raw[:, 0] >= 0.6).sum()),
                       "scaled_60_plus": int((saved[:, 0] >= 0.6).sum()),
                       "replay_max_abs_error": float(np.abs(calibrated - saved).max())})
        for index, name in enumerate(FEATURE_COLUMNS):
            lower, upper = reference[:, index].min(), reference[:, index].max()
            inputs.append({"coin": coin, "feature": name,
                           "train_median": float(np.median(reference[:, index])),
                           "live_median": float(np.median(observed[:, index])),
                           "outside_train_range_pct": float(100 * ((observed[:, index] < lower) | (observed[:, index] > upper)).mean())})
        # Input files are immutable; fail if modified during the diagnostic.
        if file_digest(train_path) != report["input_hashes"][str(train_path)]:
            raise ValueError("Training input changed during diagnosis")
        load_saved_market(model_directory, coin)
    if file_digest(manifest_path) != manifest_digest:
        raise ValueError("Model manifest changed during diagnosis")
    return pd.DataFrame(scores), pd.DataFrame(inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True)
    parser.add_argument("--book", required=True)
    args = parser.parse_args()
    scores, inputs = diagnose(args.models, args.book)
    print("Frozen-model replay and descriptive comparisons (UP = gross return > 0.2%):")
    print(scores.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print("Training versus recorded live inputs (range comparisons are descriptive, not drift certification):")
    print(inputs.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print("PASS: frozen probabilities and return forecasts replay; observation book opened read-only; reserved test unopened")
    print("No fitting, threshold changes, reliability approval, orders or paper trial start. Outcomes are not assessed for net profit.")


if __name__ == "__main__":
    main()
