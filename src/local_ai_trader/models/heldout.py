"""Evaluate frozen models and optional saved temperatures on held-out data."""

import logging
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from local_ai_trader.data.storage import write_json, write_parquet
from local_ai_trader.features.build_features import FEATURE_COLUMNS
from local_ai_trader.models.baseline import CLASS_NAMES
from local_ai_trader.models.evaluate import evaluate_probabilities
from local_ai_trader.models.inference import load_frozen_partition, predict_frozen_models, verify_frozen_inputs
from local_ai_trader.models.train import file_digest

LOGGER = logging.getLogger(__name__)


def evaluate_heldout(run_directory: Path, split_directory: Path | None = None, calibrated: bool = False) -> tuple[Path, dict]:
    """Use saved weights/scaling/temperatures only; publish one test evaluation.

    Current settings are not read and no parameter is fitted. Calibrated mode
    reports raw and temperature-adjusted predictions together. Both modes use
    the same immutable output directory, preventing a second test evaluation.
    """
    output = run_directory / "test_evaluation"
    if output.exists():
        raise ValueError("Test evaluation already saved; inspect test_evaluation/metrics.json")
    calibration_path = run_directory / "calibration" / "calibration.json"
    if calibrated and not calibration_path.is_file():
        raise ValueError("Saved calibration required; calibrate before evaluating test data")
    snapshot = load_frozen_partition(run_directory, "test", split_directory)
    checkpoint, frame = snapshot.checkpoint, snapshot.frame
    metadata = checkpoint["training_metadata"]
    probabilities, model_hashes = predict_frozen_models(checkpoint, frame, run_directory)
    calibration_hash = None
    adjusted_names = set()
    if calibrated:
        from local_ai_trader.calibration.run import apply_saved_calibration

        adjusted, calibration_hash = apply_saved_calibration(run_directory, snapshot, probabilities, model_hashes)
        for model, values in adjusted.items():
            name = f"{model}_temperature"
            probabilities[name] = values
            adjusted_names.add(name)
    metrics = {}
    for model, values in probabilities.items():
        metrics[model] = evaluate_probabilities(frame["target_class"], values, snapshot.bins)
        if model in adjusted_names:
            metrics[model]["calibration_status"] = "temperature_scaled"
    report = {
        "schema_version": 1, "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "partition": "test", "run_id": metadata["run_id"], "rows": len(frame),
        "checkpoint_sha256": snapshot.checkpoint_hash,
        "split_manifest_sha256": metadata["input_hashes"]["split.json"],
        "test_parquet_sha256": snapshot.parquet_hash,
        "split_directory": str(snapshot.directory.resolve()), "refitted": False,
        "feature_names": list(FEATURE_COLUMNS), "class_names": list(CLASS_NAMES),
        "calibration_status": "raw_and_temperature_scaled" if calibrated else checkpoint["calibration"],
        "metrics": metrics, "model_artifact_hashes": model_hashes,
        "calibration_checkpoint_sha256": calibration_hash,
        "calibration_fitted_during_evaluation": False,
    }
    verify_frozen_inputs(snapshot, run_directory, model_hashes)
    staging = Path(tempfile.mkdtemp(prefix=".test-staging-", dir=run_directory))
    try:
        write_json(staging / "metrics.json", report)
        for model, values in probabilities.items():
            predicted = frame[["timestamp", "available_at", "target_available_at", "symbol", "exchange", "target_class", "target_return"]].copy()
            for index, label in enumerate(CLASS_NAMES):
                predicted[f"p_{label}"] = values[:, index]
            predicted["predicted_class"] = np.array(CLASS_NAMES)[values.argmax(axis=1)]
            predicted["max_probability"] = values.max(axis=1)
            write_parquet(staging / f"{model}.parquet", predicted)
        verify_frozen_inputs(snapshot, run_directory, model_hashes)
        if calibrated and file_digest(calibration_path) != calibration_hash:
            raise ValueError("Calibration changed during evaluation; no results published")
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    LOGGER.info("Evaluated %s held-out test rows using saved parameters; no fitting", len(frame))
    description = "raw and temperature-scaled probabilities" if calibrated else "probabilities are uncalibrated"
    print(f"Held-out test metrics ({description}):")
    print(pd.DataFrame([
        {"model": model, **{key: metrics[model][key] for key in ("accuracy", "f1_macro", "log_loss", "brier_score", "ece")}}
        for model in metrics
    ]).to_string(index=False))
    LOGGER.info("Saved held-out evaluation to %s", output)
    return output, report
