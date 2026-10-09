"""Fit temperature on earlier validation labels and assess on later labels."""

from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import scipy

from local_ai_trader.calibration.temperature import apply_temperature, fit_temperature
from local_ai_trader.data.storage import write_json, write_parquet
from local_ai_trader.features.build_features import FEATURE_COLUMNS
from local_ai_trader.models.baseline import CLASS_NAMES
from local_ai_trader.models.evaluate import evaluate_probabilities
from local_ai_trader.models.inference import FrozenPartition, load_frozen_partition, predict_frozen_models, verify_frozen_inputs
from local_ai_trader.models.train import file_digest
from local_ai_trader.settings import CalibrationSettings, load_calibration_settings

LOGGER = logging.getLogger(__name__)
CALIBRATION_FILENAME = "calibration.json"


def validation_regions(frame: pd.DataFrame, fraction: float) -> tuple[dict[str, pd.DataFrame], dict]:
    """Purge fit labels whose outcomes reach the first assessment prediction."""
    if isinstance(fraction, bool) or not isinstance(fraction, (float, int)) or not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Calibration fit fraction must be in (0, 1)")
    frame = frame.reset_index(drop=True)
    cut = int(len(frame) * fraction)
    if cut == 0 or cut == len(frame):
        raise ValueError("Not enough validation rows for calibration and assessment")
    assessment = frame.iloc[cut:].copy()
    earlier = frame.iloc[:cut]
    fit = earlier.loc[earlier["target_available_at"] < assessment["available_at"].iloc[0]].copy()
    if fit.empty:
        raise ValueError("Purging leaves no calibration-fit rows; use more history")
    regions = {"calibration_fit": fit, "assessment": assessment}
    report = {"fit_fraction": fraction, "purge_rule": "target_available_at < assessment_first_available_at", "purged_fit_rows": cut - len(fit)}
    for name, rows in regions.items():
        report[name] = {
            "rows": len(rows), "first_prediction_time": rows["available_at"].iloc[0].isoformat(),
            "last_prediction_time": rows["available_at"].iloc[-1].isoformat(),
            "last_label_known_at": rows["target_available_at"].max().isoformat(),
            "class_counts": {label: int(rows["target_class"].eq(label).sum()) for label in CLASS_NAMES},
        }
    return regions, report


def apply_saved_calibration(run_directory: Path, snapshot: FrozenPartition, raw: dict[str, np.ndarray], artifacts: dict[str, str]) -> tuple[dict[str, np.ndarray], str]:
    """Restore immutable scalar temperatures bound to the original saved model."""
    path = run_directory / "calibration" / CALIBRATION_FILENAME
    digest = file_digest(path)
    with path.open(encoding="utf-8") as source:
        checkpoint = json.load(source)
    if (
        checkpoint.get("schema_version") != 1 or checkpoint.get("method") != "temperature_scaling_on_probabilities"
        or checkpoint.get("feature_names") != list(FEATURE_COLUMNS)
        or checkpoint.get("class_names") != list(CLASS_NAMES)
        or checkpoint.get("base_checkpoint_sha256") != snapshot.checkpoint_hash
        or checkpoint.get("model_artifact_hashes") != artifacts
        or checkpoint.get("source_input_hashes") != snapshot.checkpoint["training_metadata"]["input_hashes"]
        or set(checkpoint["models"]) != set(raw)
    ):
        raise ValueError("Calibration does not match the frozen model, features or original inputs")
    parameters = load_calibration_settings(checkpoint["parameters"])
    known_at = pd.Timestamp(checkpoint["regions"]["calibration_fit"]["last_label_known_at"])
    assessment_start = pd.Timestamp(checkpoint["regions"]["assessment"]["first_prediction_time"])
    if known_at.tzinfo is None or assessment_start.tzinfo is None or not known_at < assessment_start:
        raise ValueError("Calibration metadata does not establish purged temporal separation")
    if snapshot.partition == "test" and not known_at < snapshot.frame["available_at"].iloc[0]:
        raise ValueError("Calibration outcomes overlap the test period")
    adjusted = {}
    for model, values in raw.items():
        temperature = checkpoint["models"][model]["temperature"]
        if isinstance(temperature, bool) or not isinstance(temperature, (float, int)) or not parameters.min_temperature <= temperature <= parameters.max_temperature:
            raise ValueError("Saved temperature is outside its declared bounds")
        adjusted[model] = apply_temperature(values, temperature, parameters.probability_floor)
    if file_digest(path) != digest:
        raise ValueError("Calibration changed during inference; no results published")
    return adjusted, digest


def create_calibration(run_directory: Path, config: CalibrationSettings, split_directory: Path | None = None) -> tuple[Path, dict]:
    """Freeze one calibration bundle; original model and test data remain untouched."""
    config = load_calibration_settings(asdict(config))
    output = run_directory / "calibration"
    if output.exists():
        raise ValueError("Calibration already saved; inspect calibration/metrics.json")
    if (run_directory / "test_evaluation").exists():
        raise ValueError("Test evaluation already exists; calibration must be frozen before test use")
    snapshot = load_frozen_partition(run_directory, "validation", split_directory)
    raw, artifacts = predict_frozen_models(snapshot.checkpoint, snapshot.frame, run_directory)
    regions, region_report = validation_regions(snapshot.frame, config.fit_fraction)
    fit = regions["calibration_fit"]
    fitted = {model: fit_temperature(fit["target_class"], values[fit.index], config) for model, values in raw.items()}
    metrics = {}
    predictions = {}
    for region, rows in regions.items():
        metrics[region] = {}
        for model, values in raw.items():
            selected = values[rows.index]
            adjusted = apply_temperature(selected, fitted[model]["temperature"], config.probability_floor)
            metrics[region][model] = {}
            for variant, scores in (("raw", selected), ("temperature", adjusted)):
                result = evaluate_probabilities(rows["target_class"], scores, snapshot.bins)
                result["calibration_status"] = "uncalibrated" if variant == "raw" else "temperature_scaled"
                metrics[region][model][variant] = result
                if region == "assessment":
                    frame = rows[["timestamp", "available_at", "target_available_at", "symbol", "exchange", "target_class", "target_return"]].copy()
                    for index, label in enumerate(CLASS_NAMES):
                        frame[f"p_{label}"] = scores[:, index]
                    frame["predicted_class"] = np.array(CLASS_NAMES)[scores.argmax(axis=1)]
                    frame["max_probability"] = scores.max(axis=1)
                    predictions[f"assessment_{model}_{variant}.parquet"] = frame
    metadata = snapshot.checkpoint["training_metadata"]
    checkpoint = {
        "schema_version": 1, "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "method": "temperature_scaling_on_probabilities", "run_id": metadata["run_id"],
        "base_checkpoint_sha256": snapshot.checkpoint_hash, "model_artifact_hashes": artifacts,
        "source_input_hashes": metadata["input_hashes"], "source_split_directory": str(snapshot.directory.resolve()),
        "feature_names": list(FEATURE_COLUMNS), "class_names": list(CLASS_NAMES),
        "parameters": asdict(config), "calibration_bins": snapshot.bins,
        "regions": region_report, "models": fitted, "model_weights_refitted": False,
        "test_data_opened": False, "prediction_device": "cpu",
        "assessment_previously_used_for_model_comparison": True,
        "runtime_versions": {"numpy": np.__version__, "scipy": scipy.__version__},
    }
    verify_frozen_inputs(snapshot, run_directory, artifacts)
    staging = Path(tempfile.mkdtemp(prefix=".calibration-staging-", dir=run_directory))
    try:
        write_json(staging / CALIBRATION_FILENAME, checkpoint)
        write_json(staging / "metrics.json", metrics)
        for filename, frame in predictions.items():
            write_parquet(staging / filename, frame)
        verify_frozen_inputs(snapshot, run_directory, artifacts)
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    LOGGER.info("Calibration fit: %s rows; purged %s overlapping labels; assessment: %s rows", len(fit), region_report["purged_fit_rows"], len(regions["assessment"]))
    print("Later-validation calibration assessment (not the reserved test partition):")
    print(pd.DataFrame([
        {"model": model, "variant": variant, "temperature": 1.0 if variant == "raw" else fitted[model]["temperature"], **{name: scores[name] for name in ("accuracy", "f1_macro", "log_loss", "brier_score", "ece")}}
        for model, variants in metrics["assessment"].items() for variant, scores in variants.items()
    ]).to_string(index=False))
    LOGGER.info("Saved calibration to %s; base weights unchanged; test partition not opened", output)
    return output, checkpoint
