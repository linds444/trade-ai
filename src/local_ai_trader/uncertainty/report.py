"""Describe uncertainty in already evaluated, immutable walk-forward predictions."""

import logging
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from local_ai_trader.data.storage import write_json, write_parquet
from local_ai_trader.models.baseline import CLASS_NAMES, encode_labels
from local_ai_trader.models.train import file_digest
from local_ai_trader.uncertainty.scores import model_disagreement, uncertainty_scores
from local_ai_trader.walkforward.completed import load_completed_execution
from local_ai_trader.walkforward.results import TRUTH_COLUMNS
from local_ai_trader.walkforward.run import read_json

LOGGER = logging.getLogger(__name__)
PROBABILITY_COLUMNS = [f"p_{label}" for label in CLASS_NAMES]
ALIGNMENT_COLUMNS = [*TRUTH_COLUMNS, "fold"]


def align_predictions(first: pd.DataFrame, second: pd.DataFrame) -> None:
    """Require equal timed outcomes and folds before comparing model distributions."""
    try:
        pd.testing.assert_frame_equal(first[ALIGNMENT_COLUMNS].reset_index(drop=True), second[ALIGNMENT_COLUMNS].reset_index(drop=True))
    except (AssertionError, KeyError) as error:
        raise ValueError("Uncertainty comparison predictions have mismatched times, markets, outcomes or folds") from error


def summarize_scores(frame: pd.DataFrame, scores: dict[str, np.ndarray]) -> dict:
    """Fixed descriptive entropy bands assess reused outcomes, never select gates."""
    probabilities = frame[PROBABILITY_COLUMNS].to_numpy()
    labels = encode_labels(frame.target_class)
    correct = probabilities.argmax(axis=1) == labels
    bands = np.minimum((scores["entropy_normalized"] * 5).astype(int), 4)
    result = {"rows": len(frame)}
    for name, values in scores.items():
        result[f"mean_{name}"] = float(values.mean())
        result[f"p90_{name}"] = float(np.quantile(values, 0.9))
    result["entropy_bands"] = [
        {
            "lower_inclusive": index / 5, "upper": (index + 1) / 5,
            "upper_inclusive": index == 4, "rows": int((bands == index).sum()),
            "observed_accuracy": float(correct[bands == index].mean()) if (bands == index).any() else None,
        }
        for index in range(5)
    ]
    return result


def create_uncertainty_report(directory: Path) -> tuple[Path, dict]:
    """Publish one report without fitting, rescoring models or changing decisions."""
    snapshot = load_completed_execution(directory)
    output = snapshot.output / "uncertainty"
    if output.exists():
        raise ValueError("Uncertainty report already exists; inspect uncertainty/uncertainty.json")
    summary_path = snapshot.output / "summary/summary.json"
    summary = read_json(summary_path)
    selected = snapshot.protocol.plan["model_family"]
    frames, scores, metrics, comparisons, sources = {}, {}, {}, {}, {}
    for model in ("naive", "logistic", selected):
        for variant in ("raw", "temperature"):
            name = f"{model}_{variant}"
            path = snapshot.output / "summary" / f"{name}_predictions.parquet"
            frame = pd.read_parquet(path)
            if len(frame) != summary["evaluation_rows"] or frame.available_at.duplicated().any() or not frame.available_at.is_monotonic_increasing:
                raise ValueError("Uncertainty source predictions have incompatible or overlapping rows")
            values = frame[PROBABILITY_COLUMNS].to_numpy()
            values_scores = uncertainty_scores(values)
            if not frame.predicted_class.eq(np.array(CLASS_NAMES)[values.argmax(axis=1)]).all():
                raise ValueError("Saved predicted classes disagree with saved probabilities")
            if not np.allclose(frame.max_probability, values_scores["top_class_probability"], rtol=0, atol=1e-12):
                raise ValueError("Saved maximum probabilities disagree with saved distributions")
            frames[name], scores[name] = frame, values_scores
            metrics[name] = summarize_scores(frame, values_scores)
            sources[path.name] = file_digest(path)
    for frame in frames.values():
        align_predictions(frames[f"{selected}_raw"], frame)
    for variant in ("raw", "temperature"):
        name, reference = f"{selected}_{variant}", f"logistic_{variant}"
        chosen, control = frames[name], frames[reference]
        divergence = model_disagreement(chosen[PROBABILITY_COLUMNS].to_numpy(), control[PROBABILITY_COLUMNS].to_numpy())
        comparisons[variant] = {
            "members": [name, reference], "rows": len(chosen),
            "mean_jensen_shannon_normalized": float(divergence.mean()),
            "p90_jensen_shannon_normalized": float(np.quantile(divergence, 0.9)),
            "different_top_class_fraction": float(chosen.predicted_class.ne(control.predicted_class).mean()),
        }
        scores[name]["logistic_disagreement_normalized"] = divergence
    report = {
        "schema_version": 1, "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "research_only": True, "symbol": snapshot.protocol.plan["symbol"],
        "selected_family": selected, "class_names": list(CLASS_NAMES),
        "plan_sha256": snapshot.protocol.sha256, "execution_journal_sha256": snapshot.state_hash,
        "summary_sha256": file_digest(summary_path), "source_prediction_hashes": sources,
        "evaluation_rows": summary["evaluation_rows"], "folds": summary["folds"],
        "models_or_calibration_refitted": False, "model_inference_repeated": False,
        "policy_or_risk_changed": False, "thresholds_selected": False,
        "evaluation_previously_consumed": True, "confidence_intervals_computed": False,
        "definitions": {
            "entropy_normalized": "-sum(p*log(p))/log(3); 0 deterministic, 1 uniform; 0*log(0)=0",
            "top_class_probability": "largest class probability; not a measured probability of correctness or profit",
            "top_class_margin": "largest minus second-largest class probability; range 0 to 1",
            "logistic_disagreement_normalized": "equal-weight two-model Jensen-Shannon divergence/log(2); range 0 to 1",
            "entropy_bands": "five fixed equal-width bands; final band includes 1; outcomes reused for descriptive accuracy",
            "comparison_members": "selected family versus logistic at identical times and the same probability variant",
            "variant_comparison": "raw and temperature variants are not independent ensemble members",
            "limitations": "entropy describes predictive ambiguity; disagreement is not a guarantee of epistemic uncertainty; models can agree and be wrong",
            "expected_return": "not estimated; class probabilities and uncertainty do not establish net expected returns after costs",
        },
        "metrics": metrics, "selected_vs_logistic": comparisons,
    }
    snapshot.verify()
    staging = Path(tempfile.mkdtemp(prefix=".uncertainty-staging-", dir=snapshot.output))
    try:
        write_json(staging / "uncertainty.json", report)
        for name, frame in frames.items():
            enriched = frame.copy()
            for field, values in scores[name].items():
                enriched[field] = values
            write_parquet(staging / f"{name}_uncertainty.parquet", enriched)
        snapshot.verify()
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print_uncertainty_report(report)
    LOGGER.info("Saved uncertainty report to %s; reused predictions; no fitting or policy changes", output)
    return output, report


def print_uncertainty_report(report: dict) -> None:
    print("Historical uncertainty diagnostics (not confidence in correctness or profit):")
    print(pd.DataFrame([
        {"model": name, "mean_entropy": values["mean_entropy_normalized"],
         "mean_top_probability": values["mean_top_class_probability"], "mean_margin": values["mean_top_class_margin"]}
        for name, values in report["metrics"].items()
    ]).to_string(index=False))
    print("Selected model versus logistic; raw/scaled variants compared separately:")
    print(pd.DataFrame([
        {"variant": name, "mean_disagreement": values["mean_jensen_shannon_normalized"],
         "different_top_class_fraction": values["different_top_class_fraction"]}
        for name, values in report["selected_vs_logistic"].items()
    ]).to_string(index=False))
    print("Selected temperature predictions by fixed entropy band (reused outcomes):")
    print(pd.DataFrame(report["metrics"][f"{report['selected_family']}_temperature"]["entropy_bands"]).fillna("-").to_string(index=False))
