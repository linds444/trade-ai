"""Freeze rolling calendar partitions before fitting or comparing fold outcomes."""

from dataclasses import asdict, dataclass, replace
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from local_ai_trader.data.dataset import TARGET_COLUMNS
from local_ai_trader.data.split import validate_feature_dataset
from local_ai_trader.data.storage import write_json, write_parquet
from local_ai_trader.features.build_features import FEATURE_COLUMNS
from local_ai_trader.models.baseline import CLASS_NAMES
from local_ai_trader.models.train import file_digest
from local_ai_trader.settings import Settings, WalkForwardSettings, load_walkforward_settings

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    partitions: dict[str, pd.DataFrame]
    report: dict


def build_folds(
    features: pd.DataFrame, candle_seconds: int, horizon_steps: int, windows: WalkForwardSettings,
) -> list[Fold]:
    """Anchor prediction-time windows at the first complete UTC midnight.

    A label must be known strictly before its partition's end, including test.
    Calendar durations are fixed; only full folds are used and evaluations do
    not overlap. Earlier evaluated history can enter a later fold's training.
    """
    windows = load_walkforward_settings(asdict(windows))
    frame = validate_feature_dataset(features, candle_seconds, horizon_steps)
    start = frame["available_at"].iloc[0].ceil("D")
    coverage_end = frame["available_at"].iloc[-1] + pd.to_timedelta(candle_seconds, unit="s")
    durations = (windows.train_days, windows.validation_days, windows.evaluation_days)
    total_days = sum(durations)
    fractions = {"train_fraction": windows.train_days / total_days, "validation_fraction": windows.validation_days / total_days}
    folds = []
    while start + pd.to_timedelta(total_days, unit="D") <= coverage_end:
        validation_start = start + pd.to_timedelta(windows.train_days, unit="D")
        evaluation_start = validation_start + pd.to_timedelta(windows.validation_days, unit="D")
        end = evaluation_start + pd.to_timedelta(windows.evaluation_days, unit="D")
        boundaries = (start, validation_start, evaluation_start, end)
        partitions, summaries = {}, {}
        for index, name in enumerate(("train", "validation", "test")):
            first, last = boundaries[index], boundaries[index + 1]
            raw = frame.loc[(frame["available_at"] >= first) & (frame["available_at"] < last)]
            expected = durations[index] * 86400 // candle_seconds
            if len(raw) != expected or raw["available_at"].iloc[0] != first or raw["available_at"].iloc[-1] != last - pd.to_timedelta(candle_seconds, unit="s"):
                raise ValueError("Incomplete calendar window; no folds published")
            keep = raw["target_available_at"] < last
            selected = raw.loc[keep].copy().reset_index(drop=True)
            if selected.empty:
                raise ValueError(f"{name} is empty after label purging; use longer windows")
            partitions[name] = selected
            summaries[name] = {
                "raw_rows": len(raw), "rows": len(selected), "purged_rows": int((~keep).sum()),
                "purged_prediction_times": [stamp.isoformat() for stamp in raw.loc[~keep, "available_at"]],
                "first_prediction_time": selected["available_at"].iloc[0].isoformat(),
                "last_prediction_time": selected["available_at"].iloc[-1].isoformat(),
                "last_label_known_at": selected["target_available_at"].iloc[-1].isoformat(),
                "class_counts": {label: int(selected["target_class"].eq(label).sum()) for label in CLASS_NAMES},
            }
        report = {
            "split_method": "rolling_walk_forward", "fold": len(folds) + 1,
            "source_rows": total_days * 86400 // candle_seconds,
            **fractions, "test_fraction": windows.evaluation_days / total_days,
            "cut_positions": {"validation_start": windows.train_days * 86400 // candle_seconds, "test_start": (windows.train_days + windows.validation_days) * 86400 // candle_seconds},
            "purge_rule": "target_available_at < next_partition_first_available_at",
            "test_end_purge_rule": "target_available_at < evaluation_end_exclusive",
            "train_start_inclusive": start.isoformat(),
            "validation_first_prediction_time": validation_start.isoformat(),
            "test_first_prediction_time": evaluation_start.isoformat(),
            "evaluation_end_exclusive": end.isoformat(),
            "windows": asdict(windows), "partitions": summaries,
        }
        folds.append(Fold(partitions, report))
        start += pd.to_timedelta(windows.step_days, unit="D")
    if not folds:
        raise ValueError("Not enough complete history for one walk-forward fold")
    return folds


def create_walkforward_plan(source: Path, settings: Settings, model: str) -> tuple[Path, dict]:
    """Save all folds, source hashes and fixed research choices as one bundle."""
    if model not in ("mlp", "xgboost"):
        raise ValueError("Select mlp or xgboost; naive/logistic controls are included in the protocol")
    if not source.is_file():
        raise ValueError(f"Feature Parquet not found: {source}")
    metadata_path = source.with_suffix(".features.json")
    if not metadata_path.is_file():
        raise ValueError("Feature metadata is missing; use output from the features command")
    hashes = {"features": file_digest(source), "metadata": file_digest(metadata_path)}
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict) or metadata.get("feature_names") != list(FEATURE_COLUMNS):
        raise ValueError("Incompatible feature metadata allowlist")
    for name in ("candle_seconds", "momentum_steps", "rolling_window"):
        if metadata.get(name) != getattr(settings, name):
            raise ValueError(f"Feature metadata disagrees with {name}")
    target_metadata = metadata.get("target_metadata")
    if not isinstance(target_metadata, dict):
        raise ValueError("Missing target provenance")
    for name in ("horizon_steps", "flat_return_threshold"):
        if target_metadata.get(name) != getattr(settings, name):
            raise ValueError(f"Target metadata disagrees with {name}")
    destination = settings.processed_dir / "walkforward" / source.stem
    if destination.exists():
        raise ValueError("Walk-forward plan already exists; inspect its immutable plan.json")
    frame = pd.read_parquet(source)
    if metadata.get("feature_rows") != len(frame):
        raise ValueError("Feature metadata row count differs from source")
    folds = build_folds(frame, settings.candle_seconds, settings.horizon_steps, settings.walkforward)
    created_at = pd.Timestamp.now(tz="UTC").isoformat()
    first_report = folds[0].report
    # Future fitting restores these actual calendar fractions rather than 60/20/20.
    fold_settings = replace(settings, train_fraction=first_report["train_fraction"], validation_fraction=first_report["validation_fraction"])
    parameters = {name: str(value) if isinstance(value, Path) else value for name, value in asdict(fold_settings).items()}
    plan = {
        "schema_version": 1, "created_at": created_at,
        "research_only": True, "method": "rolling_walk_forward",
        "symbol": str(frame["symbol"].iloc[0]), "exchange": str(frame["exchange"].iloc[0]),
        "model_family": model, "control_models": ["naive", "logistic"],
        "probability_variant": "temperature", "raw_comparator": True,
        "settings": parameters, "feature_names": list(FEATURE_COLUMNS), "class_names": list(CLASS_NAMES),
        "source_path": str(source.resolve()), "source_metadata_path": str(metadata_path.resolve()),
        "source_hashes": hashes, "source_rows": len(frame),
        "source_first_prediction_time": frame["available_at"].iloc[0].isoformat(),
        "source_last_prediction_time": frame["available_at"].iloc[-1].isoformat(),
        "leading_rows_before_anchor": int((frame["available_at"] < pd.Timestamp(first_report["train_start_inclusive"])).sum()),
        "unused_tail_rows": int((frame["available_at"] >= pd.Timestamp(folds[-1].report["evaluation_end_exclusive"])).sum()),
        "folds": [fold.report for fold in folds],
        "protocol": {
            "anchor": "first_complete_UTC_midnight_of_prediction_times",
            "retraining": "fresh_weights_and_train_only_preprocessing_each_fold",
            "calibration": "earlier_validation_fit_later_validation_assessment_with_label_purge",
            "evaluation": "next_period_after_weights_and_temperature_frozen",
            "evaluation_overlap": False,
            "label_equality_at_boundary": "excluded_in_all_three_partitions",
            "policy_and_costs": "fixed_saved_backtest_settings_no_fold_specific_tuning",
            "performance_aggregation": "no_treating_adjacent_overlapping_horizon_labels_as_independent_samples",
            "scope": "historical_research_source_overlaps_previously_examined_periods_fresh_final_period_required",
        },
        "models_fitted": False, "fold_performance_evaluated": False,
    }

    def verify_sources() -> None:
        if file_digest(source) != hashes["features"] or file_digest(metadata_path) != hashes["metadata"]:
            raise ValueError("Walk-forward sources changed during planning; no plan published")

    verify_sources()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".walkforward-staging-", dir=destination.parent))
    try:
        file_hashes = {}
        for fold in folds:
            directory = staging / f"fold_{fold.report['fold']:02d}"
            directory.mkdir()
            split_report = {
                "schema_version": 1, "created_at": created_at,
                "source_path": str(source.resolve()), "source_sha256": hashes["features"],
                "feature_names": list(FEATURE_COLUMNS), "future_only_columns": list(TARGET_COLUMNS),
                "feature_metadata": metadata, **fold.report,
            }
            for name, rows in fold.partitions.items():
                write_parquet(directory / f"{name}.parquet", rows)
            write_json(directory / "split.json", split_report)
            for path in directory.iterdir():
                file_hashes[path.relative_to(staging).as_posix()] = file_digest(path)
        plan["fold_file_hashes"] = file_hashes
        write_json(staging / "plan.json", plan)
        verify_sources()
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print("Walk-forward research plan (UTC prediction times; no model fitting):")
    print(pd.DataFrame([
        {"fold": fold.report["fold"], "train_start": pd.Timestamp(fold.report["train_start_inclusive"]).strftime("%Y-%m-%d"), "evaluation_start": pd.Timestamp(fold.report["test_first_prediction_time"]).strftime("%Y-%m-%d"), "evaluation_end": pd.Timestamp(fold.report["evaluation_end_exclusive"]).strftime("%Y-%m-%d"), **{f"{name}_rows": fold.report["partitions"][name]["rows"] for name in ("train", "validation", "test")}}
        for fold in folds
    ]).to_string(index=False))
    LOGGER.info("Saved %s immutable folds to %s; labels purged at each boundary", len(folds), destination)
    return destination, plan
