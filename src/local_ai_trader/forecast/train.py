"""Fit return baselines and assess validation without opening held-out data."""

from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
import uuid

import numpy as np
import pandas as pd
import sklearn

from local_ai_trader.data.storage import native_path, write_json, write_parquet
from local_ai_trader.decision.cost_aware import round_trip_factor
from local_ai_trader.features.build_features import FEATURE_COLUMNS
from local_ai_trader.forecast.baseline import MODEL_NAMES, evaluate_return_predictions, fit_return_baselines, predict_return_checkpoint, return_targets, validate_ridge_alpha
from local_ai_trader.forecast.split import LABEL_AVAILABLE_COLUMN, validate_execution_features
from local_ai_trader.forecast.targets import FUTURE_COLUMNS, verify_input_hashes
from local_ai_trader.models.train import file_digest
from local_ai_trader.settings import Settings, load_backtest_settings, positive_integer, validate_split_fractions

LOGGER = logging.getLogger(__name__)
ARTIFACT_NAMES = ("checkpoint.json", "metrics.json", "validation_mean.parquet", "validation_ridge.parquet")


def _equal(report: dict, name: str, expected) -> None:
    actual = report.get(name)
    if actual != expected or (type(expected) in (bool, int) and type(actual) is not type(expected)):
        raise ValueError(f"Execution split metadata disagrees on {name}")


def _utc_stamp(value, candle_seconds: int) -> pd.Timestamp:
    if not isinstance(value, str):
        raise ValueError("Partition times must be timezone-aware strings")
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Partition times must be timezone-aware")
    stamp = stamp.tz_convert("UTC")
    if stamp != stamp.floor(f"{candle_seconds}s"):
        raise ValueError("Partition times must be interval-aligned")
    return stamp


def load_return_training_partitions(directory: Path, settings: Settings) -> tuple[dict[str, pd.DataFrame], dict, dict[Path, str]]:
    """Audit only split.json, train.parquet and validation.parquet.

    Do not call the whole-source feature/target loaders during model fitting:
    those dataset-preparation audits necessarily read later outcomes. The frozen
    manifest establishes lineage here; only train/validation files are read or
    hashed. Test and upstream full-source files remain unopened.
    """
    directory = native_path(directory.resolve())
    manifest_path = directory / "split.json"
    inputs = {manifest_path: file_digest(manifest_path)}
    with manifest_path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    if not isinstance(report, dict):
        raise ValueError("Invalid execution split metadata")
    for name, expected in (
        ("schema_version", 1), ("target_kind", "execution_open_to_open"), ("feature_names", list(FEATURE_COLUMNS)),
        ("future_only_columns", list(FUTURE_COLUMNS)), ("regression_target_column", "execution_target_return"),
        ("target_available_column", LABEL_AVAILABLE_COLUMN),
        ("purge_rule", "execution_target_available_at < next_partition_first_available_at"),
        ("models_fitted", False), ("strategy_performance_evaluated", False),
        ("candle_seconds", settings.candle_seconds), ("holding_bars", settings.horizon_steps), ("latency_bars", settings.backtest.latency_bars),
        ("train_fraction", settings.train_fraction), ("validation_fraction", settings.validation_fraction),
    ):
        _equal(report, name, expected)
    validate_split_fractions(report["train_fraction"], report["validation_fraction"])
    feature = report.get("feature_metadata")
    if not isinstance(feature, dict) or not isinstance(feature.get("target_metadata"), dict):
        raise ValueError("Execution split requires feature and target provenance")
    for name, expected in (
        ("schema_version", 1), ("feature_names", list(FEATURE_COLUMNS)), ("future_only_columns", list(FUTURE_COLUMNS)),
        ("target_kind", "execution_open_to_open"), ("regression_target_column", "execution_target_return"),
        ("target_available_column", LABEL_AVAILABLE_COLUMN), ("prediction_time", "available_at"),
        ("candle_seconds", settings.candle_seconds), ("momentum_steps", settings.momentum_steps), ("rolling_window", settings.rolling_window),
        ("models_fitted", False), ("symbol", report.get("symbol")), ("exchange", report.get("exchange")),
    ):
        _equal(feature, name, expected)
    target = feature["target_metadata"]
    if not isinstance(target.get("costs"), dict):
        raise ValueError("Execution split requires frozen diagnostic costs")
    costs = load_backtest_settings(target["costs"])
    if costs.latency_bars != settings.backtest.latency_bars:
        raise ValueError("Frozen execution costs disagree on latency_bars")
    delay = 1 + settings.backtest.latency_bars + settings.horizon_steps
    for name, expected in (
        ("target_kind", "execution_open_to_open"), ("regression_target_column", "execution_target_return"),
        ("candle_seconds", settings.candle_seconds), ("holding_bars", settings.horizon_steps), ("latency_bars", settings.backtest.latency_bars),
        ("target_known_offset_from_prediction_bars", delay), ("future_only_columns", list(FUTURE_COLUMNS)),
        ("costs", asdict(costs)), ("symbol", report.get("symbol")), ("exchange", report.get("exchange")),
        ("availability_assumption", "conservative_exit_candle_completion_for_complete_OHLCV_pipeline"),
    ):
        _equal(target, name, expected)
    _equal(report, "target_known_offset_from_prediction_bars", delay)
    _equal(feature, "target_known_offset_from_prediction_bars", delay)
    count = positive_integer(report.get("source_rows"), "source_rows")
    _equal(feature, "feature_rows", count)
    cut_train, cut_validation = int(count * settings.train_fraction), int(count * (settings.train_fraction + settings.validation_fraction))
    _equal(report, "cut_positions", {"validation_start": cut_train, "test_start": cut_validation})
    if not 0 < cut_train < cut_validation < count:
        raise ValueError("Invalid execution split cut positions")
    interval = pd.to_timedelta(settings.candle_seconds, unit="s")
    first = _utc_stamp(feature.get("first_prediction_time"), settings.candle_seconds)
    validation_start, test_start = first + cut_train * interval, first + cut_validation * interval
    _equal(report, "validation_first_prediction_time", validation_start.isoformat())
    _equal(report, "test_first_prediction_time", test_start.isoformat())
    if report.get("exchange") != "coinbase" or report.get("symbol") not in settings.symbols:
        raise ValueError("Execution split market is not enabled")
    summaries = report.get("partitions")
    if not isinstance(summaries, dict) or set(summaries) != {"train", "validation", "test"}:
        raise ValueError("Execution split requires three partition summaries")
    raw_counts = {"train": cut_train, "validation": cut_validation - cut_train, "test": count - cut_validation}
    for name, raw_rows in raw_counts.items():
        summary = summaries[name]
        if not isinstance(summary, dict):
            raise ValueError("Invalid partition summary")
        purged = 0 if name == "test" else delay
        for key, expected in (("raw_rows", raw_rows), ("purged_rows", purged), ("rows", raw_rows - purged)):
            _equal(summary, key, expected)
        positive_integer(summary["rows"], f"{name}.rows")
        if name != "test":
            boundary = validation_start if name == "train" else test_start
            _equal(summary, "purged_prediction_times", [(boundary - index * interval).isoformat() for index in range(delay, 0, -1)])
    tables = {}
    # No stat, hash or read of test.parquet or the original full datasets.
    for name in ("train", "validation"):
        path, summary = directory / f"{name}.parquet", summaries[name]
        inputs[path] = summary.get("sha256")
        verify_input_hashes(inputs)
        frame = validate_execution_features(pd.read_parquet(path), settings.candle_seconds, settings.horizon_steps, settings.backtest.latency_bars)
        _equal(summary, "rows", len(frame))
        for key, stamp in (
            ("first_prediction_time", frame.available_at.iloc[0]), ("last_prediction_time", frame.available_at.iloc[-1]),
            ("last_label_known_at", frame[LABEL_AVAILABLE_COLUMN].iloc[-1]),
        ):
            _equal(summary, key, stamp.isoformat())
        expected_first = first if name == "train" else validation_start
        boundary = validation_start if name == "train" else test_start
        if frame.available_at.iloc[0] != expected_first or frame.available_at.iloc[-1] != boundary - (delay + 1) * interval:
            raise ValueError("Execution partition prediction times disagree with cuts")
        if not (frame[LABEL_AVAILABLE_COLUMN] < boundary).all():
            raise ValueError("Execution labels overlap the next partition")
        if frame.symbol.iloc[0] != report["symbol"] or frame.exchange.iloc[0] != report["exchange"]:
            raise ValueError("Execution partition market disagrees with metadata")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            gross = frame.execution_exit_reference_open.to_numpy(dtype=float) / frame.execution_entry_reference_open.to_numpy(dtype=float) - 1
            net = (1 + gross) * round_trip_factor(costs) - 1
        if not np.array_equal(gross, return_targets(frame)) or not np.array_equal(net, frame.execution_target_net_return.to_numpy(dtype=float)):
            raise ValueError("Execution partition returns disagree with reference prices and frozen costs")
        tables[name] = frame
    verify_input_hashes(inputs)
    return tables, report, inputs


def train_return_baseline_experiment(directory: Path, settings: Settings, ridge_alpha: float = 1.0) -> tuple[Path, dict]:
    """Fit mean/Ridge once and atomically publish a validation-only experiment."""
    started = time.perf_counter()
    ridge_alpha = validate_ridge_alpha(ridge_alpha)
    directory = native_path(directory.resolve())
    tables, split_report, inputs = load_return_training_partitions(directory, settings)
    fitted = fit_return_baselines(tables["train"], settings, ridge_alpha)
    checkpoint = fitted.checkpoint()
    metrics, predictions = {}, {}
    frozen_costs = load_backtest_settings(split_report["feature_metadata"]["target_metadata"]["costs"])
    for partition, frame in tables.items():
        native, restored = fitted.predict(frame), predict_return_checkpoint(checkpoint, frame)
        metrics[partition] = {}
        for model in MODEL_NAMES:
            np.testing.assert_allclose(restored[model], native[model], rtol=1e-10, atol=1e-12)
            metrics[partition][model] = evaluate_return_predictions(return_targets(frame), restored[model])
            if partition == "validation":
                result = frame[["timestamp", "available_at", "symbol", "exchange", *FUTURE_COLUMNS]].copy()
                result["expected_gross_return"] = restored[model]
                result["diagnostic_expected_net_return"] = (1 + restored[model]) * round_trip_factor(frozen_costs) - 1
                result["forecast_valid"] = restored[model] >= -1
                predictions[model] = result
    metrics["test"] = {"evaluated": False}
    created = pd.Timestamp.now(tz="UTC")
    run_id = f"return_baseline_{checkpoint['symbol']}_{created.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    provenance = {"run_id": run_id, "source_split_directory": str(directory), "input_hashes": {path.name: digest for path, digest in inputs.items()}}
    checkpoint["training_provenance"] = provenance
    experiment = {
        **provenance, "schema_version": 1, "created_at": created.isoformat(), "target_kind": "execution_open_to_open",
        "model_types": ["training_mean_return", "scaled_ridge_regression"], "model_names": list(MODEL_NAMES),
        "symbol": checkpoint["symbol"], "exchange": checkpoint["exchange"], "feature_names": list(FEATURE_COLUMNS),
        "target_column": "execution_target_return", "return_unit": "fractional_reference_price_return",
        "settings": {name: str(value) if isinstance(value, Path) else value for name, value in asdict(settings).items()},
        "return_parameters": {"ridge_alpha": ridge_alpha, "ridge_solver": "svd", "device": "cpu"},
        "split_metadata": split_report, "training_rows": len(tables["train"]), "validation_rows": len(tables["validation"]),
        "test_data_opened": False, "upstream_full_datasets_opened": False, "test_evaluated": False,
        "strategy_performance_evaluated": False, "uncertainty_estimate": "not supplied by these point forecasts",
        "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "elapsed_seconds": time.perf_counter() - started, "prediction_files": {model: f"validation_{model}.parquet" for model in MODEL_NAMES},
        "checkpoint_path": "checkpoint.json", "metrics_path": "metrics.json",
        "limitations": "validation research on overlapping historical outcomes; no profit, independent-strategy-performance or trade authorization claim",
    }
    verify_input_hashes(inputs)
    root = native_path(settings.models_dir.resolve())
    root.mkdir(parents=True, exist_ok=True)
    output = root / run_id
    staging = Path(tempfile.mkdtemp(prefix=".return-baseline-staging-", dir=root))
    try:
        write_json(staging / "checkpoint.json", checkpoint)
        write_json(staging / "metrics.json", metrics)
        for model, frame in predictions.items():
            write_parquet(staging / f"validation_{model}.parquet", frame)
        experiment["artifact_hashes"] = {name: file_digest(staging / name) for name in ARTIFACT_NAMES}
        write_json(staging / "experiment.json", experiment)
        verify_input_hashes(inputs)
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    LOGGER.info("Fitted return baselines on %s training rows; assessed %s validation rows; test partition not opened", len(tables["train"]), len(tables["validation"]))
    print("Validation return-forecast errors (fractional gross returns; not strategy performance):")
    print(pd.DataFrame([{"model": model, **{name: metrics['validation'][model][name] for name in ('mae', 'rmse', 'r2', 'mean_error', 'mean_prediction', 'mean_observed_return')}} for model in MODEL_NAMES]).to_string(index=False))
    LOGGER.info("Saved return experiment and JSON checkpoint to %s", output)
    return output, metrics


def load_return_checkpoint(directory: Path) -> dict:
    """Audit a complete saved run without reading original train/validation/test data."""
    directory = native_path(directory.resolve())
    manifest_path = directory / "experiment.json"
    manifest_digest = file_digest(manifest_path)
    with manifest_path.open(encoding="utf-8") as stream:
        experiment = json.load(stream)
    if (
        not isinstance(experiment, dict) or type(experiment.get("schema_version")) is not int or experiment["schema_version"] != 1
        or experiment.get("target_kind") != "execution_open_to_open" or experiment.get("model_names") != list(MODEL_NAMES)
        or experiment.get("test_data_opened") is not False or experiment.get("test_evaluated") is not False
    ):
        raise ValueError("Incompatible return experiment")
    hashes = experiment.get("artifact_hashes")
    if not isinstance(hashes, dict) or set(hashes) != set(ARTIFACT_NAMES):
        raise ValueError("Return experiment requires all artifact hashes")
    inputs = {directory / name: digest for name, digest in hashes.items()}
    inputs[manifest_path] = manifest_digest
    verify_input_hashes(inputs)
    with (directory / "checkpoint.json").open(encoding="utf-8") as stream:
        checkpoint = json.load(stream)
    if not isinstance(checkpoint, dict) or checkpoint.get("training_provenance") != {name: experiment.get(name) for name in ("run_id", "source_split_directory", "input_hashes")}:
        raise ValueError("Return checkpoint provenance disagrees with experiment")
    for name in ("symbol", "exchange", "feature_names", "target_kind", "target_column", "return_unit"):
        if checkpoint.get(name) != experiment.get(name):
            raise ValueError(f"Return checkpoint metadata disagrees on {name}")
    verify_input_hashes(inputs)
    return checkpoint
