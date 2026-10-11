"""Restore a frozen protocol and audit fold inputs without reading test in fitting."""

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from local_ai_trader.features.build_features import FEATURE_COLUMNS
from local_ai_trader.models.baseline import CLASS_NAMES
from local_ai_trader.models.train import file_digest
from local_ai_trader.settings import (
    Settings, BoostingSettings, MLPSettings, CalibrationSettings, BacktestSettings,
    WalkForwardSettings, positive_integer, settings_from_config,
)


def restore_settings(values: dict) -> Settings:
    """Validate all saved settings; never consult current config or fill defaults."""
    if not isinstance(values, dict) or set(values) != set(Settings.__dataclass_fields__):
        raise ValueError("Protocol must contain the complete frozen settings schema")
    seconds = positive_integer(values["candle_seconds"], "candle_seconds")
    horizon = positive_integer(values["horizon_steps"], "horizon_steps")
    if seconds % 60:
        raise ValueError("Frozen candle interval must use whole minutes")
    for name in ("raw_dir", "processed_dir", "models_dir"):
        if not isinstance(values[name], str) or not Path(values[name]).is_absolute():
            raise ValueError("Frozen data paths must be absolute")
    if not isinstance(values["symbols"], (list, tuple)) or not values["symbols"]:
        raise ValueError("Frozen market symbols must be a nonempty list")
    for name, defaults in (("boosting", BoostingSettings), ("mlp", MLPSettings), ("calibration", CalibrationSettings), ("backtest", BacktestSettings), ("walkforward", WalkForwardSettings)):
        if not isinstance(values[name], dict) or set(values[name]) != set(defaults.__dataclass_fields__):
            raise ValueError(f"Protocol must freeze every {name} setting")
    config = {
        "market": {"exchange": "coinbase", "symbols": list(values["symbols"]), "candle_minutes": seconds // 60, "prediction_horizon_minutes": horizon * seconds // 60},
        "targets": {"flat_return_threshold": values["flat_return_threshold"]},
        "features": {name: values[name] for name in ("momentum_steps", "rolling_window")},
        "split": {name: values[name] for name in ("train_fraction", "validation_fraction")},
        "baseline": {name: values[name] for name in ("random_seed", "logistic_c", "logistic_max_iter", "naive_smoothing", "calibration_bins")},
        "download": {name: values[name] for name in ("lookback_days", "timeout_seconds", "max_attempts", "request_pause_seconds")},
        "paths": {"raw": values["raw_dir"], "processed": values["processed_dir"], "models": values["models_dir"]},
        **{name: values[name] for name in ("boosting", "mlp", "calibration", "backtest", "walkforward")},
    }
    return settings_from_config(config, Path.cwd())


@dataclass(frozen=True)
class Protocol:
    directory: Path
    plan: dict
    sha256: str
    settings: Settings

    def verify(self, include_test: bool = False) -> None:
        """During fitting, audit only train/validation/manifest, never test bytes."""
        if file_digest(self.directory / "plan.json") != self.sha256:
            raise ValueError("Frozen walk-forward plan changed during execution")
        for filename, digest in self.plan["fold_file_hashes"].items():
            if filename.endswith("/test.parquet") and not include_test:
                continue
            if file_digest(self.directory / filename) != digest:
                raise ValueError(f"Frozen fold input changed: {filename}")

    def fold_directory(self, index: int) -> Path:
        return self.directory / f"fold_{index:02d}"


def load_protocol(directory: Path) -> Protocol:
    directory = directory.resolve()
    path = directory / "plan.json"
    digest = file_digest(path)
    with path.open(encoding="utf-8") as stream:
        plan = json.load(stream)
    if (
        plan.get("schema_version") != 1 or plan.get("method") != "rolling_walk_forward"
        or plan.get("feature_names") != list(FEATURE_COLUMNS) or plan.get("class_names") != list(CLASS_NAMES)
        or plan.get("model_family") not in ("mlp", "xgboost")
        or plan.get("control_models") != ["naive", "logistic"]
        or plan.get("probability_variant") != "temperature" or plan.get("raw_comparator") is not True
        or plan.get("research_only") is not True
    ):
        raise ValueError("Unsupported frozen walk-forward protocol")
    settings = restore_settings(plan["settings"])
    folds = plan.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("Protocol has no folds")
    windows = settings.walkforward
    total = windows.train_days + windows.validation_days + windows.evaluation_days
    if settings.train_fraction != windows.train_days / total or settings.validation_fraction != windows.validation_days / total:
        raise ValueError("Frozen fractions do not match calendar windows")
    expected_hashes = {f"fold_{index:02d}/{name}" for index in range(1, len(folds) + 1) for name in ("split.json", "train.parquet", "validation.parquet", "test.parquet")}
    if set(plan["fold_file_hashes"]) != expected_hashes:
        raise ValueError("Protocol fold file allowlist is incompatible")
    previous_start, previous_end = None, None
    for index, fold in enumerate(folds, 1):
        if fold.get("fold") != index or fold.get("windows") != windows.__dict__:
            raise ValueError("Fold order or window settings differ from protocol")
        start = pd.Timestamp(fold["train_start_inclusive"])
        validation = pd.Timestamp(fold["validation_first_prediction_time"])
        evaluation = pd.Timestamp(fold["test_first_prediction_time"])
        end = pd.Timestamp(fold["evaluation_end_exclusive"])
        if start.tzinfo is None or start.utcoffset().total_seconds() != 0 or start != start.floor("D"):
            raise ValueError("Fold anchor must be UTC midnight")
        if (validation, evaluation, end) != (start + pd.to_timedelta(windows.train_days, unit="D"), start + pd.to_timedelta(windows.train_days + windows.validation_days, unit="D"), start + pd.to_timedelta(total, unit="D")):
            raise ValueError("Fold calendar boundaries differ from protocol")
        if previous_start is not None and (start != previous_start + pd.to_timedelta(windows.step_days, unit="D") or evaluation < previous_end):
            raise ValueError("Fold evaluations overlap or do not use the frozen step")
        for name, first, last in (("train", start, validation), ("validation", validation, evaluation), ("test", evaluation, end)):
            report = fold["partitions"][name]
            raw_rows = int((last - first).total_seconds() // settings.candle_seconds)
            if (
                report["raw_rows"] != raw_rows or report["rows"] != raw_rows - settings.horizon_steps
                or report["purged_rows"] != settings.horizon_steps
                or pd.Timestamp(report["first_prediction_time"]) != first
                or pd.Timestamp(report["last_prediction_time"]) != last - pd.to_timedelta((settings.horizon_steps + 1) * settings.candle_seconds, unit="s")
                or pd.Timestamp(report["last_label_known_at"]) != last - pd.to_timedelta(settings.candle_seconds, unit="s")
            ):
                raise ValueError("Fold partition does not establish complete purged calendar coverage")
        previous_start, previous_end = start, end
    protocol = Protocol(directory, plan, digest, settings)
    protocol.verify()
    for index, fold in enumerate(folds, 1):
        with (protocol.fold_directory(index) / "split.json").open(encoding="utf-8") as stream:
            split = json.load(stream)
        if any(split.get(key) != value for key, value in fold.items()):
            raise ValueError("Fold manifest differs from frozen plan")
    return protocol
