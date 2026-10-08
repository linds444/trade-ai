"""Load and validate configuration; resolve data paths against the project root."""

from dataclasses import dataclass, field
from pathlib import Path
import math
import re
import tomllib


@dataclass(frozen=True)
class BoostingSettings:
    n_estimators: int = 200
    max_depth: int = 3
    learning_rate: float = 0.05
    min_child_weight: float = 5.0
    reg_lambda: float = 5.0
    n_jobs: int = 4


@dataclass(frozen=True)
class MLPSettings:
    hidden_sizes: tuple[int, ...] = (64, 32)
    dropout: float = 0.1
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    grad_clip_norm: float = 1.0
    device: str = "cuda"
    mixed_precision: bool = True
    cpu_threads: int = 4


@dataclass(frozen=True)
class Settings:
    symbols: tuple[str, ...]
    candle_seconds: int
    horizon_steps: int
    flat_return_threshold: float
    momentum_steps: int
    rolling_window: int
    train_fraction: float
    validation_fraction: float
    random_seed: int
    logistic_c: float
    logistic_max_iter: int
    naive_smoothing: float
    calibration_bins: int
    lookback_days: int
    timeout_seconds: int
    max_attempts: int
    request_pause_seconds: float
    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    boosting: BoostingSettings = field(default_factory=BoostingSettings)
    mlp: MLPSettings = field(default_factory=MLPSettings)


def load_mlp_settings(config: dict) -> MLPSettings:
    defaults = MLPSettings()
    values = {name: config.get(name, getattr(defaults, name)) for name in defaults.__dataclass_fields__}
    hidden = values["hidden_sizes"]
    if not isinstance(hidden, (tuple, list)) or not hidden:
        raise ValueError("mlp.hidden_sizes must be a nonempty list of positive integers")
    values["hidden_sizes"] = tuple(positive_integer(size, "mlp.hidden_sizes") for size in hidden)
    for name in ("epochs", "batch_size", "cpu_threads"):
        positive_integer(values[name], f"mlp.{name}")
    for name in ("dropout", "learning_rate", "weight_decay", "grad_clip_norm"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"mlp.{name} must be a nonnegative finite number")
    if values["dropout"] >= 1 or values["learning_rate"] <= 0 or values["grad_clip_norm"] <= 0:
        raise ValueError("mlp.dropout must be below 1; learning_rate and grad_clip_norm must be positive")
    if values["device"] not in ("cpu", "cuda") or type(values["mixed_precision"]) is not bool:
        raise ValueError("mlp.device must be cpu/cuda and mixed_precision must be boolean")
    return MLPSettings(**values)


def load_boosting_settings(config: dict) -> BoostingSettings:
    defaults = BoostingSettings()
    values = {name: config.get(name, getattr(defaults, name)) for name in defaults.__dataclass_fields__}
    for name in ("n_estimators", "max_depth", "n_jobs"):
        positive_integer(values[name], f"boosting.{name}")
    for name in ("learning_rate", "min_child_weight", "reg_lambda"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"boosting.{name} must be a nonnegative finite number")
    if not 0 < values["learning_rate"] <= 1:
        raise ValueError("boosting.learning_rate must be in (0, 1]")
    return BoostingSettings(**values)


def validate_symbol(symbol: str) -> str:
    if not re.fullmatch(r"[A-Z0-9]+-[A-Z0-9]+", symbol):
        raise ValueError(f"Invalid Coinbase product identifier: {symbol!r}")
    return symbol


def positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_split_fractions(train: object, validation: object) -> tuple[float, float]:
    for name, value in (("train_fraction", train), ("validation_fraction", validation)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{name} must be a finite fraction strictly between 0 and 1")
    if train + validation >= 1:
        raise ValueError("Train and validation fractions must sum to less than 1 to leave test data")
    return float(train), float(validation)


def load_settings(path: Path) -> Settings:
    path = path.resolve()
    with path.open("rb") as source:
        config = tomllib.load(source)
    market, download, paths = config["market"], config["download"], config["paths"]
    if market["exchange"] != "coinbase":
        raise ValueError("This milestone supports only the Coinbase public candle API")
    seconds = positive_integer(market["candle_minutes"], "candle_minutes") * 60
    if seconds not in (60, 300, 900, 3600, 21600, 86400):
        raise ValueError("Unsupported Coinbase candle interval")
    horizon = positive_integer(market["prediction_horizon_minutes"], "horizon") * 60
    if horizon % seconds:
        raise ValueError("Prediction horizon must be a whole number of candle intervals")
    threshold = config["targets"]["flat_return_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("flat_return_threshold must be a numeric fraction")
    if not math.isfinite(threshold) or not 0 <= threshold < 1:
        raise ValueError("flat_return_threshold must be finite and between 0 (inclusive) and 1 (exclusive)")
    symbols = market["symbols"]
    if not isinstance(symbols, list) or not symbols or not all(isinstance(s, str) for s in symbols):
        raise ValueError("symbols must be a nonempty list of product identifiers")
    if len(set(symbols)) != len(symbols):
        raise ValueError("Duplicate symbols in settings")
    pause = float(download["request_pause_seconds"])
    if not math.isfinite(pause) or pause < 0:
        raise ValueError("request_pause_seconds must be finite and nonnegative")
    root = path.parent.parent
    rolling_window = positive_integer(config["features"]["rolling_window"], "rolling_window")
    if rolling_window < 2:
        raise ValueError("rolling_window must be at least 2")
    train_fraction, validation_fraction = validate_split_fractions(
        config["split"]["train_fraction"], config["split"]["validation_fraction"],
    )
    baseline = config["baseline"]
    for key in ("logistic_c", "naive_smoothing"):
        value = baseline[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a positive finite number")
    seed = baseline["random_seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("random_seed must be an integer in [0, 2**32)")
    return Settings(
        symbols=tuple(validate_symbol(s) for s in symbols),
        candle_seconds=seconds,
        horizon_steps=horizon // seconds,
        flat_return_threshold=float(threshold),
        momentum_steps=positive_integer(config["features"]["momentum_steps"], "momentum_steps"),
        rolling_window=rolling_window,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        random_seed=seed,
        logistic_c=float(baseline["logistic_c"]),
        logistic_max_iter=positive_integer(baseline["logistic_max_iter"], "logistic_max_iter"),
        naive_smoothing=float(baseline["naive_smoothing"]),
        calibration_bins=positive_integer(baseline["calibration_bins"], "calibration_bins"),
        lookback_days=positive_integer(download["lookback_days"], "lookback_days"),
        timeout_seconds=positive_integer(download["timeout_seconds"], "timeout_seconds"),
        max_attempts=positive_integer(download["max_attempts"], "max_attempts"),
        request_pause_seconds=pause,
        raw_dir=root / paths["raw"],
        processed_dir=root / paths["processed"],
        models_dir=root / paths["models"],
        boosting=load_boosting_settings(config.get("boosting", {})),
        mlp=load_mlp_settings(config.get("mlp", {})),
    )
