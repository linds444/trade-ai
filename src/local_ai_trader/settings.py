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
class CalibrationSettings:
    fit_fraction: float = 0.5
    min_temperature: float = 0.25
    max_temperature: float = 4.0
    probability_floor: float = 1e-12
    optimizer_tolerance: float = 1e-6
    max_iterations: int = 200


@dataclass(frozen=True)
class BacktestSettings:
    initial_cash: float = 10000.0
    allocation_fraction: float = 0.1
    max_exposure_fraction: float = 0.1
    max_drawdown_fraction: float = 0.1
    entry_probability: float = 0.55
    exit_probability: float = 0.55
    minimum_direction_margin: float = 0.15
    fee_bps: float = 60.0
    spread_bps: float = 10.0
    slippage_bps: float = 5.0
    latency_bars: int = 1


@dataclass(frozen=True)
class WalkForwardSettings:
    train_days: int = 60
    validation_days: int = 14
    evaluation_days: int = 14
    step_days: int = 14


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
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    backtest: BacktestSettings = field(default_factory=BacktestSettings)
    walkforward: WalkForwardSettings = field(default_factory=WalkForwardSettings)


def load_walkforward_settings(config: dict) -> WalkForwardSettings:
    defaults = WalkForwardSettings()
    values = {name: positive_integer(config.get(name, getattr(defaults, name)), f"walkforward.{name}") for name in defaults.__dataclass_fields__}
    if values["step_days"] < values["evaluation_days"]:
        raise ValueError("walkforward.step_days must be at least evaluation_days to avoid overlapping evaluation periods")
    return WalkForwardSettings(**values)


def load_backtest_settings(config: dict) -> BacktestSettings:
    defaults = BacktestSettings()
    values = {name: config.get(name, getattr(defaults, name)) for name in defaults.__dataclass_fields__}
    positive_integer(values["latency_bars"], "backtest.latency_bars")
    for name, value in values.items():
        if name == "latency_bars":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"backtest.{name} must be a finite number")
    if values["initial_cash"] <= 0:
        raise ValueError("backtest.initial_cash must be positive")
    for name in ("allocation_fraction", "max_exposure_fraction", "max_drawdown_fraction", "entry_probability", "exit_probability"):
        if not 0 < values[name] <= 1:
            raise ValueError(f"backtest.{name} must be in (0, 1]")
    if not 0 <= values["minimum_direction_margin"] <= 1:
        raise ValueError("backtest.minimum_direction_margin must be in [0, 1]")
    for name in ("fee_bps", "spread_bps", "slippage_bps"):
        if not 0 <= values[name] < 10000:
            raise ValueError(f"backtest.{name} must be in [0, 10000)")
    if values["spread_bps"] / 2 + values["slippage_bps"] >= 10000:
        raise ValueError("Combined adverse price impact must be below 100%")
    return BacktestSettings(**values)


def load_calibration_settings(config: dict) -> CalibrationSettings:
    defaults = CalibrationSettings()
    values = {name: config.get(name, getattr(defaults, name)) for name in defaults.__dataclass_fields__}
    for name in ("fit_fraction", "min_temperature", "max_temperature", "probability_floor", "optimizer_tolerance"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"calibration.{name} must be a positive finite number")
    if values["fit_fraction"] >= 1:
        raise ValueError("calibration.fit_fraction must be below 1")
    if not values["min_temperature"] <= 1 <= values["max_temperature"] or values["min_temperature"] >= values["max_temperature"]:
        raise ValueError("Temperature bounds must be ordered and contain 1")
    if values["probability_floor"] > 1e-6:
        raise ValueError("calibration.probability_floor must be at most 1e-6")
    positive_integer(values["max_iterations"], "calibration.max_iterations")
    return CalibrationSettings(**values)


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
        calibration=load_calibration_settings(config.get("calibration", {})),
        backtest=load_backtest_settings(config.get("backtest", {})),
        walkforward=load_walkforward_settings(config.get("walkforward", {})),
    )
