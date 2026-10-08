"""Load and validate configuration; resolve data paths against the project root."""

from dataclasses import dataclass
from pathlib import Path
import math
import re
import tomllib


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
    lookback_days: int
    timeout_seconds: int
    max_attempts: int
    request_pause_seconds: float
    raw_dir: Path
    processed_dir: Path


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
    return Settings(
        symbols=tuple(validate_symbol(s) for s in symbols),
        candle_seconds=seconds,
        horizon_steps=horizon // seconds,
        flat_return_threshold=float(threshold),
        momentum_steps=positive_integer(config["features"]["momentum_steps"], "momentum_steps"),
        rolling_window=rolling_window,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        lookback_days=positive_integer(download["lookback_days"], "lookback_days"),
        timeout_seconds=positive_integer(download["timeout_seconds"], "timeout_seconds"),
        max_attempts=positive_integer(download["max_attempts"], "max_attempts"),
        request_pause_seconds=pause,
        raw_dir=root / paths["raw"],
        processed_dir=root / paths["processed"],
    )
