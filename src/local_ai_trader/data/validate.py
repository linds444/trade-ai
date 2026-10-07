"""Shared single-market candle timeline validation for targets and features."""

import numpy as np
import pandas as pd

from local_ai_trader.settings import positive_integer


def validate_candle_timeline(candles: pd.DataFrame, candle_seconds: int) -> pd.DataFrame:
    """Return a copy with an ordered contiguous timeline normalized to UTC."""
    positive_integer(candle_seconds, "candle_seconds")
    required = {"timestamp", "available_at", "close", "symbol", "exchange"}
    if not required.issubset(candles.columns):
        raise ValueError(f"Missing candle columns: {sorted(required - set(candles.columns))}")
    if candles.empty:
        raise ValueError("Candle dataset is empty")
    frame = candles.copy().reset_index(drop=True)
    for column in ("symbol", "exchange"):
        if frame[column].isna().any() or frame[column].nunique() != 1:
            raise ValueError("Require exactly one symbol and one exchange per dataset")
    for column in ("timestamp", "available_at"):
        if frame[column].isna().any() or getattr(frame[column].dtype, "tz", None) is None:
            raise ValueError(f"{column} must contain timezone-aware, nonmissing datetimes")
        frame[column] = frame[column].dt.tz_convert("UTC")
    interval = pd.Timedelta(candle_seconds, unit="s")
    if not frame["timestamp"].eq(frame["timestamp"].dt.floor(f"{candle_seconds}s")).all():
        raise ValueError("Candle timestamps must be interval-aligned")
    if not frame["timestamp"].diff().iloc[1:].eq(interval).all():
        raise ValueError("Candles must be ordered, unique and contiguous; do not shift across gaps")
    if not frame["available_at"].eq(frame["timestamp"] + interval).all():
        raise ValueError("Candle availability must equal its open timestamp plus one interval")
    if frame["available_at"].max() > pd.Timestamp.now(tz="UTC"):
        raise ValueError("Input contains unfinished or future candles")
    if not pd.api.types.is_numeric_dtype(frame["close"]) or pd.api.types.is_bool_dtype(frame["close"]):
        raise ValueError("Close prices must be numeric")
    close = frame["close"].to_numpy(dtype=np.float64, na_value=np.nan)
    if not np.isfinite(close).all() or not (close > 0).all():
        raise ValueError("Close prices must be positive and finite")
    return frame
