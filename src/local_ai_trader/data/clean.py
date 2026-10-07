"""Validate OHLCV, remove identical duplicates, and expose missing intervals."""

from dataclasses import dataclass
import math
from numbers import Real

import pandas as pd


@dataclass(frozen=True)
class QualityReport:
    raw_rows: int
    cleaned_rows: int
    duplicate_rows_removed: int
    out_of_range_rows_removed: int
    expected_rows: int
    missing_count: int
    missing_timestamps_utc: list[str]


def clean_candles(rows: list, start: int, end: int, seconds: int) -> tuple[pd.DataFrame, QualityReport]:
    """Input format is Coinbase [epoch, low, high, open, close, volume].

    Timestamps are candle OPEN times. A candle's values become available only at
    timestamp + seconds. Missing candles are never invented or forward-filled.
    """
    if seconds <= 0 or start >= end or start % seconds or end % seconds:
        raise ValueError("Invalid or unaligned cleaning bounds")
    unique = {}
    duplicates = outside = 0
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 6:
            raise ValueError(f"Candle {index}: expected exactly six values")
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in row):
            raise ValueError(f"Candle {index}: all values must be finite numbers")
        timestamp, low, high, opening, close, volume = row
        if int(timestamp) != timestamp or int(timestamp) % seconds:
            raise ValueError(f"Candle {index}: timestamp is not interval-aligned")
        timestamp = int(timestamp)
        if not (0 < low <= min(opening, close) <= max(opening, close) <= high) or volume < 0:
            raise ValueError(f"Candle {index}: invalid OHLC prices or negative volume")
        if not start <= timestamp < end:
            outside += 1
            continue
        values = (float(low), float(high), float(opening), float(close), float(volume))
        if timestamp in unique:
            if unique[timestamp] != values:
                raise ValueError(f"Conflicting duplicate candle at epoch {timestamp}")
            duplicates += 1
        else:
            unique[timestamp] = values

    timestamps = sorted(unique)
    frame = pd.DataFrame(
        [(stamp, *unique[stamp]) for stamp in timestamps],
        columns=["timestamp", "low", "high", "open", "close", "volume"],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = frame[column].astype("float64")
    frame = frame[["timestamp", "open", "high", "low", "close", "volume"]]
    frame["available_at"] = frame["timestamp"] + pd.to_timedelta(seconds, unit="s")
    missing = [stamp for stamp in range(start, end, seconds) if stamp not in unique]
    report = QualityReport(
        raw_rows=len(rows), cleaned_rows=len(frame), duplicate_rows_removed=duplicates,
        out_of_range_rows_removed=outside, expected_rows=(end - start) // seconds,
        missing_count=len(missing),
        missing_timestamps_utc=[pd.Timestamp(stamp, unit="s", tz="UTC").isoformat() for stamp in missing],
    )
    return frame, report
