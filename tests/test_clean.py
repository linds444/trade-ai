import math

import pandas as pd
import pytest

from local_ai_trader.data.clean import clean_candles


def candle(timestamp: int) -> list:
    return [timestamp, 99, 102, 100, 101, 10]


def test_sort_deduplicate_filter_and_availability():
    rows = [candle(300), candle(0), candle(300), candle(600)]
    frame, report = clean_candles(rows, 0, 600, 300)
    assert list(frame["timestamp"]) == list(pd.to_datetime([0, 300], unit="s", utc=True))
    assert list(frame["available_at"]) == list(pd.to_datetime([300, 600], unit="s", utc=True))
    assert report.duplicate_rows_removed == 1
    assert report.out_of_range_rows_removed == 1
    assert report.expected_rows == report.cleaned_rows == 2
    assert report.missing_count == 0
    assert str(frame["timestamp"].dtype) == "datetime64[ns, UTC]"


def test_gaps_include_missing_first_middle_and_last_candle():
    frame, report = clean_candles([candle(300), candle(900)], 0, 1500, 300)
    assert len(frame) == 2
    assert report.missing_count == 3
    assert report.missing_timestamps_utc == [
        pd.Timestamp(stamp, unit="s", tz="UTC").isoformat() for stamp in [0, 600, 1200]
    ]


def test_empty_response_is_all_missing():
    frame, report = clean_candles([], 0, 600, 300)
    assert frame.empty
    assert report.missing_count == 2


def test_conflicting_duplicates_fail():
    conflicting = candle(0)
    conflicting[-1] = 11
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        clean_candles([candle(0), conflicting], 0, 300, 300)


@pytest.mark.parametrize("row", [
    [0, 99, 102, 100, 101],
    [0, 99, 102, 100, math.nan, 10],
    [0, 99, math.inf, 100, 101, 10],
    [0, 99, 102, 100, 101, -1],
    [0, -1, 102, 100, 101, 10],
    [0, 99, 100, 100, 101, 10],
    [0, 101, 102, 100, 101, 10],
    [1, 99, 102, 100, 101, 10],
    [0.5, 99, 102, 100, 101, 10],
    [False, 99, 102, 100, 101, 10],
    [0, 99, 102, "100", 101, 10],
])
def test_invalid_candles_fail(row):
    with pytest.raises(ValueError):
        clean_candles([row], 0, 300, 300)


@pytest.mark.parametrize("start,end,seconds", [(0, 0, 300), (300, 0, 300), (1, 300, 300), (0, 300, 0)])
def test_invalid_bounds_fail(start, end, seconds):
    with pytest.raises(ValueError):
        clean_candles([], start, end, seconds)
