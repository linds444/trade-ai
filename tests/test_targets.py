from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from local_ai_trader import cli
from local_ai_trader.data.dataset import TARGET_COLUMNS, build_targets, create_target_dataset
from local_ai_trader.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]


def candles(closes=None):
    if closes is None:
        closes = np.arange(100, 112, dtype=float)
    timestamps = pd.date_range("2025-01-01", periods=len(closes), freq="5min", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps, "available_at": timestamps + pd.Timedelta(5, unit="min"),
        "close": closes, "symbol": "BTC-USD", "exchange": "coinbase",
    })


@pytest.fixture
def settings(tmp_path):
    config = tmp_path / "config/settings.toml"
    config.parent.mkdir()
    config.write_text((ROOT / "config/settings.toml").read_text(), encoding="utf-8")
    return load_settings(config)


def test_six_step_indexing_and_label_availability():
    source = candles()
    result = build_targets(source, 6, 300, 0.001)
    assert len(result) == 6
    for index in range(len(result)):
        assert result.loc[index, "target_return"] == pytest.approx((106 + index) / (100 + index) - 1)
        assert result.loc[index, "target_available_at"] == source.loc[index + 6, "available_at"]
    assert (result["target_available_at"] - result["available_at"]).eq(pd.Timedelta(30, unit="min")).all()
    assert result["target_class"].tolist() == ["up"] * 6
    assert not result[list(TARGET_COLUMNS)].isna().any().any()
    assert not any(column.startswith("target_") for column in source.columns)


def test_up_flat_down():
    result = build_targets(candles([100, 120, 120, 96]), 1, 300, 0.1)
    assert result["target_class"].tolist() == ["up", "flat", "down"]
    np.testing.assert_allclose(result["target_return"], [0.2, 0, -0.2])


def test_exact_boundaries_are_flat():
    result = build_targets(candles([100, 112.5, 126.5625, 110.7421875]), 1, 300, 0.125)
    assert result["target_class"].tolist() == ["flat"] * 3


def test_decimal_boundary_roundoff_is_flat():
    result = build_targets(candles([100, 100.1, 100.1 * 0.999]), 1, 300, 0.001)
    assert result["target_class"].tolist() == ["flat", "flat"]


def test_zero_threshold_is_supported():
    result = build_targets(candles([100, 101, 101, 100]), 1, 300, 0)
    assert result["target_class"].tolist() == ["up", "flat", "down"]


def test_future_changes_cannot_modify_current_candle_values():
    source = candles()
    before = build_targets(source, 6, 300, 0.001)
    changed = source.copy()
    changed.loc[6:, "close"] *= 2
    after = build_targets(changed, 6, 300, 0.001)
    pd.testing.assert_frame_equal(before.drop(columns=list(TARGET_COLUMNS)), after.drop(columns=list(TARGET_COLUMNS)))
    assert not before["target_return"].equals(after["target_return"])


@pytest.mark.parametrize("mutation", ["gap", "duplicate", "unsorted", "wrong_availability", "misaligned", "naive", "missing_time", "mixed_symbols", "mixed_exchanges", "missing_symbol", "existing_target", "missing_close"])
def test_invalid_dataset_rejected(mutation):
    source = candles()
    if mutation == "gap":
        source = source.drop(index=5)
    elif mutation == "duplicate":
        source.loc[5, "timestamp"] = source.loc[4, "timestamp"]
    elif mutation == "unsorted":
        source = source.iloc[::-1]
    elif mutation == "wrong_availability":
        source.loc[5, "available_at"] += pd.Timedelta(5, unit="min")
    elif mutation == "misaligned":
        source["timestamp"] += pd.Timedelta(1, unit="s")
        source["available_at"] += pd.Timedelta(1, unit="s")
    elif mutation == "naive":
        source["timestamp"] = source["timestamp"].dt.tz_localize(None)
    elif mutation == "missing_time":
        source.loc[5, "timestamp"] = pd.NaT
    elif mutation == "mixed_symbols":
        source.loc[5, "symbol"] = "ETH-USD"
    elif mutation == "mixed_exchanges":
        source.loc[5, "exchange"] = "other"
    elif mutation == "missing_symbol":
        source.loc[5, "symbol"] = None
    elif mutation == "existing_target":
        source["target_return"] = 0
    else:
        source = source.drop(columns="close")
    with pytest.raises(ValueError):
        build_targets(source, 6, 300, 0.001)


@pytest.mark.parametrize("price", [0, -1, np.nan, np.inf, -np.inf])
def test_invalid_close_rejected(price):
    source = candles()
    source.loc[5, "close"] = price
    with pytest.raises(ValueError, match="positive and finite"):
        build_targets(source, 6, 300, 0.001)


@pytest.mark.parametrize("threshold", [-0.001, 1, np.inf, np.nan, True, "0.001"])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValueError):
        build_targets(candles(), 6, 300, threshold)


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True, 12, 13])
def test_invalid_or_unavailable_horizon_rejected(horizon):
    with pytest.raises(ValueError):
        build_targets(candles(), horizon, 300, 0.001)


def test_future_candles_rejected():
    source = candles()
    source["timestamp"] += pd.Timedelta(365 * 100, unit="D")
    source["available_at"] += pd.Timedelta(365 * 100, unit="D")
    with pytest.raises(ValueError, match="future"):
        build_targets(source, 6, 300, 0.001)


def test_nonstandard_index_and_timezone_normalized():
    source = candles()
    source.index = np.arange(100, 112)
    for column in ("timestamp", "available_at"):
        source[column] = source[column].dt.tz_convert("America/Los_Angeles")
    result = build_targets(source, 6, 300, 0.001)
    assert result.index.tolist() == list(range(6))
    assert str(result["target_available_at"].dtype) == "datetime64[ns, UTC]"
    assert result["target_available_at"].iloc[0] == pd.Timestamp("2025-01-01T00:35:00Z")


def test_seven_day_dataset_has_2010_labels():
    source = candles(np.full(2016, 100.0))
    result = build_targets(source, 6, 300, 0.001)
    assert len(result) == 2010
    assert result["target_class"].eq("flat").all()


def test_parquet_and_metadata_roundtrip(settings, tmp_path):
    source = tmp_path / "candles.parquet"
    candles().to_parquet(source, index=False)
    original = source.read_bytes()
    output, report = create_target_dataset(source, settings)
    restored = pd.read_parquet(output)
    assert len(restored) == 6
    assert report["class_counts"] == {"up": 6, "down": 0, "flat": 0}
    assert report["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert report["horizon_minutes"] == 30
    assert report["flat_return_threshold"] == 0.001
    assert source.read_bytes() == original
    assert json.loads(output.with_suffix(".targets.json").read_text()) == report
    with pytest.raises(ValueError, match="already exists"):
        create_target_dataset(source, replace(settings, flat_return_threshold=0.002))


def test_cli_targets(settings, tmp_path, capsys):
    source = tmp_path / "candles.parquet"
    candles().to_parquet(source, index=False)
    config = tmp_path / "config/settings.toml"
    assert cli.main(["targets", str(source), "--config", str(config)]) == 0
    output = settings.processed_dir / "targets/candles_h6.parquet"
    assert cli.main(["inspect", str(output)]) == 0
    assert "BTC-USD" in capsys.readouterr().out
    assert cli.main(["targets", str(source), "--config", str(config)]) == 1


def test_missing_source_fails(settings, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        create_target_dataset(tmp_path / "missing.parquet", settings)


@pytest.mark.parametrize("prices", [["100"] * 12, [True] * 12])
def test_nonnumeric_or_boolean_closes_fail(prices):
    with pytest.raises(ValueError, match="numeric"):
        build_targets(candles(prices), 6, 300, 0.001)


def test_overflowing_return_fails_without_output():
    with pytest.raises(ValueError, match="nonfinite returns"):
        build_targets(candles([np.finfo(float).tiny / 2, np.finfo(float).max]), 1, 300, 0.001)


@pytest.mark.parametrize("value", ["true", '"0.001"', "-0.001", "1.0", "nan"])
def test_config_rejects_invalid_flat_threshold(settings, tmp_path, value):
    config = tmp_path / "config/settings.toml"
    config.write_text(config.read_text().replace("flat_return_threshold = 0.001", f"flat_return_threshold = {value}"))
    with pytest.raises(ValueError, match="flat_return_threshold"):
        load_settings(config)
