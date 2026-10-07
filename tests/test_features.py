from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from local_ai_trader import cli
from local_ai_trader.data.dataset import TARGET_COLUMNS, build_targets, create_target_dataset
from local_ai_trader.features.build_features import FEATURE_COLUMNS, build_features, create_feature_dataset
from local_ai_trader.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]


def candles(count=80):
    close = np.arange(100, 100 + count, dtype=float)
    timestamps = pd.date_range("2025-01-01", periods=count, freq="5min", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps, "available_at": timestamps + pd.to_timedelta(300, unit="s"),
        "open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.arange(10, 10 + count, dtype=float),
        "symbol": "BTC-USD", "exchange": "coinbase",
    })


@pytest.fixture
def settings(tmp_path):
    path = tmp_path / "config/settings.toml"
    path.parent.mkdir()
    path.write_text((ROOT / "config/settings.toml").read_text(), encoding="utf-8")
    return load_settings(path)


def test_feature_values_against_independent_scalar_formulas():
    source = candles()
    result = build_features(source, 300, 6, 12)
    first = result.iloc[0]
    assert len(result) == len(source) - 12
    assert first["timestamp"] == source.loc[12, "timestamp"]
    assert first["feature_return_1"] == pytest.approx(112 / 111 - 1)
    assert first["feature_momentum"] == pytest.approx(112 / 106 - 1)
    log_returns = [math.log((101 + i) / (100 + i)) for i in range(12)]
    assert first["feature_volatility"] == pytest.approx(np.std(log_returns, ddof=0))
    assert first["feature_close_vs_mean"] == pytest.approx(112 / np.mean(np.arange(101, 113)) - 1)
    assert first["feature_volume_ratio"] == pytest.approx(22 / np.mean(np.arange(11, 23)))
    assert first["feature_range"] == pytest.approx(0.02)
    assert first["feature_body"] == pytest.approx((112 - 112 * 0.999) / (112 * 0.999))
    assert first["feature_close_location"] == pytest.approx(0.5)
    assert np.isfinite(result[list(FEATURE_COLUMNS)]).all().all()


def test_future_candle_changes_cannot_change_past_features():
    original = candles()
    changed = original.copy()
    changed.loc[41:, ["open", "high", "low", "close"]] *= 3
    changed.loc[41:, "volume"] += 500
    before = build_features(original, 300, 6, 12)
    after = build_features(changed, 300, 6, 12)
    cutoff = original.loc[40, "timestamp"]
    pd.testing.assert_frame_equal(
        before.loc[before["timestamp"] <= cutoff, list(FEATURE_COLUMNS)],
        after.loc[after["timestamp"] <= cutoff, list(FEATURE_COLUMNS)],
    )


def test_truncating_future_data_leaves_prefix_features_unchanged():
    source = candles()
    full = build_features(source, 300, 6, 12)
    prefix = build_features(source.iloc[:41], 300, 6, 12)
    pd.testing.assert_frame_equal(full.iloc[:len(prefix)], prefix)


def test_targets_are_preserved_but_never_used_as_predictors():
    labelled = build_targets(candles(), 6, 300, 0.001)
    before = build_features(labelled, 300, 6, 12)
    changed = labelled.copy()
    changed["target_return"] = 1e9
    changed["target_class"] = "down"
    changed["target_available_at"] += pd.to_timedelta(86400, unit="s")
    after = build_features(changed, 300, 6, 12)
    pd.testing.assert_frame_equal(before[list(FEATURE_COLUMNS)], after[list(FEATURE_COLUMNS)])
    pd.testing.assert_frame_equal(before[list(TARGET_COLUMNS)], labelled.iloc[12:].reset_index(drop=True)[list(TARGET_COLUMNS)])
    assert not set(FEATURE_COLUMNS).intersection(TARGET_COLUMNS)


def test_flat_zero_volume_candles_produce_finite_neutral_features():
    source = candles()
    source[["open", "high", "low", "close"]] = 100.0
    source["volume"] = 0.0
    result = build_features(source, 300, 6, 12)
    assert result["feature_volume_ratio"].eq(0).all()
    assert result["feature_close_location"].eq(0.5).all()
    for column in set(FEATURE_COLUMNS) - {"feature_close_location"}:
        assert result[column].eq(0).all()


@pytest.mark.parametrize("momentum,window,warmup", [(20, 12, 20), (6, 24, 24), (1, 2, 2)])
def test_configurable_warmup(momentum, window, warmup):
    source = candles()
    result = build_features(source, 300, momentum, window)
    assert len(result) == len(source) - warmup
    assert result["timestamp"].iloc[0] == source["timestamp"].iloc[warmup]


@pytest.mark.parametrize("column,value", [
    ("open", 0), ("high", -1), ("low", 0), ("volume", -1),
    ("volume", np.inf), ("open", np.nan), ("high", 1), ("low", 1000),
])
def test_invalid_ohlcv_fails(column, value):
    source = candles()
    source.loc[30, column] = value
    with pytest.raises(ValueError):
        build_features(source, 300, 6, 12)


@pytest.mark.parametrize("mutation", ["gap", "duplicate", "unsorted", "missing_volume", "existing_features", "short", "future"])
def test_invalid_input_fails(mutation):
    source = candles()
    if mutation == "gap":
        source = source.drop(index=30)
    elif mutation == "duplicate":
        source.loc[30, "timestamp"] = source.loc[29, "timestamp"]
    elif mutation == "unsorted":
        source = source.iloc[::-1]
    elif mutation == "missing_volume":
        source = source.drop(columns="volume")
    elif mutation == "existing_features":
        source["feature_return_1"] = 0.0
    elif mutation == "short":
        source = source.iloc[:12]
    else:
        source["timestamp"] += pd.to_timedelta(36500, unit="D")
        source["available_at"] += pd.to_timedelta(36500, unit="D")
    with pytest.raises(ValueError):
        build_features(source, 300, 6, 12)


@pytest.mark.parametrize("momentum,window", [(0, 12), (1.5, 12), (6, 1), (6, True)])
def test_invalid_parameters_fail(momentum, window):
    with pytest.raises(ValueError):
        build_features(candles(), 300, momentum, window)


def labelled_snapshot(tmp_path, settings):
    source = tmp_path / "candles.parquet"
    candles().to_parquet(source, index=False)
    return create_target_dataset(source, settings)[0]


def test_metadata_and_storage_roundtrip(tmp_path, settings):
    labelled = labelled_snapshot(tmp_path, settings)
    original = labelled.read_bytes()
    output, report = create_feature_dataset(labelled, settings)
    frame = pd.read_parquet(output)
    assert len(frame) == 62  # 80 candles - 6 unknown labels - 12 warmup
    assert report["feature_names"] == list(FEATURE_COLUMNS)
    assert report["warmup_rows_removed"] == 12
    assert report["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert report["source_rows"] == 74
    assert sum(report["class_counts"].values()) == 62
    assert json.loads(output.with_suffix(".features.json").read_text()) == report
    assert labelled.read_bytes() == original
    with pytest.raises(ValueError, match="already exists"):
        create_feature_dataset(labelled, settings)


def test_seven_day_labelled_dataset_has_1998_feature_rows():
    labelled = build_targets(candles(2016), 6, 300, 0.001)
    result = build_features(labelled, 300, 6, 12)
    assert len(result) == 1998


@pytest.mark.parametrize("key", ["horizon_steps", "candle_seconds", "flat_return_threshold"])
def test_mismatched_label_settings_fail(tmp_path, settings, key):
    labelled = labelled_snapshot(tmp_path, settings)
    changed = replace(settings, **{key: getattr(settings, key) * 2})
    with pytest.raises(ValueError, match="disagree"):
        create_feature_dataset(labelled, changed)


@pytest.mark.parametrize("mutation", ["missing_metadata", "missing_target", "bad_class", "bad_return", "bad_time"])
def test_invalid_target_snapshot_fails(tmp_path, settings, mutation):
    labelled = labelled_snapshot(tmp_path, settings)
    if mutation == "missing_metadata":
        labelled.with_suffix(".targets.json").unlink()
    else:
        frame = pd.read_parquet(labelled)
        if mutation == "missing_target":
            frame = frame.drop(columns="target_return")
        elif mutation == "bad_class":
            frame.loc[20, "target_class"] = "BUY"
        elif mutation == "bad_return":
            frame.loc[20, "target_return"] = np.nan
        else:
            frame.loc[20, "target_available_at"] += pd.to_timedelta(300, unit="s")
        frame.to_parquet(labelled, index=False)
    with pytest.raises(ValueError):
        create_feature_dataset(labelled, settings)
    assert not list((settings.processed_dir / "features").glob("*.parquet"))


def test_cli_features(tmp_path, settings, capsys):
    source = labelled_snapshot(tmp_path, settings)
    assert cli.main(["features", str(source), "--config", str(tmp_path / "config/settings.toml")]) == 0
    output = next((settings.processed_dir / "features").glob("*.parquet"))
    assert cli.main(["inspect", str(output)]) == 0
    assert "BTC-USD" in capsys.readouterr().out


@pytest.mark.parametrize("key,value", [("momentum_steps", "0"), ("momentum_steps", "true"), ("rolling_window", "1"), ("rolling_window", "1.5")])
def test_settings_reject_invalid_feature_parameters(tmp_path, settings, key, value):
    config = tmp_path / "config/settings.toml"
    original = 6 if key == "momentum_steps" else 12
    config.write_text(config.read_text().replace(f"{key} = {original}", f"{key} = {value}"))
    with pytest.raises(ValueError, match=key):
        load_settings(config)
