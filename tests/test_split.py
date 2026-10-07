from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from local_ai_trader import cli
from local_ai_trader.data import split as split_module
from local_ai_trader.data.dataset import build_targets, create_target_dataset
from local_ai_trader.data.split import chronological_split, create_split_dataset
from local_ai_trader.features.build_features import FEATURE_COLUMNS, build_features, create_feature_dataset
from local_ai_trader.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]


def candles(count=2016):
    close = 100 + np.arange(count, dtype=float) * 0.01
    timestamps = pd.date_range("2025-01-01", periods=count, freq="5min", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps, "available_at": timestamps + pd.to_timedelta(300, unit="s"),
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.full(count, 10.0),
        "symbol": "BTC-USD", "exchange": "coinbase",
    })


def feature_frame(count=2016):
    return build_features(build_targets(candles(count), 6, 300, 0.001), 300, 6, 12)


@pytest.fixture
def settings(tmp_path):
    path = tmp_path / "config/settings.toml"
    path.parent.mkdir()
    path.write_text((ROOT / "config/settings.toml").read_text(), encoding="utf-8")
    return load_settings(path)


def feature_snapshot(tmp_path, settings):
    source = tmp_path / "candles.parquet"
    candles(200).to_parquet(source, index=False)
    labelled = create_target_dataset(source, settings)[0]
    return create_feature_dataset(labelled, settings)[0]


def test_default_seven_day_split_accounting():
    source = feature_frame()
    partitions, report = chronological_split(source, 300, 6, 0.6, 0.2)
    assert len(source) == 1998
    assert {name: len(data) for name, data in partitions.items()} == {"train": 1192, "validation": 394, "test": 400}
    assert report["cut_positions"] == {"validation_start": 1198, "test_start": 1598}
    assert [report["partitions"][name]["purged_rows"] for name in partitions] == [6, 6, 0]
    retained = sum(len(data) for data in partitions.values())
    removed = sum(data["purged_rows"] for data in report["partitions"].values())
    assert retained + removed == len(source)
    for name, data in partitions.items():
        assert data["timestamp"].is_monotonic_increasing
        assert sum(report["partitions"][name]["class_counts"].values()) == len(data)


def test_label_boundary_equality_is_purged():
    source = feature_frame()
    partitions, report = chronological_split(source, 300, 6, 0.6, 0.2)
    validation_start = partitions["validation"]["available_at"].iloc[0]
    test_start = partitions["test"]["available_at"].iloc[0]
    assert (partitions["train"]["target_available_at"] < validation_start).all()
    assert (partitions["validation"]["target_available_at"] < test_start).all()
    train_cut = report["cut_positions"]["validation_start"]
    equality_row = source.iloc[train_cut - 6]
    assert equality_row["target_available_at"] == validation_start
    assert equality_row["timestamp"] not in set(partitions["train"]["timestamp"])
    preceding_row = source.iloc[train_cut - 7]
    assert preceding_row["timestamp"] == partitions["train"]["timestamp"].iloc[-1]
    assert len(report["partitions"]["train"]["purged_prediction_times"]) == 6


def test_split_does_not_shuffle_or_modify_feature_values():
    source = feature_frame()
    original = source.copy(deep=True)
    partitions, report = chronological_split(source, 300, 6, 0.6, 0.2)
    test_cut = report["cut_positions"]["test_start"]
    pd.testing.assert_frame_equal(partitions["test"], source.iloc[test_cut:].reset_index(drop=True))
    pd.testing.assert_frame_equal(source, original)
    assert set(partitions["train"]["timestamp"]).isdisjoint(partitions["validation"]["timestamp"])
    assert set(partitions["validation"]["timestamp"]).isdisjoint(partitions["test"]["timestamp"])


def test_future_test_values_do_not_change_training_partition():
    source = feature_frame()
    before, report = chronological_split(source, 300, 6, 0.6, 0.2)
    changed = source.copy()
    changed.loc[report["cut_positions"]["test_start"]:, list(FEATURE_COLUMNS)] *= 10
    changed.loc[report["cut_positions"]["test_start"]:, "target_class"] = "down"
    after, _ = chronological_split(changed, 300, 6, 0.6, 0.2)
    pd.testing.assert_frame_equal(before["train"], after["train"])
    pd.testing.assert_frame_equal(before["validation"], after["validation"])


def test_configurable_cut_positions():
    partitions, report = chronological_split(feature_frame(), 300, 6, 0.5, 0.25)
    assert report["cut_positions"] == {"validation_start": 999, "test_start": 1498}
    assert {name: len(data) for name, data in partitions.items()} == {"train": 993, "validation": 493, "test": 500}


@pytest.mark.parametrize("train,validation", [
    (0, 0.2), (-0.1, 0.2), (1, 0.2), (0.6, 0), (0.8, 0.2), (0.9, 0.2),
    (np.nan, 0.2), (0.6, np.inf), (True, 0.2), (0.6, "0.2"),
])
def test_invalid_fractions_fail(train, validation):
    with pytest.raises(ValueError):
        chronological_split(feature_frame(), 300, 6, train, validation)


def test_short_dataset_fails_after_purging():
    with pytest.raises(ValueError, match="empty after label purging"):
        chronological_split(feature_frame(38), 300, 6, 0.6, 0.2)


@pytest.mark.parametrize("mutation", ["unsorted", "gap", "missing_feature", "nonfinite_feature", "wrong_delay", "missing_label_time", "naive_label_time", "invalid_class", "nonfinite_return", "mixed_symbols"])
def test_invalid_datasets_rejected(mutation):
    source = feature_frame()
    if mutation == "unsorted":
        source = source.iloc[::-1]
    elif mutation == "gap":
        source = source.drop(index=200)
    elif mutation == "missing_feature":
        source = source.drop(columns=FEATURE_COLUMNS[0])
    elif mutation == "nonfinite_feature":
        source.loc[200, FEATURE_COLUMNS[0]] = np.inf
    elif mutation == "wrong_delay":
        source.loc[200, "target_available_at"] += pd.to_timedelta(300, unit="s")
    elif mutation == "missing_label_time":
        source.loc[200, "target_available_at"] = pd.NaT
    elif mutation == "naive_label_time":
        source["target_available_at"] = source["target_available_at"].dt.tz_localize(None)
    elif mutation == "invalid_class":
        source.loc[200, "target_class"] = "BUY"
    elif mutation == "nonfinite_return":
        source.loc[200, "target_return"] = np.nan
    else:
        source.loc[200, "symbol"] = "ETH-USD"
    with pytest.raises(ValueError):
        chronological_split(source, 300, 6, 0.6, 0.2)


def test_output_bundle_metadata_and_no_overwrite(tmp_path, settings):
    source = feature_snapshot(tmp_path, settings)
    original = source.read_bytes()
    directory, report = create_split_dataset(source, settings)
    assert {file.name for file in directory.iterdir()} == {"train.parquet", "validation.parquet", "test.parquet", "split.json"}
    assert json.loads((directory / "split.json").read_text()) == report
    assert report["feature_names"] == list(FEATURE_COLUMNS)
    assert report["source_sha256"] == hashlib.sha256(original).hexdigest()
    for name in ("train", "validation", "test"):
        assert len(pd.read_parquet(directory / f"{name}.parquet")) == report["partitions"][name]["rows"]
    assert source.read_bytes() == original
    with pytest.raises(ValueError, match="already exists"):
        create_split_dataset(source, settings)


def test_failed_bundle_write_never_publishes_partial_splits(tmp_path, settings, monkeypatch):
    source = feature_snapshot(tmp_path, settings)

    def fail_report(*args, **kwargs):
        raise OSError("Simulated disk write failure")

    monkeypatch.setattr(split_module, "write_json", fail_report)
    with pytest.raises(OSError):
        create_split_dataset(source, settings)
    split_root = settings.processed_dir / "splits"
    assert not (split_root / source.stem).exists()
    assert not list(split_root.glob(".split-staging-*"))


@pytest.mark.parametrize("key", ["candle_seconds", "momentum_steps", "rolling_window", "horizon_steps", "flat_return_threshold"])
def test_incompatible_settings_rejected(tmp_path, settings, key):
    source = feature_snapshot(tmp_path, settings)
    changed = replace(settings, **{key: getattr(settings, key) * 2})
    with pytest.raises(ValueError, match="disagree"):
        create_split_dataset(source, changed)


@pytest.mark.parametrize("mutation", ["missing_metadata", "bad_allowlist", "bad_rows"])
def test_invalid_metadata_rejected(tmp_path, settings, mutation):
    source = feature_snapshot(tmp_path, settings)
    metadata = source.with_suffix(".features.json")
    if mutation == "missing_metadata":
        metadata.unlink()
    else:
        report = json.loads(metadata.read_text())
        if mutation == "bad_allowlist":
            report["feature_names"].append("target_return")
        else:
            report["feature_rows"] += 1
        metadata.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        create_split_dataset(source, settings)


def test_cli_split(tmp_path, settings):
    source = feature_snapshot(tmp_path, settings)
    assert cli.main(["split", str(source), "--config", str(tmp_path / "config/settings.toml")]) == 0
    assert cli.main(["split", str(source), "--config", str(tmp_path / "config/settings.toml")]) == 1


@pytest.mark.parametrize("key,value", [("train_fraction", "0.0"), ("train_fraction", "true"), ("validation_fraction", "0.5"), ("validation_fraction", "nan")])
def test_config_invalid_fractions_rejected(tmp_path, settings, key, value):
    config = tmp_path / "config/settings.toml"
    original = 0.6 if key == "train_fraction" else 0.2
    config.write_text(config.read_text().replace(f"{key} = {original}", f"{key} = {value}"))
    with pytest.raises(ValueError):
        load_settings(config)
