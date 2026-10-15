from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from local_ai_trader import cli
from local_ai_trader.data.storage import native_path
from local_ai_trader.features.build_features import FEATURE_COLUMNS
from local_ai_trader.forecast import features, split, targets
from local_ai_trader.models.train import file_digest
from local_ai_trader.settings import BacktestSettings, load_settings
from test_execution_targets import candles

ROOT = Path(__file__).resolve().parents[1]


def feature_frame(count=200, latency=1, holding=6):
    labelled = targets.build_execution_targets(candles(count), 300, holding, replace(BacktestSettings(), latency_bars=latency))
    return features.build_execution_features(labelled, 300, 6, 12)


@pytest.fixture
def bundle(tmp_path):
    config = tmp_path / "config/settings.toml"
    config.parent.mkdir()
    config.write_text((ROOT / "config/settings.toml").read_text(), encoding="utf-8")
    settings = load_settings(config)
    settings.processed_dir.mkdir(parents=True)
    source = settings.processed_dir / "BTC_candles.parquet"
    candles(200).to_parquet(source, index=False)
    labelled, _ = targets.create_execution_target_dataset(source, settings)
    directory, _ = features.create_execution_feature_dataset(labelled, settings)
    return directory, settings, config


def rewrite_metadata(directory, **updates):
    path = directory / "features.json"
    metadata = json.loads(path.read_text())
    metadata.update(updates)
    path.write_text(json.dumps(metadata))


def test_143_day_split_counts_and_complete_row_accounting():
    frame = feature_frame(41184)
    partitions, report = split.chronological_execution_split(frame, 300, 6, 1, 0.6, 0.2)
    assert len(frame) == 41164
    assert report["cut_positions"] == {"validation_start": 24698, "test_start": 32931}
    assert {name: len(data) for name, data in partitions.items()} == {"train": 24690, "validation": 8225, "test": 8233}
    assert [report["partitions"][name]["purged_rows"] for name in partitions] == [8, 8, 0]
    assert sum(len(data) for data in partitions.values()) + sum(item["purged_rows"] for item in report["partitions"].values()) == len(frame)


def test_equality_and_the_two_rows_missed_by_a_classifier_purge_are_removed():
    frame = feature_frame()
    partitions, report = split.chronological_execution_split(frame, 300, 6, 1, 0.6, 0.2)
    for earlier, later, cut_key in (("train", "validation", "validation_start"), ("validation", "test", "test_start")):
        cut = report["cut_positions"][cut_key]
        boundary = partitions[later].available_at.iloc[0]
        assert frame.execution_target_available_at.iloc[cut - 8] == boundary
        assert (partitions[earlier].execution_target_available_at < boundary).all()
        assert partitions[earlier].available_at.iloc[-1] == frame.available_at.iloc[cut - 9]
        removed = set(report["partitions"][earlier]["purged_prediction_times"])
        assert len(removed) == 8
        for index in (cut - 8, cut - 7):
            assert frame.available_at.iloc[index].isoformat() in removed


@pytest.mark.parametrize("latency,holding", [(1, 1), (2, 6), (3, 12)])
def test_purge_follows_full_execution_availability_for_other_windows(latency, holding):
    partitions, report = split.chronological_execution_split(feature_frame(500, latency, holding), 300, holding, latency, 0.6, 0.2)
    expected = latency + holding + 1
    assert report["target_known_offset_from_prediction_bars"] == expected
    assert [report["partitions"][name]["purged_rows"] for name in partitions] == [expected, expected, 0]
    assert len(partitions["test"]) == report["partitions"]["test"]["raw_rows"]


def test_split_preserves_order_values_and_original_frame():
    frame = feature_frame()
    original = frame.copy(deep=True)
    partitions, report = split.chronological_execution_split(frame, 300, 6, 1, 0.6, 0.2)
    pd.testing.assert_frame_equal(frame, original)
    test_cut = report["cut_positions"]["test_start"]
    pd.testing.assert_frame_equal(partitions["test"], frame.iloc[test_cut:].reset_index(drop=True))
    timestamps = [set(partition.timestamp) for partition in partitions.values()]
    assert timestamps[0].isdisjoint(timestamps[1])
    assert timestamps[0].isdisjoint(timestamps[2])
    assert timestamps[1].isdisjoint(timestamps[2])
    assert all(data.available_at.is_monotonic_increasing for data in partitions.values())


def test_future_test_features_and_outcomes_cannot_change_earlier_partitions():
    frame = feature_frame()
    before, report = split.chronological_execution_split(frame, 300, 6, 1, 0.6, 0.2)
    changed = frame.copy()
    cut = report["cut_positions"]["test_start"]
    changed.loc[cut:, list(FEATURE_COLUMNS)] *= 10
    changed.loc[cut:, "execution_target_return"] = 0.123
    changed.loc[cut:, "execution_target_net_return"] = 0.1
    after, _ = split.chronological_execution_split(changed, 300, 6, 1, 0.6, 0.2)
    pd.testing.assert_frame_equal(before["train"], after["train"])
    pd.testing.assert_frame_equal(before["validation"], after["validation"])


@pytest.mark.parametrize("mutation", ["classifier_label", "missing_feature", "boolean_feature", "nonfinite_feature", "wrong_entry", "wrong_exit", "six_bar_known", "naive_known", "missing_known", "gap", "unordered", "nonfinite_target", "nonpositive_reference"])
def test_invalid_schemas_inputs_and_label_times_cannot_be_partitioned(mutation):
    frame = feature_frame()
    if mutation == "classifier_label":
        frame["target_return"] = 0.0
    elif mutation == "missing_feature":
        frame = frame.drop(columns=FEATURE_COLUMNS[0])
    elif mutation == "boolean_feature":
        frame[FEATURE_COLUMNS[0]] = True
    elif mutation == "nonfinite_feature":
        frame.loc[20, FEATURE_COLUMNS[0]] = np.inf
    elif mutation == "wrong_entry":
        frame["execution_entry_at"] += pd.to_timedelta(300, unit="s")
    elif mutation == "wrong_exit":
        frame["execution_exit_at"] -= pd.to_timedelta(300, unit="s")
    elif mutation == "six_bar_known":
        frame["execution_target_available_at"] -= pd.to_timedelta(600, unit="s")
    elif mutation == "naive_known":
        frame["execution_target_available_at"] = frame.execution_target_available_at.dt.tz_localize(None)
    elif mutation == "missing_known":
        frame.loc[20, "execution_target_available_at"] = pd.NaT
    elif mutation == "gap":
        frame = frame.drop(index=20)
    elif mutation == "unordered":
        frame = frame.iloc[::-1]
    elif mutation == "nonfinite_target":
        frame.loc[20, "execution_target_return"] = np.nan
    else:
        frame.loc[20, "execution_entry_reference_open"] = 0.0
    with pytest.raises(ValueError):
        split.chronological_execution_split(frame, 300, 6, 1, 0.6, 0.2)


def test_empty_validation_after_purging_is_rejected():
    with pytest.raises(ValueError, match="empty after execution label purging"):
        split.chronological_execution_split(feature_frame(40), 300, 6, 1, 0.6, 0.2)


@pytest.mark.parametrize("train,validation", [(True, 0.2), (0.8, 0.2), (0.6, np.nan)])
def test_invalid_fraction_settings_are_rejected(train, validation):
    with pytest.raises(ValueError):
        split.chronological_execution_split(feature_frame(), 300, 6, 1, train, validation)


def test_bundle_audit_hashes_no_scaling_and_immutable_split_publication(bundle):
    directory, settings, _ = bundle
    frame, metadata, inputs = features.load_execution_feature_bundle(directory)
    original = {path: path.read_bytes() for path in inputs}
    output, report = split.create_execution_split_dataset(directory, settings)
    assert {path.name for path in output.iterdir()} == {"train.parquet", "validation.parquet", "test.parquet", "split.json"}
    assert report["feature_names"] == list(FEATURE_COLUMNS)
    assert report["future_only_columns"] == list(targets.FUTURE_COLUMNS)
    assert report["feature_metadata"] == metadata
    assert report["input_hashes"] == {str(path): file_digest(path) for path in inputs}
    assert report["models_fitted"] is False
    assert report["strategy_performance_evaluated"] is False
    assert report["target_available_column"] == "execution_target_available_at"
    assert "freshness relative to prior research is not established" in report["reserved_test_status"]
    assert json.loads((output / "split.json").read_text()) == report
    assert {name: summary["rows"] for name, summary in report["partitions"].items()} == {"train": 100, "validation": 28, "test": 36}
    for name in ("train", "validation", "test"):
        assert file_digest(output / f"{name}.parquet") == report["partitions"][name]["sha256"]
    pd.testing.assert_frame_equal(pd.read_parquet(output / "train.parquet")[list(FEATURE_COLUMNS)], frame.iloc[:100][list(FEATURE_COLUMNS)].reset_index(drop=True))
    assert {path: path.read_bytes() for path in inputs} == original
    saved = (output / "split.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        split.create_execution_split_dataset(directory, settings)
    assert (output / "split.json").read_bytes() == saved


@pytest.mark.parametrize("key,value", [("feature_names", ["execution_target_return"]), ("feature_rows", 179), ("target_known_offset_from_prediction_bars", 6), ("future_only_columns", []), ("models_fitted", True)])
def test_wrong_feature_metadata_is_rejected_before_split_publication(bundle, key, value):
    directory, settings, _ = bundle
    rewrite_metadata(directory, **{key: value})
    with pytest.raises(ValueError, match="metadata disagrees"):
        split.create_execution_split_dataset(directory, settings)
    assert not (settings.processed_dir / "execution_splits").exists()


@pytest.mark.parametrize("mutation", ["missing_metadata", "changed_feature_hash", "rehash_leaked_feature", "rehash_changed_label", "upstream_hash"])
def test_feature_source_and_lineage_tampering_is_rejected(bundle, mutation):
    directory, settings, _ = bundle
    if mutation == "missing_metadata":
        (directory / "features.json").unlink()
    elif mutation == "upstream_hash":
        rewrite_metadata(directory, input_hashes={})
    else:
        path = directory / "features.parquet"
        frame = pd.read_parquet(path)
        if mutation in ("changed_feature_hash", "rehash_leaked_feature"):
            frame[FEATURE_COLUMNS[0]] = frame.execution_target_return
        else:
            frame.loc[20, "execution_target_return"] += 0.01
        frame.to_parquet(path, index=False)
        if mutation != "changed_feature_hash":
            rewrite_metadata(directory, features_sha256=file_digest(path))
    with pytest.raises(ValueError):
        split.create_execution_split_dataset(directory, settings)
    assert not (settings.processed_dir / "execution_splits").exists()


@pytest.mark.parametrize("name", ["momentum_steps", "rolling_window", "horizon_steps", "latency_bars"])
def test_current_feature_and_execution_parameters_must_match(bundle, name):
    directory, settings, _ = bundle
    changed = replace(settings, backtest=replace(settings.backtest, latency_bars=2)) if name == "latency_bars" else replace(settings, **{name: getattr(settings, name) * 2})
    with pytest.raises(ValueError, match="disagree"):
        split.create_execution_split_dataset(directory, changed)


@pytest.mark.parametrize("writer", ["write_json", "write_parquet"])
def test_failed_publication_never_leaves_partial_partitions(bundle, monkeypatch, writer):
    directory, settings, _ = bundle

    def fail(*args, **kwargs):
        raise OSError("Split disk failure")

    monkeypatch.setattr(split, writer, fail)
    with pytest.raises(OSError):
        split.create_execution_split_dataset(directory, settings)
    assert not list((settings.processed_dir / "execution_splits").iterdir())


def test_upstream_mutation_during_publication_cancels_the_bundle(bundle, monkeypatch):
    directory, settings, _ = bundle
    original_write = split.write_json

    def write(*args, **kwargs):
        original_write(*args, **kwargs)
        with (directory / "features.parquet").open("ab") as stream:
            stream.write(b"changed")

    monkeypatch.setattr(split, "write_json", write)
    with pytest.raises(ValueError, match="changed"):
        split.create_execution_split_dataset(directory, settings)
    assert not list((settings.processed_dir / "execution_splits").iterdir())


def test_cli_and_long_windows_input_output_paths(bundle):
    directory, settings, config = bundle
    assert cli.main(["execution-split", str(directory), "--config", str(config)]) == 0
    assert cli.main(["execution-split", str(directory), "--config", str(config)]) == 1
    deep = replace(settings, processed_dir=native_path(settings.processed_dir / ("deep-folder-" * 16) / ("more-folders-" * 12)))
    deep_targets, _ = targets.create_execution_target_dataset(Path(json.loads((directory / "features.json").read_text())["target_metadata"]["source_path"]), deep)
    deep_features, _ = features.create_execution_feature_dataset(deep_targets, deep)
    assert len(str(deep_features)) > 300
    deep_splits, report = split.create_execution_split_dataset(deep_features, deep)
    assert len(str(deep_splits)) > 300
    assert len(pd.read_parquet(deep_splits / "train.parquet")) == 100
    assert report["partitions"]["validation"]["purged_rows"] == 8
