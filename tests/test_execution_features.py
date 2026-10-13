from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from local_ai_trader import cli
from local_ai_trader.features.build_features import build_features, FEATURE_COLUMNS
from local_ai_trader.forecast import features, targets
from local_ai_trader.models.train import file_digest
from local_ai_trader.settings import BacktestSettings, load_settings
from test_execution_targets import candles

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bundle(tmp_path):
    config = tmp_path / "config/settings.toml"
    config.parent.mkdir()
    config.write_text((ROOT / "config/settings.toml").read_text(), encoding="utf-8")
    settings = load_settings(config)
    settings.processed_dir.mkdir(parents=True)
    path = settings.processed_dir / "BTC_candles.parquet"
    candles(80).to_parquet(path, index=False)
    directory, _ = targets.create_execution_target_dataset(path, settings)
    return directory, path, settings, config


def rewrite_metadata(directory, **updates):
    path = directory / "targets.json"
    metadata = json.loads(path.read_text())
    metadata.update(updates)
    path.write_text(json.dumps(metadata))


def test_same_proven_causal_features_and_exact_label_preservation():
    raw = candles(80)
    labelled = targets.build_execution_targets(raw, 300, 6, BacktestSettings())
    original = labelled.copy(deep=True)
    result = features.build_execution_features(labelled, 300, 6, 12)
    expected = build_features(raw, 300, 6, 12).iloc[:len(result)]
    assert len(result) == 60
    pd.testing.assert_frame_equal(result[list(targets.CANDLE_COLUMNS) + list(FEATURE_COLUMNS)], expected)
    pd.testing.assert_frame_equal(result[list(targets.FUTURE_COLUMNS)], labelled.iloc[12:][list(targets.FUTURE_COLUMNS)].reset_index(drop=True))
    pd.testing.assert_frame_equal(labelled, original)
    assert set(result.columns) == set(targets.CANDLE_COLUMNS + targets.FUTURE_COLUMNS + FEATURE_COLUMNS)


def test_every_future_label_column_is_excluded_from_feature_computation():
    labelled = targets.build_execution_targets(candles(80), 300, 6, BacktestSettings())
    before = features.build_execution_features(labelled, 300, 6, 12)
    changed = labelled.copy()
    for column in targets.FUTURE_COLUMNS:
        changed[column] = "future value must never reach a predictor"
    after = features.build_execution_features(changed, 300, 6, 12)
    pd.testing.assert_frame_equal(before[list(FEATURE_COLUMNS)], after[list(FEATURE_COLUMNS)])


def test_future_market_changes_cannot_change_earlier_model_inputs():
    raw = candles(80)
    changed = raw.copy()
    changed.loc[31:, ["open", "high", "low", "close"]] *= 5
    changed.loc[31:, "volume"] *= 100
    before = features.build_execution_features(targets.build_execution_targets(raw, 300, 6, BacktestSettings()), 300, 6, 12)
    after = features.build_execution_features(targets.build_execution_targets(changed, 300, 6, BacktestSettings()), 300, 6, 12)
    past = before.timestamp <= raw.timestamp.iloc[30]
    pd.testing.assert_frame_equal(before.loc[past, list(FEATURE_COLUMNS)], after.loc[past, list(FEATURE_COLUMNS)])
    # Some historical future labels change; past predictors still do not.
    assert not before.loc[past, "execution_target_return"].equals(after.loc[past, "execution_target_return"])


def test_truncating_future_data_keeps_complete_prefix_identical():
    raw = candles(80)
    full = features.build_execution_features(targets.build_execution_targets(raw, 300, 6, BacktestSettings()), 300, 6, 12)
    prefix = features.build_execution_features(targets.build_execution_targets(raw.iloc[:50], 300, 6, BacktestSettings()), 300, 6, 12)
    pd.testing.assert_frame_equal(full.iloc[:len(prefix)], prefix)


def test_verified_143_day_input_produces_41164_features_with_eight_bar_availability():
    labelled = targets.build_execution_targets(candles(41184), 300, 6, BacktestSettings())
    result = features.build_execution_features(labelled, 300, 6, 12)
    assert len(result) == 41164
    assert np.isfinite(result[list(FEATURE_COLUMNS)].to_numpy()).all()
    assert (result.execution_target_available_at - result.available_at).eq(pd.to_timedelta(2400, unit="s")).all()


@pytest.mark.parametrize("mutation", ["extra_numeric", "classifier_target", "missing_label", "existing_features", "duplicate_column"])
def test_unexpected_columns_cannot_expand_the_model_input_schema(mutation):
    labelled = targets.build_execution_targets(candles(80), 300, 6, BacktestSettings())
    if mutation == "extra_numeric":
        labelled["future_profit_hint"] = 1.0
    elif mutation == "classifier_target":
        labelled["target_return"] = 0.1
    elif mutation == "missing_label":
        labelled = labelled.drop(columns="execution_target_available_at")
    elif mutation == "existing_features":
        labelled["feature_return_1"] = 0.0
    else:
        labelled = pd.concat([labelled, labelled[["volume"]]], axis=1)
    with pytest.raises(ValueError, match="execution labels"):
        features.build_execution_features(labelled, 300, 6, 12)


def test_audited_bundle_is_unchanged_by_feature_publication(bundle):
    directory, source, settings, _ = bundle
    originals = {path: path.read_bytes() for path in (source, directory / "labels.parquet", directory / "targets.json")}
    labelled, metadata, inputs = targets.load_execution_target_bundle(directory)
    assert len(labelled) == 72
    assert metadata["labelled_rows"] == 72
    assert inputs == {path: file_digest(path) for path in originals}
    output, report = features.create_execution_feature_dataset(directory, settings)
    result = pd.read_parquet(output / "features.parquet")
    assert len(result) == 60
    assert report["feature_names"] == list(FEATURE_COLUMNS)
    assert report["future_only_columns"] == list(targets.FUTURE_COLUMNS)
    assert not set(report["feature_names"]).intersection(report["future_only_columns"])
    assert report["target_available_column"] == "execution_target_available_at"
    assert report["target_known_offset_from_prediction_bars"] == 8
    assert report["features_sha256"] == file_digest(output / "features.parquet")
    assert report["source_rows"] == 72
    assert report["warmup_rows_removed"] == 12
    assert report["feature_rows"] == 60
    assert report["models_fitted"] is False
    assert report["strategy_performance_evaluated"] is False
    assert json.loads((output / "features.json").read_text()) == report
    for path, original in originals.items():
        assert path.read_bytes() == original
    saved = (output / "features.parquet").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        features.create_execution_feature_dataset(directory, settings)
    assert (output / "features.parquet").read_bytes() == saved


@pytest.mark.parametrize("file", ["labels.parquet", "targets.json"])
def test_missing_target_artifacts_are_rejected(bundle, file):
    directory, _, settings, _ = bundle
    (directory / file).unlink()
    with pytest.raises(ValueError, match="requires"):
        features.create_execution_feature_dataset(directory, settings)


@pytest.mark.parametrize("file", ["original", "labels.parquet"])
def test_changed_candles_or_label_hashes_are_rejected(bundle, file):
    directory, source, settings, _ = bundle
    path = source if file == "original" else directory / file
    with path.open("ab") as stream:
        stream.write(b"mutation")
    with pytest.raises(ValueError, match="hash"):
        features.create_execution_feature_dataset(directory, settings)


@pytest.mark.parametrize("key,value", [
    ("target_kind", "close_to_close"), ("labelled_rows", 71), ("entry_row_offset", 1),
    ("target_known_offset_from_prediction_bars", 6), ("latency_bars", 2),
    ("regression_target_column", "execution_target_net_return"),
    ("future_only_columns", []), ("models_fitted", True), ("schema_version", True),
])
def test_inconsistent_metadata_cannot_change_target_contract(bundle, key, value):
    directory, _, settings, _ = bundle
    rewrite_metadata(directory, **{key: value})
    with pytest.raises(ValueError, match="metadata disagrees"):
        features.create_execution_feature_dataset(directory, settings)


@pytest.mark.parametrize("mutation", ["gross_return", "net_return", "known_at", "entry_open", "extra_feature"])
def test_semantic_audit_rejects_wrong_labels_even_with_updated_hash(bundle, mutation):
    directory, _, settings, _ = bundle
    path = directory / "labels.parquet"
    frame = pd.read_parquet(path)
    if mutation == "gross_return":
        frame.loc[20, "execution_target_return"] += 0.01
    elif mutation == "net_return":
        frame.loc[20, "execution_target_net_return"] += 0.01
    elif mutation == "known_at":
        frame["execution_target_available_at"] -= pd.to_timedelta(600, unit="s")
    elif mutation == "entry_open":
        frame.loc[20, "execution_entry_reference_open"] *= 2
    else:
        frame["feature_leaked_future_return"] = frame.execution_target_return
    frame.to_parquet(path, index=False)
    rewrite_metadata(directory, labels_sha256=file_digest(path))
    with pytest.raises(ValueError, match="original candle source"):
        features.create_execution_feature_dataset(directory, settings)


@pytest.mark.parametrize("name", ["candle_seconds", "horizon_steps", "latency_bars", "symbols"])
def test_current_timing_and_market_must_match_audited_targets(bundle, name):
    directory, _, settings, _ = bundle
    if name == "latency_bars":
        changed = replace(settings, backtest=replace(settings.backtest, latency_bars=2))
    elif name == "symbols":
        changed = replace(settings, symbols=("ETH-USD",))
    else:
        changed = replace(settings, **{name: getattr(settings, name) * 2})
    with pytest.raises(ValueError, match="disagree|not enabled"):
        features.create_execution_feature_dataset(directory, changed)


def test_new_current_costs_do_not_rewrite_frozen_diagnostic_labels(bundle):
    directory, _, settings, _ = bundle
    output, report = features.create_execution_feature_dataset(directory, replace(settings, backtest=replace(settings.backtest, fee_bps=5)))
    assert report["target_metadata"]["costs"]["fee_bps"] == 60
    labelled = pd.read_parquet(directory / "labels.parquet")
    result = pd.read_parquet(output / "features.parquet")
    pd.testing.assert_series_equal(result.execution_target_net_return, labelled.execution_target_net_return.iloc[12:].reset_index(drop=True))


@pytest.mark.parametrize("writer", ["write_json", "write_parquet"])
def test_failed_write_leaves_no_partial_feature_bundle(bundle, monkeypatch, writer):
    directory, _, settings, _ = bundle

    def fail(*args, **kwargs):
        raise OSError("Feature disk failure")

    monkeypatch.setattr(features, writer, fail)
    with pytest.raises(OSError):
        features.create_execution_feature_dataset(directory, settings)
    assert not list((settings.processed_dir / "execution_features").iterdir())


@pytest.mark.parametrize("file", ["original", "labels.parquet", "targets.json"])
def test_mutation_during_publication_prevents_a_feature_snapshot(bundle, monkeypatch, file):
    directory, source, settings, _ = bundle
    original_write = features.write_json

    def write(*args, **kwargs):
        original_write(*args, **kwargs)
        path = source if file == "original" else directory / file
        with path.open("ab") as stream:
            stream.write(b"changed")

    monkeypatch.setattr(features, "write_json", write)
    with pytest.raises(ValueError, match="changed"):
        features.create_execution_feature_dataset(directory, settings)
    assert not list((settings.processed_dir / "execution_features").iterdir())


def test_cli_supports_long_paths_without_registry_opt_in(bundle):
    _, _, settings, config = bundle
    # Force nested source/output paths beyond the Windows 260-character limit.
    # Native paths keep mkdir, Arrow, hashing and directory renames consistent.
    from local_ai_trader.data.storage import native_path

    root = native_path(settings.processed_dir / ("deep-folder-" * 16) / ("more-folders-" * 12))
    root.mkdir(parents=True)
    source = root / "long_candles.parquet"
    candles(80).to_parquet(source, index=False)
    long_settings = replace(settings, processed_dir=root)
    nested, _ = targets.create_execution_target_dataset(source, long_settings)
    assert len(str(nested)) > 300
    # The CLI output root remains the config's root; the audited inputs are long.
    assert cli.main(["execution-features", str(nested), "--config", str(config)]) == 0
    output = settings.processed_dir / "execution_features" / nested.name
    assert len(pd.read_parquet(native_path(output) / "features.parquet")) == 60
    # Exercise long output publication as well.
    deep_output, _ = features.create_execution_feature_dataset(nested, long_settings)
    assert len(str(deep_output)) > 300
    assert len(pd.read_parquet(deep_output / "features.parquet")) == 60
