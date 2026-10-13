from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from local_ai_trader import cli
from local_ai_trader.data.storage import native_path
from local_ai_trader.decision.cost_aware import round_trip_factor
from local_ai_trader.forecast import baseline, split, train
from local_ai_trader.models.train import file_digest
from local_ai_trader.settings import load_backtest_settings
from test_execution_split import bundle as feature_bundle


@pytest.fixture
def splits(feature_bundle):
    directory, settings, config = feature_bundle
    output, _ = split.create_execution_split_dataset(directory, settings)
    return output, settings, config


def read_json(path):
    return json.loads(path.read_text())


def rewrite_manifest(directory, report):
    (directory / "split.json").write_text(json.dumps(report))


def rehash_partition(directory, name):
    report = read_json(directory / "split.json")
    report["partitions"][name]["sha256"] = file_digest(directory / f"{name}.parquet")
    rewrite_manifest(directory, report)


def test_training_reads_only_train_validation_and_manifest_even_without_test_or_upstream_sources(splits, monkeypatch):
    directory, settings, _ = splits
    (directory / "test.parquet").unlink()
    original_open, original_read, original_stat = Path.open, pd.read_parquet, Path.stat
    forbidden = {"test.parquet", "features.parquet", "features.json", "labels.parquet", "targets.json", "BTC_candles.parquet"}

    def open_path(path, *args, **kwargs):
        assert path.name not in forbidden, f"Forbidden data opened: {path}"
        return original_open(path, *args, **kwargs)

    def read(path, *args, **kwargs):
        assert Path(path).name not in forbidden, f"Forbidden data read: {path}"
        return original_read(path, *args, **kwargs)

    def stat(path, *args, **kwargs):
        assert path.name not in forbidden, f"Forbidden data inspected: {path}"
        return original_stat(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", open_path)
        patch.setattr(Path, "stat", stat)
        patch.setattr(pd, "read_parquet", read)
        output, metrics = train.train_return_baseline_experiment(directory, settings)
        checkpoint = train.load_return_checkpoint(output)
    experiment = read_json(output / "experiment.json")
    assert experiment["test_data_opened"] is False
    assert experiment["upstream_full_datasets_opened"] is False
    assert experiment["test_evaluated"] is False
    assert set(experiment["input_hashes"]) == {"split.json", "train.parquet", "validation.parquet"}
    assert metrics["test"] == {"evaluated": False}
    assert checkpoint["training_rows"] == 100


def test_saved_weights_scaler_metrics_predictions_and_artifacts_match_fitted_training_data(splits):
    directory, settings, _ = splits
    originals = {path: path.read_bytes() for path in directory.iterdir()}
    tables, _, _ = train.load_return_training_partitions(directory, settings)
    output, metrics = train.train_return_baseline_experiment(directory, settings)
    checkpoint = train.load_return_checkpoint(output)
    experiment = read_json(output / "experiment.json")
    assert set(path.name for path in output.iterdir()) == set(train.ARTIFACT_NAMES) | {"experiment.json"}
    assert read_json(output / "metrics.json") == metrics
    assert experiment["training_rows"] == 100
    assert experiment["validation_rows"] == 28
    assert experiment["strategy_performance_evaluated"] is False
    assert experiment["return_parameters"] == {"ridge_alpha": 1.0, "ridge_solver": "svd", "device": "cpu"}
    assert checkpoint["training_mean_return"] == pytest.approx(tables["train"].execution_target_return.mean())
    np.testing.assert_allclose(checkpoint["scaler_mean"], tables["train"][checkpoint["feature_names"]].mean())
    for model in baseline.MODEL_NAMES:
        result = pd.read_parquet(output / f"validation_{model}.parquet")
        predicted = baseline.predict_return_checkpoint(checkpoint, tables["validation"])[model]
        np.testing.assert_allclose(result.expected_gross_return, predicted)
        pd.testing.assert_series_equal(result.execution_target_return, tables["validation"].execution_target_return)
        assert result.forecast_valid.all()
        forecasts = baseline.checkpoint_return_forecasts(checkpoint, tables["validation"], model)
        assert len(forecasts) == 28
        assert metrics["validation"][model] == baseline.evaluate_return_predictions(result.execution_target_return, result.expected_gross_return)
    for name, digest in experiment["artifact_hashes"].items():
        assert digest == file_digest(output / name)
    assert {path: path.read_bytes() for path in directory.iterdir()} == originals


def test_extreme_validation_inputs_and_changed_outcomes_do_not_fit_scaler_or_weights(splits):
    directory, settings, _ = splits
    first, first_metrics = train.train_return_baseline_experiment(directory, settings)
    checkpoint_before = train.load_return_checkpoint(first)
    frame = pd.read_parquet(directory / "validation.parquet")
    frame[checkpoint_before["feature_names"]] *= 1000
    frame["execution_exit_reference_open"] = frame.execution_entry_reference_open * (1.2 + np.arange(len(frame)) * 0.001)
    frame["execution_target_return"] = frame.execution_exit_reference_open / frame.execution_entry_reference_open - 1
    costs = load_backtest_settings(read_json(directory / "split.json")["feature_metadata"]["target_metadata"]["costs"])
    frame["execution_target_net_return"] = (1 + frame.execution_target_return) * round_trip_factor(costs) - 1
    frame.to_parquet(directory / "validation.parquet", index=False)
    rehash_partition(directory, "validation")
    second, second_metrics = train.train_return_baseline_experiment(directory, settings)
    checkpoint_after = train.load_return_checkpoint(second)
    for key in ("training_mean_return", "training_rows", "scaler_mean", "scaler_scale", "scaler_variance", "ridge_coefficients", "ridge_intercept"):
        assert checkpoint_before[key] == checkpoint_after[key]
    assert first_metrics["validation"]["ridge"] != second_metrics["validation"]["ridge"]


@pytest.mark.parametrize("name", ["train", "validation"])
def test_changed_partition_hash_is_refused_before_fit(splits, monkeypatch, name):
    directory, settings, _ = splits
    with (directory / f"{name}.parquet").open("ab") as stream:
        stream.write(b"changed")

    def fit(*args, **kwargs):
        pytest.fail("Fitting must not run on unaudited input")

    monkeypatch.setattr(train, "fit_return_baselines", fit)
    with pytest.raises(ValueError, match="hash"):
        train.train_return_baseline_experiment(directory, settings)


@pytest.mark.parametrize("mutation", ["six_bar_purge", "wrong_basis", "leaked_feature_allowlist", "net_target", "wrong_rows", "wrong_boundary", "unknown_delay", "wrong_source_cost_delay"])
def test_invalid_split_metadata_cannot_reach_model_fitting(splits, mutation):
    directory, settings, _ = splits
    report = read_json(directory / "split.json")
    if mutation == "six_bar_purge":
        report["target_known_offset_from_prediction_bars"] = 6
    elif mutation == "wrong_basis":
        report["target_kind"] = "close_to_close"
    elif mutation == "leaked_feature_allowlist":
        report["feature_names"].append("execution_target_return")
    elif mutation == "net_target":
        report["regression_target_column"] = "execution_target_net_return"
    elif mutation == "wrong_rows":
        report["partitions"]["train"]["rows"] += 1
    elif mutation == "wrong_boundary":
        report["validation_first_prediction_time"] = report["test_first_prediction_time"]
    elif mutation == "unknown_delay":
        report["feature_metadata"]["target_metadata"]["availability_assumption"] = "exit_open_immediate"
    else:
        report["feature_metadata"]["target_metadata"]["costs"]["latency_bars"] = 2
    rewrite_manifest(directory, report)
    with pytest.raises(ValueError):
        train.train_return_baseline_experiment(directory, settings)


@pytest.mark.parametrize("mutation", ["six_bar_known", "bad_reference_return", "bad_net_return", "wrong_market", "nonfinite_feature"])
def test_semantic_partition_errors_are_refused_even_if_file_hash_is_updated(splits, mutation):
    directory, settings, _ = splits
    path = directory / "train.parquet"
    frame = pd.read_parquet(path)
    if mutation == "six_bar_known":
        frame["execution_target_available_at"] -= pd.to_timedelta(600, unit="s")
    elif mutation == "bad_reference_return":
        frame.loc[10, "execution_target_return"] += 0.01
    elif mutation == "bad_net_return":
        frame.loc[10, "execution_target_net_return"] += 0.01
    elif mutation == "wrong_market":
        frame["symbol"] = "ETH-USD"
    else:
        frame.loc[10, "feature_return_1"] = np.inf
    frame.to_parquet(path, index=False)
    rehash_partition(directory, "train")
    with pytest.raises(ValueError):
        train.train_return_baseline_experiment(directory, settings)


@pytest.mark.parametrize("name", ["horizon_steps", "momentum_steps", "rolling_window", "train_fraction", "latency_bars"])
def test_current_settings_must_match_frozen_return_protocol(splits, name):
    directory, settings, _ = splits
    changed = replace(settings, backtest=replace(settings.backtest, latency_bars=2)) if name == "latency_bars" else replace(settings, **{name: getattr(settings, name) * (0.9 if name == "train_fraction" else 2)})
    with pytest.raises(ValueError, match="disagree"):
        train.train_return_baseline_experiment(directory, changed)


@pytest.mark.parametrize("writer", ["write_json", "write_parquet"])
def test_write_failure_leaves_no_partial_experiment(splits, monkeypatch, writer):
    directory, settings, _ = splits

    def fail(*args, **kwargs):
        raise OSError("Return disk failure")

    monkeypatch.setattr(train, writer, fail)
    with pytest.raises(OSError):
        train.train_return_baseline_experiment(directory, settings)
    assert not list(native_path(settings.models_dir).iterdir())


def test_training_input_mutation_during_publication_cancels_the_experiment(splits, monkeypatch):
    directory, settings, _ = splits
    original_write = train.write_json

    def write(*args, **kwargs):
        original_write(*args, **kwargs)
        with (directory / "train.parquet").open("ab") as stream:
            stream.write(b"changed")

    monkeypatch.setattr(train, "write_json", write)
    with pytest.raises(ValueError, match="changed"):
        train.train_return_baseline_experiment(directory, settings)
    assert not list(native_path(settings.models_dir).iterdir())


@pytest.mark.parametrize("artifact", list(train.ARTIFACT_NAMES))
def test_saved_artifact_mutation_is_refused_by_checkpoint_loader(splits, artifact):
    directory, settings, _ = splits
    output, _ = train.train_return_baseline_experiment(directory, settings)
    with (output / artifact).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        train.load_return_checkpoint(output)


def test_cli_and_long_windows_paths_for_training_and_json_inference(splits):
    directory, settings, config = splits
    assert cli.main(["return-baseline", str(directory), "--config", str(config)]) == 0
    models_root = native_path(settings.models_dir)
    existing = list(models_root.iterdir())
    assert len(existing) == 1
    original_manifest = (existing[0] / "experiment.json").read_bytes()
    deep = native_path(settings.processed_dir / ("deep-folder-" * 16) / ("more-folders-" * 12) / directory.name)
    deep.mkdir(parents=True)
    for name in ("split.json", "train.parquet", "validation.parquet"):
        (deep / name).write_bytes((directory / name).read_bytes())
    # Test is intentionally absent even at the long source path.
    deep_settings = replace(settings, models_dir=native_path(settings.models_dir / ("deep-model-" * 16) / ("more-folders-" * 12)))
    output, _ = train.train_return_baseline_experiment(deep, deep_settings)
    assert len(str(output)) > 300
    checkpoint = train.load_return_checkpoint(output)
    tables, _, _ = train.load_return_training_partitions(deep, deep_settings)
    assert len(baseline.checkpoint_return_forecasts(checkpoint, tables["validation"])) == 28
    assert existing[0] in list(models_root.iterdir())
    assert (existing[0] / "experiment.json").read_bytes() == original_manifest
