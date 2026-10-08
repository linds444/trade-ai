from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from local_ai_trader import cli
from local_ai_trader.data.dataset import create_target_dataset
from local_ai_trader.data.split import create_split_dataset
from local_ai_trader.features.build_features import FEATURE_COLUMNS, create_feature_dataset
from local_ai_trader.models import train as train_module
from local_ai_trader.models.baseline import CLASS_NAMES, encode_labels, feature_matrix, fit_baselines, predict_checkpoint
from local_ai_trader.models.evaluate import evaluate_probabilities
from local_ai_trader.models.train import load_training_partitions, train_baseline_experiment
from local_ai_trader.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path):
    config = tmp_path / "config/settings.toml"
    config.parent.mkdir()
    config.write_text((ROOT / "config/settings.toml").read_text(), encoding="utf-8")
    return load_settings(config)


def training_frame(count=90):
    generator = np.random.default_rng(42)
    result = pd.DataFrame(generator.normal(size=(count, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS))
    result["target_class"] = np.tile(CLASS_NAMES, count // 3 + 1)[:count]
    result["feature_return_1"] += encode_labels(result["target_class"])
    result["future_price"] = 1e12  # Explicitly excluded from the input allowlist.
    return result


def split_fixture(tmp_path, settings):
    count = 300
    close = 100 * (1 + 0.006 * np.sin(np.arange(count) * 0.21))
    timestamps = pd.date_range("2025-01-01", periods=count, freq="5min", tz="UTC")
    candles = pd.DataFrame({
        "timestamp": timestamps, "available_at": timestamps + pd.to_timedelta(300, unit="s"),
        "open": close, "high": close * 1.001, "low": close * 0.999, "close": close,
        "volume": 10 + np.arange(count) % 7, "symbol": "BTC-USD", "exchange": "coinbase",
    })
    source = tmp_path / "candles.parquet"
    candles.to_parquet(source, index=False)
    labelled = create_target_dataset(source, settings)[0]
    features = create_feature_dataset(labelled, settings)[0]
    return create_split_dataset(features, settings)[0]


def test_scaler_uses_training_statistics_only(settings):
    train = training_frame()
    fitted = fit_baselines(train, settings)
    scaler = fitted.logistic.named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, feature_matrix(train).mean(axis=0))
    before = fitted.checkpoint()
    validation = training_frame(30)
    validation.loc[:, list(FEATURE_COLUMNS)] += 1000
    fitted.predict(validation)
    assert fitted.checkpoint() == before
    assert before["training_rows"] == len(train)
    assert before["feature_names"] == list(FEATURE_COLUMNS)


def test_naive_probabilities_use_smoothed_training_counts(settings):
    train = training_frame(30)
    train["target_class"] = ["up"] * 20 + ["down"] * 8 + ["flat"] * 2
    fitted = fit_baselines(train, settings)
    np.testing.assert_allclose(fitted.naive_probabilities, np.array([21, 9, 3]) / 33)
    prediction = fitted.predict(training_frame(12))["naive"]
    assert prediction.shape == (12, 3)
    np.testing.assert_allclose(prediction, np.tile(np.array([21, 9, 3]) / 33, (12, 1)))


@pytest.mark.parametrize("binary", [False, True])
def test_json_checkpoint_inference_matches_sklearn(settings, binary):
    train = training_frame()
    if binary:
        train = train.loc[train["target_class"] != "down"]
    fitted = fit_baselines(train, settings)
    checkpoint = json.loads(json.dumps(fitted.checkpoint(), allow_nan=False))
    validation = training_frame(30)
    restored = predict_checkpoint(checkpoint, validation)
    expected = fitted.predict(validation)
    for name in expected:
        np.testing.assert_allclose(restored[name], expected[name], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(restored[name].sum(axis=1), 1)
    if binary:
        assert restored["logistic"][:, CLASS_NAMES.index("down")].tolist() == [0.0] * 30


def test_future_and_target_columns_do_not_enter_prediction(settings):
    fitted = fit_baselines(training_frame(), settings)
    validation = training_frame(30)
    before = fitted.predict(validation)
    validation["future_price"] = -1e15
    validation["target_class"] = "flat"
    validation["target_return"] = 1e15
    after = fitted.predict(validation)
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])


def test_fixed_settings_reproduce_predictions(settings):
    first = fit_baselines(training_frame(), settings)
    second = fit_baselines(training_frame(), settings)
    np.testing.assert_allclose(first.predict(training_frame(30))["logistic"], second.predict(training_frame(30))["logistic"])


def test_single_training_class_fails(settings):
    frame = training_frame()
    frame["target_class"] = "flat"
    with pytest.raises(ValueError, match="at least two"):
        fit_baselines(frame, settings)


@pytest.mark.parametrize("mutation", ["missing_feature", "nonfinite", "boolean", "bad_label"])
def test_bad_training_input_fails(settings, mutation):
    frame = training_frame()
    if mutation == "missing_feature":
        frame = frame.drop(columns=FEATURE_COLUMNS[0])
    elif mutation == "nonfinite":
        frame.loc[0, FEATURE_COLUMNS[0]] = np.nan
    elif mutation == "boolean":
        frame[FEATURE_COLUMNS[0]] = True
    else:
        frame.loc[0, "target_class"] = "BUY"
    with pytest.raises(ValueError):
        fit_baselines(frame, settings)


@pytest.mark.parametrize("mutation", ["feature_order", "class_order", "bad_scale", "bad_class_ids", "bad_shape", "nonfinite"])
def test_bad_checkpoint_rejected(settings, mutation):
    checkpoint = deepcopy(fit_baselines(training_frame(), settings).checkpoint())
    if mutation == "feature_order":
        checkpoint["feature_names"].reverse()
    elif mutation == "class_order":
        checkpoint["class_names"].reverse()
    elif mutation == "bad_scale":
        checkpoint["scaler_scale"][0] = 0
    elif mutation == "bad_class_ids":
        checkpoint["logistic_classes"] = [0, 0, 1]
    elif mutation == "bad_shape":
        checkpoint["scaler_mean"].pop()
    else:
        checkpoint["logistic_intercept"][0] = np.inf
    with pytest.raises(ValueError):
        predict_checkpoint(checkpoint, training_frame(30))


def test_perfect_probability_metrics():
    metrics = evaluate_probabilities(CLASS_NAMES, np.eye(3), 10)
    assert metrics["accuracy"] == metrics["f1_macro"] == 1
    assert metrics["brier_score"] == metrics["ece"] == 0
    assert metrics["log_loss"] == pytest.approx(0, abs=1e-12)
    assert metrics["roc_auc_ovr_macro"] == 1
    assert metrics["calibration_bins"][-1]["count"] == 3


def test_uniform_probability_metrics_have_known_values():
    metrics = evaluate_probabilities(CLASS_NAMES, np.full((3, 3), 1 / 3), 10)
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["log_loss"] == pytest.approx(np.log(3))
    assert metrics["brier_score"] == pytest.approx(2 / 3)
    assert metrics["ece"] == pytest.approx(0)


def test_ece_matches_empirical_bin_gap():
    probabilities = np.array([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]])
    metrics = evaluate_probabilities(["up", "down"], probabilities, 10)
    assert metrics["ece"] == pytest.approx(0.3)
    assert metrics["roc_auc_ovr_per_class"]["flat"] is None
    assert metrics["roc_auc_ovr_macro"] is None


@pytest.mark.parametrize("probabilities", [
    [[0.8, 0.1, 0.2]], [[1.1, -0.1, 0]], [[np.nan, 0, 1]], [[0.5, 0.5]],
])
def test_invalid_probability_distributions_fail(probabilities):
    with pytest.raises(ValueError):
        evaluate_probabilities(["up"], np.array(probabilities))


def test_experiment_does_not_open_test_file_and_saves_artifacts(tmp_path, settings, capsys):
    directory = split_fixture(tmp_path, settings)
    (directory / "test.parquet").write_bytes(b"Deliberately unreadable held-out data")
    output, metrics = train_baseline_experiment(directory, settings)
    experiment = json.loads((output / "experiment.json").read_text())
    checkpoint = json.loads((output / "checkpoint.json").read_text())
    assert experiment["test_data_opened"] is False
    assert metrics["test"] == {"evaluated": False}
    assert set(experiment["input_hashes"]) == {"split.json", "train.parquet", "validation.parquet"}
    assert set(path.name for path in output.iterdir()) == {"checkpoint.json", "experiment.json", "metrics.json", "validation_naive.parquet", "validation_logistic.parquet"}
    tables, _ = load_training_partitions(directory, settings)
    predicted = predict_checkpoint(checkpoint, tables["validation"])
    saved = pd.read_parquet(output / "validation_logistic.parquet")
    np.testing.assert_allclose(saved[[f"p_{name}" for name in CLASS_NAMES]], predicted["logistic"])
    assert len(saved) == metrics["validation"]["logistic"]["rows"]
    assert "Validation metrics" in capsys.readouterr().out


def test_future_validation_changes_cannot_change_fitted_weights(tmp_path, settings):
    directory = split_fixture(tmp_path, settings)
    tables, _ = load_training_partitions(directory, settings)
    before = fit_baselines(tables["train"], settings).checkpoint()
    validation = tables["validation"].copy()
    validation.loc[:, list(FEATURE_COLUMNS)] += 100
    validation.to_parquet(directory / "validation.parquet", index=False)
    changed, _ = load_training_partitions(directory, settings)
    assert fit_baselines(changed["train"], settings).checkpoint() == before


def test_failed_experiment_write_never_publishes_partial_run(tmp_path, settings, monkeypatch):
    directory = split_fixture(tmp_path, settings)

    def fail(*args, **kwargs):
        raise OSError("Simulated full disk")

    monkeypatch.setattr(train_module, "write_json", fail)
    with pytest.raises(OSError):
        train_baseline_experiment(directory, settings)
    assert not list(settings.models_dir.iterdir())


def test_bad_partition_metadata_fails(tmp_path, settings):
    directory = split_fixture(tmp_path, settings)
    path = directory / "split.json"
    report = json.loads(path.read_text())
    report["partitions"]["train"]["rows"] += 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="row count"):
        load_training_partitions(directory, settings)


def test_cli_baseline(tmp_path, settings):
    directory = split_fixture(tmp_path, settings)
    assert cli.main(["baseline", str(directory), "--config", str(tmp_path / "config/settings.toml")]) == 0


@pytest.mark.parametrize("key,value", [
    ("logistic_c", "0.0"), ("logistic_c", "nan"), ("naive_smoothing", "-1.0"),
    ("logistic_max_iter", "0"), ("random_seed", "true"), ("random_seed", "-1"), ("calibration_bins", "0"),
])
def test_invalid_baseline_config_fails(tmp_path, settings, key, value):
    config = tmp_path / "config/settings.toml"
    originals = {"logistic_c": "1.0", "naive_smoothing": "1.0", "logistic_max_iter": "1000", "random_seed": "42", "calibration_bins": "10"}
    config.write_text(config.read_text().replace(f"{key} = {originals[key]}", f"{key} = {value}"))
    with pytest.raises(ValueError, match=key):
        load_settings(config)
