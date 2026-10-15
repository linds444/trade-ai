from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from local_ai_trader import cli
from local_ai_trader.calibration import run as calibration_run
from local_ai_trader.calibration.run import create_calibration, validation_regions
from local_ai_trader.calibration.temperature import apply_temperature, fit_temperature
from local_ai_trader.data.dataset import create_target_dataset
from local_ai_trader.data.split import create_split_dataset
from local_ai_trader.features.build_features import create_feature_dataset
from local_ai_trader.models import heldout, inference
from local_ai_trader.models.boosting import train_boosting_experiment
from local_ai_trader.models.heldout import evaluate_heldout
from local_ai_trader.models.evaluate import evaluate_probabilities
from local_ai_trader.models.train import train_baseline_experiment
from local_ai_trader.settings import CalibrationSettings, load_calibration_settings, load_settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def experiment_factory(tmp_path):
    config = tmp_path / "config/settings.toml"
    config.parent.mkdir()
    config.write_text((ROOT / "config/settings.toml").read_text(), encoding="utf-8")
    settings = load_settings(config)
    settings = replace(settings, boosting=replace(settings.boosting, n_estimators=8, n_jobs=1), mlp=replace(settings.mlp, hidden_sizes=(12, 6), epochs=2, batch_size=32, device="cpu", cpu_threads=1))
    timestamps = pd.date_range("2025-01-01", periods=300, freq="5min", tz="UTC")
    close = 100 * (1 + 0.006 * np.sin(np.arange(300) * 0.21))
    candles = pd.DataFrame({
        "timestamp": timestamps, "available_at": timestamps + pd.to_timedelta(300, unit="s"),
        "open": close, "high": close * 1.001, "low": close * 0.999, "close": close,
        "volume": 10 + np.arange(300) % 7, "symbol": "BTC-USD", "exchange": "coinbase",
    })
    source = tmp_path / "candles.parquet"
    candles.to_parquet(source, index=False)
    targets = create_target_dataset(source, settings)[0]
    features = create_feature_dataset(targets, settings)[0]
    splits = create_split_dataset(features, settings)[0]

    def make(kind="baseline"):
        if kind == "baseline":
            run = train_baseline_experiment(splits, settings)[0]
        elif kind == "xgboost":
            run = train_boosting_experiment(splits, settings)[0]
        else:
            pytest.importorskip("torch")
            from local_ai_trader.models.mlp import train_mlp_experiment

            run = train_mlp_experiment(splits, settings)[0]
        return run, splits, settings, config

    return make


def test_temperature_recovers_known_empirical_distribution():
    probabilities = np.tile([0.8, 0.1, 0.1], (6, 1))
    result = fit_temperature(["up"] * 4 + ["down", "flat"], probabilities, CalibrationSettings())
    assert result["temperature"] == pytest.approx(1.5, abs=1e-5)
    adjusted = apply_temperature(probabilities, result["temperature"])
    np.testing.assert_allclose(adjusted, np.tile([2 / 3, 1 / 6, 1 / 6], (6, 1)), atol=1e-6)
    assert result["fit_objective"] < result["objective_at_identity"]


def test_uniform_probabilities_keep_identity_and_class_predictions():
    uniform = np.full((9, 3), 1 / 3)
    result = fit_temperature(["up", "down", "flat"] * 3, uniform, CalibrationSettings())
    assert result["temperature"] == 1
    probabilities = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.0, 0.2, 0.8]])
    for temperature in (0.25, 1.0, 4.0):
        adjusted = apply_temperature(probabilities, temperature)
        np.testing.assert_array_equal(adjusted.argmax(axis=1), probabilities.argmax(axis=1))
        np.testing.assert_allclose(adjusted.sum(axis=1), 1, atol=1e-12)


def test_optimizer_failure_is_not_published_as_a_temperature():
    probabilities = np.tile([0.8, 0.1, 0.1], (6, 1))
    with pytest.raises(ValueError, match="did not converge"):
        fit_temperature(["up"] * 4 + ["down", "flat"], probabilities, CalibrationSettings(max_iterations=1))


def test_classwise_reliability_measures_directional_event_frequencies():
    probabilities = np.tile([0.7, 0.2, 0.1], (10, 1))
    metrics = evaluate_probabilities(["up"] * 7 + ["down"] * 3, probabilities)
    up = metrics["classwise_calibration_bins"]["up"][7]
    assert up["count"] == 10
    assert up["mean_probability"] == pytest.approx(0.7)
    assert up["observed_frequency"] == pytest.approx(0.7)
    assert metrics["classwise_ece"]["up"] == pytest.approx(0)
    assert metrics["classwise_ece"]["down"] == pytest.approx(0.1)
    assert metrics["classwise_ece_macro"] == pytest.approx(0.2 / 3)
    assert metrics["classwise_calibration_bins"]["up"][0]["observed_frequency"] is None


@pytest.mark.parametrize("temperature", [0, -1, True, float("nan"), float("inf"), 1e-320])
def test_invalid_or_unrepresentable_temperatures_fail(temperature):
    with pytest.raises(ValueError):
        apply_temperature(np.array([[0.8, 0.1, 0.1]]), temperature)


@pytest.mark.parametrize("probabilities", [[[0.8, 0.1, 0.2]], [[-0.1, 0.3, 0.8]], [[float("nan"), 0, 1]], [[0.5, 0.5]]])
def test_invalid_probability_distributions_fail(probabilities):
    with pytest.raises(ValueError):
        apply_temperature(np.array(probabilities), 1)


def test_partition_purges_labels_at_the_boundary(experiment_factory):
    _, splits, settings, _ = experiment_factory()
    frame = pd.read_parquet(splits / "validation.parquet")
    regions, report = validation_regions(frame, 0.5)
    fit, assessment = regions["calibration_fit"], regions["assessment"]
    assert report["purged_fit_rows"] == settings.horizon_steps
    assert (fit["target_available_at"] < assessment["available_at"].iloc[0]).all()
    assert frame.iloc[len(frame) // 2 - settings.horizon_steps]["target_available_at"] == assessment["available_at"].iloc[0]
    assert len(assessment) == len(frame) - len(frame) // 2
    with pytest.raises(ValueError, match="no calibration-fit rows"):
        validation_regions(frame, 0.1)


def test_only_purged_earlier_validation_labels_fit_temperature(experiment_factory, monkeypatch):
    run, splits, settings, _ = experiment_factory()
    original_fit = calibration_run.fit_temperature
    frame = pd.read_parquet(splits / "validation.parquet")
    expected, _ = validation_regions(frame, 0.5)
    seen = []

    def fit(labels, probabilities, config):
        pd.testing.assert_series_equal(labels, expected["calibration_fit"]["target_class"])
        assert len(probabilities) == len(labels)
        seen.append(len(labels))
        return original_fit(labels, probabilities, config)

    monkeypatch.setattr(calibration_run, "fit_temperature", fit)
    original_read = pd.read_parquet
    original_digest = inference.file_digest
    loaded, hashed = [], []

    def read(path, *args, **kwargs):
        loaded.append(Path(path).name)
        return original_read(path, *args, **kwargs)

    def digest(path):
        hashed.append(Path(path).name)
        return original_digest(path)

    monkeypatch.setattr(pd, "read_parquet", read)
    monkeypatch.setattr(inference, "file_digest", digest)
    (splits / "test.parquet").write_bytes(b"Reserved test data must never be read or hashed")
    output, checkpoint = create_calibration(run, settings.calibration)
    assert loaded == ["validation.parquet"]
    assert "test.parquet" not in hashed
    assert len(seen) == 2
    assert checkpoint["test_data_opened"] is False
    assert checkpoint["model_weights_refitted"] is False
    assert (output / "metrics.json").exists()


@pytest.mark.parametrize("kind", ["baseline", "xgboost", "mlp"])
def test_calibration_and_frozen_evaluation_never_refit_prediction_models(experiment_factory, kind, monkeypatch):
    run, splits, settings, config = experiment_factory(kind)
    before = {path.name: path.read_bytes() for path in run.iterdir()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Saved model evaluation must not fit prediction weights or preprocessing")

    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    monkeypatch.setattr(XGBClassifier, "fit", forbidden)
    if kind == "mlp":
        import torch

        monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    calibration, checkpoint = create_calibration(run, settings.calibration)
    original_calibration = (calibration / "calibration.json").read_bytes()
    monkeypatch.setattr(calibration_run, "fit_temperature", forbidden)
    config.write_text("Invalid current settings")
    output, report = evaluate_heldout(run, calibrated=True)
    assert report["calibration_fitted_during_evaluation"] is False
    assert report["refitted"] is False
    assert report["calibration_checkpoint_sha256"]
    assert set(report["metrics"]) == set(checkpoint["models"]) | {f"{model}_temperature" for model in checkpoint["models"]}
    for model in checkpoint["models"]:
        assert report["metrics"][model]["accuracy"] == report["metrics"][f"{model}_temperature"]["accuracy"]
        assert report["metrics"][f"{model}_temperature"]["calibration_status"] == "temperature_scaled"
        assert (output / f"{model}_temperature.parquet").exists()
    for name, content in before.items():
        assert (run / name).read_bytes() == content
    assert (calibration / "calibration.json").read_bytes() == original_calibration
    with pytest.raises(ValueError, match="already saved"):
        evaluate_heldout(run, calibrated=False)


def test_assessment_changes_cannot_change_fitted_temperature(experiment_factory, monkeypatch):
    run, splits, settings, _ = experiment_factory()
    first, _ = create_calibration(run, settings.calibration)
    original = json.loads((first / "calibration.json").read_text())["models"]
    copied = run.parent / "copied-frozen-run"
    shutil.copytree(run, copied, ignore=shutil.ignore_patterns("calibration"))
    original_predict = calibration_run.predict_frozen_models

    def changed(checkpoint, frame, directory):
        values, artifacts = original_predict(checkpoint, frame, directory)
        cut = int(len(frame) * settings.calibration.fit_fraction)
        for model in values:
            values[model][cut:] = [0.98, 0.01, 0.01]
        return values, artifacts

    monkeypatch.setattr(calibration_run, "predict_frozen_models", changed)
    _, altered = create_calibration(copied, settings.calibration)
    assert altered["models"] == original


def test_missing_calibration_is_detected_before_loading_test_data(experiment_factory, monkeypatch):
    run, _, _, _ = experiment_factory()

    def forbidden(*args, **kwargs):
        raise AssertionError("Test data must remain unopened")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises(ValueError, match="Saved calibration required"):
        evaluate_heldout(run, calibrated=True)


def test_calibration_refuses_overwrite_and_post_test_fitting(experiment_factory):
    run, splits, settings, _ = experiment_factory()
    create_calibration(run, settings.calibration)
    with pytest.raises(ValueError, match="already saved"):
        create_calibration(run, settings.calibration)
    other, _, _, _ = experiment_factory()
    evaluate_heldout(other)
    with pytest.raises(ValueError, match="before test use"):
        create_calibration(other, settings.calibration)


def test_failed_calibration_write_leaves_no_partial_bundle(experiment_factory, monkeypatch):
    run, splits, settings, _ = experiment_factory()

    def fail(*args, **kwargs):
        raise OSError("Simulated full disk")

    monkeypatch.setattr(calibration_run, "write_json", fail)
    with pytest.raises(OSError):
        create_calibration(run, settings.calibration)
    assert not (run / "calibration").exists()
    assert not list(run.glob(".calibration-staging-*"))


def test_validation_changed_during_fitting_prevents_publication(experiment_factory, monkeypatch):
    run, splits, settings, _ = experiment_factory()
    original_fit = calibration_run.fit_temperature

    def changed(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        with (splits / "validation.parquet").open("ab") as source:
            source.write(b" ")
        return result

    monkeypatch.setattr(calibration_run, "fit_temperature", changed)
    with pytest.raises(ValueError, match="Validation dataset changed"):
        create_calibration(run, settings.calibration)
    assert not (run / "calibration").exists()


def test_calibration_changed_during_test_scoring_prevents_publication(experiment_factory, monkeypatch):
    run, splits, settings, _ = experiment_factory()
    create_calibration(run, settings.calibration)
    original_evaluate = heldout.evaluate_probabilities

    def changed(*args, **kwargs):
        result = original_evaluate(*args, **kwargs)
        with (run / "calibration/calibration.json").open("ab") as source:
            source.write(b" ")
        return result

    monkeypatch.setattr(heldout, "evaluate_probabilities", changed)
    with pytest.raises(ValueError, match="Calibration changed during evaluation"):
        evaluate_heldout(run, calibrated=True)
    assert not (run / "test_evaluation").exists()


@pytest.mark.parametrize("mutation", ["base_hash", "classes", "temperature", "model_keys", "overlap"])
def test_incompatible_calibration_is_rejected(experiment_factory, mutation):
    run, splits, settings, _ = experiment_factory()
    output, checkpoint = create_calibration(run, settings.calibration)
    if mutation == "base_hash":
        checkpoint["base_checkpoint_sha256"] = "wrong"
    elif mutation == "classes":
        checkpoint["class_names"].reverse()
    elif mutation == "temperature":
        checkpoint["models"]["naive"]["temperature"] = float("nan")
    elif mutation == "model_keys":
        checkpoint["models"].pop("logistic")
    else:
        checkpoint["regions"]["calibration_fit"]["last_label_known_at"] = checkpoint["regions"]["assessment"]["first_prediction_time"]
    (output / "calibration.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate_heldout(run, calibrated=True)
    assert not (run / "test_evaluation").exists()


@pytest.mark.parametrize("name,value", [
    ("fit_fraction", 1), ("min_temperature", 2), ("max_temperature", 0.5),
    ("min_temperature", 0), ("probability_floor", 0.1), ("optimizer_tolerance", float("nan")),
    ("max_iterations", 0),
])
def test_invalid_calibration_config_fails(name, value):
    with pytest.raises(ValueError):
        load_calibration_settings({name: value})


def test_calibration_cli_saves_assessment_without_test_evaluation(experiment_factory, capsys):
    run, splits, settings, config = experiment_factory()
    assert cli.main(["calibrate", str(run), "--config", str(config)]) == 0
    assert "Later-validation calibration assessment" in capsys.readouterr().out
    assert not (run / "test_evaluation").exists()
