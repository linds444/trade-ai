from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from local_ai_trader import cli
from local_ai_trader.data.dataset import create_target_dataset
from local_ai_trader.data.split import create_split_dataset
from local_ai_trader.features.build_features import FEATURE_COLUMNS, create_feature_dataset
from local_ai_trader.models import boosting, heldout
from local_ai_trader.models.boosting import fit_boosting, predict_boosting_checkpoint, train_boosting_experiment
from local_ai_trader.models.heldout import evaluate_heldout
from local_ai_trader.models.train import load_training_partitions
from local_ai_trader.settings import load_boosting_settings, load_settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dataset(tmp_path):
    config = tmp_path / "config/settings.toml"
    config.parent.mkdir()
    config.write_text((ROOT / "config/settings.toml").read_text(), encoding="utf-8")
    settings = load_settings(config)
    settings = replace(settings, boosting=replace(settings.boosting, n_estimators=12, n_jobs=1))
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
    return splits, settings


def test_only_training_features_and_labels_enter_fit(dataset, monkeypatch):
    splits, settings = dataset
    tables, _ = load_training_partitions(splits, settings)
    original_fit = XGBClassifier.fit
    fitted_rows = []

    def fit(model, inputs, labels, **kwargs):
        fitted_rows.append(len(inputs))
        assert list(inputs.columns) == list(FEATURE_COLUMNS)
        np.testing.assert_array_equal(inputs.to_numpy(), tables["train"][list(FEATURE_COLUMNS)].to_numpy())
        assert not kwargs  # No eval_set or early stopping on validation labels.
        return original_fit(model, inputs, labels, **kwargs)

    monkeypatch.setattr(XGBClassifier, "fit", fit)
    (splits / "test.parquet").write_bytes(b"Held-out data must not be read or hashed")
    run, metrics = train_boosting_experiment(splits, settings)
    assert fitted_rows == [len(tables["train"])]
    experiment = json.loads((run / "experiment.json").read_text())
    assert experiment["test_data_opened"] is False
    assert experiment["validation_used_for_fitting"] is False
    assert metrics["test"] == {"evaluated": False}
    assert set(experiment["input_hashes"]) == {"split.json", "train.parquet", "validation.parquet"}


def test_saved_predictions_round_trip_and_ignore_future_columns(dataset):
    splits, settings = dataset
    run, _ = train_boosting_experiment(splits, settings)
    checkpoint = json.loads((run / "checkpoint.json").read_text())
    frame = pd.read_parquet(splits / "validation.parquet")
    expected = pd.read_parquet(run / "validation_xgboost.parquet")[["p_up", "p_down", "p_flat"]].to_numpy()
    frame["target_class"] = "up"
    frame["target_return"] = 1e15
    frame["future_price"] = -1e15
    actual = predict_boosting_checkpoint(checkpoint, frame, run)["xgboost"]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(actual.sum(axis=1), 1, atol=1e-7)


def test_validation_changes_do_not_change_saved_weights(dataset):
    splits, settings = dataset
    first, _ = train_boosting_experiment(splits, settings)
    validation = pd.read_parquet(splits / "validation.parquet")
    validation.loc[:, list(FEATURE_COLUMNS)] += 100
    validation.to_parquet(splits / "validation.parquet", index=False)
    second, _ = train_boosting_experiment(splits, settings)
    assert (first / "xgboost.json").read_bytes() == (second / "xgboost.json").read_bytes()


def test_missing_training_class_has_clear_error(dataset):
    splits, settings = dataset
    frame = pd.read_parquet(splits / "train.parquet")
    frame = frame.loc[frame["target_class"] != "down"]
    with pytest.raises(ValueError, match="all three training classes"):
        fit_boosting(frame, settings)


@pytest.mark.parametrize("mutation", ["feature_order", "class_order", "preprocessing", "model_path", "model_bytes"])
def test_incompatible_checkpoint_or_model_is_rejected(dataset, mutation):
    splits, settings = dataset
    run, _ = train_boosting_experiment(splits, settings)
    checkpoint = deepcopy(json.loads((run / "checkpoint.json").read_text()))
    if mutation == "feature_order":
        checkpoint["feature_names"].reverse()
    elif mutation == "class_order":
        checkpoint["class_names"].reverse()
    elif mutation == "preprocessing":
        checkpoint["preprocessing"] = "refit_scaler"
    elif mutation == "model_path":
        checkpoint["model_file"] = "../other-model.json"
    else:
        with (run / "xgboost.json").open("ab") as source:
            source.write(b" ")
    with pytest.raises(ValueError):
        predict_boosting_checkpoint(checkpoint, pd.read_parquet(splits / "validation.parquet"), run)


def test_frozen_test_evaluation_never_fits_and_preserves_model(dataset, monkeypatch):
    splits, settings = dataset
    run, _ = train_boosting_experiment(splits, settings)
    before = {path.name: path.read_bytes() for path in run.iterdir()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen evaluation must not fit")

    monkeypatch.setattr(XGBClassifier, "fit", forbidden)
    monkeypatch.setattr(boosting, "fit_boosting", forbidden)
    output, report = evaluate_heldout(run)
    assert set(report["metrics"]) == {"xgboost"}
    assert report["refitted"] is False
    assert report["rows"] == len(pd.read_parquet(splits / "test.parquet"))
    assert report["model_artifact_hashes"]["xgboost.json"]
    assert (output / "xgboost.parquet").exists()
    for name, content in before.items():
        assert (run / name).read_bytes() == content
    with pytest.raises(ValueError, match="already saved"):
        evaluate_heldout(run)


def test_failed_write_leaves_no_published_experiment(dataset, monkeypatch):
    splits, settings = dataset

    def fail(*args, **kwargs):
        raise OSError("Simulated full disk")

    monkeypatch.setattr(boosting, "write_json", fail)
    with pytest.raises(OSError):
        train_boosting_experiment(splits, settings)
    assert not list(settings.models_dir.iterdir())


def test_model_change_during_test_scoring_prevents_publication(dataset, monkeypatch):
    splits, settings = dataset
    run, _ = train_boosting_experiment(splits, settings)
    original_evaluate = heldout.evaluate_probabilities

    def changed(*args, **kwargs):
        result = original_evaluate(*args, **kwargs)
        with (run / "xgboost.json").open("ab") as source:
            source.write(b" ")
        return result

    monkeypatch.setattr(heldout, "evaluate_probabilities", changed)
    with pytest.raises(ValueError, match="Model artifact changed"):
        evaluate_heldout(run)
    assert not (run / "test_evaluation").exists()


def test_input_change_during_fitting_is_rejected(dataset, monkeypatch):
    splits, settings = dataset
    original_fit = boosting.fit_boosting

    def changed(frame, settings):
        result = original_fit(frame, settings)
        with (splits / "train.parquet").open("ab") as source:
            source.write(b" ")
        return result

    monkeypatch.setattr(boosting, "fit_boosting", changed)
    with pytest.raises(ValueError, match="Training inputs changed"):
        train_boosting_experiment(splits, settings)
    assert not list(settings.models_dir.iterdir())


@pytest.mark.parametrize("name,value", [
    ("n_estimators", 0), ("max_depth", True), ("n_jobs", -1),
    ("learning_rate", 0), ("learning_rate", 1.1), ("learning_rate", float("nan")),
    ("min_child_weight", -1), ("reg_lambda", float("inf")),
])
def test_invalid_boosting_settings_fail(name, value):
    with pytest.raises(ValueError):
        load_boosting_settings({name: value})


def test_boosting_cli_creates_validation_experiment(dataset, capsys):
    splits, settings = dataset
    config = settings.models_dir.parent.parent / "config/settings.toml"
    assert cli.main(["boosting", str(splits), "--config", str(config)]) == 0
    assert "Validation metrics" in capsys.readouterr().out
