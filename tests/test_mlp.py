from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from local_ai_trader import cli
from local_ai_trader.data.dataset import create_target_dataset
from local_ai_trader.data.split import create_split_dataset
from local_ai_trader.features.build_features import FEATURE_COLUMNS, create_feature_dataset
from local_ai_trader.models import heldout, mlp
from local_ai_trader.models.heldout import evaluate_heldout
from local_ai_trader.models.mlp import fit_mlp, predict_mlp_checkpoint, train_mlp_experiment
from local_ai_trader.models.train import load_training_partitions
from local_ai_trader.settings import load_mlp_settings, load_settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dataset(tmp_path):
    config = tmp_path / "config/settings.toml"
    config.parent.mkdir()
    config.write_text((ROOT / "config/settings.toml").read_text().replace('device = "cuda"', 'device = "cpu"').replace("epochs = 30", "epochs = 3"), encoding="utf-8")
    settings = load_settings(config)
    settings = replace(settings, mlp=replace(settings.mlp, hidden_sizes=(12, 6), batch_size=32, cpu_threads=1))
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
    return splits, settings, config


def test_normalization_uses_training_only_and_handles_constant_feature(dataset):
    splits, settings, _ = dataset
    tables, _ = load_training_partitions(splits, settings)
    frame = tables["train"].copy()
    frame[FEATURE_COLUMNS[0]] = 2.0
    frame[FEATURE_COLUMNS[1]] = 1 + (np.arange(len(frame)) % 2) * np.finfo(float).eps
    fitted = fit_mlp(frame, settings)
    np.testing.assert_allclose(fitted.mean, frame[list(FEATURE_COLUMNS)].mean())
    # Stable variance algorithms can differ at rounding scale on constant columns.
    np.testing.assert_allclose(fitted.variance, frame[list(FEATURE_COLUMNS)].var(ddof=0), atol=1e-27)
    assert fitted.scale[0] == 1
    assert fitted.scale[1] == 1  # Do not amplify rounding in an effectively constant feature.
    before = fitted.mean.copy(), fitted.scale.copy()
    validation = tables["validation"].copy()
    validation.loc[:, list(FEATURE_COLUMNS)] += 100
    fitted.predict(validation)
    np.testing.assert_array_equal(fitted.mean, before[0])
    np.testing.assert_array_equal(fitted.scale, before[1])


def test_fit_receives_only_training_rows_and_test_is_unread(dataset, monkeypatch):
    splits, settings, _ = dataset
    tables, _ = load_training_partitions(splits, settings)
    original_fit = mlp.fit_mlp
    fitted_rows = []

    def fit(frame, config):
        pd.testing.assert_frame_equal(frame, tables["train"])
        fitted_rows.append(len(frame))
        return original_fit(frame, config)

    monkeypatch.setattr(mlp, "fit_mlp", fit)
    (splits / "test.parquet").write_bytes(b"Test data must not be opened or hashed")
    run, metrics = train_mlp_experiment(splits, settings)
    experiment = json.loads((run / "experiment.json").read_text())
    assert fitted_rows == [len(tables["train"])]
    assert experiment["test_data_opened"] is False
    assert experiment["validation_used_for_fitting"] is False
    assert experiment["early_stopping"] is False
    assert experiment["hardware"]["mixed_precision"] is False
    assert len(experiment["training_history"]) == settings.mlp.epochs
    assert set(experiment["input_hashes"]) == {"split.json", "train.parquet", "validation.parquet"}
    assert metrics["test"] == {"evaluated": False}


def test_checkpoint_inference_matches_saved_predictions_and_ignores_targets(dataset):
    splits, settings, _ = dataset
    run, _ = train_mlp_experiment(splits, settings)
    checkpoint = json.loads((run / "checkpoint.json").read_text())
    validation = pd.read_parquet(splits / "validation.parquet")
    saved = pd.read_parquet(run / "validation_mlp.parquet")[["p_up", "p_down", "p_flat"]].to_numpy()
    validation["target_class"] = "BUY"
    validation["target_return"] = 1e15
    validation["future_price"] = -1e15
    for _ in range(2):  # Evaluation mode disables dropout on every restore.
        values = predict_mlp_checkpoint(checkpoint, validation, run)["mlp"]
        np.testing.assert_array_equal(values, saved)
        np.testing.assert_allclose(values.sum(axis=1), 1, atol=1e-12)


def test_validation_changes_cannot_change_weights_or_normalization(dataset):
    splits, settings, _ = dataset
    first, _ = train_mlp_experiment(splits, settings)
    validation = pd.read_parquet(splits / "validation.parquet")
    validation.loc[:, list(FEATURE_COLUMNS)] += 100
    validation.to_parquet(splits / "validation.parquet", index=False)
    second, _ = train_mlp_experiment(splits, settings)
    first_state = torch.load(first / "mlp_state.pt", weights_only=True)
    second_state = torch.load(second / "mlp_state.pt", weights_only=True)
    for name in first_state:
        torch.testing.assert_close(first_state[name], second_state[name], rtol=0, atol=0)
    before = json.loads((first / "checkpoint.json").read_text())
    after = json.loads((second / "checkpoint.json").read_text())
    assert before["scaler_mean"] == after["scaler_mean"]
    assert before["scaler_scale"] == after["scaler_scale"]


def test_missing_class_and_unavailable_cuda_have_clear_errors(dataset, monkeypatch):
    splits, settings, _ = dataset
    frame = pd.read_parquet(splits / "train.parquet")
    with pytest.raises(ValueError, match="all three training classes"):
        fit_mlp(frame.loc[frame["target_class"] != "down"], settings)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    settings = replace(settings, mlp=replace(settings.mlp, device="cuda"))
    with pytest.raises(ValueError, match="CUDA requested but unavailable"):
        fit_mlp(frame, settings)


@pytest.mark.parametrize("mutation", ["feature_order", "class_order", "bad_scale", "bad_mean", "bad_architecture", "model_path", "weight_bytes"])
def test_bad_checkpoint_or_changed_weights_are_rejected(dataset, mutation):
    splits, settings, _ = dataset
    run, _ = train_mlp_experiment(splits, settings)
    checkpoint = deepcopy(json.loads((run / "checkpoint.json").read_text()))
    if mutation == "feature_order":
        checkpoint["feature_names"].reverse()
    elif mutation == "class_order":
        checkpoint["class_names"].reverse()
    elif mutation == "bad_scale":
        checkpoint["scaler_scale"][0] = 0
    elif mutation == "bad_mean":
        checkpoint["scaler_mean"].pop()
    elif mutation == "bad_architecture":
        checkpoint["model_config"]["hidden_sizes"] = []
    elif mutation == "model_path":
        checkpoint["model_file"] = "../other-state.pt"
    else:
        with (run / "mlp_state.pt").open("ab") as source:
            source.write(b" ")
    with pytest.raises(ValueError):
        predict_mlp_checkpoint(checkpoint, pd.read_parquet(splits / "validation.parquet"), run)


def test_frozen_evaluation_does_not_fit_and_preserves_original_artifacts(dataset, monkeypatch):
    splits, settings, config = dataset
    run, _ = train_mlp_experiment(splits, settings)
    before = {path.name: path.read_bytes() for path in run.iterdir()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen evaluation must not optimize")

    monkeypatch.setattr(mlp, "fit_mlp", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    monkeypatch.setattr(mlp.StandardScaler, "fit", forbidden)
    config.write_text("Invalid current configuration")
    output, report = evaluate_heldout(run)
    assert report["refitted"] is False
    assert report["rows"] == len(pd.read_parquet(splits / "test.parquet"))
    assert set(report["metrics"]) == {"mlp"}
    assert report["model_artifact_hashes"]["mlp_state.pt"]
    assert (output / "mlp.parquet").exists()
    for name, content in before.items():
        assert (run / name).read_bytes() == content


def test_failed_write_cleans_staging_directory(dataset, monkeypatch):
    splits, settings, _ = dataset

    def fail(*args, **kwargs):
        raise OSError("Simulated full disk")

    monkeypatch.setattr(mlp, "write_json", fail)
    with pytest.raises(OSError):
        train_mlp_experiment(splits, settings)
    assert not list(settings.models_dir.iterdir())


def test_training_inputs_changed_during_fit_prevent_publication(dataset, monkeypatch):
    splits, settings, _ = dataset
    original_fit = mlp.fit_mlp

    def changed(frame, config):
        fitted = original_fit(frame, config)
        with (splits / "validation.parquet").open("ab") as source:
            source.write(b" ")
        return fitted

    monkeypatch.setattr(mlp, "fit_mlp", changed)
    with pytest.raises(ValueError, match="Training inputs changed"):
        train_mlp_experiment(splits, settings)
    assert not list(settings.models_dir.iterdir())


def test_weights_changed_during_test_scoring_prevent_publication(dataset, monkeypatch):
    splits, settings, _ = dataset
    run, _ = train_mlp_experiment(splits, settings)
    original_evaluate = heldout.evaluate_probabilities

    def changed(*args, **kwargs):
        result = original_evaluate(*args, **kwargs)
        with (run / "mlp_state.pt").open("ab") as source:
            source.write(b" ")
        return result

    monkeypatch.setattr(heldout, "evaluate_probabilities", changed)
    with pytest.raises(ValueError, match="Model artifact changed"):
        evaluate_heldout(run)
    assert not (run / "test_evaluation").exists()


@pytest.mark.parametrize("name,value", [
    ("hidden_sizes", []), ("hidden_sizes", [True]), ("hidden_sizes", [0]),
    ("dropout", 1), ("dropout", float("nan")), ("epochs", 0), ("batch_size", True),
    ("learning_rate", 0), ("weight_decay", -1), ("grad_clip_norm", 0),
    ("device", "auto"), ("mixed_precision", 1), ("cpu_threads", 0),
])
def test_invalid_mlp_config_is_rejected(name, value):
    with pytest.raises(ValueError):
        load_mlp_settings({name: value})


def test_mlp_cli_creates_experiment(dataset, capsys):
    splits, settings, config = dataset
    assert cli.main(["mlp", str(splits), "--config", str(config)]) == 0
    assert "Validation metrics" in capsys.readouterr().out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_cuda_training_amp_and_checkpoint_round_trip(dataset):
    splits, settings, _ = dataset
    settings = replace(settings, mlp=replace(settings.mlp, device="cuda", mixed_precision=True))
    run, metrics = train_mlp_experiment(splits, settings)
    experiment = json.loads((run / "experiment.json").read_text())
    assert experiment["hardware"]["training_device"] == "cuda"
    assert experiment["hardware"]["mixed_precision"] is True
    assert experiment["hardware"]["peak_allocated_bytes"] > 0
    checkpoint = json.loads((run / "checkpoint.json").read_text())
    validation = pd.read_parquet(splits / "validation.parquet")
    expected = pd.read_parquet(run / "validation_mlp.parquet")[["p_up", "p_down", "p_flat"]].to_numpy()
    actual = predict_mlp_checkpoint(checkpoint, validation, run, device="cpu")["mlp"]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
    assert metrics["test"] == {"evaluated": False}
