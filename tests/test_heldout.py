import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from local_ai_trader import cli
from local_ai_trader.data.dataset import create_target_dataset
from local_ai_trader.data.split import create_split_dataset
from local_ai_trader.features.build_features import create_feature_dataset
from local_ai_trader.models import heldout, train
from local_ai_trader.models.heldout import evaluate_heldout
from local_ai_trader.models.train import train_baseline_experiment
from local_ai_trader.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def experiment(tmp_path):
    config = tmp_path / "config/settings.toml"
    config.parent.mkdir()
    config.write_text((ROOT / "config/settings.toml").read_text(), encoding="utf-8")
    settings = load_settings(config)
    count = 300
    close = 100 * (1 + 0.006 * np.sin(np.arange(count) * 0.21))
    timestamps = pd.date_range("2025-01-01", periods=count, freq="5min", tz="UTC")
    frame = pd.DataFrame({
        "timestamp": timestamps, "available_at": timestamps + pd.to_timedelta(300, unit="s"),
        "open": close, "high": close * 1.001, "low": close * 0.999, "close": close,
        "volume": 10 + np.arange(count) % 7, "symbol": "BTC-USD", "exchange": "coinbase",
    })
    source = tmp_path / "candles.parquet"
    frame.to_parquet(source, index=False)
    targets = create_target_dataset(source, settings)[0]
    features = create_feature_dataset(targets, settings)[0]
    splits = create_split_dataset(features, settings)[0]
    run = train_baseline_experiment(splits, settings)[0]
    return run, splits, config


def test_evaluation_never_fits_and_preserves_original_experiment(experiment, monkeypatch, capsys):
    run, splits, config = experiment
    before = {name: (run / name).read_bytes() for name in ("checkpoint.json", "metrics.json", "experiment.json")}

    def forbidden(*args, **kwargs):
        raise AssertionError("Held-out evaluation must not fit")

    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    monkeypatch.setattr(train, "fit_baselines", forbidden)
    output, report = evaluate_heldout(run)
    assert report["partition"] == "test"
    assert report["refitted"] is False
    assert report["rows"] == len(pd.read_parquet(splits / "test.parquet"))
    for name, content in before.items():
        assert (run / name).read_bytes() == content
    assert json.loads((output / "metrics.json").read_text()) == report
    for model in ("naive", "logistic"):
        probabilities = pd.read_parquet(output / f"{model}.parquet")
        np.testing.assert_allclose(probabilities[["p_up", "p_down", "p_flat"]].sum(axis=1), 1)
        assert report["metrics"][model]["rows"] == len(probabilities)
    assert "Held-out test metrics" in capsys.readouterr().out


def test_only_test_partition_is_loaded(experiment, monkeypatch):
    run, splits, config = experiment
    original_read = pd.read_parquet
    loaded = []

    def read(path, *args, **kwargs):
        loaded.append(Path(path).name)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(heldout.pd, "read_parquet", read)
    evaluate_heldout(run)
    assert loaded == ["test.parquet"]


def test_current_settings_are_not_read(experiment):
    run, splits, config = experiment
    config.write_text("Deliberately invalid current configuration")
    assert cli.main(["evaluate", str(run)]) == 0


def test_repeat_evaluation_is_refused(experiment):
    run, _, _ = experiment
    evaluate_heldout(run)
    with pytest.raises(ValueError, match="already saved"):
        evaluate_heldout(run)


def test_split_location_override_requires_original_bundle(experiment, tmp_path):
    run, splits, _ = experiment
    moved = tmp_path / "relocated"
    shutil.copytree(splits, moved)
    shutil.rmtree(splits)
    output, report = evaluate_heldout(run, moved)
    assert report["split_directory"] == str(moved.resolve())


@pytest.mark.parametrize("filename", ["train.parquet", "validation.parquet", "split.json"])
def test_modified_training_inputs_are_rejected(experiment, filename):
    run, splits, _ = experiment
    with (splits / filename).open("ab") as source:
        source.write(b"changed")
    with pytest.raises(ValueError, match="differs from the frozen"):
        evaluate_heldout(run)
    assert not (run / "test_evaluation").exists()


@pytest.mark.parametrize("mutation", ["wrong_rows", "wrong_market", "wrong_availability", "bad_class", "nonfinite_feature", "reordered"])
def test_invalid_test_partition_is_rejected(experiment, mutation):
    run, splits, _ = experiment
    test_path = splits / "test.parquet"
    frame = pd.read_parquet(test_path)
    if mutation == "wrong_rows":
        frame = frame.iloc[:-1]
    elif mutation == "wrong_market":
        frame["symbol"] = "ETH-USD"
    elif mutation == "wrong_availability":
        frame.loc[0, "target_available_at"] += pd.to_timedelta(300, unit="s")
    elif mutation == "bad_class":
        frame.loc[0, "target_class"] = "BUY"
    elif mutation == "nonfinite_feature":
        frame.loc[0, "feature_momentum"] = np.inf
    else:
        frame = frame.iloc[::-1]
    frame.to_parquet(test_path, index=False)
    with pytest.raises(ValueError):
        evaluate_heldout(run)
    assert not (run / "test_evaluation").exists()


def test_failed_write_leaves_no_partial_test_evaluation(experiment, monkeypatch):
    run, _, _ = experiment

    def fail(*args, **kwargs):
        raise OSError("Simulated full disk")

    monkeypatch.setattr(heldout, "write_json", fail)
    with pytest.raises(OSError):
        evaluate_heldout(run)
    assert not (run / "test_evaluation").exists()
    assert not list(run.glob(".test-staging-*"))


def test_checkpoint_change_during_evaluation_is_rejected(experiment, monkeypatch):
    run, _, _ = experiment
    original_predict = heldout.predict_checkpoint

    def changed(checkpoint, frame):
        with (run / "checkpoint.json").open("ab") as source:
            source.write(b" ")
        return original_predict(checkpoint, frame)

    monkeypatch.setattr(heldout, "predict_checkpoint", changed)
    with pytest.raises(ValueError, match="Checkpoint changed"):
        evaluate_heldout(run)


def test_test_data_change_during_evaluation_is_rejected(experiment, monkeypatch):
    run, splits, _ = experiment
    original_predict = heldout.predict_checkpoint

    def changed(checkpoint, frame):
        with (splits / "test.parquet").open("ab") as source:
            source.write(b" ")
        return original_predict(checkpoint, frame)

    monkeypatch.setattr(heldout, "predict_checkpoint", changed)
    with pytest.raises(ValueError, match="Test dataset changed"):
        evaluate_heldout(run)
