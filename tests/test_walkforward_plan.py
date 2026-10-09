from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from local_ai_trader import cli
from local_ai_trader.features.build_features import FEATURE_COLUMNS
from local_ai_trader.models.baseline import CLASS_NAMES
from local_ai_trader.models.train import file_digest, load_training_partitions
from local_ai_trader.settings import WalkForwardSettings, load_settings, load_walkforward_settings
from local_ai_trader.walkforward import plan as planner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def features():
    timestamps = pd.date_range("2026-05-09T01:00:00Z", periods=41166, freq="5min")
    times = timestamps + pd.to_timedelta(300, unit="s")
    frame = pd.DataFrame({
        "timestamp": timestamps, "available_at": times, "open": 100.0, "close": 100.0,
        "symbol": "BTC-USD", "exchange": "coinbase",
        "target_available_at": times + pd.to_timedelta(1800, unit="s"),
        "target_class": np.array(CLASS_NAMES)[np.arange(len(times)) % 3],
        "target_return": np.tile([0.002, -0.002, 0.0], len(times) // 3),
    })
    for index, name in enumerate(FEATURE_COLUMNS):
        frame[name] = 0.001 * (index + 1) * np.sin(np.arange(len(times)))
    return frame


@pytest.fixture
def source(tmp_path, features):
    config = tmp_path / "config/settings.toml"
    config.parent.mkdir()
    config.write_text((ROOT / "config/settings.toml").read_text(), encoding="utf-8")
    settings = load_settings(config)
    directory = settings.processed_dir / "features"
    directory.mkdir(parents=True)
    path = directory / "BTC_features.parquet"
    features.to_parquet(path, index=False)
    metadata = {
        "schema_version": 1, "feature_names": list(FEATURE_COLUMNS), "feature_rows": len(features),
        "candle_seconds": 300, "momentum_steps": 6, "rolling_window": 12,
        "target_metadata": {"horizon_steps": 6, "flat_return_threshold": 0.001},
    }
    path.with_suffix(".features.json").write_text(json.dumps(metadata), encoding="utf-8")
    return path, settings, config


def test_default_protocol_creates_four_full_calendar_folds_with_all_boundaries_purged(features):
    folds = planner.build_folds(features, 300, 6, WalkForwardSettings())
    assert len(folds) == 4
    assert folds[0].report["train_start_inclusive"] == "2026-05-10T00:00:00+00:00"
    assert folds[0].report["test_first_prediction_time"] == "2026-07-23T00:00:00+00:00"
    assert folds[-1].report["evaluation_end_exclusive"] == "2026-09-17T00:00:00+00:00"
    for fold in folds:
        assert [len(fold.partitions[name]) for name in ("train", "validation", "test")] == [17274, 4026, 4026]
        boundaries = ("validation_first_prediction_time", "test_first_prediction_time", "evaluation_end_exclusive")
        for name, boundary_name in zip(("train", "validation", "test"), boundaries):
            boundary = pd.Timestamp(fold.report[boundary_name])
            selected = fold.partitions[name]
            assert (selected.target_available_at < boundary).all()
            assert selected.target_available_at.iloc[-1] == boundary - pd.to_timedelta(300, unit="s")
            assert fold.report["partitions"][name]["purged_rows"] == 6
            dropped = pd.to_datetime(fold.report["partitions"][name]["purged_prediction_times"], utc=True)
            assert dropped[0] + pd.to_timedelta(1800, unit="s") == boundary
    for previous, following in zip(folds, folds[1:]):
        assert previous.report["evaluation_end_exclusive"] == following.report["test_first_prediction_time"]
        assert previous.partitions["test"].target_available_at.iloc[-1] < following.partitions["test"].available_at.iloc[0]


def test_midnight_anchor_and_calendar_dates_do_not_depend_on_host_timezone(features):
    for name in ("timestamp", "available_at", "target_available_at"):
        features[name] = features[name].dt.tz_convert("America/Los_Angeles")
    folds = planner.build_folds(features, 300, 6, WalkForwardSettings())
    assert folds[0].report["train_start_inclusive"] == "2026-05-10T00:00:00+00:00"
    assert str(folds[0].partitions["train"].available_at.dt.tz) == "UTC"


def test_exact_midnight_is_kept_and_short_final_window_is_not_a_fold(features):
    midnight = features.loc[features.available_at >= pd.Timestamp("2026-05-10T00:00:00Z")]
    assert planner.build_folds(midnight, 300, 6, WalkForwardSettings())[0].report["train_start_inclusive"] == "2026-05-10T00:00:00+00:00"
    shortened = features.loc[features.available_at <= pd.Timestamp("2026-09-16T23:50:00Z")]
    assert len(planner.build_folds(shortened, 300, 6, WalkForwardSettings())) == 3


def test_rolling_training_can_use_previous_evaluation_history_but_not_future_outcomes(features):
    folds = planner.build_folds(features, 300, 6, WalkForwardSettings())
    earlier_test = folds[0].partitions["test"]
    latest_training = folds[-1].partitions["train"]
    assert latest_training.available_at.isin(earlier_test.available_at).any()
    for fold in folds:
        assert fold.partitions["train"].target_available_at.max() < fold.partitions["validation"].available_at.min()


def test_changes_after_a_fold_end_cannot_change_that_folds_data(features):
    original = planner.build_folds(features, 300, 6, WalkForwardSettings())[0]
    changed = features.copy()
    mask = changed.available_at >= pd.Timestamp(original.report["evaluation_end_exclusive"])
    changed.loc[mask, ["close", "target_return"]] = [500, 0.5]
    changed.loc[mask, "target_class"] = "up"
    altered = planner.build_folds(changed, 300, 6, WalkForwardSettings())[0]
    assert altered.report == original.report
    for name in original.partitions:
        pd.testing.assert_frame_equal(altered.partitions[name], original.partitions[name])


def test_larger_steps_leave_disjoint_evaluation_windows(features):
    folds = planner.build_folds(features, 300, 6, WalkForwardSettings(step_days=28))
    assert len(folds) == 2
    assert pd.Timestamp(folds[0].report["evaluation_end_exclusive"]) < pd.Timestamp(folds[1].report["test_first_prediction_time"])


@pytest.mark.parametrize("mutation", ["gap", "wrong_horizon", "invalid_feature", "invalid_label", "short_history"])
def test_invalid_source_or_insufficient_history_fails(features, mutation):
    if mutation == "gap":
        features = features.drop(1000)
    elif mutation == "wrong_horizon":
        features.loc[1000, "target_available_at"] += pd.to_timedelta(300, unit="s")
    elif mutation == "invalid_feature":
        features.loc[1000, FEATURE_COLUMNS[0]] = float("nan")
    elif mutation == "invalid_label":
        features.loc[1000, "target_class"] = "buy"
    else:
        features = features.iloc[:100]
    with pytest.raises(ValueError):
        planner.build_folds(features, 300, 6, WalkForwardSettings())


@pytest.mark.parametrize("name,value", [
    ("train_days", 0), ("validation_days", 1.5), ("evaluation_days", True),
    ("step_days", 13), ("step_days", 0),
])
def test_invalid_walkforward_settings_fail(name, value):
    with pytest.raises(ValueError):
        load_walkforward_settings({name: value})


def test_plan_records_frozen_choices_file_hashes_and_usable_training_metadata(source, monkeypatch):
    path, settings, _ = source

    def forbidden(*args, **kwargs):
        raise AssertionError("Planning must not fit or score prediction models")

    monkeypatch.setattr("local_ai_trader.models.train.fit_baselines", forbidden)
    destination, report = planner.create_walkforward_plan(path, settings, "mlp")
    assert report["models_fitted"] is False
    assert report["fold_performance_evaluated"] is False
    assert report["research_only"] is True
    assert report["model_family"] == "mlp"
    assert report["control_models"] == ["naive", "logistic"]
    assert report["settings"]["mlp"] == {**settings.mlp.__dict__, "hidden_sizes": settings.mlp.hidden_sizes}
    assert report["settings"]["backtest"] == settings.backtest.__dict__
    assert report["leading_rows_before_anchor"] == 275
    assert report["unused_tail_rows"] == 3451
    assert len(report["fold_file_hashes"]) == 16
    for filename, digest in report["fold_file_hashes"].items():
        assert file_digest(destination / filename) == digest
    assert report["source_hashes"]["features"] == file_digest(path)
    assert json.loads((destination / "plan.json").read_text())["settings"]["mlp"]["hidden_sizes"] == list(settings.mlp.hidden_sizes)
    actual_settings = replace(settings, train_fraction=report["settings"]["train_fraction"], validation_fraction=report["settings"]["validation_fraction"])
    original_read = pd.read_parquet

    def checked_read(path, *args, **kwargs):
        assert Path(path).name != "test.parquet"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", checked_read)
    tables, split_report = load_training_partitions(destination / "fold_01", actual_settings)
    assert len(tables["train"]) == 17274
    assert len(tables["validation"]) == 4026
    assert split_report["split_method"] == "rolling_walk_forward"


def test_plan_is_immutable_even_if_another_family_is_requested(source):
    path, settings, _ = source
    destination, _ = planner.create_walkforward_plan(path, settings, "mlp")
    original = (destination / "plan.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        planner.create_walkforward_plan(path, settings, "xgboost")
    assert (destination / "plan.json").read_bytes() == original


@pytest.mark.parametrize("writer", ["write_json", "write_parquet"])
def test_failed_writes_publish_no_partial_plan(source, monkeypatch, writer):
    path, settings, _ = source

    def fail(*args, **kwargs):
        raise OSError("Simulated full disk")

    monkeypatch.setattr(planner, writer, fail)
    with pytest.raises(OSError):
        planner.create_walkforward_plan(path, settings, "mlp")
    assert not list((settings.processed_dir / "walkforward").iterdir())


@pytest.mark.parametrize("changed_file", ["features", "metadata"])
def test_source_mutation_prevents_publication(source, monkeypatch, changed_file):
    path, settings, _ = source
    target = path if changed_file == "features" else path.with_suffix(".features.json")
    original_write = planner.write_json

    def changed(*args, **kwargs):
        original_write(*args, **kwargs)
        with target.open("ab") as stream:
            stream.write(b" ")

    monkeypatch.setattr(planner, "write_json", changed)
    with pytest.raises(ValueError, match="sources changed"):
        planner.create_walkforward_plan(path, settings, "mlp")
    assert not list((settings.processed_dir / "walkforward").iterdir())


@pytest.mark.parametrize("mutation", ["row_count", "horizon", "interval", "feature_names"])
def test_incompatible_feature_metadata_is_rejected(source, mutation):
    path, settings, _ = source
    metadata_path = path.with_suffix(".features.json")
    metadata = json.loads(metadata_path.read_text())
    if mutation == "row_count":
        metadata["feature_rows"] -= 1
    elif mutation == "horizon":
        metadata["target_metadata"]["horizon_steps"] = 1
    elif mutation == "interval":
        metadata["candle_seconds"] = 60
    else:
        metadata["feature_names"] = list(reversed(FEATURE_COLUMNS))
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        planner.create_walkforward_plan(path, settings, "mlp")
    assert not (settings.processed_dir / "walkforward").exists()


def test_planning_cli_publishes_a_research_protocol_only(source, capsys):
    path, settings, config = source
    assert cli.main(["walkforward-plan", str(path), "--model", "xgboost", "--config", str(config)]) == 0
    assert "no model fitting" in capsys.readouterr().out
    assert not settings.models_dir.exists()
