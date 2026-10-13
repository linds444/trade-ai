from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from local_ai_trader import cli
from local_ai_trader.data.storage import native_path
from local_ai_trader.decision import preview
from local_ai_trader.forecast import split, train
from local_ai_trader.models.train import file_digest
from local_ai_trader.walkforward.completed import load_completed_execution
from local_ai_trader.walkforward import run as runner
from test_execution_split import bundle as feature_bundle
from test_walkforward_run import make_plan, forbidden


@pytest.fixture
def return_run(feature_bundle):
    source, settings, _ = feature_bundle
    directory, _ = split.create_execution_split_dataset(source, settings)
    output, _ = train.train_return_baseline_experiment(directory, settings)
    return output, directory


@pytest.fixture
def prepared(make_plan, return_run):
    directory, config = make_plan()
    runner.run_walkforward(directory)
    return directory, config, return_run[0], return_run[1]


def test_preview_projects_one_causal_row_and_preserves_existing_outputs_without_refitting(prepared, monkeypatch):
    directory, config, return_run, _ = prepared
    old, _ = preview.create_decision_preview(directory, config)
    execution = directory / "execution"
    before = runner.file_manifest(execution)
    model_before = runner.file_manifest(return_run)
    original_forecasts = preview.checkpoint_return_forecasts
    calls = []

    def forecasts(checkpoint, frame, model):
        assert list(frame.columns) == list(preview.RETURN_INPUT_COLUMNS)
        assert len(frame) == 1
        assert not any(name.startswith(("target_", "execution_")) for name in frame.columns)
        result = original_forecasts(checkpoint, frame, model)
        calls.append(result[0])
        return result

    monkeypatch.setattr(preview, "checkpoint_return_forecasts", forecasts)
    monkeypatch.setattr(train, "fit_return_baselines", forbidden)
    monkeypatch.setattr("local_ai_trader.models.inference.predict_frozen_models", forbidden)
    for name in ("fit_family", "create_calibration", "evaluate_heldout"):
        monkeypatch.setattr(runner, name, forbidden)
    output, report = preview.create_decision_preview(directory, config, return_run, "mean")
    assert len(calls) == 1
    assert output.parent.name == "forecast_decision_previews"
    assert report["decision"]["action"] == "AVOID"
    assert report["decision"]["reason"] == "expected_return_does_not_cover_costs_and_margin"
    assert report["expected_gross_return_forecast"] == calls[0].expected_gross_return
    assert report["return_forecast_status"] == "saved_execution_aligned_point_forecast"
    assert report["classifier_inference_repeated"] is False
    assert report["return_inference_rows"] == 1
    assert report["orders_created"] is False
    assert report["model_or_forecast_fitted"] is False
    assert report["strategy_performance_evaluated"] is False
    assert report["return_model"]["reserved_return_test_data_opened"] is False
    assert report["return_model"]["inference_input"]["fold"] == 2
    assert report["return_model"]["forecast"]["prediction_time"] == report["prediction_time"]
    assert json.loads((output / "preview.json").read_text()) == report
    after = runner.file_manifest(execution)
    assert {name: digest for name, digest in after.items() if not name.startswith("forecast_decision_previews/")} == before
    assert runner.file_manifest(return_run) == model_before
    assert json.loads((old / "preview.json").read_text())["expected_gross_return_forecast"] is None
    saved = (output / "preview.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        preview.create_decision_preview(directory, config, return_run, "mean")
    assert (output / "preview.json").read_bytes() == saved


def test_reserved_return_test_and_full_return_sources_are_never_read_by_preview(prepared, monkeypatch):
    directory, config, return_run, return_splits = prepared
    experiment = json.loads((return_run / "experiment.json").read_text())
    forbidden_paths = {Path(name).resolve() for name in experiment["split_metadata"]["input_hashes"]}
    forbidden_paths.update((return_splits / name).resolve() for name in ("train.parquet", "validation.parquet", "test.parquet"))
    original_open, original_read = Path.open, pd.read_parquet

    def open_path(path, *args, **kwargs):
        assert path.resolve() not in forbidden_paths, f"Return source opened: {path}"
        return original_open(path, *args, **kwargs)

    def read(path, *args, **kwargs):
        assert Path(path).resolve() not in forbidden_paths, f"Return source read: {path}"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)
    monkeypatch.setattr(pd, "read_parquet", read)
    _, report = preview.create_decision_preview(directory, config, return_run)
    assert report["return_model"]["reserved_return_test_data_opened"] is False


@pytest.mark.parametrize("name", ["momentum_steps", "rolling_window", "horizon_steps", "latency_bars", "symbol"])
def test_incompatible_return_model_is_refused_before_inference(prepared, monkeypatch, name):
    directory, config, run, _ = prepared
    model = train.load_return_model_snapshot(run)
    if name == "symbol":
        model = replace(model, checkpoint={**model.checkpoint, "symbol": "ETH-USD"})
    elif name == "latency_bars":
        model = replace(model, settings=replace(model.settings, backtest=replace(model.settings.backtest, latency_bars=2)))
    else:
        model = replace(model, settings=replace(model.settings, **{name: getattr(model.settings, name) * 2}))
    monkeypatch.setattr(preview, "load_return_model_snapshot", lambda path: model)
    monkeypatch.setattr(preview, "checkpoint_return_forecasts", forbidden)
    with pytest.raises(ValueError, match="disagree"):
        preview.create_decision_preview(directory, config, run)
    assert not (directory / "execution/forecast_decision_previews").exists()


def test_historical_signal_cannot_use_a_return_model_with_labels_known_later(prepared, monkeypatch):
    directory, config, run, _ = prepared
    model = train.load_return_model_snapshot(run)
    snapshot = load_completed_execution(directory)
    chosen = pd.read_parquet(snapshot.output / "summary/xgboost_temperature_predictions.parquet")
    model = replace(model, checkpoint={**model.checkpoint, "training_last_label_known_at": chosen.available_at.iloc[-1].isoformat()})
    monkeypatch.setattr(preview, "load_return_model_snapshot", lambda path: model)
    with pytest.raises(ValueError, match="training label availability"):
        preview.create_decision_preview(directory, config, run)


def test_future_labels_and_unselected_rows_cannot_affect_the_projected_forecast(prepared, monkeypatch):
    directory, _, run, _ = prepared
    snapshot, model = load_completed_execution(directory), train.load_return_model_snapshot(run)
    chosen = pd.read_parquet(snapshot.output / "summary/xgboost_temperature_predictions.parquet")
    row = chosen.iloc[-1]
    packet = preview.ProbabilityPrediction(row.symbol, row.available_at.to_pydatetime(), tuple(float(row[name]) for name in preview.PROBABILITY_COLUMNS))
    before, _ = preview.return_preview_forecast(snapshot, model, chosen, packet, "ridge")
    original_read = pd.read_parquet
    target_path = snapshot.protocol.fold_directory(2) / "test.parquet"

    def read(path, *args, **kwargs):
        frame = original_read(path, *args, **kwargs)
        if Path(path) == target_path:
            frame["target_return"] = 1000.0
            frame["target_class"] = "flat"
            frame["target_available_at"] += pd.to_timedelta(30, unit="D")
            frame["future_profit_hint"] = 1000000.0
            frame.loc[frame.available_at < row.available_at, list(preview.FEATURE_COLUMNS)] *= 1000
        return frame

    monkeypatch.setattr(pd, "read_parquet", read)
    after, _ = preview.return_preview_forecast(snapshot, model, chosen, packet, "ridge")
    assert before == after


def test_missing_matching_causal_row_is_refused(prepared, monkeypatch):
    directory, _, run, _ = prepared
    snapshot, model = load_completed_execution(directory), train.load_return_model_snapshot(run)
    chosen = pd.read_parquet(snapshot.output / "summary/xgboost_temperature_predictions.parquet")
    row = chosen.iloc[-1]
    packet = preview.ProbabilityPrediction(row.symbol, row.available_at.to_pydatetime(), tuple(float(row[name]) for name in preview.PROBABILITY_COLUMNS))
    original_read = pd.read_parquet
    path = snapshot.protocol.fold_directory(2) / "test.parquet"

    def read(source, *args, **kwargs):
        frame = original_read(source, *args, **kwargs)
        return frame.loc[frame.available_at != row.available_at] if Path(source) == path else frame

    monkeypatch.setattr(pd, "read_parquet", read)
    with pytest.raises(ValueError, match="input row"):
        preview.return_preview_forecast(snapshot, model, chosen, packet, "ridge")


def test_buy_candidate_reports_pending_independent_risk_without_orders(prepared, monkeypatch):
    directory, config, run, _ = prepared
    original_decide = preview.decide_cost_aware

    def decide(*args, **kwargs):
        result = original_decide(*args, **kwargs)
        return replace(result, action="BUY", reason="entry_conditions_met_pending_risk", risk_check_required=True)

    monkeypatch.setattr(preview, "decide_cost_aware", decide)
    _, report = preview.create_decision_preview(directory, config, run)
    assert report["risk_status"] == "required_before_execution_not_requested"
    assert report["orders_created"] is False


def test_failed_write_or_model_mutation_prevents_publication(prepared, monkeypatch):
    directory, config, run, _ = prepared
    original_write = preview.write_json

    def write(*args, **kwargs):
        original_write(*args, **kwargs)
        with (run / "checkpoint.json").open("ab") as stream:
            stream.write(b"changed")

    monkeypatch.setattr(preview, "write_json", write)
    with pytest.raises(ValueError, match="changed"):
        preview.create_decision_preview(directory, config, run)
    parent = directory / "execution/forecast_decision_previews"
    assert parent.is_dir()
    assert not list(parent.iterdir())


def test_failed_nested_preview_write_leaves_no_partial_report(prepared, monkeypatch):
    directory, config, run, _ = prepared

    def fail(*args, **kwargs):
        raise OSError("Forecast preview disk failure")

    monkeypatch.setattr(preview, "write_json", fail)
    with pytest.raises(OSError, match="disk failure"):
        preview.create_decision_preview(directory, config, run)
    assert not list((directory / "execution/forecast_decision_previews").iterdir())


def test_forecast_preview_publishes_under_long_windows_paths(prepared, monkeypatch):
    directory, config, run, _ = prepared
    snapshot = load_completed_execution(directory)
    deep = native_path(snapshot.output.parent / ("long-preview-" * 15) / ("more-nested-" * 13) / "execution")
    shutil.copytree(snapshot.output, deep)
    deep_snapshot = replace(snapshot, output=deep)
    monkeypatch.setattr(preview, "load_completed_execution", lambda path: deep_snapshot)
    output, report = preview.create_decision_preview(directory, config, run)
    assert len(str(output)) > 300
    assert json.loads((output / "preview.json").read_text()) == report
    assert report["return_inference_rows"] == 1


@pytest.mark.parametrize("name", ["training_last_label_known_at", "holding_bars", "training_rows", "ridge_alpha"])
def test_rehashed_checkpoint_metadata_cannot_disagree_with_saved_training_provenance(return_run, name):
    run, _ = return_run
    path = run / "checkpoint.json"
    checkpoint = json.loads(path.read_text())
    checkpoint[name] = "2026-02-01T00:00:00+00:00" if name == "training_last_label_known_at" else checkpoint[name] * 2
    path.write_text(json.dumps(checkpoint))
    experiment = json.loads((run / "experiment.json").read_text())
    experiment["artifact_hashes"]["checkpoint.json"] = file_digest(path)
    (run / "experiment.json").write_text(json.dumps(experiment))
    with pytest.raises(ValueError, match="metadata disagrees"):
        train.load_return_model_snapshot(run)


def test_cli_optional_return_run_dispatch_preserves_default_behavior(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(preview, "create_decision_preview", lambda *args, **kwargs: calls.append((args, kwargs)))
    plan, config, run = tmp_path / "plan", tmp_path / "settings.toml", tmp_path / "return_run"
    assert cli.main(["decision-preview", str(plan), "--config", str(config)]) == 0
    assert calls[-1] == ((plan, config), {})
    assert cli.main(["decision-preview", str(plan), "--config", str(config), "--return-run", str(run)]) == 0
    assert calls[-1] == ((plan, config), {"return_run": run, "return_model": "ridge"})
    with pytest.raises(SystemExit):
        cli.main(["decision-preview", str(plan), "--expected-return", "0.03"])
