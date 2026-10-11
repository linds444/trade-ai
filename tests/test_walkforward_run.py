from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from local_ai_trader import cli
from local_ai_trader.features.build_features import FEATURE_COLUMNS
from local_ai_trader.models.baseline import CLASS_NAMES
from local_ai_trader.models.evaluate import evaluate_probabilities
from local_ai_trader.settings import load_settings, WalkForwardSettings
from local_ai_trader.walkforward.plan import create_walkforward_plan
from local_ai_trader.walkforward.protocol import load_protocol, restore_settings
from local_ai_trader.walkforward import run as runner
from local_ai_trader.walkforward import results

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def make_plan(tmp_path):
    def make(family="xgboost"):
        root = tmp_path / family
        config = root / "config/settings.toml"
        config.parent.mkdir(parents=True)
        config.write_text((ROOT / "config/settings.toml").read_text(), encoding="utf-8")
        settings = load_settings(config)
        settings = replace(
            settings, walkforward=WalkForwardSettings(1, 1, 1, 1),
            boosting=replace(settings.boosting, n_estimators=8, n_jobs=1),
            mlp=replace(settings.mlp, device="cpu", mixed_precision=False, epochs=2, hidden_sizes=(12, 6), cpu_threads=1),
            backtest=replace(settings.backtest, entry_probability=0.01, exit_probability=0.99, minimum_direction_margin=0),
        )
        times = pd.date_range("2026-01-01T00:00:00Z", periods=1440, freq="5min")
        angle = np.arange(len(times)) * 0.21
        prices = 100 * np.exp(0.006 * np.sin(angle))
        frame = pd.DataFrame({
            "timestamp": times, "available_at": times + pd.to_timedelta(300, unit="s"),
            "target_available_at": times + pd.to_timedelta(2100, unit="s"),
            "open": prices, "close": prices * 1.0001, "symbol": "BTC-USD", "exchange": "coinbase",
            "target_class": np.array(CLASS_NAMES)[np.arange(len(times)) % 3],
            "target_return": np.tile([0.002, -0.002, 0.0], len(times) // 3),
        })
        for index, name in enumerate(FEATURE_COLUMNS):
            frame[name] = np.sin(angle * (index + 1)) * 0.001
        path = root / "features.parquet"
        frame.to_parquet(path, index=False)
        path.with_suffix(".features.json").write_text(json.dumps({
            "feature_names": list(FEATURE_COLUMNS), "feature_rows": len(frame), "candle_seconds": 300,
            "momentum_steps": 6, "rolling_window": 12,
            "target_metadata": {"horizon_steps": 6, "flat_return_threshold": 0.001},
        }), encoding="utf-8")
        destination, _ = create_walkforward_plan(path, settings, family)
        assert len(load_protocol(destination).plan["folds"]) == 2
        return destination, config
    return make


def forbidden(*args, **kwargs):
    raise AssertionError("Completed bundles must not be fitted or evaluated again")


@pytest.mark.parametrize("family", ["xgboost", "mlp"])
def test_end_to_end_freezes_all_models_before_test_io_and_reconciles_reset_accounts(make_plan, monkeypatch, family):
    directory, config = make_plan(family)
    protocol = load_protocol(directory)
    config.write_text("invalid current config must never be read", encoding="utf-8")
    calibrated = []
    original_calibrate = runner.create_calibration
    original_read = pd.read_parquet
    original_digest = runner.file_digest
    original_simulate = results.simulate

    def calibrate(*args, **kwargs):
        answer = original_calibrate(*args, **kwargs)
        calibrated.append(args[0])
        return answer

    def checked_read(path, *args, **kwargs):
        if Path(path).name == "test.parquet":
            assert len(calibrated) == 4
        return original_read(path, *args, **kwargs)

    def checked_digest(path):
        if Path(path).name == "test.parquet":
            assert len(calibrated) == 4
        return original_digest(path)

    def checked_simulate(candles, *args, **kwargs):
        assert set(candles) == {"timestamp", "available_at", "open", "close", "symbol", "exchange"}
        return original_simulate(candles, *args, **kwargs)

    monkeypatch.setattr(runner, "create_calibration", calibrate)
    monkeypatch.setattr(pd, "read_parquet", checked_read)
    monkeypatch.setattr("local_ai_trader.walkforward.protocol.file_digest", checked_digest)
    monkeypatch.setattr(results, "simulate", checked_simulate)
    output, report = runner.run_walkforward(directory)
    assert report["folds"] == 2
    assert report["evaluation_rows"] == 564
    assert len(report["fold_metrics"]) == 12
    assert len(report["financial_summary"]) == 8
    assert report["all_models_calibrated_before_first_evaluation"] is True
    assert report["independent_sample_assumption"] is False
    state = runner.read_json(output / "state.json")
    assert state["complete"] is True
    paths = [output / entry["run"] for fold in state["folds"].values() for key, entry in fold.items() if key in ("baseline", "selected")]
    assert len(set(paths)) == 4
    for index in (1, 2):
        backtest = runner.read_json(output / f"fold_{index:02d}/backtest/backtest.json")
        for name, scores in backtest["metrics"].items():
            trades = original_read(output / f"fold_{index:02d}/backtest/{name}_trades.parquet")
            fills = original_read(output / f"fold_{index:02d}/backtest/{name}_fills.parquet")
            assert trades.net_pnl.sum() == pytest.approx(scores["final_equity"] - scores["initial_cash"])
            assert fills.fee.sum() == pytest.approx(scores["fees_paid"])
        for key in ("baseline", "selected"):
            cp = runner.read_json(output / state["folds"][str(index)][key]["run"] / "checkpoint.json")
            assert cp["training_metadata"]["input_hashes"]["train.parquet"] == protocol.plan["fold_file_hashes"][f"fold_{index:02d}/train.parquet"]
            if key == "baseline" or family == "mlp":
                train = original_read(directory / f"fold_{index:02d}/train.parquet")
                np.testing.assert_allclose(cp["scaler_mean"], train[list(FEATURE_COLUMNS)].mean().to_numpy(), atol=1e-15)
    for name, scores in report["pooled_prediction_metrics"].items():
        table = original_read(output / f"summary/{name}_predictions.parquet")
        expected = evaluate_probabilities(table.target_class, table[[f"p_{c}" for c in CLASS_NAMES]].to_numpy(), 10)
        assert scores["log_loss"] == pytest.approx(expected["log_loss"])
        net_returns = [runner.read_json(output / f"fold_{i:02d}/backtest/backtest.json")["metrics"][name]["net_return"] for i in (1, 2)]
        assert report["financial_summary"][name]["mean_fold_net_return"] == pytest.approx(np.mean(net_returns))
    fingerprints = runner.file_manifest(output)
    for function in ("fit_family", "create_calibration", "evaluate_heldout", "publish_fold_backtest", "publish_summary"):
        monkeypatch.setattr(runner, function, forbidden)
    _, resumed = runner.run_walkforward(directory, resume=True)
    assert resumed == report
    assert runner.file_manifest(output) == fingerprints
    with pytest.raises(ValueError, match="--resume"):
        runner.run_walkforward(directory)


@pytest.mark.parametrize("field", ["backtest", "symbols", "candle_seconds", "models_dir", "random_seed"])
def test_frozen_settings_are_complete_and_validated(make_plan, field):
    directory, _ = make_plan()
    values = runner.read_json(directory / "plan.json")["settings"]
    if field == "backtest":
        values[field].pop("fee_bps")
    else:
        values[field] = {"symbols": [], "candle_seconds": True, "models_dir": "relative", "random_seed": -1}[field]
    with pytest.raises(ValueError):
        restore_settings(values)


@pytest.mark.parametrize("artifact", ["train.parquet", "validation.parquet", "split.json"])
def test_input_mutation_fails_before_any_fitting(make_plan, monkeypatch, artifact):
    directory, _ = make_plan()
    with (directory / "fold_01" / artifact).open("ab") as stream:
        stream.write(b"changed")
    monkeypatch.setattr(runner, "fit_family", forbidden)
    with pytest.raises(ValueError, match="input changed"):
        runner.run_walkforward(directory)


def test_resume_recovers_atomic_models_calibration_and_evaluation_after_journal_write_failure(make_plan, monkeypatch):
    directory, _ = make_plan()
    original_write = runner.write_json
    failed = False

    def fail_after_test(path, value):
        nonlocal failed
        if Path(path).name == "state.json" and any(directory.rglob("test_evaluation/metrics.json")) and not failed:
            failed = True
            raise OSError("Interrupted journal update")
        return original_write(path, value)

    monkeypatch.setattr(runner, "write_json", fail_after_test)
    with pytest.raises(OSError, match="Interrupted"):
        runner.run_walkforward(directory)
    saved = next(directory.rglob("test_evaluation/metrics.json"))
    original_bytes = saved.read_bytes()
    original_evaluate = runner.evaluate_heldout

    def evaluate(path, *args, **kwargs):
        assert not (path / "test_evaluation").exists()
        return original_evaluate(path, *args, **kwargs)

    monkeypatch.setattr(runner, "write_json", original_write)
    monkeypatch.setattr(runner, "fit_family", forbidden)
    monkeypatch.setattr(runner, "create_calibration", forbidden)
    monkeypatch.setattr(runner, "evaluate_heldout", evaluate)
    _, report = runner.run_walkforward(directory, resume=True)
    assert report["evaluation_rows"] == 564
    assert saved.read_bytes() == original_bytes
    assert not (directory / "execution/running.lock").exists()


def test_resume_adopts_model_published_before_pointer_was_saved(make_plan, monkeypatch):
    directory, _ = make_plan()
    original_write = runner.write_json

    def fail(path, value):
        if Path(path).name == "state.json":
            raise OSError("Journal unavailable")
        return original_write(path, value)

    monkeypatch.setattr(runner, "write_json", fail)
    with pytest.raises(OSError):
        runner.run_walkforward(directory)
    saved_model = next(directory.rglob("checkpoint.json"))
    before = saved_model.read_bytes()
    original_fit = runner.fit_family

    def checked_fit(protocol, index, family, root):
        assert not (index == 1 and family == "baseline")
        return original_fit(protocol, index, family, root)

    monkeypatch.setattr(runner, "write_json", original_write)
    monkeypatch.setattr(runner, "fit_family", checked_fit)
    runner.run_walkforward(directory, resume=True)
    assert saved_model.read_bytes() == before


def test_mutated_test_during_fitting_is_rejected_before_evaluation(make_plan, monkeypatch):
    directory, _ = make_plan()
    original_calibrate = runner.create_calibration

    def calibrate(*args, **kwargs):
        answer = original_calibrate(*args, **kwargs)
        with (directory / "fold_01/test.parquet").open("ab") as stream:
            stream.write(b"changed")
        return answer

    monkeypatch.setattr(runner, "create_calibration", calibrate)
    monkeypatch.setattr(runner, "evaluate_heldout", forbidden)
    with pytest.raises(ValueError, match="test.parquet"):
        runner.run_walkforward(directory)
    assert not list(directory.rglob("test_evaluation"))


def test_active_lock_and_changed_plan_binding_block_resume(make_plan):
    directory, _ = make_plan()
    output = directory / "execution"
    output.mkdir()
    lock = output / "running.lock"
    lock.write_text("active", encoding="utf-8")
    with pytest.raises(ValueError, match="running.lock"):
        runner.run_walkforward(directory, resume=True)
    assert lock.read_text() == "active"
    lock.unlink()
    (output / "protocol.json").write_text(json.dumps({"schema_version": 1, "plan_sha256": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="different frozen protocol"):
        runner.run_walkforward(directory, resume=True)


@pytest.mark.parametrize("mutation", ["path_escape", "end_boundary", "missing_setting"])
def test_incompatible_plan_fails(make_plan, mutation):
    directory, _ = make_plan()
    plan = runner.read_json(directory / "plan.json")
    if mutation == "path_escape":
        plan["fold_file_hashes"]["../../outside"] = "wrong"
    elif mutation == "end_boundary":
        plan["folds"][0]["evaluation_end_exclusive"] = "2026-01-06T00:00:00Z"
    else:
        plan["settings"].pop("random_seed")
    (directory / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError):
        load_protocol(directory)


def test_cli_dispatch_has_no_current_config_option(make_plan, monkeypatch):
    directory, _ = make_plan()
    called = []
    monkeypatch.setattr(runner, "run_walkforward", lambda path, resume: called.append((path, resume)))
    assert cli.main(["walkforward-run", str(directory), "--resume"]) == 0
    assert called == [(directory, True)]
    with pytest.raises(SystemExit):
        cli.main(["walkforward-run", str(directory), "--config", "anything"])


def test_failed_backtest_publication_preserves_evaluation_and_resumes_without_refitting(make_plan, monkeypatch):
    directory, _ = make_plan()
    original_write = results.write_parquet

    def fail(*args, **kwargs):
        raise OSError("Ledger disk failure")

    monkeypatch.setattr(results, "write_parquet", fail)
    with pytest.raises(OSError, match="Ledger"):
        runner.run_walkforward(directory)
    fold = directory / "execution/fold_01"
    assert not (fold / "backtest").exists()
    assert not list(fold.glob(".backtest-staging-*"))
    saved = {path: path.read_bytes() for path in directory.rglob("test_evaluation/metrics.json")}
    assert len(saved) == 2
    original_evaluate = runner.evaluate_heldout

    def evaluate(path, *args, **kwargs):
        assert "fold_01" not in str(path)
        return original_evaluate(path, *args, **kwargs)

    monkeypatch.setattr(results, "write_parquet", original_write)
    monkeypatch.setattr(runner, "fit_family", forbidden)
    monkeypatch.setattr(runner, "create_calibration", forbidden)
    monkeypatch.setattr(runner, "evaluate_heldout", evaluate)
    runner.run_walkforward(directory, resume=True)
    assert all(path.read_bytes() == value for path, value in saved.items())


@pytest.mark.parametrize("artifact", ["native_weights", "calibration", "predictions", "backtest", "summary"])
def test_completed_artifact_mutation_rejects_resume(make_plan, monkeypatch, artifact):
    directory, _ = make_plan()
    output, _ = runner.run_walkforward(directory)
    paths = {
        "native_weights": list(directory.rglob("xgboost.json")),
        "calibration": list(directory.rglob("calibration/calibration.json")),
        "predictions": list(directory.rglob("test_evaluation/xgboost.parquet")),
        "backtest": list(directory.rglob("backtest/cash_equity.parquet")),
        "summary": [output / "summary/fold_metrics.parquet"],
    }
    with paths[artifact][0].open("ab") as stream:
        stream.write(b" ")  # JSON remains readable; recorded byte identity must still fail.
    for function in ("fit_family", "create_calibration", "evaluate_heldout"):
        monkeypatch.setattr(runner, function, forbidden)
    with pytest.raises(ValueError, match="changed"):
        runner.run_walkforward(directory, resume=True)
