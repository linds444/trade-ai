from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from local_ai_trader.cli import main
from local_ai_trader.data.storage import native_path, write_json, write_parquet
from local_ai_trader.hyperliquid import feed, history, train
from local_ai_trader.models.train import file_digest
from local_ai_trader.risk.entry_policy import SignalEvidence, evaluate_entry_signal
from local_ai_trader.settings import BoostingSettings, MLPSettings

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 1, 5, 0, 5, 10, tzinfo=timezone.utc)
FAST = train.TrainingRecipe(boosting=BoostingSettings(n_estimators=8, n_jobs=1), mlp=MLPSettings(epochs=2, hidden_sizes=(8,), cpu_threads=1))


def make_history(root, monkeypatch, *, flat=False, coins=("BTC", "ETH"), gap=False):
    config = native_path(root / "config/hyperliquid.toml")
    config.parent.mkdir(parents=True)
    text = (ROOT / "config/hyperliquid.toml").read_text(encoding="utf-8")
    roster = 'coins = ["xyz:HOOD", "xyz:NVDA", "SOL", "ETH", "xyz:TSLA", "BTC", "kPEPE", "DOGE"]'
    config.write_text(text.replace(roster, "coins = " + json.dumps(list(coins))), encoding="utf-8")
    settings = feed.load_hyperliquid_settings(config)
    client = feed.HyperliquidPublicClient(settings, clock=lambda: NOW)
    def info(request):
        if request["type"] == "metaAndAssetCtxs":
            group = [coin for coin in coins if feed.BY_COIN[coin].dex == request.get("dex", "")]
            return [{"collateralToken": 0, "universe": [{"name": coin, "szDecimals": 4, "maxLeverage": 20,
                    **({"deployerFeeScale": "1", "growthMode": "enabled"} if ":" in coin else {})} for coin in group]},
                    [{"markPx": "100", "oraclePx": "100", "funding": "0.00001"} for _ in group]]
        req = request["req"]
        values = []
        for index, stamp in enumerate(range(req["startTime"], req["endTime"], 300000)):
            if gap and index == 200:
                continue
            price = 100 if flat is True else (100+0.5*(index//12) if flat == "two" else 100+0.5*np.sin(index/3))
            values.append({"s": req["coin"], "i": "5m", "t": stamp, "T": stamp+299999,
                           "o": str(price), "c": str(price+0.01), "h": str(price+0.1), "l": str(price-0.1), "v": "10", "n": 3})
        return values
    monkeypatch.setattr(client, "info", info)
    output, _ = history.create_history_bundle(client, config, bars=500)
    return output


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    return make_history(tmp_path, monkeypatch)


def test_all_market_training_keeps_test_and_full_sources_entirely_unopened(prepared, monkeypatch):
    original_open, original_stat = Path.open, Path.stat
    forbidden = {"test.parquet", "raw.json", "candles.parquet", "features.parquet"}
    def guarded_open(path, *args, **kwargs):
        if path.name in forbidden:
            pytest.fail(f"Training attempted to open {path.name}")
        return original_open(path, *args, **kwargs)
    def guarded_stat(path, *args, **kwargs):
        if path.name in forbidden:
            pytest.fail(f"Training attempted to stat {path.name}")
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    output, report = train.train_hyperliquid_models(prepared, device="cpu", recipe=FAST)
    assert output.parent == prepared.parent.parent / "models"
    assert "research/data/hyperliquid" not in str(output)
    assert report["test_opened"] is False and report["test_hashed"] is False
    for coin in ("BTC", "ETH"):
        checkpoint, _, source_id = train.load_saved_market(output, coin)
        assert set(checkpoint["classifiers"]) == set(train.MODEL_NAMES)
        assert checkpoint["history_manifest_sha256"] == file_digest(prepared / "history.json")
        assert checkpoint["confidence"] is None
        assert source_id.startswith("sha256:")
        assert report["markets"][coin]["calibration"]["purged_rows"] == 8
        assert report["markets"][coin]["metrics"]["test"]["evaluated"] is False


def test_calibration_and_base_models_freeze_before_later_validation_predictions(prepared, monkeypatch):
    original = train.predict_classifiers
    seen = []
    def observe(checkpoint, frame, root):
        temperatures = bool(checkpoint["temperatures"])
        if temperatures:
            assert (root / "checkpoint.json").is_file()
            assert set(checkpoint["temperatures"]) == set(checkpoint["classifiers"])
            assert frame.available_at.min() > pd.Timestamp(checkpoint["calibration_last_label_known_at"])
        else:
            assert not (root / "checkpoint.json").exists()
        seen.append(temperatures)
        return original(checkpoint, frame, root)
    monkeypatch.setattr(train, "predict_classifiers", observe)
    _, report = train.train_hyperliquid_models(prepared, device="cpu", recipe=FAST)
    assert seen == [False, True, False, True]
    assert report["markets"]["BTC"]["primary_model"] == "mlp"
    assert report["markets"]["ETH"]["primary_model"] == "xgboost"


def test_saved_inference_reproduces_metrics_and_never_refits(prepared, monkeypatch):
    output, report = train.train_hyperliquid_models(prepared, device="cpu", recipe=FAST)
    tables, _, _, _ = train.load_training_inputs(prepared)
    _, assessment, _ = train.calibration_partitions(tables["BTC"]["validation"], FAST)
    def no_fit(*args, **kwargs):
        pytest.fail("Saved inference must not refit")
    monkeypatch.setattr(train, "fit_baselines", no_fit)
    monkeypatch.setattr(train, "fit_boosting", no_fit)
    monkeypatch.setattr(train.Ridge, "fit", no_fit)
    result = train.predict_saved_market(output, "BTC", assessment)
    stored = pd.read_parquet(output / "BTC/assessment_predictions.parquet")
    for name, values in result["probabilities"].items():
        np.testing.assert_allclose(values, stored[[f"{name}_temperature_p_{label}" for label in train.CLASS_NAMES]].to_numpy(), rtol=1e-10, atol=1e-12)
    for name, values in result["return_forecasts"].items():
        np.testing.assert_allclose(values, stored[f"return_{name}"], rtol=1e-10, atol=1e-12)
    assert result["confidence"] is None and result["reliability_validation_id"] is None
    gate = evaluate_entry_signal(SignalEvidence("BUY", 0.9, result["calibration_id"], result["confidence"], result["reliability_validation_id"]), 0)
    assert gate.status == "REVIEW" and gate.reason == "validated_confidence_unavailable"
    assert not report["forward_strategy_started"]


def test_scaling_and_mean_return_use_training_only(prepared):
    tables, source, _, _ = train.load_training_inputs(prepared)
    fitted = train._return_checkpoint(tables["BTC"]["train"], 1)
    np.testing.assert_allclose(fitted["scaler_mean"], train.feature_matrix(tables["BTC"]["train"]).mean(axis=0), atol=1e-12)
    assert fitted["training_mean"] == pytest.approx(tables["BTC"]["train"].execution_target_return.mean())
    old = deepcopy(fitted)
    tables["BTC"]["validation"]["execution_target_return"] = 1e9
    assert train._return_checkpoint(tables["BTC"]["train"], 1) == old


def test_source_train_or_validation_tampering_is_refused(prepared):
    path = prepared / "BTC/train.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "feature_return_1"] = 1
    write_parquet(path, frame)
    with pytest.raises(ValueError, match="input changed"):
        train.load_training_inputs(prepared)


@pytest.mark.parametrize("change", [
    lambda frame: frame.__setitem__("symbol", "ETH-USDC"),
    lambda frame: frame.__setitem__("exchange", "coinbase"),
    lambda frame: frame.__setitem__("target_class", "invalid"),
    lambda frame: frame.__setitem__("execution_target_return", 1),
    lambda frame: frame.__setitem__("feature_return_1", np.nan),
    lambda frame: frame.__setitem__("feature_return_1", 1+2j),
    lambda frame: frame.__setitem__("execution_entry_at", frame.timestamp),
    lambda frame: frame.__setitem__("volume", -1),
])
def test_invalid_partitions_are_refused_even_with_updated_artifact_hash(prepared, change):
    path = prepared / "BTC/train.parquet"
    frame = pd.read_parquet(path)
    change(frame)
    # Arrow does not store complex arrays; exercise their validation directly.
    manifest = json.loads((prepared / "history.json").read_text(encoding="utf-8"))
    if pd.api.types.is_complex_dtype(frame.feature_return_1):
        with pytest.raises(ValueError):
            train.validate_partition(frame, "BTC", manifest["assets"]["BTC"]["partitions"]["train"], manifest["protocol"])
        return
    write_parquet(path, frame)
    manifest["artifact_hashes"]["BTC/train.parquet"] = file_digest(path)
    write_json(prepared / "history.json", manifest)
    with pytest.raises(ValueError):
        train.load_training_inputs(prepared)


def test_history_protocol_changes_are_refused(prepared):
    path = prepared / "history.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["protocol"]["latency_bars"] = 2
    write_json(path, report)
    with pytest.raises(ValueError, match="protocol disagree"):
        train.load_training_inputs(prepared)


def test_missing_training_classes_publish_explicit_unavailability(tmp_path, monkeypatch):
    prepared = make_history(tmp_path, monkeypatch, flat=True)
    output, report = train.train_hyperliquid_models(prepared, device="cpu", recipe=FAST)
    checkpoint, _, _ = train.load_saved_market(output, "BTC")
    assert set(checkpoint["classifiers"]) == {"naive"}
    assert report["markets"]["BTC"]["status"] == "primary_model_unavailable"
    assert set(checkpoint["unavailable_models"]) == {"logistic", "xgboost", "mlp"}
    assert checkpoint["primary_model"] == "mlp"  # No substitution after outcomes.


def test_two_training_classes_use_binary_logistic_with_explicit_class_alignment(tmp_path, monkeypatch):
    prepared = make_history(tmp_path, monkeypatch, flat="two")
    output, report = train.train_hyperliquid_models(prepared, device="cpu", recipe=FAST)
    checkpoint, root, _ = train.load_saved_market(output, "BTC")
    assert set(checkpoint["classifiers"]) == {"naive", "logistic"}
    assert checkpoint["classifiers"]["logistic"]["logistic_probability_link"] == "binary_sigmoid"
    tables, _, _, _ = train.load_training_inputs(prepared)
    predictions = train.predict_classifiers(checkpoint, tables["BTC"]["validation"], root)["logistic"]
    assert predictions.shape[1] == 3
    assert (predictions[:, train.CLASS_NAMES.index("down")] == 0).all()
    assert report["markets"]["BTC"]["status"] == "primary_model_unavailable"


def test_gap_safe_prepared_partitions_can_train_without_forcing_contiguous_full_history(tmp_path, monkeypatch):
    prepared = make_history(tmp_path, monkeypatch, gap=True)
    output, report = train.train_hyperliquid_models(prepared, device="cpu", recipe=FAST)
    assert report["markets"]["BTC"]["status"] == "trained"
    assert train.load_saved_market(output, "BTC")[0]["coin"] == "BTC"


@pytest.mark.parametrize("artifact", ["BTC/checkpoint.json", "BTC/xgboost.json", "BTC/mlp_state.pt", "BTC/metrics.json"])
def test_saved_artifact_changes_are_refused(prepared, monkeypatch, artifact):
    output, _ = train.train_hyperliquid_models(prepared, device="cpu", recipe=FAST)
    with (output / artifact).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="input changed"):
        train.load_saved_market(output, "BTC")


def test_changed_venue_binding_with_updated_hash_is_refused(prepared):
    output, report = train.train_hyperliquid_models(prepared, device="cpu", recipe=FAST)
    path = output / "BTC/checkpoint.json"
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    checkpoint["coin"] = "ETH"
    write_json(path, checkpoint)
    report["artifact_hashes"]["BTC/checkpoint.json"] = file_digest(path)
    write_json(output / "models.json", report)
    with pytest.raises(ValueError, match="market/protocol disagrees"):
        train.load_saved_market(output, "BTC")


def test_source_mutation_or_failure_after_first_market_publishes_no_partial_batch(prepared, monkeypatch):
    original = train._assess_market
    def mutate(coin, tables, recipe, root, source):
        result = original(coin, tables, recipe, root, source)
        with (prepared / "input_config.toml").open("a", encoding="utf-8") as stream:
            stream.write("\n# changed during fitting\n")
        return result
    monkeypatch.setattr(train, "_assess_market", mutate)
    with pytest.raises(ValueError, match="input changed"):
        train.train_hyperliquid_models(prepared, device="cpu", recipe=FAST)
    assert list((prepared.parent.parent / "models").iterdir()) == []


def test_cuda_is_never_silently_replaced_with_cpu(prepared, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA requested but unavailable"):
        train.train_hyperliquid_models(prepared, device="cuda", recipe=FAST)
    assert not (prepared.parent.parent / "models").exists()


def test_future_or_precalibration_inference_is_refused(prepared):
    output, _ = train.train_hyperliquid_models(prepared, device="cpu", recipe=FAST)
    tables, _, _, _ = train.load_training_inputs(prepared)
    with pytest.raises(ValueError, match="follow calibration"):
        train.predict_saved_market(output, "BTC", tables["BTC"]["train"].iloc[:1])
    frame = tables["BTC"]["validation"].iloc[-1:].copy()
    frame["available_at"] = datetime.now(timezone.utc)+timedelta(days=1)
    with pytest.raises(ValueError, match="follow calibration"):
        train.predict_saved_market(output, "BTC", frame)
    frame["symbol"] = "ETH-USDC"
    with pytest.raises(ValueError, match="market disagrees"):
        train.predict_saved_market(output, "BTC", frame)


def test_single_causal_row_inference_needs_no_future_label_columns(prepared):
    output, _ = train.train_hyperliquid_models(prepared, device="cpu", recipe=FAST)
    tables, _, _, _ = train.load_training_inputs(prepared)
    frame = tables["BTC"]["validation"].iloc[-1:][list(train.CANDLE_COLUMNS+train.FEATURE_COLUMNS)]
    result = train.predict_saved_market(output, "BTC", frame)
    assert result["probabilities"]["mlp"].shape == (1, 3)
    assert result["return_forecasts"]["ridge"].shape == (1,)


def test_deep_windows_paths_and_hip3_identifiers(tmp_path, monkeypatch):
    import torch
    import xgboost
    original_load, original_save = xgboost.XGBClassifier.load_model, torch.save
    def memory_load(model, value):
        assert isinstance(value, bytearray), "Native XGBoost must not open extended Windows filenames"
        return original_load(model, value)
    def stream_save(value, stream, *args, **kwargs):
        assert hasattr(stream, "write"), "Native Torch writer must receive an open Python file"
        return original_save(value, stream, *args, **kwargs)
    monkeypatch.setattr(xgboost.XGBClassifier, "load_model", memory_load)
    monkeypatch.setattr(torch, "save", stream_save)
    root = native_path(tmp_path / ("deep-"+"x"*70) / ("nested-"+"y"*70) / ("root-"+"z"*70))
    prepared = make_history(root, monkeypatch, coins=("xyz:HOOD", "kPEPE"))
    output, report = train.train_hyperliquid_models(prepared, device="cpu", recipe=FAST)
    assert len(str(output / "xyz_HOOD/mlp_state.pt")) > 310
    assert train.load_saved_market(output, "xyz:HOOD")[0]["symbol"] == "xyz:HOOD-USDC"
    assert train.load_saved_market(output, "kPEPE")[0]["coin"] == "kPEPE"
    assert set(report["markets"]) == {"xyz:HOOD", "kPEPE"}


def test_reader_preserves_compatibility_with_previous_native_model_files(prepared, tmp_path):
    import torch
    from local_ai_trader.models.mlp import fit_mlp, predict_mlp_checkpoint
    tables, _, _, _ = train.load_training_inputs(prepared)
    frame = tables["BTC"]["train"]
    root = native_path(tmp_path / "old-native-models")
    root.mkdir()
    boosted = train.fit_boosting(frame, FAST)
    boosted.save_model(root / "xgboost.json")
    checkpoint = {"schema_version": 1, "model_type": "xgboost", "calibration": "none", "preprocessing": "identity",
                  "probability_postprocessing": "float64_row_normalization", "feature_names": list(train.FEATURE_COLUMNS),
                  "class_names": list(train.CLASS_NAMES), "model_file": "xgboost.json", "model_sha256": file_digest(root / "xgboost.json")}
    np.testing.assert_allclose(train.predict_boosting_checkpoint(checkpoint, frame, root)["xgboost"], train.predict_boosting(boosted, frame), rtol=1e-7, atol=1e-8)
    recipe = train.replace(FAST, mlp=train.replace(FAST.mlp, device="cpu"))
    neural = fit_mlp(frame, recipe)
    torch.save(neural.model.state_dict(), root / "mlp_state.pt")
    checkpoint = {"schema_version": 1, "model_type": "mlp", "calibration": "none", "preprocessing": "training_population_standardization",
                  "probability_postprocessing": "float64_softmax", "feature_names": list(train.FEATURE_COLUMNS), "class_names": list(train.CLASS_NAMES),
                  "model_file": "mlp_state.pt", "model_sha256": file_digest(root / "mlp_state.pt"), "model_config": train.asdict(recipe.mlp),
                  "scaler_mean": neural.mean.tolist(), "scaler_scale": neural.scale.tolist()}
    np.testing.assert_allclose(predict_mlp_checkpoint(checkpoint, frame, root)["mlp"], neural.predict(frame), rtol=1e-7, atol=1e-8)


def test_cli_exposes_explicit_device_and_saved_directory(prepared, monkeypatch, capsys):
    original = train.train_hyperliquid_models
    monkeypatch.setattr(train, "train_hyperliquid_models", lambda path, device: original(path, device=device, recipe=FAST))
    assert main(["hyperliquid-train", str(prepared), "--device", "cpu"]) == 0
    output = capsys.readouterr().out
    assert "test files neither opened nor hashed" in output
    assert "otherwise eligible BUY candidates require REVIEW" in output
    assert "Saved Hyperliquid models to" in output
