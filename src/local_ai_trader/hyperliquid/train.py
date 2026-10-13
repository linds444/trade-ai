"""Frozen venue-specific models; calibration/assessment read validation only.

Preparation lineage is established by its manifest. Fitting deliberately never
opens, stats or hashes reserved test data or full raw/candle/feature sources.
"""

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import tempfile
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from local_ai_trader.calibration.temperature import apply_temperature, fit_temperature
from local_ai_trader.data.storage import native_path, write_json, write_parquet
from local_ai_trader.features.build_features import FEATURE_COLUMNS
from local_ai_trader.forecast.baseline import evaluate_return_predictions
from local_ai_trader.forecast.baseline import return_feature_matrix
from local_ai_trader.forecast.targets import CANDLE_COLUMNS
from local_ai_trader.hyperliquid.feed import load_hyperliquid_settings
from local_ai_trader.hyperliquid.history import LABEL_COLUMNS, research_protocol
from local_ai_trader.hyperliquid.markets import BY_COIN
from local_ai_trader.models.baseline import CLASS_NAMES, encode_labels, feature_matrix, fit_baselines, predict_checkpoint
from local_ai_trader.models.boosting import fit_boosting, predict_boosting, predict_boosting_checkpoint
from local_ai_trader.models.evaluate import evaluate_probabilities
from local_ai_trader.models.train import file_digest
from local_ai_trader.settings import BoostingSettings, CalibrationSettings, MLPSettings

LOGGER = logging.getLogger(__name__)
MODEL_NAMES = ("naive", "logistic", "xgboost", "mlp")


@dataclass(frozen=True)
class TrainingRecipe:
    random_seed: int = 42
    logistic_c: float = 1.0
    logistic_max_iter: int = 1000
    naive_smoothing: float = 1.0
    calibration_bins: int = 10
    ridge_alpha: float = 1.0
    boosting: BoostingSettings = field(default_factory=BoostingSettings)
    mlp: MLPSettings = field(default_factory=MLPSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)


def _verify(inputs):
    for path, digest in inputs.items():
        if file_digest(path) != digest:
            raise ValueError(f"Hyperliquid training input changed: {path.name}")


def _timestamp(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("Require nonmissing timezone-aware model timestamps")
    return result.tz_convert("UTC")


def validate_partition(frame, coin, summary, protocol):
    expected = set(CANDLE_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMNS + ("segment_id",))
    if frame.empty or not frame.columns.is_unique or set(frame.columns) != expected or len(frame) != summary["rows"]:
        raise ValueError("Hyperliquid model partition schema/count disagrees")
    if frame.symbol.nunique(dropna=False) != 1 or frame.symbol.iloc[0] != f"{coin}-USDC" or frame.exchange.nunique(dropna=False) != 1 or frame.exchange.iloc[0] != "hyperliquid":
        raise ValueError("Hyperliquid model partition market disagrees")
    frame = frame.copy().reset_index(drop=True)
    for name in ("timestamp", "available_at", "execution_entry_at", "execution_exit_at", "execution_target_available_at"):
        if frame[name].isna().any() or getattr(frame[name].dtype, "tz", None) is None:
            raise ValueError("Partition times must be nonmissing timezone-aware values")
        frame[name] = frame[name].dt.tz_convert("UTC")
    stamps = frame.timestamp.dt.as_unit("ns").astype("int64").to_numpy()
    if (stamps % 300_000_000_000).any() or (np.diff(stamps) <= 0).any():
        raise ValueError("Prediction times must be ordered, unique and interval-aligned")
    for name, bars in (("available_at", 1), ("execution_entry_at", 2), ("execution_exit_at", 8), ("execution_target_available_at", 9)):
        if not frame[name].eq(frame.timestamp + pd.to_timedelta(bars * 300, unit="s")).all():
            raise ValueError("Execution label availability/timing disagrees")
    if frame.execution_target_available_at.max() > pd.Timestamp.now(tz="UTC"):
        raise ValueError("Training/validation contains unknown future outcomes")
    for name in ("open", "high", "low", "close", "volume", "execution_entry_reference_open", "execution_exit_reference_open", "execution_target_return"):
        if not pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(frame[name]) or pd.api.types.is_complex_dtype(frame[name]):
            raise ValueError("Model prices/returns must be real numeric values")
        values = frame[name].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (name == "volume" and (values < 0).any()) or (name not in ("volume", "execution_target_return") and (values <= 0).any()):
            raise ValueError("Model prices/returns must be finite and valid")
    if (frame.high < frame[["open", "close", "low"]].max(axis=1)).any() or (frame.low > frame[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Partition OHLC relationships disagree")
    return_feature_matrix(frame)
    encode_labels(frame.target_class)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        returns = frame.execution_exit_reference_open.to_numpy() / frame.execution_entry_reference_open.to_numpy() - 1
    if not np.isfinite(returns).all() or not np.array_equal(returns, frame.execution_target_return.to_numpy()):
        raise ValueError("Partition gross returns disagree with reference prices")
    threshold = protocol["gross_return_class_threshold"]
    classes = np.where(returns > threshold, "up", np.where(returns < -threshold, "down", "flat"))
    if not np.array_equal(classes, frame.target_class.to_numpy()):
        raise ValueError("Partition classes disagree with gross-return threshold")
    for key, actual in (("first_prediction_time", frame.available_at.iloc[0]), ("last_prediction_time", frame.available_at.iloc[-1]), ("last_label_known_at", frame.execution_target_available_at.iloc[-1])):
        if _timestamp(summary[key]) != actual:
            raise ValueError("Partition times disagree with source manifest")
    return frame


def load_training_inputs(directory):
    """Read only manifest, exact config bytes, train and validation files."""
    directory = native_path(Path(directory)).resolve()
    manifest = directory / "history.json"
    inputs = {manifest: file_digest(manifest)}
    with manifest.open(encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("schema_version") != 1 or report.get("mode") != "hyperliquid_research_preparation" or report.get("exchange") != "hyperliquid" or report.get("source_mode") != "observed" or report.get("instrument") != "perpetual" or report.get("collateral_currency") != "USDC":
        raise ValueError("Require an observed Hyperliquid research archive")
    if any(report.get(name) is not False for name in ("models_fitted", "strategy_performance_evaluated", "forward_strategy_started", "paper_fills_created", "orders_created", "live_execution_enabled")):
        raise ValueError("Require research-preparation lineage without execution claims")
    config = directory / "input_config.toml"
    inputs[config] = report["input_config_sha256"]
    _verify(inputs)
    if report["artifact_hashes"]["input_config.toml"] != inputs[config]:
        raise ValueError("Research config hash disagrees")
    settings = load_hyperliquid_settings(config)
    if list(settings.coins) != report["coins"] or set(report["assets"]) != set(settings.coins) or report["protocol"] != research_protocol(settings, report["protocol"]["requested_closed_bars"]):
        raise ValueError("Research contracts or frozen protocol disagree")
    tables = {}
    for coin in settings.coins:
        asset = report["assets"][coin]
        if asset["contract"] != coin or asset["directory"] != BY_COIN[coin].filename or asset["underlying_units_per_contract"] != BY_COIN[coin].underlying_units_per_contract:
            raise ValueError("Research contract registry disagrees")
        if asset["status"] != "ready":
            continue
        tables[coin] = {}
        for name in ("train", "validation"):
            path = directory / BY_COIN[coin].filename / f"{name}.parquet"
            inputs[path] = report["artifact_hashes"][f"{BY_COIN[coin].filename}/{name}.parquet"]
            _verify(inputs)
            tables[coin][name] = validate_partition(pd.read_parquet(path), coin, asset["partitions"][name], report["protocol"])
        train, validation = tables[coin]["train"], tables[coin]["validation"]
        if train.execution_target_available_at.max() >= validation.available_at.min() or validation.execution_target_available_at.max() >= _timestamp(asset["partitions"]["test"]["first_prediction_time"]):
            raise ValueError("Research labels overlap the next partition")
    _verify(inputs)
    return tables, report, settings, inputs


def calibration_partitions(validation, recipe):
    cut = int(len(validation) * recipe.calibration.fit_fraction)
    if not 0 < cut < len(validation):
        raise ValueError("Not enough validation rows for calibration and assessment")
    assessment = validation.iloc[cut:].copy().reset_index(drop=True)
    earlier = validation.iloc[:cut]
    fit = earlier.loc[earlier.execution_target_available_at < assessment.available_at.iloc[0]].reset_index(drop=True)
    if fit.empty:
        raise ValueError("Calibration partition empty after execution-label purge")
    return fit, assessment, {"fit_rows": len(fit), "assessment_rows": len(assessment), "purged_rows": len(earlier)-len(fit),
                             "fit_last_label_known_at": fit.execution_target_available_at.max().isoformat(),
                             "assessment_first_prediction_time": assessment.available_at.iloc[0].isoformat()}


def _return_checkpoint(train, alpha):
    inputs = feature_matrix(train)
    scaler = StandardScaler().fit(inputs)
    regressor = Ridge(alpha=alpha, solver="svd").fit(scaler.transform(inputs), train.execution_target_return.to_numpy())
    checkpoint = {"feature_names": list(FEATURE_COLUMNS), "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                  "coefficients": regressor.coef_.tolist(), "intercept": float(regressor.intercept_),
                  "training_mean": float(train.execution_target_return.mean()), "ridge_alpha": alpha, "solver": "svd"}
    np.testing.assert_allclose(predict_returns(checkpoint, train)["ridge"], regressor.predict(scaler.transform(inputs)), rtol=1e-10, atol=1e-12)
    return checkpoint


def predict_returns(checkpoint, frame):
    width = len(FEATURE_COLUMNS)
    if checkpoint.get("feature_names") != list(FEATURE_COLUMNS) or checkpoint.get("solver") != "svd":
        raise ValueError("Incompatible Hyperliquid return checkpoint")
    values = []
    for name in ("scaler_mean", "scaler_scale", "coefficients"):
        array = np.asarray(checkpoint[name])
        if array.shape != (width,) or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError("Invalid saved return parameters")
        values.append(array.astype(float))
    mean, scale, coefficients = values
    scalars = [checkpoint[name] for name in ("intercept", "training_mean", "ridge_alpha")]
    if any(type(value) not in (int, float) or not np.isfinite(value) for value in scalars) or (scale <= 0).any() or checkpoint["ridge_alpha"] <= 0:
        raise ValueError("Invalid saved return scaling/scalars")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ridge = ((return_feature_matrix(frame) - mean) / scale) @ coefficients + checkpoint["intercept"]
    if not np.isfinite(ridge).all():
        raise ValueError("Nonfinite saved return prediction")
    return {"mean": np.full(len(frame), checkpoint["training_mean"]), "ridge": ridge}


def _fit_market(train, recipe, root):
    """Fit every declared family from train only; absent classes are explicit."""
    classifiers, unavailable, hardware = {}, {}, {}
    labels = encode_labels(train.target_class)
    counts = np.bincount(labels, minlength=3)
    classifiers["naive"] = {"probabilities": ((counts+recipe.naive_smoothing)/(len(labels)+3*recipe.naive_smoothing)).tolist()}
    if len(np.unique(labels)) >= 2:
        fitted = fit_baselines(train, recipe)
        classifiers["logistic"] = fitted.checkpoint()
        for name, actual in fitted.predict(train).items():
            np.testing.assert_allclose(predict_checkpoint(classifiers["logistic"], train)[name], actual, rtol=1e-10, atol=1e-12)
    else:
        unavailable["logistic"] = "fewer_than_two_training_classes"
    if len(np.unique(labels)) == 3:
        boosted = fit_boosting(train, recipe)
        boosted.save_model(root / "xgboost.json")
        classifiers["xgboost"] = {"schema_version": 1, "model_type": "xgboost", "calibration": "none", "preprocessing": "identity",
                                  "probability_postprocessing": "float64_row_normalization", "feature_names": list(FEATURE_COLUMNS), "class_names": list(CLASS_NAMES),
                                  "model_file": "xgboost.json", "model_sha256": file_digest(root / "xgboost.json")}
        np.testing.assert_allclose(predict_boosting_checkpoint(classifiers["xgboost"], train, root)["xgboost"], predict_boosting(boosted, train), rtol=1e-7, atol=1e-8)
        from local_ai_trader.models.mlp import fit_mlp, predict_mlp_checkpoint
        import torch
        neural = fit_mlp(train, recipe)
        torch.save({key: value.detach().cpu() for key, value in neural.model.state_dict().items()}, root / "mlp_state.pt")
        classifiers["mlp"] = {"schema_version": 1, "model_type": "mlp", "calibration": "none", "preprocessing": "training_population_standardization",
                              "probability_postprocessing": "float64_softmax", "feature_names": list(FEATURE_COLUMNS), "class_names": list(CLASS_NAMES),
                              "model_file": "mlp_state.pt", "model_sha256": file_digest(root / "mlp_state.pt"), "model_config": asdict(recipe.mlp),
                              "scaler_mean": neural.mean.tolist(), "scaler_scale": neural.scale.tolist(), "scaler_variance": neural.variance.tolist(), "training_rows": len(train)}
        hardware = {"mlp": neural.hardware, "training_history": neural.history}
        # Compare on the training device; future observed inference is CPU.
        restored = predict_mlp_checkpoint(classifiers["mlp"], train, root, device=str(neural.device))["mlp"]
        np.testing.assert_allclose(restored, neural.predict(train), rtol=1e-7, atol=1e-8)
    else:
        unavailable.update(xgboost="missing_training_class", mlp="missing_training_class")
    return classifiers, _return_checkpoint(train, recipe.ridge_alpha), unavailable, hardware


def predict_classifiers(checkpoint, frame, root):
    classifiers = checkpoint["classifiers"]
    naive = np.asarray(classifiers["naive"]["probabilities"], dtype=float)
    if naive.shape != (3,) or not np.isfinite(naive).all() or (naive < 0).any() or not np.isclose(naive.sum(), 1):
        raise ValueError("Invalid training-frequency probabilities")
    result = {"naive": np.tile(naive, (len(feature_matrix(frame)), 1))}
    if "logistic" in classifiers:
        result["logistic"] = predict_checkpoint(classifiers["logistic"], frame)["logistic"]
    if "xgboost" in classifiers:
        result.update(predict_boosting_checkpoint(classifiers["xgboost"], frame, root))
    if "mlp" in classifiers:
        from local_ai_trader.models.mlp import predict_mlp_checkpoint
        result.update(predict_mlp_checkpoint(classifiers["mlp"], frame, root))
    return result


def _assess_market(coin, tables, recipe, root, source):
    train = tables["train"]
    fit, assessment, split = calibration_partitions(tables["validation"], recipe)
    classifiers, returns, unavailable, hardware = _fit_market(train, recipe, root)
    primary = "mlp" if coin == "BTC" else "xgboost"
    checkpoint = {"schema_version": 1, "mode": "hyperliquid_frozen_models", "coin": coin, "symbol": f"{coin}-USDC", "exchange": "hyperliquid",
                  "source_mode": "observed", "instrument": "perpetual", "collateral_currency": "USDC", "feature_names": list(FEATURE_COLUMNS),
                  "class_names": list(CLASS_NAMES), "protocol": source["protocol"], "training_rows": len(train),
                  "training_last_label_known_at": train.execution_target_available_at.max().isoformat(),
                  "primary_model": primary, "primary_model_selection": "fixed before calibration/assessment: BTC MLP; all other contracts XGBoost",
                  "classifiers": classifiers, "returns": returns, "unavailable_models": unavailable,
                  "history_manifest_sha256": source["history_manifest_sha256"], "source_config_sha256": source["input_config_sha256"],
                  "temperatures": {}, "calibration_split": split,
                  "confidence": None, "reliability_validation_id": None, "reliability_status": "independent_reliability_measure_unvalidated"}
    fitting = predict_classifiers(checkpoint, fit, root)
    for name, probabilities in fitting.items():
        checkpoint["temperatures"][name] = fit_temperature(fit.target_class, probabilities, recipe.calibration)
    checkpoint["calibration_last_label_known_at"] = fit.execution_target_available_at.max().isoformat()
    checkpoint["frozen_at"] = datetime.now(timezone.utc).isoformat()
    # All weights, preprocessing and temperatures are frozen before assessment.
    write_json(root / "checkpoint.json", checkpoint)
    frozen_digest = file_digest(root / "checkpoint.json")
    raw = predict_classifiers(checkpoint, assessment, root)
    return_predictions = predict_returns(returns, assessment)
    metrics = {"classifiers": {}, "returns": {}, "test": {"opened": False, "hashed": False, "evaluated": False},
               "confidence_available": False, "reliability_status": checkpoint["reliability_status"]}
    predicted = assessment[["timestamp", "available_at", "execution_target_available_at", "target_class", "execution_target_return", "symbol", "exchange"]].copy()
    for name, probabilities in raw.items():
        calibrated = apply_temperature(probabilities, checkpoint["temperatures"][name]["temperature"])
        metrics["classifiers"][name] = {}
        for variant, values in (("raw", probabilities), ("temperature", calibrated)):
            scores = evaluate_probabilities(assessment.target_class, values, recipe.calibration_bins)
            scores["calibration_status"] = "raw" if variant == "raw" else "temperature_fitted_on_earlier_validation"
            metrics["classifiers"][name][variant] = scores
            for index, label in enumerate(CLASS_NAMES):
                predicted[f"{name}_{variant}_p_{label}"] = values[:, index]
    for name, values in return_predictions.items():
        metrics["returns"][name] = evaluate_return_predictions(assessment.execution_target_return.to_numpy(), values)
        predicted[f"return_{name}"] = values
    write_parquet(root / "assessment_predictions.parquet", predicted)
    write_json(root / "metrics.json", metrics)
    write_json(root / "training.json", {"hardware": hardware, "recipe": asdict(recipe), "class_order": list(CLASS_NAMES),
                                       "validation_used_for_base_model_fitting": False, "early_stopping": False, "test_opened": False})
    if file_digest(root / "checkpoint.json") != frozen_digest:
        raise ValueError("Frozen Hyperliquid checkpoint changed during assessment")
    return {"status": "trained" if primary in classifiers else "primary_model_unavailable", "primary_model": primary,
            "training_rows": len(train), "calibration": split, "unavailable_models": unavailable,
            "metrics": metrics, "confidence_available": False}


def train_hyperliquid_models(directory, *, device="cuda", recipe=None):
    if device not in ("cuda", "cpu"):
        raise ValueError("Hyperliquid training device must be cuda or cpu")
    recipe = recipe or TrainingRecipe()
    recipe = replace(recipe, mlp=replace(recipe.mlp, device=device))
    tables, source, settings, inputs = load_training_inputs(directory)
    source["history_manifest_sha256"] = inputs[native_path(Path(directory)).resolve() / "history.json"]
    if not tables:
        raise ValueError("No Hyperliquid markets have sufficient prepared history")
    # Refuse missing CUDA before fitting or publishing any part of a batch.
    try:
        from local_ai_trader.models.mlp import training_device
    except ImportError as error:
        raise ValueError("PyTorch is unavailable in this project interpreter; use the verified CUDA environment") from error
    training_device(device)
    # The copied config preserves relative paths verbatim. Its new archive
    # location must not reinterpret those paths as a nested project root.
    parent = native_path(Path(directory)).resolve().parent.parent / "models"
    parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    output = parent / f"models_{now.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:8]}"
    staging = Path(tempfile.mkdtemp(prefix=".models-staging-", dir=parent))
    report = {"schema_version": 1, "mode": "hyperliquid_model_research", "exchange": "hyperliquid", "source_mode": "observed",
              "source_history_directory": str(native_path(Path(directory)).resolve()), "created_at": now.isoformat(),
              "recipe": asdict(recipe), "coins": list(settings.coins), "protocol": source["protocol"],
              "history_manifest_sha256": source["history_manifest_sha256"], "source_config_sha256": source["input_config_sha256"],
              "input_hashes": {str(path): digest for path, digest in inputs.items()}, "markets": {},
              "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
              "test_opened": False, "test_hashed": False, "forward_strategy_started": False, "paper_fills_created": False, "live_execution_enabled": False}
    try:
        for coin in settings.coins:
            if coin not in tables:
                report["markets"][coin] = {"status": "insufficient_history", "confidence_available": False}
                continue
            LOGGER.info("Training Hyperliquid %s: naive/logistic/XGBoost/MLP and mean/Ridge", coin)
            root = staging / BY_COIN[coin].filename
            root.mkdir()
            report["markets"][coin] = _assess_market(coin, tables[coin], recipe, root, source)
            _verify(inputs)
        report["artifact_hashes"] = {path.relative_to(staging).as_posix(): file_digest(path) for path in staging.rglob("*") if path.is_file()}
        write_json(staging / "models.json", report)
        _verify(inputs)
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print("Hyperliquid frozen-model assessment (later validation; reserved tests unopened):")
    for coin, market in report["markets"].items():
        if "metrics" not in market:
            print(f"{coin}: {market['status']}")
            continue
        print(f"{coin}: train {market['training_rows']}; calibration {market['calibration']['fit_rows']}; purged {market['calibration']['purged_rows']}; assessment {market['calibration']['assessment_rows']}; fixed primary {market['primary_model']}")
        rows = [{"model": name, "variant": variant, **{key: values[key] for key in ("accuracy", "f1_macro", "log_loss", "ece")}}
                for name, variants in market["metrics"]["classifiers"].items() for variant, values in variants.items()]
        print(pd.DataFrame(rows).to_string(index=False))
        print(pd.DataFrame([{"return_model": name, **{key: values[key] for key in ("mae", "rmse", "r2")}}
                            for name, values in market["metrics"]["returns"].items()]).to_string(index=False))
    print("PASS: venue-specific weights, temperatures and return forecasts saved; test files neither opened nor hashed")
    print("Independent reliability confidence: unavailable; otherwise eligible BUY candidates require REVIEW; paper strategy month has not started")
    print(f"Saved Hyperliquid models to {output}")
    return output, report


def load_saved_market(directory, coin):
    """Audit saved artifacts only; no training data or model fitting needed."""
    directory = native_path(Path(directory)).resolve()
    with (directory / "models.json").open(encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("schema_version") != 1 or report.get("mode") != "hyperliquid_model_research" or report.get("exchange") != "hyperliquid" or report.get("source_mode") != "observed" or coin not in BY_COIN or coin not in report["coins"]:
        raise ValueError("Unsupported saved Hyperliquid model batch")
    if report.get("live_execution_enabled") is not False or report.get("test_opened") is not False or report.get("test_hashed") is not False:
        raise ValueError("Saved model execution/test contract disagrees")
    market = report["markets"][coin]
    if market["status"] not in ("trained", "primary_model_unavailable"):
        raise ValueError("Requested market lacks trained models")
    prefix = BY_COIN[coin].filename + "/"
    hashes = {name: digest for name, digest in report["artifact_hashes"].items() if name.startswith(prefix)}
    if prefix+"checkpoint.json" not in hashes or any(Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name or ":" in name for name in hashes):
        raise ValueError("Unsafe/incomplete saved market artifacts")
    _verify({directory / name: digest for name, digest in hashes.items()})
    root = directory / BY_COIN[coin].filename
    with (root / "checkpoint.json").open(encoding="utf-8") as stream:
        checkpoint = json.load(stream)
    expected = {"schema_version": 1, "mode": "hyperliquid_frozen_models", "coin": coin, "symbol": f"{coin}-USDC", "exchange": "hyperliquid",
                "source_mode": "observed", "instrument": "perpetual", "collateral_currency": "USDC", "feature_names": list(FEATURE_COLUMNS),
                "class_names": list(CLASS_NAMES), "protocol": report["protocol"], "primary_model": market["primary_model"],
                "history_manifest_sha256": report["history_manifest_sha256"], "source_config_sha256": report["source_config_sha256"],
                "confidence": None, "reliability_validation_id": None, "reliability_status": "independent_reliability_measure_unvalidated"}
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved Hyperliquid checkpoint market/protocol disagrees")
    if set(checkpoint["temperatures"]) != set(checkpoint["classifiers"]) or not set(checkpoint["classifiers"]) <= set(MODEL_NAMES):
        raise ValueError("Saved classifier calibration set disagrees")
    if checkpoint["primary_model"] != ("mlp" if coin == "BTC" else "xgboost"):
        raise ValueError("Saved primary model differs from the fixed selection rule")
    _verify({directory / name: digest for name, digest in hashes.items()})
    return checkpoint, root, "sha256:"+hashes[prefix+"checkpoint.json"]


def predict_saved_market(directory, coin, frame):
    """Infer after all fitting labels were known; preserve unavailable reliability."""
    checkpoint, root, source_id = load_saved_market(directory, coin)
    if frame.empty or frame.symbol.nunique(dropna=False) != 1 or frame.symbol.iloc[0] != checkpoint["symbol"] or frame.exchange.nunique(dropna=False) != 1 or frame.exchange.iloc[0] != "hyperliquid":
        raise ValueError("Observed model inference market disagrees")
    if frame.available_at.isna().any() or getattr(frame.available_at.dtype, "tz", None) is None or not (frame.available_at > _timestamp(checkpoint["calibration_last_label_known_at"])).all() or frame.available_at.max() > pd.Timestamp.now(tz="UTC"):
        raise ValueError("Inference must follow calibration label availability and use completed observations")
    probabilities = predict_classifiers(checkpoint, frame, root)
    calibrated = {name: apply_temperature(values, checkpoint["temperatures"][name]["temperature"]) for name, values in probabilities.items()}
    returns = predict_returns(checkpoint["returns"], frame)
    if any((values <= -1).any() for values in returns.values()):
        raise ValueError("Return forecast at or below -100% is invalid; no clipping is applied")
    if "sha256:"+file_digest(root / "checkpoint.json") != source_id:
        raise ValueError("Checkpoint changed during saved inference")
    return {"probabilities": calibrated, "raw_probabilities": probabilities, "return_forecasts": returns,
            "primary_model": checkpoint["primary_model"], "confidence": None, "reliability_validation_id": None, "calibration_id": source_id}
