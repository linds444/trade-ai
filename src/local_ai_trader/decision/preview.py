"""Historical flat-position preview with an optional audited return forecaster."""

from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from local_ai_trader.data.storage import native_path, write_json
from local_ai_trader.decision.cost_aware import ProbabilityPrediction, decide_cost_aware, load_decision_settings
from local_ai_trader.features.build_features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from local_ai_trader.forecast.baseline import MODEL_NAMES, checkpoint_return_forecasts
from local_ai_trader.forecast.train import load_return_model_snapshot
from local_ai_trader.models.train import file_digest
from local_ai_trader.uncertainty.report import PROBABILITY_COLUMNS, align_predictions
from local_ai_trader.walkforward.completed import load_completed_execution

LOGGER = logging.getLogger(__name__)
RETURN_INPUT_COLUMNS = ("timestamp", "available_at", "open", "close", "symbol", "exchange", *FEATURE_COLUMNS)


def return_preview_forecast(snapshot, return_snapshot, chosen: pd.DataFrame, prediction: ProbabilityPrediction, model: str):
    """Infer one row from the matching frozen classifier fold, with no labels."""
    settings, return_settings = snapshot.protocol.settings, return_snapshot.settings
    for name in ("candle_seconds", "horizon_steps", "momentum_steps", "rolling_window"):
        if getattr(settings, name) != getattr(return_settings, name):
            raise ValueError(f"Return model and classifier feature/timing settings disagree on {name}")
    if return_settings.backtest.latency_bars != settings.backtest.latency_bars:
        raise ValueError("Return model and classifier execution latency disagree")
    if return_snapshot.checkpoint["symbol"] != prediction.symbol or return_snapshot.checkpoint["exchange"] != "coinbase":
        raise ValueError("Return model and classifier market disagree")
    if chosen.available_at.duplicated().any() or not chosen.available_at.is_monotonic_increasing:
        raise ValueError("Saved classifier signals must be unique and chronological")
    row = chosen.iloc[-1]
    if isinstance(row.fold, (bool, np.bool_)) or not isinstance(row.fold, (int, np.integer)) or int(row.fold) != len(snapshot.protocol.plan["folds"]):
        raise ValueError("Latest classifier signal must belong to the final frozen fold")
    fold = int(row.fold)
    directory = snapshot.protocol.fold_directory(fold)
    with (directory / "split.json").open(encoding="utf-8") as stream:
        split_metadata = json.load(stream)
    if split_metadata["feature_metadata"].get("schema_version", FEATURE_SCHEMA_VERSION) != FEATURE_SCHEMA_VERSION:
        raise ValueError("Classifier and return feature schemas disagree")
    path = directory / "test.parquet"
    frame = pd.read_parquet(path)
    matching = frame.loc[frame.available_at.eq(pd.Timestamp(prediction.prediction_time))]
    if len(matching) != 1 or matching.timestamp.iloc[0] != row.timestamp or matching.symbol.iloc[0] != prediction.symbol or matching.exchange.iloc[0] != "coinbase":
        raise ValueError("Return input row and saved classifier signal disagree")
    # Project before inference: classifier labels and execution outcomes never
    # enter the return model, even when present in the historical source table.
    causal = matching.loc[:, list(RETURN_INPUT_COLUMNS)].reset_index(drop=True)
    forecast = checkpoint_return_forecasts(return_snapshot.checkpoint, causal, model)[0]
    return forecast, {"fold": fold, "source_feature_file": str(path), "source_feature_sha256": file_digest(path), "input_columns": list(RETURN_INPUT_COLUMNS), "input_rows": 1}


def create_decision_preview(directory: Path, config_path: Path, return_run: Path | None = None, return_model: str = "ridge") -> tuple[Path, dict]:
    """Explain a candidate from saved data without inventing forecasts or orders."""
    snapshot = load_completed_execution(directory)
    if return_model not in MODEL_NAMES or (return_run is None and return_model != "ridge"):
        raise ValueError("Select a saved return run before choosing a return model")
    config_path = native_path(config_path.resolve())
    config_hash = file_digest(config_path)
    policy = load_decision_settings(config_path)
    settings = snapshot.protocol.settings
    return_snapshot = load_return_model_snapshot(return_run) if return_run is not None else None
    if return_snapshot is None:
        output = snapshot.output / "decision_preview"
    else:
        model_hash = return_snapshot.input_hashes[return_snapshot.directory / "experiment.json"]
        output = snapshot.output / "forecast_decision_previews" / f"{model_hash[:12]}_{return_model}_{config_hash[:12]}"
    if output.exists():
        raise ValueError("Decision preview already exists; inspect its preview.json")
    family = snapshot.protocol.plan["model_family"]
    chosen_path = snapshot.output / "summary" / f"{family}_temperature_predictions.parquet"
    reference_path = snapshot.output / "summary/logistic_temperature_predictions.parquet"
    chosen, reference = pd.read_parquet(chosen_path), pd.read_parquet(reference_path)
    align_predictions(chosen, reference)
    if chosen.empty:
        raise ValueError("Decision preview requires saved predictions")

    def packet(frame):
        row = frame.iloc[-1]
        # Only model probabilities and causal identity/time reach the policy.
        return ProbabilityPrediction(row.symbol, row.available_at.to_pydatetime(), tuple(float(row[name]) for name in PROBABILITY_COLUMNS))

    prediction, comparison = packet(chosen), packet(reference)
    forecast, forecast_inputs = None, None
    if return_snapshot is not None:
        forecast, forecast_inputs = return_preview_forecast(snapshot, return_snapshot, chosen, prediction, return_model)
    decision = decide_cost_aware(prediction, False, policy, settings.backtest, settings.candle_seconds, settings.horizon_steps, forecast=forecast, reference=comparison)
    report = {
        "schema_version": 1, "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "mode": "historical_flat_entry_diagnostic", "research_only": True,
        "symbol": prediction.symbol, "prediction_time": prediction.prediction_time.isoformat(),
        "model": family, "probability_variant": "temperature", "holding_assumption": False,
        "probabilities": dict(zip(("p_up", "p_down", "p_flat"), prediction.probabilities)),
        "expected_gross_return_forecast": None if forecast is None else forecast.expected_gross_return,
        "return_forecast_status": "unavailable_from_current_classification_models" if forecast is None else "saved_execution_aligned_point_forecast",
        "decision": asdict(decision), "policy": asdict(policy),
        "costs": asdict(settings.backtest), "costs_source": "frozen_walkforward_plan",
        "return_forecast_contract": {"basis": "execution_open_to_open", "candle_seconds": settings.candle_seconds, "latency_bars": settings.backtest.latency_bars, "holding_bars": settings.horizon_steps},
        "risk_status": "required_before_execution_not_requested" if decision.risk_check_required else "not_requested_no_buy_candidate", "orders_created": False,
        "model_inference_repeated": forecast is not None, "classifier_inference_repeated": False,
        "return_inference_rows": 0 if forecast is None else 1, "model_or_forecast_fitted": False,
        "strategy_performance_evaluated": False, "existing_backtests_changed": False,
        "plan_sha256": snapshot.protocol.sha256, "execution_journal_sha256": snapshot.state_hash,
        "policy_config_sha256": config_hash,
        "source_prediction_hashes": {chosen_path.name: file_digest(chosen_path), reference_path.name: file_digest(reference_path)},
        "limitations": "historical snapshot, not a current market decision; forecast timing, freshness and separate risk authorization are required for a future paper coordinator",
    }
    if return_snapshot is not None:
        report["return_model"] = {
            "run_id": return_snapshot.experiment["run_id"], "run_directory": str(return_snapshot.directory), "model": return_model,
            "artifact_hashes": {path.name: digest for path, digest in return_snapshot.input_hashes.items()},
            "training_last_label_known_at": return_snapshot.checkpoint["training_last_label_known_at"],
            "forecast": {**asdict(forecast), "prediction_time": forecast.prediction_time.isoformat()},
            "inference_input": forecast_inputs, "uncertainty_estimate": "not supplied by this point forecaster",
            "reserved_return_test_data_opened": False,
        }

    def verify():
        snapshot.verify()
        if return_snapshot is not None:
            return_snapshot.verify()
        if file_digest(config_path) != config_hash:
            raise ValueError("Decision policy config changed during preview")

    verify()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".decision-preview-staging-", dir=output.parent))
    try:
        write_json(staging / "preview.json", report)
        verify()
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print("Historical cost-aware entry preview (flat-position assumption):")
    print(f"{prediction.symbol}: {decision.action}; reason: {decision.reason}")
    print(f"Gross break-even return after both sides' costs: {100 * decision.break_even_gross_return:.4f}%")
    print(f"Required gross return including configured net margin: > {100 * decision.required_gross_return:.4f}%")
    if forecast is None:
        print("Expected return forecast: unavailable; independent risk authorization has not been requested")
    else:
        print(f"Expected gross return forecast ({return_model}): {100 * forecast.expected_gross_return:.4f}%")
        print(f"Expected net return after modeled costs: {100 * decision.expected_net_return:.4f}%")
        print("Independent risk authorization: not requested; no orders created")
    LOGGER.info("Saved decision preview to %s; no orders or model fitting", output)
    return output, report
