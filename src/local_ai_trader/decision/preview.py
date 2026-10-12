"""Historical flat-position preview; current classifiers have no return forecast."""

from dataclasses import asdict
import logging
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from local_ai_trader.data.storage import native_path, write_json
from local_ai_trader.decision.cost_aware import ProbabilityPrediction, decide_cost_aware, load_decision_settings
from local_ai_trader.models.train import file_digest
from local_ai_trader.uncertainty.report import PROBABILITY_COLUMNS, align_predictions
from local_ai_trader.walkforward.completed import load_completed_execution

LOGGER = logging.getLogger(__name__)


def create_decision_preview(directory: Path, config_path: Path) -> tuple[Path, dict]:
    """Explain a candidate from saved data without inventing forecasts or orders."""
    snapshot = load_completed_execution(directory)
    output = snapshot.output / "decision_preview"
    if output.exists():
        raise ValueError("Decision preview already exists; inspect decision_preview/preview.json")
    config_path = native_path(config_path.resolve())
    config_hash = file_digest(config_path)
    policy = load_decision_settings(config_path)
    settings = snapshot.protocol.settings
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
    decision = decide_cost_aware(prediction, False, policy, settings.backtest, settings.candle_seconds, settings.horizon_steps, forecast=None, reference=comparison)
    report = {
        "schema_version": 1, "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "mode": "historical_flat_entry_diagnostic", "research_only": True,
        "symbol": prediction.symbol, "prediction_time": prediction.prediction_time.isoformat(),
        "model": family, "probability_variant": "temperature", "holding_assumption": False,
        "probabilities": dict(zip(("p_up", "p_down", "p_flat"), prediction.probabilities)),
        "expected_gross_return_forecast": None,
        "return_forecast_status": "unavailable_from_current_classification_models",
        "decision": asdict(decision), "policy": asdict(policy),
        "costs": asdict(settings.backtest), "costs_source": "frozen_walkforward_plan",
        "return_forecast_contract": {"basis": "execution_open_to_open", "candle_seconds": settings.candle_seconds, "latency_bars": settings.backtest.latency_bars, "holding_bars": settings.horizon_steps},
        "risk_status": "not_requested_no_buy_candidate", "orders_created": False,
        "model_inference_repeated": False, "model_or_forecast_fitted": False,
        "strategy_performance_evaluated": False, "existing_backtests_changed": False,
        "plan_sha256": snapshot.protocol.sha256, "execution_journal_sha256": snapshot.state_hash,
        "policy_config_sha256": config_hash,
        "source_prediction_hashes": {chosen_path.name: file_digest(chosen_path), reference_path.name: file_digest(reference_path)},
        "limitations": "historical snapshot, not a current market decision; forecast timing, freshness and separate risk authorization are required for a future paper coordinator",
    }

    def verify():
        snapshot.verify()
        if file_digest(config_path) != config_hash:
            raise ValueError("Decision policy config changed during preview")

    verify()
    staging = Path(tempfile.mkdtemp(prefix=".decision-preview-staging-", dir=snapshot.output))
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
    print("Expected return forecast: unavailable; independent risk authorization has not been requested")
    LOGGER.info("Saved decision preview to %s; no orders or model fitting", output)
    return output, report
