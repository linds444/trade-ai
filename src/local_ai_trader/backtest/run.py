"""Publish audited validation-assessment simulations; test access is unsupported."""

from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from local_ai_trader.backtest.engine import simulate
from local_ai_trader.calibration.run import CALIBRATION_FILENAME, apply_saved_calibration, validation_regions
from local_ai_trader.data.storage import write_json, write_parquet
from local_ai_trader.models.inference import load_frozen_partition, predict_frozen_models, verify_frozen_inputs
from local_ai_trader.models.train import file_digest
from local_ai_trader.settings import BacktestSettings, load_backtest_settings

LOGGER = logging.getLogger(__name__)


def create_backtest(
    run_directory: Path, model: str, variant: str, config: BacktestSettings,
    split_directory: Path | None = None,
) -> tuple[Path, dict]:
    """Research trading assumptions on the same later-validation assessment only."""
    config = load_backtest_settings(asdict(config))
    if variant not in ("raw", "temperature") or model not in ("naive", "logistic", "xgboost", "mlp"):
        raise ValueError("Select a supported model and raw/temperature variant")
    calibration_path = run_directory / "calibration" / CALIBRATION_FILENAME
    if not calibration_path.is_file():
        raise ValueError("Saved calibration required to identify the validation assessment period")
    calibration_hash = file_digest(calibration_path)
    with calibration_path.open(encoding="utf-8") as source:
        calibration = json.load(source)
    key = hashlib.sha256(json.dumps({"model": model, "variant": variant, "settings": asdict(config), "calibration_sha256": calibration_hash}, sort_keys=True).encode()).hexdigest()[:12]
    output = run_directory / "backtests" / f"validation_{model}_{variant}_{key}"
    if output.exists():
        raise ValueError(f"Backtest already saved for these settings; inspect {output}")
    snapshot = load_frozen_partition(run_directory, "validation", split_directory)
    raw, artifacts = predict_frozen_models(snapshot.checkpoint, snapshot.frame, run_directory)
    if model not in raw:
        raise ValueError(f"Model {model} is not present in this saved experiment")
    adjusted, restored_hash = apply_saved_calibration(run_directory, snapshot, raw, artifacts)
    if restored_hash != calibration_hash:
        raise ValueError("Calibration changed during backtest; no results published")
    regions, region_report = validation_regions(snapshot.frame, calibration["parameters"]["fit_fraction"])
    if region_report != calibration["regions"]:
        raise ValueError("Assessment period differs from the frozen calibration metadata")
    assessment = regions["assessment"]
    # Future labels, forward returns and features never enter policy or accounting.
    candles = assessment[["timestamp", "available_at", "open", "close", "symbol", "exchange"]].reset_index(drop=True)
    probabilities = (raw if variant == "raw" else adjusted)[model][assessment.index]
    settings = snapshot.checkpoint["training_metadata"]["settings"]
    results = {
        strategy: simulate(candles, probabilities, config, settings["candle_seconds"], settings["horizon_steps"], strategy)
        for strategy in ("policy", "cash", "buy_and_hold")
    }
    report = {
        "schema_version": 1, "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "research_only": True, "partition": "later_validation_assessment",
        "assessment_previously_used_for_model_selection": True,
        "classification_test_evaluation_already_exists": (run_directory / "test_evaluation").exists(),
        "test_data_opened": False, "model_or_calibration_refitted": False,
        "model": model, "variant": variant,
        "run_id": snapshot.checkpoint["training_metadata"]["run_id"],
        "symbol": candles["symbol"].iloc[0], "exchange": candles["exchange"].iloc[0],
        "settings": asdict(config), "candle_seconds": settings["candle_seconds"],
        "maximum_holding_bars": settings["horizon_steps"],
        "assessment_rows": len(candles),
        "simulation_start": candles["timestamp"].iloc[0].isoformat(),
        "simulation_end": candles["timestamp"].iloc[-1].isoformat(),
        "base_checkpoint_sha256": snapshot.checkpoint_hash,
        "calibration_checkpoint_sha256": calibration_hash,
        "model_artifact_hashes": artifacts,
        "source_input_hashes": snapshot.checkpoint["training_metadata"]["input_hashes"],
        "source_split_directory": str(snapshot.directory.resolve()),
        "assumptions": {
            "market": "single_asset_spot_long_only_no_leverage",
            "sizing": "allocation_fraction_of_cash_once_at_entry_no_rebalancing",
            "signal_time": "candle_available_at_close",
            "fill_time": "open_after_latency_bars_full_bars_following_signal",
            "buy_fill": "open * (1 + (spread_bps/2 + slippage_bps)/10000)",
            "sell_fill": "open * (1 - (spread_bps/2 + slippage_bps)/10000)",
            "fees": "fee_bps_on_executed_notional_each_side",
            "terminal_exit": "forced_liquidation_at_predefined_last_open_with_costs",
            "equity_and_risk_marks": "opens_only_not_intrabar",
            "drawdown_gate": "block_new_entries_at_or_above_limit_exits_remain_allowed",
            "buy_and_hold": "same_allocation_and_costs_first_possible_fill_then_final_open_exit",
            "turnover": "sum_executed_buy_and_sell_notional_divided_by_initial_cash",
            "sharpe_sortino": "open_to_open_returns_365_day_annualization_zero_risk_free",
            "gross_pnl": "reference_open_price_attribution_for_actual_quantities_not_a_cost_free_rerun",
            "limitations": ["no_order_book_or_liquidity_capacity_model", "no_intrabar_stops", "no_partial_fills_or_minimum_order_sizes", "no_interest_or_tax", "probability_thresholds_do_not_estimate_expected_return_after_costs", "short_validation_period_annualized_ratios_are_descriptive"],
        },
        "metrics": {strategy: result.metrics for strategy, result in results.items()},
    }
    verify_frozen_inputs(snapshot, run_directory, artifacts)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".backtest-staging-", dir=output.parent))
    try:
        write_json(staging / "backtest.json", report)
        for strategy, result in results.items():
            for name in ("equity", "fills", "trades", "decisions"):
                write_parquet(staging / f"{strategy}_{name}.parquet", getattr(result, name))
        verify_frozen_inputs(snapshot, run_directory, artifacts)
        if file_digest(calibration_path) != calibration_hash:
            raise ValueError("Calibration changed during backtest; no results published")
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    LOGGER.info("Simulated %s later-validation candles; test partition not opened", len(candles))
    print("Validation research backtest (costs included; not unseen strategy performance):")
    print(pd.DataFrame([
        {"strategy": name, **{key: scores[key] for key in ("net_return", "max_drawdown", "number_of_trades", "fees_paid", "turnover")}}
        for name, scores in report["metrics"].items()
    ]).to_string(index=False))
    LOGGER.info("Saved backtest to %s", output)
    return output, report
