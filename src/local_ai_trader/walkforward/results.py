"""Fixed-policy fold backtests and descriptive aggregation of reset accounts."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable

import numpy as np
import pandas as pd

from local_ai_trader.backtest.engine import simulate
from local_ai_trader.data.storage import write_json, write_parquet
from local_ai_trader.models.baseline import CLASS_NAMES
from local_ai_trader.models.evaluate import evaluate_probabilities
from local_ai_trader.models.inference import load_frozen_partition
from local_ai_trader.walkforward.protocol import Protocol

TRUTH_COLUMNS = ["timestamp", "available_at", "target_available_at", "symbol", "exchange", "target_class", "target_return"]


def read_predictions(path: Path, truth: pd.DataFrame) -> pd.DataFrame:
    """Saved signals must align exactly with the frozen fold's timed outcomes."""
    frame = pd.read_parquet(path)
    try:
        pd.testing.assert_frame_equal(frame[TRUTH_COLUMNS].reset_index(drop=True), truth[TRUTH_COLUMNS].reset_index(drop=True))
    except (AssertionError, KeyError) as error:
        raise ValueError("Saved evaluation predictions do not match frozen fold outcomes") from error
    return frame


def publish_fold_backtest(
    protocol: Protocol, index: int, fold_output: Path, runs: dict[str, Path],
    verify: Callable[[], None],
) -> dict:
    """Use frozen test signals and original opens; no refitting or tuning."""
    settings = protocol.settings
    snapshot = load_frozen_partition(runs["baseline"], "test", protocol.fold_directory(index))
    if not (snapshot.frame["target_available_at"] < pd.Timestamp(protocol.plan["folds"][index - 1]["evaluation_end_exclusive"])).all():
        raise ValueError("Fold outcomes cross the frozen evaluation end")
    candles = snapshot.frame[["timestamp", "available_at", "open", "close", "symbol", "exchange"]].copy()
    results = {}
    for family, directory in runs.items():
        names = ("naive", "logistic") if family == "baseline" else (protocol.plan["model_family"],)
        for model in names:
            for variant in ("raw", "temperature"):
                filename = model if variant == "raw" else f"{model}_temperature"
                predicted = read_predictions(directory / "test_evaluation" / f"{filename}.parquet", snapshot.frame)
                probabilities = predicted[[f"p_{label}" for label in CLASS_NAMES]].to_numpy(dtype=float)
                results[f"{model}_{variant}"] = simulate(candles, probabilities, settings.backtest, settings.candle_seconds, settings.horizon_steps)
    for strategy in ("cash", "buy_and_hold"):
        results[strategy] = simulate(candles, probabilities, settings.backtest, settings.candle_seconds, settings.horizon_steps, strategy)
    report = {
        "schema_version": 1, "research_only": True, "fold": index,
        "plan_sha256": protocol.sha256, "test_parquet_sha256": snapshot.parquet_hash,
        "settings": asdict(settings.backtest), "model_or_calibration_refitted": False,
        "account_reset_each_fold": True,
        "evaluation_start": protocol.plan["folds"][index - 1]["test_first_prediction_time"],
        "evaluation_end_exclusive": protocol.plan["folds"][index - 1]["evaluation_end_exclusive"],
        "simulation_start_open": candles.timestamp.iloc[0].isoformat(),
        "simulation_terminal_open": candles.timestamp.iloc[-1].isoformat(),
        "assumptions": {
            "positions": "one_spot_long_no_leverage_or_overlap",
            "fills": "open_after_frozen_latency_bars_half_spread_and_adverse_slippage",
            "fees": "frozen_fee_bps_each_side_on_fill_notional",
            "horizon_exit": "saved_horizon_bars_since_execution",
            "terminal_exit": "predefined_last_selected_open_with_costs",
            "risk": "frozen_entry_exposure_and_drawdown_checks_at_opens",
            "benchmarks": "same_allocation_costs_and_dates_cash_and_buy_and_hold",
            "limitations": "no_intrabar_stop_order_book_capacity_partial_fill_tax_or_interest_model",
        },
        "metrics": {name: result.metrics for name, result in results.items()},
    }
    output = fold_output / "backtest"
    if output.exists():
        raise ValueError("Fold backtest already exists; resume verifies the saved bundle")
    verify()
    staging = Path(tempfile.mkdtemp(prefix=".backtest-staging-", dir=fold_output))
    try:
        write_json(staging / "backtest.json", report)
        for name, result in results.items():
            for ledger in ("equity", "fills", "trades", "decisions"):
                write_parquet(staging / f"{name}_{ledger}.parquet", getattr(result, ledger))
        verify()
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report


def publish_summary(protocol: Protocol, output: Path, runs: dict[int, dict[str, Path]], verify: Callable[[], None]) -> dict:
    """Pool prediction scores; summarize independent reset-account simulations."""
    pooled: dict[str, list[pd.DataFrame]] = {}
    fold_rows = []
    financial: dict[str, list[dict]] = {}
    for index, families in runs.items():
        fold_directory = output / f"fold_{index:02d}"
        with (fold_directory / "backtest/backtest.json").open(encoding="utf-8") as source:
            backtest = json.load(source)
        for name, scores in backtest["metrics"].items():
            financial.setdefault(name, []).append(scores)
        for family, directory in families.items():
            models = ("naive", "logistic") if family == "baseline" else (protocol.plan["model_family"],)
            with (directory / "test_evaluation/metrics.json").open(encoding="utf-8") as stream:
                evaluated = json.load(stream)
            for model in models:
                for variant in ("raw", "temperature"):
                    filename = model if variant == "raw" else f"{model}_temperature"
                    name = f"{model}_{variant}"
                    frame = pd.read_parquet(directory / "test_evaluation" / f"{filename}.parquet")
                    frame["fold"] = index
                    pooled.setdefault(name, []).append(frame)
                    fold_rows.append({
                        "fold": index, "model": model, "variant": variant,
                        **{key: evaluated["metrics"][filename][key] for key in ("accuracy", "f1_macro", "log_loss", "brier_score", "ece")},
                        **{key: backtest["metrics"][name][key] for key in ("net_return", "max_drawdown", "number_of_trades", "fees_paid")},
                    })
    pooled_frames, pooled_metrics = {}, {}
    for name, tables in pooled.items():
        frame = pd.concat(tables, ignore_index=True)
        if frame.available_at.duplicated().any() or not frame.available_at.is_monotonic_increasing:
            raise ValueError("Pooled evaluation predictions overlap or are unordered")
        metrics = evaluate_probabilities(frame.target_class, frame[[f"p_{label}" for label in CLASS_NAMES]].to_numpy(dtype=float), protocol.settings.calibration_bins)
        metrics["calibration_status"] = "temperature_scaled" if name.endswith("_temperature") else "uncalibrated"
        pooled_frames[name], pooled_metrics[name] = frame, metrics
    financial_summary = {}
    for name, scores in financial.items():
        returns = [row["net_return"] for row in scores]
        financial_summary[name] = {
            "number_of_folds": len(scores), "mean_fold_net_return": float(np.mean(returns)),
            "best_fold_net_return": max(returns), "worst_fold_net_return": min(returns),
            "worst_fold_max_drawdown": max(row["max_drawdown"] for row in scores),
            "total_trades": sum(row["number_of_trades"] for row in scores),
            "sum_net_pnl_reset_accounts": sum(row["final_equity"] - row["initial_cash"] for row in scores),
            "sum_fees_paid": sum(row["fees_paid"] for row in scores),
            "sum_price_impact_cost": sum(row["spread_and_slippage_cost"] for row in scores),
        }
    report = {
        "schema_version": 1, "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "research_only": True, "plan_sha256": protocol.sha256,
        "symbol": protocol.plan["symbol"], "selected_family": protocol.plan["model_family"],
        "folds": len(runs), "evaluation_rows": len(next(iter(pooled_frames.values()))),
        "all_models_calibrated_before_first_evaluation": True,
        "settings_source": "immutable_plan_not_current_config",
        "fold_metrics": fold_rows, "pooled_prediction_metrics": pooled_metrics,
        "financial_summary": financial_summary,
        "aggregation": "prediction_scores_pooled_account_returns_descriptive_each_fold_resets_cash_and_risk",
        "independent_sample_assumption": False, "confidence_intervals_computed": False,
        "fresh_final_strategy_period_required": True,
    }
    verify()
    staging = Path(tempfile.mkdtemp(prefix=".summary-staging-", dir=output))
    try:
        write_json(staging / "summary.json", report)
        write_parquet(staging / "fold_metrics.parquet", pd.DataFrame(fold_rows))
        for name, frame in pooled_frames.items():
            write_parquet(staging / f"{name}_predictions.parquet", frame)
        verify()
        os.rename(staging, output / "summary")
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report


def print_summary(report: dict) -> None:
    chosen = report["selected_family"]
    print("Walk-forward historical research: selected temperature-scaled model by fold")
    print(pd.DataFrame([
        {key: row[key] for key in ("fold", "log_loss", "brier_score", "ece", "net_return", "max_drawdown", "number_of_trades")}
        for row in report["fold_metrics"] if row["model"] == chosen and row["variant"] == "temperature"
    ]).to_string(index=False))
    print("Pooled prediction scores and mean returns across reset fold accounts:")
    print(pd.DataFrame([
        {"model": name, "log_loss": scores["log_loss"], "brier_score": scores["brier_score"], "ece": scores["ece"], "mean_fold_return": report["financial_summary"][name]["mean_fold_net_return"], "trades": report["financial_summary"][name]["total_trades"]}
        for name, scores in report["pooled_prediction_metrics"].items()
    ]).to_string(index=False))
