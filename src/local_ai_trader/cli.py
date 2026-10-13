"""Small command-line entry point for collection and local analytical queries."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import logging
from pathlib import Path
import sqlite3

import duckdb

from local_ai_trader.data.clean import clean_candles
from local_ai_trader.data.download import download_candles
from local_ai_trader.data.dataset import create_target_dataset
from local_ai_trader.data.split import create_split_dataset
from local_ai_trader.features.build_features import create_feature_dataset
from local_ai_trader.data.storage import write_json, write_parquet
from local_ai_trader.settings import load_settings, validate_symbol
from local_ai_trader.models.train import train_baseline_experiment
from local_ai_trader.models.heldout import evaluate_heldout

LOGGER = logging.getLogger(__name__)


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Include a timezone in dates, for example 2026-01-01T00:00:00Z")
    return parsed.astimezone(timezone.utc)


def range_bounds(start: str | None, end: str | None, seconds: int, days: int) -> tuple[int, int]:
    if bool(start) != bool(end):
        raise ValueError("Specify both --start and --end, or neither")
    now = datetime.now(timezone.utc)
    closed_boundary = int(now.timestamp()) // seconds * seconds
    if start is None:
        return closed_boundary - days * 86400, closed_boundary
    first, last = parse_utc(start), parse_utc(end)
    if first.microsecond or last.microsecond:
        raise ValueError("Date bounds must not include fractional seconds")
    first_epoch, last_epoch = int(first.timestamp()), int(last.timestamp())
    if first_epoch >= last_epoch or first_epoch % seconds or last_epoch % seconds:
        raise ValueError("Date bounds must be ordered and aligned to the candle interval")
    if last_epoch > closed_boundary:
        raise ValueError("The end includes an unfinished or future candle")
    return first_epoch, last_epoch


def collect(config: Path, start: str | None, end: str | None, symbols: list[str] | None) -> None:
    settings = load_settings(config)
    first, last = range_bounds(start, end, settings.candle_seconds, settings.lookback_days)
    selected = settings.symbols if symbols is None else tuple(validate_symbol(s) for s in symbols)
    for symbol in selected:
        stem = f"coinbase_{symbol}_{settings.candle_seconds}s_{first}_{last}"
        raw_path = settings.raw_dir / f"{stem}.json"
        parquet_path = settings.processed_dir / f"{stem}.parquet"
        quality_path = settings.processed_dir / f"{stem}.quality.json"
        # Exact ranges are immutable snapshots. Avoid stale Parquet after a failed refresh.
        if any(path.exists() for path in (raw_path, parquet_path, quality_path)):
            raise ValueError(f"Output already exists for {symbol} and this range; choose a new range or move existing files")
        payload = download_candles(symbol, first, last, settings)
        write_json(raw_path, payload)
        rows = [row for page in payload["pages"] for row in page["candles"]]
        frame, report = clean_candles(rows, first, last, settings.candle_seconds)
        write_json(quality_path, {
            "exchange": "coinbase", "symbol": symbol,
            "granularity_seconds": settings.candle_seconds,
            "start_inclusive": first, "end_exclusive": last,
            "prediction_horizon_steps": settings.horizon_steps,
            "raw_path": str(raw_path), **asdict(report),
        })
        if report.missing_count:
            raise ValueError(
                f"{symbol}: {report.missing_count} missing candles. See {quality_path}. "
                "Raw data saved; no processed Parquet written."
            )
        frame["symbol"] = symbol
        frame["exchange"] = "coinbase"
        write_parquet(parquet_path, frame)
        LOGGER.info("Saved %s rows to %s; no missing intervals", len(frame), parquet_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Save public candles and validated Parquet")
    download.add_argument("--config", type=Path, default=Path("config/settings.toml"))
    download.add_argument("--start", help="Inclusive, timezone-aware and interval-aligned")
    download.add_argument("--end", help="Exclusive, timezone-aware and interval-aligned")
    download.add_argument("--symbols", nargs="+", help="Override configured Coinbase products")
    reprocess = commands.add_parser("reprocess", help="Build a complete subrange from saved raw responses without downloading")
    reprocess.add_argument("path", type=Path, help="Original raw download JSON")
    reprocess.add_argument("--start", required=True, help="Inclusive, timezone-aware and interval-aligned")
    reprocess.add_argument("--end", required=True, help="Exclusive, timezone-aware and interval-aligned")
    reprocess.add_argument("--config", type=Path, default=Path("config/settings.toml"))
    walkforward = commands.add_parser("walkforward-plan", help="Freeze rolling calendar folds before model fitting")
    walkforward.add_argument("path", type=Path, help="Feature Parquet with metadata")
    walkforward.add_argument("--model", required=True, choices=("mlp", "xgboost"), help="Selected family; naive/logistic controls are recorded too")
    walkforward.add_argument("--config", type=Path, default=Path("config/settings.toml"))
    walkforward_run = commands.add_parser("walkforward-run", help="Execute frozen folds with fresh models, calibration and cost-aware evaluation")
    walkforward_run.add_argument("path", type=Path, help="Directory containing immutable plan.json")
    walkforward_run.add_argument("--resume", action="store_true", help="Verify and reuse completed bundles after interruption")
    uncertainty = commands.add_parser("uncertainty", help="Audit entropy and model disagreement in saved walk-forward predictions")
    uncertainty.add_argument("path", type=Path, help="Directory containing a completed immutable walk-forward plan")
    decision_preview = commands.add_parser("decision-preview", help="Explain a cost-aware entry candidate using saved historical probabilities")
    decision_preview.add_argument("path", type=Path, help="Completed walk-forward plan directory")
    decision_preview.add_argument("--config", type=Path, default=Path("config/settings.toml"), help="Read the separate decision policy table; costs remain bound to the plan")
    decision_preview.add_argument("--return-run", type=Path, help="Saved execution-aligned return-baseline run; performs one audited point inference")
    decision_preview.add_argument("--return-model", choices=("mean", "ridge"), default="ridge", help="Return model within the saved run (default: ridge)")
    paper_risk = commands.add_parser("paper-risk-demo", help="Exercise independent entry checks using synthetic inputs; no orders")
    paper_risk.add_argument("--config", type=Path, default=Path("config/settings.toml"))
    paper_demo = commands.add_parser("paper-demo", help="Persist a synthetic paper buy/exit and verify restart; no broker connection")
    paper_demo.add_argument("--config", type=Path, default=Path("config/settings.toml"))
    paper_status = commands.add_parser("paper-status", help="Audit an existing paper SQLite account without fetching prices")
    paper_status.add_argument("path", type=Path, help="Path to account.sqlite")
    hyperliquid = commands.add_parser("hyperliquid-feed", help="Validate public Hyperliquid mainnet books and completed candles; no trading")
    hyperliquid.add_argument("--config", type=Path, default=Path("config/hyperliquid.toml"))
    hyperliquid.add_argument("--watch", action="store_true", help="Save each new completed 5-minute window until Ctrl+C")
    perp_demo = commands.add_parser("perpetual-paper-demo", help="Synthetic persisted USDC margin/funding workflow; no strategy or orders")
    perp_demo.add_argument("--config", type=Path, default=Path("config/hyperliquid.toml"))
    perp_status = commands.add_parser("perpetual-paper-status", help="Audit recorded USDC perpetual balances without fetching prices")
    perp_status.add_argument("path", type=Path)
    inspect = commands.add_parser("inspect", help="Query a Parquet dataset using DuckDB")
    inspect.add_argument("path", type=Path)
    targets = commands.add_parser("targets", help="Build causal-time-labelled forward-return targets")
    targets.add_argument("path", type=Path, help="Original candle Parquet file")
    targets.add_argument("--config", type=Path, default=Path("config/settings.toml"))
    execution_targets = commands.add_parser("execution-targets", help="Build delayed-entry open-to-open labels for a return forecaster")
    execution_targets.add_argument("path", type=Path, help="Original complete candle Parquet")
    execution_targets.add_argument("--config", type=Path, default=Path("config/settings.toml"))
    execution_features = commands.add_parser("execution-features", help="Build causal features from an audited execution target bundle")
    execution_features.add_argument("path", type=Path, help="Directory containing labels.parquet and targets.json")
    execution_features.add_argument("--config", type=Path, default=Path("config/settings.toml"))
    execution_split = commands.add_parser("execution-split", help="Create chronological return partitions purged by execution label availability")
    execution_split.add_argument("path", type=Path, help="Directory containing features.parquet and features.json")
    execution_split.add_argument("--config", type=Path, default=Path("config/settings.toml"))
    return_baseline = commands.add_parser("return-baseline", help="Fit training-mean and scaled Ridge return forecasts; assess validation only")
    return_baseline.add_argument("path", type=Path, help="Directory containing execution splits and split.json")
    return_baseline.add_argument("--config", type=Path, default=Path("config/settings.toml"))
    features = commands.add_parser("features", help="Build trailing features from a labelled dataset")
    features.add_argument("path", type=Path, help="Labelled target Parquet file")
    features.add_argument("--config", type=Path, default=Path("config/settings.toml"))
    split = commands.add_parser("split", help="Create chronological partitions with overlapping labels purged")
    split.add_argument("path", type=Path, help="Feature Parquet file")
    split.add_argument("--config", type=Path, default=Path("config/settings.toml"))
    baseline = commands.add_parser("baseline", help="Fit naive/logistic models; evaluate validation only")
    baseline.add_argument("path", type=Path, help="Directory containing purged splits and split.json")
    baseline.add_argument("--config", type=Path, default=Path("config/settings.toml"))
    boosting = commands.add_parser("boosting", help="Fit a fixed XGBoost model; evaluate validation only")
    boosting.add_argument("path", type=Path, help="Directory containing purged splits and split.json")
    boosting.add_argument("--config", type=Path, default=Path("config/settings.toml"))
    mlp = commands.add_parser("mlp", help="Train a small PyTorch MLP; evaluate validation only")
    mlp.add_argument("path", type=Path, help="Directory containing purged splits and split.json")
    mlp.add_argument("--config", type=Path, default=Path("config/settings.toml"))
    calibrate = commands.add_parser("calibrate", help="Fit temperature on earlier validation; assess later validation")
    calibrate.add_argument("path", type=Path, help="Saved model experiment directory")
    calibrate.add_argument("--config", type=Path, default=Path("config/settings.toml"))
    calibrate.add_argument("--splits", type=Path, help="Override split location while verifying original input hashes")
    evaluate = commands.add_parser("evaluate", help="Evaluate a frozen baseline run on its held-out test data")
    evaluate.add_argument("path", type=Path, help="Saved baseline experiment directory")
    evaluate.add_argument("--splits", type=Path, help="Override split location while verifying original input hashes")
    evaluate.add_argument("--calibrated", action="store_true", help="Report raw and saved-temperature predictions without fitting")
    backtest = commands.add_parser("backtest", help="Simulate a frozen model on later validation only, with costs")
    backtest.add_argument("path", type=Path, help="Saved experiment with a calibration bundle")
    backtest.add_argument("--model", required=True, choices=("naive", "logistic", "xgboost", "mlp"))
    backtest.add_argument("--variant", choices=("raw", "temperature"), default="temperature")
    backtest.add_argument("--config", type=Path, default=Path("config/settings.toml"))
    backtest.add_argument("--splits", type=Path, help="Override split location while verifying original input hashes")
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if arguments.command == "download":
            collect(arguments.config, arguments.start, arguments.end, arguments.symbols)
        elif arguments.command == "reprocess":
            from local_ai_trader.data.reprocess import reprocess_download

            settings = load_settings(arguments.config)
            first, last = range_bounds(arguments.start, arguments.end, settings.candle_seconds, settings.lookback_days)
            reprocess_download(arguments.path, first, last, settings)
        elif arguments.command == "walkforward-plan":
            from local_ai_trader.walkforward.plan import create_walkforward_plan

            create_walkforward_plan(arguments.path, load_settings(arguments.config), arguments.model)
        elif arguments.command == "walkforward-run":
            from local_ai_trader.walkforward.run import run_walkforward

            run_walkforward(arguments.path, resume=arguments.resume)
        elif arguments.command == "uncertainty":
            from local_ai_trader.uncertainty.report import create_uncertainty_report

            create_uncertainty_report(arguments.path)
        elif arguments.command == "decision-preview":
            from local_ai_trader.decision.preview import create_decision_preview

            if arguments.return_run is None and arguments.return_model == "ridge":
                create_decision_preview(arguments.path, arguments.config)
            else:
                create_decision_preview(arguments.path, arguments.config, return_run=arguments.return_run, return_model=arguments.return_model)
        elif arguments.command == "targets":
            create_target_dataset(arguments.path, load_settings(arguments.config))
        elif arguments.command == "paper-risk-demo":
            from local_ai_trader.risk.demo import paper_risk_demo

            paper_risk_demo(arguments.config)
        elif arguments.command == "paper-demo":
            from local_ai_trader.paper.demo import paper_demo

            paper_demo(arguments.config)
        elif arguments.command == "paper-status":
            from local_ai_trader.paper.demo import paper_status

            paper_status(arguments.path)
        elif arguments.command == "hyperliquid-feed":
            from local_ai_trader.hyperliquid.feed import run_hyperliquid_feed

            run_hyperliquid_feed(arguments.config, watch=arguments.watch)
        elif arguments.command == "perpetual-paper-demo":
            from local_ai_trader.paper.perpetual_demo import perpetual_demo

            perpetual_demo(arguments.config)
        elif arguments.command == "perpetual-paper-status":
            from local_ai_trader.paper.perpetual_demo import perpetual_status

            perpetual_status(arguments.path)
        elif arguments.command == "execution-targets":
            from local_ai_trader.forecast.targets import create_execution_target_dataset

            create_execution_target_dataset(arguments.path, load_settings(arguments.config))
        elif arguments.command == "features":
            create_feature_dataset(arguments.path, load_settings(arguments.config))
        elif arguments.command == "execution-features":
            from local_ai_trader.forecast.features import create_execution_feature_dataset

            create_execution_feature_dataset(arguments.path, load_settings(arguments.config))
        elif arguments.command == "split":
            create_split_dataset(arguments.path, load_settings(arguments.config))
        elif arguments.command == "execution-split":
            from local_ai_trader.forecast.split import create_execution_split_dataset

            create_execution_split_dataset(arguments.path, load_settings(arguments.config))
        elif arguments.command == "baseline":
            train_baseline_experiment(arguments.path, load_settings(arguments.config))
        elif arguments.command == "return-baseline":
            from local_ai_trader.forecast.baseline import load_return_ridge_alpha
            from local_ai_trader.forecast.train import train_return_baseline_experiment

            train_return_baseline_experiment(arguments.path, load_settings(arguments.config), load_return_ridge_alpha(arguments.config))
        elif arguments.command == "boosting":
            from local_ai_trader.models.boosting import train_boosting_experiment

            train_boosting_experiment(arguments.path, load_settings(arguments.config))
        elif arguments.command == "mlp":
            try:
                from local_ai_trader.models.mlp import train_mlp_experiment
            except ModuleNotFoundError as error:
                if error.name == "torch":
                    raise ValueError("PyTorch is required for mlp; use the verified CUDA environment") from error
                raise
            train_mlp_experiment(arguments.path, load_settings(arguments.config))
        elif arguments.command == "backtest":
            from local_ai_trader.backtest.run import create_backtest

            create_backtest(arguments.path, arguments.model, arguments.variant, load_settings(arguments.config).backtest, arguments.splits)
        elif arguments.command == "calibrate":
            from local_ai_trader.calibration.run import create_calibration

            create_calibration(arguments.path, load_settings(arguments.config).calibration, arguments.splits)
        elif arguments.command == "evaluate":
            evaluate_heldout(arguments.path, arguments.splits, calibrated=arguments.calibrated)
        else:
            if not arguments.path.is_file():
                raise ValueError(f"Parquet file not found: {arguments.path}")
            with duckdb.connect() as database:
                # TIMESTAMPTZ display otherwise follows the host's local timezone.
                database.execute("SET TimeZone = 'UTC'")
                summary = database.execute(
                    "SELECT symbol, exchange, count(*) AS rows, min(timestamp) AS first_open, "
                    "max(timestamp) AS last_open FROM read_parquet(?) GROUP BY symbol, exchange",
                    [str(arguments.path.resolve())],
                ).fetchdf()
            print(summary.to_string(index=False))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, sqlite3.Error, duckdb.Error) as error:
        LOGGER.error("%s", error)
        return 1
