"""Small command-line entry point for collection and local analytical queries."""

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path

import duckdb

from local_ai_trader.data.clean import clean_candles
from local_ai_trader.data.download import download_candles
from local_ai_trader.data.storage import write_json, write_parquet
from local_ai_trader.settings import load_settings, validate_symbol

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
    inspect = commands.add_parser("inspect", help="Query a Parquet dataset using DuckDB")
    inspect.add_argument("path", type=Path)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if arguments.command == "download":
            collect(arguments.config, arguments.start, arguments.end, arguments.symbols)
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
    except (ValueError, OSError, RuntimeError, KeyError, duckdb.Error) as error:
        LOGGER.error("%s", error)
        return 1
