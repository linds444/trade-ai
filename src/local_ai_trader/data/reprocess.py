"""Build a strictly complete subrange from an immutable saved API response."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

from local_ai_trader.data.clean import clean_candles
from local_ai_trader.data.storage import write_json, write_parquet
from local_ai_trader.settings import Settings, validate_symbol

LOGGER = logging.getLogger(__name__)


def reprocess_download(source: Path, start: int, end: int, settings: Settings) -> tuple[Path, dict]:
    """Select [start, end) without network requests, gap filling or source edits.

    Original raw responses remain the source of truth. The new quality report
    records their path, byte hash and original bounds. An incomplete selection
    saves gap evidence only; a complete selection saves a new candle snapshot.
    """
    if not source.is_file():
        raise ValueError(f"Saved raw download not found: {source}")
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    payload = json.loads(data)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or payload.get("exchange") != "coinbase":
        raise ValueError("Require a schema-1 saved Coinbase raw download")
    symbol = payload.get("symbol")
    if not isinstance(symbol, str):
        raise ValueError("Raw download is missing its symbol")
    validate_symbol(symbol)
    seconds = settings.candle_seconds
    if type(payload.get("granularity_seconds")) is not int or payload["granularity_seconds"] != seconds:
        raise ValueError("Raw candle interval differs from current configuration")
    original_start, original_end = payload.get("start_inclusive"), payload.get("end_exclusive")
    if type(original_start) is not int or type(original_end) is not int or original_start >= original_end or original_start % seconds or original_end % seconds:
        raise ValueError("Raw download has invalid original bounds")
    if type(start) is not int or type(end) is not int or not original_start <= start < end <= original_end or start % seconds or end % seconds:
        raise ValueError("Selected bounds must be aligned and contained in the saved download")
    if end > int(datetime.now(timezone.utc).timestamp()) // seconds * seconds:
        raise ValueError("Selected range contains unfinished or future candles")
    pages = payload.get("pages")
    if not isinstance(pages, list) or not pages or any(not isinstance(page, dict) or not isinstance(page.get("candles"), list) for page in pages):
        raise ValueError("Raw download must contain pages of candle rows")
    stem = f"coinbase_{symbol}_{seconds}s_{start}_{end}"
    destination = settings.processed_dir / f"{stem}.parquet"
    quality_path = settings.processed_dir / f"{stem}.quality.json"
    if destination.exists() or quality_path.exists() or (settings.raw_dir / f"{stem}.json").exists():
        raise ValueError("Output already exists for this range; saved snapshots cannot be overwritten")
    rows = [row for page in pages for row in page["candles"]]
    frame, quality = clean_candles(rows, start, end, seconds)
    report = {
        "schema_version": 1, "exchange": "coinbase", "symbol": symbol,
        "granularity_seconds": seconds, "start_inclusive": start, "end_exclusive": end,
        "prediction_horizon_steps": settings.horizon_steps,
        "raw_path": str(source.resolve()), "raw_sha256": digest,
        "source_start_inclusive": original_start, "source_end_exclusive": original_end,
        "source_downloaded_at": payload.get("downloaded_at"),
        "reprocessed_at": datetime.now(timezone.utc).isoformat(),
        "processing": "complete_subrange_of_saved_raw_no_gap_filling", **asdict(quality),
    }

    def verify_source() -> None:
        with source.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError("Raw source changed during reprocessing; no results published")

    verify_source()
    if quality.missing_count:
        write_json(quality_path, report)
        raise ValueError(f"{symbol}: {quality.missing_count} missing candles in selected range. See {quality_path}; no processed Parquet written")
    frame["symbol"], frame["exchange"] = symbol, "coinbase"
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".reprocess-staging-", dir=destination.parent))
    try:
        write_json(staging / quality_path.name, report)
        write_parquet(staging / destination.name, frame)
        verify_source()
        if destination.exists() or quality_path.exists():
            raise ValueError("Output appeared during reprocessing; no snapshot overwritten")
        os.rename(staging / quality_path.name, quality_path)
        os.rename(staging / destination.name, destination)
    finally:
        shutil.rmtree(staging)
    LOGGER.info("Saved %s rows to %s; no missing intervals; reused saved raw responses", len(frame), destination)
    return destination, report
