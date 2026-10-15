"""Venue-bound research preparation, with causal windows reset across every gap.

This prepares all three partitions but fits/scores no model. Training code must
read only its declared partitions rather than invoke the full reconstruction
audit below while fitting. All prices and volumes retain native contract units.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

import numpy as np
import pandas as pd

from local_ai_trader.data.storage import native_path, write_json, write_parquet
from local_ai_trader.features.build_features import FEATURE_COLUMNS, build_features
from local_ai_trader.hyperliquid.feed import (
    HyperliquidPublicClient, SECONDS, clean_hyperliquid_candles,
    load_hyperliquid_settings, parse_contexts,
)
from local_ai_trader.hyperliquid.markets import BY_COIN
from local_ai_trader.models.train import file_digest
from local_ai_trader.risk.paper import _utc

MAXIMUM_CLOSED_BARS = 4999  # The forming candle may consume the 5000th API slot.
LATENCY_BARS = 1
HOLDING_BARS = 6
EXIT_OFFSET = 1 + LATENCY_BARS + HOLDING_BARS
CLASS_THRESHOLD = 0.002
MINIMUM_PARTITION_ROWS = {"train": 128, "validation": 64, "test": 64}
LABEL_COLUMNS = (
    "execution_entry_at", "execution_exit_at", "execution_entry_reference_open",
    "execution_exit_reference_open", "execution_target_return",
    "execution_target_available_at", "target_class",
)


def research_protocol(settings, bars):
    if type(bars) is not int or not 128 <= bars <= MAXIMUM_CLOSED_BARS:
        raise ValueError("Hyperliquid research history requires 128–4999 closed bars")
    return {
        "requested_closed_bars": bars, "candle_seconds": SECONDS,
        "feature_names": list(FEATURE_COLUMNS), "feature_schema_version": 1,
        "momentum_steps": settings.momentum_steps, "rolling_window": settings.rolling_window,
        "latency_bars": LATENCY_BARS, "holding_bars": HOLDING_BARS,
        "entry_row_offset": 1 + LATENCY_BARS, "exit_row_offset": EXIT_OFFSET,
        "target_known_offset_from_prediction_bars": EXIT_OFFSET,
        "prediction_time": "available_at", "target_kind": "execution_open_to_open",
        "return_formula": "open[t+8] / open[t+2] - 1",
        "target_availability": "exit candle completion; 8 bars after prediction",
        "future_only_columns": list(LABEL_COLUMNS),
        "gross_return_class_threshold": CLASS_THRESHOLD, "class_order": ["down", "flat", "up"],
        "train_fraction": 0.6, "validation_fraction": 0.2, "test_fraction": 0.2,
        "minimum_partition_rows": MINIMUM_PARTITION_ROWS,
        "purge_rule": "execution_target_available_at < next_partition_first_available_at",
        "gap_policy": "reset warmup and discard label windows across every missing candle",
        "normalization": "none; eventual scaling must fit training rows only",
        "costs_in_labels": False, "funding_in_labels": False,
        "cost_note": "gross reference-open labels; forecasts still require current executable costs and funding",
        "prices_and_volume": "native contract units; kPEPE is not divided by 1000",
        "reserved_test_status": "prepared only; no model or strategy scored; independence from prior research not established",
    }


def build_research_rows(candles, settings):
    """Build features/labels separately inside contiguous candle segments."""
    warmup = max(settings.momentum_steps, settings.rolling_window)
    frames, segments = [], []
    nanoseconds = candles.timestamp.dt.as_unit("ns").astype("int64")
    group = nanoseconds.diff().ne(SECONDS * 1_000_000_000).cumsum()
    for segment_id, (_, segment) in enumerate(candles.groupby(group, sort=False), start=1):
        segment = segment.reset_index(drop=True)
        summary = {"segment_id": segment_id, "candle_rows": len(segment),
                   "first_open": segment.timestamp.iloc[0].isoformat(),
                   "last_open": segment.timestamp.iloc[-1].isoformat(),
                   "research_rows": max(0, len(segment) - warmup - EXIT_OFFSET)}
        segments.append(summary)
        if summary["research_rows"] == 0:
            continue
        # Future columns are attached only after the original feature builder.
        features = build_features(segment, SECONDS, settings.momentum_steps, settings.rolling_window)
        features = features.iloc[:-EXIT_OFFSET].copy().reset_index(drop=True)
        indices = np.arange(warmup, len(segment) - EXIT_OFFSET)
        entry = segment.iloc[indices + 1 + LATENCY_BARS].reset_index(drop=True)
        exit = segment.iloc[indices + EXIT_OFFSET].reset_index(drop=True)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            returns = exit.open.to_numpy() / entry.open.to_numpy() - 1
        if not np.isfinite(returns).all():
            raise ValueError("Hyperliquid return labels are nonfinite")
        features["execution_entry_at"] = entry.timestamp
        features["execution_exit_at"] = exit.timestamp
        features["execution_entry_reference_open"] = entry.open
        features["execution_exit_reference_open"] = exit.open
        features["execution_target_return"] = returns
        features["execution_target_available_at"] = exit.available_at
        features["target_class"] = np.where(returns > CLASS_THRESHOLD, "up", np.where(returns < -CLASS_THRESHOLD, "down", "flat"))
        features["segment_id"] = segment_id
        frames.append(features)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return result, segments


def split_research_rows(rows):
    """Chronological 60/20/20 split with availability purging, including equality."""
    cuts = (int(len(rows) * 0.6), int(len(rows) * 0.8))
    if rows.empty or not 0 < cuts[0] < cuts[1] < len(rows):
        return {}, {"status": "insufficient_history", "partitions": {}}
    boundaries = {"train": rows.available_at.iloc[cuts[0]], "validation": rows.available_at.iloc[cuts[1]]}
    raw = {"train": rows.iloc[:cuts[0]], "validation": rows.iloc[cuts[0]:cuts[1]], "test": rows.iloc[cuts[1]:]}
    partitions, summaries = {}, {}
    for name, frame in raw.items():
        keep = frame.execution_target_available_at < boundaries[name] if name in boundaries else pd.Series(True, index=frame.index)
        selected = frame.loc[keep].reset_index(drop=True)
        partitions[name] = selected
        summaries[name] = {"raw_rows": len(frame), "rows": len(selected), "purged_rows": int((~keep).sum()),
                           "first_prediction_time": selected.available_at.iloc[0].isoformat() if len(selected) else None,
                           "last_prediction_time": selected.available_at.iloc[-1].isoformat() if len(selected) else None,
                           "last_label_known_at": selected.execution_target_available_at.iloc[-1].isoformat() if len(selected) else None}
    ready = all(len(partitions[name]) >= count for name, count in MINIMUM_PARTITION_ROWS.items())
    return partitions, {"status": "ready" if ready else "insufficient_history", "partitions": summaries}


def derive_history(raw, settings, protocol):
    """Reconstruct quality, causal rows and partitions from complete raw evidence."""
    if protocol != research_protocol(settings, protocol.get("requested_closed_bars")):
        raise ValueError("Hyperliquid research protocol disagrees")
    start, end = raw["start_inclusive"], raw["end_exclusive"]
    created = _utc(datetime.fromisoformat(raw["captured_at"]))
    if type(start) is not int or type(end) is not int or start < 0 or end - start != protocol["requested_closed_bars"] * SECONDS or start % SECONDS or end % SECONDS or end > int(created.timestamp()) // SECONDS * SECONDS:
        raise ValueError("Invalid completed Hyperliquid history bounds")
    if set(raw["candles"]) != set(settings.coins) or set(raw["candles_received_at"]) != set(settings.coins):
        raise ValueError("Research archive requires every configured contract")
    dex_coins = {}
    for coin in settings.coins:
        dex_coins.setdefault(BY_COIN[coin].dex, []).append(coin)
    if set(raw["contexts"]) != set(dex_coins) or set(raw["contexts_received_at"]) != set(dex_coins):
        raise ValueError("Research archive DEX metadata disagrees")
    contracts = {}
    for dex, coins in dex_coins.items():
        metadata = raw["contexts"][dex]
        receipt = _utc(datetime.fromisoformat(raw["contexts_received_at"][dex]))
        if not datetime.fromtimestamp(end, timezone.utc) <= receipt <= created:
            raise ValueError("Invalid research metadata receipt")
        contracts.update(parse_contexts(metadata, tuple(coins), receipt))
        if type(metadata[0].get("collateralToken")) is not int or metadata[0]["collateralToken"] != 0:
            raise ValueError("Research contracts must verify USDC collateral")
    summary, artifacts = {}, {}
    for coin in settings.coins:
        receipt = _utc(datetime.fromisoformat(raw["candles_received_at"][coin]))
        if not datetime.fromtimestamp(end, timezone.utc) <= receipt <= created:
            raise ValueError("Invalid research candle receipt")
        candles, quality = clean_hyperliquid_candles(raw["candles"][coin], coin, start, end)
        if (candles.available_at > pd.Timestamp(created)).any():
            raise ValueError("Research contains unavailable candle outcomes")
        rows, segments = build_research_rows(candles, settings)
        partitions, split = split_research_rows(rows)
        prefix = BY_COIN[coin].filename
        artifacts[f"{prefix}/candles.parquet"] = candles
        if not rows.empty:
            artifacts[f"{prefix}/features.parquet"] = rows
        if split["status"] == "ready":
            for name, frame in partitions.items():
                artifacts[f"{prefix}/{name}.parquet"] = frame
        summary[coin] = {"symbol": BY_COIN[coin].symbol, "contract": coin, "directory": prefix,
                         "underlying_units_per_contract": BY_COIN[coin].underlying_units_per_contract,
                         "metadata_observed_at": contracts[coin]["context_received_at"],
                         "size_decimals": contracts[coin]["size_decimals"],
                         "quality": asdict(quality), "segments": segments,
                         "research_rows": len(rows), "discarded_window_rows": len(candles) - len(rows), **split}
    return summary, artifacts


def load_history_bundle(directory):
    """Full preparation audit; reconstruct all artifacts, including reserved test."""
    directory = native_path(Path(directory)).resolve()
    with (directory / "history.json").open(encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("schema_version") != 1 or report.get("mode") != "hyperliquid_research_preparation" or report.get("exchange") != "hyperliquid" or report.get("instrument") != "perpetual" or report.get("collateral_currency") != "USDC" or report.get("source_mode") != "observed":
        raise ValueError("Unsupported Hyperliquid research archive")
    hashes = report["artifact_hashes"]
    if not {"raw.json", "input_config.toml"} <= set(hashes) or any(Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name or ":" in name for name in hashes):
        raise ValueError("Unsafe or incomplete research artifact paths")
    for name, digest in hashes.items():
        if file_digest(directory / name) != digest:
            raise ValueError(f"Research artifact changed: {name}")
    settings = load_hyperliquid_settings(directory / "input_config.toml")
    if list(settings.coins) != report["coins"] or hashes["input_config.toml"] != report["input_config_sha256"]:
        raise ValueError("Research configuration binding disagrees")
    with (directory / "raw.json").open(encoding="utf-8") as stream:
        raw = json.load(stream)
    assets, frames = derive_history(raw, settings, report["protocol"])
    if set(hashes) != {"raw.json", "input_config.toml", *frames} or assets != report["assets"] or raw["captured_at"] != report["created_at"]:
        raise ValueError("Research summary disagrees with raw evidence")
    if any(report.get(name) is not False for name in ("models_fitted", "strategy_performance_evaluated", "forward_strategy_started", "paper_fills_created", "orders_created", "live_execution_enabled")):
        raise ValueError("Research archive cannot claim fitting, strategy performance or execution")
    for name, frame in frames.items():
        try:
            pd.testing.assert_frame_equal(pd.read_parquet(directory / name), frame, check_exact=True)
        except AssertionError as error:
            raise ValueError(f"Research rows disagree with raw evidence: {name}") from error
    # Detect mutation during the reconstruction as well as at its start.
    for name, digest in hashes.items():
        if file_digest(directory / name) != digest:
            raise ValueError("Research archive changed during audit")
    return report


def create_history_bundle(client, config_path, *, bars=MAXIMUM_CLOSED_BARS):
    config_path = native_path(Path(config_path)).resolve()
    settings = load_hyperliquid_settings(config_path)
    if client.settings != settings:
        raise ValueError("Research client and configuration disagree")
    config_digest = file_digest(config_path)
    protocol = research_protocol(settings, bars)
    end = int(_utc(client.clock()).timestamp()) // SECONDS * SECONDS
    start = end - bars * SECONDS
    raw = {"start_inclusive": start, "end_exclusive": end, "candles": {}, "candles_received_at": {},
           "contexts": {}, "contexts_received_at": {}}
    def fetch_candles(coin):
        payload = client.info({"type": "candleSnapshot", "req": {"coin": coin, "interval": "5m", "startTime": start * 1000, "endTime": end * 1000}})
        return payload, _utc(client.clock()).isoformat()
    dexes = sorted({BY_COIN[coin].dex for coin in settings.coins})
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {coin: pool.submit(fetch_candles, coin) for coin in settings.coins}
        metadata = {dex: pool.submit(client._contexts, dex) for dex in dexes}
        for coin, job in jobs.items():
            raw["candles"][coin], raw["candles_received_at"][coin] = job.result()
        for dex, job in metadata.items():
            raw["contexts"][dex], receipt = job.result()
            raw["contexts_received_at"][dex] = receipt.isoformat()
    raw["captured_at"] = _utc(client.clock()).isoformat()
    assets, frames = derive_history(raw, settings, protocol)
    report = {"schema_version": 1, "mode": "hyperliquid_research_preparation", "source_mode": "observed",
              "exchange": "hyperliquid", "instrument": "perpetual", "collateral_currency": "USDC",
              "created_at": raw["captured_at"], "coins": list(settings.coins), "protocol": protocol, "assets": assets,
              "input_config_sha256": config_digest, "models_fitted": False, "strategy_performance_evaluated": False,
              "forward_strategy_started": False, "paper_fills_created": False, "orders_created": False, "live_execution_enabled": False}
    parent = settings.observations.parent / "research"
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(raw["captured_at"]).strftime("%Y%m%dT%H%M%S%fZ")
    output = parent / f"history_{stamp}_{uuid4().hex[:8]}"
    staging = Path(tempfile.mkdtemp(prefix=".history-staging-", dir=parent))
    try:
        write_json(staging / "raw.json", raw)
        (staging / "input_config.toml").write_bytes(config_path.read_bytes())
        for name, frame in frames.items():
            write_parquet(staging / name, frame)
        names = ["raw.json", "input_config.toml", *frames]
        report["artifact_hashes"] = {name: file_digest(staging / name) for name in names}
        write_json(staging / "history.json", report)
        load_history_bundle(staging)
        if file_digest(config_path) != config_digest:
            raise ValueError("Research configuration changed during preparation")
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output, report


def prepare_hyperliquid_history(config_path, *, bars=MAXIMUM_CLOSED_BARS):
    settings = load_hyperliquid_settings(config_path)
    output, report = create_history_bundle(HyperliquidPublicClient(settings), config_path, bars=bars)
    print("Hyperliquid venue-specific research preparation (native contract units; no fitting):")
    for coin, asset in report["assets"].items():
        counts = "; ".join(f"{name} {part['rows']} (purged {part['purged_rows']})" for name, part in asset["partitions"].items())
        print(f"{coin}: {asset['quality']['cleaned_rows']} completed candles; missing {asset['quality']['missing_count']}; causal research rows {asset['research_rows']}; {asset['status']}")
        if counts:
            print(f"  {counts}")
    ready = sum(asset["status"] == "ready" for asset in report["assets"].values())
    print(f"PASS: raw evidence, gap-safe features and execution labels audited; {ready}/{len(settings.coins)} markets have prepared partitions")
    print("Research preparation only: test partitions unscored; reliability unvalidated; forward strategy month has not started")
    print(f"Saved Hyperliquid research to {output}")
    return output, report
