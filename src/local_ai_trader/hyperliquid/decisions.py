"""Audited live model/decision previews, with no account or execution authority."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
from uuid import uuid4

import numpy as np
import pandas as pd

from local_ai_trader.data.storage import native_path, write_json
from local_ai_trader.features.build_features import FEATURE_COLUMNS
from local_ai_trader.hyperliquid.feed import HyperliquidPublicClient, load_hyperliquid_settings, parse_book
from local_ai_trader.hyperliquid.markets import BY_COIN
from local_ai_trader.hyperliquid.train import load_saved_market, predict_saved_market
from local_ai_trader.models.train import file_digest
from local_ai_trader.paper.perpetual import RISK_LIMITS, modeled_entry_net_return
from local_ai_trader.risk.entry_policy import SignalEvidence, evaluate_entry_signal
from local_ai_trader.risk.paper import _utc

LOGGER = logging.getLogger(__name__)
SLIPPAGE_BPS = 1.0


def decide_market(asset, prediction, policy):
    """Flat-position entry preview, using the frozen primary and Ridge forecast."""
    primary = prediction["primary_model"]
    probabilities = prediction["probabilities"].get(primary)
    quote = {"bid": asset["book"]["bid"], "ask": asset["book"]["ask"], "fee_bps": asset["taker_fee_bps_assumption"]}
    gross = float(prediction["return_forecasts"]["ridge"][0])
    # One potential settlement over 30 minutes, priced with the current oracle.
    # This is an expected-cost assumption, never a settled funding payment.
    fill = quote["ask"] * (1 + SLIPPAGE_BPS / 10000)
    funding_budget = max(0.0, asset["hourly_funding_rate"]) * asset["oracle_price"] / fill
    net = modeled_entry_net_return(quote, gross, SLIPPAGE_BPS, funding_budget)
    if probabilities is not None and (np.asarray(probabilities).shape != (1, 3) or not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any() or not np.isclose(probabilities.sum(), 1)):
        raise ValueError("Invalid saved directional probabilities")
    probability = None if probabilities is None else float(probabilities[0, 0])  # explicit up/down/flat order
    if probability is None:
        action, reason = "AVOID", "primary_model_unavailable"
    elif probability < policy.minimum_probability:
        action, reason = "AVOID", "up_probability_below_minimum"
    elif net <= RISK_LIMITS["minimum_net_margin"]:
        action, reason = "AVOID", "forecast_does_not_cover_costs_and_margin"
    else:
        action, reason = "BUY", "candidate_covers_modeled_costs"
    evidence = SignalEvidence(action, probability, prediction["calibration_id"], prediction["confidence"], prediction["reliability_validation_id"])
    gate = evaluate_entry_signal(evidence, 0, policy)
    return {"action": action, "decision_reason": reason, "signal_gate": asdict(gate), "primary_model": primary,
            "calibrated_up_probability": probability, "calibrated_probabilities_up_down_flat": None if probabilities is None else probabilities[0].tolist(),
            "expected_gross_return": gross, "modeled_net_return": net, "net_margin_required": RISK_LIMITS["minimum_net_margin"],
            "current_funding_cost_budget": funding_budget, "funding_budget_kind": "one_potential_hour_at_current_rate_and_oracle; no funding credits",
            "taker_fee_bps": quote["fee_bps"], "slippage_bps_per_side": SLIPPAGE_BPS,
            "return_denominator": "entry fill notional; same convention as perpetual paper account",
            "confidence": prediction["confidence"], "reliability_validation_id": prediction["reliability_validation_id"],
            "calibration_id": prediction["calibration_id"], "account_risk_authorized": False,
            "position_assumption": "flat; zero occupied positions for signal preview only"}


def create_decision_preview(client, config_path, model_directory, *, track_outcomes=False):
    config_path, model_directory = native_path(Path(config_path)).resolve(), native_path(Path(model_directory)).resolve()
    config_digest = file_digest(config_path)
    if load_hyperliquid_settings(config_path) != client.settings:
        raise ValueError("Scanner configuration disagrees with client")
    model_digest = file_digest(model_directory / "models.json")
    with (model_directory / "models.json").open(encoding="utf-8") as stream:
        models = json.load(stream)
    if models["coins"] != list(client.settings.coins):
        raise ValueError("Scanner and model market rosters disagree")
    for name in ("momentum_steps", "rolling_window"):
        if models["protocol"][name] != getattr(client.settings, name):
            raise ValueError("Scanner features disagree with frozen models")
    if any(models["protocol"].get(name) != value for name, value in (("candle_seconds", 300), ("latency_bars", 1), ("holding_bars", 6), ("target_known_offset_from_prediction_bars", 8))):
        raise ValueError("Scanner requires frozen delayed-entry, six-bar execution labels")
    bindings = {}
    for coin in client.settings.coins:
        if models["markets"][coin]["status"] == "insufficient_history":
            bindings[coin] = None
            continue
        checkpoint, _, source_id = load_saved_market(model_directory, coin)
        if checkpoint["protocol"] != models["protocol"]:
            raise ValueError("Scanner model timing protocol disagrees")
        bindings[coin] = source_id
    # Load/infer model libraries before executable-quote freshness checks.
    snapshot, raw, candles, features = client.snapshot()
    boundary = int(datetime.fromisoformat(snapshot["created_at"]).timestamp()) // 300 * 300
    results, pending = {}, {}
    for coin in client.settings.coins:
        asset = snapshot["assets"][coin]
        eligible = features[coin].loc[features[coin].available_at == pd.Timestamp(boundary-300, unit="s", tz="UTC")]
        if len(eligible) != 1:
            raise ValueError("Require exactly one causal signal with elapsed one-bar execution delay")
        if bindings[coin] is None:
            # A missing forecaster is never replaced by a numerical forecast.
            results[coin] = {"action": "AVOID", "decision_reason": "models_unavailable", "signal_gate": {"status": "BLOCK", "reason": "action_not_entry", "eligible_for_account_risk_check": False},
                             "calibrated_up_probability": None, "expected_gross_return": None, "modeled_net_return": None, "confidence": None, "account_risk_authorized": False}
        else:
            inference_frame = features[coin].iloc[-2:] if track_outcomes else eligible
            batch = predict_saved_market(model_directory, coin, inference_frame)
            prediction = {**batch, "probabilities": {name: values[:1] for name, values in batch["probabilities"].items()},
                          "return_forecasts": {name: values[:1] for name, values in batch["return_forecasts"].items()}}
            if prediction["calibration_id"] != bindings[coin]:
                raise ValueError("Saved scanner model changed during capture")
            results[coin] = decide_market(asset, prediction, client.settings.entry_policy)
            if track_outcomes and batch["primary_model"] in batch["probabilities"]:
                signal = features[coin].available_at.iloc[-1].to_pydatetime()
                pending[coin] = {"coin": coin, "signal_at": signal.isoformat(), "issued_at": _utc(client.clock()).isoformat(),
                                 "primary_model": batch["primary_model"], "calibration_id": batch["calibration_id"],
                                 "probabilities": batch["probabilities"][batch["primary_model"]][-1].tolist(),
                                 "ridge_return": float(batch["return_forecasts"]["ridge"][-1]),
                                 "mean_return": float(batch["return_forecasts"]["mean"][-1]),
                                 "features": {name: float(features[coin][name].iloc[-1]) for name in FEATURE_COLUMNS}}
        results[coin].update(symbol=BY_COIN[coin].symbol, coin=coin, signal_at=eligible.available_at.iloc[0].isoformat(),
                             inference_at=_utc(client.clock()).isoformat(), features={name: float(eligible[name].iloc[0]) for name in FEATURE_COLUMNS},
                             underlying_units_per_contract=BY_COIN[coin].underlying_units_per_contract)
    now = _utc(client.clock())
    def verify():
        at = _utc(client.clock())
        if int(at.timestamp()) // 300 * 300 != boundary:
            raise ValueError("Scanner candle boundary advanced during capture")
        for coin, asset in snapshot["assets"].items():
            parse_book(raw[f"{coin}_book"], coin, at, client.settings.maximum_book_age_seconds)
            age = (at-datetime.fromisoformat(asset["context_received_at"])).total_seconds()
            if not 0 <= age < client.settings.maximum_book_age_seconds:
                raise ValueError("Scanner mark/funding context is future-dated or stale")
        if file_digest(config_path) != config_digest or file_digest(model_directory / "models.json") != model_digest:
            raise ValueError("Scanner config/model manifest changed during capture")
        for coin, source_id in bindings.items():
            if source_id is not None and load_saved_market(model_directory, coin)[2] != source_id:
                raise ValueError("Scanner model changed before publication")
    verify()
    parent = client.settings.observations.parent / "decisions"
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / f"preview_{now.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:8]}"
    staging = Path(tempfile.mkdtemp(prefix=".decision-staging-", dir=parent))
    report = {"schema_version": 1, "mode": "hyperliquid_live_decision_preview", "source_mode": "observed", "created_at": now.isoformat(),
              "exchange": "hyperliquid", "instrument": "perpetual", "currency": "USDC", "model_directory": str(model_directory),
              "model_manifest_sha256": model_digest, "input_config_sha256": config_digest, "market_snapshot": snapshot, "decisions": results,
              "entry_policy": asdict(client.settings.entry_policy), "models_fitted": False, "test_opened": False,
              "paper_account_created": False, "paper_fills_created": False, "orders_created": False, "forward_strategy_started": False, "live_execution_enabled": False}
    if track_outcomes:
        report["pending_forward_predictions"] = pending
    try:
        write_json(staging / "raw_market.json", raw)
        (staging / "input_config.toml").write_bytes(config_path.read_bytes())
        report["artifact_hashes"] = {name: file_digest(staging / name) for name in ("raw_market.json", "input_config.toml")}
        write_json(staging / "decisions.json", report)
        verify()
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print("Hyperliquid live model scanner (one-bar-delayed signals; flat-position preview):")
    rows = [{"symbol": row["symbol"], "action": row["action"], "gate": row["signal_gate"]["status"],
             "p_up": row["calibrated_up_probability"], "gross_pct": None if row["expected_gross_return"] is None else 100*row["expected_gross_return"],
             "net_pct": None if row["modeled_net_return"] is None else 100*row["modeled_net_return"],
             "reason": row["signal_gate"]["reason"] if row["signal_gate"]["status"] == "REVIEW" else row["decision_reason"]} for row in results.values()]
    print(pd.DataFrame(rows).to_string(index=False))
    print("PASS: saved models inferred causal delayed signals; current quotes, modeled costs and signal gates checked")
    print("Preview only: independent account risk not requested; no fills; reliability unvalidated; strategy month not started")
    print(f"Saved live decision preview to {output}")
    return output, report


def run_decision_scanner(config_path, model_directory, *, watch=False, track_outcomes=False):
    settings = load_hyperliquid_settings(config_path)
    client = HyperliquidPublicClient(settings)
    last_boundary = None
    try:
        while True:
            boundary = int(client.clock().timestamp()) // 300
            if watch and boundary == last_boundary:
                time.sleep(settings.poll_seconds)
                continue
            try:
                output, report = create_decision_preview(client, config_path, model_directory, track_outcomes=track_outcomes)
                if track_outcomes:
                    from local_ai_trader.hyperliquid.forward import track_preview
                    track_preview(client, output, report)
                last_boundary = int(datetime.fromisoformat(report["created_at"]).timestamp()) // 300
            except (ValueError, RuntimeError, OSError, KeyError) as error:
                if not watch:
                    raise
                LOGGER.error("Live decision preview refused: %s", error)
            if not watch:
                return
            time.sleep(settings.poll_seconds)
    except KeyboardInterrupt:
        print("Hyperliquid live decision scanner stopped")
