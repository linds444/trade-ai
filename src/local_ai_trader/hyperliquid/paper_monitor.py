"""Observed paper-book maintenance, with entry execution deliberately paused.

This adapter records signal skips and maintains marks/funding/reducing exits.
It never submits an entry, grants reliability, or starts a profitability trial.
"""

from dataclasses import asdict
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

from local_ai_trader.data.storage import native_path, write_json
from local_ai_trader.hyperliquid.decisions import SLIPPAGE_BPS
from local_ai_trader.hyperliquid.feed import parse_book, parse_contexts
from local_ai_trader.hyperliquid.funding import _publication_lock, apply_recorded_funding
from local_ai_trader.hyperliquid.markets import BY_COIN
from local_ai_trader.models.train import file_digest
from local_ai_trader.paper.ledger import _hash
from local_ai_trader.paper.perpetual import PerpetualPaperAccount, RISK_LIMITS
from local_ai_trader.risk.entry_policy import SignalEvidence, evaluate_entry_signal
from local_ai_trader.risk.paper import _utc

LOGGER = logging.getLogger(__name__)


def _binding(client, report):
    if report.get("mode") != "hyperliquid_live_decision_preview" or report.get("source_mode") != "observed" or report.get("live_execution_enabled") is not False:
        raise ValueError("Paper monitor requires an observed, non-executing preview")
    coins = list(client.settings.coins)
    if set(report["decisions"]) != set(coins) or set(report["market_snapshot"]["assets"]) != set(coins) or report["entry_policy"] != asdict(client.settings.entry_policy):
        raise ValueError("Paper monitor source roster/policy disagrees")
    return {"schema_version": 1, "mode": "hyperliquid_paper_monitor", "source_mode": "observed",
            "model_manifest_sha256": report["model_manifest_sha256"], "input_config_sha256": report["input_config_sha256"],
            "coins": coins, "initial_collateral": 10000.0, "slippage_bps": SLIPPAGE_BPS,
            "entry_policy": asdict(client.settings.entry_policy), "risk": dict(RISK_LIMITS),
            "entries_enabled": False, "profitability_trial_started": False}


def _quotes(client, report, raw):
    now = _utc(client.clock())
    assets = report["market_snapshot"]["assets"]
    contexts = {}
    for dex in dict.fromkeys(BY_COIN[coin].dex for coin in client.settings.coins):
        coins = tuple(coin for coin in client.settings.coins if BY_COIN[coin].dex == dex)
        received = {assets[coin]["context_received_at"] for coin in coins}
        payload = raw[f"{dex}_contexts" if dex else "contexts"]
        if len(received) != 1 or payload[0].get("collateralToken") != 0:
            raise ValueError("Paper monitor context/collateral disagrees")
        at = _utc(datetime.fromisoformat(received.pop()))
        if not 0 <= (now-at).total_seconds() < client.settings.maximum_book_age_seconds:
            raise ValueError("Paper monitor mark context is future-dated or stale")
        contexts.update(parse_contexts(payload, coins, at))
    quotes = []
    for coin in client.settings.coins:
        context, asset = contexts[coin], assets[coin]
        book = parse_book(raw[f"{coin}_book"], coin, now, client.settings.maximum_book_age_seconds)
        multiplier = 1.0
        if BY_COIN[coin].dex:
            scale = context["deployer_fee_scale"]
            multiplier = (1+scale) if scale < 1 else 2*scale
            multiplier *= 0.1 if context["growth_mode"] == "enabled" else 1
        fee = client.settings.taker_fee_bps * multiplier
        if any(asset[name] != context[name] for name in ("mark_price", "size_decimals", "context_received_at")) or asset["taker_fee_bps_assumption"] != fee or any(asset["book"][name] != book[name] for name in ("bid", "ask", "exchange_time")):
            raise ValueError("Paper monitor quote disagrees with archived raw evidence")
        quotes.append({"coin": coin, "at": min(book["exchange_time"], context["context_received_at"]),
                       "bid": book["bid"], "ask": book["ask"], "mark": context["mark_price"],
                       "fee_bps": fee, "size_decimals": context["size_decimals"], "source_mode": "observed"})
    return quotes


def _account(client, binding):
    parent = native_path(client.settings.observations.parent / "paper").resolve()
    parent.mkdir(parents=True, exist_ok=True)
    identity = _hash(binding)
    root = parent / ("monitor_" + identity[:16])
    with _publication_lock(parent, label="paper monitor"):
        if not root.exists():
            staging = Path(tempfile.mkdtemp(prefix=".monitor-staging-", dir=parent))
            try:
                PerpetualPaperAccount.create(staging / "account.sqlite", coins=client.settings.coins,
                    policy=client.settings.entry_policy, slippage_bps=SLIPPAGE_BPS,
                    source_mode="observed", now=_utc(client.clock()), run_binding=binding, entries_enabled=False)
                write_json(staging / "run.json", binding)
                os.rename(staging, root)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        if not (root / "account.sqlite").is_file():
            raise ValueError("Paper monitor account is missing; refusing to reset the book")
        cache = getattr(client, "_paper_monitors", {})
        account = cache.get(identity)
        if account is None:
            account = PerpetualPaperAccount(root / "account.sqlite")  # full startup replay audit
            cache[identity] = account
            client._paper_monitors = cache
        contract = account.contract()
        if json.loads((root / "run.json").read_text(encoding="utf-8")) != binding or contract.get("run_binding") != binding or contract["source_mode"] != "observed" or contract["entry_policy"] != binding["entry_policy"] or contract["coins"] != binding["coins"] or contract["slippage_bps"] != SLIPPAGE_BPS or contract["initial_collateral"] != binding["initial_collateral"] or contract.get("initial_entries_enabled") is not False:
            raise ValueError("Paper monitor account/run binding changed")
    return account, root


def monitor_preview(client, preview_path, report):
    """Maintain a paused paper account from one freshly audited scanner preview."""
    preview_path = native_path(Path(preview_path)).resolve()
    preview_hash = file_digest(preview_path / "decisions.json")
    if json.loads((preview_path / "decisions.json").read_text(encoding="utf-8")) != report:
        raise ValueError("Paper monitor preview changed")
    binding = _binding(client, report)
    for filename in ("raw_market.json", "input_config.toml"):
        if file_digest(preview_path / filename) != report["artifact_hashes"][filename]:
            raise ValueError("Paper monitor source artifact changed")
    if file_digest(preview_path / "input_config.toml") != binding["input_config_sha256"] or file_digest(native_path(Path(report["model_directory"])) / "models.json") != binding["model_manifest_sha256"]:
        raise ValueError("Paper monitor config/model binding changed")
    raw = json.loads((preview_path / "raw_market.json").read_text(encoding="utf-8"))
    quotes = _quotes(client, report, raw)  # refuse stale/future/mutated evidence before any write
    account, root = _account(client, binding)
    with _publication_lock(root, label="paper monitor"):
        result = _maintain(client, preview_path, report, preview_hash, binding, quotes, account, root)
    status = result["account"]
    print(f"Observed paper monitor: simulated wallet {status['wallet_balance']:.6f} USDC; equity {status['equity']:.6f}; fills {status['fill_count']}; positions {len(status['positions'])}; missing funding {len(status['missing_funding'])}")
    print("Entries paused: reliability validation and venue risk integration remain; profitability month has not started")
    print(f"Saved observed paper account to {account.path}")
    return result


def _maintain(client, preview_path, report, preview_hash, binding, quotes, account, root):
    receipt = root / "observations" / (preview_path.name + ".json")
    if receipt.is_file():
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        checksum = saved.pop("receipt_sha256", None)
        if _hash(saved) != checksum or saved["source_preview_sha256"] != preview_hash or saved["run_binding_sha256"] != _hash(binding):
            raise ValueError("Paper monitor receipt changed")
        saved["receipt_sha256"] = checksum
        return saved
    before = account.status()
    reviews = {}
    for coin, row in report["decisions"].items():
        gate = evaluate_entry_signal(SignalEvidence(row["action"], row["calibrated_up_probability"],
            row.get("calibration_id"), row["confidence"], row.get("reliability_validation_id")), len(before["positions"]), client.settings.entry_policy)
        reviews[coin] = {"signal_at": row["signal_at"], "action": row["action"], "decision_reason": row["decision_reason"],
                         "signal_gate": asdict(gate), "account_risk_authorized": False,
                         "execution_status": "paused" if gate.eligible_for_account_risk_check else gate.status.lower(),
                         "execution_reason": "paper_monitor_entries_paused" if gate.eligible_for_account_risk_check else gate.reason}
    # Entry is never invoked, even if a future adapter supplies a passing gate.
    if before["entries_enabled"]:
        account.execute("monitor-pause:" + preview_hash, "pause", {"enabled": False}, clock=client.clock)
    marked = account.execute("monitor-mark:" + preview_hash, "mark", {
        "quotes": quotes, "source_preview": str(preview_path), "source_preview_sha256": preview_hash,
        "source_decisions": report["decisions"]}, clock=client.clock)
    exits = []
    by_coin = {quote["coin"]: quote for quote in quotes}
    # Reducing exits precede funding archive IO. Missing funding cannot trap a holding.
    for coin, position in account.status()["positions"].items():
        due = datetime.fromisoformat(position["exit_due_at"])
        if _utc(client.clock()) >= due and datetime.fromisoformat(by_coin[coin]["at"]) >= due:
            exits.append(account.execute("monitor-horizon:" + position["position_id"], "exit", {
                "quote": by_coin[coin], "position_id": position["position_id"], "reason": "horizon"}, clock=client.clock))
    funding, funding_error = [], None
    if account.status()["missing_funding"]:
        try:
            funding, _ = apply_recorded_funding(account.path, client.settings.observations.parent / "funding", clock=client.clock)
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            funding_error = str(error)
            LOGGER.error("Paper monitor funding remains provisional: %s", error)
    status = account.status()
    result = {"schema_version": 1, "mode": "hyperliquid_paper_monitor_observation", "source_mode": "observed",
              "observed_at": _utc(client.clock()).isoformat(), "source_preview": str(preview_path), "source_preview_sha256": preview_hash,
              "account_path": str(account.path), "run_binding_sha256": _hash(binding), "mark_result": marked,
              "entry_reviews": reviews, "exit_results": exits, "funding_results": funding, "funding_error": funding_error,
              "account": status, "profitability_trial_started": False, "reliability_validated": False,
              "orders_created": False, "live_execution_enabled": False}
    result["receipt_sha256"] = _hash(result)
    write_json(receipt, result)
    return result
