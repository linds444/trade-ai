"""Independent, execution-time authorization for a simulated spot-long entry."""

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import tomllib

from local_ai_trader.decision.cost_aware import CostAwareDecision
from local_ai_trader.settings import BacktestSettings, load_backtest_settings, load_settings, positive_integer, validate_symbol


@dataclass(frozen=True)
class PaperRiskSettings:
    new_entries_enabled: bool = True
    allocation_fraction: float = 0.10
    max_position_exposure_fraction: float = 0.10
    max_total_exposure_fraction: float = 0.20
    max_drawdown_fraction: float = 0.10
    max_daily_loss_fraction: float = 0.02
    max_open_positions: int = 2
    max_signal_age_bars: int = 2
    max_quote_age_seconds: int = 30
    max_account_age_seconds: int = 30
    authorization_ttl_seconds: int = 5
    minimum_entry_cash: float = 10.0
    allowed_symbols: tuple[str, ...] = ("BTC-USD", "ETH-USD")
    candle_seconds: int = 300
    holding_bars: int = 6


MARKET_FIELDS = {"allowed_symbols", "candle_seconds", "holding_bars"}


def validate_paper_risk_settings(config: PaperRiskSettings) -> PaperRiskSettings:
    if not isinstance(config, PaperRiskSettings) or type(config.new_entries_enabled) is not bool:
        raise ValueError("Require valid paper risk settings")
    for name in ("allocation_fraction", "max_position_exposure_fraction", "max_total_exposure_fraction", "max_drawdown_fraction", "max_daily_loss_fraction"):
        value = getattr(config, name)
        if not _number(value, positive=True) or value > 1:
            raise ValueError(f"paper_risk.{name} must be a finite fraction in (0, 1]")
    if config.max_position_exposure_fraction > config.max_total_exposure_fraction:
        raise ValueError("Per-position exposure cannot exceed total exposure")
    for name in ("max_open_positions", "max_signal_age_bars", "max_quote_age_seconds", "max_account_age_seconds", "authorization_ttl_seconds", "candle_seconds", "holding_bars"):
        positive_integer(getattr(config, name), f"paper_risk.{name}")
    if not _number(config.minimum_entry_cash, positive=True):
        raise ValueError("paper_risk.minimum_entry_cash must be positive and finite")
    if not isinstance(config.allowed_symbols, tuple) or not config.allowed_symbols or not all(isinstance(symbol, str) for symbol in config.allowed_symbols) or len(set(config.allowed_symbols)) != len(config.allowed_symbols):
        raise ValueError("Paper risk requires unique allowed market symbols")
    for symbol in config.allowed_symbols:
        validate_symbol(symbol)
        if not symbol.endswith("-USD"):
            raise ValueError("Paper entry sizing requires markets quoted in USD")
    return config


def load_paper_risk_settings(path: Path) -> PaperRiskSettings:
    """Read a separate risk table; market timing comes from the primary config."""
    settings = load_settings(path)
    with path.open("rb") as stream:
        values = tomllib.load(stream).get("paper_risk", {})
    if not isinstance(values, dict) or set(values) - (set(PaperRiskSettings.__dataclass_fields__) - MARKET_FIELDS):
        raise ValueError("Unsupported paper_risk setting")
    config = validate_paper_risk_settings(PaperRiskSettings(**values, allowed_symbols=settings.symbols, candle_seconds=settings.candle_seconds, holding_bars=settings.horizon_steps))
    if config.max_signal_age_bars <= settings.backtest.latency_bars:
        raise ValueError("Paper signal freshness window must extend beyond execution latency")
    return config


@dataclass(frozen=True)
class MarkedPosition:
    symbol: str
    quantity: float
    mark_price: float
    marked_at: datetime


@dataclass(frozen=True)
class PendingEntry:
    symbol: str
    cash_outlay: float
    marked_notional: float
    marked_at: datetime


@dataclass(frozen=True)
class PaperAccountSnapshot:
    version: int
    as_of: datetime
    cash: float
    peak_equity: float
    day_start_equity: float
    day_start_at: datetime
    positions: tuple[MarkedPosition, ...] = ()
    pending_entries: tuple[PendingEntry, ...] = ()
    mode: str = "paper"
    currency: str = "USD"


@dataclass(frozen=True)
class PaperEntryRequest:
    symbol: str
    signal_at: datetime
    quote_at: datetime
    reference_price: float
    decision: CostAwareDecision
    candle_seconds: int = 300
    holding_bars: int = 6
    latency_bars: int = 1


@dataclass(frozen=True)
class PaperEntryAuthorization:
    symbol: str
    signal_at: datetime
    issued_at: datetime
    expires_at: datetime
    account_version: int
    request_sha256: str
    reference_price: float
    fill_price: float
    quantity: float
    cash_outlay: float
    fee: float
    price_impact_cost: float
    projected_equity: float
    projected_position_exposure_fraction: float
    projected_total_exposure_fraction: float
    mode: str = "paper"


@dataclass(frozen=True)
class PaperRiskResult:
    allowed: bool
    reason: str
    authorization: PaperEntryAuthorization | None = None


def _number(value, *, positive=False) -> bool:
    try:
        return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value) and (value > 0 if positive else value >= 0)
    except (ValueError, TypeError, OverflowError):
        return False


def _utc(value) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Risk timestamps must be timezone-aware datetimes")
    return value.astimezone(timezone.utc)


def _freshness(value: datetime, now: datetime, seconds: int, name: str) -> str | None:
    age = (now - _utc(value)).total_seconds()
    if age < 0:
        return f"future_{name}"
    if age >= seconds:
        return f"stale_{name}"
    return None


def _loss_limit(equity: float, baseline: float, fraction: float) -> bool:
    # Test the equity boundary directly; 1 - 9000/10000 rounds below 0.1.
    return equity <= baseline * (1 - fraction) + 1e-12 * baseline


def authorize_paper_entry(request: PaperEntryRequest, account: PaperAccountSnapshot, config: PaperRiskSettings, costs: BacktestSettings, now: datetime) -> PaperRiskResult:
    """Size and check one entry at fill time, without mutating or placing orders.

    The coordinator must recompute fresh marks, rerun this gate at execution,
    compare the account version and commit the paper fill atomically. Entry
    pauses, loss limits and uncertainty filters must never veto reducing exits.
    Pending entries consume cash and possible exposure; their gains never inflate
    projected equity. All quantities and fees here are simulated assumptions.
    """
    config = validate_paper_risk_settings(config)
    costs = load_backtest_settings(asdict(costs))
    now = _utc(now)

    def deny(reason):
        return PaperRiskResult(False, reason)

    if not isinstance(request, PaperEntryRequest) or not isinstance(request.decision, CostAwareDecision):
        return deny("invalid_entry_request")
    if request.decision.action != "BUY":
        return deny("decision_not_buy")
    if request.decision.risk_check_required is not True or not _number(request.decision.expected_net_return, positive=True):
        return deny("invalid_buy_candidate")
    if not config.new_entries_enabled:
        return deny("new_entries_paused")
    if request.symbol not in config.allowed_symbols:
        return deny("market_not_allowed")
    if not _number(request.reference_price, positive=True):
        return deny("invalid_reference_price")
    if any(type(getattr(request, name)) is not int for name in ("candle_seconds", "holding_bars", "latency_bars")) or (request.candle_seconds, request.holding_bars, request.latency_bars) != (config.candle_seconds, config.holding_bars, costs.latency_bars):
        return deny("execution_contract_mismatch")
    try:
        signal, quote = _utc(request.signal_at), _utc(request.quote_at)
        if signal.microsecond or int(signal.timestamp()) % config.candle_seconds:
            return deny("signal_not_interval_aligned")
        for stamp, seconds, name in ((signal, config.max_signal_age_bars * config.candle_seconds, "signal"), (quote, config.max_quote_age_seconds, "quote")):
            reason = _freshness(stamp, now, seconds, name)
            if reason:
                return deny(reason)
        eligible = signal + timedelta(seconds=costs.latency_bars * config.candle_seconds)
        if now < eligible:
            return deny("execution_latency_not_elapsed")
        if quote < eligible:
            return deny("quote_precedes_execution")
        if not isinstance(account, PaperAccountSnapshot) or account.mode != "paper" or account.currency != "USD":
            return deny("invalid_paper_account")
        if type(account.version) is not int or account.version < 0 or not _number(account.cash) or not all(_number(value, positive=True) for value in (account.peak_equity, account.day_start_equity)):
            return deny("invalid_paper_account")
        account_at, day_start = _utc(account.as_of), _utc(account.day_start_at)
        reason = _freshness(account_at, now, config.max_account_age_seconds, "account")
        if reason:
            return deny(reason)
        if day_start != now.replace(hour=0, minute=0, second=0, microsecond=0):
            return deny("utc_day_baseline_unavailable")
        if day_start > account_at or account.day_start_equity > account.peak_equity + 1e-12 * account.peak_equity:
            return deny("inconsistent_day_baseline")
        if not isinstance(account.positions, tuple) or not isinstance(account.pending_entries, tuple):
            return deny("invalid_portfolio")
        symbols, marked_values, reservations, pending_values, pending_losses, expiries = set(), [], [], [], [], []
        for entries, entry_type, name in ((account.positions, MarkedPosition, "position"), (account.pending_entries, PendingEntry, "pending_entry")):
            for entry in entries:
                if not isinstance(entry, entry_type) or entry.symbol not in config.allowed_symbols or entry.symbol in symbols:
                    return deny("invalid_portfolio")
                marked_at = _utc(entry.marked_at)
                if marked_at > account_at:
                    return deny("mark_after_account_snapshot")
                reason = _freshness(marked_at, now, config.max_account_age_seconds, f"{name}_mark")
                if reason:
                    return deny(reason)
                symbols.add(entry.symbol)
                expiries.append(marked_at + timedelta(seconds=config.max_account_age_seconds))
                if entry_type is MarkedPosition:
                    if not _number(entry.quantity, positive=True) or not _number(entry.mark_price, positive=True):
                        return deny("invalid_portfolio")
                    marked_values.append(entry.quantity * entry.mark_price)
                else:
                    if not _number(entry.cash_outlay, positive=True) or not _number(entry.marked_notional, positive=True):
                        return deny("invalid_portfolio")
                    reservations.append(entry.cash_outlay)
                    pending_values.append(entry.marked_notional)
                    pending_losses.append(max(entry.cash_outlay - entry.marked_notional, 0))
        holding = math.fsum(marked_values)
        pending_exposure, reserved_cash = math.fsum(pending_values), math.fsum(reservations)
        equity = account.cash + holding
        if not _number(equity, positive=True) or not math.isfinite(pending_exposure) or reserved_cash > account.cash + 1e-12 * equity or equity > account.peak_equity + 1e-12 * equity:
            return deny("inconsistent_account_equity_or_reservations")
        if _loss_limit(equity, account.peak_equity, config.max_drawdown_fraction):
            return deny("drawdown_limit")
        if _loss_limit(equity, account.day_start_equity, config.max_daily_loss_fraction):
            return deny("daily_loss_limit")
        if request.symbol in {position.symbol for position in account.positions}:
            return deny("position_already_open")
        if request.symbol in {entry.symbol for entry in account.pending_entries}:
            return deny("entry_already_pending")
        if len(symbols) + 1 > config.max_open_positions:
            return deny("position_count_limit")
        budget = equity * config.allocation_fraction
        if budget < config.minimum_entry_cash:
            return deny("below_minimum_entry_cash")
        if budget > account.cash - reserved_cash + 1e-12 * equity:
            return deny("insufficient_unreserved_cash")
        impact, fee_rate = (costs.spread_bps / 2 + costs.slippage_bps) / 10000, costs.fee_bps / 10000
        fill_price = request.reference_price * (1 + impact)
        quantity = budget / (fill_price * (1 + fee_rate))
        notional, marked = quantity * fill_price, quantity * request.reference_price
        fee = notional * fee_rate
        outlay = notional + fee
        projected = equity - math.fsum(pending_losses) - outlay + marked
        if not all(_number(value, positive=True) for value in (fill_price, quantity, outlay, marked, projected)):
            return deny("invalid_projected_entry")
        if _loss_limit(projected, account.peak_equity, config.max_drawdown_fraction):
            return deny("projected_drawdown_limit")
        if _loss_limit(projected, account.day_start_equity, config.max_daily_loss_fraction):
            return deny("projected_daily_loss_limit")
        position_fraction = marked / projected
        total_fraction = (holding + pending_exposure + marked) / projected
        if position_fraction > config.max_position_exposure_fraction + 1e-12:
            return deny("position_exposure_limit")
        if any(value / projected > config.max_position_exposure_fraction + 1e-12 for value in marked_values + pending_values):
            return deny("existing_position_exposure_limit")
        if total_fraction > config.max_total_exposure_fraction + 1e-12:
            return deny("total_exposure_limit")
        expires_at = min(
            now + timedelta(seconds=config.authorization_ttl_seconds),
            signal + timedelta(seconds=config.max_signal_age_bars * config.candle_seconds),
            quote + timedelta(seconds=config.max_quote_age_seconds),
            account_at + timedelta(seconds=config.max_account_age_seconds),
            day_start + timedelta(days=1), *expiries,
        )
        if expires_at <= now:
            return deny("authorization_window_elapsed")
        fingerprint = hashlib.sha256(json.dumps({"request": asdict(request), "account": asdict(account), "risk_settings": asdict(config), "costs": asdict(costs)}, sort_keys=True, default=lambda value: _utc(value).isoformat(), allow_nan=False).encode()).hexdigest()
        authorization = PaperEntryAuthorization(request.symbol, signal, now, expires_at, account.version, fingerprint, float(request.reference_price), fill_price, quantity, outlay, fee, quantity * (fill_price - request.reference_price), projected, position_fraction, total_fraction)
        return PaperRiskResult(True, "paper_entry_approved", authorization)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError):
        return deny("invalid_account_or_entry_data")


def paper_authorization_is_current(authorization: PaperEntryAuthorization, request: PaperEntryRequest, account: PaperAccountSnapshot, config: PaperRiskSettings, costs: BacktestSettings, now: datetime) -> bool:
    """Recheck a simulated approval against fresh inputs, without consuming it.

    This is not a bearer token or a lock. The paper coordinator must use this
    check and commit against the same account version in one atomic transaction.
    """
    if not isinstance(authorization, PaperEntryAuthorization) or not isinstance(account, PaperAccountSnapshot):
        return False
    try:
        now, issued, expiry = _utc(now), _utc(authorization.issued_at), _utc(authorization.expires_at)
        if authorization.mode != "paper" or authorization.account_version != account.version or not issued <= now < expiry or expiry > issued + timedelta(seconds=config.authorization_ttl_seconds):
            return False
        result = authorize_paper_entry(request, account, config, costs, now)
        if not result.allowed or expiry > result.authorization.expires_at:
            return False
        return replace(result.authorization, issued_at=authorization.issued_at, expires_at=authorization.expires_at) == authorization
    except (ValueError, TypeError, OverflowError, AttributeError):
        return False
