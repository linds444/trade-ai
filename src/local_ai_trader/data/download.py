"""Paginate the unauthenticated Coinbase Exchange candle endpoint."""

from datetime import datetime, timezone
import json
import logging
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from local_ai_trader.settings import Settings, validate_symbol

LOGGER = logging.getLogger(__name__)
BASE_URL = "https://api.exchange.coinbase.com"
# Below Coinbase's 300-candle limit; inclusive endpoints may overlap.
PAGE_INTERVALS = 250


def iso_time(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def request_candles(url: str, settings: Settings) -> list:
    request = Request(url, headers={"User-Agent": "local-ai-trader/0.1", "Accept": "application/json"})
    for attempt in range(settings.max_attempts):
        try:
            with urlopen(request, timeout=settings.timeout_seconds) as response:
                payload = json.load(response)
            if not isinstance(payload, list):
                raise ValueError("Coinbase returned a non-list candle response")
            return payload
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(
                    f"Coinbase HTTP {error.code}. Check product availability and regional access."
                ) from error
            last_error = error
        except (URLError, TimeoutError) as error:
            last_error = error
        if attempt + 1 < settings.max_attempts:
            delay = 2 ** attempt
            LOGGER.warning("Candle request failed; retrying in %s seconds", delay)
            time.sleep(delay)
    raise RuntimeError(f"Coinbase request failed after {settings.max_attempts} attempts") from last_error


def download_candles(symbol: str, start: int, end: int, settings: Settings) -> dict:
    """Fetch [start, end); retain source rows and page bounds for auditability."""
    validate_symbol(symbol)
    if start >= end or start % settings.candle_seconds or end % settings.candle_seconds:
        raise ValueError("Download bounds must be ordered and aligned to the candle interval")
    if end > int(time.time()) // settings.candle_seconds * settings.candle_seconds:
        raise ValueError("Download range includes an unfinished candle")
    pages = []
    cursor = start
    while cursor < end:
        page_end = min(cursor + PAGE_INTERVALS * settings.candle_seconds, end)
        query = urlencode({
            "start": iso_time(cursor), "end": iso_time(page_end),
            "granularity": settings.candle_seconds,
        })
        url = f"{BASE_URL}/products/{symbol}/candles?{query}"
        LOGGER.info("Downloading %s: %s to %s", symbol, iso_time(cursor), iso_time(page_end))
        rows = request_candles(url, settings)
        pages.append({"start": cursor, "end": page_end, "candles": rows})
        cursor = page_end
        if cursor < end:
            time.sleep(settings.request_pause_seconds)
    return {
        "schema_version": 1, "exchange": "coinbase", "symbol": symbol,
        "granularity_seconds": settings.candle_seconds,
        "start_inclusive": start, "end_exclusive": end,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "pages": pages,
    }
