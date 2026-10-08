from dataclasses import replace
import io
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from local_ai_trader import cli
from local_ai_trader.data import download
from local_ai_trader.data.clean import clean_candles
from local_ai_trader.data.storage import write_json
from local_ai_trader.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
START = 1735689600  # 2025-01-01T00:00:00Z


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "config" / "settings.toml"
    path.parent.mkdir()
    path.write_text((ROOT / "config/settings.toml").read_text(), encoding="utf-8")
    return path


def row(timestamp):
    return [timestamp, 99, 102, 100, 101, 10]


def test_pagination_overlap_and_half_open_range(monkeypatch, config):
    settings = replace(load_settings(config), request_pause_seconds=0)
    requested = []

    def fetch(url, settings):
        query = parse_qs(urlparse(url).query)
        start = int(cli.parse_utc(query["start"][0]).timestamp())
        end = int(cli.parse_utc(query["end"][0]).timestamp())
        requested.append((start, end))
        assert end - start <= download.PAGE_INTERVALS * settings.candle_seconds
        return [row(stamp) for stamp in range(end, start - 1, -300)]

    monkeypatch.setattr(download, "request_candles", fetch)
    end = START + 501 * 300
    payload = download.download_candles("BTC-USD", START, end, settings)
    rows = [candle for page in payload["pages"] for candle in page["candles"]]
    frame, report = clean_candles(rows, START, end, 300)
    assert len(requested) == 3
    assert requested[0][0] == START
    assert requested[-1][1] == end
    assert len(frame) == 501
    assert report.duplicate_rows_removed == 2
    assert report.out_of_range_rows_removed == 1
    assert report.missing_count == 0


@pytest.mark.parametrize("status,attempts", [(403, 1), (429, 3), (503, 3)])
def test_http_failures_are_bounded(monkeypatch, config, status, attempts):
    calls = []

    def fail(request, timeout):
        calls.append(request.full_url)
        raise HTTPError(request.full_url, status, "failure", {}, None)

    monkeypatch.setattr(download, "urlopen", fail)
    monkeypatch.setattr(download.time, "sleep", lambda delay: None)
    with pytest.raises(RuntimeError):
        download.request_candles("https://example.com", load_settings(config))
    assert len(calls) == attempts


def test_retry_then_success(monkeypatch, config):
    calls = []

    def fetch(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise HTTPError(request.full_url, 429, "busy", {}, None)
        return io.BytesIO(json.dumps([row(START)]).encode())

    monkeypatch.setattr(download, "urlopen", fetch)
    monkeypatch.setattr(download.time, "sleep", lambda delay: None)
    assert download.request_candles("https://example.com", load_settings(config)) == [row(START)]
    assert len(calls) == 2


def mock_payload(symbol, first, last, settings):
    return {"pages": [{"candles": [row(stamp) for stamp in range(first, last, 300)]}]}


def test_collection_parquet_and_duckdb_query(monkeypatch, config, capsys):
    monkeypatch.setattr(cli, "download_candles", mock_payload)
    assert cli.main([
        "download", "--config", str(config), "--start", "2025-01-01T00:00:00Z",
        "--end", "2025-01-01T01:00:00Z",
    ]) == 0
    files = sorted((config.parent.parent / "data/processed").glob("*.parquet"))
    assert len(files) == 2
    frame = pd.read_parquet(files[0])
    assert len(frame) == 12
    assert frame["symbol"].unique().tolist() == ["BTC-USD"]
    assert (frame["available_at"] > frame["timestamp"]).all()

    # Simulate a Windows machine whose DuckDB session defaults to Pacific time.
    real_connect = cli.duckdb.connect

    def pacific_connection(*args, **kwargs):
        database = real_connect(*args, **kwargs)
        database.execute("SET TimeZone = 'America/Los_Angeles'")
        return database

    monkeypatch.setattr(cli.duckdb, "connect", pacific_connection)
    assert cli.main(["inspect", str(files[0])]) == 0
    output = capsys.readouterr().out
    assert "BTC-USD" in output
    assert "2025-01-01 00:00:00+00:00" in output
    assert "2025-01-01 00:55:00+00:00" in output


def test_gap_saves_evidence_without_parquet(monkeypatch, config):
    def incomplete(symbol, first, last, settings):
        return {"pages": [{"candles": [row(first)]}]}

    monkeypatch.setattr(cli, "download_candles", incomplete)
    assert cli.main([
        "download", "--config", str(config), "--start", "2025-01-01T00:00:00Z",
        "--end", "2025-01-01T01:00:00Z", "--symbols", "BTC-USD",
    ]) == 1
    root = config.parent.parent
    assert len(list((root / "data/raw").glob("*.json"))) == 1
    assert not list((root / "data/processed").glob("*.parquet"))
    report = json.loads(next((root / "data/processed").glob("*.quality.json")).read_text())
    assert report["missing_count"] == 11


def test_existing_snapshot_is_not_overwritten(monkeypatch, config):
    monkeypatch.setattr(cli, "download_candles", mock_payload)
    args = ["download", "--config", str(config), "--start", "2025-01-01T00:00:00Z",
            "--end", "2025-01-01T01:00:00Z", "--symbols", "BTC-USD"]
    assert cli.main(args) == 0
    parquet = next((config.parent.parent / "data/processed").glob("*.parquet"))
    original = parquet.read_bytes()
    assert cli.main(args) == 1
    assert parquet.read_bytes() == original


def test_failed_json_write_preserves_previous_file(tmp_path):
    path = tmp_path / "raw.json"
    write_json(path, {"ok": True})
    with pytest.raises(ValueError):
        write_json(path, {"invalid": float("nan")})
    assert json.loads(path.read_text()) == {"ok": True}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("start,end", [
    ("2025-01-01", "2025-01-02"),
    ("2025-01-01T00:01:00Z", "2025-01-02T00:00:00Z"),
    ("2025-01-02T00:00:00Z", "2025-01-01T00:00:00Z"),
    ("2025-01-01T00:00:00.1Z", "2025-01-02T00:00:00Z"),
    ("2025-01-01T00:00:00Z", None),
    ("2099-01-01T00:00:00Z", "2099-01-02T00:00:00Z"),
])
def test_invalid_date_ranges_rejected(start, end):
    with pytest.raises(ValueError):
        cli.range_bounds(start, end, 300, 7)


def test_timezone_conversion():
    first, last = cli.range_bounds("2024-12-31T19:00:00-05:00", "2024-12-31T20:00:00-05:00", 300, 7)
    assert first == START
    assert last - first == 3600


def test_default_range_ends_at_closed_boundary():
    first, last = cli.range_bounds(None, None, 300, 7)
    assert first % 300 == last % 300 == 0
    assert last - first == 7 * 86400


def test_settings_match_initial_market(config):
    settings = load_settings(config)
    assert settings.symbols == ("BTC-USD", "ETH-USD")
    assert settings.horizon_steps == 6
    assert settings.candle_seconds == 300
    assert settings.raw_dir == config.parent.parent / "data/raw"


def test_unsupported_exchange_rejected(config):
    config.write_text(config.read_text().replace('exchange = "coinbase"', 'exchange = "unknown"'))
    with pytest.raises(ValueError, match="supports only"):
        load_settings(config)


@pytest.mark.parametrize("symbol", ["../BTC-USD", "btc-usd", "BTC/USD", "BTC-USD?query"])
def test_invalid_product_identifiers_rejected(symbol):
    with pytest.raises(ValueError):
        cli.validate_symbol(symbol)
