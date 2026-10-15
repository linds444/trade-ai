import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from local_ai_trader import cli
from local_ai_trader.data import reprocess
from local_ai_trader.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
START = 1735689600


@pytest.fixture
def archive(tmp_path):
    config = tmp_path / "config/settings.toml"
    config.parent.mkdir()
    config.write_text((ROOT / "config/settings.toml").read_text(), encoding="utf-8")
    settings = load_settings(config)
    settings.raw_dir.mkdir(parents=True)
    source = settings.raw_dir / f"coinbase_BTC-USD_300s_{START}_{START + 15 * 300}.json"
    rows = [[stamp, 99, 102, 100, 101, 10] for stamp in range(START, START + 15 * 300, 300) if stamp != START + 2 * 300]
    rows += [rows[-1], [START + 15 * 300, 99, 102, 100, 101, 10]]
    payload = {
        "schema_version": 1, "exchange": "coinbase", "symbol": "BTC-USD",
        "granularity_seconds": 300, "start_inclusive": START, "end_exclusive": START + 15 * 300,
        "downloaded_at": "2026-01-01T00:00:00+00:00", "pages": [{"candles": rows}],
    }
    source.write_text(json.dumps(payload), encoding="utf-8")
    return source, settings, config


def test_complete_subrange_preserves_original_evidence_and_records_provenance(archive):
    source, settings, _ = archive
    original = source.read_bytes()
    settings.processed_dir.mkdir(parents=True)
    old_report = settings.processed_dir / f"{source.stem}.quality.json"
    old_report.write_text('{"missing_count": 1}', encoding="utf-8")
    destination, report = reprocess.reprocess_download(source, START + 5 * 300, START + 15 * 300, settings)
    frame = pd.read_parquet(destination)
    assert len(frame) == report["expected_rows"] == report["cleaned_rows"] == 10
    assert report["missing_count"] == 0
    assert report["duplicate_rows_removed"] == 1
    assert frame.timestamp.iloc[0] == pd.Timestamp(START + 5 * 300, unit="s", tz="UTC")
    assert frame.timestamp.iloc[-1] == pd.Timestamp(START + 14 * 300, unit="s", tz="UTC")
    assert frame.symbol.eq("BTC-USD").all()
    assert report["raw_sha256"] == hashlib.sha256(original).hexdigest()
    assert report["raw_path"] == str(source.resolve())
    assert report["source_start_inclusive"] == START
    assert report["source_end_exclusive"] == START + 15 * 300
    assert source.read_bytes() == original
    assert json.loads(old_report.read_text()) == {"missing_count": 1}
    assert json.loads(destination.with_suffix(".quality.json").read_text()) == report
    assert not list(settings.processed_dir.glob(".reprocess-staging-*"))


def test_gap_in_selected_range_saves_evidence_without_inventing_candles(archive):
    source, settings, _ = archive
    with pytest.raises(ValueError, match="1 missing candles"):
        reprocess.reprocess_download(source, START + 300, START + 5 * 300, settings)
    assert not list(settings.processed_dir.glob("*.parquet"))
    report = json.loads(next(settings.processed_dir.glob("*.quality.json")).read_text())
    assert report["missing_count"] == 1
    assert report["cleaned_rows"] == 3


@pytest.mark.parametrize("first,last", [
    (START - 300, START + 5 * 300), (START + 300, START + 16 * 300),
    (START + 1, START + 5 * 300), (START + 300, START + 5 * 300 + 1),
    (START + 5 * 300, START + 5 * 300), (True, START + 5 * 300),
])
def test_unaligned_or_outside_source_ranges_fail(archive, first, last):
    source, settings, _ = archive
    with pytest.raises(ValueError, match="Selected bounds"):
        reprocess.reprocess_download(source, first, last, settings)
    assert not settings.processed_dir.exists()


@pytest.mark.parametrize("field,value", [
    ("granularity_seconds", 60), ("symbol", "../BTC-USD"), ("exchange", "other"),
    ("schema_version", 2), ("pages", []), ("start_inclusive", True),
])
def test_incompatible_raw_provenance_fails(archive, field, value):
    source, settings, _ = archive
    payload = json.loads(source.read_text())
    payload[field] = value
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        reprocess.reprocess_download(source, START + 5 * 300, START + 15 * 300, settings)
    assert not settings.processed_dir.exists()


def test_output_is_immutable(archive):
    source, settings, _ = archive
    first, last = START + 5 * 300, START + 15 * 300
    destination, _ = reprocess.reprocess_download(source, first, last, settings)
    original = destination.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        reprocess.reprocess_download(source, first, last, settings)
    assert destination.read_bytes() == original


def test_failed_parquet_write_cleans_staging_and_publishes_nothing(archive, monkeypatch):
    source, settings, _ = archive

    def fail(*args, **kwargs):
        raise OSError("Simulated full disk")

    monkeypatch.setattr(reprocess, "write_parquet", fail)
    with pytest.raises(OSError):
        reprocess.reprocess_download(source, START + 5 * 300, START + 15 * 300, settings)
    assert not list(settings.processed_dir.iterdir())


def test_source_change_before_publication_is_rejected(archive, monkeypatch):
    source, settings, _ = archive
    original_write = reprocess.write_parquet

    def changed(*args, **kwargs):
        original_write(*args, **kwargs)
        with source.open("ab") as stream:
            stream.write(b" ")

    monkeypatch.setattr(reprocess, "write_parquet", changed)
    with pytest.raises(ValueError, match="Raw source changed"):
        reprocess.reprocess_download(source, START + 5 * 300, START + 15 * 300, settings)
    assert not list(settings.processed_dir.iterdir())


def test_reprocess_cli_uses_saved_data_without_network(archive, monkeypatch):
    source, settings, config = archive

    def forbidden(*args, **kwargs):
        raise AssertionError("Reprocessing must not download candles")

    monkeypatch.setattr(cli, "download_candles", forbidden)
    assert cli.main([
        "reprocess", str(source), "--config", str(config),
        "--start", "2025-01-01T00:25:00Z", "--end", "2025-01-01T01:15:00Z",
    ]) == 0
    assert len(list(settings.processed_dir.glob("*.parquet"))) == 1
