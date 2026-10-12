import json
import os
from pathlib import Path

import pandas as pd
import pytest

from local_ai_trader.data.storage import native_path, write_json, write_parquet


def test_long_output_paths_round_trip_json_and_parquet(tmp_path):
    parent = tmp_path.joinpath(*(f"nested_{index}_" + "x" * 60 for index in range(4)))
    parquet = parent / "assessment_naive_temperature.parquet"
    metadata = parent / "calibration.json"
    assert len(str(parquet)) > 300
    expected = pd.DataFrame({"probability": [0.25, 0.75], "class": ["up", "flat"]})
    write_parquet(parquet, expected)
    write_json(metadata, {"rows": 2})
    pd.testing.assert_frame_equal(pd.read_parquet(native_path(parquet)), expected)
    assert json.loads(native_path(metadata).read_text(encoding="utf-8")) == {"rows": 2}
    assert not list(native_path(parent).glob("*.tmp"))
    if os.name == "nt":
        assert str(native_path(parquet)).startswith("\\\\?\\")


def test_failed_parquet_write_keeps_existing_output_and_cleans_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "saved.parquet"
    expected = pd.DataFrame({"x": [1, 2]})
    write_parquet(path, expected)
    before = path.read_bytes()

    def fail(frame, output, **kwargs):
        assert hasattr(output, "write")
        output.write(b"incomplete parquet")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    with pytest.raises(OSError, match="disk full"):
        write_parquet(path, expected)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_native_path_is_idempotent_and_relative_paths_resolve_on_windows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = Path("nested") / "output.parquet"
    converted = native_path(source)
    assert native_path(converted) == converted
    if os.name == "nt":
        assert str(converted).startswith("\\\\?\\")
        assert converted.is_absolute()
    else:
        assert converted == source
