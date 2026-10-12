"""Atomically replace output files after a complete successful write."""

import json
from pathlib import Path
import os
import tempfile

import pandas as pd


def native_path(path: Path) -> Path:
    """Use extended absolute Windows paths without requiring registry changes.

    Python, Arrow and model files share deeply nested walk-forward directories.
    Keep the extended prefix on the directory so every derived path, including
    temporary files and atomic directory renames, uses the same Windows namespace.
    """
    if os.name != "nt":
        return path
    absolute = os.path.abspath(path)
    if absolute.startswith("\\\\?\\"):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


def write_json(path: Path, payload: object) -> None:
    path = native_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, allow_nan=False)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path = native_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        # Reuse the already-open temporary file; Arrow need not resolve its path.
        with os.fdopen(descriptor, "wb") as output:
            frame.to_parquet(output, engine="pyarrow", index=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
