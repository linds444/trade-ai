# Trade AI

A local, Windows-first market research project for an NVIDIA RTX 3060 (12 GB).
The first milestone collects and validates historical data. It does not yet
train a model, make trading recommendations, or execute orders.

## Verified starting environment

The user's Windows computer passed these checks before development:

- Windows 11 Home, 64-bit; Python 3.12.10; Git 2.54.0.
- NVIDIA RTX 3060, 12 GB VRAM; driver 617.42.
- PyTorch 2.11.0+cu128, CUDA runtime 12.8; successful GPU matrix computation.
- NumPy 2.5.3; successful NumPy -> GPU -> NumPy conversion.

These are user-reported results, not GPU measurements from the development
workspace. `scripts/check_gpu.py` repeats the check on the machine running it.

## Windows setup

Use **PowerShell**, from the root of your cloned repository. No administrator
terminal, WSL, Docker, separate CUDA Toolkit, or API credentials are required.

For a new environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

If your environment already exists, skip its creation. Installing the core
dependencies does not install or replace PyTorch. For a fresh GPU environment,
install the previously verified PyTorch build separately:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python.exe scripts/check_gpu.py
```

### Reuse the environment already verified in this conversation

If you clone this repository into `C:\Users\lingr\trade-ai`, reuse your existing
environment without rebuilding it. From the cloned repository root:

```powershell
$projectPython = "$env:USERPROFILE\local-ai-trader\.venv\Scripts\python.exe"
& $projectPython -m pip install -r requirements-dev.txt
& $projectPython -m pytest -q
& $projectPython scripts/check_gpu.py
```

Use `& $projectPython` in place of `.\.venv\Scripts\python.exe` in the examples
below. No activation or execution-policy changes are necessary.

## First historical download

`config/settings.toml` defaults to Coinbase **BTC-USD and ETH-USD**, five-minute
candles, a planned 30-minute prediction horizon (six candles), and seven days of
history. TOML uses Python's standard library and avoids a YAML dependency.
The prediction horizon is metadata for the next milestone; targets are not
generated yet. USD spot candles are not interchangeable with USDT markets.

Start with a fixed one-hour smoke test, not a training dataset:

```powershell
.\.venv\Scripts\python.exe -m local_ai_trader download --start 2026-01-01T00:00:00Z --end 2026-01-01T01:00:00Z
```

Expected: **12 validated rows per symbol**, with no missing intervals. Raw
responses go to `data/raw/`; Parquet and quality reports go to `data/processed/`.
Dates must include a timezone and align with the candle interval. The requested
range is **start-inclusive and end-exclusive**, in UTC.
The inspection command explicitly displays UTC, regardless of the computer's
local timezone.

After that succeeds, the default download fetches the latest seven days of fully
closed candles (2,016 per symbol if complete):

```powershell
.\.venv\Scripts\python.exe -m local_ai_trader download
```

Query one saved file locally with DuckDB:

```powershell
$dataset = Get-ChildItem data\processed\*BTC-USD*.parquet | Select-Object -First 1
.\.venv\Scripts\python.exe -m local_ai_trader inspect $dataset.FullName
```

Collection uses public HTTPS requests with bounded retries and page throttling.
If Coinbase denies access, fails, or reports gaps, stop and inspect the error;
the program does not silently substitute another exchange. Coinbase may omit
intervals with no trades. A gap is evidence to investigate, not a candle to invent.

## Data guarantees and limits

- Raw page responses, request bounds, exchange, product and retrieval time are saved.
- OHLC prices must be positive and internally consistent; volume must be
  nonnegative. Nonfinite numbers, malformed rows and misaligned timestamps fail.
- Candles are sorted in UTC. Identical duplicates are removed; conflicting
  duplicates fail. Rows outside the requested range are excluded and counted.
- Missing leading, internal and trailing intervals are reported. Any gap prevents
  writing processed Parquet. Raw evidence and a quality report remain available.
- `timestamp` is the candle **open** time. `available_at` is its close time:
  a candle's full OHLCV must never be used earlier than that.
- Future and unfinished candles are excluded. Output ranges are immutable;
  an existing raw, quality or Parquet file prevents overwrite. To retry a failed
  range, move its existing files aside first so the original evidence is retained.
- Files are replaced atomically after successful writes. A download that fails
  partway through its requests does not produce a complete raw snapshot.
- Each symbol is collected independently; one can succeed before another fails.
- Two selected assets do not remove survivorship or asset-selection bias. Later
  research must document its universe and avoid generalizing beyond these assets.

## Tests and dependencies

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Tests cover pagination boundaries, timestamp availability, duplicate conflicts,
invalid prices, missing intervals, retry behavior, output preservation, timezone
conversion, Parquet round trips and DuckDB queries. Tests use synthetic candles
and need neither network access nor a GPU.

Core dependencies have compatibility ranges. After a verified installation,
record its exact resolved versions for local reproducibility:

```powershell
.\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding utf8 requirements-lock-local.txt
```

That local snapshot can include an absolute editable-install path; keep it as an
environment record rather than assuming it is portable to another computer.

## Next milestones

1. Define forward-return and up/down/flat targets with precise availability times.
2. Build causal features and chronological splits, purging overlapping forward
   labels at split boundaries before comparing a naive baseline and logistic regression.
3. Compare gradient boosting and a small GPU MLP; measure calibration separately.
4. Add cost-aware backtesting, walk-forward evaluation and local experiment logs.
5. Add a Transformer only when evidence warrants it, then uncertainty, configurable
   policy, a separate risk gate, paper trading, API and dashboard.

Every change must demonstrate improvement against the previous model on suitable
unseen data. Training, validation, test, backtest and paper-trading performance
will be reported separately. A profitable backtest does not establish live profitability.
