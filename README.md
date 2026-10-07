# Trade AI

A local, Windows-first market research project for an NVIDIA RTX 3060 (12 GB).
The current milestone collects and validates historical data, builds
forward-return targets, and computes causal features. It does not yet train a model, make trading
recommendations, or execute orders.

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
The prediction horizon also controls the separate target-generation command.
USD spot candles are not interchangeable with USDT markets.

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
Target tests additionally cover six-step indexing, label availability, inclusive
flat boundaries, insufficient future data, nonfinite returns and rejection of
gaps, duplicates, multiple assets, unfinished candles and malformed prices.
Feature tests compare calculations to independent scalar formulas, alter future
candles, truncate future history, change target values, check zero-volume and
flat candles, and validate dataset provenance and immutable storage.

Core dependencies have compatibility ranges. After a verified installation,
record its exact resolved versions for local reproducibility:

```powershell
.\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding utf8 requirements-lock-local.txt
```

That local snapshot can include an absolute editable-install path; keep it as an
environment record rather than assuming it is portable to another computer.

## Forward-return targets

From a validated candle file, generate labels in a separate Parquet snapshot:

```powershell
.\.venv\Scripts\python.exe -m local_ai_trader targets data/processed/coinbase_BTC-USD_300s_1790804400_1791409200.parquet
```

This filename matches the user's verified seven-day BTC download. Replace it
with another original candle file when needed. Outputs appear in
`data/processed/targets/`, with a `_h6.parquet` suffix and a `.targets.json`
metadata file. A 2,016-candle dataset yields **2,010 labelled rows** because the
last six future outcomes are unknown within that snapshot. Original candle
datasets are preserved, and target snapshots cannot be overwritten.

For a prediction made at candle close time `available_at[t]`, the target is:

```text
target_return[t] = close[t + 6] / close[t] - 1
target_available_at[t] = available_at[t + 6]
```

For example, the 10:00 candle is fully known at 10:05. Its six-step outcome uses
the 10:30 candle's close, known at 10:35: exactly 30 minutes after prediction
time. This target is a close-to-close price outcome, not an assumed executable
trade return.

The provisional `targets.flat_return_threshold = 0.001` is a fractional return:

- `up`: return above +0.1%.
- `down`: return below -0.1%.
- `flat`: return between those boundaries, inclusive.

A machine-precision tolerance (eight float64 epsilons) absorbs numerical
roundoff at the two boundaries; it is recorded in the metadata. The threshold
is a research choice, not a policy threshold, fee allowance, calibrated
probability or prediction. Later changes must be evaluated using training and
validation data without tuning on the held-out test set.

The metadata records the source SHA-256, horizon, threshold, class counts and
label availability. **All `target_*` columns contain future-only information**
and must be excluded from model inputs. Chronological splitting is not yet
implemented: the next milestone must purge overlapping labels using
`target_available_at`, not simply slice consecutive rows or split randomly.

## Causal features

Generate model inputs from a labelled target snapshot:

```powershell
.\.venv\Scripts\python.exe -m local_ai_trader features data/processed/targets/coinbase_BTC-USD_300s_1790804400_1791409200_h6.parquet
```

The default settings use a six-candle momentum interval and 12-candle trailing
window, both configurable in `config/settings.toml`. The eight inputs are:

| Feature | Calculation at candle t | Research purpose |
| --- | --- | --- |
| `feature_return_1` | `close[t] / close[t-1] - 1` | Most recent price move |
| `feature_momentum` | `close[t] / close[t-6] - 1` | Recent 30-minute direction |
| `feature_volatility` | Population standard deviation of the last 12 log returns | Recent variation in five-minute returns; not annualized |
| `feature_close_vs_mean` | `close[t] / mean(close[t-11:t]) - 1` | Position relative to the recent price level |
| `feature_volume_ratio` | `volume[t] / mean(volume[t-11:t])` | Activity relative to the recent baseline |
| `feature_range` | `(high[t] - low[t]) / close[t]` | Current candle's price range |
| `feature_body` | `(close[t] - open[t]) / open[t]` | Signed movement within the candle |
| `feature_close_location` | `(close[t] - low[t]) / (high[t] - low[t])` | Where the candle finished within its range |

The table's ranges include both endpoints. Rolling windows include the current
fully closed candle and older candles. Twelve log returns require 13 closes,
so the first 12 rows are warmup: **2,010 labelled rows become 1,998 feature rows**.
A window containing only zero volume uses ratio 0; a zero-width candle uses
neutral close location 0.5. Other nonfinite results fail rather than silently
removing arbitrary internal rows.

Feature Parquet snapshots and `.features.json` reports are stored under
`data/processed/features/`. Reports include the ordered feature allowlist,
parameters, warmup count, class counts, source hash and original target metadata.
The target horizon, interval and flat threshold must agree with the current
settings; a mismatch fails instead of relabelling old data. Original targets
are preserved. Existing feature snapshots cannot be overwritten.

No fitted scaling or normalization is applied yet. Each feature is a hypothesis
to evaluate against a baseline, not an established source of predictive signal.
Future models must select only the eight `feature_names` in the metadata.
Targets and timestamps stay in the dataset for supervision and chronological
evaluation; they are not model inputs.

## Next milestones

1. Build chronological splits, purging overlapping forward labels at split
   boundaries before comparing a naive baseline and logistic regression.
2. Compare gradient boosting and a small GPU MLP; measure calibration separately.
3. Add cost-aware backtesting, walk-forward evaluation and local experiment logs.
4. Add a Transformer only when evidence warrants it, then uncertainty, configurable
   policy, a separate risk gate, paper trading, API and dashboard.

Every change must demonstrate improvement against the previous model on suitable
unseen data. Training, validation, test, backtest and paper-trading performance
will be reported separately. A profitable backtest does not establish live profitability.
