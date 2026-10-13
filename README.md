# Trade AI

A local, Windows-first market research project for an NVIDIA RTX 3060 (12 GB).
Research includes causal features and purged splits, calibrated classifier and
return baselines, walk-forward cost accounting, decision/risk checks and a
persistent spot-paper execution foundation.

The current deployment target is **eight Hyperliquid perpetuals, 1× leverage,
long only**. Its public mainnet data feed validates and archives live observations
without a wallet key. A separate restartable USDC perpetual accounting foundation
handles margin reservation, costs and recorded funding. Venue-bound models can
now train, calibrate and assess all eight markets in one batch. Recorded-funding
coordination, separately validated reliability and liquidation rules remain integration steps.
Recent Hyperliquid history can now prepare all eight markets' causal features,
execution labels and purged research partitions in one command. Live execution is disabled.

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

## Reuse a complete portion of a saved raw download

If repeated requests leave unresolved gaps in an earlier part of a download,
an explicitly selected complete subrange can be processed without downloading
everything again. For the April–September archive with gaps on May 8, select
May 9 through September 29 (exclusive), **143 days / 41,184 candles**:

```powershell
& $projectPython -m local_ai_trader reprocess data/raw/coinbase_BTC-USD_300s_1775001600_1790640000.json --start "2026-05-09T00:00:00Z" --end "2026-09-29T00:00:00Z"
```

The command makes no API requests and applies the same OHLC, deduplication and
strict gap checks as collection. Bounds must be aligned and contained in the
original archive; its candle interval must match configuration. It preserves
the original raw responses and quality report, then writes a separate Parquet
snapshot and quality report recording the raw path, SHA-256, original bounds
and processing time. Existing snapshots cannot be overwritten. If the selected
range still has gaps, only a new gap report is saved. Failed writes or changes
to the source during processing prevent candle publication.

Choose the research range based on data availability before evaluating model
results, and record the excluded period. The command does not fill missing
candles, join disjoint periods, or change previously trained models.

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
Split tests check strict label boundaries, chronological ordering, row accounting,
settings compatibility and cleanup after a failed bundle write.
Baseline tests verify train-only scaling, class probability order, JSON inference
round trips, known metric values, failure cleanup, and that validation experiments
can run with an unreadable held-out test file.
Held-out tests forbid scaler/classifier fitting, preserve original experiment
files, verify frozen input hashes and reject partial or repeated publication.

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
and must be excluded from model inputs. Chronological splitting is
performed by the separate split command using `target_available_at`; it never
shuffles rows or randomly divides time-series data.

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

## Chronological splits

Create separate train, validation and held-out test files:

```powershell
.\.venv\Scripts\python.exe -m local_ai_trader split data/processed/features/coinbase_BTC-USD_300s_1790804400_1791409200_h6_features.parquet
```

The fractions are configured under `[split]` in `config/settings.toml`:
60% training, 20% validation, with the remainder reserved for testing. The cut
positions are `floor(N * train_fraction)` and
`floor(N * (train_fraction + validation_fraction))`.

After assigning the chronological periods, the code removes any training row
whose `target_available_at` is **at or after the first validation prediction
time**. It applies the same rule to validation against the first test prediction
time. This is stricter than just sorting and slicing: overlapping future labels
must not cross those boundaries. A label known exactly at the next period's
start is purged conservatively. The test rows already have known outcomes, so
no further end purging is necessary.

For the user's verified 1,998-row files:

| Partition | Before purging | Purged labels | Retained rows |
| --- | ---: | ---: | ---: |
| Train | 1,198 | 6 | 1,192 |
| Validation | 400 | 6 | 394 |
| Test | 400 | 0 | 400 |

Validation and test features may legitimately use older candle history. The
restriction applies to future labels entering earlier training/model-selection
periods; past prices do not become unavailable at an arbitrary split boundary.

Outputs are stored under `data/processed/splits/<feature-file-stem>/`:
`train.parquet`, `validation.parquet`, `test.parquet` and `split.json`.
The report records the source hash, feature allowlist and provenance, cut
positions, purged prediction times, class counts and availability bounds.
All files are staged and then published as one directory; failed writes do not
leave a partially published split bundle. Existing bundles cannot be overwritten.

The split command fits no scalers or models. The baseline command fits preprocessing
and model parameters only on the training file and reports validation metrics;
the test file is reserved for evaluation of frozen choices. Seven
days is an engineering sample, not sufficient evidence of predictive stability
or trading profitability. Longer history and walk-forward evaluation follow.

## First probability baselines

After pulling this milestone, install its new scikit-learn dependency from the
repository root. Use the already verified Python environment if reusing it:

```powershell
& $projectPython -m pip install -r requirements-dev.txt
& $projectPython -m pytest -q
```

Fit and compare models for one asset's verified split directory:

```powershell
& $projectPython -m local_ai_trader baseline data/processed/splits/coinbase_BTC-USD_300s_1790804400_1791409200_h6_features
```

The command fits two baselines on the training rows:

- **Naive:** constant class probabilities estimated from training counts, with
  a configurable Laplace pseudocount (`naive_smoothing = 1.0`). This avoids zero
  probability for a class absent from that training period.
- **Logistic regression:** `StandardScaler` fitted on training features, followed
  by a regularized classifier (`C = 1.0`, `lbfgs`, seed 42). Parameters live under
  `[baseline]` in `config/settings.toml`. Validation data is only transformed,
  never used to fit the scaler or model. Training needs at least two classes;
  a missing third class receives zero classifier probability. Failed convergence
  stops the run instead of publishing a checkpoint.

Both models produce ordered `p_up`, `p_down`, `p_flat` distributions. These small
classical models run on the CPU; GPU training comes with the custom PyTorch model.
The experiment reads only `train.parquet`, `validation.parquet` and `split.json`.
It verifies the manifest, availability boundaries and model-input allowlist.
**The held-out test Parquet is never opened or hashed by this command.**

The printed table reports validation accuracy, macro F1, log loss, Brier score
and expected calibration error. JSON metrics also include per-class precision,
recall/F1, one-vs-rest ROC-AUC where defined, and reliability-bin data. Train and
validation results are stored separately; test results are explicitly unevaluated.
Lower log loss and Brier score are the primary initial evidence of improvement
over the naive baseline, to be checked later across longer unseen periods.

Metric conventions are recorded in the output:

- Multiclass Brier score is the mean **sum** of squared errors over all three
  classes (range 0–2), rather than a mean over classes.
- ECE uses the most probable class, equal-width confidence bins and empirical
  top-class accuracy; the final bin includes probability 1. It is not a separate
  calibration guarantee for each individual class probability.
- ROC-AUC is null for a class whose validation truth has only one outcome.
  The three-class macro AUC is null if any component is undefined.
- Probabilities remain **uncalibrated**. A maximum score is saved as
  `max_probability`, not presented as reliable confidence. No calibration model,
  trading policy, expected-return regressor or backtest is implemented yet.

Each run publishes a new directory under `data/models/baseline_<symbol>_<run-id>/`
containing `checkpoint.json`, `experiment.json`, `metrics.json`,
`validation_naive.parquet` and `validation_logistic.parquet`. It records the
dataset provenance, input hashes, feature/class order, settings, seed, package
versions and timings. The JSON checkpoint includes model coefficients, intercepts
and fitted normalization statistics, and its inference is checked against the
fitted sklearn pipeline before publishing. A future inference step must use these
saved statistics, not refit preprocessing. Publication is staged so a failed write
does not leave a partially published model run.

Seven days is still an engineering sample. Adjacent 30-minute labels overlap
within each partition, so candle counts are not independent outcome counts.
Validation improvement alone establishes neither held-out performance nor
trading profitability. Keep the naive baseline and freeze choices before using
the test set; do not tune repeatedly against it.

## Frozen held-out evaluation

After recording validation results and freezing model/feature choices, evaluate
an existing model run on its original held-out test dataset:

```powershell
& $projectPython -m local_ai_trader evaluate data/models/baseline_BTC-USD_20261008T021502Z_d48a8cad
```

The example matches the user's verified BTC checkpoint. This command uses its
saved coefficients, normalization statistics, feature/class order, horizon and
metric settings. It does not read the current configuration, fit a scaler, train
a classifier or select parameters. It verifies hashes of the original split
manifest and training/validation inputs before loading the test partition.

For the verified seven-day dataset, both baselines evaluate **400 test rows**.
Outputs appear in the existing model run's `test_evaluation/` directory:
`metrics.json`, `naive.parquet` and `logistic.parquet`. The report identifies the
checkpoint and test-data hashes, states `partition = test` and `refitted = false`,
and preserves the original training and validation artifacts. The original
training report's `test.evaluated = false` remains a historical record; current
test results live in the separate evaluation directory.

The full result is staged before publication; an existing test-evaluation
directory prevents repeated execution or overwrite. To move saved datasets,
use `--splits <directory>`; the original hashes must still match.

Examining these outcomes consumes this engineering holdout. Further model
selection must use training/validation data, followed by fresh unseen data or a
new predefined research split. Do not repeatedly adjust the model against this
same test period. These probability metrics remain distinct from backtesting,
paper trading or live returns, and the probabilities remain uncalibrated.

## Gradient boosting

Install the updated project dependencies into the existing Python environment:

```powershell
Set-Location "$env:USERPROFILE\trade-ai"
$projectPython = "$env:USERPROFILE\local-ai-trader\.venv\Scripts\python.exe"
& $projectPython -m pip install -r requirements-dev.txt
& $projectPython -m pytest -q
```

For the verified 90-day research period, July 1 through September 29, 2026
(`start` inclusive, `end` exclusive, UTC), each asset has 25,920 raw candles,
25,914 labels and 25,902 feature rows. Its purged chronological partitions
contain 15,535 training, 5,174 validation and 5,181 test rows. Build naive/logistic
references on these same partitions before comparing XGBoost:

```powershell
& $projectPython -m local_ai_trader boosting data/processed/splits/coinbase_BTC-USD_300s_1782864000_1790640000_h6_features
```

The `boosting` command fits **only XGBoost** and prints its validation metrics
using the same definitions as the earlier naive/logistic runs. Compare models
on the exact same split manifest, feature allowlist and target definition. A
different dataset or different baseline parameters requires a new comparison.
Use validation to compare candidates; freeze choices before evaluating test
data. Reserve the 90-day test partitions while developing these candidates.

The starting parameters under `[boosting]` are 200 trees, depth 3, learning rate
0.05, minimum child weight 5 and L2 regularization 5. These limit the capacity
of the starting model; they were not selected using test results. The seed is
shared with `[baseline]`. CPU histogram training uses four threads, all training
rows and all eight features, without fitting a scaler. All three training
classes must be present; insufficient history produces a clear error.

Training uses no validation `eval_set`, early stopping, calibration fit or
automatic parameter search. The command reads/hashes only the split manifest,
training and validation files; it never opens or hashes test Parquet. Changing
validation inputs cannot change the fitted weights. Input changes during
fitting prevent publication.

A new `data/models/boosting_<symbol>_<run-id>/` contains:

- `xgboost.json`: native model weights and input names, without pickle.
- `checkpoint.json`: model hash, feature/class order, preprocessing convention
  and frozen training metadata.
- `experiment.json` and `metrics.json`: settings, actual model parameters,
  provenance, package versions, timings and separate train/validation results.
- `validation_xgboost.parquet`: probabilities, truth and prediction times.

Softmax probabilities are converted to float64 and normalized to correct
float32 rounding. This is recorded in the checkpoint and **is not probability
calibration**. Restored native-model predictions must match the fitted model
before the complete run is published. A failed write leaves no partial run.

Once model choices are frozen, the existing `evaluate <saved-run-directory>`
command also supports XGBoost. It verifies the native model hash and input/class
order, uses the saved horizon and metric settings, and performs no fitting. Test
results are saved once under `test_evaluation/`, including the model artifact
hash. It preserves the original run files. Do not evaluate the reserved test
partition during candidate development.

## First PyTorch MLP

The `mlp` command trains a small feedforward classifier on the same eight
features and purged partitions used by the classical baselines:

```powershell
& $projectPython -m local_ai_trader mlp data/processed/splits/coinbase_BTC-USD_300s_1782864000_1790640000_h6_features
```

Use the existing environment whose PyTorch 2.11.0+cu128 build was verified on the
RTX 3060. PyTorch remains optional for data/classical-model commands; it is not
reinstalled by `requirements-dev.txt`. For a new GPU environment, the separate
`requirements-gpu.txt` specifies the verified CUDA build. The generic `neural`
extra declares compatible PyTorch versions without selecting a CUDA build.

The initial network has hidden widths 64 and 32, ReLU activations, dropout 0.1,
and three output logits (2,755 trainable parameters). Settings under `[mlp]`
specify 30 epochs, batch size 256, AdamW learning rate 0.001, weight decay 0.001
and gradient clipping at norm 1. These are fixed starting settings, not selected
using test data. No validation early stopping or best-validation checkpoint
selection occurs: the saved model is the final fixed-epoch model.

`StandardScaler` fits only on training inputs, including its treatment of
constant/nearly constant features. The checkpoint stores the resulting mean,
scale and variance in input order; inference never fits a scaler. Training
batches follow chronological order. All three training classes must be present.
Initialization and dropout are seeded; deterministic PyTorch algorithms are
enabled with the required CUDA workspace setting. Exact reproduction across
different devices, library versions or CUDA builds is not promised.

GPU training is explicit (`device = "cuda"`): unavailable CUDA stops the run
instead of silently training on CPU. Mixed precision uses CUDA float16 autocast
and gradient scaling. Scaling-overflow steps are skipped and counted in the
training history; nonfinite loss or model weights stop publication. An explicit
`device = "cpu"` supports CPU verification, with mixed precision disabled.

Each epoch logs training loss. The saved experiment records epoch history,
training device, actual mixed-precision use, GPU name/peak allocated tensor
memory where available, seed, package versions and dataset hashes. Validation
metrics use full-precision inference and float64 softmax; scores are still
**uncalibrated**. This small model verifies the neural/GPU pipeline; a GPU or a
neural network does not establish a performance advantage.

A new `data/models/mlp_<symbol>_<run-id>/` contains `mlp_state.pt` (tensor state
dictionary), `checkpoint.json`, `experiment.json`, `metrics.json` and
`validation_mlp.parquet`. The model state is restored using
`torch.load(..., weights_only=True)` with strict layer matching, and restored
predictions must match the fitted model before staged publication. Training
never opens or hashes test Parquet. Changing validation data cannot change
normalization or weights; changed inputs or failed writes leave no partial run.

After choices are frozen, `evaluate <saved-run-directory>` supports the MLP
without fitting or reading current settings. Frozen evaluation defaults to CPU
for portability and records the state-file hash. The inference helper also
accepts an explicit CUDA device. Original experiment files remain unchanged.
Keep the 90-day test partitions reserved during model and calibration development.

The test suite includes CPU checks and a small CUDA/mixed-precision training and
checkpoint test. The CUDA test is skipped on machines without an available GPU.

## Temperature calibration

Temperature scaling adjusts probability sharpness with one positive scalar per
model: `softmax(log(max(p, probability_floor)) / temperature)`. Temperature 1 is
the identity, temperatures above 1 soften predictions, and temperatures below
1 sharpen them. The tiny probability floor handles zero scores. Positive
temperature preserves each row's predicted class, so accuracy and F1 do not
change; compare log loss, Brier score and reliability data. A fitted temperature
does not guarantee improvement on a later period or perfect calibration.

Fit a separate calibration bundle for an existing saved run:

```powershell
& $projectPython -m local_ai_trader calibrate data/models/boosting_BTC-USD_20261008T175708Z_5e9dc4f5
```

This example uses the verified 90-day BTC XGBoost checkpoint. The command
supports naive/logistic, XGBoost and MLP runs. It restores model weights and
preprocessing without fitting them and loads only **validation Parquet**.
Hashes audit the original manifest/training/validation files. The test Parquet
is never opened or hashed. Saved models are restored on CPU, so calibration
does not need another GPU training run.

With `[calibration].fit_fraction = 0.5`, the first half of validation is the
calibration-fit period and the second half is the assessment period. Fit rows
whose outcomes reach the first assessment prediction are purged. For each
90-day asset this leaves **2,581 fit rows**, six labels purged, and **2,587
assessment rows**. Fit labels are strictly known before assessment begins;
the original validation-to-test purge is also checked.

SciPy bounded scalar minimization fits temperature using only fit-period log
loss. Fixed bounds 0.25–4 include temperature 1; identity and both endpoints
are checked explicitly. Failed optimization stops publication. Assessment
labels and scores cannot influence temperature fitting, and the command does
not automatically undo calibration based on assessment results.

The printed table compares raw and adjusted scores on **the same later
validation rows**. Earlier full-validation tables used 5,174 rows and are not
directly comparable to this smaller assessment. The later segment already
contributed to earlier model comparisons, so it is a diagnostic calibration
assessment, not a fresh independent test. Keep the original 90-day test
partitions reserved while choosing models and calibration settings.

An immutable `<saved-run>/calibration/` contains:

- `calibration.json`: scalar temperatures, optimizer results, configuration,
  source/model hashes, purged temporal regions and class counts.
- `metrics.json`: separate fit-period and assessment metrics for raw and
  temperature-adjusted predictions.
- `assessment_<model>_<variant>.parquet`: assessment probabilities and truth.

Metrics retain the existing top-label ECE convention and now also include
`classwise_calibration_bins`, `classwise_ece` and `classwise_ece_macro`. Each
class's bin records mean predicted probability, observed event frequency and
count. These are reliability-curve data: for example, the `up` bins compare
`p_up` with how often `up` actually occurred. Empty bins have null means; counts
are not independent sample counts because adjacent horizon labels overlap.

Publication is staged, failed writes leave no partial bundle, and changed
inputs or model artifacts prevent publication. The original checkpoint,
weights and experiment metrics are preserved. Repeated calibration or
calibration after an existing test evaluation is refused. A `--splits` override
supports relocated datasets only when the original hashes still match.

After all choices are frozen, this optional evaluation mode reports raw and
saved-temperature predictions together, without fitting any parameter:

```powershell
& $projectPython -m local_ai_trader evaluate <saved-run-directory> --calibrated
```

It verifies calibration/model/input hashes and purged fit-label availability,
ignores current settings, and records the calibration checkpoint hash. Raw
and calibrated modes share the same immutable `test_evaluation/` destination,
preventing a second test evaluation. Missing calibration is rejected before
test data is loaded. Do not run this on reserved test data during development.

## Validation backtesting

The first simulator is a **research tool on later validation assessment data**.
It restores the saved model and temperature without fitting and identifies the
same assessment period recorded by calibration. It loads only validation
Parquet, audits original input/model hashes, and never opens or hashes test
Parquet or reads classification test metrics. It can run after a classification
test evaluation, but that test has then been consumed: policy development needs
a fresh chronological period for independent strategy evaluation.

```powershell
& $projectPython -m local_ai_trader backtest data/models/mlp_BTC-USD_20261008T181958Z_c65a0348 --model mlp --variant temperature
```

`--model` also accepts `naive`, `logistic` and `xgboost` when present in the saved
experiment. `--variant raw` permits a separate validation comparison. Calibration
is required for either variant to establish the assessment period. `--config`
selects current `[backtest]` assumptions; candle interval and horizon come from
the saved model. `--splits` permits relocation only with matching original hashes.
There is deliberately no test-partition option or threshold optimizer.

The initial policy is single-asset, spot, long-only, with one position and one
pending order at most. BUY requires `p_up >= entry_probability` and
`p_up - p_down >= minimum_direction_margin`. SELL closes an existing position
when the analogous down thresholds pass. Otherwise an open position is HOLD;
an unmet entry threshold is AVOID. These configurable demonstration rules are
not optimized. Class probabilities do not supply expected return, so the policy
does not claim that a signal covers trading costs or interpret max probability
as a separate confidence estimate.

Predictions become available at candle close. With `latency_bars = 1`, the
simulator waits one whole following candle and fills at the next open: a 00:05
signal fills at 00:10, never at the price used to generate that signal. BUY and
SELL signals have the same delay. Entry sizes spend `allocation_fraction` of
cash, including fees. There is no leverage, shorting, overlapping position or
rebalancing. A position exits after the saved horizon's number of bars **since
execution**, or an earlier delayed SELL. This holding window starts later than
the model's label window, which starts at prediction time. Remaining positions
are liquidated at the last open, a boundary fixed in advance; both sides pay costs.
The final candle's close and probabilities do not affect simulation results.

Default assumptions are $10,000 cash, 10% entry allocation, a 10% maximum
entry exposure, 10% drawdown entry limit, 0.55 directional probability and 0.15
directional margin. The separate risk module checks available cash, exposure
**after entry costs**, and drawdown at execution time. Exits remain permitted.
It blocks new entries at the drawdown limit; this is not a guaranteed loss cap.
Positions are marked at opens, so intrabar drawdown is unobserved and an existing
position's exposure can grow after entry as prices move. Full portfolio/daily-loss
limits and live paper-trading gates remain later work.

Costs are configurable: **60 bps (0.6%) fees each side**, 10 bps total spread,
and 5 bps adverse slippage each side. These are illustrative assumptions, not
your actual Coinbase account fee tier. BUY price is open plus half spread and
slippage; SELL price is open minus them. Fees apply to actual fill notional.
High turnover can therefore overwhelm small predictive advantages. No order-book
capacity, partial fills, minimum order sizes, taxes or cash interest are modeled.

Each immutable `<saved-run>/backtests/validation_<model>_<variant>_<hash>/`
contains `backtest.json` with assumptions, provenance and metrics, plus equity,
fill, trade and decision Parquets for policy, cash and buy-and-hold. Changed
settings create a different validation research bundle; an identical repeat
cannot overwrite results. Failed writes or changed inputs leave no partial bundle.
Buy-and-hold uses the same allocation, costs and first possible entry time,
holds until the final open, and does not use model signals. Each asset has its
own simulated account; these are not combined portfolio results.

Metrics include net return, fees, price-impact cost, gross price PnL attribution,
maximum drawdown, Sharpe, Sortino, win rate, profit factor, trade count, turnover,
exposure and blocked entries. Completed-trade net PnL must reconcile with final
cash. Turnover is both-side executed notional divided by initial capital. Annualized
ratios use open-to-open equity returns and 365-day crypto years with zero risk-free
return; a short validation window does not establish stable annual performance.
Undefined ratios, no-trade win rate and profit factor with no losses are null.
Gross PnL is cost attribution for actual quantities, not a cost-free simulation.

## Walk-forward planning

Freeze chronological fold datasets before fitting or reviewing fold performance:

```powershell
& $projectPython -m local_ai_trader walkforward-plan data/processed/features/coinbase_BTC-USD_300s_1778284800_1790640000_h6_features.parquet --model mlp
& $projectPython -m local_ai_trader walkforward-plan data/processed/features/coinbase_ETH-USD_300s_1778284800_1790640000_h6_features.parquet --model xgboost
```

`[walkforward]` defaults to rolling **60-day training, 14-day validation and
14-day evaluation** windows, advancing by 14 days. Windows use `available_at`
prediction times and start at the first complete UTC midnight after feature
warmup. Only full windows are included. Evaluation windows cannot overlap;
earlier evaluation history can legitimately enter later training once its
outcomes are known. Each fold will require fresh weights and train-only scaling.

For the verified May 9–September 29 feature datasets, the anchor is May 10 UTC
and four folds are planned:

| Fold | Training start | Validation start | Evaluation start | Evaluation end (exclusive) |
| --- | --- | --- | --- | --- |
| 1 | May 10 | July 9 | July 23 | August 6 |
| 2 | May 24 | July 23 | August 6 | August 20 |
| 3 | June 7 | August 6 | August 20 | September 3 |
| 4 | June 21 | August 20 | September 3 | September 17 |

Labels whose outcomes become known at or after a partition's end are excluded
in **all three partitions**, including equality. This removes six rows from
each boundary and prevents the last fold's labels crossing its evaluation end.
Each fold therefore contains **17,274 training, 4,026 validation and 4,026
evaluation rows**. Earlier validation will fit temperature; later validation
will assess it using the existing additional purge. Adjacent 30-minute targets
inside each partition still overlap and are not independent observations.

An immutable `data/processed/walkforward/<feature-stem>/` contains `plan.json`
and `fold_01` through `fold_04`, each with training/validation/test Parquets and
a compatible `split.json`. The plan records source hashes, hashes of every fold
file, unused leading/tail rows and fixed model/calibration/backtest settings.
The selected family is recorded along with naive/logistic controls and a raw
probability comparator. Source changes or failed writes prevent publication.

Calendar fractions are 60/88 and 14/88; the execution runner restores
these and the other frozen settings from the protocol. This command only
prepares folds: it does not fit models or report fold performance. Do not change
model or policy settings in response to individual evaluation-fold results.
These dates overlap previously examined research periods, so walk-forward
results are historical stability evidence. A fresh later period remains
necessary for an independent final strategy evaluation.

## Walk-forward execution

Run a saved plan with its frozen settings:

```powershell
& $projectPython -m local_ai_trader walkforward-run data/processed/walkforward/coinbase_BTC-USD_300s_1778284800_1790640000_h6_features
& $projectPython -m local_ai_trader walkforward-run data/processed/walkforward/coinbase_ETH-USD_300s_1778284800_1790640000_h6_features
```

There is no current-config or parameter override. Each fold fits fresh selected
MLP/XGBoost weights and fresh naive/logistic controls using training data only.
It then fits scalar temperatures on earlier validation, purges overlapping
outcomes, and assesses later validation. **Every fold's models and temperatures
are saved before any evaluation Parquet is read or hashed.** Evaluation outcomes
can enter later folds' training history, but never tune the frozen research
recipe. Raw and temperature-scaled probabilities are always retained together;
assessment results do not change the chosen variant or thresholds.

The plan's `execution/` directory contains each fold's model checkpoints,
calibration, paired evaluation predictions and eight cost-aware simulations:
raw/scaled naive, logistic and the selected family, plus cash and buy-and-hold.
Simulations use the frozen delayed fills, fees, spread, slippage and risk gates.
Policies receive candle prices and saved probabilities without future labels.
Positions liquidate at the predefined last selected open; the final purged
label rows are excluded from the simulated price timeline too.

Each fold starts with the same cash and resets positions and risk history.
`execution/summary/summary.json` reports pooled prediction scores, per-fold
performance, mean/best/worst fold returns, worst fold drawdown and totals for
trades and costs. This is a summary of separate reset accounts, not a continuous
portfolio or a compounded return. Evaluation windows do not overlap, but
30-minute labels within them do; confidence intervals are not computed.

If interrupted, rerun the same command with `--resume`. Saved checkpoints,
calibration and evaluation bundles are verified and reused, including bundles
whose atomic publication completed just before a journal-write failure.
Completed evaluations are not recomputed. Changed input or artifact hashes
stop execution. A `running.lock` prevents concurrent runs; after a hard process
crash, verify that the recorded process has stopped before removing a stale
lock and resuming. Failed writes retain previously completed research results.

The runner supports nested Windows paths longer than 260 characters using
extended absolute paths. Atomic Parquet writes reuse an open temporary file.
No registry edits, administrator access or folder relocation are required.
Automated tests run on Windows and Linux, including a long-path file round trip.

## Uncertainty diagnostics

After a walk-forward run is complete, describe its saved probability distributions:

```powershell
& $projectPython -m local_ai_trader uncertainty data/processed/walkforward/coinbase_BTC-USD_300s_1778284800_1790640000_h6_features
& $projectPython -m local_ai_trader uncertainty data/processed/walkforward/coinbase_ETH-USD_300s_1778284800_1790640000_h6_features
```

This first uncertainty milestone computes normalized Shannon entropy, top-class
probability and the margin between the largest two probabilities. Entropy is
zero for a deterministic distribution and one for a uniform three-class
distribution. It describes ambiguity within the prediction. Low entropy can
occur in an incorrect prediction; its complement is not calibrated confidence.
Top-class probability is retained under its explicit name.

The selected MLP/XGBoost distribution is also compared with logistic regression
at identical prediction times using equal-weight Jensen-Shannon divergence,
normalized by `log(2)`. Zero means identical distributions; one means disjoint
supports. Disagreement exposes different model views, but correlated models can
agree and be wrong. Raw and temperature-scaled variants are compared separately
against the same variant of logistic; those two variants of one model are not
treated as independent ensemble members. Naive controls retain entropy scores.

Reports require a complete, plan-bound execution and verify all recorded model,
calibration, evaluation, backtest and summary hashes. They reuse saved predictions
without fitting or running model inference. `execution/uncertainty/` contains an
immutable JSON report and six enriched prediction Parquets. Original results,
decisions and accounting remain unchanged. Five predefined entropy bands show
descriptive accuracy on already consumed outcomes; they do not select trading
thresholds or establish a new unseen evaluation.

Entropy is inexpensive and uses one forward prediction. Model disagreement
requires comparable model outputs. Monte Carlo dropout requires repeated model
passes, and independently trained ensembles require additional checkpoints;
these are later options when evaluation justifies their cost. None of these
scores estimates expected return after fees. A future cost-aware entry policy
needs return-size forecasts aligned with its execution and holding horizon.

## Cost-aware decision candidates

The separate `[decision]` table configures the future paper policy's probability,
entropy and model-disagreement limits and its required net-return margin. These
are engineering starting points, not thresholds optimized on historical test
results. `DecisionSettings` remains separate from frozen research `Settings`,
so adding this table does not invalidate previously saved walk-forward plans.

The policy requires a structured `ReturnForecast` before proposing a BUY. Its
symbol, prediction time, candle interval, entry latency and holding duration
must match the prediction and execution assumptions. Its target is fractional
reference-price return from **delayed entry open to horizon exit open**. Existing
close-to-close classification targets have different timing and cannot supply
this forecast. Current classifiers supply no expected-return estimate, so new
entries produce `AVOID / expected_return_unavailable`.

With per-side fee fraction `f` and adverse price impact
`i = (full_spread_bps / 2 + slippage_bps) / 10000`, entry outlay includes the
buy fee and exit proceeds deduct the sell fee:

```
factor = (1 - i) * (1 - f) / ((1 + i) * (1 + f))
expected_net_return = (1 + expected_gross_return) * factor - 1
required_gross_return = (1 + minimum_expected_net_return) / factor - 1
```

This return is per unit of entry cash outlay, not the whole account. The frozen
60-bps-per-side fee, 10-bps full spread and 5-bps slippage imply approximately
**1.4099% gross break-even**. The initial 0.2% net margin requires gross return
**strictly above 1.6127%**. This is a model-based mean threshold under fixed
costs, not a guaranteed realized profit. Earlier policy/risk exits can have a
different return than the fixed-horizon forecast.

Valid entries also require the configured up-probability and direction margin,
entropy limit and an aligned reference-model comparison by default. A BUY is
only a candidate: the independent risk gate must still authorize execution and
position size. SELL/HOLD logic for open positions does not depend on entry
return estimates, entropy limits or reference availability. Freshness and account
state checks belong to the future paper coordinator and risk gate.

Inspect an already completed run's last saved prediction with:

```powershell
& $projectPython -m local_ai_trader decision-preview data/processed/walkforward/coinbase_BTC-USD_300s_1778284800_1790640000_h6_features
```

The command writes one immutable `execution/decision_preview/preview.json` with
probabilities, uncertainty, cost hurdles, current decision settings and source
hashes. It assumes a flat position and uses the plan's original costs. It is a
historical diagnostic, not a current market signal or new strategy evaluation.
It creates no orders, fits no models, and does not alter recorded backtests.
The next forecasting milestone must train and validate an execution-aligned
return model before a real BUY candidate can reach the paper risk gate.

## Execution-aligned return targets

Prepare the return forecaster's supervised targets from the original complete
candles, using the current timeframe, holding horizon and latency assumptions:

```powershell
& $projectPython -m local_ai_trader execution-targets data/processed/coinbase_BTC-USD_300s_1778284800_1790640000.parquet
& $projectPython -m local_ai_trader execution-targets data/processed/coinbase_ETH-USD_300s_1778284800_1790640000.parquet
```

For a prediction at `available_at[t]`, entry occurs at
`timestamp[t + 1 + latency_bars]` and horizon exit occurs at
`timestamp[t + 1 + latency_bars + holding_bars]`. The regression target is the
fractional change between those **reference open prices**. A separate diagnostic
net label applies the same multiplicative fees/spread/slippage conversion as the
cost-aware policy. Its unit is entry cash outlay, independent of position size.
It represents a fixed-horizon outcome, not a replay of earlier SELL/risk exits.

With five-minute candles, one latency bar and six holding bars, a signal at
00:05 enters at 00:10 and exits at 00:40. This pipeline uses completed OHLCV
candles, so it conservatively marks the target as known at the exit candle's
completion, **00:45**. The label is available eight bars (40 minutes) after
prediction. Future splits must purge against `execution_target_available_at`;
the classifier's six-bar purge is insufficient for these targets.

Eight trailing source rows lack an exit candle and are removed. Each verified
143-day, 41,184-row source therefore yields **41,176 execution labels** before
feature warmup or split purging. Original causal candle columns are preserved.
Future entry/exit times, reference prices and both return labels are explicitly
marked as future-only metadata/targets, never model inputs. OHLCV, market,
availability and contiguous-timeline checks reject incomplete or malformed data.

An immutable `data/processed/execution_targets/<source-stem>_h6_l1/` contains
`labels.parquet` and `targets.json`. Metadata records source and label hashes,
timing offsets, formulas, conservative availability, costs and future-only
columns. Atomic publication prevents partial bundles and rejects source changes.
This command does not fit a model or evaluate a new strategy.

## Causal features for execution-aligned returns

Build the same eight trailing OHLCV features for the new return targets:

```powershell
& $projectPython -m local_ai_trader execution-features data/processed/execution_targets/coinbase_BTC-USD_300s_1778284800_1790640000_h6_l1
& $projectPython -m local_ai_trader execution-features data/processed/execution_targets/coinbase_ETH-USD_300s_1778284800_1790640000_h6_l1
```

The command takes a target **bundle directory**, not the classifier's target
Parquet. It verifies the original candle, label and metadata hashes, reconstructs
the execution targets from their original candles, and checks the complete
stored schema and timing contract before building features. Keep the original
candle Parquet at the path recorded in `targets.json`.

Only the nine completed causal candle columns reach feature calculation.
All future entry/exit prices, times, returns and availability are attached
afterward as labels. Changing future labels or future market data cannot change
earlier model inputs. Models must select the ordered eight `feature_names`,
never every numeric column. No normalization or global statistics are fitted;
eventual scaling must use the training partition alone.

With default settings, **41,176 execution labels become 41,164 feature rows**
after removing 12 leading warmup rows. The eight-bar label availability remains
unchanged and must drive future split purging. Existing classifier splits and
their six-bar purge cannot be reused for these targets.

An immutable `data/processed/execution_features/<target-bundle-name>/` contains
`features.parquet` and `features.json`. Publication is atomic, including long
Windows paths without a registry change. Metadata records the feature allowlist,
all future-only columns, upstream hashes, output hash, frozen target costs and
timing, and feature parameters. Current fee changes do not rewrite old diagnostic
net labels. No source artifact is modified, and reruns cannot overwrite output.
This stage does not train a return model or measure strategy performance.

## Chronological execution-return splits

Create separate regression partitions from an audited execution feature bundle:

```powershell
& $projectPython -m local_ai_trader execution-split data/processed/execution_features/coinbase_BTC-USD_300s_1778284800_1790640000_h6_l1
& $projectPython -m local_ai_trader execution-split data/processed/execution_features/coinbase_ETH-USD_300s_1778284800_1790640000_h6_l1
```

The command verifies all upstream hashes and reconstructs features and labels
before partitioning. It preserves the chronological order and feature values,
using the configured 60%/20%/20% cut positions. No scaling or model fitting occurs.
Retain a training or validation row only if `execution_target_available_at` is
**strictly before** the next partition's first prediction time. The default
execution labels require **eight purged rows at each boundary**, including the
row whose label becomes known exactly at the boundary. A six-bar classifier
purge would leak two extra rows of outcomes into each earlier partition.

For each verified **41,164-row** source, the default partitions contain:

| Partition | Raw rows | Purged rows | Retained rows |
| --- | ---: | ---: | ---: |
| Train | 24,698 | 8 | 24,690 |
| Validation | 8,233 | 8 | 8,225 |
| Test | 8,233 | 0 | 8,233 |

Unknown trailing labels were already removed by the target command. Test needs
no additional end purge, and validation/test features can use older candle
history. Eventual scaler fitting must use training rows only. The reserved test
partition has not been scored by a return model or new strategy; these dates
overlap previous classifier research and are not a fresh final strategy period.

An immutable `data/processed/execution_splits/<feature-bundle-name>/` contains
`train.parquet`, `validation.parquet`, `test.parquet` and `split.json`. Its manifest
records all input hashes, each partition's output hash, cut positions, purged
prediction times, label availability, the feature allowlist and source lineage.
Publication is atomic, supports long Windows paths and cannot overwrite an
existing bundle. Existing classifier datasets and experiment protocols are
unchanged. Return forecasting and train-only preprocessing are the next step.

## Execution-aligned return baselines

Fit two point-forecast baselines from the new regression splits:

```powershell
& $projectPython -m local_ai_trader return-baseline data/processed/execution_splits/coinbase_BTC-USD_300s_1778284800_1790640000_h6_l1
& $projectPython -m local_ai_trader return-baseline data/processed/execution_splits/coinbase_ETH-USD_300s_1778284800_1790640000_h6_l1
```

- **Mean:** the arithmetic mean of training `execution_target_return`, repeated
  for every prediction. It is a control for changing market drift.
- **Ridge:** `StandardScaler` fitted on training rows alone, followed by a
  regularized linear return regressor with intercept and deterministic SVD.
  The separate `[return_forecast]` table sets `ridge_alpha = 1.0`. It does not
  change the frozen classifier Settings or old walk-forward protocols.

Both use only the eight feature names. The supervised target is the fractional
gross reference-price return from delayed entry open to fixed-horizon exit open;
fees, spread and slippage remain the decision layer's cost conversion. The
regressor runs on CPU, needs no new dependencies, and is a starting comparison
before more complex return models.

For each prepared 143-day dataset, **24,690 training rows** fit the model and
scaler; **8,225 validation rows** assess them. The trainer verifies manifest
semantics, per-partition hashes, strict eight-bar purges, timed labels and
reference-price returns. It opens/hashes only `split.json`, `train.parquet` and
`validation.parquet`. The reserved `test.parquet` and full upstream candle,
target and feature datasets remain unopened. Do not invoke the dataset-preparation
loaders from a fitting phase: those loaders reconstruct entire historical sources.

The console reports validation MAE, RMSE, R-squared, mean error, mean forecast and
mean observed return. These are **fractions**: `0.001` is `0.1%`. Lower MAE/RMSE is
better; negative R-squared means the validation outcomes' own mean would have
lower squared error. That validation mean is only an explanatory metric baseline,
not a forecast available in advance. R-squared is `null` for constant outcomes.
Overlapping labels are not independent observations, and these error metrics
do not establish profitability, trade confidence or strategy performance.

An atomic immutable `data/models/return_baseline_<symbol>_<UTC>_<id>/` contains:

- `checkpoint.json`: training mean, exact scaler statistics, Ridge weights,
  market/timing basis, training label availability and input provenance;
- `metrics.json`: training diagnostics and validation errors; test unevaluated;
- `validation_mean.parquet` and `validation_ridge.parquet`: aligned point
  forecasts and historical outcomes, with fixed-cost net diagnostics;
- `experiment.json`: settings, model parameters, package versions, input hashes
  and a hash for every model/metric/prediction artifact.

JSON checkpoint inference uses saved weights and scaling, with no refitting or
pickle loading. Structured `ReturnForecast` packets carry the symbol, signal
time, interval, latency, holding horizon and `execution_open_to_open` basis
required by the cost-aware policy. Packets cannot precede or equal the last known
training label. Unconstrained Ridge outputs below `-1` are counted as invalid in
diagnostics and refused as forecast packets; predictions are never clipped.
These point forecasts supply no uncertainty interval or correctness probability.

This command does not change the existing historical decision previews, select a
trading threshold, backtest a new policy, or submit orders. A saved return model
still needs assessment and an explicit integration into inference and the
separate paper risk coordinator. No fresh independent strategy period is claimed.

## Historical decision preview with a saved return forecaster

Supply a trained return run to the existing preview command:

```powershell
& $projectPython -m local_ai_trader decision-preview data/processed/walkforward/coinbase_BTC-USD_300s_1778284800_1790640000_h6_features --return-run "data/models/return_baseline_BTC-USD_<UTC>_<id>"
```

Replace the placeholder with the actual saved run directory. `--return-model`
defaults to `ridge`; `mean` selects the same run's training-mean control. Without
`--return-run`, the original preview behavior is preserved. There is no option
to type in an invented expected return.

The command audits the saved classifier execution and return experiment,
including model/metric/prediction hashes and recorded training provenance.
Market, candle interval, holding period, execution latency, feature schema,
momentum and rolling windows must match. It locates the final classifier signal
in its frozen fold's feature table and projects only causal candle metadata and
the eight feature columns into **one** return inference. Future labels are never
inputs. The signal must occur after all return-model training labels were known.
Classifier inference is reused; no classifier or return model is fitted again.
Reserved return test and full return source datasets remain unopened.

The preview prints expected gross return, expected net return under the frozen
walk-forward costs, break-even gross return, required margin and the decision
reason. Source-run diagnostic fees do not replace the classifier plan's cost
assumptions. A forecast that fails cost coverage yields `AVOID`; a `BUY` candidate
still requires independent risk authorization and creates no orders.

An immutable report is saved under
`execution/forecast_decision_previews/<return-manifest-hash>_<model>_<policy-hash>/`.
It records the signal, point forecast, model and feature-source hashes, timing
basis, policy/costs and separate risk status. The original `decision_preview`
and all model/backtest/uncertainty bundles remain unchanged. Atomic publication
is refused if a model, classifier source or policy config changes during preview.

This is a historical flat-position explanation, not a current market instruction
or a new strategy evaluation. The classifiers' recorded outcomes were already
consumed by research, point forecasts supply no uncertainty interval, and a
fresh later period plus freshness/account/risk checks remain necessary for the
eventual paper coordinator.

## Independent paper entry risk checks

The pure `risk.paper.authorize_paper_entry` function accepts a BUY candidate,
timestamped quote and paper account snapshot. It sizes a simulated spot-long
entry and returns an approval or a specific refusal. It creates no orders,
changes no account state and never opens the saved historical datasets.
Historical AVOID previews do not request entry authorization.

The separate `[paper_risk]` table configures these starting limits:

| Check | Default |
| --- | --- |
| Entry cash budget | 10% of current total equity |
| Per-market / total marked exposure | 10% / 20% of projected equity |
| Maximum drawdown / UTC daily loss | 10% / 2% |
| Positions including pending entries | 2 |
| Signal age | Strictly less than 2 candles |
| Quote / account and portfolio mark age | Strictly less than 30 seconds |
| Approval lifetime | At most 5 seconds |
| Minimum entry cash outlay | $10 |

Markets, candle duration and holding horizon come from the primary configuration;
execution latency and modeled fee/spread/slippage come from its cost settings.
The signal must be interval-aligned, its execution delay must have elapsed, and
the quote must be from that eligible execution time or later. Future timestamps,
stale historical signals, invalid prices, missing current-day equity baselines,
duplicate holdings and inconsistent reservations are refused. The account must
explicitly be a paper account denominated in USD.

Current equity includes cash and every marked holding. Pending entries reserve
cash, count toward position and exposure caps, and conservatively reduce projected
equity by their possible entry costs; prospective gains do not increase it.
New quantity includes adverse fill impact and entry fees within the cash budget.
Projected equity, after those costs, must pass both loss limits and all exposure
limits. Loss-limit equality blocks an entry. Setting `new_entries_enabled = false`
pauses entries. **This entry gate must never veto a reducing exit**; SELL/HOLD
actions are not entry requests.

Approvals carry exact simulated sizing, fees, input fingerprint and account
version. They expire at the earliest of their lifetime, any input freshness
deadline or UTC midnight. `paper_authorization_is_current` reruns all checks and
refuses changed inputs, sizing or account versions. The paper coordinator
must recheck fresh inputs and commit against the same version atomically. These
functions alone provide no persistent ledger, transaction lock or order execution.
The existing immutable research protocols and historical simulator remain unchanged.

Exercise the gate without models or an account ledger:

```powershell
& $projectPython -m local_ai_trader paper-risk-demo --config config/settings.toml
```

This diagnostic uses **invented probabilities, a positive test return and synthetic
accounts**. Under the default config, one fresh candidate is approved; stale data,
paused entries, loss boundaries, pending reservations and position limits are
blocked. Nothing is saved or traded. The synthetic positive forecast is not a
forecast for either market and does not change historical AVOID decisions.

## Persistent paper account and execution coordinator

`paper.ledger.PaperLedger` stores a USD spot-long simulated account in SQLite.
It uses Python's standard library and makes no broker requests. Each operation
locks the account with `BEGIN IMMEDIATE`, refreshes observed marks, checks risk
against the locked version and saves cash, holdings, fills, closed trades and its
journal in one transaction. Full synchronous commits preserve SQLite recovery.
A failure before commit rolls back every change. Competing processes reread the
committed account rather than spending against an old snapshot.

Creation freezes the market/timing, cost and risk contract, hashes it and refuses
an existing file. Accounts start with the configured simulated cash, not a deposit.
An entry accepts a structured candidate and quote plus timestamped marks for
existing holdings. It obtains its own approval inside the transaction and checks
expiry again before commit; outside approvals cannot execute an order. Real-time
callers leave `now` unset so freshness uses the clock after the lock is acquired.
The explicit clock parameter is for deterministic tests/replay.

Every mutation requires a stable operation ID. Retrying identical inputs returns
the original result, including after restart, and adds no fill or account version.
Using the same ID with changed inputs is refused. Refusals are journaled too; a
new attempt needs a new ID. An optional expected account version refuses a stale
worker. There are no outstanding broker orders or persistent entry queues in this
coordinator: an approved fill commits immediately. Earlier signal scheduling
belongs to the upcoming current-data loop, and execution still checks latency.

Entries use the existing modeled adverse buy impact and entry fees within their
cash budget. Positions record actual fill time, original quote time and a holding
deadline starting at the fill. Full exits use fresh sell quotes with adverse
impact and exit fees. Policy, risk and manual exits bypass entry pauses and loss
filters. Holding-deadline exits require a quote from the deadline or later, use
stable position-specific IDs, and cannot accidentally close a replacement
position. Missed deadlines after downtime wait for a fresh quote; no past fill
price is invented. A late or early exit differs from the forecaster's fixed
open-to-open target window.

Daily limits keep the recorded UTC opening equity. When the account was flat at
midnight, the coordinator reconstructs that value from initial cash and prior
fills. An overnight holding requires explicitly supplied observed midnight marks.
If they are unavailable, new entries remain blocked and exits remain available.
Selling an overnight position cannot erase that day's missing baseline or losses.

Restart and `audit()` reconstruct cash and inventory from all fills, verify
modeled costs, closed trade PnL, operation hashes, version sequence and the last
journaled account/marks. Marked equity minus starting cash equals realized plus
unrealized net PnL. Status is read-only and reports the last marked values; it
does not fetch current prices. Runtime entry pause/resume is persisted separately
and cannot override a disabled frozen risk setting. Extended native Windows paths
support deeply nested account files without changing the user's registry.

Run a controlled persistence demonstration:

```powershell
& $projectPython -m local_ai_trader paper-demo --config config/settings.toml
```

This creates a new **synthetic** account under
`data/paper/synthetic_<UTC>_<id>/account.sqlite`, records an invented $100 reference
entry, retries it, reopens the database and fast-forwards the test clock to a $102
holding-deadline exit. It audits exactly two fills, one closed trade and no open
holdings. Its hand-picked prices and positive test forecast are not model outputs
or strategy performance. The console prints the saved account path. Paper
databases are ignored by Git; historical data, models and AVOID previews remain
unchanged.

Inspect a saved account without changing it:

```powershell
& $projectPython -m local_ai_trader paper-status "data/paper/<account-directory>/account.sqlite"
```

## Hyperliquid perpetual market-data integration

The current deployment target is **eight Hyperliquid perpetuals, 1× leverage,
long only, with USDC collateral**. Its separate `config/hyperliquid.toml` records
that contract, a provisional base-tier taker fee of **4.5 bps (0.045%) per side**,
and a minimum 30-day forward paper period. No referral, volume or staking discount
is assumed. This configuration has no wallet, private key or signing settings;
`live_execution_enabled` must remain false.

| Scanner symbol | Hyperliquid contract | Contract unit |
| --- | --- | --- |
| HOOD | `xyz:HOOD` | 1 underlying unit |
| NVDA | `xyz:NVDA` | 1 underlying unit |
| SOL/USD | `SOL` | 1 SOL |
| ETH/USD | `ETH` | 1 ETH |
| TSLA | `xyz:TSLA` | 1 underlying unit |
| BTC/USD | `BTC` | 1 BTC |
| PEPE/USD | `kPEPE` | **1,000 PEPE** |
| DOGE/USD | `DOGE` | 1 DOGE |

HOOD/NVDA/TSLA are HIP-3 equity perpetuals, not ownership of shares. Their `xyz`
metadata is fetched separately from validator-operated crypto perpetuals. Both
venues must report USDC collateral (token index zero); a change refuses the
observation. Display symbols retain the requested `/USD` labels, while settlement
is explicitly USDC. PEPE display prices divide contract prices by 1,000; candles,
model features, execution quantities and funding remain in **contract units**.
Daily quote notional is never multiplied by that conversion. HIP-3 deployer fee
scale and growth mode adjust the base fee assumption separately per contract;
currently the three configured equity perpetuals imply 0.9 bps per side before
account discounts. Fee settings are read again each observation.

The intended flow is **Source → Decision → Gate → Paper**. `MarketState` supplies
the same fields for live observations and a seeded, explicitly synthetic offline
universe: price, fractional 24h change, quote notional volume, causal momentum,
news and regime. Live news/regime are unavailable until their sources/models are
connected; generated data is never a fallback for a failed live feed.

The separate signal policy defaults to **60% calibrated probability, 70% validated
confidence and five occupied positions**, counting pending entry reservations.
`evaluate_entry_signal` names every rule, returning BLOCK or REVIEW when evidence
is missing/low. Confidence needs a separate validated reliability measure; model
probability and entropy are not relabeled as confidence. Passing this gate only
makes a candidate eligible for independent account risk authorization. HOLD,
AVOID and SELL do not create new entries; reducing exits use their own path.
These policy targets are saved in observations, but the source collector has no
connected decision model, review queue or execution coordinator yet. It does not
report an active signal gate or claim calibrated offline decisions/performance.

Read actual public **mainnet** market data without an API key:

```powershell
& $projectPython -m local_ai_trader hyperliquid-feed --config config/hyperliquid.toml
```

The command reads only the public `https://api.hyperliquid.xyz/info` endpoint:
perpetual metadata/asset contexts, eight order books and recent five-minute
candles. It does not use testnet prices. It has no exchange-order, withdrawal,
wallet or signing client. Its request allowlist also refuses unrequested action
types and extra signature/wallet fields. Transient HTTP/rate-limit errors receive
bounded retries; permanent failures surface without switching data sources.

The console prints display prices, contract best bid/ask, current hourly funding,
completed candle counts and causal feature counts. Book age uses the exchange's
millisecond timestamp and must be strictly under 30 seconds at validation and
again before publication. Reception does not freshen a stale book. Asset contexts
do not include an exchange timestamp; their reception time is recorded separately
and is not represented as an exchange event time.

Candle history and features are processed before books are requested, so a slow
history response cannot age an already received quote. A stale capture retries
the observation once using new exchange responses; persistently stale quotes
still refuse publication. A future-dated quote refuses immediately. Timestamp
errors identify the contract, signed age, exchange UTC, local UTC and required
age range. A negative age can indicate a PC clock behind exchange time; check
Windows **Settings → Time & language → Date & time → Sync now**. The collector
never changes the PC clock, permits future quotes or increases the age limit.

The default window contains 24 completed candles and 12 feature rows after the
original warmup. Hyperliquid's inclusive candle-close millisecond becomes
`available_at = open + 300 seconds`. The current unfinished candle is excluded
even if the endpoint returns it. Missing intervals, conflicting duplicates,
wrong coins/intervals, invalid OHLCV, crossed/empty books and invalid contract
metadata are refused. An interval boundary crossed during fetching is retried
with a new completed window. The existing eight trailing features are computed
without labels, fitted scaling or model inference.

A new immutable `data/hyperliquid/live/snapshot_<UTC>_<id>/` contains raw public
responses, eight completed candle and feature Parquets, and `snapshot.json` with
configuration/artifact hashes, quality, source times and contract context. Frames
use `exchange = hyperliquid` and symbols `BTC-USDC` / `ETH-USDC`; the manifest
explicitly identifies perpetuals. Publication is atomic and supports long native
Windows paths. Filenames replace the `xyz:` colon with an underscore, preserving
the original exchange identifiers inside reports and frames. A stale book,
advanced candle boundary or changed config before
publication leaves no published bundle. Runtime observations are ignored by Git.

Optionally collect each new completed five-minute window until Ctrl+C:

```powershell
& $projectPython -m local_ai_trader hyperliquid-feed --config config/hyperliquid.toml --watch
```

Watch mode checks for the next boundary every ten seconds and archives once per
completed window, avoiding repeated full-window files within the same candle.
Data errors are logged and retried; stopping is clean. This is a data collector,
not a continuous trading or liquidation service.

Hyperliquid pays funding hourly. A positive rate charges longs; a negative rate
credits them. The documented payment is **position size × oracle price × actual
hourly funding rate**, using the oracle rather than mark price. The context's
current rate is not guaranteed future funding. These observations do not charge
funding to an account or fabricate paper fills/performance.

The earlier Coinbase datasets/models are historical spot research, and the
existing persistent ledger settles simulated spot buys in USD. They are not
silently relabeled or connected as a Hyperliquid perpetual strategy. Next work
must connect the separate USDC accounting foundation to observed funding, venue
liquidation rules and venue-bound model training/inference before starting the
actual forward paper trial. Merely running
the data collector does **not** start its performance clock. A profitable month
must include modeled fees, slippage and funding; it is a review milestone and
does not enable live execution automatically. Live execution requires a separate
implementation and explicit approval at that time.

An execution adapter boundary makes a future broker implementation replaceable;
going live also needs signed order handling, exchange/account reconciliation and
venue-specific risk checks. It is not enabled by changing one paper-fill file.

Primary references: [public API](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint),
[funding](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding),
[fees](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees), and the
[official Python information client](https://github.com/hyperliquid-dex/hyperliquid-python-sdk/blob/master/hyperliquid/info.py).

## Next milestones

1. Connect recorded funding, venue liquidation rules and Hyperliquid-trained models.
2. Connect the saved models and public feed to a scheduled forward paper loop.
3. Add a local API and dashboard showing forecasts, uncertainty, decisions,
   risk refusals and paper-account results.

More complex models remain conditional on evidence. Training, validation, test,
backtest and paper results will be reported separately. The current models and
historical policies have not demonstrated a profitable edge after costs.

## USDC perpetual paper accounting foundation

`PerpetualPaperAccount` is separate from the historical USD spot ledger. Its
virtual USDC wallet retains collateral principal; a 1× position reserves initial
margin equal to quantity × entry fill price. Wallet cash moves only for fees,
realized fill-price PnL and recorded funding. Equity adds unrealized mark-to-market
PnL. Available collateral uses the lower of wallet and equity minus reserved
margin, preventing unrealized gains from being spent. Quantities and oracle
prices remain in exchange contract units, including `kPEPE`.

The account freezes its source mode, policy, risk profile and slippage assumption
at creation. Defaults preserve 10% per-position exposure, 20% total exposure,
10% drawdown and 2% UTC daily loss. Five positions is a ceiling; total exposure
still applies. Independent signal checks require calibrated probability and a
separately validated reliability measure. The trusted model adapter must verify
their artifact references; nonempty IDs are audit references, not a calibration
procedure performed by the account. Missing reliability routes to REVIEW.
Entries also require fresh portfolio marks, delayed causal signal timing and
forecast coverage of spread, slippage, both fees, a nonnegative funding cost
budget and a 0.2% net margin. Missing funding-cost estimates route to REVIEW.

Funding requires a settled UTC-hour rate with provenance and an oracle recorded
within 30 seconds of that settlement. Positive rates debit longs, negative rates
credit them, using **quantity × oracle price × settled hourly rate**. Current
context funding is not used as a guaranteed settled rate. Oracle observations
near an exchange settlement are explicitly modeled estimates in observed-mode
accounts, not reconstructed exact historical exchange oracles. Synthetic funding
cannot be inserted into observed-mode accounts.

New accounts preserve the actual published funding event milliseconds, not just
the UTC-hour label. A position opened or closed between the nominal hour and
actual settlement is charged only if held at the real event. Until its event is
known, entries in the first 30 seconds of an hour have conservative unresolved
funding coverage. Older hour-based accounts retain their original semantics;
they are not silently converted into observed accounts with actual event timing.

Every potentially held settlement must have a recorded event. Missing hours make
performance provisional and block new entries. They do not block reducing exits.
Later funding can settle a closed position's outstanding liability, updating its
net PnL and wallet without inventing another fill. The same coin/hour cannot be
charged twice or silently changed. Holding deadlines start at actual fill time;
missed deadlines use current fresh quotes. Overnight holdings without a known
UTC opening-equity baseline block entries for that day; closing them does not
erase the missing baseline. Pauses and loss limits never veto a fresh exit.

SQLite `BEGIN IMMEDIATE` and full synchronous transactions save the operation,
state and checksum chain together. Stable operation IDs replay their original
receipts across restarts, including blocks; changed inputs refuse. Quotes and
the five-second execution window are rechecked before commit. Startup and status
perform complete deterministic journal replay; mutations check the current state
and journal head, and duplicate replays verify their own receipt checksums. A
normal exception or worker death before commit leaves no partial fill or margin
reservation. Deep native Windows paths work without a registry change.

Run a controlled accounting demonstration:

```powershell
& $projectPython -m local_ai_trader perpetual-paper-demo --config config/hyperliquid.toml
```

It creates an explicitly **synthetic** account under
`data/hyperliquid/paper/synthetic_perpetual_<UTC>_<id>/account.sqlite`, opens an
invented-price long, verifies a duplicate after restart, posts one invented
hourly funding charge, retries it without double charging, then exits after
restart. Two fills, one closed trade, released margin and no missing funding must
reconcile. Prices, probabilities, reliability inputs, forecasts and funding are
invented test data; any positive PnL is not strategy performance. Existing live
observations and spot accounts remain separate. No new dependencies are needed.

Inspect last recorded values without fetching prices:

```powershell
& $projectPython -m local_ai_trader perpetual-paper-status "data/hyperliquid/paper/<account-directory>/account.sqlite"
```

This foundation models conservative collateral reservation, not exact isolated
or cross-margin liquidation mechanics. It is not yet wired to a live strategy or
liquidation monitor, and does not start the 30-day trial. The recorded-funding
adapter described below is available for the future coordinator.
Venue-specific margin/liquidation rules and verified model/source adapters must
be connected before that trial begins. Live execution remains unavailable.

## Recorded Hyperliquid funding and oracle evidence

Read the latest 24 UTC funding hours for the configured eight contracts:

```powershell
& $projectPython -m local_ai_trader hyperliquid-funding --config config/hyperliquid.toml
```

The collector uses only public mainnet `/info` funding history, DEX metadata,
asset contexts and order books. It preserves the actual publication timestamp
(typically milliseconds after the hour), settled signed rate, premium and the
time the response became available. Invalid/future event times, conflicting
same-hour records, wrong coins and nonfinite rates are refused. Missing rate
hours are named, never replaced by current context funding forecasts.

Funding history does **not** contain historical oracle prices. The collector
records an observed current oracle with its reception time and an independently
fresh exchange-timestamped book. Contexts have no exchange timestamp, so these
observations remain explicitly modeled proxies. Only a recorded oracle within
strictly 30 seconds of the actual funding event can pair with its settled rate.
The closest valid observation is chosen deterministically. A current mid-hour
oracle cannot supply an unobserved old settlement; no forward-fill, interpolation
or mark-price substitution is performed.

An immutable `data/hyperliquid/funding/capture_<UTC>_<id>/` stores `raw.json`, the
exact input TOML and `capture.json`, with artifact/config hashes and all receipt
times. Loading recomputes the summary from raw evidence. Publication takes an OS
lock, checks prior economic values for conflicts, rechecks configuration and book
freshness, and atomically renames the staged directory. A failed publication
leaves no capture. OS locks release on process death without requiring manual
stale-lock removal; their small marker file is retained. Native deep Windows
paths work without enabling a registry setting.

Collect upcoming settlement windows until Ctrl+C:

```powershell
& $projectPython -m local_ai_trader hyperliquid-funding --config config/hyperliquid.toml --watch
```

At the default ten-second polling interval, watch mode archives oracle samples
in the 30 seconds before and after each hour, plus changed settled history.
Away from those windows it checks once a minute and avoids repeated full-response
bundles when rates are unchanged. Funding history is cached for up to 60 seconds
and refreshed when the hour changes. Near-hour samples remain available when a
settled rate appears later. Preview pairing uses only the latest 24 funding hours
and relevant captures rather than scanning a growing month of data each tick.
Data errors are logged and retried without inventing observations.

On first launch, **zero accounting-ready settlements and 192 rates lacking
historical oracle evidence can be normal**: 24 hours × eight markets were observed
after their settlements. As future hourly oracle windows are recorded, those new
events become ready when their rates are published. The collector itself creates
no account, fill, model or order, and does not start the forward strategy clock.

`apply_recorded_funding` is the future coordinator adapter. It accepts only an
observed-mode account with actual event timing, audits the archive, posts only
that account's missing eligible events with stable IDs and leaves unmatched
hours provisional. Recorded response availability and oracle times must precede
application, preventing retrospective future-data use. Repeating the adapter
does not charge another settlement or create a trade. Observed oracle proxies
remain estimates in account reports. Existing synthetic accounts cannot consume
these mainnet records.

## Hyperliquid research preparation for all eight markets

In a second PowerShell window, while the funding collector continues running:

```powershell
& $projectPython -m local_ai_trader hyperliquid-history --config config/hyperliquid.toml
```

This one operation requests recent five-minute candles for every configured
contract, verifies current DEX metadata and USDC collateral, and prepares causal
features, execution-aligned gross-return/class labels and chronological
60/20/20 partitions. The public candle API retains at most 5,000 candles per
interval. The default requests **4,999 closed candles** (about 17.36 days), leaving
room for an inclusive forming candle, which is removed. This is recent bootstrap
data, not a replacement for collecting a longer venue-specific history.
`--bars` accepts 128–4,999; small datasets may be explicitly insufficient.

Prices, volumes and model inputs remain in native contract units, including
kPEPE. Only the original eight `feature_*` columns are model inputs. Prediction
time is the current candle's completion; entry is the next five-minute opening
after a one-bar delay, holding is six bars (30 minutes), and outcome availability
is conservatively eight bars (40 minutes) after prediction. Gross returns use
`open[t+8] / open[t+2] - 1`. A fixed ±0.2% gross-return threshold produces
down/flat/up labels; it is a research definition rather than a profitability
threshold. Fees and funding are excluded from gross labels and must be considered
by the eventual cost-aware decision/account adapters.

Missing candles are reported by timestamp and never filled. Every contiguous
segment restarts feature warmup and removes unknown trailing labels, so neither
rolling statistics nor future returns cross a gap. Short segments contribute no
research rows. Labels must be known **strictly before** the next partition's
first prediction, including removal at the equality boundary. Prepared markets
require at least 128 training, 64 validation and 64 test rows after purging;
insufficient markets preserve their raw/cleaned data and are named explicitly.
These minimums are dataset checks, not evidence of model quality or reliability.

An immutable `data/hyperliquid/research/history_<UTC>_<id>/` contains exact config
bytes, raw public responses with per-request receipt times, a `history.json`
protocol/quality manifest, and per-contract candle, feature and partition
Parquets. Windows-safe directory names retain original contract identifiers in
metadata. Atomic staging, checksums and a full reconstruction audit verify each
derived artifact before publication; failures leave existing bundles intact.
`load_history_bundle` is a **preparation audit** that reads all three partitions.
Future fitting code must read only its declared partitions and must not invoke
that full audit while promising to leave reserved test outcomes unopened.

A genuine mainnet run prepared all eight markets with 4,999 closed candles,
4,979 research rows, 2,979 training / 988 validation / 996 reserved test rows, and
eight purged labels at each earlier boundary. This prepares data without fitting
models, estimating validated reliability, scoring a strategy or starting the
30-day paper trial. Original Coinbase experiments remain separate. The next
model milestone uses this venue-specific data rather than connect Coinbase
checkpoints to new markets.

## Train and calibrate all Hyperliquid markets

Use the directory printed by `hyperliquid-history`:

```powershell
& $projectPython -m local_ai_trader hyperliquid-train "data/hyperliquid/research/history_<UTC>_<id>" --device cuda
```

The batch fits training-frequency and scaled logistic controls, fixed XGBoost
and a 30-epoch MLP, plus training-mean and scaled Ridge gross-return forecasts
for every prepared market. Defaults reuse seed 42, logistic C=1, 200 depth-three
XGBoost trees and a 64/32 MLP. Training is explicit: CUDA absence refuses the
requested GPU run before publishing any batch. `--device cpu` is available for
an explicitly chosen CPU run and automated verification. PyTorch must be present
in the selected project interpreter.

Base-model fitting and preprocessing use training rows only, without early
stopping, validation eval sets or validation-driven hyperparameter changes.
Logistic and Ridge parameters are saved as strict JSON; XGBoost uses its native
JSON and MLP restoration loads tensors with `weights_only=True`. Saved inference
reproduces native predictions without refitting and checks contract, features,
class order, source hashes and fitting-label availability. Model probabilities
use the explicit **up/down/flat** order; source string labels are encoded by
name rather than reusing a different preparation order.

Earlier validation fits each model's scalar temperature. Labels reaching the
later-validation boundary are purged before fitting; all weights and temperatures
freeze before the later assessment is predicted. A gap-free default market has
2,979 training rows, 486 calibration rows, eight purged rows and 494 assessment
rows. Raw/scaled class metrics and mean/Ridge return errors are reported on that
later validation slice. Temperature fitting can worsen later metrics and is not
a promise of calibrated future performance. Fixed primary families are BTC MLP
and XGBoost for the other seven markets, selected before calibration/assessment;
results never choose a favorable replacement. Missing training classes publish
explicit family unavailability, with no substitution for the primary model.

The fitting loader reads/hashes only `history.json`, its exact copied config and
per-market train/validation Parquets. It never opens, stats or hashes reserved
test, raw, candle or full-feature files. Source preparation's manifest establishes
lineage; the whole-source reconstruction audit is not called during fitting.
Training/validation schemas, gross-return labels, class boundaries, prediction
times and partition purges are checked against the frozen protocol.

Atomic `data/hyperliquid/models/models_<UTC>_<id>/` bundles contain all market
checkpoints, native weights, temperatures, later-validation predictions, metrics,
training history/hardware, recipe and source hashes. Output follows the history
archive's data root, so relative paths in a copied config do not create a nested
project. Failed fits, changed inputs and failed writes publish no partial batch.
XGBoost native JSON is saved through Python IO and restored from checked bytes;
PyTorch weights use open Python file handles. This avoids native filename limits
on deep Windows directories while preserving compatibility with earlier model
files. No system registry changes are required.

`predict_saved_market` provides calibrated probabilities and gross-return point
forecasts from explicitly selected saved models on causal rows, without future
label columns. Its calibration ID binds the checkpoint. **Confidence and its
reliability-validation ID remain unavailable:** probability, entropy and later
validation accuracy do not establish the separately validated 70% reliability
measure. Otherwise eligible BUY candidates route to REVIEW through the existing
signal gate. These are research models without demonstrated profitability; no
orders, paper fills or forward strategy month are created by this command.

## Live model scanner and cost-aware entry previews

Keep the funding collector open. In another PowerShell window, use the model
directory printed by `hyperliquid-train`:

```powershell
& $projectPython -m local_ai_trader hyperliquid-decisions --models "data/hyperliquid/models/models_<UTC>_<id>" --config config/hyperliquid.toml --watch
```

This connects the saved primary models and Ridge return forecasts to fresh
Hyperliquid data for all eight markets. Watch mode archives one preview per
five-minute window; it does not retrain or read reserved test data. Source/model
market rosters, feature parameters, execution-label timing and model hashes must
agree. The eligible signal is the **previous** completed-window feature row:
its one-bar delay has elapsed when the current quote arrives. Using the newest
row would skip the configured execution delay. Unavailable primary models are
named and never replaced by another family or an invented return forecast.

The flat-position preview uses calibrated **up** probability, not the maximum
class probability. Entry requires the configured 60% minimum and a Ridge gross
return covering the current bid/ask spread, both taker fees, one basis point of
slippage per side, a current-rate funding budget and the 0.2% net margin. The
funding budget assumes one potential hourly settlement at the current positive
rate and oracle/entry-price ratio; negative rates supply no hoped-for credit.
This is an expected-cost assumption, not a settled funding payment or a maximum
loss bound. The scanner and perpetual account share the same cost function and
net-PnL denominator (entry fill notional).

Otherwise eligible BUY candidates pass through the existing signal gate and
remain REVIEW while validated reliability is unavailable. Low probability or
insufficient cost coverage produces AVOID with its rule. No scanner result
authorizes account risk or fills. The flat assumption does not represent real
inventory, available collateral, position limits, daily loss or liquidation
checks. Execution integration and separately validated reliability remain work
before the forward strategy trial starts.

Immutable `data/hyperliquid/decisions/preview_<UTC>_<id>/` bundles contain raw
market responses, exact config bytes, model/checkpoint references, causal inputs,
per-market forecasts, modeled costs and gate reasons. Quotes and mark/funding
contexts must remain fresh at publication; a moved candle boundary, stale/future
data or changed config/model refuses publication. Staging failures leave no
partial preview. These observations create no account, order, fill or performance
clock. `Ctrl+C` stops the scanner independently of the funding collector.

## Prospective forecast outcomes alongside the scanner

Stop only the decision scanner with `Ctrl+C`, update the branch, then restart
with the same saved models and add `--track-outcomes`:

```powershell
& $projectPython -m local_ai_trader hyperliquid-decisions --models "data/hyperliquid/models/models_<UTC>_<id>" --config config/hyperliquid.toml --watch --track-outcomes
```

The entry preview still uses the delayed eligible signal. Separately, the
tracker records the **newest completed signal** before its next future entry
opening is known. These two rows are inferred together from the same frozen
weights. Probabilities, causal features, Ridge/mean point forecasts, issuance,
publication and source-preview hashes persist in a dedicated observation book.
Issuance/publication and first insertion must all precede the one-bar-delayed
entry; a late restart never backfills an unrecorded prediction after entry.

At 40 minutes after the signal, completed real candles can resolve its delayed
entry-open to six-bar exit-open return. Earlier resolution is refused even if
future-looking inputs are supplied. The tracker first reuses the archived
scanner candle response, then requests the exact public historical window when
the pending signal falls outside that response. Missing intervals remain pending,
without interpolation or fabricated outcomes. Long outages beyond API retention
can leave permanently unavailable outcomes.

The model/config-bound `data/hyperliquid/forward/<hash>/observations.sqlite`
retains first-issued forecasts and resolved raw candle evidence. SQLite immediate
transactions and full synchronous commits make deliveries restart-safe; duplicate
forecasts/outcomes do not add counts, changed predictions or resolved economics
refuse, and a failed batch rolls back. Startup audits reconstruct outcomes from
raw candles, check record hashes and reconcile counters, including deletion
detection. Watch mode caches the audited book and uses SQL diagnostic aggregates
rather than reconstructing a growing month of outcomes every tick. Model or
configuration changes select a separate observation book.

Initially expect **8 recorded, 8 pending, 0 resolved** if all eight primary
models are available. More predictions accumulate each five-minute window; the
first outcomes mature 40 minutes after their signal time. Descriptive accuracy
includes flat labels; the separate up-probability-at-least-60% hit counts show
that directional subset. Ridge and mean MAE are gross-return forecast errors.
Adjacent targets overlap and markets correlate, so neither row counts nor these
descriptive metrics are independent samples, a confidence interval, validated
reliability, net trading profit or approval to enter. Confidence remains
unavailable and the **paper-profitability trial has not started**. This option
collects the genuinely prospective evidence needed for later reliability work;
it never fits models or opens reserved test partitions.

## Observed paper-book monitor alongside the scanner

Add `--paper-monitor` to the same scanner command to maintain a separate,
**entry-paused paper account** alongside forecast tracking. Stop only the scanner
with `Ctrl+C`; keep the funding collector running. A self-contained Windows
launch, using your actual saved model directory, is:

```powershell
Set-Location "$env:USERPROFILE\trade-ai"
$projectPython = @(
    "$PWD\.venv\Scripts\python.exe"
    "$PWD\venv\Scripts\python.exe"
    "$env:USERPROFILE\local-ai-trader\.venv\Scripts\python.exe"
    "$env:USERPROFILE\local-ai-trader\venv\Scripts\python.exe"
) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $projectPython) { throw "No project Python environment found." }
& $projectPython -m local_ai_trader hyperliquid-decisions --models "data/hyperliquid/models/models_<UTC>_<id>" --config config/hyperliquid.toml --watch --track-outcomes --paper-monitor
```

The monitor creates `data/hyperliquid/paper/monitor_<hash>/account.sqlite` with
**10,000 simulated USDC**, an observed-source contract, the fixed risk/cost policy
and model/config binding. This balance is fictional paper collateral. Synthetic
demo accounts and the existing prospective forecast book remain separate.
The account and run manifest publish together; restart replay-audits the journal
and resumes that book. A missing account under an existing run is an error rather
than a balance reset. Model/config changes select a separate monitor book.

Each new five-minute scanner observation rechecks raw books and received mark
contexts before writes. Contract prices, quantity precision and venue fees must
agree with archived evidence, preserving native `kPEPE` units. The account stores
marks and immutable source decisions with preview hashes; each observation
receipt includes the independent signal-gate result, execution skip, reconciled
account values and a checksum. Duplicate previews preserve the first receipt and
never add another mark event. A crash after account commit but before receipt
publication reuses the original operation. A running clock rechecks the
five-second operation deadline and quote freshness before transaction commit.

The monitor **never calls entry execution**, even if a supplied signal passes
the gate. An entry switch changed externally is paused again. Missing reliability
continues to route otherwise eligible BUY signals to review; AVOID decisions
stay blocked. No review notification is sent externally. Venue margin/liquidation
integration and independently validated reliability remain before enabling a
strategy trial. Zero fills, a flat paper balance and elapsed monitoring days do
not establish profitability or start the required month.

Maintenance can mark existing holdings, close due holdings against a current
quote at or after their actual deadline, and post accounting-ready recorded
funding through the existing audited archive adapter. Reducing exits precede
funding archive work, so unavailable or corrupt funding evidence cannot trap a
holding. Missing funding stays provisional; no historical oracle or fill price
is invented. This runs once per scanner window and depends on successful scanner
captures; it is not yet a continuous risk/liquidation engine. The monitor runs
before potentially slower prospective outcome resolution. Live execution stays
disabled, and no exchange trading credentials are needed.
