# Trade AI

A local, Windows-first market research project for an NVIDIA RTX 3060 (12 GB).
The current milestone collects and validates historical data, builds
forward-return targets, computes causal features, and creates purged chronological
splits, and compares naive/logistic probability baselines on validation and
held-out test data using separate commands.
It does not yet make trading recommendations or execute orders.

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

## Next milestones

1. Compare naive, logistic, XGBoost and MLP validation results on the predefined
   90-day research partitions while reserving test data.
2. Review temperature calibration on purged later-validation assessment periods,
   then freeze model/calibration choices before the reserved test evaluation.
3. Add cost-aware backtesting, walk-forward evaluation and local experiment logs.
4. Add a Transformer only when evidence warrants it, then uncertainty, configurable
   policy, a separate risk gate, paper trading, API and dashboard.

Every change must demonstrate improvement against the previous model on suitable
unseen data. Training, validation, test, backtest and paper-trading performance
will be reported separately. A profitable backtest does not establish live profitability.
