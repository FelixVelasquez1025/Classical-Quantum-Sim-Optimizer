# Simulator selection

The public workflow has one collector, one trainer and one predictor. The model
is a boosted winner classifier with a separate memory-feasibility filter. All
features are static; prediction neither runs competing simulators nor estimates
runtime. See the [model card](../models/README.md) for measured results and limits.

## 1. Collect benchmark data

Place OpenQASM 2 files in directories grouped by algorithm family. For example:

```text
circuits/
  qft/
    8.qasm
    16.qasm
  grover/
    4.qasm
    6.qasm
```

```sh
.venv/bin/python scripts/collect_simulator_data.py \
  --circuits /path/to/circuits \
  --task shots \
  --output data/simulator-runs
```

The script recursively imports `.qasm` files, records hashes and import errors,
then benchmarks each successfully imported circuit on all four simulators.
Shot tasks require explicit measurements; none are inserted automatically.
Use `--task auto` (the default) to choose shots for measured circuits and unitary
evolution for unmeasured circuits.

Defaults are 1,000 shots, five timed repetitions, one warmup, four native threads,
one parallel-shot worker, 1 GiB backend working memory and a one-hour timeout per
circuit/backend pair. `--timeout-seconds 0` removes the timeout. The memory setting
is not a process RSS cap. Pairs run sequentially in isolated worker processes.

Circuit files in a family directory use that directory's name as the family;
common size suffixes such as `_n16` and `_16` are stripped. Flat directories use
file stems instead. Organize related sizes and variants under the same family
name, and inspect the generated manifest before relying on grouped validation.

Already normalized inputs can use `--manifest FILE --imports DIR` instead of
`--circuits`. Each benchmark run saves `run.json`, its frozen `manifest.json`,
`results.jsonl` and `labels.jsonl`. Normalized artifacts remain alongside the
runs in an `imports-*` directory. Keep these artifacts for training and resume.

```sh
.venv/bin/python scripts/collect_simulator_data.py \
  --resume data/simulator-runs/RUN_DIRECTORY
```

Resume validates the saved settings and completed observations, then computes
only missing pairs. Do not pass new circuits to a resumed run. Historical local
runs with pinned runner hashes should continue using their original private
collector; the public collector supports resuming runs it created.

## 2. Train the selector

```sh
.venv/bin/python -m ml.train_selector \
  --run data/simulator-runs/RUN_DIRECTORY \
  --output data/models/my-selector
```

Repeat `--run` to combine compatible collections. The loader checks schema,
workload settings and provenance, deduplicates normalized circuits, and groups
related algorithm families. Source artifacts are hash-checked before extracting
structural features. Repetitions contribute a median timing, not extra examples.

The trainer uses the published architecture and fixed parameters. It evaluates
with family-grouped folds, then fits one final model. `--folds` and `--seed`
control evaluation and reproducibility; there are no model-comparison or tuning
modes. Existing output directories are never overwritten.

Outputs are `selector.json`, `report.json`, `training-data.jsonl` and `run.json`.
Winner training uses only resolved comparisons. Memory training uses successes
and memory/resource failures, excluding timeouts, missing results and generic
errors. Reports retain unresolved cases and measure failures and coverage as
well as exact-winner accuracy and slowdown. Fresh held-out families are still
needed for a final generalization claim.

## 3. Predict a simulator

The bundled model is the default:

```sh
.venv/bin/python -m ml.predict_selector --circuit /path/to/circuit.qasm
```

For a newly trained model:

```sh
.venv/bin/python -m ml.predict_selector \
  --model data/models/my-selector/selector.json \
  --circuit /path/to/circuit.qasm
```

Normalized artifact JSON is also supported. Output includes the chosen backend,
original winner ranking, separate memory scores, exclusions and workload
settings. Scores are uncalibrated. Unsupported tasks or removal of all eligible
candidates cause explicit abstention. Run the selected simulator through the
`dqsim` API; this CLI performs selection only.

## Circuit attribution

The bundled model uses QASMBench and the PennyLane-hosted MQT Bench collection.
See the [source citations](../docs/references.md#circuit-sources) and
[model provenance](../models/README.md#source-provenance). For your own collections,
retain upstream circuit authorship, source versions and applicable notices
alongside the generated manifests. A timing label is this project's measurement;
it does not transfer authorship of the underlying circuit.
