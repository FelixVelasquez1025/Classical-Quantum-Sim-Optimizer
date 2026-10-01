# Simulator selector

`selector.json` is the single bundled model. It predicts which simulator to use;
it does not predict runtime. Its learned parameters are stored as plain JSON,
without executable pickle objects or dependencies on private experiment files.

## Architecture

A gradient-boosted classifier ranks statevector, MPS, P-block and stabilizer using
28 circuit features plus task/shot context. A separate memory-failure classifier
uses those inputs and 29 structural features to filter unsuitable candidates.
The original ranking then chooses among the remaining backends.

The winner classifier has 80 trees, learning rate 0.05, depth 2 and minimum leaf
size 2. The memory models have 64 trees, depth 4, minimum leaf size 3 and balanced
class weights. P-block filtering uses a score threshold of 0.90; MPS filtering
is disabled. Deterministic stabilizer support and statevector allocation checks
also apply. Scores are uncalibrated.

## Training scope and measured performance

Training used [QASMBench](../docs/references.md#qasmbench) and
[MQT Bench circuits distributed by PennyLane](../docs/references.md#pennylane-dataset-distribution),
with related circuit families kept together in validation. The data contains 395 observed circuits,
370 with resolved winner labels. Measurements used macOS arm64, four native
threads, 1,000 shots, five timed repetitions, and a 1 GiB backend working-memory
setting. That setting is not a total process-memory cap.

Development evaluation of this selection pipeline used five family-held-out
folds, with tuning confined to training groups:

| Metric | Result |
| --- | ---: |
| Exact winner choices on resolved circuits | 336 / 370 (90.8%) |
| Failed selections on resolved circuits | 1 |
| Successful choices at least 10× slower than the winner | 6 |
| Observed failed selections across all 395 circuits | 24 |
| Selections with a missing observation | 1 |
| Coverage | 100% |

These are development-validation results, not an independent final test or
in-sample scores for the bundled all-data fit. The final model was fitted after
validation. The fixed-parameter public trainer evaluates its own newly trained
pipeline; it does not replay the earlier model-selection study or promise the
same score on new records.

## Limitations

The memory filter falsely excluded P-block on 10 of 264 successful P-block runs
(3.8%), all QFT circuits. P-block was fastest on those circuits. The winner
classifier already preferred MPS on them, so those exclusions did not introduce
new selection errors in this evaluation, but they remain an important weakness.
The filter exceeded the 1% false-exclusion limit used during development; the
model should be treated as a research prototype, not a validated safety filter.

The remaining resolved failure selects MPS on a 24-qubit random circuit. The
worst successful selection was about 526× slower than the fastest backend.
Different hardware, shot counts or resource settings require fresh validation.
Some measurement labels incorporate provisional repeated-run medians in training;
held-out evaluation retains the original observations. Repeated use of these
families during development limits the strength of generalization claims.

Raw datasets, machine-specific training paths, diagnostic reports, alternative
models and tuning history are not included in this model artifact.

## Source provenance

- **QASMBench:** [upstream revision
  `357b942396d5c2b7cbc1c229c585a6ef5ccaebac`](https://github.com/pnnl/QASMBench/tree/357b942396d5c2b7cbc1c229c585a6ef5ccaebac).
- **MQT Bench via PennyLane:** hosted `mqt-bench` snapshot, HTTP Last-Modified
  `Mon, 29 Apr 2024 21:55:36 GMT`, ETag
  `"50b0281166cd7005c2d69f2a41158ebb-1009"`, as saved in the import manifests.
  This is a hosted-snapshot identifier, not an MQT Git commit or content hash.
  The upstream Git revision of that snapshot was not recorded.
- **Conversion:** QASMBench inputs were normalized to the shared one-/two-qubit
  format. The PennyLane subset exporter decomposed operations, removed global
  phases where needed (preserving measurement probabilities), and appended
  terminal measurements on all wires. These are project-prepared sampling
  variants of the source circuits. The public QASM importer itself does not add
  measurements automatically.

[Full citations and dataset/software credits](../docs/references.md) identify
both the original benchmark authors and the PennyLane distribution. Model
accuracy and runtime labels are project measurements, not results reported by
those upstream publications.
