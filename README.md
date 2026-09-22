# Quantum Simulator Optimizer

The project compares classical quantum-circuit simulators to support a future ML
backend selector. Its Rust extension exposes statevector, MPS, stabilizer, and
P-block simulation through Python. OpenQASM import and selector training are not
implemented in this checkout.

## Build and test

With Rust, Python 3.10+, and `uv` installed:

```sh
uv venv --python 3.10
uv pip install maturin 'numpy>=1.26' pytest qiskit
.venv/bin/maturin build --release --interpreter .venv/bin/python
uv pip install --no-deps target/wheels/dqsim-*.whl
PYO3_PYTHON=.venv/bin/python cargo test --lib --offline
RAYON_NUM_THREADS=4 .venv/bin/python -m unittest discover -s tests -v
```

The native tests use a small circuit transport and do not require the Bosonic
packages listed in the project dependencies. `--no-deps` installs just the built
extension for this workflow; applications using Bosonic circuit models or
converters should install those packages separately. Qiskit supplies independent
reference checks for every supported standard unitary gate and random circuits.

## Statevector simulation

Inputs expose `model_dump_json()` and contain `qregs`, `cregs`, and `instructions`
matching `src/types.rs`. For example:

```python
import json
from dqsim import StatevectorSimulator

class BellCircuit:
    def model_dump_json(self):
        return json.dumps({
            "qregs": {"q": {"name": "q", "base": 0, "size": 2}},
            "cregs": {"c": {"name": "c", "base": 0, "size": 2}},
            "instructions": [
                {"kind": "h", "qubit": 0},
                {"kind": "cx", "control": 0, "target": 1},
                {"kind": "measure", "qubit": 0, "cbit": 0},
                {"kind": "measure", "qubit": 1, "cbit": 1},
            ],
        })

sim = StatevectorSimulator(seed=42, max_memory_mb=1024)
counts = sim.simulate_shots(BellCircuit(), shots=1000)  # Only "00" and "11".
```

Both `StatevectorSimulator` and the `simulate_monolithic*` functions accept:

| Option | Default | Meaning |
| --- | --- | --- |
| `max_memory_mb` | `1024` | Positive working-memory budget in MiB. |
| `max_parallel_shots` | `None` | Maximum concurrent trajectories; also bounded by memory, shot count, and the Rayon thread pool. |
| `sample_terminal` | `True` | Evolve once and sample jointly when all measurements are terminal. Set `False` to force trajectories for comparisons. |
| `seed` | `None` | Optional seed for reproducible sampling within an execution strategy. |
| `profile` | `False` | Enable timing and execution metadata. |

Set `RAYON_NUM_THREADS` before starting Python to control CPU concurrency.
Measurements, reset, and classical feedback are supported. Dynamic circuits reuse
a deterministic prefix when the memory budget permits, allocate a bounded set
of worker buffers, and reuse them across shots. With one worker, gate updates can
use the thread pool. A terminal probability table that does not fit the budget
falls back to trajectories.

The budget includes live statevectors, terminal sampling tables, and classical
worker buffers. It is not a process RSS limit: compiled instructions, count
maps, profiling data, Python outputs, and previously returned results are outside
it. Statevector dimensions and allocation sizes are checked before allocation.

`simulate()` returns one trajectory: measurement collapses its returned state.
`simulate_shots()` returns the final classical register counts, including
unwritten bits as zero; a circuit with no classical bits returns the key `""`.
`result.counts()` samples the result's current quantum state, so calling it after
a measured `simulate()` does not reproduce the distribution of fresh circuit
executions. Terminal sampling preserves repeated measurements, correlations, and
the last write to each classical bit. Exact seeded counts can differ between
terminal sampling and trajectories while representing the same distribution.

Amplitudes use little-endian wire indices: qubit 0 is the least significant bit.
For `result.probabilities(qubits=...)` and `result.counts(qubits=...)`, the first
requested qubit is the most significant output bit. Duplicate or out-of-range
query qubits raise `ValueError`. The `result.statevector` property makes an
independent NumPy copy, requiring another full statevector allocation.

## Implemented statevector improvements

- Correct RCCX and RC3X relative phases; reject unknown gates and classical
  operations rather than silently skipping them. `remote_cu1` requires one phase
  parameter and now applies it.
- Validate registers, gate operands, parameters, and nested conditionals before
  running either API, including branches that are not taken.
- Compile and fuse gates once per invocation using shared execution semantics.
  Preserve small rotations, specialize permutations and diagonal gates, and
  precompute dense-gate layouts. Execution releases the Python GIL.
- Extract full probabilities in O(2^n), and k-qubit marginals in O(k * 2^n),
  replacing the previous O(2^(n+k)) scan. Sampling reuses the probability buffer
  as its cumulative distribution.
- Reuse terminal states and deterministic prefixes; bound parallel trajectories
  by working memory. Gates, measurement, and result queries have reference tests.

The shared gate matrices also serve MPS and P-block; compatibility tests cover
both. The MPS implementation is described below.

## Profiling and performance checks

`simulate_shots(..., profile=True)` writes JSON to `dqsim_profiles/` with total,
preprocessing, compilation (`fusion_time`), and execution times, plus
`execution_strategy`, `parallel_shots`, `working_bytes`, and
`deterministic_prefix_ops`. Terminal sampling leaves `shot_times` empty because
all samples share the evolution cost. Trajectory timings exclude shared prefix
execution and initial worker allocation; use total wall time for comparisons.
Single-trajectory `result.profile.total_time` covers preprocessing through
execution. Its kernel counters describe executed compiled operations, with
reset grouped into measurement time.

Use release builds, fixed thread counts, warm-up runs, and repeated measurements.
Keep profiling disabled for timing comparisons. The focused benchmark does not
collect ML training data:

```sh
RAYON_NUM_THREADS=4 .venv/bin/python benchmarks/statevector_benchmark.py
```

It measures terminal sampling, dynamic trajectories, unitary evolution, and full
probability extraction. `--extension PATH` loads a separately built native module
in a fresh process for before/after checks; `--repeats N` changes the sample count.

The recorded [before/after measurements](benchmarks/statevector_results.json)
compare commit `3c37e16` with these changes on macOS arm64, Python 3.10.9, four
Rayon threads, and five timed repetitions after warm-up:

| Case | Before median | After median | Speedup |
| --- | ---: | ---: | ---: |
| 12 qubits, 256 terminal shots | 159.95 ms | 3.03 ms | 52.7× |
| 10 qubits, 128 dynamic shots | 13.62 ms | 2.56 ms | 5.3× |
| 16-qubit unitary evolution | 29.32 ms | 8.38 ms | 3.5× |
| 14-qubit full probabilities | 86.57 ms | 0.409 ms | 211.6× |

These are focused implementation checks, not general performance guarantees or
simulator-selection training labels. The statevector regressions include Qiskit comparisons and shared-kernel
compatibility checks. Clippy's
remaining five warnings are in the existing stabilizer implementation.


## MPS simulation

`MpsSimulator` now keeps a mixed-canonical matrix product state through evolution,
measurement, reset, and classical feedback. QR moves the orthogonality center;
a bounded complex Jacobi SVD handles two-site updates. Reconstruction regressions
cover failures of the previous complex SVD path. Truncated states are normalized.
Single-qubit gates are fused and applied in place. Routing retains useful tensor
orders, chooses a direction using bond-size estimates, and implements logical
SWAPs by relabeling wires.

```python
from dqsim import MpsSimulator

sim = MpsSimulator(seed=42, max_memory_mb=256, profile=True)
result = sim.simulate(circuit)  # Returns a compact MpsResult.
counts = result.counts(shots=1000, seed=42)
print(result.bond_dimensions, result.diagnostics)
# Explicit dense export for small circuits only:
amplitudes = result.statevector
```

`simulate_monolithic(..., mode="mps")` also returns `MpsResult`, replacing the
previous dense `SimulationResult`. It preserves `.num_qubits`, `.classical_bits`,
`.statevector`, `.probabilities(qubits=None)`, `.counts(shots=1000, qubits=None,
seed=None)`, and `.fidelity(numpy_statevector)`. Dense export returns an independent
NumPy array and raises `MemoryError` if its dimension or working allocation is too
large. No dense array is created by ordinary simulation, measurement, or sampling.

Additional compact queries:

- `result.amplitude("0101")`: one amplitude, with the highest logical qubit first.
  Strings can exceed 64 bits.
- `result.expectation_value("ZIIX")`: expectation of a Pauli product, highest
  logical qubit first. Supply exactly one character per qubit.
- `result.probabilities([q0, q1])`: marginal probabilities through tensor
  contractions, with `q0` as the most significant output bit. Full distributions
  are still exponential output and are checked against memory limits.
- `result.bond_dimensions` and `result.qubit_order`: current bonds and the logical
  qubit at each position in the tensor chain. Query outputs always use logical
  qubit indices, regardless of routing.
- `result.diagnostics`: bond sizes, discarded weight, norm squared, routing and
  decomposition counters, and peak memory estimates. `result.profile` is a
  dictionary including `total_time` when profiling is enabled, otherwise `None`.

Options are accepted by the constructor and the `simulate_monolithic*` wrappers:

| Option | Default | Meaning |
| --- | --- | --- |
| `max_bond_dimension` | `None` | Optional positive cap on each bond. |
| `truncation_threshold` | `1e-12` | Legacy absolute singular-value cutoff. |
| `max_discarded_weight` | `0.0` | Optional per-update relative squared Schmidt-weight budget, in `[0, 1)`. Zero disables this additional approximation. |
| `max_memory_mb` | `1024` | Positive working-memory allowance in MiB. |
| `max_parallel_shots` | `1` | Maximum concurrent dynamic trajectories; bounded by the Rayon pool and available memory. |
| `sample_terminal` | `True` | Evolve once and sample the tensor chain for terminal measurements. |
| `seed` | `None` | Reproducible sampling seed. |
| `profile` | `False` | Enable timing; shot profiling is selected on `simulate_shots(..., profile=True)`. |

For no intentional truncation, use `max_bond_dimension=None`,
`truncation_threshold=0.0`, and `max_discarded_weight=0.0`. Numerical rank removal
still discards singular directions below a dimension-scaled machine-precision
cutoff. The three approximation controls are independent: an absolute cutoff or
bond cap can require more loss than the weight budget. `discarded_weight` sums
relative losses over executed updates; it is a diagnostic, **not final-state
infidelity**. `bond_cap_truncations` records updates forced below the otherwise
selected rank. Compare simulators using a consistent accuracy target and output
task before collecting ML labels.

Terminal sampling preserves correlations, repeated measurements, classical
register order, and the last write to each classical bit. Dynamic circuits reuse
a deterministic prefix when it fits, with one reusable MPS per worker. Shot RNGs
are indexed by shot number, so changing the worker count preserves seeded counts.
Terminal sampling and forced trajectories can produce different seeded counts.
Rust evolution and result queries release the Python GIL.

The memory allowance covers tensors, wire maps, conservative QR/SVD workspace,
query scratch, and classical shot workspace. Shared prefix storage is reserved
before dividing memory between workers. `peak_working_bytes` is the largest
single-state working estimate, not aggregate process RSS. Python/runtime overhead,
compiled instructions, count dictionaries, profile data, and previously returned
objects are outside the allowance. Dense queries include their output buffer (and
an estimated dictionary allowance for probabilities). An unpredictable increase
in a worker's bond sizes can exhaust its share; reduce `max_parallel_shots` or
increase `max_memory_mb` in that case.

Unsupported gates and classical operations now raise errors, including operations
inside untaken conditionals. Decompose gates with more than two operands before
using MPS. `remote_cu1` and `remote_rzz` apply their required phase parameter.

The MPS tests cover every supported standard one-/two-qubit gate against Qiskit,
random circuits and routing, the previous numerical failures, truncation, compact
queries, validation, profiles, and 100-qubit GHZ sampling and feedback under small
memory budgets. Run focused performance checks with:

```sh
RAYON_NUM_THREADS=4 .venv/bin/python benchmarks/mps_benchmark.py
```

This measures terminal sampling versus forced trajectories, a 100-qubit GHZ
circuit, dynamic execution, and unitary evolution. It uses a release build, warm-up,
and repeated timings; it does not collect selector training data.

The complete suite passes 19 Rust tests and 40 Python tests.
Recorded [MPS measurements](benchmarks/mps_results.json) use macOS arm64,
Python 3.10.9, four Rayon threads, and five repetitions. These compare execution
strategies in the upgraded implementation, not against the old simulator:

| Case | Median |
| --- | ---: |
| 20-qubit GHZ, 1,000 terminal samples | 3.35 ms |
| Same circuit, 1,000 forced trajectories | 24.89 ms |
| 100-qubit GHZ, 1,000 terminal samples | 13.56 ms |
| 20-qubit dynamic circuit, 100 shots | 2.59 ms |
| 10-qubit unitary circuit, four layers | 0.95 ms |

These focused checks demonstrate the new execution paths; they are not general
performance guarantees or simulator-selection labels.
