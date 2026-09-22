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

The shared gate matrices and numerical kernels also serve MPS and P-block;
small compatibility tests cover both. Their simulator-specific algorithms have
not been redesigned here.

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
simulator-selection training labels. The 18 Rust tests and 25 Python tests pass,
including Qiskit comparisons and shared-kernel compatibility checks. Clippy's
remaining five warnings are in the existing stabilizer implementation.
