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
compatibility checks.


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

## P-block simulation

`PBlockSimulator` stores independent groups of qubits as separate dense blocks.
It starts with one block per qubit, merges blocks when an operation needs them,
and splits off measured or reset qubits. Node ownership no longer forces qubits
into a single dense block. Memory therefore depends on the largest interacting
groups, rather than necessarily on the total number of qubits. Circuits that
entangle all their qubits can still require a full exponential statevector.

Both ordinary circuit objects exposing `model_dump_json()` and distributed
circuit objects are accepted:

```python
from dqsim import PBlockSimulator

sim = PBlockSimulator(seed=42, max_memory_mb=256, profile=True)
result = sim.simulate(circuit)
print(result.block_qubits, result.diagnostics)
counts = result.counts(shots=1000, seed=42)
# Explicit dense export for small circuits only:
amplitudes = result.statevector
```

Evolution, measurements, reset, and nested classical feedback follow one global
instruction order. Shared classical state is preserved across nodes. Unsupported
operations, invalid operands, and malformed transport metadata raise errors,
including operations in untaken conditionals. Standard one- through five-qubit
gates and parameterized `remote_cu1`/`remote_rzz` use the shared statevector gate
compiler. Compilation fuses single-qubit gates without dropping small rotations.
Specialized controlled and diagonal kernels avoid dense gate matrices; known
control values avoid unnecessary merges. Logical SWAPs relabel wires.

The constructor and `simulate_distributed*` wrappers accept:

| Option | Default | Meaning |
| --- | --- | --- |
| `max_memory_mb` | `1024` | Positive working-memory allowance in MiB. |
| `max_block_qubits` | `None` | Optional positive cap on any dense block; exceeding it raises `MemoryError`. |
| `max_parallel_shots` | `1` | Maximum concurrent trajectories, bounded by memory, shots, and the Rayon pool. |
| `sample_terminal` | `True` | Evolve once and sample independent blocks for terminal measurements. |
| `split_separable` | `False` | Check touched wires for single-qubit factors after gates. |
| `max_split_qubits` | `12` | Largest block considered by optional separability checks. |
| `separation_tolerance` | `0.0` | Maximum relative squared reconstruction residual for an optional split, in `[0, 1)`. |
| `seed` | `None` | Reproducible sampling seed within an execution strategy. |
| `profile` | `False` | Enable result timing; select shot profiling with `simulate_shots(..., profile=True)`. |

Defaults introduce no intentional approximation. Measurement splitting is always
enabled. Optional separability checks can recover single-qubit factors after
disentangling gates, but cost extra work and do not search all possible block
partitions. Zero tolerance requires an exactly zero floating-point reconstruction
residual, so some separable states may remain grouped. A positive tolerance
explicitly permits approximation; accepted factors are normalized.
`separation_residual` sums accepted relative residuals and is not final-state
infidelity. The block-size cap never silently truncates the state.

`simulate()` returns a compact `PBlockResult`, preserving `.num_qubits`,
`.physical_qubits`, `.classical_bits`, `.statevector`, and
`.probabilities(qubits=None)`. Additional queries include:

- `.counts(shots=1000, qubits=None, seed=None)`: sample blocks directly, preserving
  correlations within each block.
- `.amplitude("0101")`: one amplitude without constructing the full statevector.
- `.expectation_value("ZIIX")`: expectation of a Pauli product.
- `.block_qubits`: physical wires in each current block.
- `.diagnostics`: block sizes, norm, merge/split counters, separation residuals,
  and peak memory estimates. `.profile` adds `total_time` when enabled, otherwise
  it is `None`.

Physical labels may be sparse, and total qubit counts may exceed 64. Amplitude
and Pauli strings list the highest physical label first, with one character per
present wire. Dense array bit `i` corresponds to `physical_qubits[i]`, sorted in
ascending order. Probability/count queries take physical labels, with the first
requested wire as the most significant output bit; the default is descending
physical order. Unknown or duplicate query wires raise `ValueError`.
Dense export returns an independent NumPy array. Both dense export and full
probability dictionaries require exponential output and are memory checked.

As with the other backends, `simulate()` returns one collapsed trajectory;
`result.counts()` samples that current state. Use `simulate_shots()` for fresh
circuit executions. Terminal sampling preserves repeated measurements and the
last write to each classical bit. Dynamic shots reuse a deterministic prefix
and worker buffers when memory permits. Trajectory seeds are indexed by shot,
so worker count does not change seeded counts. Terminal sampling and trajectories
can produce different seeded counts. Rust execution releases the Python GIL.

The memory allowance covers quantum amplitudes, conservative block metadata,
merge/split and query scratch, and classical worker workspace. Shared prefix
storage is reserved before dividing memory between workers. If terminal sampling
tables or prefix copies do not fit, execution falls back to trajectories without
them. Compiled input, count maps, profiling data, Python/runtime overhead, and
previous results are outside the allowance. It is not a process RSS limit.
Unexpected block growth may exhaust a worker's share; reduce
`max_parallel_shots` or increase the budget in that case.

### Distributed ordering

Distributed inputs expose `circuits` and `qubits_per_node`, keyed by identical
node IDs, with unique physical-wire ownership. The preferred ordering metadata
uses stable string event IDs:

```python
distributed.operation_ids = {0: ["prepare", "entangle"], 1: ["entangle", "read"]}
distributed.operation_order = {"prepare": 0, "entangle": 1, "read": 2}
```

Each ID list matches that node's serialized instructions. Copies of a shared
event use the same ID and identical instruction contents, and execute once.
Separate occurrences require separate IDs. Global order must be unambiguous and
agree with every node's instruction order. Shared classical register ranges may
be identical across nodes; partial overlaps are rejected.

Legacy `_instruction_index` mappings from Python object IDs to global positions
remain supported with a matching Python `instructions` list on every node.
Missing order entries and repeated use of the same instruction object within
one node now raise errors; use explicit IDs for those repeated occurrences.

### P-block validation and measurements

Shot profiling writes `pblock_*` JSON under `dqsim_profiles/`, recording total,
preprocessing, compilation, and execution times, the chosen strategy and worker
count, prefix length, and block counters. `peak_working_bytes` is the largest
single-pool working estimate, rather than aggregate process RSS.

P-block coverage
includes Qiskit gate and random-circuit comparisons, feedback, ordering and
deduplication, compact queries, optional splitting, memory fallbacks, and a
100-qubit state stored as fifty Bell-pair blocks under a 1 MiB allowance.

```sh
RAYON_NUM_THREADS=4 .venv/bin/python benchmarks/pblock_benchmark.py
```

Recorded [P-block measurements](benchmarks/pblock_results.json) use a release
build on macOS arm64, Python 3.10.9, four Rayon threads, and five repetitions
after warm-up:

| Case | Median |
| --- | ---: |
| 16-qubit product state, one node, 16 shots | 0.094 ms |
| Same product state, sixteen nodes, 16 shots | 0.158 ms |
| 20-qubit Bell pairs, 1,000 terminal samples | 2.00 ms |
| Same circuit, 1,000 forced trajectories | 5.73 ms |
| 100-qubit Bell pairs, 1,000 terminal samples | 8.74 ms |
| 20-qubit dynamic circuit, 100 shots | 0.677 ms |

Both node layouts keep the product state in singleton blocks; additional node
transport has parsing overhead. These timings compare paths in this
implementation and are not before/after speedups or selector training labels.


## Stabilizer simulation

`StabilizerSimulator` now uses a contiguous tableau with packed `u64` X/Z rows
and separate signs. Pauli multiplication tracks the full phase modulo four,
fixing a sign error that previously produced impossible measurement outcomes.
Gate compilation, register validation, and Clifford eligibility checks run before
execution, including for zero shots and untaken conditional branches. Rust
compilation, evolution, and numerical result queries release the Python GIL.

```python
from dqsim import StabilizerSimulator

sim = StabilizerSimulator(seed=42, max_memory_mb=256, profile=True)
if sim.supports(circuit):
    result = sim.simulate(circuit)
    print(result.stabilizers, result.diagnostics)
    counts = result.counts(shots=1000, seed=42)
    # Fresh executions, including measurements, reset and classical feedback:
    circuit_counts = sim.simulate_shots(circuit, shots=1000)
```

`supports()` tests whether every instruction can be compiled to a supported
Clifford operation. It returns `False` for non-Clifford gates or unsupported
parameter choices; malformed circuits and unknown transport operations raise
`ValueError`. It does not allocate a tableau or check simulation memory needs.
Use it as an eligibility filter before benchmarking for selector training.
It is a conservative, instruction-level check: it does not discover cancellations
between non-Clifford gates, such as two consecutive T gates.

Supported fixed gates are identity/U0, X, Y, Z, H, S, Sdg, SX, SXdg, CX, CY, CZ,
and SWAP. Measurement, reset, barriers, and nested register-equality feedback are
supported. Parameterized RX/RY/RZ, P/U1, U/U2/U3, RXX/RZZ, and controlled rotation
and U gates are accepted when their parameters pass the angle policy below and
the resulting gate preserves the Pauli group. For example, `RX(pi/2)` and
`CRX(pi)` are accepted; `CRX(pi/2)` and `CP(pi/2)` are rejected. Remote CX/CZ,
Bell-link gates, and Clifford-valued remote CU1/RZZ are supported as well.
T, Toffoli, and other non-Clifford operations require another backend.

The constructor and `simulate_monolithic*` wrappers accept:

| Option | Default | Meaning |
| --- | --- | --- |
| `max_memory_mb` | `1024` | Positive working-memory allowance in MiB. |
| `max_parallel_shots` | `1` | Maximum concurrent trajectories, also bounded by memory, shots, and Rayon threads. |
| `sample_terminal` | `True` | Compile a correlated terminal sampler instead of replaying every shot. |
| `clifford_tolerance` | `0.0` | Absolute angle tolerance in radians for snapping parameters to multiples of pi/2; must be in `[0, pi/4)`. |
| `seed` | `None` | Reproducible sampling seed within an execution strategy. |
| `profile` | `False` | Enable result timing; shot profiling is selected with `simulate_shots(..., profile=True)`. |

At zero tolerance each parameter must equal the floating-point expression
`k * (pi/2)`, with `abs(k) <= 2**40`. A tiny nonzero rotation is rejected rather
than discarded. A positive tolerance explicitly permits approximating angles by
nearby Clifford values; leave it at zero for exact-backend comparisons. After
snapping, controlled and U-family gates undergo a separate Pauli-conjugation
check, so quarter-turn parameters alone do not guarantee eligibility. All
recognized gates then evolve with binary arithmetic. Global phase is not tracked.

Single-qubit Clifford operations are compiled to signed Pauli permutations and
fused exactly. X, Y, Z and Sdg each need one tableau pass. Row multiplication uses
word-level XOR/popcount operations, and deterministic measurements reuse a
scratch row. Gate updates take O(n) time; measurements can take O(n^2 / 64 + n)
with the packed representation. Memory remains quadratic in the qubit count.

Terminal computational-basis outcomes form a uniform affine binary space. The
sampler finds its independent directions from the stabilizers' X components and
one valid outcome by projection, then draws correlated samples from that space.
This preserves repeated measurements, classical-bit order, unwritten zero bits,
and the last write to each classical bit. It never constructs a statevector or
an exponential probability table for shot sampling. Dynamic circuits reuse a
deterministic prefix and one tableau per worker when memory permits. Seeds are
indexed by shot number, so changing the trajectory worker count preserves counts;
terminal sampling can produce different seeded counts. Reference-frame sampling
for dynamic circuits and inverse-tableau execution are not implemented.

`simulate()` and `simulate_monolithic(..., mode="stabilizer")` return a compact
`StabilizerResult` containing one trajectory. Its queries include:

- `.num_qubits` and `.classical_bits`: the state size and final written bits.
- `.stabilizers`: signed Pauli generator strings, such as `+XX` and `+ZZ`.
- `.expectation_value("YY")`: Pauli expectation, exactly -1, 0, or 1.
- `.counts(shots=1000, qubits=None, seed=None)`: sample the current quantum state.
- `.probability("0101")`: probability of one computational-basis outcome.
- `.probabilities(qubits=None)`: dictionary of integer outcomes and nonzero
  probabilities. Only the support is enumerated; a large GHZ state has two entries,
  while a product of n plus states needs 2**n entries and can exceed the budget.
- `.diagnostics`: tableau allocation estimate and gate/measurement counters.
  `.profile` adds `total_time` when enabled, otherwise it is `None`.

Strings list the highest logical qubit first. In an explicit query list, the
first qubit is the most significant output bit. Unknown and duplicate query
qubits raise `ValueError`. Bitstrings and Python integer outcome keys support more
than 64 qubits. Result queries preserve the state; sampling/probability queries
reserve a temporary tableau copy. This API does not expose a dense statevector.
Sampling a measured result samples its collapsed state; use `simulate_shots()`
for fresh circuit executions.

The working allowance covers tableaus, retained prefixes, sampler workspace,
classical worker buffers, and query scratch/output estimates. Sampler construction
falls back to trajectories when it does not fit, and prefix reuse is disabled
when an additional tableau would exceed the allowance. Workers reuse their
buffers, with concurrency bounded before allocation. Compiled instructions,
input parsing, count maps, profiling data, Python/runtime overhead, and previously
returned objects are outside the allowance; it is not a process RSS limit.

Shot profiling writes `stabilizer_*` JSON under `dqsim_profiles/`. It records
preprocessing, compilation, execution, and total time, `execution_strategy`,
`parallel_shots`, `deterministic_prefix_ops`, `working_bytes`, and measurement
counters. `terminal_rank` is the number of independent random bits in the sampled
measurement space before classical overwrites. Counters describe work actually
executed; terminal sampling performs measurements once, while trajectories
perform them per shot. Profiles aggregate counters without retaining individual
shot outputs or timings. The strategy is `terminal_affine`, `prefix_trajectories`,
`trajectories`, or `empty`.

The complete suite passes 21 Rust tests and 79 Python tests. Stabilizer tests
cover the original phase regression, the full Pauli multiplication table,
symplectic invariants across word boundaries, Qiskit gate and random-circuit
comparisons, exact enumeration of dynamic measurement branches, large compact
queries, validation, seeded execution, and memory-limited fallback paths.

Run the focused release benchmark with:

```sh
RAYON_NUM_THREADS=4 .venv/bin/python benchmarks/stabilizer_benchmark.py
```

`--extension PATH` loads a separately built native module in a fresh process for
before/after comparisons. Profiling is disabled; each case has a warm-up and five
timed repetitions by default. These focused timings are not selector training
labels or general performance guarantees.

Recorded [before/after measurements](benchmarks/stabilizer_results.json) compare
the previous stabilizer implementation with these upgrades on macOS arm64,
Python 3.10.9, and four Rayon threads. Each uses its default execution strategy:
previously independent trajectories, now terminal affine sampling.

| Case | Before median | After median | Speedup |
| --- | ---: | ---: | ---: |
| 100-qubit GHZ, 1,000 shots | 148.18 ms | 0.624 ms | 237.3× |
| 500-qubit GHZ, 100 shots | 1,572.11 ms | 8.629 ms | 182.2× |
| 100-qubit GHZ, one shot | 0.733 ms | 0.303 ms | 2.4× |

These circuits have only two possible outcomes and particularly benefit from
shared evolution and correlated sampling. The results do not predict performance
on arbitrary Clifford circuits or dynamic workloads.
