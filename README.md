# Quantum Simulator Optimizer

A Python/Rust project for selecting a classical quantum-circuit simulator from
static circuit features. It includes four simulation backends and a small,
direct-classification ML pipeline.

| Backend | Representation | Useful circuit structure |
| --- | --- | --- |
| Statevector | Dense complex amplitudes | General circuits whose full state fits in memory |
| Matrix product state (MPS) | A chain of tensors | Circuits with limited entanglement across tensor cuts |
| P-block | Independent dense qubit blocks | Circuits that preserve small independent groups |
| Stabilizer | Packed Clifford tableau | Supported Clifford circuits, including large systems |

These are structural advantages, not universal performance rankings. Execution
time depends on circuit order, measurements, entanglement, resource limits and
hardware.

The intended workflow is **circuit → predict a simulator → simulate → display
results**. The current prediction CLI performs the selection step; the simulator
API executes circuits separately. It does not yet provide a combined application.

## Build and test

The documented collection workflow targets macOS and Linux (it uses POSIX file
locking). Use Python 3.10+, a Rust toolchain, and `uv`. From the repository root:

```sh
uv venv --python 3.10
uv pip install --python .venv/bin/python maturin 'numpy>=1.26' pytest \
  'qiskit>=2.5,<3' 'scikit-learn>=1.5,<1.8'
.venv/bin/maturin build --release --interpreter .venv/bin/python
uv pip install --python .venv/bin/python --no-deps target/wheels/dqsim-*.whl
RAYON_NUM_THREADS=4 .venv/bin/python -m pytest
PYO3_PYTHON=.venv/bin/python cargo test --lib --offline
```

The core library requires NumPy. The `qasm`, `ml`, and `test` dependency extras
are listed in `pyproject.toml`; `test` includes the dependencies for the public
Python suite. The `ml/` tools and bundled model are run from the repository
checkout; they are not installed by the native `dqsim` wheel.

The first Rust build needs access to download crate dependencies. The offline
Rust test command above works after those dependencies have been cached.

## Simulate a circuit

Save this as `bell.qasm`:

```qasm
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
```

```python
from dqsim import StatevectorSimulator, load_qasm

circuit = load_qasm("bell.qasm")
simulator = StatevectorSimulator(seed=42, max_memory_mb=1024)
counts = simulator.simulate_shots(circuit, shots=1000)
print(counts)  # Outcomes are "00" and "11".
```

`MpsSimulator`, `PBlockSimulator`, and `StabilizerSimulator` expose the same shot
simulation interface. Their result-query APIs and supported operations differ;
see the [simulator API reference](docs/simulators.md).

OpenQASM 2 import preserves supported measurements, resets and classical
conditions. Custom and larger gates are decomposed into the shared transport.
OpenQASM 3 and unsupported control flow raise errors. No measurements are added
automatically.

## Collect, train and predict

One public script imports local OpenQASM 2 circuits and collects repeated timings
for all four backends. Keep related sizes/variants in the same family directory.

```sh
.venv/bin/python scripts/collect_simulator_data.py \
  --circuits /path/to/circuits --task shots

.venv/bin/python -m ml.train_selector \
  --run data/simulator-runs/RUN_DIRECTORY \
  --output data/models/my-selector
```

The repository includes one trained selector: a boosted winner classifier with
a memory-feasibility filter. Predict using the bundled model:

```sh
.venv/bin/python -m ml.predict_selector --circuit bell.qasm
```

Use `--model data/models/my-selector/selector.json` for your own trained model.
The predictor selects a simulator; it does not estimate runtime or execute the
circuit. See the [workflow guide](ml/README.md) for collection settings, resume,
training and evaluation.

The selected pipeline achieved **90.8% exact winner accuracy** on 370 resolved
development comparisons, with one failed selection and six choices over 10×
slower than the winner. Its memory filter can incorrectly exclude viable
P-block choices on QFT circuits. These are development results, not a deployment
guarantee; see the [model card](models/README.md) for scope and limitations.

## Repository layout

- `src/`: Rust simulator implementations and Python bindings.
- `python/dqsim/`: Python interface and OpenQASM import.
- `ml/`: the selected model, static features, training, evaluation and prediction.
- `models/`: one trained selector and its model card.
- `scripts/collect_simulator_data.py`: circuit import and resumable data collection.
- `tests/`: public simulator, parser and ML tests.
- `benchmarks/`: reproducible simulator performance checks.
- `docs/`: public API documentation.

Alternative models, experiment runners, internal reports and local datasets are excluded
from version control. Public code and tests do not require that workspace.

## Data sources and acknowledgments

Training circuits were drawn from [QASMBench](https://github.com/pnnl/QASMBench)
and [MQT Bench](https://github.com/munich-quantum-toolkit/bench), with MQT circuits
obtained through [PennyLane's hosted dataset](https://pennylane.ai/datasets/single-dataset/mqt-bench).
We credit the upstream circuit authors and PennyLane's dataset distribution.
The benchmark timings and model evaluation are this project's measurements.
See [references and acknowledgments](docs/references.md) for paper citations,
software credits and [BibTeX entries](docs/references.bib).

## License

This project is available under the [MIT license](LICENSE).
