# References and acknowledgments

## Circuit sources

The model's training circuits came from two benchmark collections. MQT Bench
was accessed through PennyLane's hosted dataset; these are not three independent
circuit collections. Circuit authorship belongs to the upstream contributors.
The runtime measurements, selection labels, model fitting and evaluation reported
in this repository were produced by this project.

### QASMBench

Ang Li, Samuel Stein, Sriram Krishnamoorthy, and James Ang.
**QASMBench: A Low-Level Quantum Benchmark Suite for NISQ Evaluation and
Simulation.** *ACM Transactions on Quantum Computing* (2022, following the
upstream recommended citation).
[Paper and DOI](https://doi.org/10.1145/3550488) ·
[Repository and citation guidance](https://github.com/pnnl/QASMBench#citation-format).

Used as an OpenQASM circuit source for training-data collection. The imported
revision is recorded in the [model's source provenance](../models/README.md#source-provenance).
QASMBench also credits the original contributors of individual routines in its
benchmark tables and circuit files.

### MQT Bench

Nils Quetschlich, Lukas Burgholzer, and Robert Wille.
**MQT Bench: Benchmarking Software and Design Automation Tools for Quantum
Computing.** *Quantum* **7**, 1062 (2023).
[Paper and DOI](https://doi.org/10.22331/q-2023-07-20-1062) ·
[Repository and citation guidance](https://github.com/munich-quantum-toolkit/bench#cite-this).

Used as the original source of the benchmark families distributed through
PennyLane. Credit MQT Bench when describing the circuits, even when they were
obtained through PennyLane rather than generated with the MQT Python package.

### PennyLane dataset distribution

[PennyLane's MQT Bench dataset](https://pennylane.ai/datasets/single-dataset/mqt-bench)
provided the hosted snapshot used here. Its historical loading identifier is
`qml.data.load('other', name='mqt-bench')`. The snapshot identity and our conversion
steps are recorded in the [model card](../models/README.md#source-provenance).

For the PennyLane software used to inspect and convert the dataset:
Ville Bergholm et al. **PennyLane: Automatic differentiation of hybrid
quantum-classical computations.** arXiv:1811.04968 (2018).
[Paper](https://arxiv.org/abs/1811.04968) ·
[Upstream citation guidance](https://github.com/PennyLaneAI/pennylane#authors).
This credits dataset access and conversion; the selector uses scikit-learn.

## Software used by this project

| Resource | Role | Reference |
| --- | --- | --- |
| scikit-learn | Fitting the boosted winner classifier and random-forest memory filters; grouped evaluation | Fabian Pedregosa et al., **Scikit-learn: Machine Learning in Python**, *JMLR* **12**, 2825–2830 (2011). [Paper](https://jmlr.org/papers/v12/pedregosa11a.html) |
| Qiskit | OpenQASM parsing, gate definitions, circuit conversion, and independent simulator tests | [Qiskit SDK](https://github.com/Qiskit/qiskit) |
| Qiskit Aer | External baseline in the private comparison experiments | [Aer project](https://github.com/Qiskit/qiskit-aer), [AerSimulator documentation](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html) |
| NumPy | Numerical arrays, feature processing, and reference checks | [NumPy project](https://numpy.org/) |

The Python dependencies and Rust crates are declared in `pyproject.toml` and
`Cargo.toml`; resolved Rust versions are in `Cargo.lock`. Experiment records
capture their own software versions. Dependency use does not imply upstream
endorsement of this project's model or performance claims.

## Simulator background

These references describe established simulation methods used by the project:

- Scott Aaronson and Daniel Gottesman. **Improved Simulation of Stabilizer
  Circuits.** *Physical Review A* **70**, 052328 (2004).
  [Paper](https://arxiv.org/abs/quant-ph/0406196).
  Background for the stabilizer/destabilizer tableau representation.
- Guifré Vidal. **Efficient Classical Simulation of Slightly Entangled Quantum
  Computations.** *Physical Review Letters* **91**, 147902 (2003).
  [Paper](https://arxiv.org/abs/quant-ph/0301063).
  Background for low-entanglement tensor/MPS simulation.

## Reuse and citation files

[references.bib](references.bib) contains reusable bibliographic entries for the
papers and the hosted dataset above. It cites upstream resources, not a paper
claiming that this repository's experimental results have been peer reviewed.

Raw benchmark files are not included in the public repository. Upstream materials
retain their own licenses and notices; this project's MIT license does not
replace them. See QASMBench's [LICENSE](https://github.com/pnnl/QASMBench/blob/master/LICENSE)
and [NOTICE](https://github.com/pnnl/QASMBench/blob/master/NOTICE),
[MQT Bench's license](https://github.com/munich-quantum-toolkit/bench/blob/main/LICENSE),
and the PennyLane dataset page for the hosted resource. Preserve any applicable
per-circuit attribution when distributing circuit subsets.

Citation guidance checked against upstream sources on 2026-10-01. The PennyLane
snapshot used here is historical; no claim is made that it matches today's MQT
Bench release.
