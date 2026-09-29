"""Native quantum circuit simulators.

Circuit inputs expose ``model_dump_json()`` using the register/instruction schema.
"""

from ._core import (
    MpsSimulator,
    MpsResult,
    PBlockResult,
    PBlockSimulator,
    SimulationProfile,
    SimulationResult,
    StabilizerSimulator,
    StabilizerResult,
    StatevectorSimulator,
    simulate_distributed,
    simulate_distributed_shots,
    simulate_monolithic,
    simulate_monolithic_shots,
)

__all__ = [
    "MpsSimulator",
    "MpsResult",
    "PBlockResult",
    "PBlockSimulator",
    "SimulationProfile",
    "SimulationResult",
    "StabilizerSimulator",
    "StabilizerResult",
    "StatevectorSimulator",
    "simulate_distributed",
    "simulate_distributed_shots",
    "simulate_monolithic",
    "simulate_monolithic_shots",
]

# Parser dependencies are loaded only when an import helper is called.
from .qasm import ImportedCircuit, QASMImportError, from_qiskit, load_qasm

__all__ += ["ImportedCircuit", "QASMImportError", "from_qiskit", "load_qasm"]
