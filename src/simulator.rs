use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::codecs::{
    parse_distributed_mode, parse_monolithic_mode, parse_mps_options, parse_pblock_options,
    parse_stabilizer_options, parse_statevector_options, DistributedSimulationMode,
    MonolithicSimulationMode,
};
use crate::distributed::pblock::PBlockSimulator;
use crate::monolithic::mps::MpsSimulator;
use crate::monolithic::stabilizer::StabilizerSimulator;
use crate::monolithic::statevector::StatevectorSimulator;

#[pyfunction]
#[pyo3(signature = (circuit, mode="state_vector", seed=None, profile=false, **options))]
pub fn simulate_monolithic(
    py: Python,
    circuit: &Bound<PyAny>, // todo: tighten this
    mode: &str,
    seed: Option<u64>,
    profile: bool,
    options: Option<&Bound<PyDict>>,
) -> PyResult<PyObject> {
    match parse_monolithic_mode(mode)? {
        MonolithicSimulationMode::StateVector => {
            let options = parse_statevector_options(options)?;
            let sim = StatevectorSimulator::new(
                seed,
                profile,
                options.max_memory_mb,
                options.max_parallel_shots,
                options.sample_terminal,
            )?;
            Ok(sim.simulate(py, circuit)?.into_py(py))
        }
        MonolithicSimulationMode::Mps => {
            let options = parse_mps_options(options)?;
            let sim = MpsSimulator::new(
                seed,
                options.max_bond_dimension,
                options.truncation_threshold,
                options.max_discarded_weight,
                options.max_memory_mb,
                options.max_parallel_shots,
                options.sample_terminal,
                profile,
            )?;
            Ok(sim.simulate(py, circuit)?.into_py(py))
        }
        MonolithicSimulationMode::Stabilizer => {
            let opts = parse_stabilizer_options(options)?;
            let sim = StabilizerSimulator::new(
                seed,
                opts.max_memory_mb,
                opts.max_parallel_shots,
                opts.sample_terminal,
                opts.clifford_tolerance,
                profile,
            )?;
            Ok(sim.simulate(py, circuit)?.into_py(py))
        }
    }
}

#[pyfunction]
#[pyo3(signature = (distributed, mode="p_block", seed=None, profile=false, **options))]
pub fn simulate_distributed(
    py: Python,
    distributed: &Bound<PyAny>, // todo: tighten this
    mode: &str,
    seed: Option<u64>,
    profile: bool,
    options: Option<&Bound<PyDict>>,
) -> PyResult<PyObject> {
    match parse_distributed_mode(mode)? {
        DistributedSimulationMode::PBlock => {
            let opts = parse_pblock_options(options)?;
            let sim = PBlockSimulator::new(
                seed,
                opts.max_memory_mb,
                opts.max_block_qubits,
                opts.max_parallel_shots,
                opts.sample_terminal,
                opts.split_separable,
                opts.max_split_qubits,
                opts.separation_tolerance,
                profile,
            )?;
            Ok(sim.simulate(py, distributed)?.into_py(py))
        }
    }
}

#[pyfunction]
#[pyo3(signature = (circuit, mode="state_vector", shots=1000, seed=None, profile=false, **options))]
pub fn simulate_monolithic_shots(
    py: Python,
    circuit: &Bound<PyAny>, // todo: tighten this
    mode: &str,
    shots: usize,
    seed: Option<u64>,
    profile: bool,
    options: Option<&Bound<PyDict>>,
) -> PyResult<PyObject> {
    match parse_monolithic_mode(mode)? {
        MonolithicSimulationMode::StateVector => {
            let options = parse_statevector_options(options)?;
            let sim = StatevectorSimulator::new(
                seed,
                false,
                options.max_memory_mb,
                options.max_parallel_shots,
                options.sample_terminal,
            )?;
            sim.simulate_shots(py, circuit, shots, profile)
        }
        MonolithicSimulationMode::Mps => {
            let options = parse_mps_options(options)?;
            let sim = MpsSimulator::new(
                seed,
                options.max_bond_dimension,
                options.truncation_threshold,
                options.max_discarded_weight,
                options.max_memory_mb,
                options.max_parallel_shots,
                options.sample_terminal,
                profile,
            )?;
            Ok(sim.simulate_shots(py, circuit, shots, profile)?.into_py(py))
        }
        MonolithicSimulationMode::Stabilizer => {
            let opts = parse_stabilizer_options(options)?;
            let sim = StabilizerSimulator::new(
                seed,
                opts.max_memory_mb,
                opts.max_parallel_shots,
                opts.sample_terminal,
                opts.clifford_tolerance,
                profile,
            )?;
            Ok(sim.simulate_shots(py, circuit, shots, profile)?.into_py(py))
        }
    }
}

#[pyfunction]
#[pyo3(signature = (distributed, mode="p_block", shots=1000, seed=None, profile=false, **options))]
pub fn simulate_distributed_shots(
    py: Python,
    distributed: &Bound<PyAny>, // todo: tighten this
    mode: &str,
    shots: usize,
    seed: Option<u64>,
    profile: bool,
    options: Option<&Bound<PyDict>>,
) -> PyResult<PyObject> {
    match parse_distributed_mode(mode)? {
        DistributedSimulationMode::PBlock => {
            let opts = parse_pblock_options(options)?;
            let sim = PBlockSimulator::new(
                seed,
                opts.max_memory_mb,
                opts.max_block_qubits,
                opts.max_parallel_shots,
                opts.sample_terminal,
                opts.split_separable,
                opts.max_split_qubits,
                opts.separation_tolerance,
                profile,
            )?;
            Ok(sim
                .simulate_shots(py, distributed, shots, profile)?
                .into_py(py))
        }
    }
}
