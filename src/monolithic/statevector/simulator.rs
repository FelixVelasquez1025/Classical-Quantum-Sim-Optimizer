use std::collections::HashMap;
use std::time::Instant;

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

use super::execution::{compile, execute, ProfileAcc};
use super::shots::{check_budget, run_shots, state_layout, zero_state};
use super::validation::validate_circuit;
use crate::engine::{marginal_probs, sample_counts};
use crate::profiling::write_shots_profile;
use crate::types::Circuit;

type C = Complex64;

// ---------------------------------------------------------------------------
// ShotsProfile (simulate_shots instrumentation, dumped to JSON on disk)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct ShotsProfile {
    preprocessing_time: f64,
    fusion_time: f64,
    shots_total_time: f64,
    total_time: f64,
    num_shots: usize,
    shot_times: Vec<f64>,
    execution_strategy: String,
    parallel_shots: usize,
    working_bytes: usize,
    deterministic_prefix_ops: usize,
}

// ---------------------------------------------------------------------------
// SimulationProfile
// ---------------------------------------------------------------------------

#[pyclass]
pub struct SimulationProfile {
    #[pyo3(get)]
    pub apply_one_qubit_calls: u64,
    #[pyo3(get)]
    pub apply_one_qubit_time: f64,
    #[pyo3(get)]
    pub apply_n_qubit_calls: u64,
    #[pyo3(get)]
    pub apply_n_qubit_time: f64,
    #[pyo3(get)]
    pub measure_qubit_calls: u64,
    #[pyo3(get)]
    pub measure_qubit_time: f64,
    #[pyo3(get)]
    pub total_time: f64,
}

#[pymethods]
impl SimulationProfile {
    fn __repr__(&self) -> String {
        let total = self.total_time.max(1e-9);
        format!(
            "SimulationProfile (total: {:.2} ms)\n  apply_one_qubit : {:4} calls  {:8.2} ms  ({:.1}%)\n  apply_n_qubit   : {:4} calls  {:8.2} ms  ({:.1}%)\n  measure_qubit   : {:4} calls  {:8.2} ms  ({:.1}%)",
            self.total_time * 1000.0,
            self.apply_one_qubit_calls,
            self.apply_one_qubit_time * 1000.0,
            100.0 * self.apply_one_qubit_time / total,
            self.apply_n_qubit_calls,
            self.apply_n_qubit_time * 1000.0,
            100.0 * self.apply_n_qubit_time / total,
            self.measure_qubit_calls,
            self.measure_qubit_time * 1000.0,
            100.0 * self.measure_qubit_time / total,
        )
    }
}

// ---------------------------------------------------------------------------
// SimulationResult
// ---------------------------------------------------------------------------

#[pyclass]
pub struct SimulationResult {
    sv: Vec<C>,
    #[pyo3(get)]
    pub num_qubits: usize,
    cbits: HashMap<usize, i32>,
    prof: Option<Py<SimulationProfile>>,
}

impl SimulationResult {
    pub(crate) fn new(
        sv: Vec<C>,
        num_qubits: usize,
        cbits: HashMap<usize, i32>,
        prof: Option<Py<SimulationProfile>>,
    ) -> Self {
        Self {
            sv,
            num_qubits,
            cbits,
            prof,
        }
    }
}

#[pymethods]
impl SimulationResult {
    /// Copy the amplitudes to an independent NumPy array of shape (2^n,).
    #[getter]
    fn statevector<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<C>> {
        self.sv.clone().into_pyarray_bound(py)
    }

    /// Final classical register state. Keys are absolute cbit indices.
    #[getter]
    fn classical_bits(&self, py: Python) -> PyResult<PyObject> {
        let d = PyDict::new_bound(py);
        for (&k, &v) in &self.cbits {
            d.set_item(k, v)?;
        }
        Ok(d.into())
    }

    /// Profiling data, or None if the simulator was not run with profile=True.
    #[getter]
    fn profile(&self, py: Python) -> PyObject {
        match &self.prof {
            None => py.None(),
            Some(p) => p.clone_ref(py).into_py(py),
        }
    }

    /// Full or marginal probability distribution. Keys are integer basis states.
    /// qubits[0] is MSB of the output index. If None, all qubits in descending order.
    #[pyo3(signature = (qubits=None))]
    fn probabilities(&self, py: Python, qubits: Option<Vec<usize>>) -> PyResult<PyObject> {
        let qs: Vec<usize> = qubits.unwrap_or_else(|| (0..self.num_qubits).rev().collect());
        validate_query(&qs, self.num_qubits)?;
        let probs = py.allow_threads(|| marginal_probs(&self.sv, self.num_qubits, &qs));
        let d = PyDict::new_bound(py);
        for (j, p) in probs.iter().enumerate() {
            if *p > 0.0 {
                d.set_item(j, p)?;
            }
        }
        Ok(d.into())
    }

    /// Sample the distribution. Bitstrings have qubits[0] as the leftmost (MSB) character.
    #[pyo3(signature = (shots=1000, qubits=None, seed=None))]
    fn counts(
        &self,
        py: Python,
        shots: usize,
        qubits: Option<Vec<usize>>,
        seed: Option<u64>,
    ) -> PyResult<PyObject> {
        let qs: Vec<usize> = qubits.unwrap_or_else(|| (0..self.num_qubits).rev().collect());
        validate_query(&qs, self.num_qubits)?;
        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };
        let c = py
            .allow_threads(|| sample_counts(&self.sv, self.num_qubits, shots, &mut rng, Some(&qs)));
        let d = PyDict::new_bound(py);
        for (k, v) in c {
            d.set_item(k, v)?;
        }
        Ok(d.into())
    }

    /// Compute |<self|other>|^2.
    fn fidelity(&self, other: PyReadonlyArray1<C>) -> f64 {
        let arr = other.as_array();
        if arr.len() != self.sv.len() {
            return 0.0;
        }
        let dot: C = self
            .sv
            .iter()
            .zip(arr.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        dot.norm_sqr()
    }
}

fn validate_query(qubits: &[usize], n: usize) -> PyResult<()> {
    for (i, &q) in qubits.iter().enumerate() {
        if q >= n || qubits[..i].contains(&q) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Query qubits must be distinct and within the statevector's range",
            ));
        }
    }
    Ok(())
}

fn read_circuit(circuit: &Bound<PyAny>) -> PyResult<Circuit> {
    let json: String = circuit.call_method0("model_dump_json")?.extract()?;
    let circuit: Circuit = serde_json::from_str(&json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Circuit JSON parse error: {e}"))
    })?;
    validate_circuit(&circuit).map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(circuit)
}

/// Dense, double-precision simulation with shared compiled gate execution.
#[pyclass]
pub struct StatevectorSimulator {
    seed: Option<u64>,
    profile: bool,
    memory_budget: usize,
    max_parallel_shots: Option<usize>,
    sample_terminal: bool,
}

#[pymethods]
impl StatevectorSimulator {
    #[new]
    #[pyo3(signature = (seed=None, profile=false, *, max_memory_mb=1024, max_parallel_shots=None, sample_terminal=true))]
    pub fn new(
        seed: Option<u64>,
        profile: bool,
        max_memory_mb: usize,
        max_parallel_shots: Option<usize>,
        sample_terminal: bool,
    ) -> PyResult<Self> {
        let memory_budget = max_memory_mb
            .checked_mul(1024 * 1024)
            .filter(|&v| v > 0 && v <= isize::MAX as usize)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "max_memory_mb must be positive and representable in bytes",
                )
            })?;
        if max_parallel_shots == Some(0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_parallel_shots must be positive or None",
            ));
        }
        Ok(Self {
            seed,
            profile,
            memory_budget,
            max_parallel_shots,
            sample_terminal,
        })
    }

    /// Return classical counts; terminal measurements share a single evolution.
    /// Dynamic circuits use independent trajectories with bounded worker buffers.
    #[pyo3(signature = (circuit, shots=1000, profile=false))]
    pub fn simulate_shots(
        &self,
        py: Python,
        circuit: &Bound<PyAny>,
        shots: usize,
        profile: bool,
    ) -> PyResult<PyObject> {
        let total_t0 = Instant::now();
        let profile = profile || self.profile;
        let circuit = read_circuit(circuit)?;
        let preprocessing_time = total_t0.elapsed().as_secs_f64();
        let fusion_t0 = Instant::now();
        let plan = py
            .allow_threads(|| compile(&circuit.instructions))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let fusion_time = fusion_t0.elapsed().as_secs_f64();
        let seed = self.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let shots_t0 = Instant::now();
        let output = py
            .allow_threads(|| {
                run_shots(
                    &plan,
                    circuit.num_qubits(),
                    circuit.num_cbits(),
                    shots,
                    seed,
                    self.memory_budget,
                    self.max_parallel_shots,
                    self.sample_terminal,
                    profile,
                )
            })
            .map_err(pyo3::exceptions::PyMemoryError::new_err)?;
        let shots_total_time = shots_t0.elapsed().as_secs_f64();
        let result = PyDict::new_bound(py);
        for (key, count) in &output.counts {
            result.set_item(key, count)?;
        }
        if profile {
            let data = ShotsProfile {
                preprocessing_time,
                fusion_time,
                shots_total_time,
                total_time: total_t0.elapsed().as_secs_f64(),
                num_shots: shots,
                shot_times: output.shot_times,
                execution_strategy: output.strategy.into(),
                parallel_shots: output.parallel_shots,
                working_bytes: output.working_bytes,
                deterministic_prefix_ops: output.prefix_ops,
            };
            write_shots_profile("statevector", &data).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to write shots profile: {e}"
                ))
            })?;
        }
        Ok(result.into())
    }

    /// Execute one trajectory. Measurements collapse its returned state.
    pub fn simulate(&self, py: Python, circuit: &Bound<PyAny>) -> PyResult<SimulationResult> {
        let total_t0 = Instant::now();
        let circuit = read_circuit(circuit)?;
        let n = circuit.num_qubits();
        let num_cbits = circuit.num_cbits();
        let (len, bytes) = state_layout(n).map_err(pyo3::exceptions::PyMemoryError::new_err)?;
        let required = bytes.checked_add(num_cbits).ok_or_else(|| {
            pyo3::exceptions::PyMemoryError::new_err("Working memory size overflow")
        })?;
        check_budget(required, self.memory_budget)
            .map_err(pyo3::exceptions::PyMemoryError::new_err)?;
        let plan = py
            .allow_threads(|| compile(&circuit.instructions))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let seed = self.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let (state, cbits, acc) = py
            .allow_threads(|| {
                let mut state = zero_state(len)?;
                let mut cbits = vec![0; num_cbits];
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                let mut acc = self.profile.then(ProfileAcc::default);
                execute(&plan, &mut state, n, &mut cbits, &mut rng, true, &mut acc);
                Ok::<_, String>((state, cbits, acc))
            })
            .map_err(pyo3::exceptions::PyMemoryError::new_err)?;
        let prof = acc
            .map(|a| {
                Py::new(
                    py,
                    SimulationProfile {
                        apply_one_qubit_calls: a.oq_calls,
                        apply_one_qubit_time: a.oq_time,
                        apply_n_qubit_calls: a.nq_calls,
                        apply_n_qubit_time: a.nq_time,
                        measure_qubit_calls: a.mq_calls,
                        measure_qubit_time: a.mq_time,
                        total_time: total_t0.elapsed().as_secs_f64(),
                    },
                )
            })
            .transpose()?;
        Ok(SimulationResult::new(
            state,
            n,
            cbits
                .into_iter()
                .enumerate()
                .map(|(i, b)| (i, b as i32))
                .collect(),
            prof,
        ))
    }
}
