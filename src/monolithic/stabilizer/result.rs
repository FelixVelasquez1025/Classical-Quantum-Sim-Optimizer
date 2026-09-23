use super::{
    engine::{allocation, check, Tableau},
    execution::Stats,
    sampling::{workspace, Affine},
};
use pyo3::{prelude::*, types::PyDict};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

pub(super) fn error(e: String) -> PyErr {
    if e.contains("memory") || e.contains("allocation") {
        pyo3::exceptions::PyMemoryError::new_err(e)
    } else {
        pyo3::exceptions::PyRuntimeError::new_err(e)
    }
}
pub(super) fn sum(a: usize, b: usize) -> Result<usize, String> {
    a.checked_add(b)
        .ok_or_else(|| "Stabilizer allocation overflow".into())
}
#[pyclass]
pub struct StabilizerResult {
    pub(super) state: Tableau,
    pub(super) cbits: HashMap<usize, i32>,
    pub(super) budget: usize,
    pub(super) stats: Stats,
    pub(super) profile: bool,
    pub(super) elapsed: f64,
}
impl StabilizerResult {
    fn query(&self, qs: Option<Vec<usize>>) -> PyResult<Vec<usize>> {
        let count = qs.as_ref().map_or(self.state.n, Vec::len);
        let extra = count
            .checked_mul(64)
            .ok_or_else(|| error("Stabilizer allocation overflow".into()))?;
        check(
            sum(allocation(self.state.n).map_err(error)?, extra).map_err(error)?,
            self.budget,
        )
        .map_err(error)?;
        let qs = qs.unwrap_or_else(|| (0..self.state.n).rev().collect());
        let mut seen = std::collections::HashSet::new();
        if qs.iter().any(|&q| q >= self.state.n || !seen.insert(q)) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Query qubits must be distinct and within the result",
            ));
        }
        Ok(qs)
    }
    fn sampler(&self, qs: &[usize]) -> Result<Affine, String> {
        check(
            sum(
                allocation(self.state.n)?
                    .checked_mul(2)
                    .ok_or("Stabilizer allocation overflow")?,
                workspace(qs.len())?,
            )?,
            self.budget,
        )?;
        Ok(Affine::build(
            &mut self.state.clone(),
            qs,
            &mut Stats::default(),
            false,
        ))
    }
}
#[pymethods]
impl StabilizerResult {
    #[getter]
    fn num_qubits(&self) -> usize {
        self.state.n
    }
    #[getter]
    fn classical_bits(&self) -> HashMap<usize, i32> {
        self.cbits.clone()
    }
    #[getter]
    fn stabilizers(&self, py: Python) -> PyResult<Vec<String>> {
        let output = self
            .state
            .n
            .checked_add(128)
            .and_then(|v| v.checked_mul(self.state.n))
            .ok_or_else(|| error("Stabilizer allocation overflow".into()))?;
        check(
            sum(allocation(self.state.n).map_err(error)?, output).map_err(error)?,
            self.budget,
        )
        .map_err(error)?;
        Ok(py.allow_threads(|| self.state.generators()))
    }
    #[getter]
    fn diagnostics(&self, py: Python) -> PyResult<PyObject> {
        let d = PyDict::new_bound(py);
        d.set_item("num_qubits", self.state.n)?;
        d.set_item("tableau_bytes", allocation(self.state.n).map_err(error)?)?;
        d.set_item("gate_calls", self.stats.gate_calls)?;
        d.set_item("measure_calls", self.stats.measure_calls)?;
        d.set_item("random_measurements", self.stats.random_measurements)?;
        d.set_item(
            "deterministic_measurements",
            self.stats.deterministic_measurements,
        )?;
        d.set_item("measure_time", self.stats.measure_time)?;
        Ok(d.into())
    }
    #[getter]
    fn profile(&self, py: Python) -> PyResult<PyObject> {
        if !self.profile {
            return Ok(py.None());
        }
        let obj = self.diagnostics(py)?;
        obj.bind(py)
            .downcast::<PyDict>()?
            .set_item("total_time", self.elapsed)?;
        Ok(obj)
    }
    /// Pauli strings and default query outputs put the highest logical wire first.
    fn expectation_value(&mut self, py: Python, pauli: &str) -> PyResult<i8> {
        if pauli.len() != self.state.n || !pauli.bytes().all(|c| b"IXYZ".contains(&c)) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Expected one I, X, Y or Z per qubit, highest index first",
            ));
        }
        check(
            sum(
                allocation(self.state.n).map_err(error)?,
                self.state.words * 16,
            )
            .map_err(error)?,
            self.budget,
        )
        .map_err(error)?;
        let mut x = vec![0; self.state.words];
        let mut z = vec![0; self.state.words];
        for (q, c) in pauli.bytes().rev().enumerate() {
            if c == b'X' || c == b'Y' {
                x[q / 64] |= 1 << (q % 64);
            }
            if c == b'Z' || c == b'Y' {
                z[q / 64] |= 1 << (q % 64);
            }
        }
        Ok(py.allow_threads(|| self.state.expectation(&x, &z)))
    }
    #[pyo3(signature=(shots=1000,qubits=None,seed=None))]
    fn counts(
        &self,
        py: Python,
        shots: usize,
        qubits: Option<Vec<usize>>,
        seed: Option<u64>,
    ) -> PyResult<HashMap<String, usize>> {
        let qs = self.query(qubits)?;
        if shots == 0 {
            return Ok(HashMap::new());
        }
        py.allow_threads(|| {
            let sampler = self.sampler(&qs)?;
            let mut bits = vec![0; sampler.words];
            let mut rng =
                ChaCha8Rng::seed_from_u64(seed.unwrap_or_else(|| rand::thread_rng().gen()));
            let mut counts = HashMap::new();
            for _ in 0..shots {
                sampler.sample(&mut bits, &mut rng);
                *counts.entry(sampler.bitstring(&bits)).or_insert(0) += 1;
            }
            Ok(counts)
        })
        .map_err(error)
    }
    #[pyo3(signature=(qubits=None))]
    fn probabilities(&self, py: Python, qubits: Option<Vec<usize>>) -> PyResult<PyObject> {
        let qs = self.query(qubits)?;
        let pairs = py
            .allow_threads(|| -> Result<_, String> {
                let s = self.sampler(&qs)?;
                let size = 1usize
                    .checked_shl(s.rank.try_into().unwrap_or(u32::MAX))
                    .ok_or("Stabilizer probability allocation overflow")?;
                let output = size
                    .checked_mul(sum(qs.len(), 128)?)
                    .ok_or("Stabilizer probability allocation overflow")?;
                check(
                    sum(
                        sum(allocation(self.state.n)?, workspace(qs.len())?)?,
                        output,
                    )?,
                    self.budget,
                )?;
                let prob = 2.0f64.powf(-(s.rank as f64));
                let mut bits = vec![0; s.words];
                Ok((0..size)
                    .map(|i| {
                        s.outcome(i, &mut bits);
                        (s.bitstring(&bits), prob)
                    })
                    .collect::<Vec<_>>())
            })
            .map_err(error)?;
        let out = PyDict::new_bound(py);
        let int = py.import_bound("builtins")?.getattr("int")?;
        for (bits, p) in pairs {
            out.set_item(
                int.call1((if bits.is_empty() { "0" } else { &bits }, 2))?,
                p,
            )?;
        }
        Ok(out.into())
    }
    fn probability(&self, py: Python, bitstring: &str) -> PyResult<f64> {
        if bitstring.len() != self.state.n || !bitstring.bytes().all(|c| c == b'0' || c == b'1') {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Expected one binary digit per qubit, highest index first",
            ));
        }
        py.allow_threads(|| -> Result<f64, String> {
            let s = self.sampler(&(0..self.state.n).rev().collect::<Vec<_>>())?;
            let mut bits = vec![0; s.words];
            for (i, c) in bitstring.bytes().enumerate() {
                if c == b'1' {
                    bits[i / 64] |= 1 << (i % 64);
                }
            }
            Ok(if s.contains(&mut bits) {
                2.0f64.powf(-(s.rank as f64))
            } else {
                0.
            })
        })
        .map_err(error)
    }
}
