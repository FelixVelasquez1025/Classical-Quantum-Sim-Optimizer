use super::engine::{checked_bytes, Mps};
use num_complex::Complex64 as C;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{prelude::*, types::PyDict};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

pub(super) fn error(message: String) -> PyErr {
    if message.contains("memory") || message.contains("allocation") {
        pyo3::exceptions::PyMemoryError::new_err(message)
    } else {
        pyo3::exceptions::PyRuntimeError::new_err(message)
    }
}
pub(super) fn validate_query(qs: &[usize], n: usize) -> PyResult<()> {
    for (i, q) in qs.iter().enumerate() {
        if *q >= n || qs[..i].contains(q) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Query qubits must be distinct and in range",
            ));
        }
    }
    Ok(())
}
fn dimension(n: usize) -> PyResult<usize> {
    if n >= usize::BITS as usize {
        return Err(pyo3::exceptions::PyMemoryError::new_err(
            "Dense MPS output dimension is too large",
        ));
    }
    Ok(1usize << n)
}
pub(super) fn stats_dict<'py>(py: Python<'py>, mps: &Mps) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    let s = &mps.stats;
    d.set_item("svd_calls", s.svd_calls)?;
    d.set_item("svd_time", s.svd_time)?;
    d.set_item("routing_swaps", s.routing_swaps)?;
    d.set_item("center_moves", s.center_moves)?;
    d.set_item("peak_bond_dimension", s.peak_bond_dimension)?;
    d.set_item("peak_tensor_bytes", s.peak_tensor_bytes)?;
    d.set_item("peak_working_bytes", s.peak_working_bytes)?;
    d.set_item("discarded_weight", s.discarded_weight)?;
    d.set_item("truncations", s.truncations)?;
    d.set_item("bond_cap_truncations", s.bond_cap_truncations)?;
    d.set_item("norm_squared", mps.norm_squared())?;
    d.set_item("bond_dimensions", mps.bonds())?;
    d.set_item("qubit_order", mps.order())?;
    Ok(d)
}

#[pyclass]
pub struct MpsResult {
    pub(super) state: Mps,
    pub(super) cbits: HashMap<usize, i32>,
    pub(super) total_time: f64,
}
#[pymethods]
impl MpsResult {
    #[getter]
    fn num_qubits(&self) -> usize {
        self.state.len()
    }
    #[getter]
    fn classical_bits(&self) -> HashMap<usize, i32> {
        self.cbits.clone()
    }
    #[getter]
    fn bond_dimensions(&self) -> Vec<usize> {
        self.state.bonds()
    }
    #[getter]
    fn qubit_order(&self) -> Vec<usize> {
        self.state.order()
    }
    #[getter]
    fn diagnostics(&self, py: Python) -> PyResult<PyObject> {
        Ok(stats_dict(py, &self.state)?.into())
    }
    #[getter]
    fn profile(&self, py: Python) -> PyResult<PyObject> {
        if !self.state.options.profile {
            return Ok(py.None());
        }
        let d = stats_dict(py, &self.state)?;
        d.set_item("total_time", self.total_time)?;
        Ok(d.into())
    }
    /// Explicit dense export in logical little-endian order, bounded by memory.
    #[getter]
    fn statevector<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<C>>> {
        let n = self.state.len();
        let size = dimension(n)?;
        let output = checked_bytes(size, 16).map_err(error)?;
        self.state.query_memory(output, false).map_err(error)?;
        let values = py.allow_threads(|| {
            let mut values = Vec::with_capacity(size);
            let mut bits = vec![0; n];
            let (mut work, mut next) = (Vec::new(), Vec::new());
            for basis in 0..size {
                for (q, b) in bits.iter_mut().enumerate() {
                    *b = (basis >> q) & 1;
                }
                values.push(
                    self.state
                        .amplitude_with_buffers(&bits, &mut work, &mut next),
                );
            }
            values
        });
        Ok(values.into_pyarray_bound(py))
    }
    /// Bitstring is MSB first, and can exceed the native machine word width.
    fn amplitude(&self, py: Python, bitstring: &str) -> PyResult<PyObject> {
        if bitstring.len() != self.state.len() || !bitstring.bytes().all(|b| b == b'0' || b == b'1')
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Expected one binary digit per qubit, MSB first",
            ));
        }
        let bits: Vec<_> = bitstring
            .bytes()
            .rev()
            .map(|b| (b - b'0') as usize)
            .collect();
        self.state.query_memory(0, false).map_err(error)?;
        let amp = py.allow_threads(|| self.state.amplitude(&bits));
        Ok(pyo3::types::PyComplex::from_doubles_bound(py, amp.re, amp.im).into())
    }
    #[pyo3(signature=(qubits=None))]
    fn probabilities(&self, py: Python, qubits: Option<Vec<usize>>) -> PyResult<PyObject> {
        let qs = qubits.unwrap_or_else(|| (0..self.state.len()).rev().collect());
        validate_query(&qs, self.state.len())?;
        let size = dimension(qs.len())?;
        // Include a conservative allowance for the returned Python dictionary.
        self.state
            .query_memory(checked_bytes(size, 96).map_err(error)?, true)
            .map_err(error)?;
        let probs = py
            .allow_threads(|| -> Result<Vec<f64>, String> {
                if qs.len() == self.state.len() {
                    let mut probs = Vec::with_capacity(size);
                    let mut bits = vec![0; qs.len()];
                    let (mut work, mut next) = (Vec::new(), Vec::new());
                    for basis in 0..size {
                        for (i, &q) in qs.iter().enumerate() {
                            bits[q] = (basis >> (qs.len() - 1 - i)) & 1;
                        }
                        probs.push(
                            self.state
                                .amplitude_with_buffers(&bits, &mut work, &mut next)
                                .norm_sqr(),
                        );
                    }
                    return Ok(probs);
                }
                let identity = [
                    [C::new(1., 0.), C::default()],
                    [C::default(), C::new(1., 0.)],
                ];
                let mut ops = vec![identity; self.state.len()];
                let mut probs = Vec::with_capacity(size);
                for basis in 0..size {
                    for (i, &q) in qs.iter().enumerate() {
                        let bit = (basis >> (qs.len() - 1 - i)) & 1;
                        ops[q] = [[C::default(); 2]; 2];
                        ops[q][bit][bit] = C::new(1., 0.);
                    }
                    probs.push(self.state.contract(&ops)?.re.max(0.));
                }
                Ok(probs)
            })
            .map_err(error)?;
        let d = PyDict::new_bound(py);
        for (i, p) in probs.into_iter().enumerate() {
            if p > 0. {
                d.set_item(i, p)?;
            }
        }
        Ok(d.into())
    }
    #[pyo3(signature=(shots=1000, qubits=None, seed=None))]
    fn counts(
        &self,
        py: Python,
        shots: usize,
        qubits: Option<Vec<usize>>,
        seed: Option<u64>,
    ) -> PyResult<HashMap<String, usize>> {
        let qs = qubits.unwrap_or_else(|| (0..self.state.len()).rev().collect());
        validate_query(&qs, self.state.len())?;
        let mut rng = ChaCha8Rng::seed_from_u64(seed.unwrap_or_else(|| rand::thread_rng().gen()));
        py.allow_threads(|| {
            let mut counts = HashMap::new();
            for _ in 0..shots {
                let bits = self.state.sample(&mut rng)?;
                let key: String = qs
                    .iter()
                    .map(|&q| if bits[q] == 0 { '0' } else { '1' })
                    .collect();
                *counts.entry(key).or_insert(0) += 1;
            }
            Ok(counts)
        })
        .map_err(error)
    }
    /// Pauli string is MSB first (like a displayed basis state).
    fn expectation_value(&self, py: Python, pauli: &str) -> PyResult<f64> {
        if pauli.len() != self.state.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Expected one Pauli per qubit",
            ));
        }
        let identity = [
            [C::new(1., 0.), C::default()],
            [C::default(), C::new(1., 0.)],
        ];
        let ops = pauli
            .bytes()
            .rev()
            .map(|p| match p {
                b'I' => Ok(identity),
                b'X' => Ok(crate::gates::X),
                b'Y' => Ok(crate::gates::Y),
                b'Z' => Ok(crate::gates::Z),
                _ => Err(pyo3::exceptions::PyValueError::new_err(
                    "Pauli string must contain only I, X, Y, Z",
                )),
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(py
            .allow_threads(|| self.state.contract(&ops))
            .map_err(error)?
            .re)
    }
    fn fidelity(&self, py: Python, other: PyReadonlyArray1<C>) -> PyResult<f64> {
        let n = self.state.len();
        let size = dimension(n)?;
        if other.len()? != size {
            return Ok(0.);
        }
        self.state.query_memory(0, false).map_err(error)?;
        let reference = other.as_array();
        // The NumPy view remains borrowed while the GIL is released.
        Ok(py.allow_threads(|| {
            let mut bits = vec![0; n];
            let mut dot = C::default();
            for (basis, v) in reference.iter().enumerate() {
                for (q, b) in bits.iter_mut().enumerate() {
                    *b = (basis >> q) & 1;
                }
                dot += self.state.amplitude(&bits).conj() * v;
            }
            dot.norm_sqr()
        }))
    }
}
