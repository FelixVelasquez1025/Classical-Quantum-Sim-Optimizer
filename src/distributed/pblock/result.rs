use super::model::{bytes, dimension, sample, BlockPool, C};
use numpy::{IntoPyArray, PyArray1};
use pyo3::{
    prelude::*,
    types::{PyComplex, PyDict},
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

pub(super) fn error(message: String) -> PyErr {
    if message.contains("memory")
        || message.contains("allocation")
        || message.contains("max_block_qubits")
    {
        pyo3::exceptions::PyMemoryError::new_err(message)
    } else {
        pyo3::exceptions::PyRuntimeError::new_err(message)
    }
}
#[pyclass]
pub struct PBlockResult {
    pub(super) pool: BlockPool,
    pub(super) physical: Vec<usize>,
    pub(super) cbits: HashMap<usize, i32>,
    pub(super) elapsed: f64,
}
impl PBlockResult {
    fn query(&self, qs: Option<Vec<usize>>) -> PyResult<Vec<usize>> {
        let qs = qs.unwrap_or_else(|| self.physical.iter().rev().copied().collect());
        let mut local = Vec::with_capacity(qs.len());
        for q in qs {
            let i = self.physical.binary_search(&q).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Physical qubit {q} is not in the result"
                ))
            })?;
            if local.contains(&i) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Query qubits must be distinct",
                ));
            }
            local.push(i);
        }
        Ok(local)
    }
}
#[pymethods]
impl PBlockResult {
    #[getter]
    fn num_qubits(&self) -> usize {
        self.physical.len()
    }
    #[getter]
    fn physical_qubits(&self) -> Vec<usize> {
        self.physical.clone()
    }
    #[getter]
    fn classical_bits(&self) -> HashMap<usize, i32> {
        self.cbits.clone()
    }
    #[getter]
    fn block_qubits(&self) -> Vec<Vec<usize>> {
        self.pool
            .block_qubits()
            .into_iter()
            .map(|qs| qs.into_iter().map(|q| self.physical[q]).collect())
            .collect()
    }
    #[getter]
    fn diagnostics(&self, py: Python) -> PyResult<PyObject> {
        let s = &self.pool.stats;
        let d = PyDict::new_bound(py);
        d.set_item("merge_calls", s.merge_calls)?;
        d.set_item("merge_time", s.merge_time)?;
        d.set_item("merge_output_amplitudes", s.merge_output_amplitudes)?;
        d.set_item("split_calls", s.split_calls)?;
        d.set_item("separation_checks", s.separation_checks)?;
        d.set_item("separation_time", s.separation_time)?;
        d.set_item("separation_residual", s.separation_residual)?;
        d.set_item("skipped_controlled_gates", s.skipped_controlled_gates)?;
        d.set_item("peak_block_qubits", s.peak_block_qubits)?;
        d.set_item("peak_live_amplitudes", s.peak_live_amplitudes)?;
        d.set_item("peak_working_bytes", s.peak_working_bytes)?;
        d.set_item(
            "block_sizes",
            self.pool
                .block_qubits()
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
        )?;
        d.set_item(
            "live_amplitudes",
            self.pool
                .blocks
                .iter()
                .flatten()
                .map(|b| b.state.len())
                .sum::<usize>(),
        )?;
        d.set_item("norm_squared", self.pool.norm_squared())?;
        Ok(d.into())
    }
    #[getter]
    fn profile(&self, py: Python) -> PyResult<PyObject> {
        if !self.pool.options.profile {
            return Ok(py.None());
        }
        let obj = self.diagnostics(py)?;
        obj.bind(py)
            .downcast::<PyDict>()?
            .set_item("total_time", self.elapsed)?;
        Ok(obj)
    }
    /// Explicit dense export: local bit i corresponds to physical_qubits[i].
    #[getter]
    fn statevector<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<C>>> {
        let size = dimension(self.num_qubits()).map_err(error)?;
        self.pool
            .check(
                bytes(size, 16)
                    .map_err(error)?
                    .checked_add(bytes(self.num_qubits(), 8).map_err(error)?)
                    .ok_or_else(|| error("P-block allocation overflow".into()))?,
            )
            .map_err(error)?;
        let state = py.allow_threads(|| {
            let mut bits = vec![0; self.num_qubits()];
            let mut state = Vec::with_capacity(size);
            for basis in 0..size {
                for (i, bit) in bits.iter_mut().enumerate() {
                    *bit = (basis >> i) & 1;
                }
                state.push(self.pool.amplitude(&bits));
            }
            state
        });
        Ok(state.into_pyarray_bound(py))
    }
    fn amplitude(&self, py: Python, bitstring: &str) -> PyResult<PyObject> {
        if bitstring.len() != self.num_qubits()
            || !bitstring.bytes().all(|c| c == b'0' || c == b'1')
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Expected one binary digit per physical qubit, highest physical index first",
            ));
        }
        self.pool
            .check(bytes(self.num_qubits(), 8).map_err(error)?)
            .map_err(error)?;
        let bits: Vec<_> = bitstring
            .bytes()
            .rev()
            .map(|c| (c - b'0') as usize)
            .collect();
        let amp = py.allow_threads(|| self.pool.amplitude(&bits));
        Ok(PyComplex::from_doubles_bound(py, amp.re, amp.im).into())
    }
    #[pyo3(signature=(qubits=None))]
    fn probabilities(&self, py: Python, qubits: Option<Vec<usize>>) -> PyResult<PyObject> {
        let qs = self.query(qubits)?;
        let size = dimension(qs.len()).map_err(error)?;
        let mut table_size = 0usize;
        for b in self.pool.blocks.iter().flatten() {
            let k = qs.iter().filter(|q| b.qubits.contains(q)).count();
            table_size = table_size
                .checked_add(dimension(k).map_err(error)?)
                .ok_or_else(|| error("P-block allocation overflow".into()))?;
        }
        let extra = bytes(size, 96)
            .map_err(error)?
            .checked_add(bytes(table_size, 8).map_err(error)?)
            .ok_or_else(|| error("P-block allocation overflow".into()))?;
        self.pool.check(extra).map_err(error)?;
        let probs = py.allow_threads(|| {
            let tables: Vec<_> = self
                .pool
                .blocks
                .iter()
                .flatten()
                .map(|b| {
                    let selected: Vec<_> = qs
                        .iter()
                        .enumerate()
                        .filter_map(|(i, q)| {
                            b.qubits.iter().position(|w| w == q).map(|local| (i, local))
                        })
                        .collect();
                    let local: Vec<_> = selected.iter().map(|(_, q)| *q).collect();
                    (
                        selected,
                        crate::engine::marginal_probs(&b.state, b.qubits.len(), &local),
                    )
                })
                .collect();
            (0..size)
                .map(|basis| {
                    tables
                        .iter()
                        .map(|(selected, probs)| {
                            let index = selected.iter().fold(0, |i, (pos, _)| {
                                (i << 1) | ((basis >> (qs.len() - 1 - pos)) & 1)
                            });
                            probs[index]
                        })
                        .product::<f64>()
                })
                .collect::<Vec<_>>()
        });
        let d = PyDict::new_bound(py);
        for (basis, p) in probs.into_iter().enumerate() {
            if p > 0. {
                d.set_item(basis, p)?;
            }
        }
        Ok(d.into())
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
        let seed = seed.unwrap_or_else(|| rand::thread_rng().gen());
        py.allow_threads(|| {
            let mut counts = HashMap::new();
            if shots == 0 {
                return Ok(counts);
            }
            let tables = self.pool.sampling_tables()?;
            let mut bits = vec![0; self.num_qubits()];
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            for _ in 0..shots {
                sample(&tables, &mut bits, &mut rng);
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
    /// Pauli characters follow descending physical-qubit order.
    fn expectation_value(&self, py: Python, pauli: &str) -> PyResult<f64> {
        if pauli.len() != self.num_qubits() || !pauli.bytes().all(|c| b"IXYZ".contains(&c)) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Expected one I, X, Y or Z per physical qubit",
            ));
        }
        self.pool
            .check(bytes(self.num_qubits(), 1).map_err(error)?)
            .map_err(error)?;
        let ops: Vec<_> = pauli.bytes().rev().collect();
        Ok(py
            .allow_threads(|| {
                self.pool
                    .blocks
                    .iter()
                    .flatten()
                    .map(|b| {
                        b.state
                            .iter()
                            .enumerate()
                            .map(|(basis, &amp)| {
                                let mut out = basis;
                                let mut phase = C::new(1., 0.);
                                for (i, &q) in b.qubits.iter().enumerate() {
                                    let bit = (basis >> i) & 1;
                                    match ops[q] {
                                        b'X' => out ^= 1 << i,
                                        b'Y' => {
                                            out ^= 1 << i;
                                            phase *= C::new(0., if bit == 0 { 1. } else { -1. });
                                        }
                                        b'Z' if bit == 1 => {
                                            phase = -phase;
                                        }
                                        _ => {}
                                    }
                                }
                                b.state[out].conj() * phase * amp
                            })
                            .sum::<C>()
                    })
                    .product::<C>()
            })
            .re)
    }
}
