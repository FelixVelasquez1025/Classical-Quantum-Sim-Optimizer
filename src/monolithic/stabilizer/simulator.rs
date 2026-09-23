use super::{
    engine::{allocation, check, Tableau},
    execution::{compile, Op, Stats},
    result::{error, sum, StabilizerResult},
    sampling::{bit, workspace, Affine},
};
use crate::{profiling::write_shots_profile, types::Circuit};
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::Serialize;
use std::{collections::HashMap, time::Instant};

fn parse(obj: &Bound<PyAny>) -> PyResult<Circuit> {
    let json: String = obj.call_method0("model_dump_json")?.extract()?;
    let circuit: Circuit = serde_json::from_str(&json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Circuit JSON parse error: {e}"))
    })?;
    crate::validation::validate_circuit(&circuit)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(circuit)
}
#[derive(Serialize, Default)]
struct Report {
    preprocessing_time: f64,
    fusion_time: f64,
    shots_total_time: f64,
    total_time: f64,
    num_shots: usize,
    execution_strategy: String,
    parallel_shots: usize,
    deterministic_prefix_ops: usize,
    memory_budget_bytes: usize,
    working_bytes: usize,
    terminal_rank: Option<usize>,
    #[serde(flatten)]
    stats: Stats,
}
#[pyclass]
pub struct StabilizerSimulator {
    seed: Option<u64>,
    budget: usize,
    max_parallel_shots: usize,
    sample_terminal: bool,
    clifford_tolerance: f64,
    profile: bool,
}
#[pymethods]
impl StabilizerSimulator {
    #[new]
    #[pyo3(signature=(seed=None,*,max_memory_mb=1024,max_parallel_shots=1,sample_terminal=true,clifford_tolerance=0.0,profile=false))]
    pub fn new(
        seed: Option<u64>,
        max_memory_mb: usize,
        max_parallel_shots: usize,
        sample_terminal: bool,
        clifford_tolerance: f64,
        profile: bool,
    ) -> PyResult<Self> {
        let budget = max_memory_mb
            .checked_mul(1024 * 1024)
            .filter(|&v| v > 0 && v <= isize::MAX as usize)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "max_memory_mb must be positive and representable",
                )
            })?;
        if max_parallel_shots == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_parallel_shots must be positive",
            ));
        }
        if !clifford_tolerance.is_finite()
            || !(0.0..std::f64::consts::FRAC_PI_4).contains(&clifford_tolerance)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "clifford_tolerance must be finite and in [0, pi/4)",
            ));
        }
        Ok(Self {
            seed,
            budget,
            max_parallel_shots,
            sample_terminal,
            clifford_tolerance,
            profile,
        })
    }
    /// Algorithmic eligibility only; malformed circuits raise ValueError. Memory
    /// requirements are checked separately by simulation, before allocation.
    fn supports(&self, py: Python, circuit: &Bound<PyAny>) -> PyResult<bool> {
        let circuit = parse(circuit)?;
        Ok(py.allow_threads(|| compile(&circuit.instructions, self.clifford_tolerance).is_ok()))
    }
    pub fn simulate(&self, py: Python, circuit: &Bound<PyAny>) -> PyResult<StabilizerResult> {
        let start = Instant::now();
        let circuit = parse(circuit)?;
        let ops = py
            .allow_threads(|| compile(&circuit.instructions, self.clifford_tolerance))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let n = circuit.num_qubits();
        let classical = self.classical(circuit.num_cbits()).map_err(error)?;
        check(
            sum(allocation(n).map_err(error)?, classical).map_err(error)?,
            self.budget,
        )
        .map_err(error)?;
        let seed = self.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let (state, cbits, stats) = py
            .allow_threads(|| -> Result<_, String> {
                let mut state = Tableau::new(n, self.budget - classical)?;
                let mut cbits = HashMap::new();
                let mut stats = Stats::default();
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                for op in &ops {
                    op.run(&mut state, &mut cbits, &mut rng, &mut stats, self.profile);
                }
                Ok((state, cbits, stats))
            })
            .map_err(error)?;
        Ok(StabilizerResult {
            state,
            cbits,
            stats,
            budget: self.budget - classical,
            profile: self.profile,
            elapsed: start.elapsed().as_secs_f64(),
        })
    }
    #[pyo3(signature=(circuit,shots=1000,profile=false))]
    pub fn simulate_shots(
        &self,
        py: Python,
        circuit: &Bound<PyAny>,
        shots: usize,
        profile: bool,
    ) -> PyResult<HashMap<String, usize>> {
        let start = Instant::now();
        let circuit = parse(circuit)?;
        let mut report = Report {
            preprocessing_time: start.elapsed().as_secs_f64(),
            num_shots: shots,
            memory_budget_bytes: self.budget,
            ..Report::default()
        };
        let t = Instant::now();
        let ops = py
            .allow_threads(|| compile(&circuit.instructions, self.clifford_tolerance))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        report.fusion_time = t.elapsed().as_secs_f64();
        let seed = self.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let t = Instant::now();
        let counts = py
            .allow_threads(|| {
                self.shots(
                    &ops,
                    circuit.num_qubits(),
                    circuit.num_cbits(),
                    shots,
                    seed,
                    profile,
                    &mut report,
                )
            })
            .map_err(error)?;
        report.shots_total_time = t.elapsed().as_secs_f64();
        report.total_time = start.elapsed().as_secs_f64();
        if profile {
            write_shots_profile("stabilizer", &report).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to write stabilizer profile: {e}"
                ))
            })?;
        }
        Ok(counts)
    }
}
impl StabilizerSimulator {
    fn classical(&self, nc: usize) -> Result<usize, String> {
        nc.checked_mul(64)
            .ok_or_else(|| "Stabilizer classical allocation overflow".into())
    }
    #[allow(clippy::too_many_arguments)]
    fn shots(
        &self,
        ops: &[Op],
        n: usize,
        nc: usize,
        shots: usize,
        seed: u64,
        profile: bool,
        report: &mut Report,
    ) -> Result<HashMap<String, usize>, String> {
        let mut counts = HashMap::new();
        let tb = allocation(n)?;
        let classical = self.classical(nc)?;
        let worker_bytes = sum(tb, classical)?;
        check(worker_bytes, self.budget)?;
        if shots == 0 {
            report.execution_strategy = "empty".into();
            return Ok(counts);
        }
        let prefix_len = ops.iter().take_while(|op| op.unitary()).count();
        let mut prefix = Tableau::new(n, self.budget - classical)?;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut cbits = HashMap::new();
        for op in &ops[..prefix_len] {
            op.run(
                &mut prefix,
                &mut cbits,
                &mut rng,
                &mut report.stats,
                profile,
            );
        }
        report.working_bytes = worker_bytes;
        if self.sample_terminal && ops[prefix_len..].iter().all(Op::terminal) {
            if prefix_len == ops.len() {
                counts.insert("0".repeat(nc), shots);
                report.execution_strategy = "terminal_affine".into();
                report.parallel_shots = 1;
                report.deterministic_prefix_ops = prefix_len;
                report.terminal_rank = Some(0);
                return Ok(counts);
            }
            // Reserve wire discovery storage before allocating it. Repeated
            // measurements do not make the sampling plan grow with gate count.
            let plan_bytes = n.checked_mul(9).ok_or("Stabilizer allocation overflow")?;
            if sum(worker_bytes, plan_bytes)? <= self.budget {
                let mut qs = Vec::with_capacity(n);
                let mut seen = vec![false; n];
                for op in &ops[prefix_len..] {
                    if let Op::Measure(q, _) = op {
                        if !seen[*q] {
                            seen[*q] = true;
                            qs.push(*q);
                        }
                    }
                }
                qs.sort_unstable();
                drop(seen);
                let sampling_bytes = sum(sum(worker_bytes, plan_bytes)?, workspace(qs.len())?)?;
                if sampling_bytes <= self.budget {
                    let sampler = Affine::build(&mut prefix, &qs, &mut report.stats, profile);
                    let mut bits = vec![0; sampler.words];
                    let mut outputs = vec![usize::MAX; nc];
                    for op in &ops[prefix_len..] {
                        if let Op::Measure(q, c) = op {
                            outputs[*c] = qs.binary_search(q).unwrap();
                        }
                    }
                    for _ in 0..shots {
                        sampler.sample(&mut bits, &mut rng);
                        let key: String = outputs
                            .iter()
                            .rev()
                            .map(|&i| {
                                if i != usize::MAX && bit(&bits, i) {
                                    '1'
                                } else {
                                    '0'
                                }
                            })
                            .collect();
                        *counts.entry(key).or_insert(0) += 1;
                    }
                    report.execution_strategy = "terminal_affine".into();
                    report.parallel_shots = 1;
                    report.deterministic_prefix_ops = prefix_len;
                    report.working_bytes = sampling_bytes;
                    report.terminal_rank = Some(sampler.rank);
                    return Ok(counts);
                }
            }
        }

        let retain = prefix_len > 0 && sum(tb, worker_bytes)? <= self.budget;
        let reserved = if retain { tb } else { 0 };
        let workers = self
            .max_parallel_shots
            .min(shots)
            .min(rayon::current_num_threads())
            .min((self.budget - reserved) / worker_bytes)
            .max(1);
        report.parallel_shots = workers;
        report.deterministic_prefix_ops = if retain { prefix_len } else { 0 };
        report.working_bytes = sum(reserved, workers * worker_bytes)?;
        report.execution_strategy = if retain {
            "prefix_trajectories"
        } else {
            "trajectories"
        }
        .into();
        // Drop an unretained prefix before any worker allocates another tableau.
        let prefix = if retain {
            Some(prefix)
        } else {
            drop(prefix);
            None
        };
        let results = (0..workers)
            .into_par_iter()
            .map(|worker| -> Result<_, String> {
                let mut state = if let Some(p) = &prefix {
                    p.clone()
                } else {
                    Tableau::new(n, tb)?
                };
                let mut cbits = HashMap::new();
                let mut local = HashMap::new();
                let mut stats = Stats::default();
                for (iteration, shot) in (worker..shots).step_by(workers).enumerate() {
                    if iteration > 0 {
                        if let Some(p) = &prefix {
                            state.restore(p);
                        } else {
                            state.reset();
                        }
                    }
                    cbits.clear();
                    let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(shot as u64));
                    for op in &ops[if retain { prefix_len } else { 0 }..] {
                        op.run(&mut state, &mut cbits, &mut rng, &mut stats, profile);
                    }
                    *local.entry(format_cbits(&cbits, nc)).or_insert(0) += 1;
                }
                Ok((local, stats))
            })
            .collect::<Result<Vec<_>, String>>()?;
        for (local, stats) in results {
            report.stats.add(&stats);
            for (k, v) in local {
                *counts.entry(k).or_insert(0) += v;
            }
        }
        Ok(counts)
    }
}

fn format_cbits(cbits: &HashMap<usize, i32>, n: usize) -> String {
    (0..n)
        .rev()
        .map(|q| {
            if cbits.get(&q).copied().unwrap_or(0) == 0 {
                '0'
            } else {
                '1'
            }
        })
        .collect()
}
