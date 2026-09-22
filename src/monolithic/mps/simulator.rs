use super::{
    engine::{checked_bytes, Mps, Options, Stats},
    execution::{compile, Op},
    result::{error, MpsResult},
};
use crate::{
    profiling::write_shots_profile,
    types::{format_cbits, Circuit},
};
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::time::Instant;

#[pyclass]
pub struct MpsSimulator {
    seed: Option<u64>,
    options: Options,
    max_parallel_shots: usize,
    sample_terminal: bool,
}

#[derive(Serialize)]
struct ShotsProfile {
    preprocessing_time: f64,
    shots_total_time: f64,
    total_time: f64,
    num_shots: usize,
    execution_strategy: String,
    parallel_shots: usize,
    deterministic_prefix_ops: usize,
    #[serde(flatten)]
    stats: Stats,
}
fn merge_stats(a: &mut Stats, b: &Stats) {
    a.svd_calls += b.svd_calls;
    a.svd_time += b.svd_time;
    a.routing_swaps += b.routing_swaps;
    a.center_moves += b.center_moves;
    a.peak_bond_dimension = a.peak_bond_dimension.max(b.peak_bond_dimension);
    a.peak_tensor_bytes = a.peak_tensor_bytes.max(b.peak_tensor_bytes);
    a.peak_working_bytes = a.peak_working_bytes.max(b.peak_working_bytes);
    a.discarded_weight += b.discarded_weight;
    a.truncations += b.truncations;
    a.bond_cap_truncations += b.bond_cap_truncations;
}

fn prepare(circuit: &Bound<PyAny>) -> PyResult<(Circuit, Vec<Op>)> {
    let json: String = circuit.call_method0("model_dump_json")?.extract()?;
    let circuit: Circuit = serde_json::from_str(&json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Circuit JSON parse error: {e}"))
    })?;
    crate::validation::validate_circuit(&circuit)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let ops = compile(&circuit.instructions).map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((circuit, ops))
}
#[pymethods]
impl MpsSimulator {
    #[new]
    #[pyo3(signature=(seed=None,max_bond_dimension=None,truncation_threshold=1e-12,*,max_discarded_weight=0.0,max_memory_mb=1024,max_parallel_shots=1,sample_terminal=true,profile=false))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        seed: Option<u64>,
        max_bond_dimension: Option<usize>,
        truncation_threshold: f64,
        max_discarded_weight: f64,
        max_memory_mb: usize,
        max_parallel_shots: usize,
        sample_terminal: bool,
        profile: bool,
    ) -> PyResult<Self> {
        if max_bond_dimension == Some(0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_bond_dimension must be positive or None",
            ));
        }
        if !truncation_threshold.is_finite() || truncation_threshold < 0. {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "truncation_threshold must be finite and non-negative",
            ));
        }
        if !max_discarded_weight.is_finite() || !(0.0..1.0).contains(&max_discarded_weight) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_discarded_weight must be finite and in [0,1)",
            ));
        }
        let memory = max_memory_mb
            .checked_mul(1024 * 1024)
            .filter(|&x| x > 0 && x <= isize::MAX as usize)
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
        Ok(Self {
            seed,
            options: Options {
                max_bond: max_bond_dimension,
                threshold: truncation_threshold,
                discarded_budget: max_discarded_weight,
                memory,
                profile,
            },
            max_parallel_shots,
            sample_terminal,
        })
    }
    pub fn simulate(&self, py: Python, circuit: &Bound<PyAny>) -> PyResult<MpsResult> {
        let start = Instant::now();
        let (circuit, ops) = prepare(circuit)?;
        let mut rng =
            ChaCha8Rng::seed_from_u64(self.seed.unwrap_or_else(|| rand::thread_rng().gen()));
        let (state, cbits) = py
            .allow_threads(|| -> Result<_, String> {
                let mut state = Mps::new(circuit.num_qubits(), self.options.clone())?;
                let mut cbits = HashMap::new();
                for op in &ops {
                    op.run(&mut state, &mut cbits, &mut rng)?;
                }
                state.move_center(0)?;
                Ok((state, cbits))
            })
            .map_err(error)?;
        Ok(MpsResult {
            state,
            cbits,
            total_time: start.elapsed().as_secs_f64(),
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
        let (circuit, ops) = prepare(circuit)?;
        let preprocessing_time = start.elapsed().as_secs_f64();
        let n = circuit.num_qubits();
        let nc = circuit.num_cbits();
        let classical_bytes = checked_bytes(nc, 32).map_err(error)?;
        if classical_bytes >= self.options.memory {
            return Err(error(
                "Classical shot workspace exceeds max_memory_mb".into(),
            ));
        }
        let base_seed = self.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let mut options = self.options.clone();
        options.profile = profile;
        options.memory -= classical_bytes;
        let prefix_len = ops.iter().take_while(|op| op.unitary()).count();
        let terminal = self.sample_terminal && ops[prefix_len..].iter().all(Op::terminal);
        let exec_start = Instant::now();
        let (counts, stats, strategy, workers, used_prefix) = py
            .allow_threads(|| {
                self.run_shots(
                    n,
                    nc,
                    &ops,
                    shots,
                    base_seed,
                    options,
                    classical_bytes,
                    prefix_len,
                    terminal,
                )
            })
            .map_err(error)?;
        if profile {
            let report = ShotsProfile {
                preprocessing_time,
                shots_total_time: exec_start.elapsed().as_secs_f64(),
                total_time: start.elapsed().as_secs_f64(),
                num_shots: shots,
                execution_strategy: strategy.into(),
                parallel_shots: workers,
                deterministic_prefix_ops: used_prefix,
                stats,
            };
            write_shots_profile("mps", &report).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to write MPS profile: {e}"
                ))
            })?;
        }
        Ok(counts)
    }
}

type ShotOutput = (HashMap<String, usize>, Stats, &'static str, usize, usize);
impl MpsSimulator {
    #[allow(clippy::too_many_arguments)]
    fn run_shots(
        &self,
        n: usize,
        nc: usize,
        ops: &[Op],
        shots: usize,
        base_seed: u64,
        options: Options,
        classical_bytes: usize,
        prefix_len: usize,
        terminal: bool,
    ) -> Result<ShotOutput, String> {
        let mut counts = HashMap::new();
        let mut stats = Stats::default();
        if shots == 0 {
            return Ok((counts, stats, "empty", 0, 0));
        }
        let mut prefix = Mps::new(n, options.clone())?;
        let mut rng = ChaCha8Rng::seed_from_u64(base_seed);
        for op in &ops[..prefix_len] {
            op.run(&mut prefix, &mut HashMap::new(), &mut rng)?;
        }
        if terminal {
            prefix.move_center(0)?;
            if !ops[prefix_len..]
                .iter()
                .any(|op| matches!(op, Op::Measure(..)))
            {
                counts.insert("0".repeat(nc), shots);
                return Ok((counts, prefix.stats, "terminal_sampling", 1, prefix_len));
            }
            for shot in 0..shots {
                let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(shot as u64));
                let bits = prefix.sample(&mut rng)?;
                let mut cbits = HashMap::new();
                for op in &ops[prefix_len..] {
                    if let Op::Measure(q, c) = op {
                        cbits.insert(*c, bits[*q] as i32);
                    }
                }
                *counts.entry(format_cbits(&cbits, nc)).or_insert(0) += 1;
            }
            return Ok((counts, prefix.stats, "terminal_sampling", 1, prefix_len));
        }
        // Keep a prefix only when a trajectory plus workspace can coexist.
        let retain = prefix_len > 0
            && prefix.bytes().saturating_mul(3) < options.memory
            && prefix
                .bytes()
                .saturating_add(prefix.stats.peak_working_bytes)
                <= options.memory;
        let reserved = if retain { prefix.bytes() } else { 0 };
        let initial = checked_bytes(n, 128)?
            .saturating_add(classical_bytes)
            .max(
                prefix
                    .stats
                    .peak_working_bytes
                    .saturating_add(classical_bytes),
            )
            .max(4096);
        if retain {
            merge_stats(&mut stats, &prefix.stats);
            prefix.stats = Stats::default();
        }
        let workers = self
            .max_parallel_shots
            .min(shots)
            .min(rayon::current_num_threads())
            .min(((options.memory - reserved) / initial).max(1));
        let mut worker_options = options.clone();
        worker_options.memory =
            (options.memory - reserved - (workers - 1) * classical_bytes) / workers;
        let prefix = if retain { Some(prefix) } else { None };
        let run_worker = |worker: usize| -> Result<_, String> {
            let mut local = HashMap::new();
            let mut stats = Stats::default();
            let mut state = if let Some(p) = &prefix {
                if p.bytes() > worker_options.memory {
                    return Err("MPS trajectory exceeds per-worker memory budget; reduce max_parallel_shots".into());
                }
                let mut copy = p.clone();
                copy.options = worker_options.clone();
                copy
            } else {
                Mps::new(n, worker_options.clone())?
            };
            for (iteration, shot) in (worker..shots).step_by(workers).enumerate() {
                if iteration > 0 {
                    if let Some(p) = &prefix {
                        state.restore(p);
                    } else {
                        state.reset_zero();
                    }
                }
                let mut cbits = HashMap::new();
                let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(shot as u64));
                let start = if retain { prefix_len } else { 0 };
                for op in &ops[start..] {
                    op.run(&mut state, &mut cbits, &mut rng)?;
                }
                *local.entry(format_cbits(&cbits, nc)).or_insert(0) += 1;
                merge_stats(&mut stats, &state.stats);
            }
            Ok((local, stats))
        };
        let batches = (0..workers)
            .into_par_iter()
            .map(run_worker)
            .collect::<Result<Vec<_>, String>>()?;
        for (local, s) in batches {
            for (k, v) in local {
                *counts.entry(k).or_insert(0) += v;
            }
            merge_stats(&mut stats, &s);
        }
        Ok((
            counts,
            stats,
            if retain {
                "prefix_trajectories"
            } else {
                "trajectories"
            },
            workers,
            if retain { prefix_len } else { 0 },
        ))
    }
}
