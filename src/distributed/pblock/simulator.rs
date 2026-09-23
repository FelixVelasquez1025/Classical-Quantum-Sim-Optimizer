use super::{
    execution::{compile, Op},
    input::prepare,
    model::{bytes, sample, BlockPool, Options, Stats},
    result::{error, PBlockResult},
};
use crate::{profiling::write_shots_profile, types::format_cbits};
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::Serialize;
use std::{collections::HashMap, time::Instant};

#[pyclass]
pub struct PBlockSimulator {
    seed: Option<u64>,
    options: Options,
    max_parallel_shots: usize,
    sample_terminal: bool,
}
#[derive(Serialize)]
struct ShotsProfile {
    preprocessing_time: f64,
    fusion_time: f64,
    shots_total_time: f64,
    total_time: f64,
    num_shots: usize,
    execution_strategy: String,
    parallel_shots: usize,
    deterministic_prefix_ops: usize,
    memory_budget_bytes: usize,
    #[serde(flatten)]
    stats: Stats,
}
type ShotOutput = (HashMap<String, usize>, Stats, &'static str, usize, usize);

#[pymethods]
impl PBlockSimulator {
    #[new]
    #[pyo3(signature=(seed=None,*,max_memory_mb=1024,max_block_qubits=None,max_parallel_shots=1,sample_terminal=true,split_separable=false,max_split_qubits=12,separation_tolerance=0.0,profile=false))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        seed: Option<u64>,
        max_memory_mb: usize,
        max_block_qubits: Option<usize>,
        max_parallel_shots: usize,
        sample_terminal: bool,
        split_separable: bool,
        max_split_qubits: usize,
        separation_tolerance: f64,
        profile: bool,
    ) -> PyResult<Self> {
        let memory = max_memory_mb
            .checked_mul(1024 * 1024)
            .filter(|&v| v > 0 && v <= isize::MAX as usize)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "max_memory_mb must be positive and representable",
                )
            })?;
        if max_block_qubits == Some(0)
            || max_block_qubits.is_some_and(|n| n >= usize::BITS as usize)
        {
            return Err(pyo3::exceptions::PyValueError::new_err("max_block_qubits must be positive and smaller than the machine word width, or None"));
        }
        if max_parallel_shots == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_parallel_shots must be positive",
            ));
        }
        if max_split_qubits == 0 || max_split_qubits >= usize::BITS as usize {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_split_qubits must be positive and smaller than the machine word width",
            ));
        }
        if !separation_tolerance.is_finite() || !(0.0..1.0).contains(&separation_tolerance) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "separation_tolerance must be finite and in [0,1)",
            ));
        }
        Ok(Self {
            seed,
            options: Options {
                memory,
                max_block_qubits,
                split_separable,
                max_split_qubits,
                separation_tolerance,
                profile,
            },
            max_parallel_shots,
            sample_terminal,
        })
    }
    pub fn simulate(&self, py: Python, distributed: &Bound<PyAny>) -> PyResult<PBlockResult> {
        let start = Instant::now();
        let input = prepare(distributed, self.options.memory)?;
        let ops = compile(&input.instructions).map_err(pyo3::exceptions::PyValueError::new_err)?;
        let options = self.working_options(input.num_cbits, self.options.profile)?;
        let seed = self.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let (pool, cbits) = py
            .allow_threads(|| -> Result<_, String> {
                let mut pool = BlockPool::new(input.physical.len(), options)?;
                let mut cbits = HashMap::new();
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                for op in &ops {
                    op.run(&mut pool, &mut cbits, &mut rng, true)?;
                }
                Ok((pool, cbits))
            })
            .map_err(error)?;
        Ok(PBlockResult {
            pool,
            physical: input.physical,
            cbits,
            elapsed: start.elapsed().as_secs_f64(),
        })
    }
    #[pyo3(signature=(distributed,shots=1000,profile=false))]
    pub fn simulate_shots(
        &self,
        py: Python,
        distributed: &Bound<PyAny>,
        shots: usize,
        profile: bool,
    ) -> PyResult<HashMap<String, usize>> {
        let start = Instant::now();
        let input = prepare(distributed, self.options.memory)?;
        let preprocessing_time = start.elapsed().as_secs_f64();
        let fusion_start = Instant::now();
        let ops = compile(&input.instructions).map_err(pyo3::exceptions::PyValueError::new_err)?;
        let fusion_time = fusion_start.elapsed().as_secs_f64();
        let options = self.working_options(input.num_cbits, profile)?;
        let seed = self.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let exec_start = Instant::now();
        let (counts, stats, strategy, workers, prefix) = py
            .allow_threads(|| {
                self.shots(
                    &ops,
                    input.physical.len(),
                    input.num_cbits,
                    shots,
                    seed,
                    options,
                )
            })
            .map_err(error)?;
        if profile {
            let report = ShotsProfile {
                preprocessing_time,
                fusion_time,
                shots_total_time: exec_start.elapsed().as_secs_f64(),
                total_time: start.elapsed().as_secs_f64(),
                num_shots: shots,
                execution_strategy: strategy.into(),
                parallel_shots: workers,
                deterministic_prefix_ops: prefix,
                memory_budget_bytes: self.options.memory,
                stats,
            };
            write_shots_profile("pblock", &report).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to write P-block profile: {e}"
                ))
            })?;
        }
        Ok(counts)
    }
}
impl PBlockSimulator {
    fn working_options(&self, nc: usize, profile: bool) -> PyResult<Options> {
        let classical = bytes(nc, 32).map_err(error)?;
        if classical >= self.options.memory {
            return Err(error(
                "P-block classical workspace exceeds max_memory_mb".into(),
            ));
        }
        let mut options = self.options.clone();
        options.memory -= classical;
        options.profile = profile;
        Ok(options)
    }
    fn shots(
        &self,
        ops: &[Op],
        n: usize,
        nc: usize,
        shots: usize,
        seed: u64,
        options: Options,
    ) -> Result<ShotOutput, String> {
        let mut counts = HashMap::new();
        let mut stats = Stats::default();
        if shots == 0 {
            return Ok((counts, stats, "empty", 0, 0));
        }
        let prefix_len = ops.iter().take_while(|op| op.unitary()).count();
        let mut prefix = BlockPool::new(n, options.clone())?;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut cbits = HashMap::new();
        for op in &ops[..prefix_len] {
            op.run(&mut prefix, &mut cbits, &mut rng, true)?;
        }
        if self.sample_terminal && ops[prefix_len..].iter().all(Op::terminal) {
            if prefix_len == ops.len() {
                counts.insert("0".repeat(nc), shots);
                return Ok((counts, prefix.stats, "terminal_sampling", 1, prefix_len));
            }
            match prefix.sampling_tables() {
                Ok(tables) => {
                    prefix.stats.peak_working_bytes = prefix.stats.peak_working_bytes.max(
                        prefix.bytes()
                            + tables.iter().map(|(_, v)| v.len() * 8).sum::<usize>()
                            + n * 64,
                    );
                    let mut bits = vec![0; n];
                    for shot in 0..shots {
                        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(shot as u64));
                        sample(&tables, &mut bits, &mut rng);
                        cbits.clear();
                        for op in &ops[prefix_len..] {
                            if let Op::Measure(q, c) = op {
                                cbits.insert(*c, bits[*q] as i32);
                            }
                        }
                        *counts.entry(format_cbits(&cbits, nc)).or_insert(0) += 1;
                    }
                    return Ok((counts, prefix.stats, "terminal_sampling", 1, prefix_len));
                }
                Err(e) if e.contains("memory") => {}
                Err(e) => return Err(e),
            }
        }
        stats.add(&prefix.stats);
        let retain = prefix_len > 0
            && prefix.bytes().saturating_mul(2) < options.memory
            && prefix
                .bytes()
                .saturating_add(prefix.stats.peak_working_bytes)
                <= options.memory;
        let reserved = if retain { prefix.bytes() } else { 0 };
        let classical = bytes(nc, 32)?;
        let per_worker = prefix
            .stats
            .peak_working_bytes
            .max(prefix.bytes())
            .saturating_add(classical)
            .max(4096);
        let workers = self
            .max_parallel_shots
            .min(shots)
            .min(rayon::current_num_threads())
            .min(((options.memory - reserved) / per_worker).max(1));
        let mut worker_options = options;
        worker_options.memory =
            (worker_options.memory - reserved - (workers - 1) * classical) / workers;
        prefix.stats = Stats::default();
        let prefix = if retain {
            Some(prefix)
        } else {
            drop(prefix);
            None
        };
        let run_worker = |worker| -> Result<_, String> {
            let mut local = HashMap::new();
            let mut local_stats = Stats::default();
            let mut pool = if let Some(p) = &prefix {
                if p.bytes() > worker_options.memory {
                    return Err(
                        "P-block prefix exceeds worker memory budget; reduce max_parallel_shots"
                            .into(),
                    );
                }
                let mut copy = p.clone();
                copy.options = worker_options.clone();
                copy
            } else {
                BlockPool::new(n, worker_options.clone())?
            };
            let mut cbits = HashMap::new();
            for (iteration, shot) in (worker..shots).step_by(workers).enumerate() {
                if iteration > 0 {
                    if let Some(p) = &prefix {
                        pool.restore(p);
                    } else {
                        pool.reset_zero();
                    }
                }
                cbits.clear();
                let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(shot as u64));
                for op in &ops[if retain { prefix_len } else { 0 }..] {
                    op.run(&mut pool, &mut cbits, &mut rng, workers == 1)?;
                }
                *local.entry(format_cbits(&cbits, nc)).or_insert(0) += 1;
                local_stats.add(&pool.stats);
            }
            Ok((local, local_stats))
        };
        let batches = (0..workers)
            .into_par_iter()
            .map(run_worker)
            .collect::<Result<Vec<_>, String>>()?;
        for (local, s) in batches {
            for (k, v) in local {
                *counts.entry(k).or_insert(0) += v;
            }
            stats.add(&s);
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
