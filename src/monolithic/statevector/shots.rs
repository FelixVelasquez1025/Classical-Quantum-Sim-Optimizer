use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use num_complex::Complex64 as C;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use super::execution::{execute, Operation};
use crate::engine::marginal_probs;

pub(super) struct ShotOutput {
    pub counts: HashMap<String, usize>,
    pub shot_times: Vec<f64>,
    pub strategy: &'static str,
    pub parallel_shots: usize,
    /// Numerical state/CDF/classical worker buffers; excludes outputs and the compiled plan.
    pub working_bytes: usize,
    pub prefix_ops: usize,
}

pub(super) fn state_layout(n: usize) -> Result<(usize, usize), String> {
    let len = 1usize
        .checked_shl(n.try_into().map_err(|_| "Qubit count overflow")?)
        .ok_or("State dimension overflow")?;
    let bytes = len
        .checked_mul(std::mem::size_of::<C>())
        .filter(|&v| v <= isize::MAX as usize)
        .ok_or("State byte size overflow")?;
    Ok((len, bytes))
}

pub(super) fn zero_state(len: usize) -> Result<Vec<C>, String> {
    let mut state = Vec::new();
    state
        .try_reserve_exact(len)
        .map_err(|_| "Unable to allocate statevector")?;
    state.resize(len, C::default());
    state[0] = C::new(1.0, 0.0);
    Ok(state)
}

pub(super) fn check_budget(required: usize, budget: usize) -> Result<(), String> {
    if required > budget {
        return Err(format!("Simulation needs at least {required} working bytes, exceeding max_memory_mb budget ({budget} bytes)"));
    }
    Ok(())
}

fn format_bits(cbits: &[u8]) -> String {
    cbits
        .iter()
        .rev()
        .map(|&bit| char::from(b'0' + bit))
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_shots(
    ops: &[Operation],
    n: usize,
    num_cbits: usize,
    shots: usize,
    seed: u64,
    budget: usize,
    max_parallel_shots: Option<usize>,
    sample_terminal: bool,
    profile: bool,
) -> Result<ShotOutput, String> {
    let empty = |strategy| ShotOutput {
        counts: HashMap::new(),
        shot_times: vec![],
        strategy,
        parallel_shots: 0,
        working_bytes: 0,
        prefix_ops: 0,
    };
    if shots == 0 {
        return Ok(empty("empty"));
    }
    let (len, state_bytes) = state_layout(n)?;
    let worker_bytes = state_bytes
        .checked_add(num_cbits)
        .ok_or("Working memory size overflow")?;
    check_budget(worker_bytes, budget)?;
    let prefix_end = ops
        .iter()
        .position(|op| !op.is_unitary())
        .unwrap_or(ops.len());
    let terminal = ops[prefix_end..]
        .iter()
        .all(|op| matches!(op, Operation::Measure { .. }));
    if sample_terminal && terminal {
        // The final write to each classical bit wins. Repeated measurements of one
        // qubit reuse the same joint sample, preserving all quantum correlations.
        let mut mapping = BTreeMap::new();
        for op in &ops[prefix_end..] {
            if let Operation::Measure { qubit, cbit } = op {
                mapping.insert(*cbit, *qubit);
            }
        }
        if mapping.is_empty() {
            let mut out = empty("terminal_sampling");
            out.counts.insert("0".repeat(num_cbits), shots);
            return Ok(out);
        }
        let mut qubits: Vec<_> = mapping.values().copied().collect();
        qubits.sort_unstable();
        qubits.dedup();
        qubits.reverse(); // Full-register sampling takes the direct probability path.
        let mapping: Vec<_> = mapping
            .into_iter()
            .map(|(cbit, qubit)| {
                let shift = qubits.len() - 1 - qubits.iter().position(|&q| q == qubit).unwrap();
                (cbit, shift)
            })
            .collect();
        let cdf_bytes = (1usize << qubits.len())
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or("Sampling memory overflow")?;
        if let Some(required) = worker_bytes
            .checked_add(cdf_bytes)
            .filter(|&size| size <= budget)
        {
            let mut state = zero_state(len)?;
            let mut cbits = vec![0; num_cbits];
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            execute(
                &ops[..prefix_end],
                &mut state,
                n,
                &mut cbits,
                &mut rng,
                true,
                &mut None,
            );
            let mut cdf = marginal_probs(&state, n, &qubits);
            let mut total = 0.0;
            for p in &mut cdf {
                total += *p;
                *p = total;
            }
            if !total.is_finite() || total <= 0.0 {
                return Err("Invalid state normalization when sampling".into());
            }
            for p in &mut cdf {
                *p /= total;
            }
            *cdf.last_mut().unwrap() = 1.0;
            drop(state);
            let mut sampled: HashMap<usize, usize> = HashMap::new();
            // Seed each shot independently, so reproducibility does not depend on scheduling.
            for shot in 0..shots {
                let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(shot as u64));
                let draw: f64 = rng.gen();
                let idx = cdf.partition_point(|&p| p <= draw).min(cdf.len() - 1);
                *sampled.entry(idx).or_default() += 1;
            }
            let mut counts = HashMap::new();
            for (idx, count) in sampled {
                for &(cbit, shift) in &mapping {
                    cbits[cbit] = ((idx >> shift) & 1) as u8;
                }
                *counts.entry(format_bits(&cbits)).or_default() += count;
            }
            return Ok(ShotOutput {
                counts,
                shot_times: vec![],
                strategy: "terminal_sampling",
                parallel_shots: 1,
                working_bytes: required,
                prefix_ops: prefix_end,
            });
        }
        // Fall back to trajectories when a sampling table would exceed the budget.
    }

    // Cache a deterministic prefix only when both it and at least one worker fit.
    let cache_prefix = shots > 1 && prefix_end > 0 && state_bytes <= budget - worker_bytes;
    let prefix_bytes = if cache_prefix { state_bytes } else { 0 };
    let workers = shots
        .min(max_parallel_shots.unwrap_or(rayon::current_num_threads()))
        .min(rayon::current_num_threads())
        .min((budget - prefix_bytes) / worker_bytes)
        .max(1);
    let prefix = if cache_prefix {
        let mut state = zero_state(len)?;
        let mut cbits = vec![0; num_cbits];
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        execute(
            &ops[..prefix_end],
            &mut state,
            n,
            &mut cbits,
            &mut rng,
            true,
            &mut None,
        );
        Some(state)
    } else {
        None
    };
    let tail = if cache_prefix {
        &ops[prefix_end..]
    } else {
        ops
    };
    let results: Result<Vec<_>, String> = (0..workers)
        .into_par_iter()
        .map(|worker| {
            let mut state = zero_state(len)?;
            let mut cbits = vec![0; num_cbits];
            let mut counts = HashMap::new();
            let mut times = Vec::new();
            for shot in (worker..shots).step_by(workers) {
                let start = profile.then(Instant::now);
                if let Some(prefix) = &prefix {
                    state.copy_from_slice(prefix);
                } else {
                    state.fill(C::default());
                    state[0] = C::new(1.0, 0.0);
                }
                cbits.fill(0);
                let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(shot as u64));
                execute(
                    tail,
                    &mut state,
                    n,
                    &mut cbits,
                    &mut rng,
                    workers == 1,
                    &mut None,
                );
                *counts.entry(format_bits(&cbits)).or_insert(0usize) += 1;
                if let Some(start) = start {
                    times.push((shot, start.elapsed().as_secs_f64()));
                }
            }
            Ok((counts, times))
        })
        .collect();
    let mut counts = HashMap::new();
    let mut shot_times = if profile { vec![0.0; shots] } else { vec![] };
    for (local, times) in results? {
        for (bits, count) in local {
            *counts.entry(bits).or_default() += count;
        }
        for (shot, elapsed) in times {
            shot_times[shot] = elapsed;
        }
    }
    Ok(ShotOutput {
        counts,
        shot_times,
        strategy: "trajectories",
        parallel_shots: workers,
        working_bytes: prefix_bytes + workers * worker_bytes,
        prefix_ops: if cache_prefix { prefix_end } else { 0 },
    })
}

#[cfg(test)]
mod tests {
    use super::super::execution::compile;
    use super::*;
    use crate::types::Instruction;

    #[test]
    fn terminal_sampling_keeps_correlations_and_last_classical_write() {
        use Instruction::*;
        let ops = compile(&[
            H { qubit: 0 },
            Cx {
                control: 0,
                target: 1,
            },
            Measure { qubit: 0, cbit: 0 },
            Measure { qubit: 1, cbit: 1 },
            Measure { qubit: 1, cbit: 2 },
        ])
        .unwrap();
        let result = run_shots(&ops, 2, 3, 1024, 5, 1024, None, true, false).unwrap();
        assert_eq!(result.strategy, "terminal_sampling");
        assert_eq!(result.counts.values().sum::<usize>(), 1024);
        assert!(result.counts.keys().all(|k| k == "000" || k == "111"));
        assert_eq!(result.counts.len(), 2);
    }

    #[test]
    fn memory_budget_selects_one_worker_and_sampling_fallback() {
        let ops = compile(&[
            Instruction::H { qubit: 0 },
            Instruction::Measure { qubit: 0, cbit: 0 },
        ])
        .unwrap();
        let out = run_shots(&ops, 2, 1, 10, 5, 65, Some(8), true, false).unwrap();
        assert_eq!(out.strategy, "trajectories");
        assert_eq!(out.parallel_shots, 1);
        assert_eq!(out.working_bytes, 65);
        assert!(run_shots(&ops, 2, 1, 10, 5, 64, None, true, false).is_err());
    }

    #[test]
    fn overflowing_state_size_rejected_without_allocating() {
        assert!(state_layout(usize::BITS as usize).is_err());
        assert!(state_layout(usize::BITS as usize - 3).is_err());
    }
}
