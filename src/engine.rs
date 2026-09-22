use num_complex::Complex64;
use rand::Rng;
use rayon::prelude::*;

type C = Complex64;

/// Parallelize when the statevector has at least 2^PAR_THRESHOLD amplitudes.
const PAR_THRESHOLD: usize = 12;

/// Maximum gate arity (C4x = 5 qubits → dim = 32).
const MAX_DIM: usize = 32;

// ---------------------------------------------------------------------------
// apply_one_qubit
// Apply a 2×2 unitary to `target` qubit in-place.
// `controls` is a slice of (wire, must_be_one) pairs.
//
// Parallelism strategy: split state into blocks of `2*half` (each block holds
// exactly one (low, high) pair).  The outer par_chunks_mut handles many blocks
// (small target qubit); the inner par_iter_mut zip handles large blocks (large
// target qubit).  This keeps access contiguous at both levels.
// ---------------------------------------------------------------------------

pub fn apply_one_qubit(
    state: &mut [C],
    u: &[[C; 2]; 2],
    target: usize,
    n: usize,
    controls: &[(usize, bool)],
) {
    let inclusion_mask: usize = controls.iter().fold(0, |m, &(w, _)| m | (1 << w));
    let desired_mask: usize =
        controls.iter().fold(0, |m, &(w, flag)| if flag { m | (1 << w) } else { m });
    let half = 1 << target;
    let block = 2 * half;
    let dim = 1 << n;

    let u00 = u[0][0];
    let u01 = u[0][1];
    let u10 = u[1][0];
    let u11 = u[1][1];

    if n >= PAR_THRESHOLD {
        // Split state into contiguous blocks of size `block = 2*half`.
        // Each block's lower half maps to gate input 0, upper half to input 1.
        // Outer par_chunks_mut gives parallelism for small target qubits (many blocks).
        // Inner par_iter_mut zip gives parallelism for large target qubits (large blocks).
        state.par_chunks_mut(block).enumerate().for_each(|(ci, chunk)| {
            let base_i = ci * block;
            let (lo, hi) = chunk.split_at_mut(half);

            if lo.len() >= 1024 {
                // Large block — too few outer chunks for good parallelism, use inner.
                if inclusion_mask == 0 {
                    lo.par_iter_mut().zip(hi.par_iter_mut()).for_each(|(a_r, b_r)| {
                        let a = *a_r;
                        let b = *b_r;
                        *a_r = u00 * a + u01 * b;
                        *b_r = u10 * a + u11 * b;
                    });
                } else {
                    lo.par_iter_mut()
                        .zip(hi.par_iter_mut())
                        .enumerate()
                        .for_each(|(k, (a_r, b_r))| {
                            let i = base_i + k;
                            if (i & inclusion_mask) == desired_mask {
                                let a = *a_r;
                                let b = *b_r;
                                *a_r = u00 * a + u01 * b;
                                *b_r = u10 * a + u11 * b;
                            }
                        });
                }
            } else {
                // Small block — outer loop provides enough parallelism; sequential inner.
                for (k, (a_r, b_r)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                    let i = base_i + k;
                    if inclusion_mask == 0 || (i & inclusion_mask) == desired_mask {
                        let a = *a_r;
                        let b = *b_r;
                        *a_r = u00 * a + u01 * b;
                        *b_r = u10 * a + u11 * b;
                    }
                }
            }
        });
    } else {
        let mut i = 0;
        while i < dim {
            if (i & half) == 0 && (inclusion_mask == 0 || (i & inclusion_mask) == desired_mask) {
                let j = i | half;
                let a = state[i];
                let b = state[j];
                state[i] = u00 * a + u01 * b;
                state[j] = u10 * a + u11 * b;
            }
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// apply_n_qubit
// Apply a 2^k × 2^k unitary to k arbitrary qubits in-place.
// qubits[0] = MSB of the gate's index space, qubits[k-1] = LSB.
// ---------------------------------------------------------------------------

pub fn apply_n_qubit(state: &mut [C], u: &[Vec<C>], qubits: &[usize], n: usize) {
    let k = qubits.len();
    let dim = 1 << k;
    let n_states = 1 << n;
    let mask: usize = qubits.iter().fold(0, |m, &q| m | (1 << q));

    // Pre-compute target-bit offsets once — the same for every base index.
    let mut offsets = [0usize; MAX_DIM];
    for (j, offset_slot) in offsets.iter_mut().enumerate().take(dim) {
        let mut offset = 0usize;
        for (bit, &q) in qubits.iter().enumerate() {
            if (j >> (k - 1 - bit)) & 1 == 1 {
                offset |= 1 << q;
            }
        }
        *offset_slot = offset;
    }

    if n >= PAR_THRESHOLD {
        // Safety: different base values produce fully disjoint index sets.
        // Each base has all target-qubit bits = 0; offsets fill in all 2^k
        // combinations of those bits. Two distinct bases differ in a non-target
        // bit, so their index sets cannot overlap.
        let ptr = state.as_mut_ptr() as usize;
        (0..n_states)
            .into_par_iter()
            .filter(|&i| (i & mask) == 0)
            .for_each(|base| {
                let p = ptr as *mut C;
                let mut v = [C::new(0.0, 0.0); MAX_DIM];
                let mut w = [C::new(0.0, 0.0); MAX_DIM];
                let mut idx = [0usize; MAX_DIM];

                for j in 0..dim {
                    idx[j] = base + offsets[j];
                    v[j] = unsafe { *p.add(idx[j]) };
                }
                for row in 0..dim {
                    let mut acc = C::new(0.0, 0.0);
                    for col in 0..dim {
                        acc += u[row][col] * v[col];
                    }
                    w[row] = acc;
                }
                for j in 0..dim {
                    unsafe { *p.add(idx[j]) = w[j] };
                }
            });
    } else {
        let mut v = vec![C::new(0.0, 0.0); dim];
        let mut w = vec![C::new(0.0, 0.0); dim];
        let mut idx = vec![0usize; dim];

        for base in (0..n_states).filter(|&i| (i & mask) == 0) {
            for j in 0..dim {
                idx[j] = base + offsets[j];
            }
            for j in 0..dim {
                v[j] = state[idx[j]];
            }
            for row in 0..dim {
                let mut acc = C::new(0.0, 0.0);
                for col in 0..dim {
                    acc += u[row][col] * v[col];
                }
                w[row] = acc;
            }
            for j in 0..dim {
                state[idx[j]] = w[j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// measure_qubit
// Collapse the statevector on `qubit`, renormalise, return outcome (0 or 1).
//
// Parallelism strategy: elements with (i & bit) != 0 form contiguous runs of
// length `bit` interleaved with runs of equal length where the bit is clear.
// par_chunks over blocks of `2*bit` lets us sum/zero/scale each run without
// any scatter/filter overhead and with fully contiguous memory access.
// ---------------------------------------------------------------------------

pub fn measure_qubit<R: Rng>(state: &mut [C], qubit: usize, n: usize, rng: &mut R) -> u8 {
    let n_states = 1 << n;
    let bit = 1 << qubit;

    let p1: f64 = if n >= PAR_THRESHOLD {
        // Each block of 2*bit elements has its upper half with the bit set.
        state
            .par_chunks(2 * bit)
            .map(|chunk| chunk[bit..].iter().map(|c| c.norm_sqr()).sum::<f64>())
            .sum()
    } else {
        (0..n_states)
            .filter(|&i| (i & bit) != 0)
            .map(|i| state[i].norm_sqr())
            .sum()
    };

    let outcome = if rng.gen::<f64>() < p1 { 1u8 } else { 0u8 };

    if n >= PAR_THRESHOLD {
        if outcome == 1 {
            let norm = p1.sqrt().max(1e-15);
            state.par_chunks_mut(2 * bit).for_each(|chunk| {
                let (lo, hi) = chunk.split_at_mut(bit);
                for c in lo {
                    *c = C::new(0.0, 0.0);
                }
                for c in hi {
                    *c /= norm;
                }
            });
        } else {
            let norm = (1.0 - p1).max(0.0).sqrt().max(1e-15);
            state.par_chunks_mut(2 * bit).for_each(|chunk| {
                let (lo, hi) = chunk.split_at_mut(bit);
                for c in hi {
                    *c = C::new(0.0, 0.0);
                }
                for c in lo {
                    *c /= norm;
                }
            });
        }
    } else {
        if outcome == 1 {
            let norm = p1.sqrt().max(1e-15);
            for i in (0..n_states).filter(|&i| (i & bit) == 0) {
                state[i] = C::new(0.0, 0.0);
            }
            for i in (0..n_states).filter(|&i| (i & bit) != 0) {
                state[i] /= norm;
            }
        } else {
            let norm = (1.0 - p1).max(0.0).sqrt().max(1e-15);
            for i in (0..n_states).filter(|&i| (i & bit) != 0) {
                state[i] = C::new(0.0, 0.0);
            }
            for i in (0..n_states).filter(|&i| (i & bit) == 0) {
                state[i] /= norm;
            }
        }
    }

    outcome
}

// ---------------------------------------------------------------------------
// Sequential variants — always single-threaded, for use inside parallel shot loops
// to avoid nested Rayon thread pool contention.
// ---------------------------------------------------------------------------

/// Sequential apply_one_qubit — always single-threaded, for use inside parallel shot loops.
pub fn apply_one_qubit_seq(
    state: &mut [C],
    u: &[[C; 2]; 2],
    target: usize,
    n: usize,
    controls: &[(usize, bool)],
) {
    let inclusion_mask: usize = controls.iter().fold(0, |m, &(w, _)| m | (1 << w));
    let desired_mask: usize =
        controls.iter().fold(0, |m, &(w, flag)| if flag { m | (1 << w) } else { m });
    let half = 1 << target;
    let dim = 1 << n;
    let u00 = u[0][0];
    let u01 = u[0][1];
    let u10 = u[1][0];
    let u11 = u[1][1];
    let mut i = 0;
    while i < dim {
        if (i & half) == 0 && (inclusion_mask == 0 || (i & inclusion_mask) == desired_mask) {
            let j = i | half;
            let a = state[i];
            let b = state[j];
            state[i] = u00 * a + u01 * b;
            state[j] = u10 * a + u11 * b;
        }
        i += 1;
    }
}

/// Sequential apply_n_qubit — always single-threaded.
pub fn apply_n_qubit_seq(state: &mut [C], u: &[Vec<C>], qubits: &[usize], n: usize) {
    let k = qubits.len();
    let dim = 1 << k;
    let n_states = 1 << n;
    let mask: usize = qubits.iter().fold(0, |m, &q| m | (1 << q));
    let mut offsets = [0usize; MAX_DIM];
    for (j, offset_slot) in offsets.iter_mut().enumerate().take(dim) {
        let mut offset = 0usize;
        for (bit, &q) in qubits.iter().enumerate() {
            if (j >> (k - 1 - bit)) & 1 == 1 {
                offset |= 1 << q;
            }
        }
        *offset_slot = offset;
    }
    let mut v = vec![C::new(0.0, 0.0); dim];
    let mut w = vec![C::new(0.0, 0.0); dim];
    let mut idx = vec![0usize; dim];
    for base in (0..n_states).filter(|&i| (i & mask) == 0) {
        for j in 0..dim {
            idx[j] = base + offsets[j];
        }
        for j in 0..dim {
            v[j] = state[idx[j]];
        }
        for row in 0..dim {
            let mut acc = C::new(0.0, 0.0);
            for col in 0..dim {
                acc += u[row][col] * v[col];
            }
            w[row] = acc;
        }
        for j in 0..dim {
            state[idx[j]] = w[j];
        }
    }
}

/// Sequential measure_qubit — always single-threaded.
pub fn measure_qubit_seq<R: Rng>(state: &mut [C], qubit: usize, n: usize, rng: &mut R) -> u8 {
    let n_states = 1 << n;
    let bit = 1 << qubit;
    let p1: f64 = (0..n_states)
        .filter(|&i| (i & bit) != 0)
        .map(|i| state[i].norm_sqr())
        .sum();
    let outcome = if rng.gen::<f64>() < p1 { 1u8 } else { 0u8 };
    if outcome == 1 {
        let norm = p1.sqrt().max(1e-15);
        for i in (0..n_states).filter(|&i| (i & bit) == 0) {
            state[i] = C::new(0.0, 0.0);
        }
        for i in (0..n_states).filter(|&i| (i & bit) != 0) {
            state[i] /= norm;
        }
    } else {
        let norm = (1.0 - p1).max(0.0).sqrt().max(1e-15);
        for i in (0..n_states).filter(|&i| (i & bit) != 0) {
            state[i] = C::new(0.0, 0.0);
        }
        for i in (0..n_states).filter(|&i| (i & bit) == 0) {
            state[i] /= norm;
        }
    }
    outcome
}

// ---------------------------------------------------------------------------
// marginal_probs
// Returns a Vec of length 2^k giving probabilities for each basis state of
// the specified qubits. qubits[0] = MSB of output index.
// ---------------------------------------------------------------------------

pub fn marginal_probs(state: &[C], n: usize, qubits: &[usize]) -> Vec<f64> {
    let k = qubits.len();
    let dim = 1 << k;
    let n_states = 1 << n;

    let mask: usize = qubits.iter().fold(0, |m, &q| m | (1 << q));
    let mut probs = vec![0.0f64; dim];

    for (j, prob) in probs.iter_mut().enumerate() {
        let mut desired = 0usize;
        for (bit, &q) in qubits.iter().enumerate() {
            if (j >> (k - 1 - bit)) & 1 == 1 {
                desired |= 1 << q;
            }
        }
        *prob = (0..n_states)
            .filter(|&i| (i & mask) == desired)
            .map(|i| state[i].norm_sqr())
            .sum();
    }

    probs
}

// ---------------------------------------------------------------------------
// sample_counts
// Sample `shots` outcomes from the state distribution.
// Returns a HashMap of (bitstring, count) pairs.
// ---------------------------------------------------------------------------

pub fn sample_counts<R: Rng>(
    state: &[C],
    n: usize,
    shots: usize,
    rng: &mut R,
    qubits: Option<&[usize]>,
) -> std::collections::HashMap<String, usize> {
    let all_qubits: Vec<usize>;
    let q = match qubits {
        Some(qs) => qs,
        None => {
            all_qubits = (0..n).rev().collect();
            &all_qubits
        }
    };

    let k = q.len();
    let probs = marginal_probs(state, n, q);

    let mut cdf = vec![0.0f64; probs.len()];
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        cdf[i] = acc;
    }
    *cdf.last_mut().unwrap() = 1.0;

    let mut counts = std::collections::HashMap::new();
    for _ in 0..shots {
        let r: f64 = rng.gen();
        let idx = cdf.partition_point(|&c| c < r).min((1 << k) - 1);
        let bits = format!("{:0>width$b}", idx, width = k);
        *counts.entry(bits).or_insert(0) += 1;
    }
    counts
}
