use num_complex::Complex64;
use rand::Rng;
use rayon::prelude::*;

type C = Complex64;

/// Keep small vectors within one worker, especially inside parallel shot loops.
const PAR_THRESHOLD: usize = 12;
const PAR_CHUNK: usize = 1024;
/// Largest dense gate supported by the instruction set (C4X).
const MAX_QUBITS: usize = 5;
const MAX_DIM: usize = 1 << MAX_QUBITS;

fn state_qubits(state: &[C]) -> usize {
    assert!(
        state.len().is_power_of_two(),
        "State length must be a nonzero power of two"
    );
    state.len().trailing_zeros() as usize
}

fn validate_state(state: &[C], n: usize) {
    assert!(
        n < usize::BITS as usize,
        "Qubit count exceeds addressable state size"
    );
    assert_eq!(
        state.len(),
        1usize << n,
        "State length does not match qubit count"
    );
}

fn qubit_mask(qubits: &[usize], n: usize) -> usize {
    let mut mask = 0;
    for &q in qubits {
        assert!(q < n, "Qubit index is out of range");
        let bit = 1usize << q;
        assert_eq!(mask & bit, 0, "Qubit indices must be distinct");
        mask |= bit;
    }
    mask
}

fn control_masks(controls: &[(usize, bool)], targets: usize, n: usize) -> (usize, usize) {
    let mut mask = 0;
    let mut desired = 0;
    for &(q, value) in controls {
        assert!(q < n, "Control index is out of range");
        let bit = 1usize << q;
        assert_eq!(
            (mask | targets) & bit,
            0,
            "Controls must be distinct from each other and targets"
        );
        mask |= bit;
        if value {
            desired |= bit;
        }
    }
    (mask, desired)
}

/// Iterate each target-zero/target-one pair once, with disjoint mutable slices.
fn for_each_pair<F>(
    state: &mut [C],
    target: usize,
    controls: &[(usize, bool)],
    parallel: bool,
    update: F,
) where
    F: Fn(&mut C, &mut C) + Sync + Send,
{
    let n = state_qubits(state);
    let target_mask = qubit_mask(&[target], n);
    let (mask, desired) = control_masks(controls, target_mask, n);
    let half = target_mask;
    let block = 2 * half;
    let parallel = parallel && n >= PAR_THRESHOLD;
    let process = |ci: usize, chunk: &mut [C]| {
        let base = ci * block;
        let (lo, hi) = chunk.split_at_mut(half);
        if parallel && half >= PAR_CHUNK {
            lo.par_iter_mut()
                .zip(hi.par_iter_mut())
                .enumerate()
                .with_min_len(PAR_CHUNK)
                .for_each(|(offset, (a, b))| {
                    if ((base + offset) & mask) == desired {
                        update(a, b);
                    }
                });
        } else {
            for (offset, (a, b)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                if ((base + offset) & mask) == desired {
                    update(a, b);
                }
            }
        }
    };
    if parallel {
        state
            .par_chunks_mut(block)
            .enumerate()
            .for_each(|(ci, chunk)| process(ci, chunk));
    } else {
        for (ci, chunk) in state.chunks_mut(block).enumerate() {
            process(ci, chunk);
        }
    }
}

fn apply_one_qubit_impl(
    state: &mut [C],
    u: &[[C; 2]; 2],
    target: usize,
    n: usize,
    controls: &[(usize, bool)],
    parallel: bool,
) {
    validate_state(state, n);
    for_each_pair(state, target, controls, parallel, |a, b| {
        let old_a = *a;
        let old_b = *b;
        *a = u[0][0] * old_a + u[0][1] * old_b;
        *b = u[1][0] * old_a + u[1][1] * old_b;
    });
}

/// Apply a 2x2 gate, optionally conditioned on any number of other qubits.
pub fn apply_one_qubit(
    state: &mut [C],
    u: &[[C; 2]; 2],
    target: usize,
    n: usize,
    controls: &[(usize, bool)],
) {
    apply_one_qubit_impl(state, u, target, n, controls, true);
}

/// Single-threaded variant for use within parallel shot loops.
pub fn apply_one_qubit_seq(
    state: &mut [C],
    u: &[[C; 2]; 2],
    target: usize,
    n: usize,
    controls: &[(usize, bool)],
) {
    apply_one_qubit_impl(state, u, target, n, controls, false);
}

/// X and controlled X are permutations, requiring no complex arithmetic.
pub fn apply_x(state: &mut [C], target: usize, controls: &[(usize, bool)], parallel: bool) {
    for_each_pair(state, target, controls, parallel, std::mem::swap);
}

/// Apply diagonal entries in qubits[0]-as-MSB order.
pub fn apply_diagonal(state: &mut [C], qubits: &[usize], diagonal: &[C], parallel: bool) {
    let n = state_qubits(state);
    qubit_mask(qubits, n);
    assert_eq!(
        diagonal.len(),
        1usize << qubits.len(),
        "Diagonal size does not match gate arity"
    );
    let update = |i: usize, amplitude: &mut C| {
        let j = qubits.iter().fold(0, |j, &q| (j << 1) | ((i >> q) & 1));
        *amplitude *= diagonal[j];
    };
    if parallel && n >= PAR_THRESHOLD {
        state
            .par_iter_mut()
            .enumerate()
            .with_min_len(PAR_CHUNK)
            .for_each(|(i, a)| update(i, a));
    } else {
        for (i, amplitude) in state.iter_mut().enumerate() {
            update(i, amplitude);
        }
    }
}

/// SWAP and controlled SWAP, using disjoint slices even for distant targets.
pub fn apply_swap(state: &mut [C], a: usize, b: usize, controls: &[(usize, bool)], parallel: bool) {
    let n = state_qubits(state);
    let targets = qubit_mask(&[a, b], n);
    let (mask, desired) = control_masks(controls, targets, n);
    let low_bit = 1usize << a.min(b);
    let high_bit = 1usize << a.max(b);
    let parallel = parallel && n >= PAR_THRESHOLD;
    let process_block = |ci: usize, chunk: &mut [C]| {
        let base = ci * 2 * high_bit;
        let (lo, hi) = chunk.split_at_mut(high_bit);
        let process_pair = |pi: usize, low: &mut [C], high: &mut [C]| {
            let first_index = base + pi * 2 * low_bit + low_bit;
            let left = &mut low[low_bit..];
            let right = &mut high[..low_bit];
            if parallel && low_bit >= PAR_CHUNK {
                left.par_iter_mut()
                    .zip(right.par_iter_mut())
                    .enumerate()
                    .with_min_len(PAR_CHUNK)
                    .for_each(|(offset, (x, y))| {
                        if ((first_index + offset) & mask) == desired {
                            std::mem::swap(x, y);
                        }
                    });
            } else {
                for (offset, (x, y)) in left.iter_mut().zip(right.iter_mut()).enumerate() {
                    if ((first_index + offset) & mask) == desired {
                        std::mem::swap(x, y);
                    }
                }
            }
        };
        if parallel {
            lo.par_chunks_mut(2 * low_bit)
                .zip(hi.par_chunks_mut(2 * low_bit))
                .enumerate()
                .for_each(|(pi, (low, high))| process_pair(pi, low, high));
        } else {
            for (pi, (low, high)) in lo
                .chunks_mut(2 * low_bit)
                .zip(hi.chunks_mut(2 * low_bit))
                .enumerate()
            {
                process_pair(pi, low, high);
            }
        }
    };
    if parallel {
        state
            .par_chunks_mut(2 * high_bit)
            .enumerate()
            .for_each(|(ci, chunk)| process_block(ci, chunk));
    } else {
        for (ci, chunk) in state.chunks_mut(2 * high_bit).enumerate() {
            process_block(ci, chunk);
        }
    }
}

/// Expand a compact base index by inserting zeros at the sorted gate targets.
#[inline]
fn insert_target_zeros(mut base: usize, sorted_targets: &[usize]) -> usize {
    for &q in sorted_targets {
        let low = (1usize << q) - 1;
        base = (base & low) | ((base & !low) << 1);
    }
    base
}

/// Gate-local index layout, reusable across every shot of a compiled circuit.
pub(crate) struct DenseLayout {
    qubits: [usize; MAX_QUBITS],
    offsets: [usize; MAX_DIM],
    sorted_targets: [usize; MAX_QUBITS],
    arity: usize,
}

impl DenseLayout {
    pub(crate) fn new(qubits: &[usize]) -> Self {
        let k = qubits.len();
        assert!(
            k <= MAX_QUBITS,
            "Dense gate arity exceeds supported maximum"
        );
        qubit_mask(qubits, usize::BITS as usize);
        let mut layout = Self {
            qubits: [0; MAX_QUBITS],
            offsets: [0; MAX_DIM],
            sorted_targets: [0; MAX_QUBITS],
            arity: k,
        };
        layout.qubits[..k].copy_from_slice(qubits);
        layout.sorted_targets[..k].copy_from_slice(qubits);
        layout.sorted_targets[..k].sort_unstable();
        for (j, offset) in layout.offsets[..1 << k].iter_mut().enumerate() {
            for (bit, &q) in qubits.iter().enumerate() {
                *offset |= ((j >> (k - 1 - bit)) & 1) << q;
            }
        }
        layout
    }

    pub(crate) fn apply(&self, state: &mut [C], u: &[Vec<C>], n: usize, parallel: bool) {
        // Runtime validation is retained at every unsafe scatter entry point.
        validate_state(state, n);
        let k = self.arity;
        qubit_mask(&self.qubits[..k], n);
        let dim = 1usize << k;
        assert_eq!(u.len(), dim, "Matrix row count does not match gate arity");
        assert!(
            u.iter().all(|row| row.len() == dim),
            "Matrix column count does not match gate arity"
        );
        let offsets = &self.offsets;
        let targets = &self.sorted_targets;
        let num_bases = state.len() >> k;

        if parallel && n >= PAR_THRESHOLD {
            let ptr = state.as_mut_ptr() as usize;
            (0..num_bases).into_par_iter().for_each(|compact| {
                let base = insert_target_zeros(compact, &targets[..k]);
                let p = ptr as *mut C;
                let mut values = [C::new(0.0, 0.0); MAX_DIM];
                let mut result = [C::new(0.0, 0.0); MAX_DIM];
                // Safety: validation guarantees distinct in-range targets, exact state
                // length and matrix dimensions. Each compact index expands into a
                // unique base with zero target bits. Its offsets enumerate all target
                // combinations exactly once, so workers access disjoint valid indices.
                for j in 0..dim {
                    values[j] = unsafe { *p.add(base | offsets[j]) };
                }
                for row in 0..dim {
                    for col in 0..dim {
                        result[row] += u[row][col] * values[col];
                    }
                }
                for j in 0..dim {
                    unsafe { *p.add(base | offsets[j]) = result[j] };
                }
            });
        } else {
            let mut values = [C::new(0.0, 0.0); MAX_DIM];
            let mut result = [C::new(0.0, 0.0); MAX_DIM];
            for compact in 0..num_bases {
                let base = insert_target_zeros(compact, &targets[..k]);
                for j in 0..dim {
                    values[j] = state[base | offsets[j]];
                }
                for row in 0..dim {
                    result[row] = C::new(0.0, 0.0);
                    for col in 0..dim {
                        result[row] += u[row][col] * values[col];
                    }
                }
                for j in 0..dim {
                    state[base | offsets[j]] = result[j];
                }
            }
        }
    }
}

fn apply_n_qubit_impl(state: &mut [C], u: &[Vec<C>], qubits: &[usize], n: usize, parallel: bool) {
    DenseLayout::new(qubits).apply(state, u, n, parallel);
}

/// Apply a dense gate with qubits[0] as the MSB of its matrix index.
pub fn apply_n_qubit(state: &mut [C], u: &[Vec<C>], qubits: &[usize], n: usize) {
    apply_n_qubit_impl(state, u, qubits, n, true);
}

/// Single-threaded dense application without per-gate scratch allocations.
pub fn apply_n_qubit_seq(state: &mut [C], u: &[Vec<C>], qubits: &[usize], n: usize) {
    apply_n_qubit_impl(state, u, qubits, n, false);
}

fn measure_qubit_impl<R: Rng>(
    state: &mut [C],
    qubit: usize,
    n: usize,
    rng: &mut R,
    parallel: bool,
) -> u8 {
    validate_state(state, n);
    let bit = qubit_mask(&[qubit], n);
    let parallel = parallel && n >= PAR_THRESHOLD;
    // Compute both probabilities directly: 1 - p1 loses the small branch's
    // precision and assumes perfect normalization after every previous gate.
    let probs = if parallel {
        state
            .par_chunks(PAR_CHUNK)
            .enumerate()
            .map(|(ci, chunk)| {
                let mut sums = [0.0, 0.0];
                for (offset, amplitude) in chunk.iter().enumerate() {
                    sums[usize::from(((ci * PAR_CHUNK + offset) & bit) != 0)] +=
                        amplitude.norm_sqr();
                }
                sums
            })
            .reduce(|| [0.0, 0.0], |a, b| [a[0] + b[0], a[1] + b[1]])
    } else {
        let mut sums = [0.0, 0.0];
        for chunk in state.chunks(2 * bit) {
            sums[0] += chunk[..bit].iter().map(|a| a.norm_sqr()).sum::<f64>();
            sums[1] += chunk[bit..].iter().map(|a| a.norm_sqr()).sum::<f64>();
        }
        sums
    };
    let total = probs[0] + probs[1];
    assert!(
        total.is_finite() && total > 0.0,
        "Cannot measure a state with nonfinite or zero norm"
    );
    let outcome = usize::from(rng.gen::<f64>() < probs[1] / total);
    let scale = probs[outcome].sqrt().recip();
    let collapse = |i: usize, amplitude: &mut C| {
        if usize::from((i & bit) != 0) == outcome {
            *amplitude *= scale;
        } else {
            *amplitude = C::new(0.0, 0.0);
        }
    };
    if parallel {
        state
            .par_iter_mut()
            .enumerate()
            .with_min_len(PAR_CHUNK)
            .for_each(|(i, a)| collapse(i, a));
    } else {
        for (i, amplitude) in state.iter_mut().enumerate() {
            collapse(i, amplitude);
        }
    }
    outcome as u8
}

/// Sample and collapse one qubit. Work is partitioned independently of its index.
pub fn measure_qubit<R: Rng>(state: &mut [C], qubit: usize, n: usize, rng: &mut R) -> u8 {
    measure_qubit_impl(state, qubit, n, rng, true)
}

pub fn measure_qubit_seq<R: Rng>(state: &mut [C], qubit: usize, n: usize, rng: &mut R) -> u8 {
    measure_qubit_impl(state, qubit, n, rng, false)
}

/// Marginals in one pass over the state. qubits[0] is the output MSB.
/// Full probabilities in descending qubit order use a direct O(2^n) path.
pub fn marginal_probs(state: &[C], n: usize, qubits: &[usize]) -> Vec<f64> {
    validate_state(state, n);
    qubit_mask(qubits, n);
    if qubits.len() == n && qubits.iter().copied().eq((0..n).rev()) {
        return state.iter().map(|a| a.norm_sqr()).collect();
    }
    let mut probs = vec![0.0; 1usize << qubits.len()];
    for (i, amplitude) in state.iter().enumerate() {
        let outcome = qubits
            .iter()
            .fold(0, |index, &q| (index << 1) | ((i >> q) & 1));
        probs[outcome] += amplitude.norm_sqr();
    }
    probs
}

/// Sample without collapsing the state. Normalize probabilities and reuse their
/// allocation for the CDF; format bitstrings only for distinct sampled outcomes.
pub fn sample_counts<R: Rng>(
    state: &[C],
    n: usize,
    shots: usize,
    rng: &mut R,
    qubits: Option<&[usize]>,
) -> std::collections::HashMap<String, usize> {
    let all_qubits;
    let q = match qubits {
        Some(qs) => qs,
        None => {
            all_qubits = (0..n).rev().collect::<Vec<_>>();
            &all_qubits
        }
    };
    validate_state(state, n);
    qubit_mask(q, n);
    if shots == 0 {
        return std::collections::HashMap::new();
    }
    let mut cdf = marginal_probs(state, n, q);
    let total: f64 = cdf.iter().sum();
    assert!(
        total.is_finite() && total > 0.0,
        "Cannot sample a state with nonfinite or zero norm"
    );
    let mut cumulative = 0.0;
    for probability in &mut cdf {
        cumulative += *probability;
        *probability = (cumulative / total).min(1.0);
    }
    *cdf.last_mut().unwrap() = 1.0;
    let mut counts = std::collections::HashMap::<usize, usize>::new();
    for _ in 0..shots {
        let r: f64 = rng.gen();
        // Strictly greater skips zero-probability intervals even when r == 0.
        let outcome = cdf.partition_point(|&p| p <= r);
        *counts.entry(outcome).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .map(|(outcome, count)| {
            let bits = if q.is_empty() {
                String::new()
            } else {
                format!("{:0width$b}", outcome, width = q.len())
            };
            (bits, count)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::mock::StepRng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn input(n: usize) -> Vec<C> {
        let mut state: Vec<_> = (0..1usize << n)
            .map(|i| C::new((i as f64 + 0.25).sin(), (i as f64 + 0.5).cos()))
            .collect();
        let norm = state.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        for amplitude in &mut state {
            *amplitude /= norm;
        }
        state
    }

    fn close(actual: &[C], expected: &[C]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((*a - *e).norm() < 1e-12, "Amplitude {i}: {a} != {e}");
        }
    }

    /// Independent out-of-place reference: enumerate each output basis index,
    /// and sum all input basis indices differing on the gate's target qubits.
    fn dense_reference(state: &[C], matrix: &[Vec<C>], qubits: &[usize]) -> Vec<C> {
        let mut output = vec![C::new(0.0, 0.0); state.len()];
        for (basis, result) in output.iter_mut().enumerate() {
            let row = qubits
                .iter()
                .fold(0, |row, &q| 2 * row + ((basis >> q) & 1));
            for (col, coefficient) in matrix[row].iter().enumerate() {
                let mut source = basis;
                for (position, &q) in qubits.iter().enumerate() {
                    source &= !(1usize << q);
                    source |= ((col >> (qubits.len() - 1 - position)) & 1) << q;
                }
                *result += coefficient * state[source];
            }
        }
        output
    }

    fn fourier(dim: usize) -> Vec<Vec<C>> {
        (0..dim)
            .map(|row| {
                (0..dim)
                    .map(|col| {
                        C::from_polar(
                            (dim as f64).sqrt().recip(),
                            std::f64::consts::TAU * (row * col) as f64 / dim as f64,
                        )
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn dense_gates_match_reference_for_target_order_and_parallel_execution() {
        for n in [5, PAR_THRESHOLD] {
            let original = input(n);
            for qs in [
                vec![0],
                vec![n - 1],
                vec![n - 1, 0],
                vec![0, 3, 1],
                vec![4, 2, 0, 3, 1],
                vec![],
            ] {
                let matrix = fourier(1 << qs.len());
                let expected = dense_reference(&original, &matrix, &qs);
                let mut seq = original.clone();
                apply_n_qubit_seq(&mut seq, &matrix, &qs, n);
                close(&seq, &expected);
                let mut par = original.clone();
                apply_n_qubit(&mut par, &matrix, &qs, n);
                close(&par, &expected);
            }
        }
    }

    #[test]
    fn controlled_one_qubit_and_x_match_dense_reference() {
        let matrix = fourier(2);
        let u = [[matrix[0][0], matrix[0][1]], [matrix[1][0], matrix[1][1]]];
        for n in [4, PAR_THRESHOLD] {
            for target in [0, n - 1] {
                let control = if target == 0 { n - 1 } else { 0 };
                for value in [false, true] {
                    for is_x in [false, true] {
                        let mut controlled = vec![vec![C::new(0.0, 0.0); 4]; 4];
                        for row in 0..4 {
                            for col in 0..4 {
                                controlled[row][col] = if row / 2 != col / 2 {
                                    C::new(0.0, 0.0)
                                } else if row / 2 == usize::from(value) {
                                    if is_x {
                                        C::new(f64::from(row % 2 != col % 2), 0.0)
                                    } else {
                                        u[row % 2][col % 2]
                                    }
                                } else {
                                    C::new(f64::from(row == col), 0.0)
                                };
                            }
                        }
                        let original = input(n);
                        let expected = dense_reference(&original, &controlled, &[control, target]);
                        for parallel in [false, true] {
                            let mut actual = original.clone();
                            if is_x {
                                apply_x(&mut actual, target, &[(control, value)], parallel);
                            } else {
                                apply_one_qubit_impl(
                                    &mut actual,
                                    &u,
                                    target,
                                    n,
                                    &[(control, value)],
                                    parallel,
                                );
                            }
                            close(&actual, &expected);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn diagonal_and_controlled_swap_match_dense_reference() {
        for n in [4, PAR_THRESHOLD, PAR_THRESHOLD + 1] {
            let original = input(n);
            for qs in [vec![0, n - 1], vec![n - 1, 0]] {
                let diagonal: Vec<_> = (0..4).map(|j| C::from_polar(1.0, j as f64 * 0.3)).collect();
                let mut matrix = vec![vec![C::new(0.0, 0.0); 4]; 4];
                for j in 0..4 {
                    matrix[j][j] = diagonal[j];
                }
                let expected = dense_reference(&original, &matrix, &qs);
                for parallel in [false, true] {
                    let mut actual = original.clone();
                    apply_diagonal(&mut actual, &qs, &diagonal, parallel);
                    close(&actual, &expected);
                }
            }
            for (a, b) in [(0, n - 1), (n - 1, 0), (n - 1, n - 2)] {
                let control = (0..n).find(|&q| q != a && q != b).unwrap();
                for controls in [vec![], vec![(control, true)], vec![(control, false)]] {
                    let qubits = if controls.is_empty() {
                        vec![a, b]
                    } else {
                        vec![control, a, b]
                    };
                    let dim = 1 << qubits.len();
                    let mut matrix = vec![vec![C::new(0.0, 0.0); dim]; dim];
                    (0..dim).for_each(|source| {
                        let enabled =
                            controls.is_empty() || (source / 4 == usize::from(controls[0].1));
                        let dest = if enabled {
                            (source & !3) | ((source & 1) << 1) | ((source & 2) >> 1)
                        } else {
                            source
                        };
                        matrix[dest][source] = C::new(1.0, 0.0);
                    });
                    let expected = dense_reference(&original, &matrix, &qubits);
                    for parallel in [false, true] {
                        let mut actual = original.clone();
                        apply_swap(&mut actual, a, b, &controls, parallel);
                        close(&actual, &expected);
                    }
                }
            }
        }
    }

    #[test]
    fn marginal_probabilities_cover_arbitrary_order_and_empty_selection() {
        let state: Vec<_> = (1..=8)
            .map(|i| C::new((i as f64 / 36.0).sqrt(), 0.0))
            .collect();
        for (qubits, expected) in [
            (vec![0, 2], vec![4.0, 12.0, 6.0, 14.0]),
            (vec![2, 0], vec![4.0, 6.0, 12.0, 14.0]),
            (vec![], vec![36.0]),
            (vec![2, 1, 0], (1..=8).map(f64::from).collect()),
        ] {
            let actual = marginal_probs(&state, 3, &qubits);
            for (a, e) in actual.iter().zip(expected) {
                assert!((a - e / 36.0).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn counts_normalize_and_skip_zero_probability_at_zero_rng() {
        let state = [C::new(0.0, 0.0), C::new(2.0, 0.0)];
        let mut rng = StepRng::new(0, 0);
        assert_eq!(
            sample_counts(&state, 1, 10, &mut rng, None).get("1"),
            Some(&10)
        );
        assert_eq!(
            sample_counts(&state, 1, 7, &mut rng, Some(&[])).get(""),
            Some(&7)
        );
        assert!(sample_counts(&state, 1, 0, &mut rng, None).is_empty());
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let counts = sample_counts(
            &[C::new(2.0, 0.0), C::new(2.0, 0.0)],
            1,
            10_000,
            &mut rng,
            None,
        );
        assert_eq!(counts.values().sum::<usize>(), 10_000);
        assert!((counts["0"] as isize - 5000).abs() < 250);
        assert!(sample_counts(&[C::new(1.0, 0.0)], 0, 1, &mut rng, None).contains_key(""));
    }

    #[test]
    fn measurement_normalizes_both_branches_and_high_targets() {
        for n in [1, PAR_THRESHOLD] {
            for target in [0, n - 1] {
                for outcome in [0, 1] {
                    let mut state = vec![C::new(0.0, 0.0); 1 << n];
                    state[0] = C::new(2.0, 0.0);
                    state[1 << target] = C::new(2.0, 0.0);
                    let mut rng = StepRng::new(if outcome == 1 { 0 } else { u64::MAX }, 0);
                    let measured = measure_qubit(&mut state, target, n, &mut rng);
                    assert_eq!(measured, outcome);
                    assert!((state.iter().map(|a| a.norm_sqr()).sum::<f64>() - 1.0).abs() < 1e-14);
                    assert_eq!(state[(outcome as usize) << target], C::new(1.0, 0.0));
                    let mut repeated_rng = ChaCha8Rng::seed_from_u64(9);
                    assert_eq!(
                        measure_qubit_seq(&mut state, target, n, &mut repeated_rng),
                        outcome
                    );
                }
            }
        }
    }

    #[test]
    fn dense_scatter_rejects_invalid_layouts_before_access() {
        let cases = [
            (12, vec![0, 0], fourier(4), 1usize << 12),
            (12, vec![12], fourier(2), 1usize << 12),
            (12, vec![0], fourier(4), 1usize << 12),
            (
                12,
                vec![0],
                vec![vec![C::new(0.0, 0.0)], vec![C::new(0.0, 0.0)]],
                1usize << 12,
            ),
            (12, vec![0], fourier(2), 16usize),
            (12, vec![0, 1, 2, 3, 4, 5], fourier(64), 1usize << 12),
        ];
        for (n, qs, matrix, len) in cases {
            assert!(std::panic::catch_unwind(|| {
                let mut state = vec![C::new(0.0, 0.0); len];
                apply_n_qubit(&mut state, &matrix, &qs, n);
            })
            .is_err());
        }
    }
}
