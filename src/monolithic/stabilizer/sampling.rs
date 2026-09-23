//! Computational-basis outcomes of a stabilizer state are uniform on an affine
//! binary space. The X parts of its stabilizers generate the translation space.
use super::{engine::Tableau, execution::Stats};
use rand::Rng;
use std::time::Instant;

pub(super) fn workspace(k: usize) -> Result<usize, String> {
    let w = k.checked_add(63).ok_or("Stabilizer allocation overflow")? / 64;
    k.checked_mul(w)
        .and_then(|v| v.checked_mul(8))
        .and_then(|v| v.checked_add(k.checked_mul(32)?))
        .and_then(|v| v.checked_add(w.checked_mul(24)?))
        .and_then(|v| v.checked_add(256))
        .filter(|&v| v <= isize::MAX as usize)
        .ok_or_else(|| "Stabilizer allocation overflow".into())
}
pub(super) struct Affine {
    pub k: usize,
    pub words: usize,
    pub rank: usize,
    basis: Vec<u64>,
    pivots: Vec<usize>,
    pub offset: Vec<u64>,
}
impl Affine {
    // Caller reserves tableau + workspace before entering. Consumes the quantum
    // state by projecting once, with zero chosen for every random measurement.
    pub fn build(state: &mut Tableau, qubits: &[usize], stats: &mut Stats, profile: bool) -> Self {
        let k = qubits.len();
        let words = k.div_ceil(64);
        let mut out = Self {
            k,
            words,
            rank: 0,
            basis: vec![0; k * words],
            pivots: vec![usize::MAX; k],
            offset: vec![0; words],
        };
        let mut row = vec![0; words];
        for r in state.n..2 * state.n {
            row.fill(0);
            for (i, &q) in qubits.iter().enumerate() {
                if state.xbit(r, q) {
                    row[i / 64] |= 1 << (i % 64);
                }
            }
            for p in 0..k {
                if row[p / 64] & (1 << (p % 64)) == 0 {
                    continue;
                }
                if out.pivots[p] != usize::MAX {
                    let b = out.pivots[p] * words;
                    for (w, v) in row.iter_mut().enumerate() {
                        *v ^= out.basis[b + w];
                    }
                } else {
                    let b = out.rank * words;
                    out.basis[b..b + words].copy_from_slice(&row);
                    out.pivots[p] = out.rank;
                    out.rank += 1;
                    break;
                }
            }
        }
        let start = profile.then(Instant::now);
        for (i, &q) in qubits.iter().enumerate() {
            let pivot = state.pivot(q);
            stats.record(pivot.is_some());
            if state.project(q, pivot, false) {
                out.offset[i / 64] |= 1 << (i % 64);
            }
        }
        if let Some(start) = start {
            stats.measure_time += start.elapsed().as_secs_f64();
        }
        out
    }
    pub fn sample(&self, bits: &mut [u64], rng: &mut impl Rng) {
        bits.copy_from_slice(&self.offset);
        for r in 0..self.rank {
            if rng.gen::<bool>() {
                for (w, v) in bits.iter_mut().enumerate() {
                    *v ^= self.basis[r * self.words + w];
                }
            }
        }
    }
    pub fn outcome(&self, index: usize, bits: &mut [u64]) {
        bits.copy_from_slice(&self.offset);
        for r in 0..self.rank {
            if index & (1 << r) != 0 {
                for (w, v) in bits.iter_mut().enumerate() {
                    *v ^= self.basis[r * self.words + w];
                }
            }
        }
    }
    pub fn contains(&self, bits: &mut [u64]) -> bool {
        for (v, &o) in bits.iter_mut().zip(&self.offset) {
            *v ^= o;
        }
        for p in 0..self.k {
            if bits[p / 64] & (1 << (p % 64)) != 0 {
                if self.pivots[p] == usize::MAX {
                    return false;
                }
                let b = self.pivots[p] * self.words;
                for (w, v) in bits.iter_mut().enumerate() {
                    *v ^= self.basis[b + w];
                }
            }
        }
        true
    }
    pub fn bitstring(&self, bits: &[u64]) -> String {
        (0..self.k)
            .map(|i| if bit(bits, i) { '1' } else { '0' })
            .collect()
    }
}
pub(super) fn bit(bits: &[u64], i: usize) -> bool {
    bits[i / 64] & (1 << (i % 64)) != 0
}
