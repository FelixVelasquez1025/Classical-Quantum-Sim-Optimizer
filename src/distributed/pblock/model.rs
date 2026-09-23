//! Product of independent dense blocks. Node placement is not a state partition.
use crate::{engine, monolithic::statevector::execution::Operation};
use num_complex::Complex64;
use rand::Rng;
use rayon::prelude::*;
use serde::Serialize;
use std::time::Instant;
pub(super) type C = Complex64;
pub(super) type SamplingTables = Vec<(Vec<usize>, Vec<f64>)>;

#[derive(Clone)]
pub(super) struct Options {
    pub memory: usize,
    pub max_block_qubits: Option<usize>,
    pub split_separable: bool,
    pub max_split_qubits: usize,
    pub separation_tolerance: f64,
    pub profile: bool,
}
#[derive(Clone, Default, Serialize)]
pub(super) struct Stats {
    pub merge_calls: usize,
    pub merge_time: f64,
    pub merge_output_amplitudes: usize,
    pub split_calls: usize,
    pub separation_checks: usize,
    pub separation_time: f64,
    pub separation_residual: f64,
    pub skipped_controlled_gates: usize,
    pub peak_block_qubits: usize,
    pub peak_live_amplitudes: usize,
    pub peak_working_bytes: usize,
}
impl Stats {
    pub fn add(&mut self, other: &Self) {
        self.merge_calls += other.merge_calls;
        self.merge_time += other.merge_time;
        self.merge_output_amplitudes = self
            .merge_output_amplitudes
            .saturating_add(other.merge_output_amplitudes);
        self.split_calls += other.split_calls;
        self.separation_checks += other.separation_checks;
        self.separation_time += other.separation_time;
        self.separation_residual += other.separation_residual;
        self.skipped_controlled_gates += other.skipped_controlled_gates;
        self.peak_block_qubits = self.peak_block_qubits.max(other.peak_block_qubits);
        self.peak_live_amplitudes = self.peak_live_amplitudes.max(other.peak_live_amplitudes);
        self.peak_working_bytes = self.peak_working_bytes.max(other.peak_working_bytes);
    }
}
#[derive(Clone)]
pub(super) struct Block {
    pub state: Vec<C>,
    pub qubits: Vec<usize>,
}
#[derive(Clone)]
pub(super) struct BlockPool {
    pub blocks: Vec<Option<Block>>,
    positions: Vec<(usize, usize)>,
    known: Vec<Option<bool>>,
    free: Vec<usize>,
    pub options: Options,
    pub stats: Stats,
    live: usize,
}
pub(super) fn bytes(n: usize, size: usize) -> Result<usize, String> {
    n.checked_mul(size)
        .filter(|&v| v <= isize::MAX as usize)
        .ok_or_else(|| "P-block allocation size overflow".into())
}
pub(super) fn dimension(n: usize) -> Result<usize, String> {
    if n >= usize::BITS as usize {
        return Err("P-block dense allocation dimension exceeds addressable memory".into());
    }
    let dim = 1usize << n;
    bytes(dim, 16)?;
    Ok(dim)
}
fn insert_bit(i: usize, q: usize, bit: usize) -> usize {
    let low = (1usize << q) - 1;
    (i & low) | ((i & !low) << 1) | (bit << q)
}
impl BlockPool {
    pub fn new(n: usize, options: Options) -> Result<Self, String> {
        if bytes(n, 288)? > options.memory {
            return Err("P-block initial state exceeds max_memory_mb".into());
        }
        let blocks = (0..n)
            .map(|q| {
                Some(Block {
                    state: vec![C::new(1., 0.), C::default()],
                    qubits: vec![q],
                })
            })
            .collect();
        let mut pool = Self {
            blocks,
            positions: (0..n).map(|q| (q, 0)).collect(),
            known: vec![Some(false); n],
            free: Vec::new(),
            options,
            stats: Stats::default(),
            live: 2 * n,
        };
        pool.stats.peak_block_qubits = usize::from(n > 0);
        pool.record();
        Ok(pool)
    }
    pub fn len(&self) -> usize {
        self.positions.len()
    }
    pub fn bytes(&self) -> usize {
        self.live * 16 + self.len() * 256
    }
    pub fn check(&self, extra: usize) -> Result<usize, String> {
        let total = self
            .bytes()
            .checked_add(extra)
            .ok_or("P-block allocation size overflow")?;
        if total > self.options.memory {
            Err("P-block working allocation exceeds max_memory_mb".into())
        } else {
            Ok(total)
        }
    }
    fn reserve(&mut self, extra: usize) -> Result<(), String> {
        let total = self.check(extra)?;
        self.stats.peak_working_bytes = self.stats.peak_working_bytes.max(total);
        Ok(())
    }
    fn record(&mut self) {
        self.stats.peak_live_amplitudes = self.stats.peak_live_amplitudes.max(self.live);
        self.stats.peak_working_bytes = self.stats.peak_working_bytes.max(self.bytes());
    }
    pub fn block_qubits(&self) -> Vec<Vec<usize>> {
        self.blocks
            .iter()
            .flatten()
            .map(|b| b.qubits.clone())
            .collect()
    }
    pub fn norm_squared(&self) -> f64 {
        self.blocks
            .iter()
            .flatten()
            .map(|b| b.state.iter().map(|a| a.norm_sqr()).sum::<f64>())
            .product()
    }
    pub fn restore(&mut self, source: &Self) {
        let same = self
            .blocks
            .iter()
            .zip(&source.blocks)
            .all(|(a, b)| match (a, b) {
                (Some(a), Some(b)) => a.qubits == b.qubits,
                (None, None) => true,
                _ => false,
            });
        if same {
            for (a, b) in self.blocks.iter_mut().zip(&source.blocks) {
                if let (Some(a), Some(b)) = (a, b) {
                    a.state.copy_from_slice(&b.state);
                }
            }
        } else {
            self.blocks.clear();
            self.blocks.extend(source.blocks.iter().cloned());
        }
        self.positions.clone_from(&source.positions);
        self.known.clone_from(&source.known);
        self.free.clone_from(&source.free);
        self.live = source.live;
        self.stats = Stats::default();
        self.stats.peak_block_qubits = source
            .blocks
            .iter()
            .flatten()
            .map(|b| b.qubits.len())
            .max()
            .unwrap_or(0);
        self.record();
    }
    pub fn reset_zero(&mut self) {
        self.blocks.clear();
        self.free.clear();
        for q in 0..self.len() {
            self.blocks.push(Some(Block {
                state: vec![C::new(1., 0.), C::default()],
                qubits: vec![q],
            }));
            self.positions[q] = (q, 0);
            self.known[q] = Some(false);
        }
        self.live = 2 * self.len();
        self.stats = Stats::default();
        self.stats.peak_block_qubits = usize::from(self.len() > 0);
        self.record();
    }
    fn merge(&mut self, a: usize, b: usize, parallel: bool) -> Result<usize, String> {
        if a == b {
            return Ok(a);
        }
        let n = self.blocks[a].as_ref().unwrap().qubits.len()
            + self.blocks[b].as_ref().unwrap().qubits.len();
        if self.options.max_block_qubits.is_some_and(|limit| n > limit) {
            return Err(
                "P-block merge exceeds max_block_qubits; no entanglement was discarded".into(),
            );
        }
        let dim = dimension(n)?;
        self.reserve(bytes(dim, 16)?)?;
        let start = self.options.profile.then(Instant::now);
        let left = self.blocks[a].as_ref().unwrap();
        let right = self.blocks[b].as_ref().unwrap();
        let mut state = vec![C::default(); dim];
        let fill = |i: usize, chunk: &mut [C]| {
            for (out, amp) in chunk.iter_mut().zip(&left.state) {
                *out = *amp * right.state[i];
            }
        };
        if parallel && dim >= 4096 {
            state
                .par_chunks_mut(left.state.len())
                .enumerate()
                .for_each(|(i, c)| fill(i, c));
        } else {
            for (i, c) in state.chunks_mut(left.state.len()).enumerate() {
                fill(i, c);
            }
        }
        let mut qubits = left.qubits.clone();
        qubits.extend_from_slice(&right.qubits);
        self.live = self.live - left.state.len() - right.state.len() + dim;
        for (i, &q) in qubits.iter().enumerate() {
            self.positions[q] = (a, i);
        }
        self.blocks[a] = Some(Block { state, qubits });
        self.blocks[b] = None;
        self.free.push(b);
        self.stats.peak_block_qubits = self.stats.peak_block_qubits.max(n);
        self.stats.merge_calls += 1;
        self.stats.merge_output_amplitudes = self.stats.merge_output_amplitudes.saturating_add(dim);
        if let Some(t) = start {
            self.stats.merge_time += t.elapsed().as_secs_f64();
        }
        self.record();
        Ok(a)
    }
    fn ensure(&mut self, qs: &[usize], parallel: bool) -> Result<usize, String> {
        let mut blocks: Vec<_> = qs.iter().map(|&q| self.positions[q].0).collect();
        blocks.sort_unstable();
        blocks.dedup();
        blocks.sort_by_key(|&idx| self.blocks[idx].as_ref().unwrap().qubits.len());
        let first = blocks[0];
        for &idx in &blocks[1..] {
            self.merge(first, idx, parallel)?;
        }
        Ok(first)
    }
    fn refresh_known(&mut self, q: usize) {
        let (idx, _) = self.positions[q];
        let b = self.blocks[idx].as_ref().unwrap();
        self.known[q] = if b.qubits.len() == 1 {
            if b.state[1] == C::default() {
                Some(false)
            } else if b.state[0] == C::default() {
                Some(true)
            } else {
                None
            }
        } else {
            None
        };
    }
    fn controls(&mut self, controls: &[(usize, bool)]) -> Option<Vec<(usize, bool)>> {
        let mut active = Vec::new();
        for &(q, on) in controls {
            match self.known[q] {
                Some(value) if value != on => {
                    self.stats.skipped_controlled_gates += 1;
                    return None;
                }
                Some(_) => {}
                None => active.push((q, on)),
            }
        }
        Some(active)
    }
    pub fn one(
        &mut self,
        q: usize,
        m: &[[C; 2]; 2],
        controls: &[(usize, bool)],
        parallel: bool,
    ) -> Result<(), String> {
        let Some(active) = self.controls(controls) else {
            return Ok(());
        };
        if !active.is_empty() {
            let b = self.blocks[self.positions[q].0].as_ref().unwrap();
            if b.qubits.len() == 1 {
                let a = b.state[0];
                let z = b.state[1];
                if m[0][0] * a + m[0][1] * z == a && m[1][0] * a + m[1][1] * z == z {
                    self.stats.skipped_controlled_gates += 1;
                    return Ok(());
                }
            }
        }
        let mut qs: Vec<_> = active.iter().map(|c| c.0).collect();
        qs.push(q);
        let idx = self.ensure(&qs, parallel)?;
        let cs: Vec<_> = active
            .iter()
            .map(|&(c, on)| (self.positions[c].1, on))
            .collect();
        let target = self.positions[q].1;
        let block = self.blocks[idx].as_mut().unwrap();
        if *m == crate::gates::X {
            engine::apply_x(&mut block.state, target, &cs, parallel);
        } else if parallel {
            engine::apply_one_qubit(&mut block.state, m, target, block.qubits.len(), &cs);
        } else {
            engine::apply_one_qubit_seq(&mut block.state, m, target, block.qubits.len(), &cs);
        }
        for &wire in &qs {
            self.refresh_known(wire);
        }
        self.maybe_split(&qs)?;
        Ok(())
    }
    fn swap(&mut self, a: usize, b: usize) {
        let (ia, pa) = self.positions[a];
        let (ib, pb) = self.positions[b];
        self.blocks[ia].as_mut().unwrap().qubits[pa] = b;
        self.blocks[ib].as_mut().unwrap().qubits[pb] = a;
        self.positions.swap(a, b);
        self.known.swap(a, b);
    }
    pub fn gate(&mut self, wires: &[usize], op: &Operation, parallel: bool) -> Result<(), String> {
        match op {
            Operation::One {
                target,
                matrix,
                controls,
            } => {
                let cs: Vec<_> = controls.iter().map(|&(q, on)| (wires[q], on)).collect();
                self.one(wires[*target], matrix, &cs, parallel)?;
            }
            Operation::X { target, controls } => {
                let cs: Vec<_> = controls.iter().map(|&(q, on)| (wires[q], on)).collect();
                self.one(wires[*target], &crate::gates::X, &cs, parallel)?;
            }
            Operation::Swap { a, b, controls } => {
                let cs: Vec<_> = controls.iter().map(|&(q, on)| (wires[q], on)).collect();
                let Some(active) = self.controls(&cs) else {
                    return Ok(());
                };
                if active.is_empty() {
                    self.swap(wires[*a], wires[*b]);
                } else {
                    let mut qs: Vec<_> = active.iter().map(|c| c.0).collect();
                    qs.extend([wires[*a], wires[*b]]);
                    let idx = self.ensure(&qs, parallel)?;
                    let cs: Vec<_> = active
                        .iter()
                        .map(|&(q, on)| (self.positions[q].1, on))
                        .collect();
                    let (a, b) = (self.positions[wires[*a]].1, self.positions[wires[*b]].1);
                    engine::apply_swap(
                        &mut self.blocks[idx].as_mut().unwrap().state,
                        a,
                        b,
                        &cs,
                        parallel,
                    );
                    for &q in &qs {
                        self.refresh_known(q);
                    }
                    self.maybe_split(&qs)?;
                }
            }
            Operation::Diagonal { qubits, diagonal } => {
                let qs: Vec<_> = qubits.iter().map(|&q| wires[q]).collect();
                let active: Vec<_> = qs
                    .iter()
                    .copied()
                    .filter(|&q| self.known[q].is_none())
                    .collect();
                let reduced: Vec<_> = (0..1usize << active.len())
                    .map(|basis| {
                        let mut index = 0;
                        let mut cursor = 0;
                        for &q in &qs {
                            let bit = if let Some(bit) = self.known[q] {
                                usize::from(bit)
                            } else {
                                let b = (basis >> (active.len() - 1 - cursor)) & 1;
                                cursor += 1;
                                b
                            };
                            index = (index << 1) | bit;
                        }
                        diagonal[index]
                    })
                    .collect();
                if active.is_empty() || reduced.iter().all(|&v| v == reduced[0]) {
                    let idx = self.positions[qs[0]].0;
                    for amp in &mut self.blocks[idx].as_mut().unwrap().state {
                        *amp *= reduced[0];
                    }
                } else {
                    let idx = self.ensure(&active, parallel)?;
                    let local: Vec<_> = active.iter().map(|&q| self.positions[q].1).collect();
                    engine::apply_diagonal(
                        &mut self.blocks[idx].as_mut().unwrap().state,
                        &local,
                        &reduced,
                        parallel,
                    );
                    self.maybe_split(&active)?;
                }
            }
            Operation::Dense { layout, matrix } => {
                let qs: Vec<_> = layout.qubits().iter().map(|&q| wires[q]).collect();
                let idx = self.ensure(&qs, parallel)?;
                let local: Vec<_> = qs.iter().map(|&q| self.positions[q].1).collect();
                let b = self.blocks[idx].as_mut().unwrap();
                engine::DenseLayout::new(&local).apply(
                    &mut b.state,
                    matrix,
                    b.qubits.len(),
                    parallel,
                );
                for &q in &qs {
                    self.refresh_known(q);
                }
                self.maybe_split(&qs)?;
            }
            Operation::Noop => {}
            _ => return Err("Internal error: nonunitary operation in a P-block gate".into()),
        }
        Ok(())
    }
    /// Install a separated wire, preserving the old local order of the rest.
    fn install_split(&mut self, q: usize, single: [C; 2], remaining: Vec<C>) {
        let (idx, local) = self.positions[q];
        let old = self.blocks[idx].take().unwrap();
        let slot = self
            .free
            .pop()
            .expect("A multi-qubit block leaves a free slot");
        let mut qs = old.qubits;
        qs.remove(local);
        self.live = self.live - old.state.len() + remaining.len() + 2;
        for (i, &wire) in qs.iter().enumerate() {
            self.positions[wire] = (idx, i);
        }
        self.blocks[idx] = Some(Block {
            state: remaining,
            qubits: qs,
        });
        self.blocks[slot] = Some(Block {
            state: single.to_vec(),
            qubits: vec![q],
        });
        self.positions[q] = (slot, 0);
        self.stats.split_calls += 1;
        self.refresh_known(q);
        let rest = self.blocks[idx].as_ref().unwrap().qubits.clone();
        for wire in rest {
            self.refresh_known(wire);
        }
        self.record();
    }
    pub fn measure(&mut self, q: usize, rng: &mut impl Rng, parallel: bool) -> Result<u8, String> {
        let (idx, local) = self.positions[q];
        let size = self.blocks[idx].as_ref().unwrap().state.len();
        if size > 2 {
            self.reserve(bytes(size / 2 + 2, 16)?)?;
        }
        let b = self.blocks[idx].as_mut().unwrap();
        let value = if parallel {
            engine::measure_qubit(&mut b.state, local, b.qubits.len(), rng)
        } else {
            engine::measure_qubit_seq(&mut b.state, local, b.qubits.len(), rng)
        };
        if size > 2 {
            let remaining = (0..size / 2)
                .map(|i| b.state[insert_bit(i, local, value as usize)])
                .collect();
            let mut single = [C::default(); 2];
            single[value as usize] = C::new(1., 0.);
            self.install_split(q, single, remaining);
        }
        self.known[q] = Some(value != 0);
        Ok(value)
    }
    fn maybe_split(&mut self, qs: &[usize]) -> Result<(), String> {
        if !self.options.split_separable {
            return Ok(());
        }
        for &q in qs {
            let (idx, local) = self.positions[q];
            let b = self.blocks[idx].as_ref().unwrap();
            if b.qubits.len() < 2 || b.qubits.len() > self.options.max_split_qubits {
                continue;
            }
            let size = b.state.len();
            let extra = bytes(size / 2 + 2, 16)?;
            if self.check(extra).is_err() {
                continue;
            }
            self.reserve(extra)?;
            let start = self.options.profile.then(Instant::now);
            self.stats.separation_checks += 1;
            let b = self.blocks[idx].as_ref().unwrap();
            let pivot = (0..size / 2)
                .max_by(|&i, &j| {
                    let weight = |i| {
                        b.state[insert_bit(i, local, 0)].norm_sqr()
                            + b.state[insert_bit(i, local, 1)].norm_sqr()
                    };
                    weight(i).total_cmp(&weight(j))
                })
                .unwrap();
            let mut single = [
                b.state[insert_bit(pivot, local, 0)],
                b.state[insert_bit(pivot, local, 1)],
            ];
            let norm = (single[0].norm_sqr() + single[1].norm_sqr()).sqrt();
            if norm == 0. || !norm.is_finite() {
                return Err("P-block separation encountered an invalid norm".into());
            }
            for amp in &mut single {
                *amp /= norm;
            }
            let mut remaining = Vec::with_capacity(size / 2);
            let mut loss = 0.;
            let mut total = 0.;
            for i in 0..size / 2 {
                let x = [
                    b.state[insert_bit(i, local, 0)],
                    b.state[insert_bit(i, local, 1)],
                ];
                let value = single[0].conj() * x[0] + single[1].conj() * x[1];
                remaining.push(value);
                for s in 0..2 {
                    loss += (x[s] - single[s] * value).norm_sqr();
                    total += x[s].norm_sqr();
                }
            }
            let relative = loss / total;
            if relative <= self.options.separation_tolerance {
                let kept: f64 = remaining.iter().map(|v| v.norm_sqr()).sum();
                let scale = (total / kept).sqrt();
                for amp in &mut remaining {
                    *amp *= scale;
                }
                self.install_split(q, single, remaining);
                self.stats.separation_residual += relative;
            }
            if let Some(t) = start {
                self.stats.separation_time += t.elapsed().as_secs_f64();
            }
        }
        Ok(())
    }
    pub fn amplitude(&self, bits: &[usize]) -> C {
        self.blocks
            .iter()
            .flatten()
            .map(|b| {
                let index = b
                    .qubits
                    .iter()
                    .enumerate()
                    .fold(0, |acc, (i, &q)| acc | (bits[q] << i));
                b.state[index]
            })
            .product()
    }
    pub fn sampling_tables(&self) -> Result<SamplingTables, String> {
        self.check(
            bytes(self.live, 8)?
                .checked_add(bytes(self.len(), 64)?)
                .ok_or("P-block allocation size overflow")?,
        )?;
        self.blocks
            .iter()
            .flatten()
            .map(|b| {
                let mut total = 0.;
                let cdf: Vec<_> = b
                    .state
                    .iter()
                    .map(|v| {
                        total += v.norm_sqr();
                        total
                    })
                    .collect();
                if total <= 0. || !total.is_finite() {
                    return Err("P-block sampling encountered an invalid norm".into());
                }
                Ok((b.qubits.clone(), cdf))
            })
            .collect()
    }
}
pub(super) fn sample(tables: &[(Vec<usize>, Vec<f64>)], bits: &mut [usize], rng: &mut impl Rng) {
    for (qs, cdf) in tables {
        let draw = rng.gen::<f64>() * cdf.last().unwrap();
        let index = cdf.partition_point(|&p| p <= draw).min(cdf.len() - 1);
        for (i, &q) in qs.iter().enumerate() {
            bits[q] = (index >> i) & 1;
        }
    }
}
