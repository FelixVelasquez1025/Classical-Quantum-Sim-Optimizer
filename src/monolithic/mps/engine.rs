//! Mixed-canonical MPS. `center` is the sole tensor carrying the state norm.
use crate::gates;
use nalgebra::DMatrix;
use num_complex::Complex64 as C;
use rand::Rng;
use serde::Serialize;
use std::time::Instant;

#[derive(Clone)]
pub(super) struct Options {
    pub max_bond: Option<usize>,
    pub threshold: f64,
    pub discarded_budget: f64,
    pub memory: usize,
    pub profile: bool,
}

#[derive(Clone, Default, Serialize)]
pub(super) struct Stats {
    pub svd_calls: usize,
    pub svd_time: f64,
    pub routing_swaps: usize,
    pub center_moves: usize,
    pub peak_bond_dimension: usize,
    pub peak_tensor_bytes: usize,
    pub peak_working_bytes: usize,
    pub discarded_weight: f64,
    pub truncations: usize,
    pub bond_cap_truncations: usize,
}

#[derive(Clone)]
struct Tensor {
    left: usize,
    right: usize,
    data: Vec<C>,
}
impl Tensor {
    fn zero(left: usize, right: usize) -> Self {
        Self {
            left,
            right,
            data: vec![C::default(); 2 * left * right],
        }
    }
    fn get(&self, l: usize, s: usize, r: usize) -> C {
        self.data[(2 * l + s) * self.right + r]
    }
    fn set(&mut self, l: usize, s: usize, r: usize, v: C) {
        self.data[(2 * l + s) * self.right + r] = v;
    }
}

#[derive(Clone)]
pub(super) struct Mps {
    tensors: Vec<Tensor>,
    center: usize,
    // Both directions are maintained as routing changes the tensor order.
    positions: Vec<usize>,
    labels: Vec<usize>,
    pub options: Options,
    pub stats: Stats,
}

pub(super) fn checked_bytes(items: usize, size: usize) -> Result<usize, String> {
    items
        .checked_mul(size)
        .ok_or_else(|| "MPS allocation size overflow".into())
}
impl Mps {
    pub fn new(n: usize, options: Options) -> Result<Self, String> {
        let bytes = checked_bytes(
            n,
            2 * 16 + std::mem::size_of::<Tensor>() + 2 * std::mem::size_of::<usize>(),
        )?;
        if bytes > options.memory {
            return Err("MPS initial tensors exceed max_memory_mb".into());
        }
        let tensors = (0..n)
            .map(|_| {
                let mut t = Tensor::zero(1, 1);
                t.data[0] = C::new(1., 0.);
                t
            })
            .collect();
        let mut out = Self {
            tensors,
            center: 0,
            positions: (0..n).collect(),
            labels: (0..n).collect(),
            options,
            stats: Stats::default(),
        };
        out.record();
        Ok(out)
    }
    pub fn len(&self) -> usize {
        self.tensors.len()
    }
    pub fn bytes(&self) -> usize {
        self.tensors
            .iter()
            .map(|t| t.data.len() * 16)
            .sum::<usize>()
            + self.len() * (std::mem::size_of::<Tensor>() + 2 * std::mem::size_of::<usize>())
    }
    // Conservative allowance for decomposition factors, clones and library workspace.
    fn workspace(&mut self, elements: usize) -> Result<(), String> {
        let bytes = checked_bytes(elements, 16 * 16)?
            .checked_add(self.bytes())
            .ok_or("MPS allocation size overflow")?;
        self.check_memory(bytes)?;
        self.stats.peak_working_bytes = self.stats.peak_working_bytes.max(bytes);
        Ok(())
    }
    pub fn check_memory(&self, bytes: usize) -> Result<(), String> {
        if bytes > self.options.memory {
            Err("MPS working allocation exceeds max_memory_mb".into())
        } else {
            Ok(())
        }
    }
    /// Include temporary vectors and an optional retained output allocation.
    pub fn query_memory(&self, extra: usize, contraction: bool) -> Result<(), String> {
        let max = self
            .tensors
            .iter()
            .map(|t| t.left.max(t.right))
            .max()
            .unwrap_or(1);
        let scratch = if contraction {
            checked_bytes(checked_bytes(max, max)?, 64)?
        } else {
            checked_bytes(max, 64)?
        };
        let bytes = self
            .bytes()
            .checked_add(scratch)
            .and_then(|v| v.checked_add(self.len().checked_mul(80)?))
            .and_then(|v| v.checked_add(extra))
            .ok_or("MPS allocation size overflow")?;
        self.check_memory(bytes)
    }
    /// Restore a trajectory while reusing allocations when tensor shapes agree.
    pub fn restore(&mut self, source: &Self) {
        self.center = source.center;
        self.positions.clone_from(&source.positions);
        self.labels.clone_from(&source.labels);
        self.tensors.truncate(source.tensors.len());
        for (i, t) in source.tensors.iter().enumerate() {
            if let Some(dst) = self.tensors.get_mut(i) {
                dst.left = t.left;
                dst.right = t.right;
                dst.data.clone_from(&t.data);
                // Do not retain buffers from a much larger previous trajectory.
                dst.data.shrink_to_fit();
            } else {
                self.tensors.push(t.clone());
            }
        }
        self.stats = Stats::default();
        self.record();
    }
    pub fn reset_zero(&mut self) {
        for t in &mut self.tensors {
            t.left = 1;
            t.right = 1;
            t.data.clear();
            t.data.shrink_to(2);
            t.data.resize(2, C::default());
            t.data[0] = C::new(1., 0.);
        }
        for i in 0..self.len() {
            self.positions[i] = i;
            self.labels[i] = i;
        }
        self.center = 0;
        self.stats = Stats::default();
        self.record();
    }
    fn record(&mut self) {
        self.stats.peak_bond_dimension = self
            .stats
            .peak_bond_dimension
            .max(self.tensors.iter().map(|t| t.right).max().unwrap_or(1));
        self.stats.peak_tensor_bytes = self.stats.peak_tensor_bytes.max(self.bytes());
        self.stats.peak_working_bytes = self.stats.peak_working_bytes.max(self.bytes());
    }
    pub fn bonds(&self) -> Vec<usize> {
        self.tensors
            .iter()
            .take(self.len().saturating_sub(1))
            .map(|t| t.right)
            .collect()
    }
    pub fn order(&self) -> Vec<usize> {
        self.labels.clone()
    }
    pub fn norm_squared(&self) -> f64 {
        self.tensors
            .get(self.center)
            .map(|t| t.data.iter().map(|x| x.norm_sqr()).sum())
            .unwrap_or(1.)
    }
    pub fn move_center(&mut self, target: usize) -> Result<(), String> {
        if self.len() == 0 {
            return Ok(());
        }
        while self.center < target {
            self.shift_right()?;
        }
        while self.center > target {
            self.shift_left()?;
        }
        Ok(())
    }
    fn shift_right(&mut self) -> Result<(), String> {
        let q = self.center;
        let (l, b, r) = (
            self.tensors[q].left,
            self.tensors[q].right,
            self.tensors[q + 1].right,
        );
        self.workspace(checked_bytes(2 * l, b)?.max(checked_bytes(2 * b, r)?))?;
        let a = &self.tensors[q];
        let matrix = DMatrix::from_fn(2 * l, b, |i, j| a.get(i / 2, i % 2, j));
        let (u, v) = matrix.qr().unpack();
        let k = u.ncols();
        let mut left = Tensor::zero(l, k);
        let mut right = Tensor::zero(k, r);
        for i in 0..l {
            for s in 0..2 {
                for j in 0..k {
                    left.set(i, s, j, u[(2 * i + s, j)]);
                }
            }
        }
        let old = &self.tensors[q + 1];
        for i in 0..k {
            for s in 0..2 {
                for j in 0..r {
                    right.set(i, s, j, (0..b).map(|x| v[(i, x)] * old.get(x, s, j)).sum());
                }
            }
        }
        self.tensors[q] = left;
        self.tensors[q + 1] = right;
        self.center += 1;
        self.stats.center_moves += 1;
        self.record();
        Ok(())
    }
    fn shift_left(&mut self) -> Result<(), String> {
        let q = self.center;
        let (l, b, r) = (
            self.tensors[q - 1].left,
            self.tensors[q].left,
            self.tensors[q].right,
        );
        self.workspace(checked_bytes(2 * r, b)?.max(checked_bytes(2 * l, b)?))?;
        let a = &self.tensors[q];
        let matrix = DMatrix::from_fn(2 * r, b, |i, j| a.get(j, i / r, i % r).conj());
        let (u, v) = matrix.qr().unpack();
        let k = u.ncols();
        let mut left = Tensor::zero(l, k);
        let mut right = Tensor::zero(k, r);
        for i in 0..k {
            for s in 0..2 {
                for j in 0..r {
                    right.set(i, s, j, u[(s * r + j, i)].conj());
                }
            }
        }
        let old = &self.tensors[q - 1];
        for i in 0..l {
            for s in 0..2 {
                for j in 0..k {
                    left.set(
                        i,
                        s,
                        j,
                        (0..b).map(|x| old.get(i, s, x) * v[(j, x)].conj()).sum(),
                    );
                }
            }
        }
        self.tensors[q - 1] = left;
        self.tensors[q] = right;
        self.center -= 1;
        self.stats.center_moves += 1;
        self.record();
        Ok(())
    }
    pub fn apply_1q(&mut self, qubit: usize, mat: &[[C; 2]; 2]) {
        let t = &mut self.tensors[self.positions[qubit]];
        let diagonal = mat[0][1] == C::default() && mat[1][0] == C::default();
        for l in 0..t.left {
            for r in 0..t.right {
                let a = t.get(l, 0, r);
                let b = t.get(l, 1, r);
                if diagonal {
                    t.set(l, 0, r, mat[0][0] * a);
                    t.set(l, 1, r, mat[1][1] * b);
                } else {
                    t.set(l, 0, r, mat[0][0] * a + mat[0][1] * b);
                    t.set(l, 1, r, mat[1][0] * a + mat[1][1] * b);
                }
            }
        }
    }
    // A logical SWAP only relabels wires; it need not change the tensor chain.
    pub fn swap(&mut self, a: usize, b: usize) {
        let (x, y) = (self.positions[a], self.positions[b]);
        self.labels.swap(x, y);
        self.positions.swap(a, b);
    }
    fn route_swap(&mut self, q: usize) -> Result<(), String> {
        self.adjacent(q, &gates::swap())?;
        self.labels.swap(q, q + 1);
        self.positions[self.labels[q]] = q;
        self.positions[self.labels[q + 1]] = q + 1;
        self.stats.routing_swaps += 1;
        Ok(())
    }
    pub fn apply_2q(&mut self, a: usize, b: usize, mat: &[[C; 4]; 4]) -> Result<(), String> {
        let (x, y) = (self.positions[a], self.positions[b]);
        let (lo, hi) = (x.min(y), x.max(y));
        // Estimate intermediate matrix sizes in both directions. Retain the new order.
        let left_cost: f64 = (lo..hi.saturating_sub(1))
            .map(|p| (self.tensors[p].left as f64 * self.tensors[p + 1].right as f64).powi(2))
            .sum();
        let right_cost: f64 = ((lo + 1)..hi)
            .map(|p| (self.tensors[p].left as f64 * self.tensors[p + 1].right as f64).powi(2))
            .sum();
        if left_cost < right_cost {
            for p in lo..hi - 1 {
                self.route_swap(p)?;
            }
        } else {
            for p in ((lo + 1)..hi).rev() {
                self.route_swap(p)?;
            }
        }
        let (x, y) = (self.positions[a], self.positions[b]);
        let matrix = if x < y {
            *mat
        } else {
            let permutation = [0, 2, 1, 3];
            std::array::from_fn(|i| std::array::from_fn(|j| mat[permutation[i]][permutation[j]]))
        };
        self.adjacent(x.min(y), &matrix)
    }
    fn adjacent(&mut self, q: usize, mat: &[[C; 4]; 4]) -> Result<(), String> {
        self.move_center(q)?;
        let (l, b, r) = (
            self.tensors[q].left,
            self.tensors[q].right,
            self.tensors[q + 1].right,
        );
        self.workspace(checked_bytes(2 * l, 2 * r)?)?;
        let a = &self.tensors[q];
        let z = &self.tensors[q + 1];
        let mut theta = DMatrix::<C>::zeros(2 * l, 2 * r);
        for i in 0..l {
            for j in 0..r {
                let input: [C; 4] = std::array::from_fn(|s| {
                    (0..b)
                        .map(|x| a.get(i, s / 2, x) * z.get(x, s % 2, j))
                        .sum()
                });
                for out in 0..4 {
                    theta[(2 * i + out / 2, (out % 2) * r + j)] =
                        (0..4).map(|s| mat[out][s] * input[s]).sum();
                }
            }
        }
        let start = self.options.profile.then(Instant::now);
        let (u, s, vt) = super::svd::decompose(theta)?;
        if let Some(t) = start {
            self.stats.svd_time += t.elapsed().as_secs_f64();
        }
        self.stats.svd_calls += 1;
        let total: f64 = s.iter().map(|v| v * v).sum();
        if !total.is_finite() || total <= 0. {
            return Err("MPS SVD produced an invalid norm".into());
        }
        let mut keep = s.len();
        // Legacy absolute cutoff and optional relative discarded-weight budget.
        let mut loss = 0.;
        while keep > 1 {
            let w = s[keep - 1] * s[keep - 1];
            if s[keep - 1] <= self.options.threshold
                || (self.options.discarded_budget > 0.
                    && (loss + w) / total <= self.options.discarded_budget)
            {
                loss += w;
                keep -= 1;
            } else {
                break;
            }
        }
        if let Some(cap) = self.options.max_bond {
            if keep > cap {
                self.stats.bond_cap_truncations += 1;
                keep = cap;
            }
        }
        let discarded: f64 = s.iter().skip(keep).map(|v| v * v).sum::<f64>() / total;
        let kept: f64 = s.iter().take(keep).map(|v| v * v).sum();
        if kept <= 0. || !kept.is_finite() {
            return Err("MPS truncation removed the state".into());
        }
        if discarded > 0. {
            self.stats.truncations += 1;
            self.stats.discarded_weight += discarded;
        }
        let scale = 1. / kept.sqrt();
        let mut a = Tensor::zero(l, keep);
        let mut z = Tensor::zero(keep, r);
        for i in 0..l {
            for s in 0..2 {
                for j in 0..keep {
                    a.set(i, s, j, u[(2 * i + s, j)]);
                }
            }
        }
        for i in 0..keep {
            for state in 0..2 {
                for j in 0..r {
                    z.set(i, state, j, vt[(i, state * r + j)] * (s[i] * scale));
                }
            }
        }
        self.tensors[q] = a;
        self.tensors[q + 1] = z;
        self.center = q + 1;
        self.record();
        Ok(())
    }
    pub fn measure(&mut self, qubit: usize, rng: &mut impl Rng) -> Result<usize, String> {
        let q = self.positions[qubit];
        self.move_center(q)?;
        let t = &mut self.tensors[q];
        let mut p = [0., 0.];
        for l in 0..t.left {
            for (s, prob) in p.iter_mut().enumerate() {
                for r in 0..t.right {
                    *prob += t.get(l, s, r).norm_sqr();
                }
            }
        }
        let total = p[0] + p[1];
        if !total.is_finite() || total <= 0. {
            return Err("MPS measurement encountered an invalid norm".into());
        }
        let outcome = usize::from(rng.gen::<f64>() * total >= p[0]);
        let scale = 1. / p[outcome].sqrt();
        for l in 0..t.left {
            for s in 0..2 {
                for r in 0..t.right {
                    t.set(
                        l,
                        s,
                        r,
                        if s == outcome {
                            t.get(l, s, r) * scale
                        } else {
                            C::default()
                        },
                    );
                }
            }
        }
        Ok(outcome)
    }
    /// Requires center=0; sample conditionals without cloning or modifying tensors.
    pub fn sample(&self, rng: &mut impl Rng) -> Result<Vec<usize>, String> {
        debug_assert!(self.center == 0 || self.len() == 0);
        self.query_memory(0, false)?;
        let mut bits = vec![0; self.len()];
        let mut work = vec![C::new(1., 0.)];
        let mut next = [Vec::new(), Vec::new()];
        for (q, t) in self.tensors.iter().enumerate() {
            let mut p = [0., 0.];
            for s in 0..2 {
                next[s].resize(t.right, C::default());
                for (r, value) in next[s].iter_mut().enumerate() {
                    *value = (0..t.left).map(|l| work[l] * t.get(l, s, r)).sum();
                    p[s] += value.norm_sqr();
                }
            }
            let total = p[0] + p[1];
            if !total.is_finite() || total <= 0. {
                return Err("MPS sampling encountered an invalid norm".into());
            }
            let outcome = usize::from(rng.gen::<f64>() * total >= p[0]);
            bits[self.labels[q]] = outcome;
            std::mem::swap(&mut work, &mut next[outcome]);
            let norm = p[outcome].sqrt();
            for x in &mut work {
                *x /= norm;
            }
        }
        Ok(bits)
    }
    pub fn amplitude(&self, bits: &[usize]) -> C {
        self.amplitude_with_buffers(bits, &mut Vec::new(), &mut Vec::new())
    }
    pub fn amplitude_with_buffers(
        &self,
        bits: &[usize],
        work: &mut Vec<C>,
        next: &mut Vec<C>,
    ) -> C {
        work.clear();
        work.push(C::new(1., 0.));
        for (q, t) in self.tensors.iter().enumerate() {
            next.resize(t.right, C::default());
            for (r, value) in next.iter_mut().enumerate() {
                *value = (0..t.left)
                    .map(|l| work[l] * t.get(l, bits[self.labels[q]], r))
                    .sum();
            }
            std::mem::swap(work, next);
        }
        work[0]
    }
    /// Contract a product observable, or project selected wires for a marginal.
    pub fn contract(&self, operators: &[[[C; 2]; 2]]) -> Result<C, String> {
        self.query_memory(0, true)?;
        let mut env = DMatrix::from_element(1, 1, C::new(1., 0.));
        for (q, t) in self.tensors.iter().enumerate() {
            let mut next = DMatrix::zeros(t.right, t.right);
            let op = &operators[self.labels[q]];
            // E' = sum_st O_st A_s^dagger E A_t; O(D^3), not D^4.
            for (s, row) in op.iter().enumerate() {
                for (k, &element) in row.iter().enumerate() {
                    if element == C::default() {
                        continue;
                    }
                    let mut tmp = DMatrix::<C>::zeros(t.left, t.right);
                    for i in 0..t.left {
                        for r in 0..t.right {
                            tmp[(i, r)] = (0..t.left).map(|j| env[(i, j)] * t.get(j, k, r)).sum();
                        }
                    }
                    for i in 0..t.right {
                        for j in 0..t.right {
                            next[(i, j)] += element
                                * (0..t.left)
                                    .map(|l| t.get(l, s, i).conj() * tmp[(l, j)])
                                    .sum::<C>();
                        }
                    }
                }
            }
            env = next;
        }
        Ok(env[(0, 0)])
    }
}
