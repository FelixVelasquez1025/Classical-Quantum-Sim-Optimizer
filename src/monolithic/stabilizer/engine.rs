//! Packed Aaronson–Gottesman tableau. Rows 0..n are destabilizers,
//! n..2n stabilizers, and 2n is reusable scratch. Pauli code is x + 2*z.
use rand::Rng;

pub(super) fn allocation(n: usize) -> Result<usize, String> {
    let words = n.checked_add(63).ok_or("Stabilizer allocation overflow")? / 64;
    let rows = n
        .checked_mul(2)
        .and_then(|v| v.checked_add(1))
        .ok_or("Stabilizer allocation overflow")?;
    rows.checked_mul(words)
        .and_then(|v| v.checked_mul(16))
        .and_then(|v| v.checked_add(rows))
        .and_then(|v| v.checked_add(256))
        .filter(|&v| v <= isize::MAX as usize)
        .ok_or_else(|| "Stabilizer allocation overflow".into())
}
pub(super) fn check(bytes: usize, budget: usize) -> Result<(), String> {
    if bytes > budget {
        Err("Stabilizer working memory exceeds max_memory_mb".into())
    } else {
        Ok(())
    }
}

#[derive(Clone)]
pub(super) struct Tableau {
    pub n: usize,
    pub words: usize,
    pub x: Vec<u64>,
    pub z: Vec<u64>,
    pub signs: Vec<bool>,
}

// P(x,z) = i^popcount(x&z) X^x Z^z. This also accounts for XZ=-iY.
fn product_phase(xs: u64, zs: u64, xt: u64, zt: u64) -> i32 {
    (xs & zs).count_ones() as i32 + (xt & zt).count_ones() as i32
        - ((xs ^ xt) & (zs ^ zt)).count_ones() as i32
        + 2 * (zs & xt).count_ones() as i32
}
impl Tableau {
    pub fn new(n: usize, budget: usize) -> Result<Self, String> {
        check(allocation(n)?, budget)?;
        let words = n.div_ceil(64);
        let rows = 2 * n + 1;
        let mut state = Self {
            n,
            words,
            x: vec![0; rows * words],
            z: vec![0; rows * words],
            signs: vec![false; rows],
        };
        state.reset();
        Ok(state)
    }
    pub fn reset(&mut self) {
        self.x.fill(0);
        self.z.fill(0);
        self.signs.fill(false);
        for q in 0..self.n {
            self.x[q * self.words + q / 64] = 1 << (q % 64);
            self.z[(q + self.n) * self.words + q / 64] = 1 << (q % 64);
        }
    }
    pub fn restore(&mut self, other: &Self) {
        self.x.copy_from_slice(&other.x);
        self.z.copy_from_slice(&other.z);
        self.signs.copy_from_slice(&other.signs);
    }
    pub fn xbit(&self, row: usize, q: usize) -> bool {
        self.x[row * self.words + q / 64] & (1 << (q % 64)) != 0
    }
    fn code(&self, row: usize, q: usize) -> usize {
        let i = row * self.words + q / 64;
        let b = q % 64;
        ((self.x[i] >> b) & 1) as usize | (((self.z[i] >> b) & 1) as usize) << 1
    }
    fn set_code(&mut self, row: usize, q: usize, code: u8) {
        let i = row * self.words + q / 64;
        let mask = 1u64 << (q % 64);
        self.x[i] = (self.x[i] & !mask) | ((code as u64 & 1) << (q % 64));
        self.z[i] = (self.z[i] & !mask) | (((code as u64 >> 1) & 1) << (q % 64));
    }
    pub fn one(&mut self, q: usize, map: &[u8; 4]) {
        for row in 0..2 * self.n {
            let out = map[self.code(row, q)];
            self.signs[row] ^= out & 4 != 0;
            self.set_code(row, q, out);
        }
    }
    pub fn two(&mut self, a: usize, b: usize, map: &[u8; 16]) {
        for row in 0..2 * self.n {
            let out = map[self.code(row, a) | (self.code(row, b) << 2)];
            self.signs[row] ^= out & 16 != 0;
            self.set_code(row, a, out & 3);
            self.set_code(row, b, (out >> 2) & 3);
        }
    }
    pub fn cx(&mut self, a: usize, b: usize) {
        for row in 0..2 * self.n {
            let ca = self.code(row, a);
            let cb = self.code(row, b);
            self.signs[row] ^= (ca & 1 != 0) && (cb & 2 != 0) && (((cb & 1) ^ (ca >> 1) ^ 1) != 0);
            self.set_code(row, a, (ca ^ (cb & 2)) as u8);
            self.set_code(row, b, (cb ^ (ca & 1)) as u8);
        }
    }
    pub fn swap(&mut self, a: usize, b: usize) {
        for row in 0..2 * self.n {
            let ca = self.code(row, a);
            let cb = self.code(row, b);
            self.set_code(row, a, cb as u8);
            self.set_code(row, b, ca as u8);
        }
    }
    fn clear_row(&mut self, row: usize) {
        self.x[row * self.words..(row + 1) * self.words].fill(0);
        self.z[row * self.words..(row + 1) * self.words].fill(0);
        self.signs[row] = false;
    }
    fn row_add(&mut self, source: usize, target: usize) {
        let mut phase = 2 * (self.signs[source] as i32 + self.signs[target] as i32);
        for w in 0..self.words {
            let s = source * self.words + w;
            let t = target * self.words + w;
            phase =
                (phase + product_phase(self.x[s], self.z[s], self.x[t], self.z[t])).rem_euclid(4);
            self.x[t] ^= self.x[s];
            self.z[t] ^= self.z[s];
        }
        debug_assert!(
            phase == 0 || phase == 2,
            "Only commuting Hermitian rows may be multiplied"
        );
        self.signs[target] = phase == 2;
    }
    /// None is a deterministic measurement; Some(row) needs a random outcome.
    pub fn pivot(&self, q: usize) -> Option<usize> {
        (self.n..2 * self.n).find(|&r| self.xbit(r, q))
    }
    pub fn measure(&mut self, q: usize, rng: &mut impl Rng) -> (bool, bool) {
        let pivot = self.pivot(q);
        let random = pivot.is_some();
        let outcome = if random { rng.gen() } else { false };
        (self.project(q, pivot, outcome), random)
    }
    pub fn project(&mut self, q: usize, pivot: Option<usize>, outcome: bool) -> bool {
        if let Some(p) = pivot {
            // The paired destabilizer anticommutes with p and is overwritten below.
            for r in 0..2 * self.n {
                if r != p && r != p - self.n && self.xbit(r, q) {
                    self.row_add(p, r);
                }
            }
            let dest = p - self.n;
            self.x
                .copy_within(p * self.words..(p + 1) * self.words, dest * self.words);
            self.z
                .copy_within(p * self.words..(p + 1) * self.words, dest * self.words);
            self.signs[dest] = self.signs[p];
            self.clear_row(p);
            self.z[p * self.words + q / 64] = 1 << (q % 64);
            self.signs[p] = outcome;
            outcome
        } else {
            let scratch = 2 * self.n;
            self.clear_row(scratch);
            for r in 0..self.n {
                if self.xbit(r, q) {
                    self.row_add(r + self.n, scratch);
                }
            }
            self.signs[scratch]
        }
    }
    pub fn expectation(&mut self, x: &[u64], z: &[u64]) -> i8 {
        let anti = |state: &Self, row: usize| -> bool {
            (0..state.words).fold(0u32, |p, w| {
                p ^ ((x[w] & state.z[row * state.words + w])
                    ^ (z[w] & state.x[row * state.words + w]))
                    .count_ones()
            }) & 1
                != 0
        };
        if (self.n..2 * self.n).any(|r| anti(self, r)) {
            return 0;
        }
        let scratch = 2 * self.n;
        self.clear_row(scratch);
        for r in 0..self.n {
            if anti(self, r) {
                self.row_add(r + self.n, scratch);
            }
        }
        if self.signs[scratch] {
            -1
        } else {
            1
        }
    }
    pub fn generators(&self) -> Vec<String> {
        (self.n..2 * self.n)
            .map(|r| {
                let mut s = String::with_capacity(self.n + 1);
                s.push(if self.signs[r] { '-' } else { '+' });
                for q in (0..self.n).rev() {
                    s.push(['I', 'X', 'Z', 'Y'][self.code(r, q)]);
                }
                s
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pauli_phase_table_is_complete() {
        // Order I,X,Z,Y; entry is exponent of i for source * target.
        let expected = [[0, 0, 0, 0], [0, 0, -1, 1], [0, 1, 0, -1], [0, -1, 1, 0]];
        for (a, row) in expected.iter().enumerate() {
            for (b, &phase) in row.iter().enumerate() {
                assert_eq!(
                    product_phase(
                        (a & 1) as u64,
                        (a >> 1) as u64,
                        (b & 1) as u64,
                        (b >> 1) as u64
                    )
                    .rem_euclid(4),
                    (phase as i32).rem_euclid(4)
                );
            }
        }
    }
    #[test]
    fn symplectic_invariants_survive_measurements_across_words() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(18);
        let mut s = Tableau::new(70, 1 << 20).unwrap();
        for _ in 0..150 {
            let a = rng.gen_range(0..70);
            let b = (a + rng.gen_range(1..70)) % 70;
            s.one(a, &[0, 2, 1, 7]);
            s.one(b, &[0, 3, 2, 5]);
            s.cx(a, b);
            s.measure(rng.gen_range(0..70), &mut rng);
            for i in 0..140 {
                for j in 0..140 {
                    let p = (0..s.words).fold(0, |p, w| {
                        p ^ ((s.x[i * s.words + w] & s.z[j * s.words + w])
                            ^ (s.z[i * s.words + w] & s.x[j * s.words + w]))
                            .count_ones()
                    }) & 1;
                    assert_eq!(p == 1, i.abs_diff(j) == 70);
                }
            }
        }
    }
}
