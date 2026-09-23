use super::engine::Tableau;
use crate::{
    gates,
    types::{Condition, Instruction},
};
use num_complex::Complex64 as C;
use rand::Rng;
use serde::Serialize;
use std::{
    collections::{BTreeMap, HashMap},
    f64::consts::FRAC_PI_2,
    time::Instant,
};

pub(super) const ID: [u8; 4] = [0, 1, 2, 3];
pub(super) const X_MAP: [u8; 4] = [0, 1, 6, 7];
#[derive(Default, Clone, Serialize)]
pub(super) struct Stats {
    pub gate_calls: u64,
    pub measure_calls: u64,
    pub random_measurements: u64,
    pub deterministic_measurements: u64,
    pub measure_time: f64,
}
impl Stats {
    pub fn add(&mut self, s: &Self) {
        self.gate_calls += s.gate_calls;
        self.measure_calls += s.measure_calls;
        self.random_measurements += s.random_measurements;
        self.deterministic_measurements += s.deterministic_measurements;
        self.measure_time += s.measure_time;
    }
    pub fn measure(
        &mut self,
        state: &mut Tableau,
        q: usize,
        rng: &mut impl Rng,
        profile: bool,
    ) -> bool {
        let start = profile.then(Instant::now);
        let (out, random) = state.measure(q, rng);
        self.record(random);
        if let Some(start) = start {
            self.measure_time += start.elapsed().as_secs_f64();
        }
        out
    }
    pub fn record(&mut self, random: bool) {
        self.measure_calls += 1;
        if random {
            self.random_measurements += 1;
        } else {
            self.deterministic_measurements += 1;
        }
    }
}
#[derive(Clone)]
pub(super) enum Op {
    One(usize, [u8; 4]),
    Two(usize, usize, [u8; 16]),
    Cx(usize, usize),
    Swap(usize, usize),
    Measure(usize, usize),
    Reset(usize),
    Conditional(Condition, Box<Op>),
    Noop,
}
impl Op {
    pub fn unitary(&self) -> bool {
        matches!(
            self,
            Self::One(..) | Self::Two(..) | Self::Cx(..) | Self::Swap(..) | Self::Noop
        )
    }
    pub fn terminal(&self) -> bool {
        matches!(self, Self::Measure(..) | Self::Noop)
    }
    pub fn run(
        &self,
        state: &mut Tableau,
        cbits: &mut HashMap<usize, i32>,
        rng: &mut impl Rng,
        stats: &mut Stats,
        profile: bool,
    ) {
        match self {
            Self::One(q, m) => state.one(*q, m),
            Self::Two(a, b, m) => state.two(*a, *b, m),
            Self::Cx(a, b) => state.cx(*a, *b),
            Self::Swap(a, b) => state.swap(*a, *b),
            Self::Measure(q, c) => {
                let v = stats.measure(state, *q, rng, profile);
                cbits.insert(*c, v as i32);
            }
            Self::Reset(q) => {
                if stats.measure(state, *q, rng, profile) {
                    state.one(*q, &X_MAP);
                }
            }
            Self::Conditional(c, op) => {
                let actual = (0..c.creg_size).fold(0u64, |v, b| {
                    v | ((*cbits.get(&(c.creg_base + b)).unwrap_or(&0) as u64) << b)
                });
                if actual == c.creg_value {
                    op.run(state, cbits, rng, stats, profile);
                }
            }
            Self::Noop => {}
        }
        if self.unitary() && !matches!(self, Self::Noop) {
            stats.gate_calls += 1;
        }
    }
}
fn unsupported() -> String {
    "Instruction is not a supported Clifford operation; use another backend or decompose into supported Clifford gates".into()
}
// The default admits only the exact f64 expression k*(pi/2). Positive tolerance
// explicitly opts into snapping. Reduce modulo 4*pi, preserving controlled phases.
fn angle(value: f64, tolerance: f64) -> Result<f64, String> {
    let k = (value / FRAC_PI_2).round();
    if !value.is_finite()
        || k.abs() > ((1u64 << 40) as f64)
        || (value - k * FRAC_PI_2).abs() > tolerance
    {
        return Err(unsupported());
    }
    Ok(k.rem_euclid(8.0) * FRAC_PI_2)
}
fn pauli(code: usize) -> [[C; 2]; 2] {
    match code {
        0 => [
            [C::new(1., 0.), C::default()],
            [C::default(), C::new(1., 0.)],
        ],
        1 => gates::X,
        2 => gates::Z,
        _ => gates::Y,
    }
}
fn conjugate<const N: usize>(u: &[[C; N]; N], p: &[[C; N]; N]) -> [[C; N]; N] {
    let left: [[C; N]; N] =
        std::array::from_fn(|i| std::array::from_fn(|j| (0..N).map(|k| u[i][k] * p[k][j]).sum()));
    std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..N).map(|k| left[i][k] * u[j][k].conj()).sum())
    })
}
fn signed_match<const N: usize>(a: &[[C; N]; N], b: &[[C; N]; N]) -> Option<bool> {
    for negative in [false, true] {
        let sign = if negative { -1. } else { 1. };
        if (0..N).all(|i| (0..N).all(|j| (a[i][j] - sign * b[i][j]).norm() < 1e-12)) {
            return Some(negative);
        }
    }
    None
}
fn one_map(m: &[[C; 2]; 2]) -> Result<[u8; 4], String> {
    let mut map = [0; 4];
    for (input, out) in map.iter_mut().enumerate() {
        let p = conjugate(m, &pauli(input));
        *out = (0..4)
            .find_map(|j| signed_match(&p, &pauli(j)).map(|s| j as u8 | ((s as u8) << 2)))
            .ok_or_else(unsupported)?;
    }
    Ok(map)
}
fn two_map(m: &[[C; 4]; 4]) -> Result<[u8; 16], String> {
    let matrices: [[[C; 4]; 4]; 16] = std::array::from_fn(|code| {
        let a = pauli(code & 3);
        let b = pauli(code >> 2);
        std::array::from_fn(|i| std::array::from_fn(|j| a[i / 2][j / 2] * b[i % 2][j % 2]))
    });
    let mut map = [0; 16];
    for (input, out) in map.iter_mut().enumerate() {
        let p = conjugate(m, &matrices[input]);
        *out = (0..16)
            .find_map(|j| signed_match(&p, &matrices[j]).map(|s| j as u8 | ((s as u8) << 4)))
            .ok_or_else(unsupported)?;
    }
    Ok(map)
}
fn compile_one(inst: &Instruction, tol: f64) -> Result<Op, String> {
    use Instruction::*;
    let a = |v| angle(v, tol);
    let single = match inst {
        Id { qubit } | U0 { qubit } => Some((*qubit, ID)),
        X { qubit } => Some((*qubit, X_MAP)),
        Y { qubit } => Some((*qubit, [0, 5, 6, 3])),
        Z { qubit } => Some((*qubit, [0, 5, 2, 7])),
        H { qubit } => Some((*qubit, [0, 2, 1, 7])),
        S { qubit } => Some((*qubit, [0, 3, 2, 5])),
        Sdg { qubit } => Some((*qubit, [0, 7, 2, 1])),
        Sx { qubit } => Some((*qubit, [0, 1, 7, 2])),
        Sxdg { qubit } => Some((*qubit, [0, 1, 3, 6])),
        Rx { qubit, theta } => Some((*qubit, one_map(&gates::rx(a(*theta)?))?)),
        Ry { qubit, theta } => Some((*qubit, one_map(&gates::ry(a(*theta)?))?)),
        Rz { qubit, phi } => Some((*qubit, one_map(&gates::rz(a(*phi)?))?)),
        P { qubit, lam } | U1 { qubit, lam } => Some((*qubit, one_map(&gates::p(a(*lam)?))?)),
        U {
            qubit,
            theta,
            phi,
            lam,
        }
        | U3 {
            qubit,
            theta,
            phi,
            lam,
        } => Some((*qubit, one_map(&gates::u3(a(*theta)?, a(*phi)?, a(*lam)?))?)),
        U2 { qubit, phi, lam } => Some((*qubit, one_map(&gates::u2(a(*phi)?, a(*lam)?))?)),
        _ => None,
    };
    if let Some((q, m)) = single {
        return Ok(Op::One(q, m));
    }
    let two = |a, b, m| Ok(Op::Two(a, b, two_map(&m)?));
    match inst {
        Cx { control, target } => Ok(Op::Cx(*control, *target)),
        Swap { a, b } => Ok(Op::Swap(*a, *b)),
        Cz { control, target } => two(*control, *target, gates::cz()),
        Cy { control, target } => two(*control, *target, gates::cy()),
        Crx {
            control,
            target,
            theta,
        } => two(*control, *target, gates::crx(a(*theta)?)),
        Cry {
            control,
            target,
            theta,
        } => two(*control, *target, gates::cry(a(*theta)?)),
        Crz {
            control,
            target,
            lam,
        } => two(*control, *target, gates::crz(a(*lam)?)),
        Cp {
            control,
            target,
            lam,
        }
        | Cu1 {
            control,
            target,
            lam,
        } => two(*control, *target, gates::cu1(a(*lam)?)),
        Cu3 {
            control,
            target,
            theta,
            phi,
            lam,
        } => two(
            *control,
            *target,
            gates::cu3(a(*theta)?, a(*phi)?, a(*lam)?),
        ),
        Cu {
            control,
            target,
            theta,
            phi,
            lam,
            gamma,
        } => two(
            *control,
            *target,
            gates::cu(a(*theta)?, a(*phi)?, a(*lam)?, a(*gamma)?),
        ),
        Rxx { a: x, b, theta } => two(*x, *b, gates::rxx(a(*theta)?)),
        Rzz { a: x, b, theta } => two(*x, *b, gates::rzz(a(*theta)?)),
        Gate {
            name,
            params,
            qubits,
        } => match name.to_lowercase().as_str() {
            "remote_cx" => Ok(Op::Cx(qubits[0], qubits[1])),
            "remote_cz" | "nonlocal_cz" => two(qubits[0], qubits[1], gates::cz()),
            "remote_link_phi_plus" | "remote_epr" | "epr" => {
                two(qubits[0], qubits[1], gates::phi_plus())
            }
            "remote_link_psi_plus" => two(qubits[0], qubits[1], gates::psi_plus()),
            "remote_link_psi_minus" => two(qubits[0], qubits[1], gates::psi_minus()),
            "remote_rzz" => two(qubits[0], qubits[1], gates::rzz(a(params[0])?)),
            "remote_cu1" => two(qubits[0], qubits[1], gates::cu1(a(params[0])?)),
            "remote_barrier" => Ok(Op::Noop),
            _ => Err(unsupported()),
        },
        Measure { qubit, cbit } => Ok(Op::Measure(*qubit, *cbit)),
        Reset { qubit } => Ok(Op::Reset(*qubit)),
        Conditional { condition, op } => Ok(Op::Conditional(
            condition.clone(),
            Box::new(compile_one(op, tol)?),
        )),
        Barrier => Ok(Op::Noop),
        _ => Err(unsupported()),
    }
}
pub(super) fn compile(instructions: &[Instruction], tol: f64) -> Result<Vec<Op>, String> {
    let mut out = Vec::new();
    let mut pending = BTreeMap::<usize, [u8; 4]>::new();
    let flush = |pending: &mut BTreeMap<usize, [u8; 4]>, out: &mut Vec<Op>| {
        for (q, m) in std::mem::take(pending) {
            if m != ID {
                out.push(Op::One(q, m));
            }
        }
    };
    for (i, inst) in instructions.iter().enumerate() {
        match compile_one(inst, tol).map_err(|e| format!("Instruction {i}: {e}"))? {
            Op::One(q, m) => {
                let old = pending.entry(q).or_insert(ID);
                *old = std::array::from_fn(|i| m[(old[i] & 3) as usize] ^ (old[i] & 4));
            }
            op => {
                flush(&mut pending, &mut out);
                out.push(op);
            }
        }
    }
    flush(&mut pending, &mut out);
    Ok(out)
}
