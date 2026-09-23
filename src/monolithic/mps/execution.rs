use super::engine::Mps;
use crate::{
    gates,
    types::{gate_matrix_1q, Condition, Instruction},
};
use num_complex::Complex64 as C;
use rand::Rng;
use std::collections::{BTreeMap, HashMap};

pub(super) enum Op {
    One(usize, [[C; 2]; 2]),
    Two(usize, usize, [[C; 4]; 4]),
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
            Self::One(..) | Self::Two(..) | Self::Swap(..) | Self::Noop
        )
    }
    pub fn terminal(&self) -> bool {
        matches!(self, Self::Measure(..) | Self::Noop)
    }
    fn compile(inst: &Instruction) -> Result<Self, String> {
        if let Some((q, m)) = gate_matrix_1q(inst) {
            return Ok(Self::One(q, m));
        }
        Ok(match inst {
            Instruction::Cx { control, target } => Op::Two(*control, *target, gates::cnot()),
            Instruction::Cz { control, target } => Op::Two(*control, *target, gates::cz()),
            Instruction::Cy { control, target } => Op::Two(*control, *target, gates::cy()),
            Instruction::Ch { control, target } => Op::Two(*control, *target, gates::ch()),
            Instruction::Swap { a, b } => Op::Swap(*a, *b),
            Instruction::Csx { control, target } => Op::Two(*control, *target, gates::csx()),
            Instruction::Crx {
                control,
                target,
                theta,
            } => Op::Two(*control, *target, gates::crx(*theta)),
            Instruction::Cry {
                control,
                target,
                theta,
            } => Op::Two(*control, *target, gates::cry(*theta)),
            Instruction::Crz {
                control,
                target,
                lam,
            } => Op::Two(*control, *target, gates::crz(*lam)),
            Instruction::Cu1 {
                control,
                target,
                lam,
            }
            | Instruction::Cp {
                control,
                target,
                lam,
            } => Op::Two(*control, *target, gates::cu1(*lam)),
            Instruction::Cu3 {
                control,
                target,
                theta,
                phi,
                lam,
            } => Op::Two(*control, *target, gates::cu3(*theta, *phi, *lam)),
            Instruction::Cu {
                control,
                target,
                theta,
                phi,
                lam,
                gamma,
            } => Op::Two(*control, *target, gates::cu(*theta, *phi, *lam, *gamma)),
            Instruction::Rxx { a, b, theta } => Op::Two(*a, *b, gates::rxx(*theta)),
            Instruction::Rzz { a, b, theta } => Op::Two(*a, *b, gates::rzz(*theta)),
            Instruction::Gate {
                name,
                qubits,
                params,
            } => match name.to_lowercase().as_str() {
                "remote_link_phi_plus" | "remote_epr" | "epr" => {
                    Op::Two(qubits[0], qubits[1], gates::phi_plus())
                }
                "remote_link_psi_minus" => Op::Two(qubits[0], qubits[1], gates::psi_minus()),
                "remote_link_psi_plus" => Op::Two(qubits[0], qubits[1], gates::psi_plus()),
                "nonlocal_cz" | "remote_cz" => Op::Two(qubits[0], qubits[1], gates::cz()),
                "remote_cx" => Op::Two(qubits[0], qubits[1], gates::cnot()),
                "remote_cu1" => Op::Two(qubits[0], qubits[1], gates::cu1(params[0])),
                "remote_rzz" => Op::Two(qubits[0], qubits[1], gates::rzz(params[0])),
                "remote_barrier" => Op::Noop,
                other => return Err(format!("Unsupported MPS gate {other:?}")),
            },
            Instruction::Measure { qubit, cbit } => Op::Measure(*qubit, *cbit),
            Instruction::Reset { qubit } => Op::Reset(*qubit),
            Instruction::Conditional { condition, op } => {
                Op::Conditional(condition.clone(), Box::new(Self::compile(op)?))
            }
            Instruction::Barrier => Op::Noop,
            _ => return Err(
                "MPS supports one- and two-qubit gates; decompose larger gates before simulation"
                    .into(),
            ),
        })
    }
    pub fn run(
        &self,
        state: &mut Mps,
        cbits: &mut HashMap<usize, i32>,
        rng: &mut impl Rng,
    ) -> Result<(), String> {
        match self {
            Self::One(q, m) => state.apply_1q(*q, m),
            Self::Two(a, b, m) => state.apply_2q(*a, *b, m)?,
            Self::Swap(a, b) => state.swap(*a, *b),
            Self::Measure(q, c) => {
                cbits.insert(*c, state.measure(*q, rng)? as i32);
            }
            Self::Reset(q) => {
                if state.measure(*q, rng)? == 1 {
                    state.apply_1q(*q, &gates::X);
                }
            }
            Self::Conditional(c, op) => {
                let mut actual = 0u64;
                for bit in 0..c.creg_size {
                    actual |= (*cbits.get(&(c.creg_base + bit)).unwrap_or(&0) as u64) << bit;
                }
                if actual == c.creg_value {
                    op.run(state, cbits, rng)?;
                }
            }
            Self::Noop => {}
        }
        Ok(())
    }
}

// Flush at every non-single-qubit operation: simple, deterministic and safe at
// classical control boundaries. Keep tiny rotations and global phases.
pub(super) fn compile(instructions: &[Instruction]) -> Result<Vec<Op>, String> {
    let mut pending: BTreeMap<usize, [[C; 2]; 2]> = BTreeMap::new();
    let mut out = Vec::new();
    for inst in instructions {
        if let Some((q, m)) = gate_matrix_1q(inst) {
            let old = pending.entry(q).or_insert([
                [C::new(1., 0.), C::default()],
                [C::default(), C::new(1., 0.)],
            ]);
            *old = std::array::from_fn(|i| {
                std::array::from_fn(|j| (0..2).map(|k| m[i][k] * old[k][j]).sum())
            });
        } else {
            for (q, m) in std::mem::take(&mut pending) {
                out.push(Op::One(q, m));
            }
            out.push(Op::compile(inst)?);
        }
    }
    for (q, m) in pending {
        out.push(Op::One(q, m));
    }
    Ok(out)
}
