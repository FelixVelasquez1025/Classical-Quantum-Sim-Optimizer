use super::model::{BlockPool, C};
use crate::{
    gates,
    monolithic::statevector::execution::Operation,
    types::{gate_matrix_1q, Condition, Instruction},
};
use rand::Rng;
use std::collections::{BTreeMap, HashMap};

pub(super) enum Op {
    Gate { wires: Vec<usize>, gate: Operation },
    Measure(usize, usize),
    Reset(usize),
    Conditional(Condition, Box<Op>),
    Noop,
}
impl Op {
    fn compile(inst: &Instruction) -> Result<Self, String> {
        Ok(match inst {
            Instruction::Measure { qubit, cbit } => Self::Measure(*qubit, *cbit),
            Instruction::Reset { qubit } => Self::Reset(*qubit),
            Instruction::Conditional { condition, op } => {
                Self::Conditional(condition.clone(), Box::new(Self::compile(op)?))
            }
            Instruction::Barrier => Self::Noop,
            Instruction::Gate { name, .. } if name.eq_ignore_ascii_case("remote_barrier") => {
                Self::Noop
            }
            _ => {
                let wires = inst.qubits();
                let mut local = inst.clone();
                local.remap_qubits(&|q| wires.iter().position(|&w| w == q).unwrap());
                let gate = Operation::compile(&local)?;
                Self::Gate { wires, gate }
            }
        })
    }
    pub fn unitary(&self) -> bool {
        matches!(self, Self::Gate { .. } | Self::Noop)
    }
    pub fn terminal(&self) -> bool {
        matches!(self, Self::Measure(..) | Self::Noop)
    }
    pub fn run(
        &self,
        pool: &mut BlockPool,
        cbits: &mut HashMap<usize, i32>,
        rng: &mut impl Rng,
        parallel: bool,
    ) -> Result<(), String> {
        match self {
            Self::Conditional(c, op) => {
                let mut actual = 0u64;
                for bit in 0..c.creg_size {
                    actual |= (*cbits.get(&(c.creg_base + bit)).unwrap_or(&0) as u64) << bit;
                }
                if actual == c.creg_value {
                    op.run(pool, cbits, rng, parallel)?;
                }
            }
            Self::Measure(q, c) => {
                cbits.insert(*c, pool.measure(*q, rng, parallel)? as i32);
            }
            Self::Reset(q) => {
                if pool.measure(*q, rng, parallel)? == 1 {
                    pool.one(*q, &gates::X, &[], parallel)?;
                }
            }
            Self::Gate { wires, gate } => pool.gate(wires, gate, parallel)?,
            Self::Noop => {}
        }
        Ok(())
    }
}
pub(super) fn compile(instructions: &[Instruction]) -> Result<Vec<Op>, String> {
    let mut pending: BTreeMap<usize, [[C; 2]; 2]> = BTreeMap::new();
    let mut out = Vec::new();
    let identity = [
        [C::new(1., 0.), C::default()],
        [C::default(), C::new(1., 0.)],
    ];
    for inst in instructions {
        if let Some((q, m)) = gate_matrix_1q(inst) {
            let old = pending.entry(q).or_insert(identity);
            *old = std::array::from_fn(|i| {
                std::array::from_fn(|j| (0..2).map(|k| m[i][k] * old[k][j]).sum())
            });
        } else {
            for (q, m) in std::mem::take(&mut pending) {
                if m != identity {
                    out.push(Op::Gate {
                        wires: vec![q],
                        gate: Operation::One {
                            target: 0,
                            matrix: m,
                            controls: vec![],
                        },
                    });
                }
            }
            let op = Op::compile(inst)?;
            if !matches!(op, Op::Noop) {
                out.push(op);
            }
        }
    }
    for (q, m) in pending {
        if m != identity {
            out.push(Op::Gate {
                wires: vec![q],
                gate: Operation::One {
                    target: 0,
                    matrix: m,
                    controls: vec![],
                },
            });
        }
    }
    Ok(out)
}
