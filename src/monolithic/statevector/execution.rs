//! Compile gates once and share their semantics between state and shot execution.
use std::collections::BTreeMap;
use std::time::Instant;

use num_complex::Complex64 as C;
use rand::Rng;

use crate::types::{gate_matrix_1q, Condition, Instruction};
use crate::{engine, gates};

#[derive(Default)]
pub(super) struct ProfileAcc {
    pub oq_calls: u64,
    pub oq_time: f64,
    pub nq_calls: u64,
    pub nq_time: f64,
    pub mq_calls: u64,
    pub mq_time: f64,
}

pub(crate) enum Operation {
    One {
        target: usize,
        matrix: gates::Mat2,
        controls: Vec<(usize, bool)>,
    },
    X {
        target: usize,
        controls: Vec<(usize, bool)>,
    },
    Swap {
        a: usize,
        b: usize,
        controls: Vec<(usize, bool)>,
    },
    Diagonal {
        qubits: Vec<usize>,
        diagonal: Vec<C>,
    },
    Dense {
        layout: Box<engine::DenseLayout>,
        matrix: Vec<Vec<C>>,
    },
    Measure {
        qubit: usize,
        cbit: usize,
    },
    Reset {
        qubit: usize,
    },
    Conditional {
        condition: Condition,
        op: Box<Operation>,
    },
    Noop,
}

impl Operation {
    pub fn is_unitary(&self) -> bool {
        !matches!(
            self,
            Self::Measure { .. } | Self::Reset { .. } | Self::Conditional { .. }
        )
    }

    fn one(target: usize, matrix: gates::Mat2, controls: Vec<(usize, bool)>) -> Self {
        if matrix == gates::X {
            Self::X { target, controls }
        } else if matrix[0][1] == C::default() && matrix[1][0] == C::default() {
            let mut qubits: Vec<_> = controls.iter().map(|&(q, _)| q).collect();
            qubits.push(target);
            let mut diagonal = vec![C::new(1.0, 0.0); 1 << qubits.len()];
            let index = controls
                .iter()
                .fold(0, |acc, &(_, on)| (acc << 1) | usize::from(on))
                << 1;
            diagonal[index] = matrix[0][0];
            diagonal[index + 1] = matrix[1][1];
            Self::Diagonal { qubits, diagonal }
        } else {
            Self::One {
                target,
                matrix,
                controls,
            }
        }
    }

    fn dense<const N: usize>(qubits: Vec<usize>, matrix: [[C; N]; N]) -> Self {
        // Preserve diagonal structure for RZZ and controlled phases.
        if matrix.iter().enumerate().all(|(i, row)| {
            row.iter()
                .enumerate()
                .all(|(j, &v)| i == j || v == C::default())
        }) {
            Self::Diagonal {
                qubits,
                diagonal: (0..N).map(|i| matrix[i][i]).collect(),
            }
        } else {
            Self::Dense {
                layout: Box::new(engine::DenseLayout::new(&qubits)),
                matrix: matrix.iter().map(|r| r.to_vec()).collect(),
            }
        }
    }

    pub(crate) fn compile(inst: &Instruction) -> Result<Self, String> {
        use Instruction::*;
        if let Some((q, matrix)) = gate_matrix_1q(inst) {
            return Ok(Self::one(q, matrix, vec![]));
        }
        let controlled = |c, t, m| Self::one(t, m, vec![(c, true)]);
        Ok(match inst {
            Cx { control, target } => controlled(*control, *target, gates::X),
            Cy { control, target } => controlled(*control, *target, gates::Y),
            Cz { control, target } => controlled(*control, *target, gates::Z),
            Ch { control, target } => controlled(*control, *target, gates::h()),
            Csx { control, target } => controlled(*control, *target, gates::sx()),
            Crx {
                control,
                target,
                theta,
            } => controlled(*control, *target, gates::rx(*theta)),
            Cry {
                control,
                target,
                theta,
            } => controlled(*control, *target, gates::ry(*theta)),
            Crz {
                control,
                target,
                lam,
            } => controlled(*control, *target, gates::rz(*lam)),
            Cu1 {
                control,
                target,
                lam,
            }
            | Cp {
                control,
                target,
                lam,
            } => controlled(*control, *target, gates::p(*lam)),
            Cu3 {
                control,
                target,
                theta,
                phi,
                lam,
            } => controlled(*control, *target, gates::u3(*theta, *phi, *lam)),
            Cu {
                control,
                target,
                theta,
                phi,
                lam,
                gamma,
            } => {
                let mut m = gates::u3(*theta, *phi, *lam);
                let phase = C::new(gamma.cos(), gamma.sin());
                for row in &mut m {
                    for v in row {
                        *v *= phase;
                    }
                }
                controlled(*control, *target, m)
            }
            Swap { a, b } => Self::Swap {
                a: *a,
                b: *b,
                controls: vec![],
            },
            Cswap {
                control,
                target1,
                target2,
            } => Self::Swap {
                a: *target1,
                b: *target2,
                controls: vec![(*control, true)],
            },
            Ccx {
                control1,
                control2,
                target,
            } => Self::X {
                target: *target,
                controls: vec![(*control1, true), (*control2, true)],
            },
            C3x {
                control1,
                control2,
                control3,
                target,
            } => Self::X {
                target: *target,
                controls: vec![(*control1, true), (*control2, true), (*control3, true)],
            },
            C4x {
                control1,
                control2,
                control3,
                control4,
                target,
            } => Self::X {
                target: *target,
                controls: vec![
                    (*control1, true),
                    (*control2, true),
                    (*control3, true),
                    (*control4, true),
                ],
            },
            C3sqrtx {
                control1,
                control2,
                control3,
                target,
            } => Self::one(
                *target,
                gates::sx(),
                vec![(*control1, true), (*control2, true), (*control3, true)],
            ),
            Rxx { a, b, theta } => Self::dense(vec![*a, *b], gates::rxx(*theta)),
            Rzz { a, b, theta } => Self::dense(vec![*a, *b], gates::rzz(*theta)),
            Rccx {
                control1,
                control2,
                target,
            } => Self::dense(vec![*control1, *control2, *target], gates::rccx()),
            Rc3x {
                control1,
                control2,
                control3,
                target,
            } => Self::dense(
                vec![*control1, *control2, *control3, *target],
                gates::rc3x(),
            ),
            Measure { qubit, cbit } => Self::Measure {
                qubit: *qubit,
                cbit: *cbit,
            },
            Reset { qubit } => Self::Reset { qubit: *qubit },
            Conditional { condition, op } => Self::Conditional {
                condition: condition.clone(),
                op: Box::new(Self::compile(op)?),
            },
            Barrier => Self::Noop,
            Gate {
                name,
                qubits,
                params,
            } => match name.to_lowercase().as_str() {
                "remote_link_phi_plus" | "remote_epr" | "epr" => {
                    Self::dense(qubits.clone(), gates::phi_plus())
                }
                "remote_link_psi_plus" => Self::dense(qubits.clone(), gates::psi_plus()),
                "remote_link_psi_minus" => Self::dense(qubits.clone(), gates::psi_minus()),
                "nonlocal_cz" | "remote_cz" => controlled(qubits[0], qubits[1], gates::Z),
                "remote_cx" => controlled(qubits[0], qubits[1], gates::X),
                "remote_rzz" => Self::dense(qubits.clone(), gates::rzz(params[0])),
                "remote_cu1" => controlled(qubits[0], qubits[1], gates::p(params[0])),
                "remote_barrier" => Self::Noop,
                _ => {
                    return Err(format!(
                        "Unsupported generic gate {name:?}; decompose it before simulating"
                    ))
                }
            },
            _ => return Err(format!("Unsupported instruction {inst:?}")),
        })
    }

    pub(super) fn execute(
        &self,
        state: &mut [C],
        n: usize,
        cbits: &mut [u8],
        rng: &mut impl Rng,
        parallel: bool,
        profile: &mut Option<ProfileAcc>,
    ) {
        // Conditionals recurse so profiling counts the actual executed operation.
        if let Self::Conditional { condition, op } = self {
            let mut actual = 0u64;
            for bit in 0..condition.creg_size {
                actual |= (cbits[condition.creg_base + bit] as u64) << bit;
            }
            if actual == condition.creg_value {
                op.execute(state, n, cbits, rng, parallel, profile);
            }
            return;
        }
        let start = profile.as_ref().map(|_| Instant::now());
        let category = match self {
            Self::One {
                target,
                matrix,
                controls,
            } => {
                if parallel {
                    engine::apply_one_qubit(state, matrix, *target, n, controls);
                } else {
                    engine::apply_one_qubit_seq(state, matrix, *target, n, controls);
                }
                usize::from(!controls.is_empty())
            }
            Self::X { target, controls } => {
                engine::apply_x(state, *target, controls, parallel);
                usize::from(!controls.is_empty())
            }
            Self::Swap { a, b, controls } => {
                engine::apply_swap(state, *a, *b, controls, parallel);
                1
            }
            Self::Diagonal { qubits, diagonal } => {
                engine::apply_diagonal(state, qubits, diagonal, parallel);
                usize::from(qubits.len() > 1)
            }
            Self::Dense { layout, matrix } => {
                layout.apply(state, matrix, n, parallel);
                1
            }
            Self::Measure { qubit, cbit } => {
                cbits[*cbit] = if parallel {
                    engine::measure_qubit(state, *qubit, n, rng)
                } else {
                    engine::measure_qubit_seq(state, *qubit, n, rng)
                };
                2
            }
            Self::Reset { qubit } => {
                let bit = if parallel {
                    engine::measure_qubit(state, *qubit, n, rng)
                } else {
                    engine::measure_qubit_seq(state, *qubit, n, rng)
                };
                if bit == 1 {
                    engine::apply_x(state, *qubit, &[], parallel);
                }
                2
            }
            Self::Noop => return,
            Self::Conditional { .. } => unreachable!(),
        };
        if let (Some(acc), Some(start)) = (profile.as_mut(), start) {
            let elapsed = start.elapsed().as_secs_f64();
            match category {
                0 => {
                    acc.oq_calls += 1;
                    acc.oq_time += elapsed;
                }
                1 => {
                    acc.nq_calls += 1;
                    acc.nq_time += elapsed;
                }
                _ => {
                    acc.mq_calls += 1;
                    acc.mq_time += elapsed;
                }
            }
        }
    }
}

pub(super) fn compile(instructions: &[Instruction]) -> Result<Vec<Operation>, String> {
    let mut pending: BTreeMap<usize, gates::Mat2> = BTreeMap::new();
    let mut result = Vec::new();
    let identity = [
        [C::new(1.0, 0.0), C::default()],
        [C::default(), C::new(1.0, 0.0)],
    ];
    let flush = |q, pending: &mut BTreeMap<usize, gates::Mat2>, result: &mut Vec<Operation>| {
        if let Some(m) = pending.remove(&q) {
            // Never discard a small but nonzero rotation using an approximate identity test.
            if m != identity {
                result.push(Operation::one(q, m, vec![]));
            }
        }
    };
    for inst in instructions {
        if let Some((q, m)) = gate_matrix_1q(inst) {
            let old = pending.get(&q).copied().unwrap_or(identity);
            let mut product = [[C::default(); 2]; 2];
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        product[i][j] += m[i][k] * old[k][j];
                    }
                }
            }
            pending.insert(q, product);
        } else {
            // Flush all pending gates at a measurement boundary so a terminal
            // measurement suffix remains terminal after compilation.
            let touched = if matches!(inst, Instruction::Barrier | Instruction::Measure { .. }) {
                pending.keys().copied().collect()
            } else {
                inst.qubits()
            };
            for q in touched {
                flush(q, &mut pending, &mut result);
            }
            let op = Operation::compile(inst)?;
            if !matches!(op, Operation::Noop) {
                result.push(op);
            }
        }
    }
    for q in pending.keys().copied().collect::<Vec<_>>() {
        flush(q, &mut pending, &mut result);
    }
    Ok(result)
}

pub(super) fn execute(
    ops: &[Operation],
    state: &mut [C],
    n: usize,
    cbits: &mut [u8],
    rng: &mut impl Rng,
    parallel: bool,
    profile: &mut Option<ProfileAcc>,
) {
    for op in ops {
        op.execute(state, n, cbits, rng, parallel, profile);
    }
}
