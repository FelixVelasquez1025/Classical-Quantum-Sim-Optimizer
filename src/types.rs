use num_complex::Complex64;
use serde::Deserialize;
use std::collections::HashMap;

type C = Complex64;

#[derive(Deserialize, Clone, Debug)]
pub struct Register {
    #[allow(dead_code)]
    pub name: String,
    pub size: usize,
    pub base: usize,
}

#[derive(Deserialize, Debug)]
pub struct Circuit {
    pub qregs: HashMap<String, Register>,
    #[serde(default)]
    pub cregs: HashMap<String, Register>,
    pub instructions: Vec<Instruction>,
}

impl Circuit {
    pub fn num_qubits(&self) -> usize {
        self.qregs
            .values()
            .map(|r| r.base + r.size)
            .max()
            .unwrap_or(0)
    }

    pub fn num_cbits(&self) -> usize {
        self.cregs
            .values()
            .map(|r| r.base + r.size)
            .max()
            .unwrap_or(0)
    }
}

/// Format a cbits map as a bitstring with MSB first (Qiskit convention).
/// Bits not written during simulation default to 0.
pub fn format_cbits(cbits: &HashMap<usize, i32>, num_cbits: usize) -> String {
    if num_cbits == 0 {
        return String::new();
    }
    (0..num_cbits)
        .rev()
        .map(|i| cbits.get(&i).copied().unwrap_or(0).to_string())
        .collect()
}

#[derive(Deserialize, Clone, Debug)]
pub struct Condition {
    pub creg_base: usize,
    pub creg_size: usize,
    pub creg_value: u64,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Instruction {
    // -----------------------------------------------------------------------
    // Single-qubit fixed
    // -----------------------------------------------------------------------
    Id {
        qubit: usize,
    },
    X {
        qubit: usize,
    },
    Y {
        qubit: usize,
    },
    Z {
        qubit: usize,
    },
    H {
        qubit: usize,
    },
    S {
        qubit: usize,
    },
    Sdg {
        qubit: usize,
    },
    T {
        qubit: usize,
    },
    Tdg {
        qubit: usize,
    },
    Sx {
        qubit: usize,
    },
    Sxdg {
        qubit: usize,
    },
    // -----------------------------------------------------------------------
    // Single-qubit parametric
    // -----------------------------------------------------------------------
    U3 {
        qubit: usize,
        theta: f64,
        phi: f64,
        lam: f64,
    },
    U2 {
        qubit: usize,
        phi: f64,
        lam: f64,
    },
    U1 {
        qubit: usize,
        lam: f64,
    },
    U {
        qubit: usize,
        theta: f64,
        phi: f64,
        lam: f64,
    },
    P {
        qubit: usize,
        lam: f64,
    },
    Rx {
        qubit: usize,
        theta: f64,
    },
    Ry {
        qubit: usize,
        theta: f64,
    },
    Rz {
        qubit: usize,
        phi: f64,
    },
    U0 {
        qubit: usize,
    },
    // -----------------------------------------------------------------------
    // Two-qubit fixed
    // -----------------------------------------------------------------------
    Cx {
        control: usize,
        target: usize,
    },
    Cz {
        control: usize,
        target: usize,
    },
    Cy {
        control: usize,
        target: usize,
    },
    Ch {
        control: usize,
        target: usize,
    },
    Swap {
        a: usize,
        b: usize,
    },
    Csx {
        control: usize,
        target: usize,
    },
    // -----------------------------------------------------------------------
    // Two-qubit parametric
    // -----------------------------------------------------------------------
    Crx {
        control: usize,
        target: usize,
        theta: f64,
    },
    Cry {
        control: usize,
        target: usize,
        theta: f64,
    },
    Crz {
        control: usize,
        target: usize,
        lam: f64,
    },
    Cu1 {
        control: usize,
        target: usize,
        lam: f64,
    },
    Cp {
        control: usize,
        target: usize,
        lam: f64,
    },
    Cu3 {
        control: usize,
        target: usize,
        theta: f64,
        phi: f64,
        lam: f64,
    },
    Cu {
        control: usize,
        target: usize,
        theta: f64,
        phi: f64,
        lam: f64,
        gamma: f64,
    },
    Rxx {
        a: usize,
        b: usize,
        theta: f64,
    },
    Rzz {
        a: usize,
        b: usize,
        theta: f64,
    },
    // -----------------------------------------------------------------------
    // Three-qubit
    // -----------------------------------------------------------------------
    Ccx {
        control1: usize,
        control2: usize,
        target: usize,
    },
    Cswap {
        control: usize,
        target1: usize,
        target2: usize,
    },
    Rccx {
        control1: usize,
        control2: usize,
        target: usize,
    },
    Rc3x {
        control1: usize,
        control2: usize,
        control3: usize,
        target: usize,
    },
    C3x {
        control1: usize,
        control2: usize,
        control3: usize,
        target: usize,
    },
    C3sqrtx {
        control1: usize,
        control2: usize,
        control3: usize,
        target: usize,
    },
    C4x {
        control1: usize,
        control2: usize,
        control3: usize,
        control4: usize,
        target: usize,
    },
    // -----------------------------------------------------------------------
    // Generic / cross-node
    // -----------------------------------------------------------------------
    Gate {
        name: String,
        #[allow(dead_code)]
        params: Vec<f64>,
        qubits: Vec<usize>,
    },
    // -----------------------------------------------------------------------
    // Measurement and classical control
    // -----------------------------------------------------------------------
    Measure {
        qubit: usize,
        cbit: usize,
    },
    Reset {
        qubit: usize,
    },
    Conditional {
        condition: Condition,
        op: Box<Instruction>,
    },
    // -----------------------------------------------------------------------
    // No-ops
    // -----------------------------------------------------------------------
    Barrier,
    Classical {
        #[allow(dead_code)]
        name: String,
    },
}

// ---------------------------------------------------------------------------
// Gate fusion
// ---------------------------------------------------------------------------

#[inline]
fn matmul2x2(a: [[C; 2]; 2], b: [[C; 2]; 2]) -> [[C; 2]; 2] {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

/// Shared single-qubit gate definitions used by both circuit compilers.
pub(crate) fn gate_matrix_1q(inst: &Instruction) -> Option<(usize, [[C; 2]; 2])> {
    use crate::gates;
    use Instruction::*;
    let (q, matrix) = match inst {
        Id { qubit } | U0 { qubit } => (
            *qubit,
            [
                [C::new(1.0, 0.0), C::default()],
                [C::default(), C::new(1.0, 0.0)],
            ],
        ),
        X { qubit } => (*qubit, gates::X),
        Y { qubit } => (*qubit, gates::Y),
        Z { qubit } => (*qubit, gates::Z),
        H { qubit } => (*qubit, gates::h()),
        S { qubit } => (*qubit, gates::s_gate()),
        Sdg { qubit } => (*qubit, gates::sdg()),
        T { qubit } => (*qubit, gates::t_gate()),
        Tdg { qubit } => (*qubit, gates::tdg()),
        Sx { qubit } => (*qubit, gates::sx()),
        Sxdg { qubit } => (*qubit, gates::sxdg()),
        U3 {
            qubit,
            theta,
            phi,
            lam,
        }
        | U {
            qubit,
            theta,
            phi,
            lam,
        } => (*qubit, gates::u3(*theta, *phi, *lam)),
        U2 { qubit, phi, lam } => (*qubit, gates::u2(*phi, *lam)),
        U1 { qubit, lam } | P { qubit, lam } => (*qubit, gates::p(*lam)),
        Rx { qubit, theta } => (*qubit, gates::rx(*theta)),
        Ry { qubit, theta } => (*qubit, gates::ry(*theta)),
        Rz { qubit, phi } => (*qubit, gates::rz(*phi)),
        _ => return None,
    };
    Some((q, matrix))
}

fn is_identity_2x2(m: &[[C; 2]; 2]) -> bool {
    (m[0][0] - C::new(1.0, 0.0)).norm() < 1e-10
        && m[0][1].norm() < 1e-10
        && m[1][0].norm() < 1e-10
        && (m[1][1] - C::new(1.0, 0.0)).norm() < 1e-10
}

/// Owned fused entry for the pblock simulator shot loop.
/// Unlike FusedInstruction<'a>, this is 'static and Send.
pub enum FusedPBlockEntry {
    /// Reference back into node_circuits by (node, local_idx).
    Original { node: usize, local_idx: usize },
    /// Pre-computed single-qubit matrix (no circuit reference needed).
    Fused1Q { qubit: usize, matrix: [[C; 2]; 2] },
}

/// Fuse consecutive single-qubit gates in the globally-sorted pblock entry stream.
/// `entries` is `&[(order, node, local_idx)]` already sorted by order.
pub fn fuse_pblock_entries(
    entries: &[(i64, usize, usize)],
    node_circuits: &HashMap<usize, Circuit>,
) -> Vec<FusedPBlockEntry> {
    let mut pending: HashMap<usize, [[C; 2]; 2]> = HashMap::new();
    let mut out: Vec<FusedPBlockEntry> = Vec::with_capacity(entries.len());

    let identity: [[C; 2]; 2] = [
        [C::new(1.0, 0.0), C::new(0.0, 0.0)],
        [C::new(0.0, 0.0), C::new(1.0, 0.0)],
    ];

    let flush_qubit =
        |q: usize, pending: &mut HashMap<usize, [[C; 2]; 2]>, out: &mut Vec<FusedPBlockEntry>| {
            if let Some(m) = pending.remove(&q) {
                if !is_identity_2x2(&m) {
                    out.push(FusedPBlockEntry::Fused1Q {
                        qubit: q,
                        matrix: m,
                    });
                }
            }
        };

    for &(_, node, local_idx) in entries {
        let inst = &node_circuits[&node].instructions[local_idx];
        if let Some((qubit, mat)) = gate_matrix_1q(inst) {
            let acc = pending.entry(qubit).or_insert(identity);
            *acc = matmul2x2(mat, *acc);
        } else {
            let touched = inst.qubits();
            for q in &touched {
                flush_qubit(*q, &mut pending, &mut out);
            }
            out.push(FusedPBlockEntry::Original { node, local_idx });
        }
    }

    let remaining: Vec<usize> = pending.keys().copied().collect();
    for q in remaining {
        flush_qubit(q, &mut pending, &mut out);
    }
    out
}

impl Instruction {
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            Instruction::X { qubit }
            | Instruction::Y { qubit }
            | Instruction::Z { qubit }
            | Instruction::H { qubit }
            | Instruction::S { qubit }
            | Instruction::Sdg { qubit }
            | Instruction::T { qubit }
            | Instruction::Tdg { qubit }
            | Instruction::Sx { qubit }
            | Instruction::Sxdg { qubit }
            | Instruction::U0 { qubit }
            | Instruction::Id { qubit }
            | Instruction::Reset { qubit } => vec![*qubit],

            Instruction::U3 { qubit, .. }
            | Instruction::U2 { qubit, .. }
            | Instruction::U1 { qubit, .. }
            | Instruction::U { qubit, .. }
            | Instruction::P { qubit, .. }
            | Instruction::Rx { qubit, .. }
            | Instruction::Ry { qubit, .. }
            | Instruction::Rz { qubit, .. } => vec![*qubit],

            Instruction::Measure { qubit, .. } => vec![*qubit],

            Instruction::Cx { control, target }
            | Instruction::Cz { control, target }
            | Instruction::Cy { control, target }
            | Instruction::Ch { control, target }
            | Instruction::Csx { control, target }
            | Instruction::Crx {
                control, target, ..
            }
            | Instruction::Cry {
                control, target, ..
            }
            | Instruction::Crz {
                control, target, ..
            }
            | Instruction::Cu1 {
                control, target, ..
            }
            | Instruction::Cp {
                control, target, ..
            }
            | Instruction::Cu3 {
                control, target, ..
            }
            | Instruction::Cu {
                control, target, ..
            } => vec![*control, *target],

            Instruction::Swap { a, b }
            | Instruction::Rxx { a, b, .. }
            | Instruction::Rzz { a, b, .. } => vec![*a, *b],

            Instruction::Ccx {
                control1,
                control2,
                target,
            }
            | Instruction::Rccx {
                control1,
                control2,
                target,
            } => {
                vec![*control1, *control2, *target]
            }

            Instruction::Cswap {
                control,
                target1,
                target2,
            } => {
                vec![*control, *target1, *target2]
            }

            Instruction::Rc3x {
                control1,
                control2,
                control3,
                target,
            }
            | Instruction::C3x {
                control1,
                control2,
                control3,
                target,
            }
            | Instruction::C3sqrtx {
                control1,
                control2,
                control3,
                target,
            } => {
                vec![*control1, *control2, *control3, *target]
            }

            Instruction::C4x {
                control1,
                control2,
                control3,
                control4,
                target,
            } => {
                vec![*control1, *control2, *control3, *control4, *target]
            }

            Instruction::Gate { qubits, .. } => qubits.clone(),

            Instruction::Conditional { op, .. } => op.qubits(),

            Instruction::Barrier | Instruction::Classical { .. } => vec![],
        }
    }
}
