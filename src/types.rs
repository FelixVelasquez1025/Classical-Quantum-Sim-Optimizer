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

#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct Condition {
    pub creg_base: usize,
    pub creg_size: usize,
    pub creg_value: u64,
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
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

impl Instruction {
    /// Remap quantum wires without changing classical addresses.
    pub(crate) fn remap_qubits(&mut self, map: &impl Fn(usize) -> usize) {
        match self {
            Self::Id { qubit } => {
                *qubit = map(*qubit);
            }
            Self::X { qubit } => {
                *qubit = map(*qubit);
            }
            Self::Y { qubit } => {
                *qubit = map(*qubit);
            }
            Self::Z { qubit } => {
                *qubit = map(*qubit);
            }
            Self::H { qubit } => {
                *qubit = map(*qubit);
            }
            Self::S { qubit } => {
                *qubit = map(*qubit);
            }
            Self::Sdg { qubit } => {
                *qubit = map(*qubit);
            }
            Self::T { qubit } => {
                *qubit = map(*qubit);
            }
            Self::Tdg { qubit } => {
                *qubit = map(*qubit);
            }
            Self::Sx { qubit } => {
                *qubit = map(*qubit);
            }
            Self::Sxdg { qubit } => {
                *qubit = map(*qubit);
            }
            Self::U3 { qubit, .. } => {
                *qubit = map(*qubit);
            }
            Self::U2 { qubit, .. } => {
                *qubit = map(*qubit);
            }
            Self::U1 { qubit, .. } => {
                *qubit = map(*qubit);
            }
            Self::U { qubit, .. } => {
                *qubit = map(*qubit);
            }
            Self::P { qubit, .. } => {
                *qubit = map(*qubit);
            }
            Self::Rx { qubit, .. } => {
                *qubit = map(*qubit);
            }
            Self::Ry { qubit, .. } => {
                *qubit = map(*qubit);
            }
            Self::Rz { qubit, .. } => {
                *qubit = map(*qubit);
            }
            Self::U0 { qubit } => {
                *qubit = map(*qubit);
            }
            Self::Cx { control, target } => {
                *control = map(*control);
                *target = map(*target);
            }
            Self::Cz { control, target } => {
                *control = map(*control);
                *target = map(*target);
            }
            Self::Cy { control, target } => {
                *control = map(*control);
                *target = map(*target);
            }
            Self::Ch { control, target } => {
                *control = map(*control);
                *target = map(*target);
            }
            Self::Swap { a, b } => {
                *a = map(*a);
                *b = map(*b);
            }
            Self::Csx { control, target } => {
                *control = map(*control);
                *target = map(*target);
            }
            Self::Crx {
                control, target, ..
            } => {
                *control = map(*control);
                *target = map(*target);
            }
            Self::Cry {
                control, target, ..
            } => {
                *control = map(*control);
                *target = map(*target);
            }
            Self::Crz {
                control, target, ..
            } => {
                *control = map(*control);
                *target = map(*target);
            }
            Self::Cu1 {
                control, target, ..
            } => {
                *control = map(*control);
                *target = map(*target);
            }
            Self::Cp {
                control, target, ..
            } => {
                *control = map(*control);
                *target = map(*target);
            }
            Self::Cu3 {
                control, target, ..
            } => {
                *control = map(*control);
                *target = map(*target);
            }
            Self::Cu {
                control, target, ..
            } => {
                *control = map(*control);
                *target = map(*target);
            }
            Self::Rxx { a, b, .. } => {
                *a = map(*a);
                *b = map(*b);
            }
            Self::Rzz { a, b, .. } => {
                *a = map(*a);
                *b = map(*b);
            }
            Self::Ccx {
                control1,
                control2,
                target,
            } => {
                *control1 = map(*control1);
                *control2 = map(*control2);
                *target = map(*target);
            }
            Self::Cswap {
                control,
                target1,
                target2,
            } => {
                *control = map(*control);
                *target1 = map(*target1);
                *target2 = map(*target2);
            }
            Self::Rccx {
                control1,
                control2,
                target,
            } => {
                *control1 = map(*control1);
                *control2 = map(*control2);
                *target = map(*target);
            }
            Self::Rc3x {
                control1,
                control2,
                control3,
                target,
            } => {
                *control1 = map(*control1);
                *control2 = map(*control2);
                *control3 = map(*control3);
                *target = map(*target);
            }
            Self::C3x {
                control1,
                control2,
                control3,
                target,
            } => {
                *control1 = map(*control1);
                *control2 = map(*control2);
                *control3 = map(*control3);
                *target = map(*target);
            }
            Self::C3sqrtx {
                control1,
                control2,
                control3,
                target,
            } => {
                *control1 = map(*control1);
                *control2 = map(*control2);
                *control3 = map(*control3);
                *target = map(*target);
            }
            Self::C4x {
                control1,
                control2,
                control3,
                control4,
                target,
            } => {
                *control1 = map(*control1);
                *control2 = map(*control2);
                *control3 = map(*control3);
                *control4 = map(*control4);
                *target = map(*target);
            }
            Self::Gate { qubits, .. } => {
                for q in qubits {
                    *q = map(*q);
                }
            }
            Self::Measure { qubit, .. } => {
                *qubit = map(*qubit);
            }
            Self::Reset { qubit } => {
                *qubit = map(*qubit);
            }
            Self::Conditional { op, .. } => op.remap_qubits(map),
            Self::Classical { .. } => {}
            Self::Barrier => {}
        }
    }
}
