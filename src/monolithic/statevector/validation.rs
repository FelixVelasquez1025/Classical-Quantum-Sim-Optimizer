use std::collections::HashMap;

use crate::types::{Circuit, Instruction, Register};

type Ranges = Vec<(usize, usize)>;

/// Validate before allocation or dispatch, including operations in untaken branches.
pub(super) fn validate_circuit(circuit: &Circuit) -> Result<(), String> {
    let qubits = register_ranges(&circuit.qregs, "quantum")?;
    let cbits = register_ranges(&circuit.cregs, "classical")?;
    // Every engine index uses a usize bit mask. Allocation limits are checked by
    // the caller separately, before constructing the exponentially sized state.
    if circuit.num_qubits() >= usize::BITS as usize {
        return Err(format!(
            "Statevector qubit count must be less than {}",
            usize::BITS
        ));
    }
    for (index, instruction) in circuit.instructions.iter().enumerate() {
        validate_instruction(instruction, &qubits, &cbits)
            .map_err(|error| format!("Instruction {index}: {error}"))?;
    }
    Ok(())
}

fn register_ranges(registers: &HashMap<String, Register>, kind: &str) -> Result<Ranges, String> {
    let mut ranges = Vec::with_capacity(registers.len());
    for (name, register) in registers {
        let end = register
            .base
            .checked_add(register.size)
            .ok_or_else(|| format!("{kind} register {name:?} has an overflowing range"))?;
        if register.size != 0 {
            ranges.push((register.base, end));
        }
    }
    ranges.sort_unstable();
    for pair in ranges.windows(2) {
        if pair[1].0 < pair[0].1 {
            return Err(format!("Overlapping {kind} registers"));
        }
    }
    Ok(ranges)
}

fn declared(index: usize, ranges: &Ranges) -> bool {
    ranges
        .iter()
        .any(|&(start, end)| start <= index && index < end)
}

fn validate_instruction(
    instruction: &Instruction,
    qubits: &Ranges,
    cbits: &Ranges,
) -> Result<(), String> {
    if let Instruction::Conditional { condition, op } = instruction {
        if condition.creg_size == 0 || condition.creg_size > u64::BITS as usize {
            return Err("Condition width must be between 1 and 64 bits".into());
        }
        let end = condition
            .creg_base
            .checked_add(condition.creg_size)
            .ok_or_else(|| "Condition has an overflowing classical range".to_string())?;
        if (condition.creg_base..end).any(|bit| !declared(bit, cbits)) {
            return Err("Condition refers to undeclared classical bits".into());
        }
        if condition.creg_size < u64::BITS as usize
            && condition.creg_value >= (1u64 << condition.creg_size)
        {
            return Err("Condition value does not fit its classical width".into());
        }
        return validate_instruction(op, qubits, cbits);
    }

    let operands = instruction.qubits();
    for (index, &qubit) in operands.iter().enumerate() {
        if !declared(qubit, qubits) {
            return Err(format!(
                "Qubit {qubit} is not declared by a quantum register"
            ));
        }
        if operands[..index].contains(&qubit) {
            return Err(format!(
                "Gate operands must be distinct; qubit {qubit} is repeated"
            ));
        }
    }

    match instruction {
        Instruction::Measure { cbit, .. } if !declared(*cbit, cbits) => {
            return Err(format!(
                "Classical bit {cbit} is not declared by a classical register"
            ));
        }
        Instruction::Classical { name } => {
            return Err(format!("Unsupported classical operation: {name:?}"));
        }
        Instruction::Gate {
            name,
            params,
            qubits,
        } => {
            let (arity, parameters) = match name.to_lowercase().as_str() {
                "remote_link_phi_plus"
                | "remote_link_psi_minus"
                | "remote_link_psi_plus"
                | "nonlocal_cz"
                | "remote_cz"
                | "remote_cx"
                | "remote_epr" => (Some(2), 0),
                "remote_rzz" | "remote_cu1" => (Some(2), 1),
                "remote_barrier" => (None, 0),
                _ => {
                    return Err(format!(
                        "Unsupported generic gate: {name:?}. Decompose it before simulating."
                    ))
                }
            };
            if let Some(arity) = arity {
                if qubits.len() != arity {
                    return Err(format!(
                        "Gate {name:?} requires {arity} qubits, received {}",
                        qubits.len()
                    ));
                }
            }
            if params.len() != parameters {
                return Err(format!(
                    "Gate {name:?} requires {parameters} parameters, received {}",
                    params.len()
                ));
            }
        }
        _ => {}
    }

    let finite_parameters = match instruction {
        Instruction::U3 {
            theta, phi, lam, ..
        }
        | Instruction::U {
            theta, phi, lam, ..
        }
        | Instruction::Cu3 {
            theta, phi, lam, ..
        } => [theta, phi, lam].iter().all(|p| p.is_finite()),
        Instruction::Cu {
            theta,
            phi,
            lam,
            gamma,
            ..
        } => [theta, phi, lam, gamma].iter().all(|p| p.is_finite()),
        Instruction::U2 { phi, lam, .. } => phi.is_finite() && lam.is_finite(),
        Instruction::U1 { lam, .. }
        | Instruction::P { lam, .. }
        | Instruction::Crz { lam, .. }
        | Instruction::Cu1 { lam, .. }
        | Instruction::Cp { lam, .. } => lam.is_finite(),
        Instruction::Rx { theta, .. }
        | Instruction::Ry { theta, .. }
        | Instruction::Crx { theta, .. }
        | Instruction::Cry { theta, .. }
        | Instruction::Rxx { theta, .. }
        | Instruction::Rzz { theta, .. } => theta.is_finite(),
        Instruction::Rz { phi, .. } => phi.is_finite(),
        Instruction::Gate { params, .. } => params.iter().all(|p| p.is_finite()),
        _ => true,
    };
    if !finite_parameters {
        return Err("Gate parameters must be finite numbers".into());
    }
    // U-family matrix construction evaluates exp(i * (phi + lam)). Both
    // operands can be finite while their sum overflows and produces NaNs.
    match instruction {
        Instruction::U3 { phi, lam, .. }
        | Instruction::U2 { phi, lam, .. }
        | Instruction::U { phi, lam, .. }
        | Instruction::Cu3 { phi, lam, .. }
        | Instruction::Cu { phi, lam, .. }
            if !(phi + lam).is_finite() =>
        {
            return Err("The combined phase phi + lam must be finite".into());
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Condition;

    fn register(name: &str, base: usize, size: usize) -> Register {
        Register {
            name: name.into(),
            base,
            size,
        }
    }

    fn circuit(instructions: Vec<Instruction>) -> Circuit {
        Circuit {
            qregs: HashMap::from([("q".into(), register("q", 0, 3))]),
            cregs: HashMap::from([("c".into(), register("c", 0, 3))]),
            instructions,
        }
    }

    fn generic(name: &str, qubits: Vec<usize>, params: Vec<f64>) -> Instruction {
        Instruction::Gate {
            name: name.into(),
            qubits,
            params,
        }
    }

    fn conditional(base: usize, size: usize, value: u64, op: Instruction) -> Instruction {
        Instruction::Conditional {
            condition: Condition {
                creg_base: base,
                creg_size: size,
                creg_value: value,
            },
            op: Box::new(op),
        }
    }

    #[test]
    fn validates_measurement_reset_and_nested_classical_control() {
        let input = circuit(vec![
            Instruction::H { qubit: 0 },
            Instruction::Cx {
                control: 0,
                target: 2,
            },
            Instruction::Measure { qubit: 2, cbit: 1 },
            conditional(
                0,
                3,
                2,
                conditional(1, 1, 1, Instruction::Reset { qubit: 0 }),
            ),
        ]);
        assert!(validate_circuit(&input).is_ok());
    }

    #[test]
    fn rejects_invalid_operands_before_any_kernel_runs() {
        for instruction in [
            Instruction::Cx {
                control: 0,
                target: 3,
            },
            Instruction::Cx {
                control: 1,
                target: 1,
            },
            Instruction::Id { qubit: 3 },
            Instruction::Measure { qubit: 0, cbit: 3 },
            conditional(0, 1, 1, Instruction::X { qubit: 99 }),
        ] {
            assert!(validate_circuit(&circuit(vec![instruction])).is_err());
        }
        // A hole between registers is not a declared wire.
        let mut input = circuit(vec![Instruction::X { qubit: 1 }]);
        input.qregs = HashMap::from([
            ("a".into(), register("a", 0, 1)),
            ("b".into(), register("b", 2, 1)),
        ]);
        assert!(validate_circuit(&input).is_err());
        // This used to reach unchecked pointers in the parallel dense kernel.
        let mut parallel_input = circuit(vec![Instruction::Cx {
            control: 0,
            target: 12,
        }]);
        parallel_input.qregs = HashMap::from([("q".into(), register("q", 0, 12))]);
        assert!(validate_circuit(&parallel_input).is_err());
    }

    #[test]
    fn rejects_overflow_overlap_and_unrepresentable_state_dimensions() {
        for ranges in [
            vec![register("a", usize::MAX, 1)],
            vec![register("a", 0, 2), register("b", 1, 2)],
            vec![register("a", 0, usize::BITS as usize)],
        ] {
            let mut input = circuit(vec![]);
            input.qregs = ranges.into_iter().map(|r| (r.name.clone(), r)).collect();
            assert!(validate_circuit(&input).is_err());
        }
        let mut input = circuit(vec![]);
        input
            .cregs
            .insert("overlap".into(), register("overlap", 1, 1));
        assert!(validate_circuit(&input).is_err());
    }

    #[test]
    fn validates_generic_gate_arity_parameters_and_support() {
        for instruction in [
            generic("remote_cu1", vec![0, 1], vec![0.4]),
            generic("remote_rzz", vec![0, 1], vec![0.4]),
            generic("REMOTE_CX", vec![0, 2], vec![]),
            generic("remote_barrier", vec![0, 1, 2], vec![]),
        ] {
            assert!(validate_circuit(&circuit(vec![instruction])).is_ok());
        }
        for instruction in [
            generic("remote_cu1", vec![0, 1], vec![]),
            generic("remote_cx", vec![0], vec![]),
            generic("remote_epr", vec![0, 1, 2], vec![]),
            generic("remote_cz", vec![0, 1], vec![0.2]),
            generic("circuit-123", vec![0], vec![]),
            generic("remote_barrier", vec![1, 1], vec![]),
            Instruction::Classical {
                name: "opaque".into(),
            },
        ] {
            assert!(validate_circuit(&circuit(vec![instruction])).is_err());
        }
    }

    #[test]
    fn rejects_nonfinite_angles_including_generic_gates() {
        for instruction in [
            Instruction::U3 {
                qubit: 0,
                theta: 0.0,
                phi: f64::NAN,
                lam: 0.0,
            },
            Instruction::Cu {
                control: 0,
                target: 1,
                theta: 0.0,
                phi: 0.0,
                lam: 0.0,
                gamma: f64::INFINITY,
            },
            generic("remote_rzz", vec![0, 1], vec![f64::NEG_INFINITY]),
            Instruction::U2 {
                qubit: 0,
                phi: f64::MAX,
                lam: f64::MAX,
            },
        ] {
            assert!(validate_circuit(&circuit(vec![instruction])).is_err());
        }
    }

    #[test]
    fn checks_condition_width_value_membership_and_overflow() {
        for (base, size, value) in [
            (0, 0, 0),
            (0, 65, 0),
            (0, 1, 2),
            (2, 2, 0),
            (usize::MAX, 2, 0),
        ] {
            let op = conditional(base, size, value, Instruction::X { qubit: 0 });
            assert!(validate_circuit(&circuit(vec![op])).is_err());
        }
        let mut input = circuit(vec![conditional(
            0,
            64,
            u64::MAX,
            Instruction::X { qubit: 0 },
        )]);
        input.cregs = HashMap::from([("c".into(), register("c", 0, 64))]);
        assert!(validate_circuit(&input).is_ok());
    }
}
