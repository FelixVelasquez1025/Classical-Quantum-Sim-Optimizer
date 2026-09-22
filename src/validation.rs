use crate::types::{Circuit, Instruction, Register};
use std::collections::HashMap;
type Ranges = Vec<(usize, usize)>;

pub(crate) fn validate_circuit(circuit: &Circuit) -> Result<(), String> {
    let qubits = register_ranges(&circuit.qregs, "quantum")?;
    let cbits = register_ranges(&circuit.cregs, "classical")?;
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
                | "remote_epr"
                | "epr" => (Some(2), 0),
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
