use crate::types::Circuit;

pub(super) fn validate_circuit(circuit: &Circuit) -> Result<(), String> {
    crate::validation::validate_circuit(circuit)?;
    if circuit.num_qubits() >= usize::BITS as usize {
        return Err(format!(
            "Statevector qubit count must be less than {}",
            usize::BITS
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Condition;
    use crate::types::{Instruction, Register};
    use std::collections::HashMap;

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
