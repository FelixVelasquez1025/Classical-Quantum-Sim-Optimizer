//! Validate the transport before allocating quantum state or compiling gates.
use crate::types::{Circuit, Instruction, Register};
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};
use std::collections::{BTreeMap, HashMap, HashSet};

pub(super) struct Input {
    pub physical: Vec<usize>,
    pub instructions: Vec<Instruction>,
    pub num_cbits: usize,
}
fn invalid(message: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(message.into())
}
fn parse(c: &Bound<PyAny>) -> PyResult<Circuit> {
    let json: String = c.call_method0("model_dump_json")?.extract()?;
    serde_json::from_str(&json).map_err(|e| invalid(format!("Circuit JSON parse error: {e}")))
}
fn check_size(n: usize, budget: usize) -> PyResult<()> {
    if n.checked_mul(256).is_none_or(|v| v > budget) {
        return Err(pyo3::exceptions::PyMemoryError::new_err(
            "P-block wire metadata exceeds max_memory_mb",
        ));
    }
    Ok(())
}

pub(super) fn prepare(obj: &Bound<PyAny>, budget: usize) -> PyResult<Input> {
    if !obj.hasattr("circuits")? {
        let mut circuit = parse(obj)?;
        crate::validation::validate_circuit(&circuit).map_err(invalid)?;
        let ranges =
            crate::validation::register_ranges(&circuit.qregs, "quantum").map_err(invalid)?;
        let n = ranges
            .iter()
            .try_fold(0usize, |acc, (a, b)| acc.checked_add(b - a))
            .ok_or_else(|| invalid("Qubit count overflow"))?;
        check_size(n, budget)?;
        let physical: Vec<_> = ranges.into_iter().flat_map(|(a, b)| a..b).collect();
        let map: HashMap<_, _> = physical.iter().enumerate().map(|(i, &q)| (q, i)).collect();
        let num_cbits = circuit.num_cbits();
        for op in &mut circuit.instructions {
            op.remap_qubits(&|q| map[&q]);
        }
        return Ok(Input {
            physical,
            instructions: circuit.instructions,
            num_cbits,
        });
    }
    let qpn: HashMap<usize, Vec<usize>> = obj.getattr("qubits_per_node")?.extract()?;
    let node_objects = obj.getattr("circuits")?;
    let node_objects = node_objects.downcast::<PyDict>()?;
    let mut nodes = node_objects.keys().extract::<Vec<usize>>()?;
    nodes.sort_unstable();
    if nodes.iter().copied().collect::<HashSet<_>>() != qpn.keys().copied().collect() {
        return Err(invalid(
            "circuits and qubits_per_node must contain the same node IDs",
        ));
    }
    let mut owned = HashSet::new();
    for qs in qpn.values() {
        for &q in qs {
            if q == usize::MAX || !owned.insert(q) {
                return Err(invalid(
                    "Physical qubits must have unique ownership and representable indices",
                ));
            }
        }
    }
    check_size(owned.len(), budget)?;
    let mut physical: Vec<_> = owned.into_iter().collect();
    physical.sort_unstable();
    let wire_map: HashMap<_, _> = physical.iter().enumerate().map(|(i, &q)| (q, i)).collect();
    let explicit = obj.hasattr("operation_ids")? || obj.hasattr("operation_order")?;
    let ids: HashMap<usize, Vec<String>> = if explicit {
        obj.getattr("operation_ids")?.extract()?
    } else {
        HashMap::new()
    };
    let order: HashMap<String, i64> = if explicit {
        obj.getattr("operation_order")?.extract()?
    } else {
        HashMap::new()
    };
    if explicit && ids.keys().copied().collect::<HashSet<_>>() != nodes.iter().copied().collect() {
        return Err(invalid("operation_ids must list every circuit node"));
    }
    let legacy: HashMap<usize, i64> = if explicit {
        HashMap::new()
    } else {
        obj.getattr("_instruction_index")?.extract()?
    };
    let mut events: HashMap<String, (i64, Instruction)> = HashMap::new();
    let mut orders = BTreeMap::new();
    let mut classical_ranges = HashSet::new();
    let mut quantum_ranges = Vec::new();
    for node in nodes {
        let c_obj = node_objects
            .get_item(node)?
            .ok_or_else(|| invalid("Missing node circuit"))?;
        let circuit = parse(&c_obj)?;
        let qr = crate::validation::register_ranges(&circuit.qregs, "quantum").map_err(invalid)?;
        let cr =
            crate::validation::register_ranges(&circuit.cregs, "classical").map_err(invalid)?;
        quantum_ranges.extend(qr);
        classical_ranges.extend(cr);
        let legacy_ids: Vec<usize> = if explicit {
            Vec::new()
        } else {
            let objects = c_obj.getattr("instructions")?;
            let objects = objects.downcast::<PyList>()?;
            if objects.len() != circuit.instructions.len() {
                return Err(invalid(
                    "Python instruction list and serialized instructions have different lengths",
                ));
            }
            objects.iter().map(|o| o.as_ptr() as usize).collect()
        };
        if explicit && ids[&node].len() != circuit.instructions.len() {
            return Err(invalid(
                "operation_ids length must match serialized instructions",
            ));
        }
        let mut seen = HashSet::new();
        let mut previous = None;
        for (i, inst) in circuit.instructions.into_iter().enumerate() {
            let (id, position) = if explicit {
                let id = &ids[&node][i];
                (
                    id.clone(),
                    *order
                        .get(id)
                        .ok_or_else(|| invalid(format!("Missing order for operation {id:?}")))?,
                )
            } else {
                let id = legacy_ids[i];
                let position = legacy.get(&id).ok_or_else(|| invalid(
                    "Missing instruction order; supply complete _instruction_index or explicit operation IDs"
                ))?;
                (id.to_string(), *position)
            };
            if !seen.insert(id.clone()) {
                return Err(invalid("Repeated operation ID within one node; each occurrence requires its own explicit ID"));
            }
            if previous.is_some_and(|p| position <= p) {
                return Err(invalid(
                    "Global operation order contradicts the node instruction order",
                ));
            }
            previous = Some(position);
            if let Some((old_pos, old_inst)) = events.get(&id) {
                if *old_pos != position || *old_inst != inst {
                    return Err(invalid(
                        "Copies of a shared operation must agree on order and contents",
                    ));
                }
            } else {
                if orders.insert(position, id.clone()).is_some() {
                    return Err(invalid(
                        "Ambiguous global order: different operations have the same position",
                    ));
                }
                events.insert(id, (position, inst));
            }
        }
    }
    for &q in &physical {
        if !quantum_ranges.iter().any(|&(a, b)| a <= q && q < b) {
            return Err(invalid(format!(
                "Physical qubit {q} has no register declaration"
            )));
        }
    }
    let qregs = physical
        .iter()
        .map(|&q| {
            (
                q.to_string(),
                Register {
                    name: q.to_string(),
                    base: q,
                    size: 1,
                },
            )
        })
        .collect();
    // Identical shared register ranges are allowed across nodes; partial overlap is not.
    let cregs = classical_ranges
        .into_iter()
        .enumerate()
        .map(|(i, (a, b))| {
            (
                i.to_string(),
                Register {
                    name: i.to_string(),
                    base: a,
                    size: b - a,
                },
            )
        })
        .collect();
    let instructions = orders
        .into_values()
        .map(|id| events.remove(&id).unwrap().1)
        .collect();
    let mut circuit = Circuit {
        qregs,
        cregs,
        instructions,
    };
    crate::validation::validate_circuit(&circuit).map_err(invalid)?;
    let num_cbits = circuit.num_cbits();
    for op in &mut circuit.instructions {
        op.remap_qubits(&|q| wire_map[&q]);
    }
    Ok(Input {
        physical,
        instructions: circuit.instructions,
        num_cbits,
    })
}
