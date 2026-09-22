"""Statevector correctness and API regressions.

Run after building the extension with ``python -m unittest discover -s tests -v``.
NumPy and dqsim are required. Installing Qiskit additionally enables independent
reference checks of every standard unitary gate and random circuits.
"""

import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dqsim import _core

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import standard_gates as qg
    from qiskit.quantum_info import Operator, Statevector
except ImportError:
    QuantumCircuit = qg = Operator = Statevector = None


class Circuit:
    """Minimal circuit transport, without the optional Bosonic model packages."""

    def __init__(self, qubits, instructions=(), cbits=0):
        self.data = {
            "qregs": {"q": {"name": "q", "base": 0, "size": qubits}},
            "cregs": {"c": {"name": "c", "base": 0, "size": cbits}},
            "instructions": list(instructions),
        }

    def model_dump_json(self):
        return json.dumps(self.data)


def inst(kind, **fields):
    return {"kind": kind, **fields}


def conditional(op, value, base=0, size=1):
    return inst(
        "conditional",
        condition={"creg_base": base, "creg_size": size, "creg_value": value},
        op=op,
    )


class StatevectorTests(unittest.TestCase):
    def simulate(self, circuit, **options):
        return _core.StatevectorSimulator(seed=1729, **options).simulate(circuit)

    def assert_distribution(self, counts, expected, shots):
        self.assertEqual(sum(counts.values()), shots)
        self.assertEqual(set(counts), set(expected))
        for outcome, probability in expected.items():
            # More than eight standard deviations for the probabilities used
            # below; this tests distributions without depending on RNG details.
            self.assertAlmostEqual(counts[outcome] / shots, probability, delta=0.08)

    def test_little_endian_statevector_and_ordered_marginals(self):
        result = self.simulate(Circuit(4, [inst("x", qubit=0), inst("x", qubit=2)]))
        expected = np.zeros(16, dtype=complex)
        expected[5] = 1
        np.testing.assert_allclose(result.statevector, expected, atol=1e-14)
        self.assertEqual(result.probabilities(), {5: 1.0})
        self.assertEqual(result.probabilities([0, 1]), {2: 1.0})
        self.assertEqual(result.probabilities([1, 0]), {1: 1.0})
        self.assertEqual(result.probabilities([]), {0: 1.0})
        self.assertEqual(result.counts(shots=17, qubits=[0, 1], seed=2), {"10": 17})
        self.assertEqual(result.counts(shots=17, qubits=[], seed=2), {"": 17})

    def test_full_and_marginal_probabilities_match_amplitudes(self):
        result = self.simulate(Circuit(5, [
            inst("ry", qubit=0, theta=0.7),
            inst("h", qubit=4),
            inst("cx", control=4, target=2),
            inst("rx", qubit=1, theta=-0.3),
            inst("rxx", a=1, b=3, theta=0.8),
        ]))
        probabilities = np.abs(result.statevector) ** 2
        actual = result.probabilities()
        np.testing.assert_allclose(
            [actual.get(i, 0) for i in range(32)], probabilities, atol=1e-14
        )
        self.assertAlmostEqual(sum(actual.values()), 1.0, places=13)
        for qubits in ([3, 0, 4], [0], [4, 3, 2, 1, 0], []):
            with self.subTest(qubits=qubits):
                expected = np.zeros(1 << len(qubits))
                for index, probability in enumerate(probabilities):
                    key = 0
                    for qubit in qubits:
                        key = (key << 1) | ((index >> qubit) & 1)
                    expected[key] += probability
                actual = result.probabilities(qubits)
                np.testing.assert_allclose(
                    [actual.get(i, 0) for i in range(len(expected))], expected, atol=1e-14
                )

    def test_tiny_rotations_are_not_discarded_by_fusion(self):
        theta = 1e-11
        result = self.simulate(Circuit(1, [inst("rx", qubit=0, theta=theta)]))
        np.testing.assert_allclose(
            result.statevector,
            [math.cos(theta / 2), -1j * math.sin(theta / 2)],
            rtol=1e-13,
            atol=1e-15,
        )

    def test_relative_controlled_x_gates_flip_the_target(self):
        for kind, controls in (("rccx", 2), ("rc3x", 3)):
            with self.subTest(gate=kind):
                gate = inst(kind, target=controls)
                gate.update({f"control{i + 1}": i for i in range(controls)})
                result = self.simulate(Circuit(controls + 1, [
                    *(inst("x", qubit=i) for i in range(controls)), gate,
                ]))
                self.assertAlmostEqual(
                    result.probabilities().get((1 << (controls + 1)) - 1, 0), 1.0
                )

    def test_terminal_sampling_preserves_joint_correlations_and_mapping(self):
        circuit = Circuit(3, [
            inst("h", qubit=0),
            inst("cx", control=0, target=2),
            inst("measure", qubit=2, cbit=1),
            inst("barrier"),
            inst("measure", qubit=0, cbit=3),
        ], cbits=5)
        for sample_terminal in (True, False):
            with self.subTest(sample_terminal=sample_terminal):
                sim = _core.StatevectorSimulator(seed=10, sample_terminal=sample_terminal)
                counts = sim.simulate_shots(circuit, shots=4096)
                self.assert_distribution(counts, {"00000": 0.5, "01010": 0.5}, 4096)
                self.assertEqual(counts, sim.simulate_shots(circuit, shots=4096))

    def test_repeated_measurements_and_overwritten_classical_bits(self):
        circuit = Circuit(3, [
            inst("h", qubit=0),
            inst("x", qubit=1),
            inst("measure", qubit=0, cbit=0),
            inst("measure", qubit=0, cbit=2),
            inst("measure", qubit=1, cbit=0),
            inst("measure", qubit=0, cbit=1),
        ], cbits=3)
        for sample_terminal in (True, False):
            with self.subTest(sample_terminal=sample_terminal):
                counts = _core.StatevectorSimulator(
                    seed=21, sample_terminal=sample_terminal
                ).simulate_shots(circuit, shots=4096)
                self.assert_distribution(counts, {"001": 0.5, "111": 0.5}, 4096)

    def test_mid_circuit_measurement_and_feedback(self):
        circuit = Circuit(2, [
            inst("h", qubit=0),
            inst("measure", qubit=0, cbit=0),
            conditional(inst("x", qubit=1), 1),
            inst("x", qubit=0),
            inst("measure", qubit=0, cbit=1),
            inst("measure", qubit=1, cbit=2),
        ], cbits=3)
        for sample_terminal in (True, False):
            with self.subTest(sample_terminal=sample_terminal):
                sim = _core.StatevectorSimulator(seed=15, sample_terminal=sample_terminal)
                counts = sim.simulate_shots(circuit, shots=4096)
                self.assert_distribution(counts, {"010": 0.5, "101": 0.5}, 4096)
                self.assertEqual(counts, sim.simulate_shots(circuit, shots=4096))

    def test_reset_and_nested_classical_conditionals(self):
        circuit = Circuit(2, [
            inst("x", qubit=0),
            inst("measure", qubit=0, cbit=2),
            inst("reset", qubit=0),
            inst("measure", qubit=0, cbit=0),
            conditional(conditional(inst("x", qubit=1), 0), 2, base=1, size=2),
            inst("measure", qubit=1, cbit=1),
        ], cbits=3)
        result = self.simulate(circuit)
        self.assertEqual(result.classical_bits, {0: 0, 1: 1, 2: 1})
        np.testing.assert_allclose(result.statevector, [0, 0, 1, 0], atol=1e-14)
        counts = _core.StatevectorSimulator(seed=22).simulate_shots(circuit, shots=64)
        self.assertEqual(counts, {"110": 64})

    def test_reset_of_entangled_qubit_retains_other_qubits_mixture(self):
        circuit = Circuit(2, [
            inst("h", qubit=0),
            inst("cx", control=0, target=1),
            inst("reset", qubit=0),
            inst("measure", qubit=0, cbit=0),
            inst("measure", qubit=1, cbit=1),
        ], cbits=2)
        counts = _core.StatevectorSimulator(seed=24).simulate_shots(circuit, shots=4096)
        self.assert_distribution(counts, {"00": 0.5, "10": 0.5}, 4096)

    def test_zero_shots_and_unwritten_classical_bits(self):
        for sample_terminal in (True, False):
            sim = _core.StatevectorSimulator(seed=30, sample_terminal=sample_terminal)
            for cbits, outcome in ((0, ""), (3, "000")):
                with self.subTest(sample_terminal=sample_terminal, cbits=cbits):
                    circuit = Circuit(1, [inst("h", qubit=0)], cbits=cbits)
                    self.assertEqual(sim.simulate_shots(circuit, shots=0), {})
                    self.assertEqual(sim.simulate_shots(circuit, shots=13), {outcome: 13})

    def test_result_sampling_is_seeded(self):
        result = self.simulate(Circuit(2, [inst("h", qubit=0), inst("cx", control=0, target=1)]))
        counts = result.counts(shots=4096, seed=77)
        self.assertEqual(counts, result.counts(shots=4096, seed=77))
        self.assert_distribution(counts, {"00": 0.5, "11": 0.5}, 4096)
        self.assertEqual(result.counts(shots=0), {})

    def test_invalid_qubits_and_operands_fail_before_execution(self):
        invalid = [
            Circuit(12, [inst("cx", control=0, target=12)]),
            Circuit(12, [inst("x", qubit=12)]),
            Circuit(2, [inst("cx", control=1, target=1)]),
            Circuit(3, [inst("ccx", control1=0, control2=0, target=2)]),
            Circuit(2, [inst("measure", qubit=0, cbit=2)], cbits=2),
            Circuit(1, [conditional(inst("x", qubit=1), 1)], cbits=1),
            Circuit(1, [conditional(inst("x", qubit=0), 0, base=1)], cbits=1),
            Circuit(2, [inst("gate", name="remote_cx", params=[], qubits=[0])]),
            Circuit(2, [inst("gate", name="remote_cu1", params=[], qubits=[0, 1])]),
        ]
        sim = _core.StatevectorSimulator(seed=1)
        for circuit in invalid:
            for method in (sim.simulate, sim.simulate_shots):
                with self.subTest(circuit=circuit.data, method=method.__name__):
                    with self.assertRaises(ValueError):
                        method(circuit)

    def test_unsupported_operations_are_never_silently_ignored(self):
        unsupported = [
            inst("gate", name="circuit-123", params=[], qubits=[0]),
            inst("gate", name="unknown", params=[], qubits=[0]),
            inst("classical", name="unsupported_classical_operation"),
        ]
        sim = _core.StatevectorSimulator(seed=1)
        for op in unsupported:
            # Unreachable branches must be validated as well.
            for instruction in (op, conditional(op, 1)):
                circuit = Circuit(1, [instruction], cbits=1)
                for method in (sim.simulate, sim.simulate_shots):
                    with self.subTest(instruction=instruction, method=method.__name__):
                        with self.assertRaises((ValueError, NotImplementedError)):
                            method(circuit)

    def test_invalid_probability_queries_fail(self):
        result = self.simulate(Circuit(2))
        for qubits in ([2], [0, 0], [1, 0, 1]):
            for method in (result.probabilities, result.counts):
                with self.subTest(qubits=qubits, method=method.__name__):
                    with self.assertRaises(ValueError):
                        method(qubits=qubits)

    def test_memory_budget_rejects_oversized_state_before_allocation(self):
        sim = _core.StatevectorSimulator(max_memory_mb=1)
        for method in (sim.simulate, sim.simulate_shots):
            with self.subTest(method=method.__name__):
                with self.assertRaises((ValueError, MemoryError)):
                    method(Circuit(20))

    def test_worker_limit_preserves_seeded_trajectory_results(self):
        circuit = Circuit(2, [
            inst("h", qubit=0),
            inst("measure", qubit=0, cbit=0),
            conditional(inst("x", qubit=1), 1),
            inst("measure", qubit=1, cbit=1),
        ], cbits=2)
        counts = [
            _core.StatevectorSimulator(
                seed=38, max_memory_mb=1, max_parallel_shots=workers, sample_terminal=False
            ).simulate_shots(circuit, shots=257)
            for workers in (1, 2, 4)
        ]
        self.assertEqual(counts[0], counts[1])
        self.assertEqual(counts[1], counts[2])

    def test_wrapper_forwards_statevector_options(self):
        circuit = Circuit(1, [inst("x", qubit=0), inst("measure", qubit=0, cbit=0)], cbits=1)
        options = {"seed": 11, "max_memory_mb": 1, "max_parallel_shots": 1, "sample_terminal": False}
        result = _core.simulate_monolithic(circuit, **options)
        np.testing.assert_allclose(result.statevector, [0, 1], atol=1e-14)
        self.assertEqual(_core.simulate_monolithic_shots(circuit, shots=19, **options), {"1": 19})

    def test_invalid_options_and_large_classical_registers(self):
        for options in ({"max_memory_mb": 0}, {"max_parallel_shots": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                _core.StatevectorSimulator(**options)
        with self.assertRaises(TypeError):
            _core.simulate_monolithic(Circuit(1), nonexistent_option=True)
        circuit = Circuit(1, cbits=2_000_000)
        sim = _core.StatevectorSimulator(max_memory_mb=1)
        for method in (sim.simulate, sim.simulate_shots):
            with self.subTest(method=method.__name__), self.assertRaises(MemoryError):
                method(circuit)

    def test_profiles_confirm_terminal_sampling_and_dynamic_worker_budget(self):
        terminal = Circuit(2, [
            inst("h", qubit=0), inst("h", qubit=1),
            inst("measure", qubit=0, cbit=0), inst("measure", qubit=1, cbit=1),
        ], cbits=2)
        dynamic = Circuit(15, [
            inst("h", qubit=0), inst("measure", qubit=0, cbit=0),
            conditional(inst("x", qubit=14), 1),
            inst("measure", qubit=14, cbit=1),
        ], cbits=2)
        # Profiling is an opt-in disk API; contain its output in a temporary directory.
        original = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                sim = _core.StatevectorSimulator(seed=12, max_memory_mb=1)
                counts = sim.simulate_shots(terminal, shots=4096, profile=True)
                self.assert_distribution(counts, {"00": .25, "01": .25, "10": .25, "11": .25}, 4096)
                data = json.loads(next(Path("dqsim_profiles").glob("*.json")).read_text())
                self.assertEqual(data["execution_strategy"], "terminal_sampling")
                # Individual shot timings cannot represent shared evolution.
                self.assertEqual(data["shot_times"], [])
                self.assertLessEqual(data["working_bytes"], 1024**2)
                sim.simulate_shots(dynamic, shots=16, profile=True)
                data = json.loads(sorted(Path("dqsim_profiles").glob("*.json"))[-1].read_text())
                self.assertEqual(data["execution_strategy"], "trajectories")
                self.assertEqual(data["parallel_shots"], 1)
                self.assertEqual(len(data["shot_times"]), 16)
                self.assertLessEqual(data["working_bytes"], 1024**2)
            finally:
                os.chdir(original)

    def test_shared_kernels_preserve_mps_and_pblock_bell_circuits(self):
        instructions = [inst("h", qubit=0), inst("cx", control=0, target=1)]
        result = _core.MpsSimulator(seed=2).simulate(Circuit(2, instructions))
        np.testing.assert_allclose(result.statevector, [2**-.5, 0, 0, 2**-.5], atol=1e-13)
        measured = Circuit(2, instructions + [
            inst("measure", qubit=0, cbit=0), inst("measure", qubit=1, cbit=1),
        ], cbits=2)
        # P-block reads object identities to preserve the distributed instruction order.
        from types import SimpleNamespace
        measured.instructions = [object() for _ in measured.data["instructions"]]
        distributed = SimpleNamespace(
            circuits={0: measured}, qubits_per_node={0: [0, 1]},
            _instruction_index={id(op): i for i, op in enumerate(measured.instructions)},
        )
        counts = _core.PBlockSimulator(seed=13).simulate_shots(distributed, shots=4096)
        self.assert_distribution(counts, {"00": .5, "11": .5}, 4096)


def standard_gate_cases():
    """(JSON kind, operand names, JSON parameters, independently defined gate)."""
    theta, phi, lam, gamma = 0.731, -0.412, 1.137, 0.289
    single = ("qubit",)
    controlled = ("control", "target")
    pair = ("a", "b")
    triple = ("control1", "control2", "target")
    quadruple = ("control1", "control2", "control3", "target")
    return [
        ("id", single, {}, qg.IGate()),
        ("u0", single, {}, qg.IGate()),
        ("x", single, {}, qg.XGate()),
        ("y", single, {}, qg.YGate()),
        ("z", single, {}, qg.ZGate()),
        ("h", single, {}, qg.HGate()),
        ("s", single, {}, qg.SGate()),
        ("sdg", single, {}, qg.SdgGate()),
        ("t", single, {}, qg.TGate()),
        ("tdg", single, {}, qg.TdgGate()),
        ("sx", single, {}, qg.SXGate()),
        ("sxdg", single, {}, qg.SXdgGate()),
        ("u3", single, {"theta": theta, "phi": phi, "lam": lam}, qg.U3Gate(theta, phi, lam)),
        ("u2", single, {"phi": phi, "lam": lam}, qg.U2Gate(phi, lam)),
        ("u1", single, {"lam": lam}, qg.U1Gate(lam)),
        ("u", single, {"theta": theta, "phi": phi, "lam": lam}, qg.UGate(theta, phi, lam)),
        ("p", single, {"lam": lam}, qg.PhaseGate(lam)),
        ("rx", single, {"theta": theta}, qg.RXGate(theta)),
        ("ry", single, {"theta": theta}, qg.RYGate(theta)),
        ("rz", single, {"phi": phi}, qg.RZGate(phi)),
        ("cx", controlled, {}, qg.CXGate()),
        ("cz", controlled, {}, qg.CZGate()),
        ("cy", controlled, {}, qg.CYGate()),
        ("ch", controlled, {}, qg.CHGate()),
        ("swap", pair, {}, qg.SwapGate()),
        ("csx", controlled, {}, qg.CSXGate()),
        ("crx", controlled, {"theta": theta}, qg.CRXGate(theta)),
        ("cry", controlled, {"theta": theta}, qg.CRYGate(theta)),
        ("crz", controlled, {"lam": lam}, qg.CRZGate(lam)),
        ("cu1", controlled, {"lam": lam}, qg.CU1Gate(lam)),
        ("cp", controlled, {"lam": lam}, qg.CPhaseGate(lam)),
        ("cu3", controlled, {"theta": theta, "phi": phi, "lam": lam}, qg.CU3Gate(theta, phi, lam)),
        ("cu", controlled, {"theta": theta, "phi": phi, "lam": lam, "gamma": gamma}, qg.CUGate(theta, phi, lam, gamma)),
        ("rxx", pair, {"theta": theta}, qg.RXXGate(theta)),
        ("rzz", pair, {"theta": theta}, qg.RZZGate(theta)),
        ("ccx", triple, {}, qg.CCXGate()),
        ("cswap", ("control", "target1", "target2"), {}, qg.CSwapGate()),
        ("rccx", triple, {}, qg.RCCXGate()),
        ("rc3x", quadruple, {}, qg.RC3XGate()),
        ("c3x", quadruple, {}, qg.XGate().control(3)),
        ("c3sqrtx", quadruple, {}, qg.SXGate().control(3, annotated=False)),
        ("c4x", ("control1", "control2", "control3", "control4", "target"), {}, qg.XGate().control(4)),
    ]


@unittest.skipIf(QuantumCircuit is None, "Qiskit is not installed; optional reference tests skipped")
class QiskitReferenceTests(unittest.TestCase):
    def test_parallel_kernels_on_distant_qubits(self):
        n = 13
        reference = QuantumCircuit(n)
        instructions = []
        for q in range(n):
            reference.u(.21 + .03 * q, -.17, .48, q)
            instructions.append(inst("u", qubit=q, theta=.21 + .03 * q, phi=-.17, lam=.48))
        for kind, names, params, gate in standard_gate_cases():
            qubits = [12, 0, 7, 3, 10][:len(names)]
            reference.append(gate, qubits)
            instructions.append(inst(kind, **dict(zip(names, qubits)), **params))
        actual = _core.StatevectorSimulator().simulate(Circuit(n, instructions)).statevector
        expected = Statevector.from_instruction(reference).data
        np.testing.assert_allclose(actual, expected, atol=4e-12, rtol=4e-12)

    def test_every_standard_gate_on_complex_superpositions_and_permuted_qubits(self):
        sim = _core.StatevectorSimulator(seed=60)
        for kind, names, params, gate in standard_gate_cases():
            n = len(names) + 1  # Include an untouched spectator qubit.
            placements = (list(range(len(names))), list(range(n - 1, 0, -1)))
            for qubits in placements:
                with self.subTest(gate=kind, qubits=qubits):
                    reference = QuantumCircuit(n)
                    instructions = []
                    for q in range(n):
                        theta, phi, lam = 0.21 + 0.37 * q, -0.13 + 0.31 * q, 0.7 - 0.09 * q
                        reference.u(theta, phi, lam, q)
                        instructions.append(inst("u", qubit=q, theta=theta, phi=phi, lam=lam))
                    # Entangled inputs expose errors hidden by computational basis states.
                    for q in range(n - 1):
                        reference.cx(q, q + 1)
                        instructions.append(inst("cx", control=q, target=q + 1))
                    reference.append(gate, qubits)
                    instructions.append(inst(kind, **dict(zip(names, qubits)), **params))
                    actual = sim.simulate(Circuit(n, instructions)).statevector
                    expected = Statevector.from_instruction(reference).data
                    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)

    def test_relative_phase_gates_match_every_operator_column(self):
        sim = _core.StatevectorSimulator(seed=62)
        for kind, names, params, gate in standard_gate_cases():
            if kind not in ("rccx", "rc3x"):
                continue
            n = len(names)
            matrix = Operator(gate).data
            for basis in range(1 << n):
                with self.subTest(gate=kind, basis=basis):
                    instructions = [inst("x", qubit=q) for q in range(n) if (basis >> q) & 1]
                    instructions.append(inst(kind, **dict(zip(names, range(n))), **params))
                    actual = sim.simulate(Circuit(n, instructions)).statevector
                    np.testing.assert_allclose(actual, matrix[:, basis], atol=1e-13, rtol=1e-13)

    def test_random_circuits_match_qiskit(self):
        rng = np.random.default_rng(1907)
        cases = standard_gate_cases()
        sim = _core.StatevectorSimulator(seed=64)
        for run in range(12):
            n = 5
            reference = QuantumCircuit(n)
            instructions = []
            for _ in range(60):
                kind, names, params, gate = cases[int(rng.integers(len(cases)))]
                qubits = rng.choice(n, size=len(names), replace=False).tolist()
                reference.append(gate, qubits)
                instructions.append(inst(kind, **dict(zip(names, qubits)), **params))
            with self.subTest(circuit=run):
                actual = sim.simulate(Circuit(n, instructions)).statevector
                expected = Statevector.from_instruction(reference).data
                np.testing.assert_allclose(actual, expected, atol=4e-12, rtol=4e-12)

    def test_remote_controlled_phase_is_applied(self):
        reference = QuantumCircuit(2)
        reference.h(0)
        reference.h(1)
        reference.cp(0.82, 1, 0)
        circuit = Circuit(2, [
            inst("h", qubit=0), inst("h", qubit=1),
            inst("gate", name="remote_cu1", params=[0.82], qubits=[1, 0]),
        ])
        actual = _core.StatevectorSimulator().simulate(circuit).statevector
        np.testing.assert_allclose(actual, Statevector.from_instruction(reference).data, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
