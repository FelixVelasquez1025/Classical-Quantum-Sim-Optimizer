"""OpenQASM transport semantics and import-policy regressions."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from dqsim import (load_qasm, from_qiskit, ImportedCircuit, QASMImportError,
                   StatevectorSimulator, MpsSimulator, PBlockSimulator, StabilizerSimulator)
try:
    from qiskit import qasm2, QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.quantum_info import Statevector
except ImportError:
    qasm2 = None


@unittest.skipIf(qasm2 is None, 'Optional Qiskit parser is not installed')
class QASMTests(unittest.TestCase):
    def load(self, text, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'circuit.qasm'
            path.write_text(text)
            return load_qasm(path, **kwargs)

    def test_register_offsets_custom_gates_and_three_qubit_lowering(self):
        text = '''OPENQASM 2.0;
include "qelib1.inc";
qreg a[1]; qreg b[3]; creg first[2]; creg last[2];
gate custom(theta) x,y,z { h x; ry(theta) y; ccx x,y,z; }
x a[0]; h b[2]; custom(pi/3) b[2],a[0],b[0];
cswap a[0],b[2],b[1];
'''
        imported = self.load(text)
        self.assertEqual(imported.num_qubits, 4)
        self.assertEqual(imported.num_cbits, 4)
        self.assertEqual(imported.data['qregs']['b']['base'], 1)
        self.assertEqual(imported.data['cregs']['last']['base'], 2)
        self.assertNotIn('ccx', {op['kind'] for op in imported.data['instructions']})
        self.assertNotIn('cswap', {op['kind'] for op in imported.data['instructions']})
        reference = Statevector(qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)).data
        for sim in (StatevectorSimulator(), MpsSimulator(truncation_threshold=0), PBlockSimulator()):
            np.testing.assert_allclose(sim.simulate(imported).statevector, reference, atol=3e-12)
        self.assertFalse(StabilizerSimulator().supports(imported))

    def test_feedback_reset_and_register_order_all_backends(self):
        text = '''OPENQASM 2.0; include "qelib1.inc";
qreg a[1]; qreg b[1]; creg first[1]; creg last[2];
gate pair x,y { x x; x y; }
x a[0]; measure a[0] -> last[1];
if(last==2) pair a[0],b[0];
measure a[0] -> first[0]; measure b[0] -> last[0];
reset b[0];
'''
        circuit = self.load(text)
        for sim in (StatevectorSimulator(seed=1), MpsSimulator(seed=1),
                    PBlockSimulator(seed=1), StabilizerSimulator(seed=1)):
            self.assertEqual(sim.simulate_shots(circuit, 32), {'110': 32})
        self.assertTrue(StabilizerSimulator().supports(circuit))

    def test_measurement_broadcast_and_no_added_measurements(self):
        text = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];'
        unitary = self.load(text)
        self.assertEqual(unitary.num_cbits, 0)
        self.assertEqual(StatevectorSimulator().simulate_shots(unitary, 5), {'': 5})
        measured = self.load(text + 'creg c[2]; measure q -> c;')
        for sim in (StatevectorSimulator(seed=17), MpsSimulator(seed=17),
                    PBlockSimulator(seed=17), StabilizerSimulator(seed=17)):
            self.assertEqual(set(sim.simulate_shots(measured, 256)), {'00','11'})

    def test_custom_definition_named_like_legacy_gate_is_respected(self):
        # Standard SWAP would leave |00> unchanged; this explicit definition flips.
        circuit = self.load('''OPENQASM 2.0; include "qelib1.inc";
gate swap a,b { x a; } qreg q[2]; swap q[0],q[1];''')
        self.assertEqual(StatevectorSimulator().simulate(circuit).probabilities(), {1: 1.})
        circuit = self.load('''OPENQASM 2.0; include "qelib1.inc";
gate sx a { z a; } qreg q[1]; h q[0]; sx q[0]; h q[0];''')
        self.assertAlmostEqual(StatevectorSimulator().simulate(circuit).probabilities()[1], 1.)

    def test_include_provenance_and_legacy_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'gates.inc').write_text('gate swap a,b { x b; }')
            path = root/'input.qasm'
            path.write_text('OPENQASM 2.0; include "qelib1.inc"; include "gates.inc"; qreg q[2]; swap q[0],q[1];')
            circuit = load_qasm(path)
            self.assertEqual(len(circuit.metadata['includes']), 1)
            self.assertEqual(StatevectorSimulator().simulate(circuit).probabilities(), {2: 1.})
            self.assertEqual(len(circuit.metadata['source_sha256']), 64)
            self.assertEqual(len(circuit.metadata['includes'][0]['sha256']), 64)
        text='OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; sx q[0];'
        self.load(text)
        with self.assertRaises(QASMImportError): self.load(text, legacy_gates=False)

    def test_limits_opaque_and_malformed_sources(self):
        text='OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; h q[0]; h q[0];'
        with self.assertRaises(QASMImportError): self.load(text, max_source_bytes=10)
        with self.assertRaises(QASMImportError): self.load(text, max_instructions=1)
        with self.assertRaises(QASMImportError): self.load('OPENQASM 2.0; qreg reg[1]; measure q[0] -> c[0];')
        with self.assertRaises(QASMImportError): self.load('OPENQASM 2.0; opaque mystery q; qreg q[1]; mystery q[0];')
        with self.assertRaises(QASMImportError): self.load('OPENQASM 3.0; qubit q;')
        with self.assertRaises(ValueError): self.load(text, max_instructions=0)
        with self.assertRaises(ValueError): self.load(text, max_source_bytes=0)

    def test_condition_snapshot_cannot_be_silently_changed(self):
        qr=QuantumRegister(2,'q');cr=ClassicalRegister(2,'c');circuit=QuantumCircuit(qr,cr)
        with circuit.if_test((cr,1)):
            circuit.measure(qr[0],cr[0])
            circuit.x(qr[1])
        with self.assertRaisesRegex(QASMImportError,'modifies its tested bits'):
            from_qiskit(circuit)
        circuit=QuantumCircuit(qr,cr)
        with circuit.if_test((cr,1)):
            circuit.x(qr[1]);circuit.measure(qr[0],cr[0])
        converted=from_qiskit(circuit)
        self.assertEqual(len(converted.data['instructions']),2)

    def test_explicit_global_phase_and_else_rejected(self):
        circuit=QuantumCircuit(1);circuit.global_phase=.3
        with self.assertRaisesRegex(QASMImportError,'global phase'): from_qiskit(circuit)
        circuit=QuantumCircuit(1,1)
        with circuit.if_test((circuit.cregs[0],0)) as otherwise:
            circuit.x(0)
        with otherwise:
            circuit.h(0)
        with self.assertRaisesRegex(QASMImportError,'Else'): from_qiskit(circuit)

    def test_transport_artifact_roundtrip(self):
        circuit=self.load('OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; x q[0];')
        rebuilt=ImportedCircuit(json.loads(circuit.model_dump_json()),dict(circuit.metadata))
        self.assertEqual(StabilizerSimulator().simulate(rebuilt).probabilities(),{1:1.})
