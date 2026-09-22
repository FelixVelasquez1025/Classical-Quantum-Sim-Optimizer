"""MPS correctness, approximation, compact queries, and execution strategies."""
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from dqsim import MpsSimulator, MpsResult, StatevectorSimulator, simulate_monolithic, simulate_monolithic_shots
from test_statevector import Circuit, inst, conditional, standard_gate_cases, QuantumCircuit, Statevector


def ghz(n):
    return [inst('h', qubit=0)] + [inst('cx', control=q, target=q+1) for q in range(n-1)]


def measured(n):
    return [inst('measure', qubit=q, cbit=q) for q in range(n)]


class MpsTests(unittest.TestCase):
    def test_uncapped_numerical_regression(self):
        ops = [inst('u3', qubit=3, theta=.731, phi=-.412, lam=1.137),
               inst('u', qubit=3, theta=.731, phi=-.412, lam=1.137),
               inst('swap', a=3, b=0), inst('sxdg', qubit=0),
               inst('cy', control=0, target=3), inst('cp', control=2, target=3, lam=1.137),
               inst('ch', control=2, target=3), inst('rxx', a=1, b=0, theta=.731),
               inst('cx', control=0, target=2)]
        circuit = Circuit(4, ops)
        expected = StatevectorSimulator().simulate(circuit).statevector
        for threshold in (0, 1e-12):
            result = MpsSimulator(truncation_threshold=threshold).simulate(circuit)
            np.testing.assert_allclose(result.statevector, expected, atol=2e-12)
            self.assertAlmostEqual(result.diagnostics['norm_squared'], 1., places=12)

    def test_truncation_preserves_norm(self):
        ops = [inst('ry', qubit=0, theta=math.pi/2), inst('cx', control=0, target=1),
               inst('ry', qubit=2, theta=.6), inst('cx', control=2, target=3),
               inst('cx', control=1, target=2), inst('cx', control=2, target=1)]
        for cap in (1, 2):
            result = MpsSimulator(max_bond_dimension=cap).simulate(Circuit(4, ops))
            self.assertAlmostEqual(sum(result.probabilities().values()), 1., places=12)
            self.assertAlmostEqual(np.vdot(result.statevector, result.statevector).real, 1., places=12)
            self.assertLessEqual(max(result.bond_dimensions), cap)
            self.assertGreater(result.diagnostics['discarded_weight'], 0)

    def test_discarded_weight_has_physical_meaning(self):
        theta = .2
        c = Circuit(2, [inst('ry', qubit=0, theta=theta), inst('cx', control=0, target=1)])
        exact = StatevectorSimulator().simulate(c).statevector
        r = MpsSimulator(max_discarded_weight=.02).simulate(c)
        self.assertEqual(r.bond_dimensions, [1])
        self.assertAlmostEqual(r.diagnostics['discarded_weight'], math.sin(theta/2)**2, places=13)
        self.assertAlmostEqual(r.fidelity(exact), 1-r.diagnostics['discarded_weight'], places=13)
        self.assertAlmostEqual(r.diagnostics['norm_squared'], 1., places=13)

    def test_large_ghz_stays_compact(self):
        n = 100
        sim = MpsSimulator(seed=31, max_memory_mb=4)
        r = sim.simulate(Circuit(n, ghz(n)))
        self.assertIsInstance(r, MpsResult)
        self.assertEqual(r.num_qubits, n)
        self.assertEqual(max(r.bond_dimensions), 2)
        self.assertLess(r.diagnostics['peak_working_bytes'], 4*1024*1024)
        self.assertAlmostEqual(abs(r.amplitude('0'*n)), 2**-.5, places=12)
        self.assertAlmostEqual(r.expectation_value('X'*n), 1., places=12)
        self.assertAlmostEqual(r.expectation_value('Z'+'I'*98+'Z'), 1., places=12)
        p = r.probabilities([99, 0])
        self.assertAlmostEqual(p.get(0, 0), .5, places=12)
        self.assertAlmostEqual(p.get(3, 0), .5, places=12)
        for counts in (r.counts(100, seed=4), sim.simulate_shots(Circuit(n, ghz(n)+measured(n), n), 100)):
            self.assertEqual(set(counts), {'0'*n, '1'*n})
            self.assertEqual(sum(counts.values()), 100)
        with self.assertRaises(MemoryError):
            _ = r.statevector
        with self.assertRaises(MemoryError):
            r.probabilities()

    def test_large_dynamic_measurement_and_reset(self):
        n = 100
        ops = ghz(n) + [inst('measure', qubit=99, cbit=0),
                        conditional(inst('x', qubit=0), 1), inst('measure', qubit=0, cbit=1),
                        inst('reset', qubit=50), inst('measure', qubit=50, cbit=2)]
        c = Circuit(n, ops, 3)
        serial = MpsSimulator(seed=19, max_memory_mb=8).simulate_shots(c, 100)
        parallel = MpsSimulator(seed=19, max_memory_mb=8, max_parallel_shots=3).simulate_shots(c, 100)
        self.assertEqual(serial, parallel)
        self.assertEqual(set(serial), {'000', '001'})
        r = MpsSimulator(seed=19, max_memory_mb=8).simulate(c)
        self.assertEqual(r.classical_bits[1], 0)
        self.assertEqual(r.classical_bits[2], 0)
        self.assertAlmostEqual(r.diagnostics['norm_squared'], 1., places=12)

    def test_terminal_repeated_measurements_and_classical_overwrites(self):
        ops = ghz(3) + [inst('measure', qubit=2, cbit=0), inst('measure', qubit=2, cbit=1),
                       inst('measure', qubit=0, cbit=2), inst('measure', qubit=1, cbit=0)]
        c = Circuit(3, ops, 4)
        for terminal in (True, False):
            counts = MpsSimulator(seed=2, sample_terminal=terminal).simulate_shots(c, 500)
            self.assertEqual(set(counts), {'0000', '0111'})
            self.assertAlmostEqual(counts['0111']/500, .5, delta=.08)
        self.assertEqual(MpsSimulator().simulate_shots(Circuit(3, ghz(3)), 5), {'': 5})
        self.assertEqual(MpsSimulator().simulate_shots(c, 0), {})

    def test_queries_respect_logical_order(self):
        c = Circuit(4, [inst('x', qubit=0), inst('swap', a=0, b=3), inst('cx', control=3, target=1)])
        r = MpsSimulator().simulate(c)
        self.assertNotEqual(r.qubit_order, list(range(4)))
        self.assertAlmostEqual(r.probabilities([1, 0]).get(2,0), 1., places=12)
        self.assertEqual(r.counts(7, qubits=[0, 1], seed=0), {'01': 7})
        self.assertAlmostEqual(r.amplitude('1010'), 1, places=12)
        self.assertAlmostEqual(r.expectation_value('ZIII'), -1, places=12)
        self.assertAlmostEqual(r.probabilities([])[0], 1., places=12)
        for query in ([4], [0, 0]):
            with self.assertRaises(ValueError): r.probabilities(query)
            with self.assertRaises(ValueError): r.counts(qubits=query)
        with self.assertRaises(ValueError): r.amplitude('00')
        with self.assertRaises(ValueError): r.expectation_value('ABCD')
        # Returned NumPy array is independent of the MPS.
        v = r.statevector
        v[:] = 0
        self.assertAlmostEqual(r.amplitude('1010'), 1, places=12)

    def test_pauli_y_and_marginals(self):
        ops = [inst('h', qubit=0), inst('s', qubit=0), inst('ry', qubit=1, theta=.43),
               inst('cx', control=0, target=2), inst('swap', a=1, b=2)]
        c = Circuit(3, ops)
        r = MpsSimulator().simulate(c)
        v = StatevectorSimulator().simulate(c)
        for qs in (None, [0], [2,0], [0,2], []):
            a,b=r.probabilities(qs),v.probabilities(qs)
            for k in a.keys()|b.keys(): self.assertAlmostEqual(a.get(k,0),b.get(k,0),places=12)
        mats={'I':np.eye(2), 'X':np.array([[0,1],[1,0]]), 'Y':np.array([[0,-1j],[1j,0]]), 'Z':np.diag([1,-1])}
        for pauli in ('YIY','IZX','ZII','XXX','III'):
            op=np.kron(np.kron(mats[pauli[0]],mats[pauli[1]]),mats[pauli[2]])
            self.assertAlmostEqual(r.expectation_value(pauli),np.vdot(v.statevector,op@v.statevector).real,places=12)

    def test_invalid_inputs_and_untaken_branches(self):
        for options in ({'max_bond_dimension':0}, {'truncation_threshold':-1}, {'truncation_threshold':float('nan')},
                        {'max_discarded_weight':1}, {'max_memory_mb':0}, {'max_parallel_shots':0}):
            with self.assertRaises((ValueError, OverflowError)): MpsSimulator(**options)
            with self.assertRaises((ValueError, OverflowError)): simulate_monolithic(Circuit(2), mode='mps', **options)
        bad = [inst('cx',control=0,target=0), inst('x',qubit=3), inst('classical',name='unknown'),
               inst('gate',name='circuit-123',qubits=[0],params=[]),
               inst('gate',name='remote_cx',qubits=[0],params=[]),
               inst('gate',name='remote_cu1',qubits=[0,1],params=[]),
               conditional(inst('x',qubit=99),1),
               conditional(inst('ccx',control1=0,control2=1,target=2),1)]
        for op in bad:
            c=Circuit(3,[op],1)
            with self.subTest(op=op):
                with self.assertRaises(ValueError): MpsSimulator().simulate(c)
                with self.assertRaises(ValueError): MpsSimulator().simulate_shots(c,0)
        with self.assertRaises(MemoryError): MpsSimulator(max_memory_mb=1).simulate(Circuit(20000))
        r=MpsSimulator(max_memory_mb=1).simulate(Circuit(20))
        with self.assertRaises(MemoryError): _=r.statevector

    def test_remote_parameterized_gates(self):
        for name in ('remote_cu1','remote_rzz'):
            c=Circuit(4,[inst('h',qubit=q) for q in range(4)]+[inst('gate',name=name,qubits=[3,0],params=[.7])])
            np.testing.assert_allclose(MpsSimulator().simulate(c).statevector,StatevectorSimulator().simulate(c).statevector,atol=1e-12)

    def test_truncated_random_states_and_sampling(self):
        rng = np.random.default_rng(823)
        for trial in range(4):
            ops = []
            for _ in range(30):
                a,b = [int(q) for q in rng.choice(6,2,replace=False)]
                ops.extend([inst('ry',qubit=a,theta=float(rng.normal())),
                            inst('rz',qubit=b,phi=float(rng.normal())),
                            inst('cx',control=a,target=b)])
            sim = MpsSimulator(max_bond_dimension=2, seed=trial)
            result = sim.simulate(Circuit(6,ops))
            self.assertAlmostEqual(result.diagnostics['norm_squared'],1.,places=12)
            self.assertAlmostEqual(sum(result.probabilities().values()),1.,places=12)
            self.assertLessEqual(max(result.bond_dimensions),2)
            # Both the direct sampler and per-shot collapse must sample this
            # same approximate state, without further unitary truncations.
            expected = result.probabilities([0,5])
            circuit = Circuit(6,ops+[inst('measure',qubit=5,cbit=0),inst('measure',qubit=0,cbit=1)],2)
            for terminal in (True,False):
                counts = MpsSimulator(max_bond_dimension=2,seed=trial,sample_terminal=terminal).simulate_shots(circuit,2000)
                for basis in range(4):
                    self.assertAlmostEqual(counts.get(format(basis,'02b'),0)/2000,expected.get(basis,0),delta=.05)

    def test_empty_state_and_wrapper(self):
        r=simulate_monolithic(Circuit(0),mode='mps',profile=True,max_discarded_weight=0)
        np.testing.assert_equal(r.statevector,[1+0j])
        self.assertEqual(r.amplitude(''),1)
        self.assertEqual(r.counts(5),{'':5})
        self.assertEqual(r.probabilities(),{0:1})
        self.assertEqual(r.expectation_value(''),1)
        self.assertIn('total_time',r.profile)
        self.assertIsNone(MpsSimulator().simulate(Circuit(1)).profile)
        self.assertEqual(simulate_monolithic_shots(Circuit(0),mode='mps',shots=5),{'':5})

    def test_profile_reports_reuse_and_routing(self):
        ops = [inst('h',qubit=0)] + [inst('cx',control=0,target=19)]*3 + measured(20)
        with tempfile.TemporaryDirectory() as directory:
            old=os.getcwd()
            try:
                os.chdir(directory)
                MpsSimulator(seed=1).simulate_shots(Circuit(20,ops,20),100,profile=True)
                report=json.loads(next(Path('dqsim_profiles').glob('mps*.json')).read_text())
            finally: os.chdir(old)
        self.assertEqual(report['execution_strategy'],'terminal_sampling')
        self.assertEqual(report['routing_swaps'],18)
        self.assertEqual(report['svd_calls'],21)
        self.assertEqual(report['parallel_shots'],1)


@unittest.skipIf(QuantumCircuit is None,'Qiskit optional reference dependency missing')
class MpsReferenceTests(unittest.TestCase):
    def test_all_one_and_two_qubit_gates(self):
        for kind,names,params,gate in standard_gate_cases():
            if len(names)>2: continue
            for positions in ([0,3],[3,0]):
                qc=QuantumCircuit(4); ops=[]
                for q in range(4):
                    qc.u(.21+.37*q,-.13+.31*q,.7-.09*q,q)
                    ops.append(inst('u',qubit=q,theta=.21+.37*q,phi=-.13+.31*q,lam=.7-.09*q))
                for q in range(3):
                    qc.cx(q,q+1); ops.append(inst('cx',control=q,target=q+1))
                qs=positions[:len(names)]
                qc.append(gate,qs); ops.append(inst(kind,**dict(zip(names,qs)),**params))
                with self.subTest(gate=kind,positions=qs):
                    np.testing.assert_allclose(MpsSimulator().simulate(Circuit(4,ops)).statevector,Statevector.from_instruction(qc).data,atol=3e-12)

    def test_random_routing_and_canonicalization(self):
        rng=np.random.default_rng(8372)
        cases=[c for c in standard_gate_cases() if len(c[1])<=2]
        for n in (4,5,7):
            for trial in range(6):
                qc=QuantumCircuit(n); ops=[]
                for _ in range(75):
                    kind,names,params,gate=cases[int(rng.integers(len(cases)))]
                    qs=[int(x) for x in rng.choice(n,len(names),replace=False)]
                    qc.append(gate,qs); ops.append(inst(kind,**dict(zip(names,qs)),**params))
                c=Circuit(n,ops); expected=Statevector.from_instruction(qc).data
                for threshold in (0,1e-12):
                    with self.subTest(n=n,trial=trial,threshold=threshold):
                        actual=MpsSimulator(truncation_threshold=threshold).simulate(c).statevector
                        np.testing.assert_allclose(actual,expected,atol=5e-12,rtol=5e-12)


if __name__=='__main__': unittest.main()
