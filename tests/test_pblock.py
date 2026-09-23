"""P-block transport, dynamic execution, factorization and reference regressions."""
import copy
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from dqsim import PBlockSimulator, StatevectorSimulator, simulate_distributed, simulate_distributed_shots
from test_statevector import Circuit, inst, conditional, standard_gate_cases, QuantumCircuit, Statevector


def distributed(n, ops, groups=None, cbits=0):
    groups = groups or {q:[q] for q in range(n)}
    owner = {q:node for node,qs in groups.items() for q in qs}
    buckets = {node:[] for node in groups}
    for op in ops:
        inner = op
        while inner['kind'] == 'conditional': inner = inner['op']
        q = next((inner[key] for key in ('qubit','control','control1','a') if key in inner), None)
        if q is None: q = (inner.get('qubits') or [next(iter(owner),0)])[0]
        buckets[owner.get(q,next(iter(groups),0))].append(op)
    circuits = {node:Circuit(n,bucket,cbits) for node,bucket in buckets.items()}
    for circuit in circuits.values(): circuit.instructions = circuit.data['instructions']
    return SimpleNamespace(circuits=circuits,qubits_per_node=groups,
                           _instruction_index={id(op):i for i,op in enumerate(ops)})


def measures(n): return [inst('measure',qubit=q,cbit=q) for q in range(n)]


def bell_pairs(n):
    return [op for q in range(0,n,2) for op in (inst('h',qubit=q),inst('cx',control=q,target=q+1))]


class PBlockTests(unittest.TestCase):
    def test_cross_node_feedback_both_execution_paths(self):
        ops=[inst('x',qubit=0),inst('measure',qubit=0,cbit=0),
             conditional(inst('x',qubit=1),1),inst('measure',qubit=1,cbit=1)]
        for prior_value in (False,True):
            current=ops[:2]+([inst('id',qubit=0)] if prior_value else [])+ops[2:]
            d=distributed(2,current,cbits=2)
            sim=PBlockSimulator(seed=3)
            self.assertEqual(sim.simulate(d).classical_bits,{0:1,1:1})
            self.assertEqual(sim.simulate_shots(d,64),{'11':64})

    def test_dynamic_feedback_correlations_and_worker_reuse(self):
        ops=bell_pairs(4)+[inst('measure',qubit=0,cbit=0),
                          conditional(inst('x',qubit=1),1),inst('reset',qubit=2)]+measures(4)
        c=Circuit(4,ops,4)
        a=PBlockSimulator(seed=91,max_parallel_shots=1).simulate_shots(c,1000)
        b=PBlockSimulator(seed=91,max_parallel_shots=4).simulate_shots(c,1000)
        self.assertEqual(a,b)
        self.assertEqual(set(a),{'0000','0001','1000','1001'})
        for count in a.values(): self.assertAlmostEqual(count/1000,.25,delta=.06)

    def test_remote_phase_gate_and_unknown_operations(self):
        ops=[inst('h',qubit=0),inst('h',qubit=1),
             inst('gate',name='remote_cu1',qubits=[0,1],params=[math.pi]),
             inst('h',qubit=0),inst('h',qubit=1)]
        c=Circuit(2,ops)
        r=PBlockSimulator().simulate(c)
        np.testing.assert_allclose(r.statevector,StatevectorSimulator().simulate(c).statevector,atol=1e-13)
        for p in r.probabilities().values(): self.assertAlmostEqual(p,.25)
        bad=[inst('gate',name='circuit-123',qubits=[0],params=[]),
             inst('gate',name='teleport',qubits=[],params=[]),inst('classical',name='unknown'),
             conditional(inst('gate',name='unknown',qubits=[],params=[]),1)]
        for op in bad:
            for c in (Circuit(2,[op],1),distributed(2,[op],cbits=1)):
                with self.assertRaises(ValueError): PBlockSimulator().simulate(c)
                with self.assertRaises(ValueError): PBlockSimulator().simulate_shots(c,0)

    def test_invalid_ownership_registers_parameters_and_queries(self):
        d=distributed(2,[inst('h',qubit=0)])
        d.qubits_per_node={0:[0],1:[0]}
        with self.assertRaises(ValueError): PBlockSimulator().simulate(d)
        bad=[inst('cx',control=0,target=0),inst('x',qubit=4),inst('measure',qubit=0,cbit=7),
             inst('u3',qubit=0,theta=.5,phi=1e308,lam=1e308),
             conditional(inst('x',qubit=99),1)]
        for op in bad:
            c=Circuit(2,[op],1)
            with self.assertRaises(ValueError): PBlockSimulator().simulate(c)
            with self.assertRaises(ValueError): PBlockSimulator().simulate_shots(c,0)
        r=PBlockSimulator().simulate(Circuit(2))
        for qs in ([3],[0,0]):
            with self.assertRaises(ValueError): r.probabilities(qs)
            with self.assertRaises(ValueError): r.counts(qubits=qs)
        with self.assertRaises(ValueError): r.amplitude('02')
        with self.assertRaises(ValueError): r.expectation_value('AB')

    def test_legacy_transport_rejects_ambiguous_order_and_repeated_objects(self):
        ops=[inst('x',qubit=0),inst('cx',control=0,target=1)]
        d=distributed(2,ops)
        np.testing.assert_equal(PBlockSimulator().simulate(d).statevector,[0,0,0,1])
        del d._instruction_index[id(ops[0])]
        with self.assertRaises(ValueError): PBlockSimulator().simulate(d)
        x=inst('x',qubit=0)
        d=distributed(1,[x,x])
        with self.assertRaises(ValueError): PBlockSimulator().simulate(d)
        d=distributed(2,ops); d._instruction_index={id(op):0 for op in ops}
        with self.assertRaises(ValueError): PBlockSimulator().simulate_shots(d,0)
        d=distributed(2,ops); d.circuits[0].instructions=[]
        with self.assertRaises(ValueError): PBlockSimulator().simulate(d)

    def test_explicit_operation_ids_and_shared_cross_node_copies(self):
        x=inst('x',qubit=0);cx=inst('cx',control=0,target=1)
        c0=Circuit(2,[x,cx]);c1=Circuit(2,[copy.deepcopy(cx)])
        d=SimpleNamespace(circuits={0:c0,1:c1},qubits_per_node={0:[0],1:[1]},
                          operation_ids={0:['x','cx'],1:['cx']},operation_order={'x':0,'cx':1})
        np.testing.assert_equal(PBlockSimulator().simulate(d).statevector,[0,0,0,1])
        # Same Python object may occur twice when distinct explicit IDs identify occurrences.
        d=SimpleNamespace(circuits={0:Circuit(1,[x,x])},qubits_per_node={0:[0]},
                          operation_ids={0:['first','second']},operation_order={'first':0,'second':1})
        np.testing.assert_equal(PBlockSimulator().simulate(d).statevector,[1,0])
        d.operation_order['second']=-1
        with self.assertRaises(ValueError): PBlockSimulator().simulate(d)
        d=SimpleNamespace(circuits={0:Circuit(2,[cx]),1:Circuit(2,[inst('cz',control=0,target=1)])},
                          qubits_per_node={0:[0],1:[1]},operation_ids={0:['gate'],1:['gate']},operation_order={'gate':0})
        with self.assertRaises(ValueError): PBlockSimulator().simulate(d)

    def test_partition_independence_and_large_product_state(self):
        n=100;ops=[inst('h',qubit=q) for q in range(n)]
        sim=PBlockSimulator(seed=1,max_memory_mb=1,max_block_qubits=2)
        r=sim.simulate(distributed(n,ops,{0:list(range(n))}))
        self.assertEqual(len(r.block_qubits),n)
        self.assertEqual(r.diagnostics['live_amplitudes'],2*n)
        self.assertAlmostEqual(r.expectation_value('X'*n),1.,places=12)
        self.assertAlmostEqual(r.amplitude('0'*n),2**(-n/2),places=25)
        counts=r.counts(20,seed=3);self.assertEqual(sum(counts.values()),20)
        self.assertTrue(all(len(key)==n for key in counts))
        with self.assertRaises(MemoryError): _=r.statevector
        with self.assertRaises(MemoryError): r.probabilities()
        p=r.probabilities([99,0]);self.assertEqual(len(p),4)
        for v in p.values(): self.assertAlmostEqual(v,.25,places=12)
        small=ops[:8]+measures(8)
        a=sim.simulate_shots(distributed(8,small,{0:list(range(8))},8),100)
        b=sim.simulate_shots(distributed(8,small,cbits=8),100)
        self.assertEqual(a,b)

    def test_large_independent_bell_pairs(self):
        n=100;sim=PBlockSimulator(seed=2,max_memory_mb=1,max_block_qubits=2)
        r=sim.simulate(Circuit(n,bell_pairs(n)))
        self.assertEqual(sorted(map(len,r.block_qubits)),[2]*50)
        self.assertEqual(r.diagnostics['peak_block_qubits'],2)
        self.assertAlmostEqual(r.expectation_value('X'*n),1.,places=12)
        self.assertAlmostEqual(r.expectation_value('Y'*n),-1. if (n//2)%2 else 1.,places=12)
        counts=sim.simulate_shots(Circuit(n,bell_pairs(n)+measures(n),n),100)
        for key in counts:
            self.assertTrue(all(key[i]==key[i+1] for i in range(0,n,2)))
        self.assertEqual(sum(counts.values()),100)

    def test_measurement_reset_release_blocks_and_preserve_phase(self):
        ops=[inst('h',qubit=0),inst('s',qubit=0),inst('cx',control=0,target=1),inst('measure',qubit=0,cbit=0)]
        for seed in range(6):
            r=PBlockSimulator(seed=seed).simulate(Circuit(2,ops,1))
            self.assertEqual(sorted(map(len,r.block_qubits)),[1,1])
            bit=r.classical_bits[0]
            expected=np.zeros(4,dtype=complex);expected[3 if bit else 0]=1j if bit else 1
            np.testing.assert_allclose(r.statevector,expected,atol=1e-13)
        r=PBlockSimulator(seed=2).simulate(Circuit(2,ops[:-1]+[inst('reset',qubit=0)]))
        self.assertEqual(sorted(map(len,r.block_qubits)),[1,1])
        self.assertAlmostEqual(r.probabilities([0]).get(0,0),1.)

    def test_noop_conditions_known_controls_and_swaps_do_not_merge(self):
        ops=[inst('cx',control=q,target=q+1) for q in range(99)]
        ops += [conditional(inst('cx',control=0,target=99),1),
                inst('gate',name='remote_barrier',qubits=list(range(100)),params=[]),
                inst('x',qubit=0),inst('swap',a=0,b=99)]
        r=PBlockSimulator(max_memory_mb=1,max_block_qubits=1).simulate(Circuit(100,ops,1))
        self.assertEqual(r.diagnostics['merge_calls'],0)
        self.assertEqual(r.amplitude('1'+'0'*99),1)
        self.assertEqual(r.diagnostics['skipped_controlled_gates'],99)

    def test_controlled_eigenstates_and_identity_diagonals_stay_separate(self):
        ops=[inst('h',qubit=0),inst('h',qubit=1),inst('cx',control=0,target=1),
             inst('rzz',a=0,b=1,theta=0.)]
        r=PBlockSimulator(max_block_qubits=1).simulate(Circuit(2,ops))
        self.assertEqual(r.diagnostics['merge_calls'],0)
        np.testing.assert_allclose(r.statevector,[.5,.5,.5,.5],atol=1e-13)

    def test_swaps_within_and_between_entangled_blocks(self):
        ops=bell_pairs(4)+[inst('s',qubit=0),inst('swap',a=0,b=3),inst('swap',a=1,b=3),
                           inst('h',qubit=3),inst('cx',control=3,target=2)]
        c=Circuit(4,ops)
        np.testing.assert_allclose(PBlockSimulator().simulate(c).statevector,StatevectorSimulator().simulate(c).statevector,atol=1e-13)

    def test_optional_separation_and_norm(self):
        ops=bell_pairs(2)+[inst('cx',control=0,target=1)]
        plain=PBlockSimulator().simulate(Circuit(2,ops))
        exact=PBlockSimulator(split_separable=True).simulate(Circuit(2,ops))
        self.assertEqual(len(plain.block_qubits),1)
        self.assertEqual(len(exact.block_qubits),2)
        np.testing.assert_allclose(exact.statevector,plain.statevector,atol=1e-13)
        entangled=Circuit(2,[inst('ry',qubit=0,theta=.02),inst('cx',control=0,target=1)])
        r=PBlockSimulator(split_separable=True,separation_tolerance=.001).simulate(entangled)
        self.assertEqual(len(r.block_qubits),2)
        self.assertGreater(r.diagnostics['separation_residual'],0)
        self.assertAlmostEqual(r.diagnostics['norm_squared'],1.)
        self.assertEqual(len(PBlockSimulator(split_separable=True).simulate(entangled).block_qubits),1)

    def test_memory_and_options_fail_before_large_allocations(self):
        for opts in ({'max_memory_mb':0},{'max_parallel_shots':0},{'max_block_qubits':0},
                     {'max_split_qubits':0},{'separation_tolerance':float('nan')}):
            with self.assertRaises(ValueError): PBlockSimulator(**opts)
        ops=[inst('h',qubit=0)]+[inst('cx',control=q,target=q+1) for q in range(19)]
        for sim in (PBlockSimulator(max_block_qubits=4),PBlockSimulator(max_memory_mb=1)):
            with self.assertRaises(MemoryError): sim.simulate(Circuit(20,ops))
        r=PBlockSimulator(max_memory_mb=1).simulate(Circuit(20))
        with self.assertRaises(MemoryError): _=r.statevector
        with self.assertRaises(MemoryError): PBlockSimulator(max_memory_mb=1).simulate(Circuit(100000))
        with self.assertRaises(MemoryError): PBlockSimulator(max_memory_mb=1).simulate_shots(Circuit(1,cbits=100000),0)

    def test_query_order_and_sparse_physical_labels(self):
        c=Circuit(0,[inst('x',qubit=100),inst('h',qubit=7)])
        c.data['qregs']={'a':{'name':'a','base':7,'size':1},'b':{'name':'b','base':100,'size':1}}
        r=PBlockSimulator().simulate(c)
        self.assertEqual(r.physical_qubits,[7,100])
        np.testing.assert_allclose(r.statevector,[0,0,2**-.5,2**-.5],atol=1e-13)
        self.assertAlmostEqual(r.probabilities([7,100])[1],.5)
        self.assertEqual(set(r.counts(100,qubits=[7,100],seed=1)),{'01','11'})
        self.assertAlmostEqual(r.amplitude('10'),2**-.5)
        self.assertAlmostEqual(r.expectation_value('ZX'),-1.)
        self.assertAlmostEqual(r.probabilities([])[0],1.)

    def test_terminal_repeated_measurements_and_classical_overwrites(self):
        ops=bell_pairs(2)+[inst('measure',qubit=1,cbit=0),inst('measure',qubit=1,cbit=1),inst('measure',qubit=0,cbit=0)]
        for terminal in (False,True):
            counts=PBlockSimulator(seed=5,sample_terminal=terminal).simulate_shots(Circuit(2,ops,3),1000)
            self.assertEqual(set(counts),{'000','011'})
            self.assertAlmostEqual(counts['011']/1000,.5,delta=.06)

    def test_tiny_gates_are_preserved(self):
        r=PBlockSimulator().simulate(Circuit(1,[inst('rx',qubit=0,theta=1e-11)]))
        self.assertAlmostEqual(r.statevector[1].imag,-5e-12,delta=1e-25)

    def test_empty_result_and_wrappers(self):
        r=simulate_distributed(Circuit(0),profile=True,max_memory_mb=1)
        np.testing.assert_equal(r.statevector,[1])
        self.assertEqual(r.amplitude(''),1)
        self.assertEqual(r.probabilities(),{0:1})
        self.assertEqual(r.counts(7),{'':7})
        self.assertEqual(r.expectation_value(''),1)
        self.assertIn('total_time',r.profile)
        self.assertEqual(simulate_distributed_shots(Circuit(0),shots=7),{'':7})
        self.assertEqual(PBlockSimulator().simulate_shots(Circuit(0),0),{})
        self.assertEqual(PBlockSimulator().simulate_shots(Circuit(1,cbits=3),7),{'000':7})

    def test_sampling_and_prefix_fall_back_when_memory_is_tight(self):
        # Two independent 16-qubit GHZ blocks fit in 3 MiB with merge scratch,
        # but their combined sampling tables and a cloned prefix do not.
        ops=[]
        for start in (0,16):
            ops.append(inst('h',qubit=start))
            ops.extend(inst('cx',control=q,target=q+1) for q in range(start,start+15))
        ops.extend([inst('measure',qubit=0,cbit=0),inst('measure',qubit=16,cbit=1)])
        with tempfile.TemporaryDirectory() as tmp:
            old=os.getcwd()
            try:
                os.chdir(tmp)
                counts=PBlockSimulator(seed=3,max_memory_mb=3,max_parallel_shots=4).simulate_shots(Circuit(32,ops,2),4,profile=True)
                report=json.loads(next(Path('dqsim_profiles').glob('*.json')).read_text())
            finally: os.chdir(old)
        self.assertEqual(sum(counts.values()),4)
        self.assertEqual(report['execution_strategy'],'trajectories')
        self.assertEqual(report['deterministic_prefix_ops'],0)
        self.assertEqual(report['parallel_shots'],1)
        self.assertLessEqual(report['peak_working_bytes'],3*1024**2)

    def test_profiles_confirm_shared_evolution(self):
        ops=bell_pairs(8)+measures(8)
        with tempfile.TemporaryDirectory() as tmp:
            old=os.getcwd()
            try:
                os.chdir(tmp)
                PBlockSimulator().simulate_shots(Circuit(8,ops,8),100,profile=True)
                p=json.loads(next(Path('dqsim_profiles').glob('*.json')).read_text())
            finally: os.chdir(old)
        self.assertEqual(p['execution_strategy'],'terminal_sampling')
        self.assertEqual(p['merge_calls'],4)
        self.assertEqual(p['peak_block_qubits'],2)
        self.assertLess(p['peak_working_bytes'],1024**2)


@unittest.skipIf(QuantumCircuit is None,'Optional Qiskit dependency missing')
class PBlockReferenceTests(unittest.TestCase):
    def test_all_standard_gates_on_complex_entangled_inputs(self):
        for kind,names,params,gate in standard_gate_cases():
            n=6;qc=QuantumCircuit(n);ops=[]
            for q in range(n):
                qc.u(.21+.27*q,-.17,.61,q)
                ops.append(inst('u',qubit=q,theta=.21+.27*q,phi=-.17,lam=.61))
            for q in range(0,n,2):qc.cx(q,q+1);ops.append(inst('cx',control=q,target=q+1))
            qs=[5,0,3,2,4][:len(names)]
            qc.append(gate,qs);ops.append(inst(kind,**dict(zip(names,qs)),**params))
            with self.subTest(gate=kind):
                r=PBlockSimulator().simulate(distributed(n,ops))
                np.testing.assert_allclose(r.statevector,Statevector.from_instruction(qc).data,atol=3e-12)

    def test_parallel_merges_and_large_block_kernels(self):
        n=13
        ops=[inst('h',qubit=q) for q in range(n)]
        # CZ creates one connected block from initially separate superpositions.
        ops += [inst('cz',control=q,target=(q+5)%n) for q in range(n)]
        ops += [inst('swap',a=12,b=0),inst('rxx',a=11,b=2,theta=.47),
                inst('c4x',control1=1,control2=12,control3=6,control4=3,target=8)]
        c=Circuit(n,ops)
        r=PBlockSimulator(max_memory_mb=8).simulate(c)
        self.assertEqual(r.diagnostics['peak_block_qubits'],n)
        np.testing.assert_allclose(r.statevector,StatevectorSimulator().simulate(c).statevector,atol=3e-12)

    def test_random_circuits_and_compact_queries(self):
        rng=np.random.default_rng(923);cases=standard_gate_cases()
        for trial in range(12):
            n=7;qc=QuantumCircuit(n);ops=[]
            for _ in range(70):
                kind,names,params,gate=cases[int(rng.integers(len(cases)))]
                qs=[int(q) for q in rng.choice(n,len(names),replace=False)]
                qc.append(gate,qs);ops.append(inst(kind,**dict(zip(names,qs)),**params))
            r=PBlockSimulator(split_separable=bool(trial%2)).simulate(Circuit(n,ops))
            v=Statevector.from_instruction(qc).data
            np.testing.assert_allclose(r.statevector,v,atol=5e-12)
            expected=StatevectorSimulator().simulate(Circuit(n,ops)).probabilities([6,0,3])
            actual=r.probabilities([6,0,3])
            for k in actual.keys()|expected.keys(): self.assertAlmostEqual(actual.get(k,0),expected.get(k,0),places=12)
            pauli='XYZIYZX';mats={'I':np.eye(2),'X':np.array([[0,1],[1,0]]),'Y':np.array([[0,-1j],[1j,0]]),'Z':np.diag([1,-1])}
            mat=np.array([[1]])
            for p in pauli:mat=np.kron(mat,mats[p])
            self.assertAlmostEqual(r.expectation_value(pauli),np.vdot(v,mat@v).real,places=12)


if __name__=='__main__': unittest.main()
