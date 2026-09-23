"""Stabilizer correctness, Clifford eligibility, compact queries and memory limits."""
import itertools
import json
import math
import os
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
from dqsim import StabilizerSimulator, StabilizerResult, simulate_monolithic, simulate_monolithic_shots
from test_statevector import Circuit, inst, conditional, QuantumCircuit, Statevector, qg


def measures(n):
    return [inst('measure', qubit=q, cbit=q) for q in range(n)]


def ghz(n):
    return [inst('h', qubit=0)] + [inst('cx', control=q, target=q+1) for q in range(n-1)]


def profile_run(sim, circuit, shots):
    original = os.getcwd()
    with tempfile.TemporaryDirectory() as directory:
        try:
            os.chdir(directory)
            counts = sim.simulate_shots(circuit, shots, profile=True)
            report = json.loads(next(Path('dqsim_profiles').glob('*.json')).read_text())
            return counts, report
        finally:
            os.chdir(original)


class StabilizerTests(unittest.TestCase):
    def test_wrong_pauli_phase_regression_both_strategies(self):
        ops = [inst('cx',control=0,target=2), inst('h',qubit=2), inst('h',qubit=1),
               inst('cx',control=2,target=0), inst('cx',control=0,target=1),
               inst('h',qubit=0), inst('h',qubit=2), inst('cx',control=1,target=2)]
        result = StabilizerSimulator().simulate(Circuit(3, ops))
        self.assertEqual(result.probabilities(), {0:.25, 3:.25, 5:.25, 6:.25})
        for terminal in (False, True):
            counts = StabilizerSimulator(seed=71, sample_terminal=terminal).simulate_shots(Circuit(3, ops+measures(3), 3), 4096)
            self.assertEqual(set(counts), {'000','011','101','110'})
            for count in counts.values(): self.assertAlmostEqual(count/4096, .25, delta=.04)

    def test_result_queries_and_ordering(self):
        result = StabilizerSimulator(profile=True).simulate(Circuit(4, [inst('x',qubit=0),inst('x',qubit=2)]))
        self.assertIsInstance(result, StabilizerResult)
        self.assertEqual(result.num_qubits,4)
        self.assertEqual(result.probabilities(), {5:1.})
        self.assertEqual(result.probabilities([0,1]),{2:1.})
        self.assertEqual(result.probabilities([1,0]),{1:1.})
        self.assertEqual(result.probabilities([]),{0:1.})
        self.assertEqual(result.counts(8,[0,1],71),{'10':8})
        self.assertEqual(result.counts(8,[],71),{'':8})
        self.assertEqual(result.probability('0101'),1.)
        self.assertEqual(result.probability('0001'),0.)
        self.assertEqual(result.expectation_value('ZIZI'),1)
        self.assertEqual(result.expectation_value('IIIZ'),-1)
        self.assertEqual(result.expectation_value('IIIX'),0)
        self.assertEqual(set(result.stabilizers),{'-IIIZ','+IIZI','-IZII','+ZIII'})
        self.assertGreater(result.profile['total_time'],0.)
        self.assertEqual(result.classical_bits,{})

    def test_large_ghz_and_cross_word_operations(self):
        n=130
        ops=ghz(n)+[inst('s',qubit=65),inst('swap',a=0,b=129),inst('cz',control=64,target=129)]
        result=StabilizerSimulator(max_memory_mb=1).simulate(Circuit(n,ops))
        self.assertEqual(result.probabilities(),{0:.5,(1<<n)-1:.5})
        self.assertEqual(result.probabilities([129,0,65]),{0:.5,7:.5})
        self.assertEqual(result.probability('0'*n),.5)
        self.assertEqual(result.probability('1'*n),.5)
        self.assertEqual(result.probability('1'+'0'*(n-1)),0.)
        self.assertEqual(result.expectation_value('Y'+'X'*(n-1)),-1)
        self.assertEqual(set(result.counts(100,seed=10)),{'0'*n,'1'*n})
        self.assertEqual(len(result.stabilizers),n)
        measured=StabilizerSimulator(seed=7,max_memory_mb=1).simulate(Circuit(n,ops+measures(n),n))
        outcome=''.join(str(measured.classical_bits[q]) for q in range(n-1,-1,-1))
        self.assertEqual(measured.counts(10,seed=7),{outcome:10})

    def test_reset_nested_feedback_and_worker_reuse(self):
        ops=ghz(3)+[inst('measure',qubit=0,cbit=0),
            conditional(conditional(inst('x',qubit=1),1),1),
            inst('reset',qubit=2)]+measures(3)
        circuit=Circuit(3,ops,3)
        a=StabilizerSimulator(seed=19,max_parallel_shots=1).simulate_shots(circuit,1024)
        b=StabilizerSimulator(seed=19,max_parallel_shots=4).simulate_shots(circuit,1024)
        self.assertEqual(a,b)
        self.assertEqual(set(a),{'000','001'})
        self.assertAlmostEqual(a['000']/1024,.5,delta=.06)
        for seed in range(10):
            result=StabilizerSimulator(seed=seed).simulate(circuit)
            self.assertEqual(result.classical_bits[1],0)
            self.assertEqual(result.classical_bits[2],0)

    def test_midcircuit_basis_changes_and_repeated_measurements(self):
        ops=[inst('h',qubit=0),inst('measure',qubit=0,cbit=0),
             inst('h',qubit=0),inst('measure',qubit=0,cbit=1),
             inst('measure',qubit=0,cbit=2)]
        counts=StabilizerSimulator(seed=29).simulate_shots(Circuit(1,ops,3),2048)
        self.assertEqual(set(counts),{'000','001','110','111'})
        for value in counts.values():self.assertAlmostEqual(value/2048,.25,delta=.04)

    def test_terminal_measurement_mapping_and_last_write(self):
        ops=ghz(2)+[inst('x',qubit=2),inst('measure',qubit=0,cbit=0),
                   inst('barrier'),inst('measure',qubit=1,cbit=2),
                   inst('measure',qubit=0,cbit=1),inst('measure',qubit=2,cbit=0)]
        for terminal in (False,True):
            sim=StabilizerSimulator(seed=38,sample_terminal=terminal)
            counts=sim.simulate_shots(Circuit(3,ops,5),1024)
            self.assertEqual(set(counts),{'00001','00111'})
            self.assertEqual(counts,sim.simulate_shots(Circuit(3,ops,5),1024))

    def test_invalid_circuit_rejected_even_without_shots(self):
        bad=[inst('h',qubit=20),inst('id',qubit=20),inst('cx',control=0,target=0),
             inst('measure',qubit=0,cbit=7),conditional(inst('x',qubit=10),1),
             conditional(inst('x',qubit=0),2),conditional(inst('x',qubit=0),0,size=65),
             inst('gate',name='remote_cx',params=[],qubits=[0]),
             inst('gate',name='remote_cu1',params=[],qubits=[0,1]),
             inst('classical',name='unknown')]
        sim=StabilizerSimulator()
        for op in bad:
            circuit=Circuit(2,[op],1)
            with self.subTest(op=op):
                with self.assertRaises(ValueError): sim.simulate(circuit)
                with self.assertRaises(ValueError): sim.simulate_shots(circuit,0)
                with self.assertRaises(ValueError): sim.supports(circuit)
        overlap=Circuit(2)
        overlap.data['qregs']['other']={'name':'other','base':1,'size':2}
        with self.assertRaises(ValueError):sim.simulate(overlap)
        overflow=Circuit(2)
        overflow.data['qregs']['q']['base']=(1<<64)-1
        with self.assertRaises(ValueError):sim.simulate(overflow)

    def test_eligibility_and_strict_angle_policy(self):
        sim=StabilizerSimulator()
        unsupported=[inst('t',qubit=0),inst('tdg',qubit=0),inst('rx',qubit=0,theta=1e-14),
            inst('rz',qubit=0,phi=math.pi/2+1e-13),inst('crx',control=0,target=1,theta=math.pi/2),
            inst('cp',control=0,target=1,lam=math.pi/2),inst('ch',control=0,target=1),
            inst('csx',control=0,target=1),inst('ccx',control1=0,control2=1,target=2),
            conditional(inst('t',qubit=0),1)]
        for op in unsupported:
            circuit=Circuit(3,[op],1)
            with self.subTest(op=op):
                self.assertFalse(sim.supports(circuit))
                with self.assertRaises(ValueError):sim.simulate_shots(circuit,0)
        for kind,field in [('rx','theta'),('ry','theta'),('rz','phi')]:
            for k in range(-8,9):self.assertTrue(sim.supports(Circuit(1,[inst(kind,qubit=0,**{field:k*math.pi/2})])))
        near=Circuit(1,[inst('rx',qubit=0,theta=math.pi+1e-10)])
        snapping=StabilizerSimulator(clifford_tolerance=1e-9)
        self.assertTrue(snapping.supports(near))
        self.assertEqual(snapping.simulate(near).probabilities(),{1:1.})
        for angle in [math.nan,math.inf,-math.inf]:
            with self.assertRaises(ValueError):sim.simulate(Circuit(1,[inst('rz',qubit=0,phi=angle)]))

    def test_bad_options_queries_and_empty_circuits(self):
        for kwargs in [dict(max_memory_mb=0),dict(max_parallel_shots=0),dict(clifford_tolerance=-1),
                       dict(clifford_tolerance=math.nan),dict(clifford_tolerance=math.pi/4)]:
            with self.assertRaises(ValueError):StabilizerSimulator(**kwargs)
        sim=StabilizerSimulator()
        result=sim.simulate(Circuit(0))
        self.assertEqual(result.probabilities(),{0:1.})
        self.assertEqual(result.counts(5),{'':5})
        self.assertEqual(result.probability(''),1.)
        self.assertEqual(result.expectation_value(''),1)
        self.assertEqual(result.stabilizers,[])
        self.assertEqual(sim.simulate_shots(Circuit(0),3),{'':3})
        self.assertEqual(sim.simulate_shots(Circuit(2,cbits=4),3),{'0000':3})
        self.assertEqual(sim.simulate_shots(Circuit(2),0),{})
        self.assertIsNone(result.profile)
        result=sim.simulate(Circuit(2))
        for qs in ([2],[0,0]):
            with self.assertRaises(ValueError):result.probabilities(qs)
            with self.assertRaises(ValueError):result.counts(0,qs)
        for p in ('X','XYQ'):
            with self.assertRaises(ValueError):result.expectation_value(p)
        with self.assertRaises(ValueError):result.probability('02')

    def test_memory_limits_sampling_fallback_and_bounded_workers(self):
        sim=StabilizerSimulator(max_memory_mb=1,max_parallel_shots=4)
        with self.assertRaises(MemoryError):sim.simulate(Circuit(2000))
        with self.assertRaises(MemoryError):sim.simulate(Circuit(1,cbits=100000))
        with self.assertRaises(MemoryError):sim.simulate(Circuit((1<<64)-1))
        # 1,300-qubit packed tableau fits 1 MiB; another tableau or full
        # terminal affine workspace does not. This must use one fresh worker.
        counts,report=profile_run(sim,Circuit(1300,[inst('x',qubit=1299)]+measures(1300),1300),2)
        self.assertEqual(counts,{'1'+'0'*1299:2})
        self.assertEqual(report['execution_strategy'],'trajectories')
        self.assertEqual(report['parallel_shots'],1)
        self.assertLessEqual(report['working_bytes'],1024**2)
        result=sim.simulate(Circuit(100,[inst('h',qubit=q) for q in range(100)]))
        with self.assertRaises(MemoryError):result.probabilities()
        self.assertEqual(len(result.counts(8,seed=1)),8)
        self.assertEqual(result.probabilities([0,99]),{0:.25,1:.25,2:.25,3:.25})

    def test_profiles_and_wrapper_options(self):
        circuit=Circuit(10,ghz(10)+measures(10),10)
        sim=StabilizerSimulator(seed=13,max_memory_mb=1)
        counts,report=profile_run(sim,circuit,1000)
        self.assertEqual(report['execution_strategy'],'terminal_affine')
        self.assertEqual(report['terminal_rank'],1)
        self.assertEqual(report['measure_calls'],10)
        self.assertEqual(report['gate_calls'],10)
        self.assertNotIn('shot_times',report)
        self.assertEqual(counts,simulate_monolithic_shots(circuit,mode='stabilizer',shots=1000,seed=13,max_memory_mb=1))
        result=simulate_monolithic(Circuit(2,ghz(2)),mode='stabilizer',max_memory_mb=1,profile=True)
        self.assertIsInstance(result,StabilizerResult)
        self.assertEqual(result.expectation_value('YY'),-1)
        forced=StabilizerSimulator(seed=13,max_parallel_shots=4,sample_terminal=False)
        _,report=profile_run(forced,circuit,100)
        self.assertEqual(report['execution_strategy'],'prefix_trajectories')
        self.assertGreater(report['parallel_shots'],0)
        self.assertEqual(report['measure_calls'],1000)
        with self.assertRaises(TypeError):simulate_monolithic(circuit,mode='stabilizer',unknown=True)


@unittest.skipIf(QuantumCircuit is None,'Qiskit reference dependency is not installed')
class StabilizerReferenceTests(unittest.TestCase):
    def assert_matches(self,result,reference):
        state=Statevector.from_instruction(reference)
        expected=state.probabilities()
        actual=result.probabilities()
        np.testing.assert_allclose([actual.get(i,0.) for i in range(len(expected))],expected,atol=3e-12)
        from qiskit.quantum_info import Pauli
        # All three-qubit Pauli expectations expose relative-phase errors that
        # computational-basis probabilities alone would miss.
        for p in itertools.product('IXYZ',repeat=reference.num_qubits):
            p=''.join(p)
            self.assertAlmostEqual(result.expectation_value(p),state.expectation_value(Pauli(p)).real,places=11)

    def test_fixed_clifford_gates_and_permuted_operands(self):
        cases=[('id',('qubit',),qg.IGate()),('u0',('qubit',),qg.IGate()),
               ('x',('qubit',),qg.XGate()),('y',('qubit',),qg.YGate()),('z',('qubit',),qg.ZGate()),
               ('h',('qubit',),qg.HGate()),('s',('qubit',),qg.SGate()),('sdg',('qubit',),qg.SdgGate()),
               ('sx',('qubit',),qg.SXGate()),('sxdg',('qubit',),qg.SXdgGate()),
               ('cx',('control','target'),qg.CXGate()),('cy',('control','target'),qg.CYGate()),
               ('cz',('control','target'),qg.CZGate()),('swap',('a','b'),qg.SwapGate())]
        for kind,names,gate in cases:
            for qubits in ([0,2][:len(names)],[2,0][:len(names)]):
                qc=QuantumCircuit(3);qc.h(0);qc.h(1);qc.s(1);qc.cx(0,2)
                ops=[inst('h',qubit=0),inst('h',qubit=1),inst('s',qubit=1),inst('cx',control=0,target=2)]
                qc.append(gate,qubits);ops.append(inst(kind,**dict(zip(names,qubits))))
                with self.subTest(kind=kind,qubits=qubits):self.assert_matches(StabilizerSimulator().simulate(Circuit(3,ops)),qc)

    def test_parameterized_cliffords_against_qiskit(self):
        for k in range(-4,9):
            a=k*math.pi/2
            cases=[('rx',{'qubit':2,'theta':a},qg.RXGate(a),[2]),
                   ('ry',{'qubit':0,'theta':a},qg.RYGate(a),[0]),
                   ('rz',{'qubit':1,'phi':a},qg.RZGate(a),[1]),
                   ('p',{'qubit':0,'lam':a},qg.PhaseGate(a),[0]),
                   ('u1',{'qubit':0,'lam':a},qg.U1Gate(a),[0]),
                   ('rxx',{'a':2,'b':0,'theta':a},qg.RXXGate(a),[2,0]),
                   ('rzz',{'a':0,'b':2,'theta':a},qg.RZZGate(a),[0,2])]
            if k%2==0:
                cases += [(name,{'control':2,'target':0,param:a},gate(a),[2,0]) for name,param,gate in
                          [('crx','theta',qg.CRXGate),('cry','theta',qg.CRYGate),('crz','lam',qg.CRZGate),
                           ('cp','lam',qg.CPhaseGate),('cu1','lam',qg.CU1Gate)]]
            for kind,fields,gate,qs in cases:
                qc=QuantumCircuit(3);qc.h(0);qc.s(0);qc.h(2);qc.cx(2,1);qc.append(gate,qs)
                ops=[inst('h',qubit=0),inst('s',qubit=0),inst('h',qubit=2),inst('cx',control=2,target=1),inst(kind,**fields)]
                with self.subTest(kind=kind,k=k):self.assert_matches(StabilizerSimulator().simulate(Circuit(3,ops)),qc)
        for kind in ('u','u3','u2','cu','cu3'):
            controlled=kind.startswith('c')
            params={'phi':math.pi/2,'lam':-math.pi/2 if controlled else math.pi}
            if kind!='u2':params['theta']=math.pi if controlled else math.pi/2
            if kind=='cu':params['gamma']=math.pi/2
            operands={'control':2,'target':0} if controlled else {'qubit':1}
            qc=QuantumCircuit(3);qc.h(0);qc.h(2)
            gate={'u':qg.UGate,'u3':qg.U3Gate,'u2':qg.U2Gate,'cu':qg.CUGate,'cu3':qg.CU3Gate}[kind]
            keys=['phi','lam'] if kind=='u2' else ['theta','phi','lam']+(['gamma'] if kind=='cu' else [])
            qc.append(gate(*(params[key] for key in keys)),[2,0] if controlled else [1])
            ops=[inst('h',qubit=0),inst('h',qubit=2),inst(kind,**operands,**params)]
            self.assert_matches(StabilizerSimulator().simulate(Circuit(3,ops)),qc)

    def test_random_clifford_circuits_and_marginals(self):
        rng=random.Random(781)
        for trial in range(24):
            n=5;qc=QuantumCircuit(n);ops=[]
            for _ in range(60):
                name=rng.choice(['h','s','sdg','x','y','z','sx','sxdg','cx','cy','cz','swap'])
                if name in ('cx','cy','cz','swap'):
                    a,b=rng.sample(range(n),2);getattr(qc,name)(a,b)
                    ops.append(inst(name,**({'a':a,'b':b} if name=='swap' else {'control':a,'target':b})))
                else:
                    q=rng.randrange(n);getattr(qc,name)(q);ops.append(inst(name,qubit=q))
            state=Statevector(qc);result=StabilizerSimulator().simulate(Circuit(n,ops))
            actual=result.probabilities()
            np.testing.assert_allclose([actual.get(i,0) for i in range(1<<n)],state.probabilities(),atol=3e-12)
            expected=np.zeros(8)
            for i,p in enumerate(state.probabilities()):expected[((i>>1)&1)*4+((i>>4)&1)*2+(i&1)]+=p
            marginal=result.probabilities([1,4,0])
            np.testing.assert_allclose([marginal.get(i,0) for i in range(8)],expected,atol=3e-12)
            from qiskit.quantum_info import Pauli
            for _ in range(20):
                pauli=''.join(rng.choice('IXYZ') for _ in range(n))
                self.assertAlmostEqual(result.expectation_value(pauli),state.expectation_value(Pauli(pauli)).real,places=11)
            for terminal in (True,False):
                counts=StabilizerSimulator(seed=trial,sample_terminal=terminal).simulate_shots(Circuit(n,ops+measures(n),n),128)
                self.assertTrue(all(actual.get(int(key,2),0)>0 for key in counts))

    def test_dynamic_circuits_against_exact_branch_enumeration(self):
        # Enumerate measurement branches with dense Qiskit states, independently
        # of the tableau algorithm and its RNG, including feedback after reset.
        rng=random.Random(843)
        def branch(op,state,weight,cbits):
            kind=op['kind']
            if kind=='conditional':
                c=op['condition']
                value=sum(cbits.get(c['creg_base']+i,0)<<i for i in range(c['creg_size']))
                return branch(op['op'],state,weight,cbits) if value==c['creg_value'] else [(state,weight,cbits)]
            if kind in ('measure','reset'):
                q=op['qubit'];out=[]
                for bit in (0,1):
                    amplitudes=state.data.copy()
                    for i in range(len(amplitudes)):
                        if (i>>q)&1!=bit:amplitudes[i]=0
                    probability=float(np.vdot(amplitudes,amplitudes).real)
                    if probability<1e-12:continue
                    collapsed=Statevector(amplitudes/math.sqrt(probability));updated=dict(cbits)
                    if kind=='measure':updated[op['cbit']]=bit
                    elif bit:collapsed=collapsed.evolve(qg.XGate(),[q])
                    out.append((collapsed,weight*probability,updated))
                return out
            qs=[op['control'],op['target']] if kind=='cx' else [op['qubit']]
            gate={'h':qg.HGate,'s':qg.SGate,'x':qg.XGate,'cx':qg.CXGate}[kind]()
            return [(state.evolve(gate,qs),weight,cbits)]
        for trial in range(10):
            ops=[]
            for segment in range(3):
                for _ in range(10):
                    kind=rng.choice(['h','s','cx'])
                    if kind=='cx':
                        a,b=rng.sample(range(3),2);ops.append(inst(kind,control=a,target=b))
                    else:ops.append(inst(kind,qubit=rng.randrange(3)))
                ops += [inst('measure',qubit=segment,cbit=segment),
                        conditional(inst('s',qubit=(segment+1)%3),1,base=segment)]
            ops += [inst('reset',qubit=1)]+measures(3)
            branches=[(Statevector.from_label('000'),1.,{})]
            for op in ops:
                branches=[child for state,w,c in branches for child in branch(op,state,w,c)]
            expected={}
            for _,weight,cbits in branches:
                key=''.join(str(cbits.get(q,0)) for q in (2,1,0))
                expected[key]=expected.get(key,0.)+weight
            counts=StabilizerSimulator(seed=trial,max_parallel_shots=4).simulate_shots(Circuit(3,ops,3),4096)
            self.assertTrue(set(counts)<=set(expected))
            for key,probability in expected.items():
                self.assertAlmostEqual(counts.get(key,0)/4096,probability,delta=.04)

    def test_remote_gates_agree_with_shared_statevector(self):
        from dqsim import StatevectorSimulator
        from qiskit.quantum_info import Pauli
        for name,params in [('remote_cx',[]),('remote_cz',[]),('nonlocal_cz',[]),
                            ('remote_epr',[]),('epr',[]),('remote_link_phi_plus',[]),
                            ('remote_link_psi_plus',[]),('remote_link_psi_minus',[]),
                            ('remote_cu1',[math.pi]),('remote_rzz',[math.pi/2])]:
            ops=[inst('h',qubit=0),inst('s',qubit=0),inst('h',qubit=1),inst('gate',name=name,params=params,qubits=[1,0])]
            c=Circuit(2,ops);result=StabilizerSimulator().simulate(c)
            state=Statevector(StatevectorSimulator().simulate(c).statevector)
            for p in itertools.product('IXYZ',repeat=2):
                p=''.join(p);self.assertAlmostEqual(result.expectation_value(p),state.expectation_value(Pauli(p)).real,places=12)
