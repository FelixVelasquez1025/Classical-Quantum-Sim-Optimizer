"""Focused P-block scaling checks; no selector training data is collected."""
import argparse
import json
import os
import platform
import statistics
import time
from types import SimpleNamespace

from dqsim import PBlockSimulator
from statevector_benchmark import Circuit, measure_all


def pairs(n):
    return [op for q in range(0,n,2) for op in (
        {'kind':'h','qubit':q}, {'kind':'cx','control':q,'target':q+1})]


def distributed(n, ops, singleton_nodes):
    groups={q:[q] for q in range(n)} if singleton_nodes else {0:list(range(n))}
    buckets={node:[] for node in groups}
    for op in ops:
        q=op.get('qubit',op.get('control',0))
        buckets[q if singleton_nodes else 0].append(op)
    circuits={node:Circuit(n,bucket,n) for node,bucket in buckets.items()}
    for c in circuits.values(): c.instructions=c.data['instructions']
    return SimpleNamespace(circuits=circuits,qubits_per_node=groups,
                           _instruction_index={id(op):i for i,op in enumerate(ops)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    if args.repeats<1: parser.error('--repeats must be positive')
    sim=PBlockSimulator(seed=71,max_memory_mb=64)
    forced=PBlockSimulator(seed=71,max_memory_mb=64,sample_terminal=False)
    product=[{'kind':'h','qubit':q} for q in range(16)]+measure_all(16)
    terminal=Circuit(20,pairs(20)+measure_all(20),20)
    large=Circuit(100,pairs(100)+measure_all(100),100)
    dynamic=Circuit(20,pairs(20)+[{'kind':'measure','qubit':0,'cbit':0},
        {'kind':'conditional','condition':{'creg_base':0,'creg_size':1,'creg_value':1},
         'op':{'kind':'x','qubit':1}},{'kind':'reset','qubit':2}]+measure_all(20),20)
    one_node=distributed(16,product,False)
    many_nodes=distributed(16,product,True)
    cases={
        'product_16q_one_node_16shots':(lambda:sim.simulate_shots(one_node,16),16),
        'product_16q_sixteen_nodes_16shots':(lambda:sim.simulate_shots(many_nodes,16),16),
        'bell_pairs_20q_terminal_1000shots':(lambda:sim.simulate_shots(terminal,1000),1000),
        'bell_pairs_20q_forced_1000shots':(lambda:forced.simulate_shots(terminal,1000),1000),
        'bell_pairs_100q_1000shots':(lambda:sim.simulate_shots(large,1000),1000),
        'dynamic_20q_100shots':(lambda:sim.simulate_shots(dynamic,100),100),
    }
    report={'platform':platform.platform(),'python':platform.python_version(),
            'rayon_threads':os.environ.get('RAYON_NUM_THREADS','default'),
            'repeats':args.repeats,'max_memory_mb':64,'seconds':{}}
    for name,(call,shots) in cases.items():
        call();times=[]
        for _ in range(args.repeats):
            start=time.perf_counter();result=call();times.append(time.perf_counter()-start)
            assert sum(result.values())==shots
        report['seconds'][name]={'median':statistics.median(times),'samples':times}
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
