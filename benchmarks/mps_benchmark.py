"""Focused MPS scaling and shot-strategy checks, not selector training data."""
import argparse
import json
import os
import platform
import statistics
import time

from statevector_benchmark import Circuit, layers, measure_all
from dqsim import MpsSimulator


def ghz(n):
    return [{'kind': 'h', 'qubit': 0}] + [
        {'kind': 'cx', 'control': q, 'target': q+1} for q in range(n-1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    sim = MpsSimulator(seed=71, max_memory_mb=64)
    forced = MpsSimulator(seed=71, max_memory_mb=64, sample_terminal=False)
    terminal = Circuit(20, ghz(20)+measure_all(20), 20)
    large = Circuit(100, ghz(100)+measure_all(100), 100)
    unitary = Circuit(10, layers(10, 4))
    dynamic = Circuit(20, ghz(20)+[
        {'kind': 'measure', 'qubit': 0, 'cbit': 0},
        {'kind': 'reset', 'qubit': 0},
    ]+measure_all(20), 20)
    cases = {
        'terminal_20q_1000shots': (lambda: sim.simulate_shots(terminal, 1000), 1000),
        'forced_trajectories_20q_1000shots': (lambda: forced.simulate_shots(terminal, 1000), 1000),
        'ghz_100q_1000shots': (lambda: sim.simulate_shots(large, 1000), 1000),
        'dynamic_20q_100shots': (lambda: sim.simulate_shots(dynamic, 100), 100),
        'unitary_10q_depth4': (lambda: sim.simulate(unitary), None),
    }
    report = {'platform': platform.platform(), 'python': platform.python_version(),
              'rayon_threads': os.environ.get('RAYON_NUM_THREADS', 'default'),
              'repeats': args.repeats, 'max_memory_mb': 64, 'seconds': {}}
    for name, (call, shots) in cases.items():
        call()
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = call()
            times.append(time.perf_counter()-start)
            if shots is not None:
                assert sum(result.values()) == shots
            else:
                assert abs(result.diagnostics['norm_squared']-1) < 1e-10
        report['seconds'][name] = {'median': statistics.median(times), 'samples': times}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
