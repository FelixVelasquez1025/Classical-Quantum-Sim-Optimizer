"""Focused stabilizer checks. Timings are not ML training labels."""
import argparse
import importlib.machinery
import importlib.util
import json
import os
import platform
import statistics
import time

from statevector_benchmark import Circuit, measure_all


def ghz(n):
    return [{'kind': 'h', 'qubit': 0}] + [
        {'kind': 'cx', 'control': q, 'target': q + 1} for q in range(n - 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extension', help='Native module from before the upgrade')
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if args.extension:
        loader = importlib.machinery.ExtensionFileLoader('_core', args.extension)
        spec = importlib.util.spec_from_loader('_core', loader)
        core = importlib.util.module_from_spec(spec)
        loader.exec_module(core)
    else:
        from dqsim import _core as core
    # Both versions use their defaults and the same seed, inputs and Rayon pool.
    # Old: one fresh tableau per shot. New: terminal affine sampling by default.
    sim = core.StabilizerSimulator(seed=71)
    cases = {
        'ghz_100q_1000shots': (Circuit(100, ghz(100) + measure_all(100), 100), 1000),
        'ghz_500q_100shots': (Circuit(500, ghz(500) + measure_all(500), 500), 100),
        'ghz_100q_1shot': (Circuit(100, ghz(100) + measure_all(100), 100), 1),
    }
    report = {
        'platform': platform.platform(), 'python': platform.python_version(),
        'rayon_threads': os.environ.get('RAYON_NUM_THREADS', 'default'),
        'repeats': args.repeats, 'seconds': {},
    }
    for name, (circuit, shots) in cases.items():
        sim.simulate_shots(circuit, shots)
        times = []
        n = circuit.data['qregs']['q']['size']
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = sim.simulate_shots(circuit, shots)
            times.append(time.perf_counter() - start)
            assert sum(result.values()) == shots
            assert set(result) <= {'0' * n, '1' * n}
        report['seconds'][name] = {'median': statistics.median(times), 'samples': times}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
