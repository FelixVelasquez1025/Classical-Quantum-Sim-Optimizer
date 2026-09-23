"""Small repeatable performance checks, not ML training-data collection.

Run against the installed package, or use --extension to load a separately built
native module in a fresh process for a before/after comparison.
"""

import argparse
import importlib.machinery
import importlib.util
import json
import os
import platform
import statistics
import time


class Circuit:
    def __init__(self, n, instructions, cbits=0):
        self.data = {
            "qregs": {"q": {"name": "q", "base": 0, "size": n}},
            "cregs": {"c": {"name": "c", "base": 0, "size": cbits}},
            "instructions": instructions,
        }

    def model_dump_json(self):
        return json.dumps(self.data)


def layers(n, depth):
    instructions = []
    for layer in range(depth):
        for q in range(n):
            instructions.append({"kind": "ry", "qubit": q, "theta": .23 + .017 * (q + layer)})
            instructions.append({"kind": "rz", "qubit": q, "phi": -.41 + .019 * layer})
        for q in range(n - 1):
            instructions.append({"kind": "cx", "control": q, "target": q + 1})
    return instructions


def measure_all(n):
    return [{"kind": "measure", "qubit": q, "cbit": q} for q in range(n)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", help="Path to an isolated native .so/.dylib")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    os.environ.setdefault("RAYON_NUM_THREADS", "4")
    if args.extension:
        loader = importlib.machinery.ExtensionFileLoader("_core", args.extension)
        spec = importlib.util.spec_from_loader("_core", loader)
        core = importlib.util.module_from_spec(spec)
        loader.exec_module(core)
    else:
        from dqsim import _core as core

    sim = core.StatevectorSimulator(seed=71)
    terminal = Circuit(12, layers(12, 8) + measure_all(12), 12)
    dynamic_ops = layers(10, 3) + [
        {"kind": "measure", "qubit": 0, "cbit": 0},
        {"kind": "conditional", "condition": {"creg_base": 0, "creg_size": 1, "creg_value": 1},
         "op": {"kind": "x", "qubit": 9}},
    ] + layers(10, 3) + measure_all(10)
    dynamic = Circuit(10, dynamic_ops, 10)
    unitary = Circuit(16, layers(16, 8))
    uniform = sim.simulate(Circuit(14, [{"kind": "h", "qubit": q} for q in range(14)]))
    cases = {
        "terminal_12q_256shots": lambda: sim.simulate_shots(terminal, shots=256),
        "dynamic_10q_128shots": lambda: sim.simulate_shots(dynamic, shots=128),
        "unitary_16q": lambda: sim.simulate(unitary),
        "full_probabilities_14q": uniform.probabilities,
    }
    result = {
        "platform": platform.platform(), "python": platform.python_version(),
        "rayon_threads": int(os.environ["RAYON_NUM_THREADS"]), "repeats": args.repeats,
        "seconds": {},
    }
    for name, call in cases.items():
        call()  # Warm up imports, allocations and the thread pool.
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            output = call()
            times.append(time.perf_counter() - start)
            if "shots" in name:
                assert sum(output.values()) == (256 if "256shots" in name else 128)
            elif name == "full_probabilities_14q":
                assert abs(sum(output.values()) - 1) < 1e-10
            del output
        result["seconds"][name] = {"median": statistics.median(times), "samples": times}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
