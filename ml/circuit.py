"""Public static circuit features, artifact validation and resource preflight.

No benchmark collection or circuit evolution runs in this module. Definitions
retain feature schema 1; structural enrichment is in ml.structural_features.
"""
import hashlib
import json
import statistics
import struct
import sys

FEATURE_VERSION = 1


def collector_module():
    """Compatibility accessor for existing feature/selector callers."""
    return sys.modules[__name__]


def gate_traits_classifier():
    """Classify cached one/two-qubit gates; never evolve a circuit state."""
    from functools import lru_cache
    import numpy as np
    import dqsim
    from dqsim.qasm import _SINGLE, _CONTROLLED, _PAIR
    from qiskit.circuit.library import standard_gates
    names = dict(id='IGate', x='XGate', y='YGate', z='ZGate', h='HGate',
                 s='SGate', sdg='SdgGate', t='TGate', tdg='TdgGate', sx='SXGate',
                 sxdg='SXdgGate', u='UGate', u1='U1Gate', u2='U2Gate', u3='U3Gate',
                 p='PhaseGate', rx='RXGate', ry='RYGate', rz='RZGate',
                 cx='CXGate', cy='CYGate', cz='CZGate', ch='CHGate', csx='CSXGate',
                 crx='CRXGate', cry='CRYGate', crz='CRZGate', cu1='CU1Gate',
                 cp='CPhaseGate', cu3='CU3Gate', cu='CUGate', swap='SwapGate',
                 rxx='RXXGate', rzz='RZZGate')
    parameters = {**_SINGLE, **_CONTROLLED, **_PAIR}
    simulator = dqsim.StabilizerSimulator(clifford_tolerance=0.0)

    @lru_cache(maxsize=16384)
    def classify(kind, values):
        fields = ('qubit',) if kind in _SINGLE else (('control', 'target') if kind in _CONTROLLED else ('a', 'b'))
        op = dict(kind=kind, **dict(zip(fields, range(len(fields)))),
                  **dict(zip(parameters[kind], values)))
        data = dict(qregs={'q': dict(name='q', base=0, size=len(fields))},
                    cregs={}, instructions=[op])
        clifford = simulator.supports(dqsim.ImportedCircuit(data, {}))
        matrix = getattr(standard_gates, names[kind])(*values).to_matrix()
        off_diagonal = matrix - np.diag(np.diag(matrix))
        diagonal = bool(np.max(np.abs(off_diagonal)) <= 1e-14)
        return clifford, diagonal

    def traits(op):
        kind = op['kind']
        return classify(kind, tuple(op[field] for field in parameters[kind]))
    return traits


def extract_features(data, traits):
    """Static features of normalized IR, counting conditional bodies once."""
    n = sum(reg['size'] for reg in data['qregs'].values())
    nc = sum(reg['size'] for reg in data.get('cregs', {}).values())
    qdepth, cdepth = [0] * n, [0] * nc
    parent, sizes = list(range(n)), [1] * n
    pairs, distances = set(), []
    cuts = [0] * (n + 1)
    counts = {f'{arity}_qubit_{group}_count': 0
              for arity in ('single', 'two') for group in ('clifford', 'non_clifford')}
    measurements = resets = conditionals = diagonal = gates = 0
    seen_measurement = nonterminal = False

    def find(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    for instruction in data['instructions']:
        op, controls = instruction, set()
        conditional = False
        while op['kind'] == 'conditional':
            conditional = True
            condition = op['condition']
            controls.update(range(condition['creg_base'], condition['creg_base'] + condition['creg_size']))
            op = op['op']
        conditionals += int(conditional)
        kind = op['kind']
        if kind == 'barrier':
            continue  # The normalized transport treats barriers as no-ops.
        if 'qubit' in op:
            qubits = [op['qubit']]
        elif 'control' in op:
            qubits = [op['control'], op['target']]
        else:
            qubits = [op['a'], op['b']]
        bits = controls | ({op['cbit']} if kind == 'measure' else set())
        level = 1 + max([qdepth[q] for q in qubits] + [cdepth[c] for c in bits], default=0)
        for q in qubits:
            qdepth[q] = level
        # Track reads as well as writes, preserving classical anti-dependencies.
        for c in bits:
            cdepth[c] = level
        if conditional or kind == 'reset' or (seen_measurement and kind != 'measure'):
            nonterminal = True
        if kind == 'measure':
            measurements += 1
            seen_measurement = True
            continue
        if kind == 'reset':
            resets += 1
            continue
        gates += 1
        clifford, is_diagonal = traits(op)
        diagonal += int(is_diagonal)
        arity = 'single' if len(qubits) == 1 else 'two'
        counts[f'{arity}_qubit_{"clifford" if clifford else "non_clifford"}_count'] += 1
        if len(qubits) == 2:
            a, b = sorted(qubits)
            pairs.add((a, b))
            distances.append(b - a)
            cuts[a] += 1
            cuts[b] -= 1
            x, y = find(a), find(b)
            if x != y:
                if sizes[x] < sizes[y]:
                    x, y = y, x
                parent[y] = x
                sizes[x] += sizes[y]
    crossing = max_crossing = 0
    for delta in cuts:
        crossing += delta
        max_crossing = max(max_crossing, crossing)
    mid_circuit = any_measurement_followed_by_work(data)
    features = dict(feature_version=FEATURE_VERSION, num_qubits=n, num_cbits=nc,
                    circuit_depth=max(qdepth + cdepth, default=0), total_gate_count=gates,
                    largest_interaction_component=max(sizes, default=0),
                    interaction_component_count=sum(find(q) == q for q in range(n)),
                    distinct_interacting_pairs=len(pairs),
                    mean_two_qubit_distance=statistics.mean(distances) if distances else 0.0,
                    max_two_qubit_distance=max(distances, default=0),
                    max_cut_crossings=max_crossing, measurement_count=measurements,
                    reset_count=resets, conditional_count=conditionals,
                    has_measurements=bool(measurements),
                    terminal_only_measurements=bool(measurements) and not mid_circuit,
                    terminal_sampling_candidate=bool(measurements) and not nonterminal,
                    has_mid_circuit_measurements=mid_circuit,
                    diagonal_gate_count=diagonal, diagonal_gate_fraction=diagonal/gates if gates else 0.0,
                    stabilizer_eligible=not any(counts[f'{a}_qubit_non_clifford_count'] for a in ('single', 'two')),
                    **counts)
    for key, value in counts.items():
        features[key.replace('_count', '_fraction')] = value/gates if gates else 0.0
    return features


def any_measurement_followed_by_work(data):
    seen = False
    for op in data['instructions']:
        while op['kind'] == 'conditional':
            op = op['op']
        if op['kind'] == 'measure':
            seen = True
        elif seen and op['kind'] != 'barrier':
            return True
    return False


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_artifact(root, entry):
    path = (root / entry['artifact']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Artifact path escapes the imports directory')
    artifact = json.loads(path.read_text())
    data = artifact['circuit']
    payload = json.dumps(data, separators=(',', ':'), allow_nan=False)
    if digest(payload.encode()) != entry['normalized_sha256']:
        raise ValueError('Normalized artifact hash does not match the manifest')
    if artifact['metadata']['source_sha256'] != entry['source_sha256']:
        raise ValueError('Source hash does not match the manifest')
    return data, artifact['metadata']


def has_measurements(data):
    for op in data['instructions']:
        while op['kind'] == 'conditional':
            op = op['op']
        if op['kind'] == 'measure':
            return True
    return False


def task_for(data, requested):
    measured = has_measurements(data)
    if requested == 'auto':
        return 'shots' if measured else 'evolve'
    if requested == 'shots' and not measured:
        raise ValueError('Shot task requires measurements; none are added automatically')
    return requested


def resource_preflight(backend, num_qubits, settings):
    """Reject only provably oversized dense states, never compact backends."""
    if backend != 'statevector':
        return None
    budget = settings['max_memory_mb'] * 1024**2
    # complex128 requires 16 bytes per amplitude. Workspaces cost extra;
    # fitting this lower bound does not guarantee that execution will fit.
    if num_qubits + 4 >= budget.bit_length():
        return dict(status='resource_exceeded', stage='preflight',
                    reason='A single complex128 statevector exceeds max_memory_mb',
                    minimum_state_bytes_power_of_two=num_qubits + 4,
                    memory_budget_bytes=budget)
    # Native dense indices use usize even if an enormous budget was requested.
    if num_qubits >= 8 * struct.calcsize('P'):
        return dict(status='resource_exceeded', stage='preflight',
                    reason='Qubit count exceeds native statevector index capacity')
    return None
