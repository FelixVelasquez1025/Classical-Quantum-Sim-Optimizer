"""Version 2 static features: basis facts and ordered potential interactions.

Only one-/two-qubit gate matrices (at most 4x4) are inspected. No amplitudes,
trajectories, bond dimensions, timings or backend outcomes enter this analysis.
Unknown means unknown: inverse gates do not recover facts lost to superposition.
Exact matrix zeros are used, without snapping small rotations to Clifford gates.
The interaction graph is a conservative historical proxy, not entanglement.
"""
from functools import lru_cache
import math

from ml.data import FEATURE_NAMES, validate_features

FEATURE_VERSION = 2
BASIS_NAMES = (
    'controlled_gate_count', 'inactive_control_count', 'inactive_control_fraction',
    'active_control_count', 'active_control_fraction',
    'unknown_control_count', 'unknown_control_fraction',
    'diagonal_reduced_count', 'diagonal_reduced_fraction',
    'unknown_creation_count', 'unknown_qubit_mean_fraction',
    'unknown_qubit_max_fraction', 'first_unknown_gate_fraction',
    'conditional_skipped_count', 'conditional_uncertain_count',
)
INTERACTION_NAMES = (
    'potential_coupling_count', 'potential_coupling_fraction',
    'potential_long_range_fraction', 'potential_mean_distance',
    'potential_max_cut_crossings', 'potential_mean_gate_position_fraction',
    'prefix_largest_component_mean_fraction',
    'prefix_largest_component_25_fraction', 'prefix_largest_component_50_fraction',
    'prefix_largest_component_75_fraction', 'potential_largest_component_fraction',
    'potential_component_count', 'potential_merge_count', 'swap_count',
)
STRUCTURAL_NAMES = BASIS_NAMES + INTERACTION_NAMES
FEATURE_SETS = dict(baseline=(), combined=STRUCTURAL_NAMES)


def base_features(features):
    if features.get('feature_version') == 1:
        return validate_features(features)
    if type(features.get('feature_version')) is not int or features['feature_version'] != FEATURE_VERSION:
        raise ValueError('Unsupported structural feature version')
    expected = set(FEATURE_NAMES + STRUCTURAL_NAMES) | {'feature_version'}
    if set(features) != expected:
        raise ValueError('Structural feature schema mismatch')
    for name in STRUCTURAL_NAMES:
        value = features[name]
        floating = name.endswith('_fraction') or name == 'potential_mean_distance'
        if (type(value) not in ((int, float) if floating else (int,))
                or not math.isfinite(value) or value < 0
                or (name.endswith('_fraction') and value > 1)):
            raise ValueError(f'Invalid structural feature: {name}')
    return validate_features(dict(feature_version=1, **{k: features[k] for k in FEATURE_NAMES}))


def input_names(feature_set='baseline'):
    if feature_set not in FEATURE_SETS:
        raise ValueError(f'Unknown feature set: {feature_set}')
    return (*FEATURE_NAMES, 'task_is_shots', 'log1p_shots', *FEATURE_SETS[feature_set])


def feature_vector(features, task, settings, feature_set='baseline'):
    from ml.selector import vector
    input_names(feature_set)  # Reject unknown names, including on empty folds.
    base = base_features(features)
    if feature_set != 'baseline' and features['feature_version'] != FEATURE_VERSION:
        raise ValueError('Structural classifier requires version 2 circuit features')
    return vector(base, task, settings) + [float(features[k]) for k in FEATURE_SETS[feature_set]]


@lru_cache(maxsize=16384)
def _matrix(kind, values):
    from qiskit.circuit.library import standard_gates as gates
    names = dict(id='IGate', x='XGate', y='YGate', z='ZGate', h='HGate',
                 s='SGate', sdg='SdgGate', t='TGate', tdg='TdgGate', sx='SXGate',
                 sxdg='SXdgGate', u='UGate', u1='U1Gate', u2='U2Gate', u3='U3Gate',
                 p='PhaseGate', rx='RXGate', ry='RYGate', rz='RZGate',
                 cx='CXGate', cy='CYGate', cz='CZGate', ch='CHGate', csx='CSXGate',
                 crx='CRXGate', cry='CRYGate', crz='CRZGate', cu1='CU1Gate',
                 cp='CPhaseGate', cu3='CU3Gate', cu='CUGate', swap='SwapGate',
                 rxx='RXXGate', rzz='RZZGate')
    matrix = getattr(gates, names[kind])(*values).to_matrix()
    matrix.flags.writeable = False
    return matrix


def gate_spec(op):
    from dqsim.qasm import _SINGLE, _CONTROLLED, _PAIR
    kind = op['kind']
    if kind in _SINGLE:
        wires, fields = (op['qubit'],), _SINGLE[kind]
    elif kind in _CONTROLLED:
        wires, fields = (op['control'], op['target']), _CONTROLLED[kind]
    elif kind in _PAIR:
        wires, fields = (op['a'], op['b']), _PAIR[kind]
    else:
        raise ValueError(f'Unsupported normalized gate: {kind}')
    return wires, tuple(op[field] for field in fields)


@lru_cache(maxsize=32768)
def _gate_facts(kind, values, before):
    """Sound bit support transfer, including unknown entangled inputs.

    Qiskit puts the first operand in the least significant bit. Every column
    compatible with known input bits is considered; interference can only remove
    support, so it cannot invalidate a bit reported as definitely 0 or 1 here.
    """
    matrix = _matrix(kind, values)
    width = len(before)
    inputs = [i for i in range(1 << width)
              if all(value is None or ((i >> bit) & 1) == value
                     for bit, value in enumerate(before))]
    outputs = {row for col in inputs for row in range(1 << width) if matrix[row, col] != 0}
    after = []
    for bit in range(width):
        possibilities = {(i >> bit) & 1 for i in outputs}
        after.append(next(iter(possibilities)) if len(possibilities) == 1 else None)
    diagonal = all(matrix[i, j] == 0 for i in range(1 << width)
                   for j in range(1 << width) if i != j)
    scalar_on_input = (diagonal and all(matrix[i, i] == matrix[inputs[0], inputs[0]] for i in inputs))
    return tuple(after), diagonal, scalar_on_input


class BasisTracker:
    """Three-valued quantum/classical facts, initialized to the all-zero state."""

    def __init__(self, n, nc):
        self.quantum = [0] * n
        self.classical = [0] * nc
        self.unknown_count = 0

    def unwrap(self, instruction):
        op, execution = instruction, True
        while op['kind'] == 'conditional':
            c = op['condition']
            actual = self.classical[c['creg_base']:c['creg_base'] + c['creg_size']]
            matches = [value is None or value == ((c['creg_value'] >> i) & 1)
                       for i, value in enumerate(actual)]
            if not all(matches):
                execution = False
            elif any(value is None for value in actual) and execution is not False:
                execution = None
            op = op['op']
        return op, execution

    def apply(self, instruction):
        op, execution = self.unwrap(instruction)
        kind = op['kind']
        if kind == 'barrier' or execution is False:
            return op, execution
        if kind in ('measure', 'reset'):
            wires = (op['qubit'],)
            after = (0,) if kind == 'reset' else (self.quantum[op['qubit']],)
            if kind == 'measure':
                bit = op['cbit']
                value = after[0]
                if execution is None and self.classical[bit] != value:
                    value = None
                self.classical[bit] = value
        else:
            wires, values = gate_spec(op)
            after, _, _ = _gate_facts(kind, values, tuple(self.quantum[q] for q in wires))
        for q, value in zip(wires, after):
            if execution is None and self.quantum[q] != value:
                value = None  # Join the executed and unexecuted branch.
            self.unknown_count += int(value is None) - int(self.quantum[q] is None)
            self.quantum[q] = value
        return op, execution


def extract_structure(data):
    n = sum(reg['size'] for reg in data['qregs'].values())
    nc = sum(reg['size'] for reg in data.get('cregs', {}).values())
    tracker = BasisTracker(n, nc)
    f = {name: 0 for name in STRUCTURAL_NAMES}
    instructions = data['instructions']

    def gate_body(op):
        while op['kind'] == 'conditional':
            op = op['op']
        return op['kind'] not in ('measure', 'reset', 'barrier')

    total = sum(gate_body(op) for op in instructions)
    parent, size, nodes = list(range(n)), [1] * n, list(range(n))
    largest, components = int(n > 0), n
    cuts = [0] * (n + 1)
    position = unknown_sum = largest_sum = diagonal_pairs = pairs = distance_sum = 0
    long_range = coupling_positions = 0
    milestones = {p: math.ceil(total * p / 100) for p in (25, 50, 75)}
    f['first_unknown_gate_fraction'] = 1.0 if total else 0.0

    def find(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    for instruction in instructions:
        op, execution = tracker.unwrap(instruction)
        kind = op['kind']
        if kind in ('measure', 'reset', 'barrier'):
            tracker.apply(instruction)
            continue
        position += 1
        wires, values = gate_spec(op)
        before = tuple(tracker.quantum[q] for q in wires)
        _, diagonal, scalar = _gate_facts(kind, values, before)
        pairs += int(len(wires) == 2)
        diagonal_pairs += int(len(wires) == 2 and diagonal)
        potential = False
        if execution is False:
            f['conditional_skipped_count'] += 1
        else:
            f['conditional_uncertain_count'] += int(execution is None)
            if 'control' in op:
                f['controlled_gate_count'] += 1
                category = {0: 'inactive', 1: 'active', None: 'unknown'}[before[0]]
                f[category + '_control_count'] += 1
            if len(wires) == 2 and diagonal and any(v is not None for v in before):
                f['diagonal_reduced_count'] += 1
            if kind == 'swap':
                f['swap_count'] += 1
                if execution is True:
                    a, b = wires
                    nodes[a], nodes[b] = nodes[b], nodes[a]
                else:
                    potential = True  # Possible branch-dependent permutation.
            elif len(wires) == 2:
                if diagonal:
                    potential = not scalar and all(v is None for v in before)
                elif 'control' in op:
                    potential = before[0] is None
                else:
                    potential = True
            if potential:
                a, b = sorted(wires)
                f['potential_coupling_count'] += 1
                distance_sum += b - a
                long_range += int(b - a > 1)
                coupling_positions += position
                cuts[a] += 1
                cuts[b] -= 1
                x, y = find(nodes[a]), find(nodes[b])
                if x != y:
                    if size[x] < size[y]:
                        x, y = y, x
                    parent[y] = x
                    size[x] += size[y]
                    largest = max(largest, size[x])
                    components -= 1
                    f['potential_merge_count'] += 1
        tracker.apply(instruction)
        created = sum(value is not None and tracker.quantum[q] is None
                      for q, value in zip(wires, before))
        f['unknown_creation_count'] += created
        unknown_sum += tracker.unknown_count
        fraction = tracker.unknown_count / n if n else 0.0
        f['unknown_qubit_max_fraction'] = max(f['unknown_qubit_max_fraction'], fraction)
        if tracker.unknown_count:
            f['first_unknown_gate_fraction'] = min(f['first_unknown_gate_fraction'], position / total)
        largest_sum += largest
        for p, milestone in milestones.items():
            if position == milestone:
                f[f'prefix_largest_component_{p}_fraction'] = largest / n if n else 0.0
    crossing = 0
    for delta in cuts:
        crossing += delta
        f['potential_max_cut_crossings'] = max(f['potential_max_cut_crossings'], crossing)
    controls, couplings = f['controlled_gate_count'], f['potential_coupling_count']
    for category in ('inactive', 'active', 'unknown'):
        f[category + '_control_fraction'] = f[category + '_control_count'] / controls if controls else 0.0
    f.update(
        diagonal_reduced_fraction=f['diagonal_reduced_count'] / diagonal_pairs if diagonal_pairs else 0.0,
        unknown_qubit_mean_fraction=unknown_sum / (n * total) if n and total else 0.0,
        potential_coupling_fraction=couplings / pairs if pairs else 0.0,
        potential_long_range_fraction=long_range / couplings if couplings else 0.0,
        potential_mean_distance=distance_sum / couplings if couplings else 0.0,
        potential_mean_gate_position_fraction=coupling_positions / (total * couplings) if couplings else 0.0,
        prefix_largest_component_mean_fraction=largest_sum / (n * total) if n and total else 0.0,
        potential_largest_component_fraction=largest / n if n else 0.0,
        potential_component_count=components,
    )
    return f


def enrich_features(data, base):
    """Preserve all v1 fields while versioning the additional static inputs."""
    validate_features(base)
    if sum(r['size'] for r in data['qregs'].values()) != base['num_qubits']:
        raise ValueError('Circuit width does not match saved features')
    result = dict(base, **extract_structure(data))
    result['feature_version'] = FEATURE_VERSION
    base_features(result)
    return result
