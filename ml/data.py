"""Validated, read-only snapshots of a single simulator collection run.

The collector may keep appending while this module reads. Only complete JSONL
lines from one read are used; no labels or timing values are invented for failed
or unfinished simulator/circuit pairs. No simulator is imported or executed.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

BACKENDS = ('statevector', 'mps', 'pblock', 'stabilizer')
STATUSES = frozenset(('ok', 'ineligible', 'memory_error', 'resource_exceeded',
                      'timeout', 'crash', 'error'))
FEATURE_NAMES = (
    'num_qubits', 'num_cbits', 'circuit_depth', 'total_gate_count',
    'largest_interaction_component', 'interaction_component_count',
    'distinct_interacting_pairs', 'mean_two_qubit_distance',
    'max_two_qubit_distance', 'max_cut_crossings', 'measurement_count',
    'reset_count', 'conditional_count', 'has_measurements',
    'terminal_only_measurements', 'terminal_sampling_candidate',
    'has_mid_circuit_measurements', 'diagonal_gate_count',
    'diagonal_gate_fraction', 'stabilizer_eligible',
    'single_qubit_clifford_count', 'single_qubit_non_clifford_count',
    'two_qubit_clifford_count', 'two_qubit_non_clifford_count',
    'single_qubit_clifford_fraction', 'single_qubit_non_clifford_fraction',
    'two_qubit_clifford_fraction', 'two_qubit_non_clifford_fraction',
)
BOOLEAN_FEATURES = frozenset((
    'has_measurements', 'terminal_only_measurements',
    'terminal_sampling_candidate', 'has_mid_circuit_measurements',
    'stabilizer_eligible',
))


def _hash(payload):
    return hashlib.sha256(payload).hexdigest()


def _object(value, context):
    if not isinstance(value, dict):
        raise ValueError(f'{context} must be a JSON object')
    return value


def _text(value, context):
    if not isinstance(value, str) or not value:
        raise ValueError(f'{context} must be a nonempty string')
    return value


def _sha(value, context):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError(f'{context} must be a lowercase SHA-256 digest')
    return value


def _number(value, context, *, positive=False, integer=False):
    valid = (type(value) is int if integer else type(value) in (int, float))
    try:
        valid = valid and math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        valid = False
    if not valid:
        adjective = 'positive' if positive else 'nonnegative'
        raise ValueError(f'{context} must be a finite {adjective} {"integer" if integer else "number"}')
    return value


def _json(payload, context):
    try:
        return json.loads(payload)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f'{context}: invalid JSON: {exc}') from exc


def validate_features(features):
    """Validate the exact static schema; return the original feature dictionary."""
    _object(features, 'features')
    if type(features.get('feature_version')) is not int or features['feature_version'] != 1:
        raise ValueError('features.feature_version must be 1')
    required = set(FEATURE_NAMES) | {'feature_version'}
    if set(features) != required:
        missing, extra = sorted(required - set(features)), sorted(set(features) - required)
        raise ValueError(f'Feature schema mismatch: missing={missing}, extra={extra}')
    for name in FEATURE_NAMES:
        value = features[name]
        if name in BOOLEAN_FEATURES:
            if type(value) is not bool:
                raise ValueError(f'Feature {name} must be a boolean')
        else:
            fractional = name.endswith('_fraction') or name == 'mean_two_qubit_distance'
            _number(value, f'Feature {name}', integer=not fractional)
            if name.endswith('_fraction') and value > 1:
                raise ValueError(f'Feature {name} must be at most 1')
    return features


def expected_options(backend, settings):
    """The exact settings used by collector version 1, without importing it."""
    options = dict(seed=settings['seed'], max_memory_mb=settings['max_memory_mb'],
                   max_parallel_shots=settings['parallel_shots'], sample_terminal=True)
    if backend == 'mps':
        options.update(max_bond_dimension=None, truncation_threshold=0.0,
                       max_discarded_weight=0.0)
    elif backend == 'pblock':
        options.update(split_separable=False)
    elif backend == 'stabilizer':
        options.update(clifford_tolerance=0.0)
    return options


def _validate_config(config):
    _object(config, 'run.json')
    for name in ('schema_version', 'feature_version'):
        if type(config.get(name)) is not int or config[name] != 1:
            raise ValueError(f'run.json {name} must be 1')
    _text(config.get('run_id'), 'run.json run_id')
    _text(config.get('normalization'), 'run.json normalization')
    backends = config.get('backends')
    if (not isinstance(backends, list)
            or any(not isinstance(backend, str) for backend in backends)
            or sorted(backends) != sorted(BACKENDS)):
        raise ValueError('run.json must list each of the four known backends exactly once')
    settings = _object(config.get('settings'), 'run.json settings')
    if settings.get('task') not in ('auto', 'shots', 'evolve'):
        raise ValueError('run.json task must be auto, shots, or evolve')
    for name in ('shots', 'repeats', 'threads', 'parallel_shots', 'max_memory_mb'):
        _number(settings.get(name), f'run.json settings.{name}', positive=True, integer=True)
    for name in ('warmups', 'seed'):
        _number(settings.get(name), f'run.json settings.{name}', integer=True)
    if settings['seed'] >= 2**64:
        raise ValueError('run.json seed must fit an unsigned 64-bit integer')
    if config.get('timeout_seconds') is not None:
        _number(config['timeout_seconds'], 'run.json timeout_seconds', positive=True)
    if 'provenance' in config:
        _object(config['provenance'], 'run.json provenance')


def _manifest(run_dir, config, warnings):
    path = run_dir / 'manifest.json'
    if not path.exists():
        warnings.append('No frozen manifest.json: circuit metadata and duplicate groups can only be checked against observed rows.')
        return {}, None
    raw = path.read_bytes()
    fingerprint = _hash(raw)
    if config.get('manifest_sha256') != fingerprint:
        raise ValueError('Frozen manifest.json hash does not match run.json')
    manifest = _object(_json(raw, 'manifest.json'), 'manifest.json')
    for manifest_key, config_key in (('normalization', 'normalization'),
                                     ('repository', 'dataset_repository'),
                                     ('revision', 'dataset_revision')):
        if manifest.get(manifest_key) != config.get(config_key):
            raise ValueError(f'manifest.json {manifest_key} conflicts with run.json')
    if not isinstance(manifest.get('circuits'), list):
        raise ValueError('manifest.json circuits must be a list')
    entries = {}
    for entry in manifest['circuits']:
        _object(entry, 'manifest entry')
        key = _text(entry.get('path'), 'manifest circuit path')
        if key in entries:
            raise ValueError(f'Duplicate circuit path in manifest: {key}')
        entries[key] = entry
    return entries, fingerprint


def _groups(records, entries):
    """Union families, source/normalized duplicates, and explicit references.

    All frozen manifest entries participate, even those with no completed row.
    Otherwise an unobserved circuit could bridge two families across folds.
    """
    metadata = dict(entries)
    metadata.update({record['path']: record['metadata'] for record in records})
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        left, right = find(a), find(b)
        if left != right:
            parent[max(left, right)] = min(left, right)

    seen = {}
    for path, entry in metadata.items():
        find(path)
        for field in ('family', 'normalized_sha256', 'source_sha256'):
            value = entry.get(field)
            if value:
                _text(value, f'{path}: {field}')
                token = (field, value)
                if token in seen:
                    union(path, seen[token])
                else:
                    seen[token] = path
        if entry.get('duplicate_of'):
            reference = _text(entry['duplicate_of'], f'{path}: duplicate_of')
            union(path, reference)
    components = {}
    for key in parent:
        components.setdefault(find(key), []).append(key)
    names = {root: 'group-' + _hash(json.dumps(sorted(paths)).encode())[:16]
             for root, paths in components.items()}
    for record in records:
        record['group'] = names[find(record['path'])]


def load_dataset(run_dir: Path):
    """Read a stable prefix of one run; validate it and group related circuits.

    Successful rows are usable as regression targets even when another backend
    is pending. ``rows`` retains failed outcomes for honest selection evaluation;
    failure/timeout wall times are never converted to simulated runtimes.
    """
    run_dir = Path(run_dir).resolve()
    config_bytes = (run_dir / 'run.json').read_bytes()
    config = _json(config_bytes, 'run.json')
    _validate_config(config)
    settings, warnings = config['settings'], []
    entries, manifest_sha = _manifest(run_dir, config, warnings)

    # Read exactly once. A writer may append after this, but this training
    # snapshot will not accidentally combine observations from different reads.
    raw = (run_dir / 'results.jsonl').read_bytes()
    prefix_end = raw.rfind(b'\n') + 1
    prefix, tail = raw[:prefix_end], raw[prefix_end:]
    if tail:
        warnings.append(f'Ignored {len(tail)} bytes in an unfinished final JSONL line.')
    by_path, statuses, extension_hashes, dqsim_versions = {}, Counter(), set(), set()
    for line_number, line in enumerate(prefix.splitlines(), 1):
        context = f'results.jsonl line {line_number}'
        row = _object(_json(line, context), context)
        if type(row.get('schema_version')) is not int or row['schema_version'] != 1:
            raise ValueError(f'{context}: schema_version must be 1')
        if row.get('run_id') != config['run_id']:
            raise ValueError(f'{context}: run_id differs from run.json')
        backend, status = row.get('backend'), row.get('status')
        if (not isinstance(backend, str) or not isinstance(status, str)
                or backend not in BACKENDS or status not in STATUSES):
            raise ValueError(f'{context}: unknown backend or status')
        if status == 'ineligible' and backend != 'stabilizer':
            raise ValueError(f'{context}: only stabilizer may be marked ineligible')
        if row.get('simulator_options') != expected_options(backend, settings):
            raise ValueError(f'{context}: simulator_options differ from the run settings')
        features = validate_features(row.get('features'))
        task = row.get('task')
        expected_task = settings['task']
        if expected_task == 'auto':
            expected_task = 'shots' if features['has_measurements'] else 'evolve'
        if task != expected_task or (task == 'shots' and not features['has_measurements']):
            raise ValueError(f'{context}: task conflicts with run settings or measurements')
        entry = _object(row.get('circuit'), f'{context}: circuit')
        path = _text(entry.get('path'), f'{context}: circuit path')
        family = _text(entry.get('family'), f'{context}: family')
        normalized_sha = _sha(entry.get('normalized_sha256'), f'{context}: normalized_sha256')
        _sha(entry.get('source_sha256'), f'{context}: source_sha256')
        if entry.get('status') != 'imported':
            raise ValueError(f'{context}: circuit is not an imported artifact')
        if manifest_sha is not None and entries.get(path) != entry:
            raise ValueError(f'{context}: circuit metadata conflicts with the frozen manifest')
        for field in ('num_qubits', 'num_cbits', 'has_measurements', 'stabilizer_eligible'):
            if field in entry and entry[field] != features[field]:
                raise ValueError(f'{context}: feature {field} conflicts with circuit metadata')
        if 'normalization' in entry and entry['normalization'] != config['normalization']:
            raise ValueError(f'{context}: conflicting normalization')

        if status == 'ok':
            _number(row.get('median_seconds'), f'{context}: median_seconds', positive=True)
            samples = row.get('samples_seconds')
            if not isinstance(samples, list) or len(samples) != settings['repeats']:
                raise ValueError(f'{context}: samples_seconds must have repeats entries')
            for sample in samples:
                _number(sample, f'{context}: sample', positive=True)
            if not math.isclose(statistics.median(samples), row['median_seconds'], rel_tol=1e-9):
                raise ValueError(f'{context}: median_seconds disagrees with samples_seconds')
        elif 'median_seconds' in row or 'samples_seconds' in row:
            raise ValueError(f'{context}: non-ok outcomes must not have runtime targets')
        if 'extension_sha256' in row:
            extension_hashes.add(_sha(row['extension_sha256'], f'{context}: extension_sha256'))
        if 'dqsim_version' in row:
            dqsim_versions.add(_text(row['dqsim_version'], f'{context}: dqsim_version'))

        if path not in by_path:
            by_path[path] = dict(path=path, family=family,
                                 normalized_sha256=normalized_sha, features=features,
                                 task=task, rows={}, metadata=entry)
        record = by_path[path]
        if (record['metadata'] != entry or record['features'] != features
                or record['task'] != task):
            raise ValueError(f'{context}: inconsistent per-circuit metadata, features, or task')
        if backend in record['rows']:
            raise ValueError(f'{context}: duplicate circuit/backend record: {path}/{backend}')
        record['rows'][backend] = row
        statuses[status] += 1

    if len(extension_hashes) > 1 or len(dqsim_versions) > 1:
        raise ValueError('A run contains conflicting native extension hashes or dqsim versions')
    records = sorted(by_path.values(), key=lambda record: record['path'])
    _groups(records, entries)
    missing = {record['path']: [b for b in BACKENDS if b not in record['rows']]
               for record in records if len(record['rows']) != len(BACKENDS)}
    if missing:
        warnings.append(f'{len(missing)} observed circuits have pending backend rows; their completed measurements remain usable.')
    expected_circuits = sum(entry.get('status') == 'imported' for entry in entries.values())
    if manifest_sha is not None and len(records) < expected_circuits:
        warnings.append(f'{expected_circuits - len(records)} imported circuits have no completed rows in this snapshot.')
    if not extension_hashes:
        warnings.append('No native extension fingerprint is available in this snapshot.')
    snapshot = dict(
        run_dir=str(run_dir), run_id=config['run_id'],
        results_sha256=_hash(raw), results_bytes=len(raw),
        completed_prefix_sha256=_hash(prefix), completed_prefix_bytes=len(prefix),
        ignored_tail_bytes=len(tail), run_config_sha256=_hash(config_bytes),
        manifest_sha256=manifest_sha, completed_rows=sum(statuses.values()),
        observed_circuits=len(records), observed_groups=len({r['group'] for r in records}),
        completed_circuits=sum(len(r['rows']) == len(BACKENDS) for r in records),
        expected_circuits=expected_circuits if manifest_sha is not None else None,
        status_counts=dict(sorted(statuses.items())), missing_backends=missing,
        extension_sha256=next(iter(extension_hashes), None),
        dqsim_version=next(iter(dqsim_versions), None),
        settings=dict(settings), provenance=dict(config.get('provenance', {})),
        warnings=warnings,
    )
    return dict(config=config, records=records, snapshot=snapshot)


# Conservative validation groups: variants stay together even when their
# source datasets use different algorithm names. These affect splitting only.
FAMILY_ALIASES = {
    **dict.fromkeys(('ghz', 'ghz_state', 'cat', 'cat_state'), 'ghz'),
    **dict.fromkeys(('deutsch', 'dj'), 'deutsch_jozsa'),
    **dict.fromkeys(('qft', 'inverseqft', 'qftentangled'), 'qft'),
    **dict.fromkeys(('qpe', 'pea', 'ipea', 'qpeexact', 'qpeinexact'), 'phase_estimation'),
    **dict.fromkeys(('grover', 'grover_noancilla', 'grover_v_chain'), 'grover'),
    **dict.fromkeys(('quantumwalks', 'qwalk_noancilla', 'qwalk_v_chain'), 'quantum_walk'),
    **dict.fromkeys(('qaoa', 'portfolioqaoa'), 'qaoa'),
    **dict.fromkeys(('vqe', 'portfoliovqe'), 'vqe'),
    **dict.fromkeys(('realamprandom', 'twolocalrandom', 'su2random'), 'random_ansatz'),
    **dict.fromkeys(('pricingcall', 'pricingput'), 'option_pricing'),
    **dict.fromkeys(('adder', 'bigadder'), 'adder'),
    **dict.fromkeys(('multiply', 'multiplier'), 'multiplier'),
}


def load_datasets(run_dirs):
    """Combine validated snapshots without mixing incompatible benchmark settings.

    Exact normalized duplicates contribute once. Prefer the most complete
    observation, then input run order; never choose based on measured speed.
    Group using *all* manifest entries before deduplication, including unobserved
    and excluded entries that can link algorithm families across sources.
    """
    import copy
    datasets = [load_dataset(path) for path in run_dirs]
    if not datasets:
        raise ValueError('At least one collection run is required')
    if len(datasets) == 1:
        return datasets[0]
    settings = dict(datasets[0]['config']['settings'])
    tasks = {r['task'] for d in datasets for r in d['records']}
    if len(tasks) != 1:
        raise ValueError('Multi-run training currently requires a common observed task')
    settings['task'] = next(iter(tasks))
    records, entries = [], {}
    for dataset in datasets:
        config, snapshot = dataset['config'], dataset['snapshot']
        candidate = dict(config['settings']); candidate['task'] = settings['task']
        if candidate != settings:
            raise ValueError('Cannot combine different benchmark settings')
        first = datasets[0]
        for field in ('normalization', 'timeout_seconds', 'feature_version'):
            if config[field] != first['config'][field]:
                raise ValueError(f'Cannot combine different {field}')
        if not snapshot['extension_sha256'] or snapshot['extension_sha256'] != first['snapshot']['extension_sha256']:
            raise ValueError('Cannot combine different or unknown native simulator builds')
        for field in ('python', 'machine', 'processor', 'cpu_count', 'platform'):
            if snapshot['provenance'].get(field) != first['snapshot']['provenance'].get(field):
                raise ValueError(f'Cannot combine different machine provenance: {field}')
        prefix = config['run_id'] + '/'
        manifest = json.loads((Path(snapshot['run_dir'])/'manifest.json').read_text())
        for entry in manifest['circuits']:
            entry = copy.deepcopy(entry)
            entry['path'] = prefix + entry['path']
            if entry.get('duplicate_of'): entry['duplicate_of'] = prefix + entry['duplicate_of']
            entries[entry['path']] = entry
        for record in dataset['records']:
            record = copy.deepcopy(record)
            record['source_run_id'] = config['run_id']
            record['source_path'] = record['path']
            record['path'] = prefix + record['path']
            record['metadata'] = entries[record['path']]
            records.append(record)
    group_entries = copy.deepcopy(entries)
    for entry in group_entries.values():
        family = entry.get('family', '').lower().replace('-', '_')
        entry['family'] = FAMILY_ALIASES.get(family, family)
    grouped = [dict(r, metadata=group_entries[r['path']]) for r in records]
    _groups(grouped, group_entries)
    for record, group in zip(records, grouped): record['group'] = group['group']
    unique = {}
    for record in records:
        key = (record['normalized_sha256'], record['task'])
        previous = unique.get(key)
        if previous is not None and previous['features'] != record['features']:
            raise ValueError('Identical normalized circuits have inconsistent features')
        if previous is None or len(record['rows']) > len(previous['rows']): unique[key] = record
    combined = sorted(unique.values(), key=lambda r: r['path'])
    snapshot = dict(source_snapshots=[d['snapshot'] for d in datasets],
                    observed_circuits=len(combined), observed_groups=len({r['group'] for r in combined}),
                    removed_duplicate_observations=len(records)-len(combined),
                    family_aliases=FAMILY_ALIASES,
                    warnings=[w for d in datasets for w in d['snapshot']['warnings']])
    config = dict(datasets[0]['config'], settings=settings,
                  source_configs=[d['config'] for d in datasets], run_id='combined')
    return dict(config=config, records=combined, snapshot=snapshot)
