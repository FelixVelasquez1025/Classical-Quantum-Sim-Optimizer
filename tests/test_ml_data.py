"""Dataset loading tests use JSON fixtures only, without running simulators."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ml.data import (BACKENDS, BOOLEAN_FEATURES, FEATURE_NAMES,
                     expected_options, load_dataset, validate_features)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def features():
    value = {name: (False if name in BOOLEAN_FEATURES else 0)
             for name in FEATURE_NAMES}
    value.update(feature_version=1, num_qubits=2, num_cbits=2,
                 circuit_depth=3, total_gate_count=2,
                 largest_interaction_component=2, interaction_component_count=1,
                 distinct_interacting_pairs=1, mean_two_qubit_distance=1.0,
                 max_two_qubit_distance=1, max_cut_crossings=1,
                 measurement_count=2, has_measurements=True,
                 terminal_only_measurements=True, terminal_sampling_candidate=True,
                 single_qubit_clifford_count=1, two_qubit_non_clifford_count=1,
                 single_qubit_clifford_fraction=0.5,
                 two_qubit_non_clifford_fraction=0.5)
    return value


def entry(path='a.qasm', family='a', **changes):
    result = dict(path=path, family=family, status='imported',
                  artifact=path + '.json', normalized_sha256=sha(path + '/normalized'),
                  source_sha256=sha(path + '/source'), num_qubits=2, num_cbits=2,
                  has_measurements=True, stabilizer_eligible=False)
    result.update(changes)
    return result


def configuration():
    return dict(schema_version=1, feature_version=1, run_id='fixture-run',
                normalization='common-one-two-qubit-v1',
                backends=list(BACKENDS), dataset_repository='fixture-repository',
                dataset_revision='fixture-revision', timeout_seconds=3600,
                settings=dict(task='auto', shots=1000, repeats=2, warmups=1,
                              threads=4, parallel_shots=1, max_memory_mb=1024, seed=71),
                provenance=dict(machine='fixture-machine', cpu_count=4))


def row(circuit, backend='mps', status='ok', config=None):
    config = config or configuration()
    result = dict(schema_version=1, run_id=config['run_id'],
                  circuit=copy.deepcopy(circuit), backend=backend, status=status,
                  features=features(), task='shots',
                  simulator_options=expected_options(backend, config['settings']))
    if status == 'ok':
        result.update(median_seconds=0.2, samples_seconds=[0.1, 0.3],
                      extension_sha256=sha('extension'), dqsim_version='0.1.2')
    return result


class MLDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, rows=None, entries=None, config=None, tail=b'', manifest=True):
        config = copy.deepcopy(config or configuration())
        entries = entries if entries is not None else [entry()]
        if manifest:
            raw_manifest = json.dumps(dict(normalization=config['normalization'],
                repository=config['dataset_repository'], revision=config['dataset_revision'],
                circuits=entries)).encode()
            (self.root / 'manifest.json').write_bytes(raw_manifest)
            config['manifest_sha256'] = hashlib.sha256(raw_manifest).hexdigest()
        (self.root / 'run.json').write_text(json.dumps(config))
        rows = rows if rows is not None else [row(entries[0])]
        payload = b''.join((json.dumps(r) + '\n').encode() for r in rows) + tail
        (self.root / 'results.jsonl').write_bytes(payload)
        return payload

    def test_completed_prefix_is_read_once_and_preserves_failures(self):
        circuit = entry()
        rows = [row(circuit, 'mps'), row(circuit, 'statevector', 'timeout'),
                row(circuit, 'stabilizer', 'ineligible')]
        payload = self.write(rows, tail=b'{"run_id": "incomplete')
        read_bytes = Path.read_bytes
        reads = []

        def read(path):
            reads.append(path.name)
            return read_bytes(path)

        with patch.object(Path, 'read_bytes', read):
            dataset = load_dataset(self.root)
        self.assertEqual(reads.count('results.jsonl'), 1)
        self.assertEqual(len(dataset['records']), 1)
        record = dataset['records'][0]
        self.assertEqual(record['rows']['statevector']['status'], 'timeout')
        self.assertNotIn('median_seconds', record['rows']['statevector'])
        self.assertEqual(record['features'], features())
        snapshot = dataset['snapshot']
        self.assertEqual(snapshot['results_sha256'], hashlib.sha256(payload).hexdigest())
        self.assertEqual(snapshot['results_bytes'], len(payload))
        self.assertEqual(snapshot['completed_rows'], 3)
        self.assertEqual(snapshot['completed_circuits'], 0)
        self.assertEqual(snapshot['missing_backends'], {'a.qasm': ['pblock']})
        self.assertGreater(snapshot['ignored_tail_bytes'], 0)
        self.assertEqual(snapshot['provenance']['machine'], 'fixture-machine')

    def test_final_json_without_newline_is_not_committed(self):
        payload = self.write([], tail=json.dumps(row(entry())).encode())
        dataset = load_dataset(self.root)
        self.assertEqual(dataset['records'], [])
        self.assertEqual(dataset['snapshot']['ignored_tail_bytes'], len(payload))

    def test_malformed_committed_line_is_an_error(self):
        self.write([], tail=b'{broken}\n')
        with self.assertRaisesRegex(ValueError, 'line 1: invalid JSON'):
            load_dataset(self.root)

    def test_groups_union_families_and_duplicates_through_unobserved_manifest(self):
        a = entry('a.qasm', 'family-a')
        # This unobserved variant connects family-a to family-b via its hash.
        bridge = entry('bridge.qasm', 'family-a', normalized_sha256=sha('shared'))
        b = entry('b.qasm', 'family-b', normalized_sha256=sha('shared'))
        c = entry('c.qasm', 'family-c', source_sha256=b['source_sha256'])
        d = entry('d.qasm', 'family-d', duplicate_of='c.qasm')
        independent = entry('independent.qasm', 'other')
        self.write([row(e) for e in (a, b, c, d, independent)],
                   [a, bridge, b, c, d, independent])
        dataset = load_dataset(self.root)
        groups = {r['path']: r['group'] for r in dataset['records']}
        self.assertEqual(len({groups[p] for p in ('a.qasm', 'b.qasm', 'c.qasm', 'd.qasm')}), 1)
        self.assertNotEqual(groups['a.qasm'], groups['independent.qasm'])
        self.assertEqual(dataset['snapshot']['observed_groups'], 2)

    def test_config_options_and_native_build_cannot_change_mid_run(self):
        circuit = entry()
        original = [row(circuit, 'mps'), row(circuit, 'pblock')]
        for edit, expected in (
            (lambda r: r.update(run_id='different'), 'run_id'),
            (lambda r: r['simulator_options'].update(max_memory_mb=2048), 'simulator_options'),
            (lambda r: r.update(extension_sha256=sha('new-build')), 'native extension'),
            (lambda r: r.update(dqsim_version='different'), 'dqsim versions'),
            (lambda r: r.update(schema_version=2), 'schema_version'),
            (lambda r: r.update(task='evolve'), 'task conflicts'),
        ):
            with self.subTest(expected=expected):
                rows = copy.deepcopy(original)
                edit(rows[1])
                self.write(rows)
                with self.assertRaisesRegex(ValueError, expected):
                    load_dataset(self.root)

    def test_duplicate_records_and_changed_circuit_data_are_rejected(self):
        circuit = entry()
        self.write([row(circuit), row(circuit)])
        with self.assertRaisesRegex(ValueError, 'duplicate circuit/backend'):
            load_dataset(self.root)
        first, second = row(circuit, 'mps'), row(circuit, 'pblock')
        second['features']['circuit_depth'] += 1
        self.write([first, second])
        with self.assertRaisesRegex(ValueError, 'inconsistent per-circuit'):
            load_dataset(self.root)
        second = row(circuit, 'pblock')
        second['circuit']['family'] = 'modified-family'
        self.write([first, second])
        with self.assertRaisesRegex(ValueError, 'frozen manifest'):
            load_dataset(self.root)

    def test_invalid_targets_are_rejected_and_failures_never_get_targets(self):
        edits = (
            lambda r: r.update(median_seconds=0),
            lambda r: r.update(median_seconds=float('nan')),
            lambda r: r.update(samples_seconds=[0.1, float('inf')]),
            lambda r: r.update(samples_seconds=[0.2]),
            lambda r: r.update(median_seconds=1.0),
            lambda r: r.update(status='timeout'),
        )
        for edit in edits:
            with self.subTest(edit=edit):
                r = row(entry())
                edit(r)
                self.write([r])
                with self.assertRaises(ValueError):
                    load_dataset(self.root)

    def test_feature_schema_is_explicit_and_validates_finite_values(self):
        self.assertEqual(len(FEATURE_NAMES), 28)
        self.assertEqual(validate_features(features()), features())
        for name, value in (('num_qubits', -1), ('num_qubits', True),
                            ('num_qubits', 2.5), ('mean_two_qubit_distance', float('nan')),
                            ('has_measurements', 1), ('diagonal_gate_fraction', 1.5),
                            ('feature_version', 2)):
            with self.subTest(name=name, value=value):
                f = features()
                f[name] = value
                with self.assertRaises(ValueError):
                    validate_features(f)
        for edit in (lambda f: f.pop('num_qubits'), lambda f: f.update(median_seconds=1)):
            f = features()
            edit(f)
            with self.assertRaisesRegex(ValueError, 'Feature schema mismatch'):
                validate_features(f)

    def test_manifest_fingerprint_and_normalization_are_verified(self):
        self.write()
        manifest = self.root / 'manifest.json'
        manifest.write_bytes(manifest.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'hash does not match'):
            load_dataset(self.root)
        self.write()
        config_path = self.root / 'run.json'
        config = json.loads(config_path.read_text())
        config['normalization'] = 'different-normalization'
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'normalization conflicts'):
            load_dataset(self.root)

    def test_missing_manifest_has_explicit_warning(self):
        self.write(manifest=False)
        dataset = load_dataset(self.root)
        self.assertEqual(dataset['snapshot']['manifest_sha256'], None)
        self.assertTrue(any('No frozen manifest' in w for w in dataset['snapshot']['warnings']))

    def test_zero_rows_is_a_valid_running_snapshot(self):
        self.write([])
        dataset = load_dataset(self.root)
        self.assertEqual(dataset['records'], [])
        self.assertEqual(dataset['snapshot']['completed_rows'], 0)
        self.assertEqual(dataset['snapshot']['expected_circuits'], 1)

    def test_ineligible_is_only_supported_for_stabilizer(self):
        self.write([row(entry(), 'mps', 'ineligible')])
        with self.assertRaisesRegex(ValueError, 'only stabilizer'):
            load_dataset(self.root)


if __name__ == '__main__':
    unittest.main()
