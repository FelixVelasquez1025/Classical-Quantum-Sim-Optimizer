"""Collector checks use fake workers only; no native simulations are executed."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    'collector', Path(__file__).resolve().parents[1] / 'scripts/collect_simulator_data.py')
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class CollectionTests(unittest.TestCase):
    def test_task_selection(self):
        unitary = {'instructions': []}
        measured = {'instructions': [{'kind': 'conditional', 'op': {'kind': 'measure'}}]}
        self.assertEqual(collector.task_for(unitary, 'auto'), 'evolve')
        self.assertEqual(collector.task_for(measured, 'auto'), 'shots')
        self.assertEqual(collector.task_for(measured, 'evolve'), 'evolve')
        with self.assertRaises(ValueError):
            collector.task_for(unitary, 'shots')

    def test_warmups_excluded_and_results_checked(self):
        simulator = SimpleNamespace(simulate=lambda circuit: SimpleNamespace(num_qubits=2))
        with patch.object(collector.time, 'perf_counter_ns', side_effect=[0, 9e9, 0, 2e9, 0, 4e9]):
            result = collector.measure(simulator, SimpleNamespace(num_qubits=2), 'evolve',
                                       {'warmups': 1, 'repeats': 2})
        self.assertEqual(result['samples_seconds'], [2, 4])
        self.assertEqual(result['median_seconds'], 3)
        simulator = SimpleNamespace(simulate_shots=lambda circuit, shots: {'0': shots - 1})
        with self.assertRaises(RuntimeError):
            collector.measure(simulator, None, 'shots', {'warmups': 0, 'repeats': 1, 'shots': 10})

    def test_labels_require_every_eligible_backend(self):
        rows = [dict(backend=b, status='ok', median_seconds=i+1)
                for i, b in enumerate(collector.BACKENDS)]
        self.assertEqual(collector.summarize(rows)['fastest_backend'], 'statevector')
        rows[-1]['status'] = 'ineligible'
        self.assertTrue(collector.summarize(rows)['complete'])
        rows[0]['status'] = 'timeout'
        self.assertIsNone(collector.summarize(rows)['fastest_backend'])
        self.assertFalse(collector.summarize(rows[1:])['complete'])
        duplicates = [dict(backend='mps', status='ok', median_seconds=1)] * 4
        self.assertFalse(collector.summarize(duplicates)['complete'])

    def test_worker_timeout_crash_and_environment(self):
        request = {'settings': {'threads': 3}}
        with patch.object(collector.subprocess, 'run', side_effect=subprocess.TimeoutExpired('fake', 2)):
            self.assertEqual(collector.run_pair(request, 2)['status'], 'timeout')
        with patch.object(collector.subprocess, 'run', return_value=SimpleNamespace(
                returncode=-9, stderr='killed', stdout='')):
            self.assertEqual(collector.run_pair(request, 2)['status'], 'crash')
        with patch.object(collector.subprocess, 'run', return_value=SimpleNamespace(
                returncode=0, stderr='', stdout='not JSON')) as run:
            self.assertEqual(collector.run_pair(request, 2)['status'], 'error')
            self.assertEqual(run.call_args.kwargs['env']['RAYON_NUM_THREADS'], '3')
            self.assertEqual(run.call_args.kwargs['env']['OPENBLAS_NUM_THREADS'], '1')

    def test_no_timeout_passed_to_subprocess(self):
        with patch.object(collector.subprocess, 'run', return_value=SimpleNamespace(
                returncode=0, stderr='', stdout='{"status": "ok"}')) as run:
            result = collector.run_pair({'settings': {'threads': 1}}, None)
        self.assertEqual(result['status'], 'ok')
        self.assertIsNone(run.call_args.kwargs['timeout'])

    def test_resource_preflight_preserves_compact_backends(self):
        settings = {'max_memory_mb': 1024}
        self.assertIsNone(collector.resource_preflight('statevector', 26, settings))
        self.assertEqual(collector.resource_preflight('statevector', 27, settings)['status'],
                         'resource_exceeded')
        self.assertEqual(collector.resource_preflight('statevector', 10000, settings)['status'],
                         'resource_exceeded')
        for backend in ('mps', 'pblock', 'stabilizer'):
            self.assertIsNone(collector.resource_preflight(backend, 433, settings))

    def test_resource_limited_winner_is_separate_from_unrestricted_label(self):
        rows = [dict(backend=b, status='ok', median_seconds=i+1)
                for i, b in enumerate(collector.BACKENDS)]
        rows[0] = dict(backend='statevector', status='resource_exceeded')
        rows[2] = dict(backend='pblock', status='memory_error')
        result = collector.summarize(rows)
        self.assertFalse(result['complete'])
        self.assertIsNone(result['fastest_backend'])
        self.assertEqual(result['fastest_feasible_backend'], 'mps')
        self.assertEqual(result['resource_limited_backends'], ['statevector', 'pblock'])
        rows[2]['status'] = 'timeout'
        self.assertIsNone(collector.summarize(rows)['fastest_feasible_backend'])

    def fixture(self, root):
        data = {'qregs': {}, 'cregs': {}, 'instructions': []}
        entry = dict(path='example.qasm', artifact='example.json', family='example',
                     status='imported', source_sha256='source',
                     normalized_sha256=collector.digest(json.dumps(data, separators=(',', ':')).encode()))
        (root/'example.json').write_text(json.dumps(
            {'circuit': data, 'metadata': {'source_sha256': 'source'}}))
        manifest = dict(repository='fixture', revision='fixture', normalization='fixture', circuits=[entry])
        (root/'manifest.json').write_text(json.dumps(manifest))
        return entry

    def test_artifact_integrity_and_path_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = self.fixture(root)
            collector.load_artifact(root, entry)
            entry['normalized_sha256'] = 'wrong'
            with self.assertRaises(ValueError):
                collector.load_artifact(root, entry)
            entry['artifact'] = '../outside.json'
            with self.assertRaises(ValueError):
                collector.load_artifact(root, entry)

    def test_large_circuit_skips_only_dense_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = self.fixture(root)
            data = {'qregs': {'q': {'name': 'q', 'base': 0, 'size': 433}},
                    'cregs': {}, 'instructions': []}
            entry['normalized_sha256'] = collector.digest(
                json.dumps(data, separators=(',', ':')).encode())
            (root/'example.json').write_text(json.dumps(
                {'circuit': data, 'metadata': {'source_sha256': 'source'}}))
            manifest = json.loads((root/'manifest.json').read_text())
            manifest['circuits'] = [entry]
            (root/'manifest.json').write_text(json.dumps(manifest))
            with patch.object(collector, 'run_pair', return_value={
                    'status': 'ok', 'median_seconds': 1.0}) as run, \
                 patch.object(collector, 'provenance', return_value={}), \
                 patch.object(collector, 'gate_traits_classifier', return_value=lambda op: (True, False)), \
                 contextlib.redirect_stdout(io.StringIO()):
                collector.main(['--manifest', str(root/'manifest.json'), '--imports', str(root),
                                '--output', str(root/'output'), '--timeout-seconds', '0'])
            self.assertEqual({call.args[0]['backend'] for call in run.call_args_list},
                             {'mps', 'pblock', 'stabilizer'})
            destination, = (root/'output').iterdir()
            rows = [json.loads(line) for line in (destination/'results.jsonl').read_text().splitlines()]
            dense, = [row for row in rows if row['backend'] == 'statevector']
            self.assertEqual(dense['status'], 'resource_exceeded')
            self.assertEqual(dense['features']['num_qubits'], 433)
            label = json.loads((destination/'labels.jsonl').read_text())
            self.assertIsNone(label['fastest_backend'])
            self.assertIsNotNone(label['fastest_feasible_backend'])

    def test_output_pipeline_with_fake_workers_only(self):
        def fake_pair(request, timeout):
            return dict(status='ok', median_seconds=1.0, samples_seconds=[1.0],
                        simulator_options=collector.options_for(request['backend'], request['settings']))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            with patch.object(collector, 'run_pair', side_effect=fake_pair) as run, \
                 patch.object(collector, 'provenance', return_value={'test': True}), \
                 patch.object(collector, 'gate_traits_classifier', return_value=lambda op: (True, False)), \
                 contextlib.redirect_stdout(io.StringIO()):
                collector.main(['--manifest', str(root/'manifest.json'), '--imports', str(root),
                                '--output', str(root/'output'), '--timeout-seconds', '0'])
            self.assertEqual(run.call_count, 4)
            self.assertTrue(all(call.args[1] is None for call in run.call_args_list))
            destination, = (root/'output').iterdir()
            self.assertIsNone(json.loads((destination/'run.json').read_text())['timeout_seconds'])
            rows = [json.loads(line) for line in (destination/'results.jsonl').read_text().splitlines()]
            self.assertEqual({row['backend'] for row in rows}, set(collector.BACKENDS))
            self.assertTrue(all(row['task'] == 'evolve' for row in rows))
            self.assertTrue(all(row['features']['total_gate_count'] == 0 for row in rows))
            label = json.loads((destination/'labels.jsonl').read_text())
            self.assertEqual(label['features'], rows[0]['features'])
            self.assertTrue(json.loads((destination/'labels.jsonl').read_text())['complete'])
            self.assertEqual((destination/'manifest.json').read_bytes(), (root/'manifest.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
