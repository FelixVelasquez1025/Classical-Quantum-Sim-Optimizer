"""Power-loss and resume tests use fake workers in temporary directories only."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_collection import collector


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        data = dict(qregs={}, cregs={}, instructions=[])
        entry = dict(path='test.qasm', artifact='test.json', family='test', status='imported',
                     source_sha256='a'*64,
                     normalized_sha256=collector.digest(json.dumps(data, separators=(',', ':')).encode()))
        (self.root/'test.json').write_text(json.dumps(dict(circuit=data, metadata={'source_sha256':'a'*64})))
        (self.root/'manifest.json').write_text(json.dumps(dict(repository='fixture', revision='fixture',
                                                             normalization='fixture', circuits=[entry])))
        for name, value in [('provenance', {}), ('current_native_hash', 'b'*64),
                            ('gate_traits_classifier', lambda op: (True, False))]:
            patcher = patch.object(collector, name, return_value=value)
            patcher.start();self.addCleanup(patcher.stop)
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__();self.addCleanup(self.stdout.__exit__, None, None, None)

    def ok(self, request, timeout):
        return dict(status='ok', median_seconds=1., samples_seconds=[1.]*5,
                    extension_sha256='b'*64, dqsim_version='fixture')

    def initial(self, interrupted=False):
        calls = []
        def run(request, timeout):
            if interrupted and calls:
                raise KeyboardInterrupt()
            calls.append(request['backend'])
            return self.ok(request, timeout)
        with patch.object(collector, 'run_pair', side_effect=run):
            args = ['--manifest', str(self.root/'manifest.json'), '--imports', str(self.root),
                    '--output', str(self.root/'runs'), '--timeout-seconds', '3600']
            if interrupted:
                with self.assertRaises(KeyboardInterrupt): collector.main(args)
            else:
                collector.main(args)
        destination, = (self.root/'runs').iterdir()
        return destination, calls

    def test_resume_runs_only_missing_pairs_and_inherits_timeout(self):
        destination, first = self.initial(interrupted=True)
        prior = (destination/'results.jsonl').read_bytes()
        with patch.object(collector, 'run_pair', side_effect=self.ok) as run:
            collector.main(['--resume', str(destination)])
        self.assertEqual(run.call_count, 3)
        self.assertNotIn(first[0], [call.args[0]['backend'] for call in run.call_args_list])
        self.assertTrue(all(call.args[1] == 3600 for call in run.call_args_list))
        self.assertTrue((destination/'results.jsonl').read_bytes().startswith(prior))
        self.assertEqual(len((destination/'results.jsonl').read_text().splitlines()), 4)
        self.assertEqual(len((destination/'labels.jsonl').read_text().splitlines()), 1)
        with patch.object(collector, 'run_pair') as run:
            collector.main(['--resume', str(destination)])
            run.assert_not_called()
        self.assertEqual(len((destination/'labels.jsonl').read_text().splitlines()), 1)

    def test_partial_tail_is_backed_up_and_labels_rebuilt(self):
        destination, _ = self.initial(interrupted=True)
        tail = b'{"status":"ok"'
        with (destination/'results.jsonl').open('ab') as stream: stream.write(tail)
        (destination/'labels.jsonl').write_text('{broken')
        with patch.object(collector, 'run_pair', side_effect=self.ok):
            collector.main(['--resume', str(destination)])
        backup, = destination.glob('results-interrupted-*.bin')
        self.assertEqual(backup.read_bytes(), tail)
        self.assertEqual(len([json.loads(x) for x in (destination/'results.jsonl').read_text().splitlines()]), 4)
        self.assertTrue(json.loads((destination/'labels.jsonl').read_text())['complete'])

    def test_missing_label_does_not_rerun_completed_work(self):
        destination, _ = self.initial()
        (destination/'labels.jsonl').unlink()
        with patch.object(collector, 'run_pair') as run:
            collector.main(['--resume', str(destination)])
            run.assert_not_called()
        self.assertEqual(len((destination/'labels.jsonl').read_text().splitlines()), 1)

    def test_saved_timeout_is_not_retried(self):
        destination, _ = self.initial(interrupted=True)
        row = json.loads((destination/'results.jsonl').read_text())
        row.pop('samples_seconds');row.pop('median_seconds')
        row['status'] = 'timeout';row['timeout_seconds'] = 3600
        (destination/'results.jsonl').write_text(json.dumps(row)+'\n')
        with patch.object(collector, 'run_pair', side_effect=self.ok) as run:
            collector.main(['--resume', str(destination)])
        self.assertEqual(run.call_count, 3)
        self.assertNotIn(row['backend'], [c.args[0]['backend'] for c in run.call_args_list])

    def test_conflicting_options_and_native_changes_leave_data_untouched(self):
        destination, _ = self.initial(interrupted=True)
        prior = (destination/'results.jsonl').read_bytes()
        for options in (['--shots', '23'], ['--timeout-seconds', '1'], ['--manifest', 'elsewhere']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                collector.main(['--resume', str(destination), *options])
        with patch.object(collector, 'current_native_hash', return_value='c'*64), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            collector.main(['--resume', str(destination)])
        self.assertEqual((destination/'results.jsonl').read_bytes(), prior)

    def test_frozen_manifest_and_run_lock(self):
        destination, _ = self.initial(interrupted=True)
        (self.root/'manifest.json').write_text('the current manifest may change')
        with collector.run_lock(destination), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            collector.main(['--resume', str(destination)])
        with patch.object(collector, 'run_pair', side_effect=self.ok):
            collector.main(['--resume', str(destination)])

    def test_committed_corruption_is_not_discarded(self):
        destination, _ = self.initial(interrupted=True)
        with (destination/'results.jsonl').open('ab') as stream: stream.write(b'{bad}\n')
        prior = (destination/'results.jsonl').read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            collector.main(['--resume', str(destination)])
        self.assertEqual((destination/'results.jsonl').read_bytes(), prior)


if __name__ == '__main__':
    unittest.main()
