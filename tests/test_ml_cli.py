"""One public command imports/collects; one trainer exports the selector."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_collection import collector
from ml.train_selector import main as train
from ml.predict_selector import main as predict
from ml.data import load_dataset


class CLITests(unittest.TestCase):
    def test_collect_train_export_predict_resume_without_benchmarking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'circuits'
            for i in range(4):
                folder = source/f'family{i}'; folder.mkdir(parents=True)
                (folder/'manifest.qasm').write_text('OPENQASM 2.0; include "qelib1.inc"; '
                    f'qreg q[{i+2}]; creg c[1]; h q[0]; t q[0]; measure q[0] -> c[0];')
            def fake_worker(request, timeout):
                if request['backend'] == 'stabilizer':
                    return dict(status='ineligible', extension_sha256='a'*64, dqsim_version='0.1.2')
                t = {'statevector': .001, 'mps': .002, 'pblock': .003}[request['backend']]
                return dict(status='ok', median_seconds=t, samples_seconds=[t, t],
                            extension_sha256='a'*64, dqsim_version='0.1.2')
            output = root/'runs'
            with patch.object(collector, 'run_pair', side_effect=fake_worker), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(collector.main(['--circuits', str(source), '--output', str(output),
                    '--repeats', '2', '--warmups', '0']), 0)
            run = next(p for p in output.iterdir() if (p/'results.jsonl').exists())
            records = load_dataset(run)['records']
            self.assertEqual(len(records), 4)
            self.assertEqual(len({r['family'] for r in records}), 4)
            raw = (run/'results.jsonl').read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(train(['--run', str(run), '--output', str(root/'model'), '--folds', '2']), 0)
            model = json.loads((root/'model/selector.json').read_text())
            self.assertEqual(model['model_type'], 'guarded_winner_classifier')
            self.assertEqual(model['training']['training_data_sha256'],
                hashlib.sha256((root/'model/training-data.jsonl').read_bytes()).hexdigest())
            self.assertEqual((run/'results.jsonl').read_bytes(), raw)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(predict(['--model', str(root/'model/selector.json'),
                    '--circuit', str(source/'family0/manifest.qasm')]), 0)
            result = json.loads(stdout.getvalue())
            self.assertEqual(result['selected_backend'], 'statevector')
            self.assertNotIn('predicted_seconds', result)
            with patch.object(collector, 'run_pair', side_effect=AssertionError('Completed pairs reran')), \
                    patch.object(collector, 'current_native_hash', return_value='a'*64), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(collector.main(['--resume', str(run)]), 0)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train(['--run', str(run), '--output', str(root/'model')])

    def test_collector_rejects_missing_inputs(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            collector.main([])
