"""Memory labels, safe filtering, strict group separation and portable inference."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.classification import fit_classifier
from ml.memory_feasibility import (filter_prediction, fit_memory_models, memory_label,
                                   predict_guarded_classifier, predict_memory_scores)
from ml.structural_features import enrich_features, feature_vector
from test_collection_features import circuit
from test_ml_classification import examples
from test_ml_selector import SETTINGS, features


def prediction(selected='mps'):
    return dict(selected_backend=selected, class_scores=dict(mps=.7, pblock=.2, statevector=.1),
                excluded_backends={}, warnings=[])


def enriched_examples():
    data = examples()
    for r in data:
        r['features'] = enrich_features(circuit(r['features']['num_qubits'], []), r['features'])
    return data


class MemoryTests(unittest.TestCase):
    def test_only_observed_memory_failures_are_positive(self):
        self.assertEqual(memory_label(dict(status='ok')), 0)
        for status in ('memory_error', 'resource_exceeded'):
            self.assertEqual(memory_label(dict(status=status)), 1)
        for status in ('timeout', 'ineligible', 'crash', 'error', None):
            self.assertIsNone(memory_label(dict(status=status)))
        self.assertIsNone(memory_label({}))

    def test_export_scores_match_sklearn(self):
        data = enriched_examples()
        for i, record in enumerate(data):
            record['rows']['mps']['status'] = 'memory_error' if i % 2 else 'ok'
        data[0]['rows']['pblock']['status'] = 'timeout'
        for feature_set in ('baseline', 'combined'):
            params = dict(n_estimators=7, max_depth=3, class_weight='balanced')
            guard = json.loads(json.dumps(fit_memory_models(data, SETTINGS, feature_set, params)))
            x = [feature_vector(r['features'], r['task'], SETTINGS, feature_set) for r in data]
            model = RandomForestClassifier(**params, random_state=71, n_jobs=1).fit(x, [i % 2 for i in range(len(data))])
            for i, r in enumerate(data):
                result = predict_memory_scores(guard, r['features'], r['task'])
                self.assertAlmostEqual(result['mps'], model.predict_proba([x[i]])[0, 1], places=12)
                self.assertEqual(result['pblock'], 0.)
            self.assertEqual(guard['models']['pblock']['training_examples'], len(data)-1)
            bad = copy.deepcopy(guard)
            bad['input_names'].reverse()
            with self.assertRaises(ValueError): predict_memory_scores(bad, data[0]['features'], 'shots')

    def test_constant_missing_and_unknown_task(self):
        data = enriched_examples()
        for r in data:
            r['rows']['mps']['status'] = 'memory_error'
            r['rows']['pblock']['status'] = 'timeout'
        guard = fit_memory_models(data, SETTINGS, 'combined', {})
        self.assertEqual(predict_memory_scores(guard, data[0]['features'], 'shots'), dict(mps=1., pblock=None))
        self.assertEqual(predict_memory_scores(guard, data[0]['features'], 'evolve'), dict(mps=None, pblock=None))

    def test_filter_preserves_scores_eligibility_and_abstention(self):
        original = prediction()
        snapshot = copy.deepcopy(original)
        result = filter_prediction(original, dict(mps=.9, pblock=.1), dict(mps=.8, pblock=None))
        self.assertEqual(result['selected_backend'], 'pblock')
        self.assertEqual(result['class_scores'], original['class_scores'])
        self.assertEqual(original, snapshot)
        original['excluded_backends'] = dict(pblock='Unsupported', statevector='Too large')
        result = filter_prediction(original, dict(mps=.9, pblock=.1), dict(mps=.8, pblock=.8))
        self.assertIsNone(result['selected_backend'])
        self.assertTrue(any('abstaining' in w for w in result['warnings']))
        original['selected_backend'] = None
        result = filter_prediction(original, dict(mps=0., pblock=0.), dict(mps=None, pblock=None))
        self.assertIsNone(result['selected_backend'])
        with self.assertRaises(ValueError):
            filter_prediction(prediction(), dict(mps=.8, pblock=.2), dict(mps=float('nan'), pblock=None))





    def test_guarded_cli_and_original_winner_identity(self):
        from ml.circuit import collector_module
        from ml.predict_selector import main
        data = circuit(2, [dict(kind='h', qubit=0), dict(kind='measure', qubit=0, cbit=0)], cbits=1)
        collector = collector_module()
        base = collector.extract_features(data, collector.gate_traits_classifier())
        record = examples()[0]
        record['features'] = enrich_features(data, base)
        original = fit_classifier([record], SETTINGS, {})
        original_copy = copy.deepcopy(original)
        guard = fit_memory_models([record], SETTINGS, 'combined', {})
        bundle = dict(model_type='guarded_winner_classifier', model_version=1, feature_version=2,
            settings=SETTINGS, winner_classifier=original, memory_filter=guard,
            thresholds=dict(mps=None, pblock=None),
            training=dict(experimental=True, run_config=dict(normalization='common-one-two-qubit-v1')))
        result = predict_guarded_classifier(bundle, record['features'], 'shots')
        self.assertEqual(result['selected_backend'], 'statevector')
        self.assertEqual(original, original_copy)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'model.json').write_text(json.dumps(bundle))
            (root/'circuit.json').write_text(json.dumps(dict(circuit=data,
                metadata=dict(normalization='common-one-two-qubit-v1'))))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(['--model', str(root/'model.json'), '--circuit', str(root/'circuit.json')]), 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result['feature_version'], 2)
            self.assertIn('memory_failure_scores', result)
            self.assertNotIn('predicted_seconds', result)
        bundle['settings'] = dict(SETTINGS, shots=999)
        with self.assertRaises(ValueError): predict_guarded_classifier(bundle, record['features'], 'shots')


if __name__ == '__main__':
    unittest.main()
