"""Group isolation and honest scoring for the single public selector."""
import copy
import unittest
from unittest.mock import patch

from ml.evaluation import evaluate, _metrics, _outcome, _resolved
from test_ml_classification import examples
from test_ml_selector import SETTINGS
from ml.structural_features import enrich_features
from test_collection_features import circuit


class EvaluationTests(unittest.TestCase):
    def test_group_holdouts_and_unchanged_records(self):
        data = examples()
        for r in data:
            r['features'] = enrich_features(circuit(r['features']['num_qubits'], []), r['features'])
        data[1]['group'] = data[0]['group']
        before = copy.deepcopy(data)
        seen = []
        def fit(train, **kwargs):
            seen.append({r['path'] for r in train})
            return {}
        with patch('ml.evaluation.fit_selector', side_effect=fit), patch('ml.evaluation.predict_selector',
                return_value=dict(selected_backend='statevector')):
            report = evaluate(data, settings=SETTINGS, folds=3)
        self.assertEqual(data, before)
        for fold, training in zip(report['folds'], seen):
            self.assertFalse(set(fold['train_groups']) & set(fold['test_groups']))
            tests = {o['path'] for o in report['metrics']['outcomes'] if o['fold'] == fold['fold']}
            self.assertFalse(training & tests)
        self.assertEqual(len(report['metrics']['outcomes']), len(data))

    def test_failures_abstentions_and_unresolved_cases_are_visible(self):
        data = examples()[:3]
        data[0]['rows']['mps'] = dict(status='memory_error')
        data[2]['rows']['mps'] = dict(status='timeout')
        outcomes = [_outcome(r, dict(selected_backend=selected), 0., 1)
                    for r, selected in zip(data, ('mps', None, 'mps'))]
        m = _metrics(outcomes)
        self.assertEqual(m['resolved_circuit_count'], 2)
        self.assertEqual(m['resolved_failure_count'], 1)
        self.assertEqual(m['resolved_abstention_count'], 1)
        self.assertEqual(m['failure_count'], 2)
        self.assertEqual(m['accuracy'], 0)
        self.assertEqual(m['slowdown_count'], 0)

    def test_unresolved_labels_and_exact_ties(self):
        for status in ('timeout', 'crash', 'error'):
            r = examples()[0]; r['rows']['mps'] = dict(status=status)
            self.assertFalse(_resolved(r))
        r = examples()[0]; r['rows'].pop('mps'); self.assertFalse(_resolved(r))
        r = examples()[0]; r['rows']['mps']['median_seconds'] = r['rows']['statevector']['median_seconds']
        self.assertTrue(_outcome(r, dict(selected_backend='mps'), 0., 1)['correct'])

    def test_invalid_fold_count(self):
        for folds in (0, 1, True):
            with self.assertRaises(ValueError): evaluate(examples(), settings=SETTINGS, folds=folds)
        with self.assertRaises(ValueError): evaluate(examples()[:1], settings=SETTINGS)
