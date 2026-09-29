"""Small synthetic ML fixtures; no quantum simulators execute."""
import json
import math
import unittest

import numpy as np


from ml.data import FEATURE_NAMES, BOOLEAN_FEATURES
from ml.selector import (fit_selector, predict_selector, eligible_backends, vector)
from ml.structural_features import enrich_features
from test_collection_features import circuit


SETTINGS = dict(task='auto', shots=1000, repeats=5, warmups=1,
                threads=4, parallel_shots=1, max_memory_mb=1024, seed=71)


def features(n=4, clifford=True):
    out = {k: (False if k in BOOLEAN_FEATURES else 0) for k in FEATURE_NAMES}
    out.update(feature_version=1, num_qubits=n, num_cbits=n, has_measurements=True,
               stabilizer_eligible=clifford, measurement_count=n)
    return out


def records():
    out = []
    for i in range(8):
        rows = {b: dict(status='ok', median_seconds=t * (i+1)) for b, t in
                [('statevector', .01), ('mps', .02), ('pblock', .03), ('stabilizer', .005)]}
        out.append(dict(path=f'c{i}', family=f'f{i}', group=f'g{i}',
                        normalized_sha256=f'{i:064x}', features=features(i+2), task='shots', rows=rows))
    return out



class SelectorTests(unittest.TestCase):
    def test_fit_export_predict_and_task_coverage(self):
        data = records()
        for r in data:
            r['features'] = enrich_features(circuit(r['features']['num_qubits'], []), r['features'])
        model = json.loads(json.dumps(fit_selector(data, settings=SETTINGS)))
        result = predict_selector(model, data[0]['features'], 'shots')
        self.assertEqual(result['selected_backend'], 'stabilizer')
        self.assertNotIn('predicted_seconds', result)
        self.assertIn('memory_failure_scores', result)
        self.assertIsNone(predict_selector(model, data[0]['features'], 'evolve')['selected_backend'])
        self.assertEqual(model['thresholds'], dict(mps=None, pblock=.9))

    def test_static_eligibility_and_feature_validation(self):
        allowed, excluded = eligible_backends(features(433, False), 'shots', SETTINGS)
        self.assertEqual(allowed, ['mps', 'pblock'])
        f = features(); f['has_measurements'] = False
        self.assertEqual(eligible_backends(f, 'shots', SETTINGS)[0], [])
        f = features(); f['circuit_depth'] = float('nan')
        with self.assertRaises(ValueError): vector(f, 'shots', SETTINGS)


if __name__ == '__main__': unittest.main()
