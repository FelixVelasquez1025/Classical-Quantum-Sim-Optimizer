import contextlib
import copy
import io
import json
import unittest
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from test_ml_selector import records, SETTINGS, features
from ml.classification import fit_classifier, predict_classifier, winner
from ml.selector import vector


def examples(classes=3):
    data = records()
    names = ['statevector', 'mps', 'pblock'][:classes]
    for i,r in enumerate(data):
        r['rows'][names[i%classes]]['median_seconds'] = 0.00001
    return data


class ClassifierTests(unittest.TestCase):
    def test_json_scores_match_sklearn_binary_and_multiclass(self):
        for classes in (2, 3):
            data = examples(classes)
            params = dict(n_estimators=5, max_depth=2, learning_rate=.1)
            model = json.loads(json.dumps(fit_classifier(data, SETTINGS, params, 71)))
            reference = GradientBoostingClassifier(**params, random_state=71)
            reference.fit([vector(r['features'],r['task'],SETTINGS) for r in data], [winner(r) for r in data])
            query = features(5)
            result = predict_classifier(model, query, 'shots')
            np.testing.assert_allclose(list(result['class_scores'].values()),
                reference.predict_proba([vector(query,'shots',SETTINGS)])[0], atol=1e-12)
            self.assertNotIn('predicted_seconds', result)

    def test_constant_and_unsupported_class_abstains(self):
        model=fit_classifier(records(),SETTINGS,{},71)
        self.assertEqual(predict_classifier(model,features(),'shots')['selected_backend'],'stabilizer')
        self.assertIsNone(predict_classifier(model,features(clifford=False),'shots')['selected_backend'])
        self.assertIsNone(predict_classifier(model,features(),'evolve')['selected_backend'])

    def test_unresolved_circuits_have_no_label(self):
        r=examples()[0];r['rows']['mps']={'status':'timeout'}
        self.assertIsNone(winner(r))
        r['rows']['mps']={'status':'memory_error'}
        self.assertEqual(winner(r),'statevector')

if __name__ == '__main__':unittest.main()
