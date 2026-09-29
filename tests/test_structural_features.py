"""Static-analysis soundness, schema compatibility and grouped experiment checks."""
import contextlib
import copy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
from qiskit.quantum_info import Operator, Statevector
from sklearn.ensemble import GradientBoostingClassifier

from ml.classification import fit_classifier, predict_classifier, winner
from ml.structural_features import (BasisTracker, FEATURE_SETS, STRUCTURAL_NAMES,
    _matrix, base_features, enrich_features, extract_structure, feature_vector, gate_spec)
from test_collection_features import circuit
from test_ml_classification import examples
from test_ml_selector import SETTINGS, features


class StructuralTests(unittest.TestCase):
    def test_order_distinguishes_diagonal_reduction_and_coupling(self):
        h0, h1 = dict(kind='h', qubit=0), dict(kind='h', qubit=1)
        phase = dict(kind='cp', control=0, target=1, lam=.3)
        early = extract_structure(circuit(2, [h0, phase, h1]))
        late = extract_structure(circuit(2, [h0, h1, phase]))
        self.assertEqual(early['diagonal_reduced_count'], 1)
        self.assertEqual(early['potential_coupling_count'], 0)
        self.assertEqual(late['diagonal_reduced_count'], 0)
        self.assertEqual(late['potential_coupling_count'], 1)
        self.assertEqual(late['potential_largest_component_fraction'], 1)

    def test_known_controls_and_swaps(self):
        ops = [dict(kind='cx', control=0, target=1), dict(kind='x', qubit=0),
               dict(kind='cx', control=0, target=1), dict(kind='h', qubit=0),
               dict(kind='cx', control=0, target=1)]
        f = extract_structure(circuit(2, ops))
        for kind in ('inactive', 'active', 'unknown'):
            self.assertEqual(f[kind + '_control_count'], 1)
            self.assertAlmostEqual(f[kind + '_control_fraction'], 1/3)
        self.assertEqual(f['potential_coupling_count'], 1)
        ops = [dict(kind='h', qubit=0), dict(kind='cx', control=0, target=1),
               dict(kind='swap', a=1, b=2), dict(kind='cz', control=0, target=2)]
        f = extract_structure(circuit(4, ops))
        self.assertEqual(f['potential_coupling_count'], 2)
        self.assertEqual(f['potential_merge_count'], 1)
        self.assertEqual(f['potential_largest_component_fraction'], .5)

    def test_conditional_join_measure_reset_and_nested_false(self):
        tracker = BasisTracker(2, 1)
        tracker.apply(dict(kind='h', qubit=0))
        tracker.apply(dict(kind='measure', qubit=0, cbit=0))
        self.assertEqual(tracker.classical, [None])
        condition = dict(creg_base=0, creg_size=1, creg_value=1)
        conditional = dict(kind='conditional', condition=condition, op=dict(kind='x', qubit=1))
        tracker.apply(conditional)
        self.assertEqual(tracker.quantum, [None, None])
        tracker.apply(dict(kind='reset', qubit=0))
        tracker.apply(dict(kind='measure', qubit=0, cbit=0))
        self.assertEqual(tracker.classical, [0])
        tracker.apply(dict(kind='reset', qubit=1))
        tracker.apply(conditional)
        self.assertEqual(tracker.quantum, [0, 0])
        nested = dict(kind='conditional', condition=condition,
                      op=dict(kind='conditional', condition=dict(condition, creg_value=0),
                              op=dict(kind='h', qubit=1)))
        tracker.apply(nested)
        self.assertEqual(tracker.quantum, [0, 0])
        self.assertEqual(tracker.unknown_count, 0)

    def test_unknown_is_not_erased_by_cancellation_or_tiny_rotation(self):
        tracker = BasisTracker(1, 0)
        tracker.apply(dict(kind='h', qubit=0))
        tracker.apply(dict(kind='h', qubit=0))
        self.assertEqual(tracker.quantum, [None])
        tracker = BasisTracker(1, 0)
        tracker.apply(dict(kind='rx', qubit=0, theta=1e-15))
        self.assertEqual(tracker.quantum, [None])

    def test_basis_support_against_independent_small_statevectors(self):
        from dqsim.qasm import _SINGLE, _CONTROLLED, _PAIR
        rng = random.Random(71)
        for kind, fields in {**_SINGLE, **_CONTROLLED, **_PAIR}.items():
            for initial in range(4):
                tracker = BasisTracker(2, 0)
                state = Statevector.from_int(initial, 4)
                tracker.quantum[:] = [initial & 1, (initial >> 1) & 1]
                operands = (dict(qubit=1) if kind in _SINGLE else
                            dict(control=1, target=0) if kind in _CONTROLLED else dict(a=1, b=0))
                op = dict(kind=kind, **operands, **{field: rng.uniform(-2, 2) for field in fields})
                # Also cover unknown inputs, including entangled input states.
                for prefix in ([], [dict(kind='h', qubit=0), dict(kind='cx', control=0, target=1)]):
                    local = copy.deepcopy(tracker)
                    reference = state.copy()
                    for gate in prefix + [op]:
                        wires, values = gate_spec(gate)
                        local.apply(gate)
                        reference = reference.evolve(Operator(_matrix(gate['kind'], values)), qargs=list(wires))
                        for q, value in enumerate(local.quantum):
                            if value is not None:
                                wrong = sum(abs(amplitude)**2 for i, amplitude in enumerate(reference.data)
                                            if ((i >> q) & 1) != value)
                                self.assertLess(wrong, 1e-24, (kind, initial, q, value))

    def test_empty_identity_and_idle_circuits(self):
        self.assertEqual(len(STRUCTURAL_NAMES), 29)
        empty = extract_structure(circuit(0, []))
        self.assertEqual(set(empty.values()), {0})
        idle = extract_structure(circuit(4, []))
        self.assertEqual(idle['potential_component_count'], 4)
        self.assertEqual(idle['potential_largest_component_fraction'], .25)
        f = extract_structure(circuit(2, [dict(kind='h', qubit=0), dict(kind='h', qubit=1),
                                          dict(kind='cp', control=0, target=1, lam=0)]))
        self.assertEqual(f['potential_coupling_count'], 0)

    def test_feature_vector_schema_and_input_validation(self):
        base = features(2)
        full = enrich_features(circuit(2, [dict(kind='h', qubit=0)]), base)
        self.assertEqual(base_features(full), base)
        self.assertEqual(len(feature_vector(full, 'shots', SETTINGS, 'combined')), 59)
        with self.assertRaisesRegex(ValueError, 'version 2'):
            feature_vector(base, 'shots', SETTINGS, 'combined')
        for changes in (dict(inactive_control_count=-1), dict(unknown_control_fraction=float('nan'))):
            with self.assertRaises(ValueError): base_features(dict(full, **changes))

    def test_legacy_classifier_still_predicts(self):
        model = fit_classifier(examples(), SETTINGS, dict(n_estimators=2), 71)
        model.pop('feature_set')
        model.pop('feature_version')
        self.assertIsNotNone(predict_classifier(model, features(), 'shots')['selected_backend'])

    def test_cli_extracts_version_two_without_simulating(self):
        from ml.circuit import collector_module
        from ml.predict_selector import main
        data = circuit(2, [dict(kind='h', qubit=0), dict(kind='cx', control=0, target=1),
                           dict(kind='measure', qubit=0, cbit=0)], cbits=1)
        collector = collector_module()
        base = collector.extract_features(data, collector.gate_traits_classifier())
        record = examples()[0]
        record['features'] = enrich_features(data, base)
        from ml.selector import fit_selector
        model = fit_selector([record], settings=SETTINGS)
        model['training'] = dict(experimental=True, run_config=dict(normalization='common-one-two-qubit-v1'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'model.json').write_text(json.dumps(model))
            (root/'circuit.json').write_text(json.dumps(dict(circuit=data,
                metadata=dict(normalization='common-one-two-qubit-v1'))))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(['--model', str(root/'model.json'), '--circuit', str(root/'circuit.json')]), 0)
            prediction = json.loads(output.getvalue())
            self.assertEqual(prediction['feature_version'], 2)
            self.assertEqual(prediction['selected_backend'], 'statevector')
            self.assertNotIn('predicted_seconds', prediction)



if __name__ == '__main__':
    unittest.main()
