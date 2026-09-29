"""Static extraction and compiler checks only; no simulator evolution."""
import math
import unittest
from ml import circuit as collector


def circuit(n, ops, cbits=0):
    return dict(qregs={'q': dict(name='q', base=0, size=n)},
                cregs={'c': dict(name='c', base=0, size=cbits)}, instructions=ops)


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.traits = staticmethod(collector.gate_traits_classifier())

    def extract(self, n, ops, cbits=0):
        return collector.extract_features(circuit(n, ops, cbits), self.traits)

    def test_angles_and_crossed_categories(self):
        f = self.extract(3, [dict(kind='rz', qubit=0, phi=math.pi/2),
                             dict(kind='rz', qubit=1, phi=math.pi/4),
                             dict(kind='cx', control=0, target=1),
                             dict(kind='cp', control=1, target=2, lam=math.pi/2)])
        for arity in ('single', 'two'):
            for group in ('clifford', 'non_clifford'):
                self.assertEqual(f[f'{arity}_qubit_{group}_count'], 1)
                self.assertEqual(f[f'{arity}_qubit_{group}_fraction'], .25)
        self.assertEqual(f['diagonal_gate_fraction'], .75)
        self.assertFalse(f['stabilizer_eligible'])
        self.assertEqual(f['circuit_depth'], 3)

    def test_disjoint_pairs_and_repeated_long_interaction(self):
        f = self.extract(5, [dict(kind='cx', control=0, target=3),
                             dict(kind='cx', control=1, target=2),
                             dict(kind='cz', control=3, target=0)])
        self.assertEqual(f['largest_interaction_component'], 2)
        self.assertEqual(f['interaction_component_count'], 3)
        self.assertEqual(f['distinct_interacting_pairs'], 2)
        self.assertEqual(f['max_cut_crossings'], 3)
        self.assertEqual(f['mean_two_qubit_distance'], 7/3)
        self.assertEqual(f['max_two_qubit_distance'], 3)
        self.assertEqual(f['circuit_depth'], 2)

    def test_terminal_and_dynamic_dependencies(self):
        measure = dict(kind='measure', qubit=0, cbit=0)
        f = self.extract(2, [dict(kind='h', qubit=0), measure, dict(kind='barrier')], 1)
        self.assertTrue(f['terminal_only_measurements'])
        self.assertFalse(f['has_mid_circuit_measurements'])
        conditional = dict(kind='conditional', condition=dict(creg_base=0, creg_size=1, creg_value=1),
                           op=dict(kind='x', qubit=1))
        f = self.extract(2, [dict(kind='h', qubit=0), measure, conditional,
                             dict(kind='reset', qubit=1)], 1)
        self.assertEqual(f['circuit_depth'], 4)
        self.assertEqual(f['conditional_count'], 1)
        self.assertEqual(f['reset_count'], 1)
        self.assertEqual(f['measurement_count'], 1)
        self.assertEqual(f['total_gate_count'], 2)
        self.assertFalse(f['terminal_only_measurements'])
        self.assertTrue(f['has_mid_circuit_measurements'])

    def test_reset_before_terminal_measurement(self):
        f = self.extract(1, [dict(kind='reset', qubit=0),
                             dict(kind='measure', qubit=0, cbit=0)], 1)
        self.assertTrue(f['terminal_only_measurements'])
        self.assertFalse(f['terminal_sampling_candidate'])

    def test_empty_and_idle_qubits(self):
        f = self.extract(0, [])
        self.assertEqual(f['circuit_depth'], 0)
        self.assertEqual(f['largest_interaction_component'], 0)
        self.assertEqual(f['diagonal_gate_fraction'], 0)
        self.assertFalse(f['terminal_only_measurements'])
        f = self.extract(4, [dict(kind='x', qubit=2)])
        self.assertEqual(f['largest_interaction_component'], 1)
        self.assertEqual(f['interaction_component_count'], 4)

    def test_all_normalized_gate_families(self):
        from dqsim.qasm import _SINGLE, _CONTROLLED, _PAIR
        for kind, fields in {**_SINGLE, **_CONTROLLED, **_PAIR}.items():
            with self.subTest(kind=kind):
                op = dict(kind=kind, **{field: 0.37 for field in fields})
                clifford, diagonal = self.traits(op)
                self.assertIsInstance(clifford, bool)
                self.assertIsInstance(diagonal, bool)
        # A parameterized U gate can be diagonal despite its gate name.
        self.assertTrue(self.traits(dict(kind='u', theta=0, phi=.2, lam=.3))[1])


if __name__ == '__main__':
    unittest.main()
