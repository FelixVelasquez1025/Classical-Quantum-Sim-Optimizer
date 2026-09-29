"""Boosted winner classifier used by the public selector."""
import math
import numpy as np

from ml.selector import eligible_backends, export_tree, tree_value
from ml.structural_features import base_features, feature_vector, input_names
from ml.evaluation import _resolved

def winner(record):
    if not _resolved(record):
        return None
    return min((b for b, r in record['rows'].items() if r['status'] == 'ok'),
               key=lambda b: (record['rows'][b]['median_seconds'], b))


def export_probability_tree(estimator):
    tree = export_tree(estimator)
    values = estimator.tree_.value[:, 0, :]
    tree['value'] = (values / values.sum(axis=1, keepdims=True)).tolist()
    return tree


def fit_classifier(records, settings, params=None, seed=71):
    from ml.selector import WINNER_PARAMETERS
    params = dict(WINNER_PARAMETERS if params is None else params)
    feature_set, kind = 'baseline', 'boosted'
    from sklearn.ensemble import GradientBoostingClassifier
    labeled = [r for r in records if winner(r) is not None]
    if not labeled:
        raise ValueError('No resolved winner labels in training fold')
    names = input_names(feature_set)
    x = [feature_vector(r['features'], r['task'], settings, feature_set) for r in labeled]
    y = [winner(r) for r in labeled]
    classes = sorted(set(y))
    bundle = dict(model_version=2 if feature_set == 'baseline' else 3,
                  model_type='winner_classifier', kind=kind, feature_set=feature_set,
                  feature_version=1 if feature_set == 'baseline' else 2,
                  input_names=list(names), settings=dict(settings), classes=classes,
                  tasks=sorted({r['task'] for r in labeled}), parameters=params,
                  training_circuits=len(labeled), feature_min=np.min(x, axis=0).tolist(),
                  feature_max=np.max(x, axis=0).tolist())
    if len(classes) == 1:
        bundle['constant'] = True
        return bundle
    model = GradientBoostingClassifier(**params, random_state=seed)
    model.fit(x, y)
    bundle['classes'] = model.classes_.tolist()
    bundle['learning_rate'] = model.learning_rate
    bundle['initial_logits'] = model._raw_predict_init(np.asarray(x[:1], dtype=np.float32))[0].tolist()
    bundle['stages'] = [[export_tree(t) for t in stage] for stage in model.estimators_]
    return bundle


def predict_classifier(bundle, features, task):
    feature_set = bundle.get('feature_set', 'baseline')
    names = input_names(feature_set)
    expected_version = 2
    if feature_set != 'baseline' or bundle.get('kind') != 'boosted':
        raise ValueError('Expected the selected boosted winner classifier')
    if (bundle.get('model_version') != expected_version
            or bundle.get('input_names') != list(names)
            or bundle.get('feature_version', 1) != (1 if feature_set == 'baseline' else 2)):
        raise ValueError('Incompatible classifier schema')
    x = np.asarray(feature_vector(features, task, bundle['settings'], feature_set), dtype=np.float32).tolist()
    allowed, excluded = eligible_backends(base_features(features), task, bundle['settings'])
    classes = bundle['classes']
    if bundle.get('constant'):
        scores = [1.]
    else:
        logits = np.asarray(bundle['initial_logits'], dtype=float)
        for stage in bundle['stages']:
            logits += bundle['learning_rate'] * np.asarray([tree_value(t, x) for t in stage])
        if len(classes) == 2:
            z = float(logits[0])
            p = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
            scores = [1-p, p]
        else:
            exp = np.exp(logits - np.max(logits))
            scores = (exp / exp.sum()).tolist()
    warnings = []
    if task not in bundle['tasks']:
        allowed = []
        warnings.append('No training labels for this task; abstaining')
    candidates = {b: score for b, score in zip(classes, scores) if b in allowed and score > 0}
    for b in allowed:
        if b not in classes:
            excluded[b] = 'No winner examples for this backend in training'
    outside = [name for name, v, lo, hi in zip(names, x, bundle['feature_min'], bundle['feature_max'])
               if v < lo-1e-6*max(1,abs(lo)) or v > hi+1e-6*max(1,abs(hi))]
    if outside:
        warnings.append('Features outside training ranges; prediction may be unreliable')
    return dict(selected_backend=max(candidates, key=candidates.get) if candidates else None,
                class_scores=dict(zip(classes, scores)), excluded_backends=excluded,
                warnings=warnings, outside_training_ranges=outside,
                score_interpretation='Uncalibrated class scores, not runtime estimates or success probabilities')
