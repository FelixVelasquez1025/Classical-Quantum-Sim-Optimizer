"""Memory-failure classification around a fixed winner classifier.

The filter learns only ok versus memory/resource failure, never seconds. Scores
are uncalibrated and thresholds must be selected with grouped validation. Only
MPS and P-block need learned filters; existing deterministic eligibility checks
continue to handle stabilizer support and dense statevector lower bounds.
"""
import copy
import math

import numpy as np

from ml.classification import export_probability_tree, predict_classifier
from ml.selector import tree_value
from ml.structural_features import feature_vector, input_names

BACKENDS = ('mps', 'pblock')
MEMORY_FAILURES = frozenset(('memory_error', 'resource_exceeded'))
def memory_label(row):
    status = row.get('status')
    return 0 if status == 'ok' else 1 if status in MEMORY_FAILURES else None


def fit_memory_models(records, settings, feature_set, params, seed=71):
    from sklearn.ensemble import RandomForestClassifier
    bundle = dict(schema_version=1, feature_set=feature_set,
                  input_names=list(input_names(feature_set)), settings=dict(settings),
                  parameters=dict(params), seed=seed, models={})
    for backend in BACKENDS:
        labeled = [r for r in records if memory_label(r['rows'].get(backend, {})) is not None]
        y = [memory_label(r['rows'][backend]) for r in labeled]
        model = dict(training_examples=len(labeled), memory_failures=sum(y),
                     success_groups=len({r['group'] for r in labeled if memory_label(r['rows'][backend]) == 0}),
                     failure_groups=len({r['group'] for r in labeled if memory_label(r['rows'][backend]) == 1}),
                     tasks=sorted({r['task'] for r in labeled}))
        if not labeled:
            model['unavailable'] = True
        elif len(set(y)) == 1:
            model['constant_failure_score'] = float(y[0])
        else:
            x = [feature_vector(r['features'], r['task'], settings, feature_set) for r in labeled]
            forest = RandomForestClassifier(**params, random_state=seed, n_jobs=1).fit(x, y)
            model.update(classes=forest.classes_.tolist(),
                         trees=[export_probability_tree(t) for t in forest.estimators_])
        bundle['models'][backend] = model
    return bundle


def predict_memory_scores(bundle, features, task):
    feature_set = bundle.get('feature_set')
    if bundle.get('schema_version') != 1 or bundle.get('input_names') != list(input_names(feature_set)):
        raise ValueError('Incompatible memory-feasibility schema')
    x = np.asarray(feature_vector(features, task, bundle['settings'], feature_set), dtype=np.float32).tolist()
    scores = {}
    for backend in BACKENDS:
        model = bundle['models'][backend]
        if model.get('unavailable') or task not in model['tasks']:
            scores[backend] = None
        elif 'constant_failure_score' in model:
            scores[backend] = model['constant_failure_score']
        else:
            index = model['classes'].index(1)
            scores[backend] = float(np.mean([tree_value(tree, x)[index] for tree in model['trees']]))
    return scores


def rejected_backends(scores, thresholds):
    if set(thresholds) != set(BACKENDS) or set(scores) != set(BACKENDS):
        raise ValueError('Memory scores and thresholds must specify MPS and P-block')
    rejected = set()
    for backend in BACKENDS:
        threshold, score = thresholds[backend], scores[backend]
        if threshold is not None and (type(threshold) not in (int, float)
                or not math.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError('Memory threshold must be null or between zero and one')
        if score is not None and (type(score) not in (int, float)
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError('Memory score must be null or between zero and one')
        if threshold is not None and score is not None and score >= threshold:
            rejected.add(backend)
    return rejected


def filter_prediction(prediction, scores, thresholds):
    """Keep original class scores/ranking, excluding only predicted failures.

    Original abstention is preserved (e.g. an unseen task). If all candidates
    are filtered out, abstain; never silently reinstate a rejected backend.
    """
    rejected = rejected_backends(scores, thresholds)
    result = copy.deepcopy(prediction)
    result.update(original_selected_backend=prediction['selected_backend'],
                  memory_failure_scores=dict(scores), memory_thresholds=dict(thresholds),
                  memory_rejected_backends=sorted(rejected),
                  memory_score_interpretation='Uncalibrated memory-failure scores, not success guarantees or runtime estimates')
    if prediction['selected_backend'] is None:
        return result
    for backend in rejected:
        if backend in result['class_scores'] and backend not in result['excluded_backends']:
            result['excluded_backends'][backend] = 'Excluded by learned memory-feasibility filter'
    candidates = {b: score for b, score in result['class_scores'].items()
                  if b not in result['excluded_backends'] and score > 0}
    result['selected_backend'] = max(candidates, key=candidates.get) if candidates else None
    if not candidates:
        result['warnings'].append('Memory filter excluded all eligible ranked candidates; abstaining')
    return result


def predict_guarded_classifier(bundle, features, task):
    winner = bundle['winner_classifier']
    guard = bundle['memory_filter']
    if (bundle.get('model_type') != 'guarded_winner_classifier' or bundle.get('model_version') != 1
            or bundle['settings'] != winner['settings'] or bundle['settings'] != guard['settings']
            or bundle.get('feature_version') != (1 if guard['feature_set'] == 'baseline' else 2)):
        raise ValueError('Incompatible guarded classifier schema/settings')
    prediction = predict_classifier(winner, features, task)
    scores = predict_memory_scores(guard, features, task)
    return filter_prediction(prediction, scores, bundle['thresholds'])
