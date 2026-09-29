"""The public selector: boosted winner classification with a memory filter."""
import math
from ml.circuit import collector_module
from ml.data import FEATURE_NAMES, validate_features

BACKENDS = ('statevector', 'mps', 'pblock', 'stabilizer')
INPUT_NAMES = (*FEATURE_NAMES, 'task_is_shots', 'log1p_shots')
WINNER_PARAMETERS = dict(n_estimators=80, learning_rate=.05, max_depth=2, min_samples_leaf=2)
MEMORY_PARAMETERS = dict(n_estimators=64, max_depth=4, min_samples_leaf=3,
                         max_features='sqrt', class_weight='balanced')
MEMORY_THRESHOLDS = dict(mps=None, pblock=.9)


def vector(features, task, settings):
    validate_features(features)
    if task not in ('shots', 'evolve'):
        raise ValueError('Task must be shots or evolve')
    values = [features[name] for name in FEATURE_NAMES]
    if any(not isinstance(x, (bool, int, float)) or not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('Features must contain finite nonnegative numbers or booleans')
    n = features['num_qubits']
    if isinstance(n, bool) or n != int(n):
        raise ValueError('num_qubits must be a nonnegative integer')
    return [float(x) for x in values] + [float(task == 'shots'),
            math.log1p(settings['shots']) if task == 'shots' else 0.0]


def eligible_backends(features, task, settings):
    vector(features, task, settings)
    excluded = {}
    if task == 'shots' and not features['has_measurements']:
        return [], {b: 'Shot sampling requires explicit measurements' for b in BACKENDS}
    if not features['stabilizer_eligible']:
        excluded['stabilizer'] = 'Circuit is not supported by the exact stabilizer compiler'
    preflight = collector_module().resource_preflight('statevector', int(features['num_qubits']), settings)
    if preflight is not None:
        excluded['statevector'] = preflight['reason']
    return [b for b in BACKENDS if b not in excluded], excluded


def export_tree(estimator):
    tree = estimator.tree_
    return dict(left=tree.children_left.tolist(), right=tree.children_right.tolist(),
                feature=tree.feature.tolist(), threshold=tree.threshold.tolist(),
                value=tree.value[:, 0, 0].tolist())


def tree_value(tree, x):
    node = 0
    for _ in range(len(tree['left'])):
        if tree['left'][node] == -1:
            return tree['value'][node]
        node = tree['left'][node] if float(x[tree['feature'][node]]) <= tree['threshold'][node] else tree['right'][node]
    raise ValueError('Invalid tree: traversal did not reach a leaf')


def fit_selector(records, *, settings, seed=71):
    """Fit the single selected architecture with fixed published parameters."""
    from ml.classification import fit_classifier
    from ml.memory_feasibility import fit_memory_models
    winner = fit_classifier(records, settings, WINNER_PARAMETERS, seed)
    guard = fit_memory_models(records, settings, 'combined', MEMORY_PARAMETERS, seed)
    return dict(model_type='guarded_winner_classifier', model_version=1,
                feature_version=2, settings=dict(settings), winner_classifier=winner,
                memory_filter=guard, thresholds=dict(MEMORY_THRESHOLDS))


def predict_selector(bundle, features, task):
    """Return one backend class (or abstention), never a runtime estimate."""
    from ml.memory_feasibility import predict_guarded_classifier
    return predict_guarded_classifier(bundle, features, task)
