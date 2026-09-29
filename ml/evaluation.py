"""Evaluate the selector on circuit families absent from their training fold.

Only static features reach the prediction functions. Held-out measurements are
used afterward to score the choice; they never provide a feasibility mask.
"""
from __future__ import annotations

import math
import statistics
import time


from .selector import fit_selector, predict_selector


BACKENDS = ("statevector", "mps", "pblock", "stabilizer")
_RESOLVED = {"ok", "resource_exceeded", "memory_error"}
_FAILURES = {"ineligible", "resource_exceeded", "memory_error", "timeout", "crash", "error"}


def _resolved(record):
    rows = record["rows"]
    if set(rows) != set(BACKENDS):
        return False
    return any(row["status"] == "ok" for row in rows.values()) and all(
        rows[backend]["status"] in _RESOLVED
        or (backend == "stabilizer" and rows[backend]["status"] == "ineligible")
        for backend in BACKENDS
    )


def _quantile(values, fraction):
    """Linearly interpolated quantile, including a single-observation sample."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _outcome(record, prediction, latency, fold):
    selected = prediction["selected_backend"]
    row = record["rows"].get(selected) if selected is not None else None
    status = row["status"] if row else ("unobserved" if selected is not None else None)
    resolved = _resolved(record)
    best = min(
        (value["median_seconds"] for value in record["rows"].values()
         if value["status"] == "ok"),
        default=None,
    ) if resolved else None
    selected_seconds = row["median_seconds"] if status == "ok" else None
    slowdown = selected_seconds / best if resolved and status == "ok" else None
    return {
        "path": record["path"],
        "family": record["family"],
        "group": record["group"],
        "normalized_sha256": record["normalized_sha256"],
        "task": record["task"],
        "fold": fold,
        "comparison_resolved": resolved,
        "selected_backend": selected,
        "selected_status": status,
        "selected_seconds": selected_seconds,
        "selection_failed": selected is not None and status in _FAILURES,
        "selection_unobserved": selected is not None and status not in _FAILURES | {"ok"},
        "abstained": selected is None,
        "fastest_seconds": best,
        "correct": (status == "ok" and math.isclose(selected_seconds, best,
                                                       rel_tol=1e-12, abs_tol=0.0))
        if resolved else None,
        "slowdown": slowdown,
        "excluded_backends": prediction.get("excluded_backends", {}),
        "warnings": prediction.get("warnings", []),
        "prediction_seconds": latency,
        "observed_statuses": {name: value["status"]
                              for name, value in record["rows"].items()},
        "observed_seconds": {name: value["median_seconds"]
                             for name, value in record["rows"].items()
                             if value["status"] == "ok"},
    }


def _metrics(outcomes):
    total = len(outcomes)
    resolved = [item for item in outcomes if item["comparison_resolved"]]
    slowdowns = [item["slowdown"] for item in outcomes if item["slowdown"] is not None]
    latencies = [item["prediction_seconds"] for item in outcomes]
    selected = sum(not item["abstained"] for item in outcomes)
    return {
        "circuit_count": total,
        "resolved_circuit_count": len(resolved),
        "censored_circuit_count": total - len(resolved),
        "selected_count": selected,
        "coverage": selected / total if total else None,
        "abstention_count": total - selected,
        "failure_count": sum(item["selection_failed"] for item in outcomes),
        "unobserved_selection_count": sum(item["selection_unobserved"] for item in outcomes),
        "resolved_failure_count": sum(item["selection_failed"] for item in resolved),
        "resolved_abstention_count": sum(item["abstained"] for item in resolved),
        "accuracy": sum(item["correct"] for item in resolved) / len(resolved)
        if resolved else None,
        "slowdown_count": len(slowdowns),
        "success_conditional_slowdown": {
            "median": statistics.median(slowdowns) if slowdowns else None,
            "geomean": math.exp(statistics.mean(math.log(value) for value in slowdowns))
            if slowdowns else None,
            "p95": _quantile(slowdowns, 0.95) if slowdowns else None,
            "max": max(slowdowns) if slowdowns else None,
        },
        "prediction_latency_seconds": {
            "median": statistics.median(latencies) if latencies else None,
            "p95": _quantile(latencies, 0.95) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "outcomes": outcomes,
    }


def evaluate(records, *, settings, folds=5, seed=71):
    """Evaluate the fixed public pipeline with disjoint family groups."""
    from sklearn.model_selection import GroupKFold
    if type(folds) is not int or folds < 2:
        raise ValueError('Use at least two folds')
    groups = [r['group'] for r in records]
    if len(set(groups)) < 2:
        raise ValueError('Evaluation needs at least two independent circuit groups')
    splitter = GroupKFold(n_splits=min(folds, len(set(groups))))
    outcomes, audit = [], []
    for fold, (train, test) in enumerate(splitter.split(records, groups=groups), 1):
        model = fit_selector([records[i] for i in train], settings=settings, seed=seed)
        for i in test:
            record = records[i]
            start = time.perf_counter()
            pred = predict_selector(model, record['features'], record['task'])
            outcomes.append(_outcome(record, pred, time.perf_counter()-start, fold))
        audit.append(dict(fold=fold, train_groups=sorted({groups[i] for i in train}),
                          test_groups=sorted({groups[i] for i in test})))
    metrics = _metrics(outcomes)
    metrics['severe_10x_count'] = sum(o['slowdown'] is not None and o['slowdown'] >= 10 for o in outcomes)
    return dict(task='winner_classification', folds=audit, metrics=metrics,
                evaluation_policy='Fixed published parameters; related families held out together',
                limitation='Evaluation on previously used development families is not an independent final test')
