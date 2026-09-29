"""Train the public selector from compatible benchmark collection runs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

from ml.circuit import load_artifact
from ml.data import load_datasets
from ml.evaluation import evaluate
from ml.selector import fit_selector
from ml.structural_features import enrich_features


def prepare_records(records, config):
    """Enrich static inputs from verified artifacts; never evolve a circuit."""
    sources = {c['run_id']: c for c in config.get('source_configs', [config])}
    result = []
    for record in records:
        source = sources[record.get('source_run_id', config['run_id'])]
        circuit, _ = load_artifact(Path(source['imports']), record['metadata'])
        result.append(dict(record, features=enrich_features(circuit, record['features'])))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True,
                        help='Benchmark run directory; repeat for compatible datasets')
    parser.add_argument('--output', type=Path, help='New model directory')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=71)
    args = parser.parse_args(argv)
    if args.folds < 2 or not 0 <= args.seed < 2**32:
        parser.error('Use at least two folds and a nonnegative 32-bit seed')
    if args.output is not None and args.output.exists():
        parser.error('Output must be a new directory')
    try:
        dataset = load_datasets(args.run)
        config = dataset['config']
        records = prepare_records(dataset['records'], config)
        report = evaluate(records, settings=config['settings'], folds=args.folds, seed=args.seed)
        model = fit_selector(records, settings=config['settings'], seed=args.seed)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    raw = ''.join(json.dumps(r, sort_keys=True, allow_nan=False) + '\n' for r in records)
    machine = config.get('provenance', {})
    model['training'] = dict(
        created_at=datetime.now(timezone.utc).isoformat(), seed=args.seed,
        training_data_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        run_config=dict(normalization=config['normalization'], settings=config['settings'],
                        provenance={k: machine[k] for k in ('machine', 'processor', 'cpu_count') if k in machine}),
        feature_scope='Static circuit properties only; no family names, outcomes or timings as inputs',
        limitations=['Uncalibrated scores', 'Memory filtering can exclude successful backends',
                     'Predictions apply to the recorded hardware and workload settings'])
    report['snapshot'] = dataset['snapshot']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    destination = args.output or Path('data/models') / stamp
    destination.mkdir(parents=True, exist_ok=False)
    for filename, value in [('selector.json', model), ('report.json', report), ('run.json', config)]:
        (destination / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    (destination / 'training-data.jsonl').write_text(raw)
    print(f'Model: {destination / "selector.json"}')
    print(f'Evaluation: {destination / "report.json"}')
    print(json.dumps({k: report['metrics'][k] for k in ('accuracy', 'resolved_failure_count', 'severe_10x_count', 'coverage')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
