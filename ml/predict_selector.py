"""Select a simulator for OpenQASM 2 or normalized JSON, without executing it."""
import argparse
import json
from pathlib import Path
import platform
import time

from ml import circuit as circuit_tools
from ml.selector import predict_selector
from ml.structural_features import enrich_features

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / 'models' / 'selector.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL, help='Defaults to the bundled selector')
    parser.add_argument('--circuit', type=Path, required=True, help='OpenQASM 2 or normalized artifact JSON')
    parser.add_argument('--task', choices=('auto', 'shots', 'evolve'))
    args = parser.parse_args(argv)
    start = time.perf_counter()
    try:
        model = json.loads(args.model.read_text())
        config = model['training']['run_config']
        if args.circuit.suffix.lower() == '.qasm':
            from dqsim import load_qasm
            imported = load_qasm(args.circuit)
            data, metadata = imported.data, imported.metadata
        else:
            artifact = json.loads(args.circuit.read_text())
            data, metadata = artifact['circuit'], artifact['metadata']
        if metadata.get('normalization') != config['normalization']:
            raise ValueError('Circuit normalization does not match the model')
        base = circuit_tools.extract_features(data, circuit_tools.gate_traits_classifier())
        features = enrich_features(data, base)
        task = circuit_tools.task_for(data, args.task or model['settings']['task'])
        result = predict_selector(model, features, task)
        machine = config.get('provenance', {})
        if machine.get('machine') != platform.machine() or machine.get('processor') != platform.processor():
            result['warnings'].append('Machine differs from training provenance; predictions may not transfer')
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    result.update(task=task, workload_settings=model['settings'], feature_version=2,
                  selection_seconds=time.perf_counter() - start,
                  selection_timing_scope='model loading, parsing, static features and prediction; excludes interpreter startup',
                  prediction_scope='Direct simulator classification under the recorded resource settings')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['selected_backend'] is not None else 2


if __name__ == '__main__':
    raise SystemExit(main())
