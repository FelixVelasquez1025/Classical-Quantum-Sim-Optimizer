#!/usr/bin/env python3
"""Collect isolated, repeated simulator timings from an imported-circuit manifest.

No work runs on import. Each circuit/backend pair has a separate subprocess and
optional wall-clock timeout; pairs run sequentially to avoid CPU contention.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import statistics
import struct
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ('statevector', 'mps', 'pblock', 'stabilizer')


# The same static definitions serve collection and model inference.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ml.circuit import (FEATURE_VERSION, gate_traits_classifier, extract_features,
                        any_measurement_followed_by_work, digest, load_artifact,
                        has_measurements, task_for, resource_preflight)


def options_for(backend, settings):
    options = dict(seed=settings['seed'], max_memory_mb=settings['max_memory_mb'],
                   max_parallel_shots=settings['parallel_shots'], sample_terminal=True)
    if backend == 'mps':
        options.update(max_bond_dimension=None, truncation_threshold=0.0, max_discarded_weight=0.0)
    elif backend == 'pblock':
        options.update(split_separable=False)
    elif backend == 'stabilizer':
        options.update(clifford_tolerance=0.0)
    return options


def measure(simulator, circuit, task, settings):
    """Time the public API only; loading, construction and checks are untimed."""
    samples = []
    for iteration in range(settings['warmups'] + settings['repeats']):
        start = time.perf_counter_ns()
        if task == 'shots':
            result = simulator.simulate_shots(circuit, shots=settings['shots'])
        else:
            result = simulator.simulate(circuit)
        elapsed = (time.perf_counter_ns() - start) / 1e9
        if task == 'shots' and sum(result.values()) != settings['shots']:
            raise RuntimeError('Returned shot count does not match the requested count')
        if task == 'evolve' and result.num_qubits != circuit.num_qubits:
            raise RuntimeError('Returned quantum width does not match the input')
        del result  # Do not retain quantum states or counts across repetitions.
        if iteration >= settings['warmups']:
            samples.append(elapsed)
    return dict(samples_seconds=samples, median_seconds=statistics.median(samples),
                min_seconds=min(samples), max_seconds=max(samples),
                stdev_seconds=statistics.stdev(samples) if len(samples) > 1 else 0.0)


def worker(request):
    # Imported only in workers, after the parent has set Rayon/BLAS limits.
    import dqsim
    data, metadata = load_artifact(Path(request['imports']), request['entry'])
    circuit = dqsim.ImportedCircuit(data, metadata)
    backend, settings = request['backend'], request['settings']
    task = task_for(data, settings['task'])
    options = options_for(backend, settings)
    cls = dict(statevector=dqsim.StatevectorSimulator, mps=dqsim.MpsSimulator,
               pblock=dqsim.PBlockSimulator, stabilizer=dqsim.StabilizerSimulator)[backend]
    simulator = cls(**options)
    info = dict(task=task, simulator_options=options,
                dqsim_version=importlib.metadata.version('dqsim'),
                extension_sha256=digest(Path(dqsim._core.__file__).read_bytes()))
    # Check current native eligibility, rather than trusting a possibly stale flag.
    if backend == 'stabilizer' and not simulator.supports(circuit):
        return dict(status='ineligible', reason='Circuit is not supported by the exact stabilizer compiler', **info)
    return dict(status='ok', **info, **measure(simulator, circuit, task, settings))


def worker_main():
    try:
        result = worker(json.load(sys.stdin))
    except MemoryError as exc:
        result = dict(status='memory_error', error=str(exc))
    except Exception as exc:
        result = dict(status='error', error_type=type(exc).__name__, error=str(exc))
    print(json.dumps(result, allow_nan=False))


def run_pair(request, timeout):
    env = os.environ.copy()
    env.update(RAYON_NUM_THREADS=str(request['settings']['threads']),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    start = time.perf_counter()
    try:
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                   input=json.dumps(request), text=True, capture_output=True,
                                   env=env, timeout=timeout, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return dict(status='timeout', timeout_seconds=timeout,
                    worker_wall_seconds=time.perf_counter()-start)
    wall = time.perf_counter()-start
    if completed.returncode:
        return dict(status='crash', returncode=completed.returncode,
                    error=completed.stderr[-8000:], worker_wall_seconds=wall)
    try:
        result = json.loads(completed.stdout)
        if result.get('status') not in {'ok','ineligible','memory_error','error'}:
            raise ValueError('Unknown worker status')
    except (ValueError, AttributeError) as exc:
        return dict(status='error', error=f'Invalid worker response: {exc}',
                    stderr=completed.stderr[-8000:], worker_wall_seconds=wall)
    return dict(result, worker_wall_seconds=wall)


def summarize(rows):
    """Do not award a training label when an eligible backend failed to finish."""
    successes = [r for r in rows if r['status'] == 'ok']
    complete = (len(rows) == len(BACKENDS)
                and {r['backend'] for r in rows} == set(BACKENDS)
                and all(r['status'] == 'ok' or
                        (r['backend'] == 'stabilizer' and r['status'] == 'ineligible')
                        for r in rows))
    resource_complete = (len(rows) == len(BACKENDS)
                         and {r['backend'] for r in rows} == set(BACKENDS)
                         and all(r['status'] in ('ok', 'resource_exceeded', 'memory_error') or
                                 (r['backend'] == 'stabilizer' and r['status'] == 'ineligible')
                                 for r in rows))
    ordered = sorted(successes, key=lambda r: r['median_seconds'])
    return dict(complete=complete, resource_complete=resource_complete,
                fastest_feasible_backend=ordered[0]['backend'] if resource_complete and ordered else None,
                resource_limited_backends=[r['backend'] for r in rows
                                           if r['status'] in ('resource_exceeded', 'memory_error')],
                fastest_backend=ordered[0]['backend'] if complete and ordered else None,
                successful_backends=[r['backend'] for r in ordered],
                runner_up_ratio=(ordered[1]['median_seconds']/ordered[0]['median_seconds']
                                 if complete and len(ordered)>1 and ordered[0]['median_seconds']>0 else None))


def provenance():
    def git(*args):
        try:
            return subprocess.check_output(['git', *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return dict(python=sys.version, platform=platform.platform(), machine=platform.machine(),
                processor=platform.processor(), cpu_count=os.cpu_count(),
                git_commit=git('rev-parse','HEAD'), git_status=git('status','--short'),
                runner_sha256=digest(Path(__file__).read_bytes()))



@contextmanager
def run_lock(destination):
    # flock is released by the OS on process exit, including power loss.
    with (destination / '.collector.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another collector is already writing this run') from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def label_for(config, entry, rows, task, features):
    return dict(features=features, run_id=config['run_id'], path=entry['path'],
                family=entry['family'], task=task,
                shots=config['settings']['shots'] if task == 'shots' else None,
                normalized_sha256=entry['normalized_sha256'], **summarize(rows))


def schedule(entries, seed):
    entries = list(entries)
    rng = random.Random(seed)
    rng.shuffle(entries)
    for entry in entries:
        backends = list(BACKENDS)
        rng.shuffle(backends)
        yield entry, backends


def durable_line(stream, value):
    stream.write(json.dumps(value, allow_nan=False) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def collect(destination, config, entries, tasks, features, imports, completed):
    settings = config['settings']
    with (destination/'results.jsonl').open('a') as results, (destination/'labels.jsonl').open('a') as labels:
        for index, (entry, backends) in enumerate(schedule(entries, settings['seed']), 1):
            rows = []
            was_complete = all((entry['path'], backend) in completed for backend in backends)
            for backend in backends:
                key = (entry['path'], backend)
                if key in completed:
                    rows.append(completed[key])
                    continue
                request = dict(imports=str(imports.resolve()), entry=entry, backend=backend, settings=settings)
                row = dict(schema_version=1, run_id=config['run_id'], circuit=entry, backend=backend,
                           features=features[entry['path']], task=tasks[entry['path']],
                           simulator_options=options_for(backend, settings))
                outcome = resource_preflight(backend, features[entry['path']]['num_qubits'], settings)
                row.update(outcome if outcome is not None else run_pair(request, config['timeout_seconds']))
                durable_line(results, row)
                completed[key] = row
                rows.append(row)
                print(f'[{index}/{len(entries)}] {entry["path"]} {backend}: {row["status"]}', flush=True)
            if not was_complete:
                durable_line(labels, label_for(config, entry, rows, tasks[entry['path']], features[entry['path']]))
    print(f'Results: {destination}')


def current_native_hash():
    import dqsim
    return digest(Path(dqsim._core.__file__).read_bytes())


def resume_run(args, argv):
    # Shared strict validation is stdlib-only; no ML dependencies are needed.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from ml.data import load_dataset
    destination = args.resume.resolve()
    if not destination.is_dir():
        raise ValueError('--resume must name an existing run directory')
    provided = {arg.split('=', 1)[0] for arg in argv if arg.startswith('--')}
    if provided & {'--manifest', '--output'}:
        raise ValueError('Resume uses its frozen manifest and original output directory')
    with run_lock(destination):
        dataset = load_dataset(destination)
        config, snapshot = dataset['config'], dataset['snapshot']
        for field, value in config['settings'].items():
            if '--' + field.replace('_', '-') in provided and getattr(args, field) != value:
                raise ValueError(f'Resume cannot change saved {field}; start a new run for different settings')
        timeout = None if args.timeout_seconds == 0 else args.timeout_seconds
        if '--timeout-seconds' in provided and timeout != config['timeout_seconds']:
            raise ValueError('Resume cannot change the saved timeout')
        imports = args.imports if '--imports' in provided else Path(config['imports'])
        manifest = json.loads((destination/'manifest.json').read_text())
        entries = [e for e in manifest['circuits'] if e['status'] == 'imported']
        completed = {(r['path'], backend): row for r in dataset['records'] for backend, row in r['rows'].items()}
        old_machine = config.get('provenance', {})
        now = provenance()
        for field in ('machine', 'processor', 'cpu_count', 'python'):
            if field in old_machine and old_machine[field] != now.get(field):
                raise ValueError(f'Resume environment differs in {field}; start a new run')
        if snapshot['extension_sha256'] and current_native_hash() != snapshot['extension_sha256']:
            raise ValueError('Native simulator build changed; start a new run')
        tasks, features = {}, {}
        traits = gate_traits_classifier()
        for entry in entries:
            data, _ = load_artifact(imports, entry)
            tasks[entry['path']] = task_for(data, config['settings']['task'])
            features[entry['path']] = extract_features(data, traits)
        for (path, backend), row in completed.items():
            if features[path] != row['features'] or tasks[path] != row['task']:
                raise ValueError(f'Feature extraction changed for {path}; start a new run')
        # Everything is validated before repairing/appending benchmark files.
        results_path = destination/'results.jsonl'
        raw = results_path.read_bytes()
        if digest(raw) != snapshot['results_sha256']:
            raise ValueError('Results changed during resume validation')
        if snapshot['ignored_tail_bytes']:
            backup = destination / ('results-interrupted-' + uuid.uuid4().hex + '.bin')
            with backup.open('xb') as stream:
                stream.write(raw[snapshot['completed_prefix_bytes']:])
                stream.flush()
                os.fsync(stream.fileno())
            with results_path.open('r+b') as stream:
                stream.truncate(snapshot['completed_prefix_bytes'])
                stream.flush()
                os.fsync(stream.fileno())
        # Labels are derived data: atomically rebuild from committed results.
        temporary = destination / ('labels-rebuild-' + uuid.uuid4().hex + '.tmp')
        with temporary.open('x') as labels:
            for entry, backends in schedule(entries, config['settings']['seed']):
                if all((entry['path'], b) in completed for b in backends):
                    rows = [completed[(entry['path'], b)] for b in backends]
                    durable_line(labels, label_for(config, entry, rows, tasks[entry['path']], features[entry['path']]))
        os.replace(temporary, destination/'labels.jsonl')
        with (destination/'resumes.jsonl').open('a') as history:
            durable_line(history, dict(resumed_at=datetime.now(timezone.utc).isoformat(),
                                      completed_pairs=len(completed), repaired_tail_bytes=snapshot['ignored_tail_bytes'],
                                      imports=str(imports.resolve()), provenance=now))
        print(f'Resuming {config["run_id"]}: {len(completed)} saved pairs; '
              f'{len(entries)*len(BACKENDS)-len(completed)} remaining.', flush=True)
        collect(destination, config, entries, tasks, features, imports, completed)
    return 0


def prepare_circuits(source, output):
    """Normalize a local OpenQASM 2 directory without running any simulator."""
    from dqsim import load_qasm
    source = source.resolve()
    if not source.is_dir():
        raise ValueError('--circuits must be a directory of OpenQASM 2 files')
    paths = sorted(source.rglob('*.qasm'))
    if not paths:
        raise ValueError('No .qasm files found under --circuits')
    imports = output / ('imports-' + uuid.uuid4().hex[:12])
    imports.mkdir(parents=True, exist_ok=False)
    entries = []
    for path in paths:
        relative = path.relative_to(source)
        name = relative.parent.name if relative.parent != Path('.') else path.stem
        family = re.sub(r'(?:_n?\d+)$', '', name.removesuffix('_transpiled')) or name
        entry = dict(path=relative.as_posix(), family=family, status='import_error',
                     source_sha256=digest(path.read_bytes()))
        try:
            circuit = load_qasm(path, include_path=(source,))
            artifact = Path('circuits') / relative.with_suffix('.json')
            target = imports / artifact
            target.parent.mkdir(parents=True, exist_ok=True)
            circuit.metadata['source'] = relative.as_posix()
            target.write_text(json.dumps(dict(circuit=circuit.data, metadata=circuit.metadata),
                                         separators=(',', ':'), allow_nan=False) + '\n')
            entry.update(status='imported', artifact=artifact.as_posix(),
                         source_sha256=circuit.metadata['source_sha256'],
                         normalized_sha256=digest(circuit.model_dump_json().encode()),
                         num_qubits=circuit.num_qubits, num_cbits=circuit.num_cbits,
                         has_measurements=has_measurements(circuit.data))
        except (ValueError, OSError) as exc:
            entry['error'] = str(exc)
        entries.append(entry)
    fingerprint = digest(json.dumps([(e['path'], e['source_sha256'], e.get('normalized_sha256'))
                                     for e in entries], sort_keys=True).encode())
    manifest = imports / 'manifest.json'
    manifest.write_text(json.dumps(dict(repository='local-openqasm', revision=fingerprint,
        normalization='common-one-two-qubit-v1', circuits=entries), indent=2) + '\n')
    count = sum(e['status'] == 'imported' for e in entries)
    print(f'Imported {count}/{len(entries)} circuits; import details: {manifest}', flush=True)
    return manifest, imports


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--resume', type=Path, help='Resume an existing run using its saved settings')
    parser.add_argument('--circuits', type=Path, help='Recursively import and benchmark a directory of OpenQASM 2 files')
    parser.add_argument('--manifest', type=Path, help='Alternatively, use an existing normalized import manifest')
    parser.add_argument('--imports', type=Path, help='Artifact directory belonging to --manifest')
    parser.add_argument('--output', type=Path, default=ROOT/'data/simulator-runs')
    parser.add_argument('--task', choices=['auto','shots','evolve'], default='auto')
    parser.add_argument('--shots', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--parallel-shots', type=int, default=1)
    parser.add_argument('--max-memory-mb', type=int, default=1024)
    parser.add_argument('--timeout-seconds', type=float, default=3600,
                        help='Timeout per circuit/backend pair in seconds; 0 disables the timeout (default: 3600)')
    parser.add_argument('--seed', type=int, default=71)
    args = parser.parse_args(argv)
    if args.resume is not None:
        if args.circuits is not None:
            parser.error('--resume uses saved circuits; do not pass --circuits')
        try:
            return resume_run(args, argv)
        except (ValueError, OSError, KeyError) as exc:
            parser.error(str(exc))
    for field in ('shots','repeats','threads','parallel_shots','max_memory_mb'):
        if getattr(args,field)<1: parser.error(f'--{field.replace("_","-")} must be positive')
    if args.warmups<0 or not 0<=args.seed<2**64:
        parser.error('Warmups must be nonnegative and seed must fit an unsigned 64-bit integer')
    if not 0 <= args.timeout_seconds < float('inf'):
        parser.error('Timeout must be nonnegative and finite; use 0 for no timeout')
    if args.timeout_seconds == 0:
        args.timeout_seconds = None
    if args.circuits is not None:
        if args.manifest is not None or args.imports is not None:
            parser.error('Use --circuits or --manifest with --imports, not both')
        try:
            args.manifest, args.imports = prepare_circuits(args.circuits, args.output)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    elif args.manifest is None or args.imports is None:
        parser.error('Supply --circuits DIR, or both --manifest FILE and --imports DIR')
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    entries = [entry for entry in manifest['circuits'] if entry['status']=='imported']
    if not entries: parser.error('Manifest has no imported circuits')
    if len({entry['path'] for entry in entries}) != len(entries): parser.error('Duplicate circuit paths in manifest')
    # Check all artifacts before starting any native execution or publishing a run.
    tasks, features = {}, {}
    traits = gate_traits_classifier()
    for entry in entries:
        data, _ = load_artifact(args.imports, entry)
        tasks[entry['path']] = task_for(data, args.task)
        features[entry['path']] = extract_features(data, traits)
    settings = {key:getattr(args,key) for key in ('task','shots','repeats','warmups','threads','parallel_shots','max_memory_mb','seed')}
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    destination = args.output/run_id
    destination.mkdir(parents=True, exist_ok=False)
    config = dict(feature_version=FEATURE_VERSION, schema_version=1, run_id=run_id, settings=settings, timeout_seconds=args.timeout_seconds,
                  backends=list(BACKENDS), manifest_sha256=digest(raw),
                  manifest=str(args.manifest.resolve()), imports=str(args.imports.resolve()),
                  dataset_repository=manifest['repository'], dataset_revision=manifest['revision'],
                  normalization=manifest['normalization'], provenance=provenance(),
                  timing_scope='public simulation call including serialization/validation/compilation and result creation; excludes process startup, artifact loading, simulator construction, checks and result destruction',
                  accuracy='No intentional truncation or angle snapping; MPS still performs numerical rank removal')
    (destination/'run.json').write_text(json.dumps(config,indent=2)+'\n')
    (destination/'manifest.json').write_bytes(raw)
    with run_lock(destination):
        collect(destination, config, entries, tasks, features, args.imports, {})
    return 0


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        worker_main()
    else:
        raise SystemExit(main())
