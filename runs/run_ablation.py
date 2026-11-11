"""Plan or execute the 13 PRISM ablations plus a full reference with matching seeds.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
from synthetic_data import configure_samples, sample_path

# condition, parameter overrides, prior transformation, penalty mode
CONDITIONS = (
    ('full', {}, 'none', 'prior'),
    ('no_prior', {'lambda_e': 0.0, 'lambda_d': 0.0}, 'none', 'prior'),
    ('matched_constant', {}, 'none', 'matched_constant'),
    ('permuted_prior', {}, 'permute_nodes', 'prior'),
    ('permuted_existence', {}, 'permute_existence', 'prior'),
    ('permuted_direction', {}, 'permute_direction', 'prior'),
    ('reversed_direction', {}, 'reverse_direction', 'prior'),
    ('no_direction', {'lambda_d': 0.0}, 'none', 'prior'),
    ('no_existence', {'lambda_e': 0.0}, 'none', 'prior'),
    ('glasso_only', {'alpha': 1.0}, 'none', 'prior'),
    ('hessian_only', {'alpha': 0.0}, 'none', 'prior'),
    ('glasso_no_direction', {'alpha': 1.0, 'lambda_d': 0.0}, 'none', 'prior'),
    ('hessian_no_direction', {'alpha': 0.0, 'lambda_d': 0.0}, 'none', 'prior'),
    ('oracle', {}, 'oracle', 'prior'),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--priors-dir', type=Path, default=ROOT / 'results/prism/priors')
    parser.add_argument('--condition', choices=[c[0] for c in CONDITIONS], action='append')
    parser.add_argument('--family', action='append')
    parser.add_argument('--dataset-seed', type=int, action='append')
    parser.add_argument('--model-seed', type=int, action='append')
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be positive')
    configure_samples(args.samples)
    study = json.loads((ROOT / 'configs/prism_study.json').read_text())
    winner = json.loads((ROOT / 'configs/prism_winner.json').read_text())
    selections = []
    for requested, allowed in ((args.family, study['families']),
                               (args.dataset_seed, study['test_dataset_seeds']),
                               (args.model_seed, study['model_seeds'])):
        if requested and not set(requested).issubset(allowed):
            parser.error(f'Selection must belong to {allowed}')
        selections.append(requested or allowed)
    families, seeds, model_seeds = selections
    sources = {name: sha256(ROOT / name) for name in (
        'runs/run_ablation.py', 'runs/ablation_worker.py', 'runs/ablation_transforms.py',
        'runs/metrics.py', 'requirements-models.txt')}
    tasks = []
    for family in families:
        for seed in seeds:
            sample = sample_path(family, seed, study='ablation')
            prior = (args.priors_dir / f'{family}__seed_{seed}.npz').resolve()
            identity = dict(data_sha256=sha256(sample), prior_sha256=sha256(prior))
            for condition, overrides, transform, mode in CONDITIONS:
                if args.condition and condition not in args.condition:
                    continue
                for ms in model_seeds:
                    directory = args.output.resolve() / 'fits' / condition / family / f'ds{seed}__ms{ms}'
                    tasks.append(dict(dataset=family, dataset_seed=seed, model_seed=ms,
                        condition_id=condition, params={**winner['params'], **overrides},
                        threshold=winner['threshold'], prior_transform=transform, lambda_mode=mode,
                        study_config=study, source_sha256=sources, **identity,
                        sample=str(sample), prior=str(prior), directory=str(directory)))
    args.output.mkdir(parents=True, exist_ok=True)
    plan = args.output / 'plan.json'
    if plan.exists():
        if json.loads(plan.read_text()) != tasks:
            raise ValueError('Output plan differs; use a new output directory')
        if args.execute and not args.resume:
            raise ValueError('Existing plan: use --resume or a new output directory')
    elif any(args.output.iterdir()):
        raise FileExistsError(args.output)
    else:
        atomic_json(plan, tasks)
    print(f'{len(tasks)} fits; execute={args.execute}', flush=True)
    if not args.execute:
        return
    # Import only during execution so planning does not require model dependencies.
    import numpy as np
    import pandas as pd
    import prism
    import torch
    package = Path(prism.__file__).parent
    runtime = dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__,
                   torch=torch.__version__, prism=getattr(prism, '__version__', None),
                   prism_sources={str(path.relative_to(package)): sha256(path)
                                  for path in sorted(package.rglob('*'))
                                  if path.is_file() and path.suffix in {'.py', '.json'}})
    runtime_file = args.output / 'runtime.json'
    if runtime_file.exists() and json.loads(runtime_file.read_text()) != runtime:
        raise ValueError('Model implementation or runtime changed; use a new output directory')
    atomic_json(runtime_file, runtime)
    outcomes = []
    for task in tasks:
        directory = Path(task['directory'])
        directory.mkdir(parents=True, exist_ok=True)
        result_file = directory / 'result.json'
        signature = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
        result = None
        if args.resume and result_file.exists():
            previous = json.loads(result_file.read_text())
            if previous.get('task_sha256') == signature and previous.get('status') == 'finished':
                artifacts = previous.get('artifact_sha256', {})
                if artifacts and all((directory / name).is_file() and sha256(directory / name) == digest
                                     for name, digest in artifacts.items()):
                    result = previous
        if result is None:
            atomic_json(directory / 'task.json', task)
            started = time.monotonic()
            with (directory / 'execution.log').open('w') as log:
                try:
                    process = subprocess.run([sys.executable, str(ROOT / 'runs/ablation_worker.py'),
                        str(directory / 'task.json')], stdout=log, stderr=log, timeout=args.timeout)
                    if process.returncode:
                        result = dict(status='error', returncode=process.returncode)
                    else:
                        result = json.loads((directory / 'worker_result.json').read_text())
                        W = np.load(directory / 'weights.npy', allow_pickle=False)
                        if W.shape != (result['n_nodes'], result['n_nodes']) or not np.isfinite(W).all():
                            raise ValueError('Invalid saved weights')
                        result['artifact_sha256'] = {name: sha256(directory / name)
                                                    for name in ('weights.npy', 'lambda.npy')}
                except subprocess.TimeoutExpired:
                    result = dict(status='timeout', runtime_seconds=time.monotonic() - started)
            result['task_sha256'] = signature
            atomic_json(result_file, result)
        outcomes.append(dict(dataset=task['dataset'], model='PRISM', arm='tuned',
            dataset_seed=task['dataset_seed'], model_seed=task['model_seed'],
            condition_id=task['condition_id'], timed_out=result['status'] == 'timeout', **result))
        print(f"{task['condition_id']} {task['dataset']} ds{task['dataset_seed']} ms{task['model_seed']}: {result['status']}", flush=True)
        # Incremental export survives interruption; raw diagnostics stay per fit.
        frame = pd.DataFrame(outcomes)
        columns = ['dataset', 'model', 'arm', 'dataset_seed', 'model_seed', 'condition_id',
                   'status', 'timed_out', 'n_nodes', 'n_samples', 'f1', 'fbeta', 'precision',
                   'recall', 'SHD', 'runtime_seconds', 'auprc', 'is_acyclic']
        csv = args.output / 'measurements.csv'
        frame.reindex(columns=columns).to_csv(csv.with_suffix('.tmp'), index=False)
        os.replace(csv.with_suffix('.tmp'), csv)
    if any(row['status'] != 'finished' for row in outcomes):
        raise SystemExit('Some fits failed or timed out; inspect execution.log and resume')


if __name__ == '__main__':
    main()
