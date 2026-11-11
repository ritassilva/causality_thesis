"""Execute one ablation in an isolated process (called by run_ablation)."""
import json
import os
import random
import sys
import time
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import torch
from prism import PRISM, is_acyclic
from prism.prism_model import PrismMLP, PrismNonlinear

from ablation_transforms import build_lambda
from metrics import compute_metrics, directed_auprc, fbeta_score


def run(task):
    study = task['study_config']
    model = PRISM(params=task['params'], training=study['training'],
                  stein=study['stein'], glasso=study['glasso'], bootstrap=study['bootstrap'],
                  threshold=task['threshold'], seed=task['model_seed'])
    with np.load(task['sample'], allow_pickle=False) as sample:
        raw = model._data(sample['X_raw'])
        truth = np.asarray(sample['true_adj'])
        labels = list(map(str, sample['col_names'])) if 'col_names' in sample else [f'X{i}' for i in range(raw.shape[1])]
    if truth.shape != (raw.shape[1], raw.shape[1]):
        raise ValueError('true_adj shape does not match X_raw')
    with np.load(task['prior'], allow_pickle=False) as cached:
        priors = {'r_G': cached['r_G'], 'r_H': cached['r_H_raw'],
                  'binary': cached['d_binary'], 'bootstrap': cached['d_bootstrap']}
    penalty, record = build_lambda(priors, task['params'], family=task['dataset'],
                                  dataset_seed=task['dataset_seed'],
                                  prior_transform=task['prior_transform'],
                                  lambda_mode=task['lambda_mode'], B=truth)
    # Same raw-to-standardized preprocessing and seeded optimizer as PRISM.fit.
    X = (raw - raw.mean(0, keepdims=True)) / raw.std(0, keepdims=True)
    random.seed(task['model_seed'])
    np.random.seed(task['model_seed'])
    torch.manual_seed(task['model_seed'])
    torch.set_num_threads(1)
    training = dict(model.config['training'])
    hidden = training.pop('hidden_dim')
    started = time.monotonic()
    optimizer = PrismNonlinear(PrismMLP([X.shape[1], hidden, 1], bias=True))
    W = optimizer.fit(X, penalty, **training, trace_every=study['hpo']['fit_trace_every'])
    runtime = time.monotonic() - started
    if not np.isfinite(W).all():
        raise FloatingPointError('Non-finite weights')
    graph = (np.abs(W) > task['threshold']).astype(int)
    np.fill_diagonal(graph, 0)
    scores = compute_metrics(truth, graph, labels)
    scores['fbeta'] = fbeta_score(scores['precision'], scores['recall'])
    scores['auprc'] = directed_auprc(truth, np.abs(W))
    directory = Path(task['directory'])
    np.save(directory / 'weights.npy', W)
    np.save(directory / 'lambda.npy', penalty)
    off = penalty[~np.eye(len(penalty), dtype=bool)]
    return dict(status='finished', **scores, runtime_seconds=runtime,
                n_nodes=X.shape[1], n_samples=X.shape[0], is_acyclic=is_acyclic(graph),
                lambda_record=record, lambda_summary=dict(min=float(off.min()),
                mean=float(off.mean()), max=float(off.max())), retries_h_negative=optimizer.retries)


if __name__ == '__main__':
    task = json.loads(Path(sys.argv[1]).read_text())
    result = run(task)
    Path(task['directory'], 'worker_result.json').write_text(json.dumps(result) + '\n')
