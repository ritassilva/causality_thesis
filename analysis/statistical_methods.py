"""Paired statistical summaries from the final experiment results."""

import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from Wilcoxon_test.wilcoxon_test import paired_wilcoxon as wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import result_tables as th


def holm(pvalues):
    values = np.asarray(pvalues, float)
    order = np.argsort(values, kind="stable")
    corrected = np.empty(len(values))
    corrected[order] = np.minimum(
        1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1))
    )
    return corrected


def paired_scope(frame, experiment, scope, metric, higher=True):
    """Average model runs within dataset blocks, excluding incomplete models.

    Each Holm family is exactly one experiment/scope/metric. Identical paired
    values yield p=1. Missing runs are omitted.
    """
    eligible = frame.groupby("model").finished.all()
    coverage = frame.groupby("model").agg(
        recorded_runs=("finished", "size"), completed_runs=("finished", "sum")
    )
    coverage["included"] = eligible
    coverage["reason"] = np.where(eligible, "complete", "unfinished_fit")
    blocks = (
        frame.loc[frame.finished & frame[metric].notna()]
        .groupby(["dataset", "dataset_seed", "model"], observed=True)[metric]
        .mean()
        .unstack("model")
    )
    # No pairwise deletion: every model in a scope must share all blocks.
    included = [m for m in eligible[eligible].index if m in blocks and blocks[m].notna().all()]
    coverage.loc[~coverage.index.isin(included) & coverage.included, "reason"] = (
        "missing_metric_block"
    )
    coverage["included"] = coverage.index.isin(included)
    pivot = blocks[included]
    records = []
    for a, b in combinations(included, 2):
        delta = (pivot[a] - pivot[b]).to_numpy(float)
        all_zero = bool(np.all(delta == 0))
        result = None if all_zero else wilcoxon(delta, alternative="two-sided")
        records.append(
            dict(
                model_a=a,
                model_b=b,
                n_pairs=len(delta),
                statistic=0.0 if all_zero else float(result.statistic),
                pvalue=1.0 if all_zero else float(result.pvalue),
                mean_a=float(pivot[a].mean()),
                mean_b=float(pivot[b].mean()),
                mean_difference=float(delta.mean()),
                median_difference=float(np.median(delta)),
                all_differences_zero=all_zero,
                fraction_a_better=float(np.mean(delta > 0 if higher else delta < 0)),
                fraction_tied=float(np.mean(delta == 0)),
            )
        )
    pairs = pd.DataFrame(records)
    if len(pairs):
        pairs["pvalue_holm"] = holm(pairs.pvalue)
        pairs["significant_holm"] = pairs.pvalue_holm < 0.05
        pairs["better_by_mean"] = [
            a if (d > 0 if higher else d < 0) else b if d != 0 else "tie"
            for a, b, d in zip(pairs.model_a, pairs.model_b, pairs.mean_difference)
        ]
        pairs["winner"] = np.where(
            pairs.significant_holm, pairs.better_by_mean, "no_significant_difference"
        )
    friedman = pd.DataFrame()
    if len(pivot.columns) >= 3 and len(pivot) >= 2:
        result = friedmanchisquare(*(pivot[c] for c in pivot))
        friedman = pd.DataFrame(
            [
                dict(
                    n_blocks=len(pivot),
                    n_models=len(pivot.columns),
                    statistic=float(result.statistic),
                    pvalue=float(result.pvalue),
                )
            ]
        )
    # Reuse publication ranking convention rather than inventing another ranking.
    rank_rows = []
    for model in included:
        wins = int((pairs.winner == model).sum()) if len(pairs) else 0
        significant = pairs.loc[pairs.significant_holm] if len(pairs) else pairs
        losses = (
            int(((significant.model_a == model) | (significant.model_b == model)).sum()) - wins
            if len(pairs)
            else 0
        )
        rank_rows.append(
            dict(
                model=model,
                mean=float(pivot[model].mean()),
                wins=wins,
                losses=losses,
                n_blocks=len(pivot),
            )
        )
    ranks = pd.DataFrame(rank_rows)
    if len(pairs):
        from Wilcoxon_test.wilcoxon_test import build_wilcoxon_rank

        ranking_input = pairs.rename(
            columns={
                "model_a": "model_1",
                "model_b": "model_2",
                "better_by_mean": "better_model_by_mean",
            }
        )
        reference_ranks = build_wilcoxon_rank(
            ranking_input,
            models=included,
            model_means={m: float(pivot[m].mean()) for m in included},
            higher_is_better=higher,
        )
        ranks = ranks.merge(
            reference_ranks[["model", "rank", "tier"]], on="model", validate="one_to_one"
        )
    tidy = pivot.stack().rename(metric).reset_index()
    for table in (pairs, friedman, ranks, coverage):
        table["experiment"] = experiment
        table["scope"] = scope
        table["metric"] = metric
        table["higher_is_better"] = higher
    return pairs, friedman, ranks, coverage.reset_index(), tidy

