"""Assess whether mutation co-occurrence is associated with panel mutation burden.

Burden is the number of mutated genes in the recorded panel, not a genome-wide
mutation-burden estimate. Pairwise Fisher comparisons are complemented by a
curveball null that preserves row and column sums of the binary mutation
matrix. The diagnostic reads the prepared cohort and saved consensus edges;
it does not regenerate datasets or refit causal models.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from compare_literature import (
    DEFAULT_VIEW,
    FINAL_MODEL,
    MUT,
    RESULTS,
    default_source_file,
    gene_of,
    kind_of,
    load_arm_skeletons,
    pair,
)
from scipy import stats

OUT = RESULTS / "hypermutation"

# Genes longos / frequentemente mutados em exomas de população (lista FLAGS,
# Shyr et al. 2014) ou fora do Cancer Gene Census -- candidatos a passageiro.
# Só serve para marcar linhas; nada é calculado de forma diferente para eles.
# Confirmar contra o COSMIC Cancer Gene Census antes de citar.
SUSPECTED_PASSENGERS = (
    "MUC17",
    "MYH8",
    "TRPV6",
    "NUP210L",
    "FAM47C",
    "SLC6A3",
    "TNRC18",
)

MUT_DATASETS = ("tcga_mut35", "tcga_mut_expr70")


# 1. Carga por doente ---------------------------------------------------------- #


def burden_threshold(counts: pd.Series, fixed: int | None) -> tuple[int, str]:
    if fixed is not None:
        return fixed, "fixed"
    q1, q3 = counts.quantile([0.25, 0.75])
    return max(4, int(np.ceil(q3 + 3 * (q3 - q1)))), "tukey_far_out_min4"


# 2. Nulo com carga e frequência fixas ---------------------------------------- #


def curveball_null(matrix: np.ndarray, n_null: int, rng: np.random.Generator) -> np.ndarray:
    """Matrizes de co-mutação (n_null x genes x genes) com margens fixas.

    Algoritmo curveball (Strona et al. 2014): dois doentes trocam entre si,
    ao acaso, os genes que um tem e o outro não. Mantém exactamente o número
    de mutações de cada doente e de doentes de cada gene. Aquecimento de
    5 x doentes trocas e 1 x doentes entre amostras.
    """
    n_rows, n_cols = matrix.shape
    rows = [set(np.flatnonzero(r)) for r in matrix]

    def run(n_trades: int) -> None:
        picks = rng.integers(n_rows, size=(n_trades, 2))
        for i, j in picks:
            if i == j:
                continue
            a, b = rows[i], rows[j]
            only_a, only_b = a - b, b - a
            if not only_a or not only_b:
                continue
            pool = list(only_a | only_b)
            rng.shuffle(pool)
            shared = a & b
            rows[i] = shared | set(pool[: len(only_a)])
            rows[j] = shared | set(pool[len(only_a) :])

    run(5 * n_rows)
    out = np.empty((n_null, n_cols, n_cols), dtype=np.int32)
    sample = np.zeros_like(matrix, dtype=np.int32)
    for n in range(n_null):
        run(n_rows)
        sample[:] = 0
        for r, genes in enumerate(rows):
            sample[r, list(genes)] = 1
        out[n] = sample.T @ sample
    return out


def fisher(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    table = [
        [int(np.sum(a & b)), int(np.sum(a & ~b))],
        [int(np.sum(~a & b)), int(np.sum(~a & ~b))],
    ]
    _, p = stats.fisher_exact(table)
    log_or = np.log(
        ((table[0][0] + 0.5) * (table[1][1] + 0.5)) / ((table[0][1] + 0.5) * (table[1][0] + 0.5))
    )
    return float(log_or), float(p)


def pair_diagnostics(
    muts: pd.DataFrame,
    high: np.ndarray,
    n_null: int,
    alpha: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    labels = list(muts.columns)
    matrix = muts.to_numpy().astype(bool)
    null = curveball_null(matrix.astype(np.int32), n_null, rng)
    rows = []
    for i, j in combinations(range(len(labels)), 2):
        a, b = matrix[:, i], matrix[:, j]
        both = a & b
        log_or, p_all = fisher(a, b)
        log_or_normal, p_normal = fisher(a[~high], b[~high])
        observed = int(both.sum())
        draws = null[:, i, j]
        gene_a, gene_b = gene_of(labels[i]), gene_of(labels[j])
        p_cooc = (1 + np.sum(draws >= observed)) / (1 + n_null)
        p_excl = (1 + np.sum(draws <= observed)) / (1 + n_null)
        rows.append(
            {
                "gene_a": gene_a,
                "gene_b": gene_b,
                "carriers_a": int(a.sum()),
                "carriers_b": int(b.sum()),
                "n_comut": observed,
                "n_comut_high_burden": int((both & high).sum()),
                "share_high_burden": (both & high).sum() / observed if observed else np.nan,
                "log_or_all": log_or,
                "p_fisher_all": p_all,
                "log_or_normal_burden": log_or_normal,
                "p_fisher_normal_burden": p_normal,
                "n_comut_expected_given_burden": float(draws.mean()),
                "p_cooc_burden": p_cooc,
                "p_excl_burden": p_excl,
                "explained_by_burden": bool(p_all < alpha and p_cooc >= alpha and p_excl >= alpha),
                "both_suspected_passengers": (
                    gene_a in SUSPECTED_PASSENGERS and gene_b in SUSPECTED_PASSENGERS
                ),
                "any_suspected_passenger": (
                    gene_a in SUSPECTED_PASSENGERS or gene_b in SUSPECTED_PASSENGERS
                ),
            }
        )
    return pd.DataFrame(rows)


# 3. Arestas dos braços ------------------------------------------------------- #


def arm_edges(within: float, view_70: str, pairs: pd.DataFrame) -> pd.DataFrame:
    views = {d: (view_70 if d == "tcga_mut_expr70" else DEFAULT_VIEW[d]) for d in MUT_DATASETS}
    skeletons, _ = load_arm_skeletons(within, views)
    by_pair = {pair(r.gene_a, r.gene_b): r._asdict() for r in pairs.itertuples(index=False)}
    rows = []
    for (dataset, arm), edges in sorted(skeletons.items()):
        mut_edges = [e for e in edges if all(kind_of(x) == MUT for x in e)]
        if not mut_edges:
            rows.append({"dataset": dataset, "arm": arm, "gene_a": None})
            continue
        for edge in sorted(mut_edges, key=sorted):
            a, b = sorted(gene_of(x) for x in edge)
            rows.append({"dataset": dataset, "arm": arm, **by_pair[pair(a, b)]})
    return pd.DataFrame(rows)


def summarise_arms(edges: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Por braço: quantas arestas mut-mut, e quantas cheiram a carga mutacional."""
    rows = []
    for (dataset, arm), group in edges.groupby(["dataset", "arm"], sort=True):
        real = group.dropna(subset=["gene_a"])
        rows.append(
            {
                "dataset": dataset,
                "arm": arm,
                "n_mut_mut_edges": len(real),
                "n_both_suspected_passengers": int(
                    real.both_suspected_passengers.astype(bool).sum()
                ),
                "n_any_suspected_passenger": int(real.any_suspected_passenger.astype(bool).sum()),
                "n_majority_high_burden": int((real.share_high_burden > 0.5).sum()),
                "n_explained_by_burden": int(real.explained_by_burden.astype(bool).sum()),
                "n_beyond_burden": int(
                    ((real.p_cooc_burden < alpha) | (real.p_excl_burden < alpha)).sum()
                ),
                "n_no_comutation": int((real.n_comut == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


