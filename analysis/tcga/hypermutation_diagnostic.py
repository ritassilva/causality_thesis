"""As arestas mutação-mutação explicam-se pela carga mutacional?

Um tumor hipermutado tem mutações em quase todos os genes longos ao mesmo
tempo, por isso dois genes passageiros aparecem juntos sem terem nada que ver
um com o outro. Sem a TMB real (o ficheiro só tem os 35 genes), a carga de
cada doente é aproximada pelo número de genes do painel que tem mutados.
Nada aqui refaz datasets nem refita modelos: lê o TSV e os consensos que já
existem.

Três passos:

1. **Carga por doente** -- ``n_mut`` = genes mutados entre os 35. Um doente
   é de carga alta se ``n_mut`` passar o limiar: por omissão o "far out" de
   Tukey, Q3 + 3*IQR, com mínimo de 4; ``--high-burden`` fixa-o à mão.

2. **Cada par de genes (595)** --
   * ``n_comut``: doentes com os dois mutados, e quantos são de carga alta
     (``share_high_burden``);
   * Fisher em todos os doentes e Fisher só nos de carga normal;
   * um nulo que mantém **ao mesmo tempo** quantas mutações tem cada doente
     e quantos doentes tem cada gene (trocas *curveball* na matriz 665x35).
     É a ideia do DISCOVER (Canisius et al. 2016): a co-ocorrência que
     sobra depois de fixar a carga de cada doente. ``p_cooc_burden`` e
     ``p_excl_burden`` dizem se o par aparece junto mais / menos do que a
     carga sozinha prevê.

   Um par é ``explained_by_burden`` se o Fisher simples o dá como
   associado (p < alpha) mas o nulo com carga fixa não (os dois p >= alpha):
   a associação existe, mas a carga chega para a produzir.

3. **As arestas de cada braço** -- as arestas mutação-mutação do consenso
   (``tcga_mut35`` e bloco mut-mut do ``tcga_mut_expr70``) com o
   diagnóstico do par ao lado, e um resumo por braço.

Correr no Mac::

    python analysis/tcga/hypermutation_diagnostic.py

Escreve em ``results/tcga/literature_comparison/hypermutation/``.
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


