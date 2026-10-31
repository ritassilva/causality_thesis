"""Reproduce the final high-confidence TCGA glioma edge selection.

Expression-to-expression candidates must, with the same direction:

1. occur in at least 80% of the 15 PRISM seeds;
2. occur in the full cohort and at least two of the three subtypes;
3. satisfy 1--2 in both expression-only and mutation-expression data; and
4. be recovered by at least one comparison method in at least two contexts
   within each data representation.

Exact known reference edges are controls rather than novel candidates. Novelty
is assessed at the gene-pair level, independently of direction: PIK3CA -> PTEN
is therefore not presented as a novel relationship because the reference
already contains the PTEN--PIK3CA pair at the protein-signalling level. This is
a novelty classification, not evidence that the inferred expression-level
direction is false.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

MIN_EDGE_RATE = 0.8
N_SEEDS = 15
PRIMARY_MODEL = "PRISM"
REQUESTED_MODELS = (PRIMARY_MODEL, "GRASP", "AVICI", "PC", "GES", "GOLEM")
EXPRESSION_FAMILY = "tcga_expr35"
JOINT_FAMILY = "tcga_mut_expr70"


def find_project_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "analysis/glioma_analysis.py").is_file():
            return candidate
    raise FileNotFoundError("Could not locate Final_analysis/glioma_analysis.py")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))
from glioma_analysis import (
    SUBTYPES,
    build_model_graphs,
    graph_edge_set,
    load_glioma_results,
)


def reference_edges() -> set[tuple[str, str]]:
    path = PROJECT_ROOT / "datasets/tcga/references/phased_relations.csv"
    reference = pd.read_csv(path)

    def variable(gene: str, kind: str) -> str:
        return f"{gene}_mut" if kind == "mut" else f"{gene}_expr_log2tpm"

    return {
        (
            variable(row.source_gene, row.source_kind),
            variable(row.target_gene, row.target_kind),
        )
        for row in reference.itertuples(index=False)
    }


def family_candidates(
    graphs: dict, family: str, edge_type: str, min_supporting_methods: int = 1
) -> dict[tuple[str, str], dict]:
    contexts = [family, *(f"{family}__{subtype}" for subtype in SUBTYPES)]
    primary_sets = {
        context: graph_edge_set(graphs[PRIMARY_MODEL][context], mode="directed")
        for context in contexts
    }
    comparison_methods = [model for model in graphs if model != PRIMARY_MODEL]
    candidates = {}
    for edge in sorted(set().union(*primary_sets.values())):
        source, target = edge
        source_kind = "mut" if source.endswith("_mut") else "expr"
        target_kind = "mut" if target.endswith("_mut") else "expr"
        if f"{source_kind}->{target_kind}" != edge_type:
            continue
        present = [context for context in contexts if edge in primary_sets[context]]
        if contexts[0] not in present or len(present) < 3:
            continue
        method_context_counts = {
            model: sum(
                (
                    edge in graph_edge_set(graphs[model][context], mode="directed")
                    for context in present
                )
            )
            for model in comparison_methods
        }
        supporting_methods = sorted(
            (model for model, count in method_context_counts.items() if count >= 2)
        )
        if len(supporting_methods) < min_supporting_methods:
            continue
        rates = []
        for context in present:
            frame = graphs[PRIMARY_MODEL][context]
            match = frame.loc[
                frame["source"].eq(source) & frame["target"].eq(target), "directed_rate"
            ]
            rates.append(float(match.iloc[0]))
        candidates[edge] = {
            "n_contexts": len(present),
            "contexts": present,
            "minimum_seed_rate": min(rates),
            "supporting_methods": supporting_methods,
            "method_context_counts": method_context_counts,
        }
    return candidates


def short_name(variable: str) -> str:
    return variable.removesuffix("_expr_log2tpm").removesuffix("_mut")


