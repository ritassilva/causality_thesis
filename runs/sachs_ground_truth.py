""" reference graph for the Sachs dataset."""

from __future__ import annotations

import numpy as np


def canonical_label(value: str) -> str:
    """Normalise a node label to the benchmark's convention.

    Labels are lower-cased and ``plcg`` is mapped to ``plc`` so that data
    columns and reference edges use the same names.
    """
    label = str(value).replace('"', "").strip().lower()
    return "plc" if label == "plcg" else label


# The 17-arc network inferred by Sachs et al. (2005, Fig. 3A), as distributed
# in the bnlearn Bayesian Network Repository -- the version often called the
# "consensus network" in the literature. Written in bnlearn's own
# capitalisation, which is also the capitalisation of the columns in
# cd3cd28.csv.
CANONICAL_EDGES: tuple[tuple[str, str], ...] = (
    ("Erk", "Akt"),
    ("Mek", "Erk"),
    ("PIP3", "PIP2"),
    ("PKA", "Akt"),
    ("PKA", "Erk"),
    ("PKA", "Jnk"),
    ("PKA", "Mek"),
    ("PKA", "P38"),
    ("PKA", "Raf"),
    ("PKC", "Jnk"),
    ("PKC", "Mek"),
    ("PKC", "P38"),
    ("PKC", "PKA"),
    ("PKC", "Raf"),
    ("Plcg", "PIP2"),
    ("Plcg", "PIP3"),
    ("Raf", "Mek"),
)

REFERENCE_LABELS = ("Raf", "Mek", "Plcg", "PIP2", "PIP3", "Erk", "Akt", "PKA", "PKC", "P38", "Jnk")


N_CANONICAL_EDGES = len(CANONICAL_EDGES)


def edge_set(edges: tuple[tuple[str, str], ...] = CANONICAL_EDGES) -> set[tuple[str, str]]:
    """The edges as canonical labels."""
    return {(canonical_label(source), canonical_label(target)) for source, target in edges}


def build_true_adj(
    labels: list[str], edges: tuple[tuple[str, str], ...] = CANONICAL_EDGES
) -> np.ndarray:
    """The gold standard as an adjacency matrix over ``labels``.

    ``labels`` are the data's column names, already canonicalised. Raises if
    the gold standard mentions a node the data does not have, if an edge is a
    self-loop, or if any pair appears in both directions.
    """
    index = {label: position for position, label in enumerate(labels)}
    directed = edge_set(edges)

    unknown = sorted({node for edge in directed for node in edge} - set(index))
    if unknown:
        raise ValueError(f"Gold standard mentions nodes absent from the data: {unknown}")

    reciprocal = sorted(tuple(sorted(pair)) for pair in directed if (pair[1], pair[0]) in directed)
    if reciprocal:
        raise ValueError(
            "Gold standard contains reciprocal pairs, which the "
            f"correct/semi-correct metric cannot score: {sorted(set(reciprocal))}"
        )

    adjacency = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for source, target in directed:
        if source == target:
            raise ValueError(f"Gold standard contains a self-loop at {source!r}")
        adjacency[index[source], index[target]] = 1

    if np.trace(adjacency):
        raise ValueError("Gold standard unexpectedly contains a self-loop")
    if int(adjacency.sum()) != len(directed):
        raise ValueError(
            f"Expected {len(directed)} edges in the adjacency, found {int(adjacency.sum())}"
        )
    return adjacency


__all__ = [
    "canonical_label",
    "REFERENCE_LABELS",
    "CANONICAL_EDGES",
    "N_CANONICAL_EDGES",
    "edge_set",
    "build_true_adj",
]
