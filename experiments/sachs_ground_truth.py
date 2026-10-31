"""Variable aliases and the 17-arc reference graph for the Sachs dataset."""

from __future__ import annotations

import numpy as np


def canonical_label(value: str) -> str:
    """Normalise a node label to the benchmark's convention.

    Identical to ``prepare_sachs_benchmark._canonical_label``: lower-cased,
    with ``plcg`` folded to ``plc`` so that the data columns, the gold standard
    and the metrics all agree on one spelling.
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

# The 20-edge file the earlier Sachs run scored against (Zenodo 7681811, MaRDI):
# the 18 literature-supported edges plus PIP3 <-> PKA. Retained only so that
# ``difference_from_legacy`` can show what changed; nothing scores against it.
LEGACY_EDGES: tuple[tuple[str, str], ...] = (
    ("PKA", "Jnk"),
    ("PKC", "Jnk"),
    ("PKA", "P38"),
    ("PKC", "P38"),
    ("PKA", "Akt"),
    ("PIP3", "Akt"),
    ("PKA", "Erk"),
    ("Mek", "Erk"),
    ("PKA", "Mek"),
    ("PKC", "Mek"),
    ("Raf", "Mek"),
    ("PKA", "Raf"),
    ("PKC", "Raf"),
    ("PIP2", "PKC"),
    ("Plcg", "PKC"),
    ("PIP3", "PIP2"),
    ("Plcg", "PIP2"),
    ("PIP3", "Plcg"),
    ("PIP3", "PKA"),
    ("PKA", "PIP3"),
)

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
    self-loop, or if any pair appears in both directions -- the last check is
    the one that would have caught the old file before it cost a benchmark run.
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


def difference_from_legacy() -> dict[str, list[tuple[str, str]]]:
    """What changed between the old 20-edge file and the canonical 17.

    Reported in the study manifest so the change of gold standard is recorded
    with the results rather than only in this docstring.
    """
    canonical, legacy = edge_set(CANONICAL_EDGES), edge_set(LEGACY_EDGES)
    reversed_pairs = sorted(
        (source, target) for source, target in legacy - canonical if (target, source) in canonical
    )
    return {
        "kept": sorted(canonical & legacy),
        "dropped": sorted(legacy - canonical),
        "added": sorted(canonical - legacy),
        "reversed": reversed_pairs,
    }


__all__ = [
    "canonical_label",
    "CANONICAL_EDGES",
    "LEGACY_EDGES",
    "N_CANONICAL_EDGES",
    "edge_set",
    "build_true_adj",
    "difference_from_legacy",
]
