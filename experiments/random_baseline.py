"""Random DAG baseline used by the Deucalion benchmark."""

from __future__ import annotations

import numpy as np


def sample_random_dag(
    n_variables: int,
    *,
    edge_probability: float = 0.5,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a random DAG from a random topological order.

    A permutation of the variables is sampled first.  Every possible edge
    from an earlier to a later variable in that order is then retained with
    probability ``edge_probability``.  Consequently, the output is always a
    DAG and has expected edge count
    ``edge_probability * n_variables * (n_variables - 1) / 2``.

    ``seed`` is intended to be the benchmark ``model_seed``.  Pass ``rng``
    instead when a caller needs to control a generator directly; the two
    options are mutually exclusive.
    """

    n_variables = int(n_variables)
    if n_variables < 0:
        raise ValueError("n_variables must be non-negative")
    if not 0.0 <= edge_probability <= 1.0:
        raise ValueError("edge_probability must be between 0 and 1")
    if seed is not None and rng is not None:
        raise ValueError("Pass either seed or rng, not both")
    generator = rng if rng is not None else np.random.default_rng(seed)

    order = generator.permutation(n_variables)
    retained = generator.random((n_variables, n_variables)) < edge_probability
    retained = np.triu(retained, k=1).astype(int)

    adjacency = np.zeros((n_variables, n_variables), dtype=int)
    # The rows/columns of this matrix are in topological-order coordinates.
    # Re-indexing both axes maps them back to the original variable labels.
    adjacency[np.ix_(order, order)] = retained
    return adjacency


__all__ = ["sample_random_dag"]
