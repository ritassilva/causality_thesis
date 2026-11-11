"""The Lambda of one ablation condition: prior transforms and constant Lambda.

A condition changes the penalty in exactly one of three ways.

* ``params`` -- the winner's parameters with an override (lambda_d = 0, alpha = 1,
  gate = "none", ...). ``lib/penalty.build_lambda`` is used unchanged.
* ``prior_transform`` -- the priors are transformed before Lambda is built:

  ``permute_nodes``      r_G, r_H and d are all relabelled by the same
                         deterministic node permutation. Lambda_perm[i, j] =
                         Lambda[p(i), p(j)], so the multiset of off-diagonal
                         penalties is EXACTLY the one of the full model and only
                         the alignment with the true graph is destroyed.
  ``permute_existence``  only r_G and r_H are relabelled (d stays correct).
  ``permute_direction``  only d is relabelled (the existence stays correct).
  ``reverse_direction``  d -> 1 - d, i.e. the SCORE order upside down.
  ``oracle``             r_G = r_H = the true skeleton and d = a true topological
                         order. READS THE TRUE GRAPH: a ceiling, not a model.

* ``lambda_mode = "matched_constant"`` -- the full model's Lambda is built and
  replaced by its own off-diagonal mean, one constant per cell. Same mean
  penalty coefficient (and same coefficient sum), no edge information at all.

The joint node permutation preserves the complete off-diagonal Lambda multiset.
The partial permutations need not preserve its mean because the direction term
contains the interaction g(e)(1-d); ``run.py`` therefore records the mean and
distribution of Lambda for every condition and cell.
"""
from __future__ import annotations

import hashlib

import numpy as np

from prism import penalty

PRIOR_KEYS = ("r_G", "r_H", "binary", "bootstrap")
EXISTENCE_KEYS = ("r_G", "r_H")
DIRECTION_KEYS = ("binary", "bootstrap")


def permutation_for(family: str, dataset_seed: int, d: int) -> np.ndarray:
    token = hashlib.sha256(f"modelo-final-ablation-v1:{family}:{dataset_seed}".encode()).digest()
    permutation = np.random.default_rng(int.from_bytes(token[:8], "little")).permutation(d)
    if np.array_equal(permutation, np.arange(d)):
        permutation = np.roll(permutation, 1)
    return permutation


def _relabel(matrix: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=float)[np.ix_(permutation, permutation)]


def topological_order(B: np.ndarray) -> list[int]:
    """A topological order of the true DAG; ties by node index, so it is unique."""
    A = (np.asarray(B) != 0).astype(int)
    np.fill_diagonal(A, 0)
    indegree = A.sum(axis=0).astype(int)
    order: list[int] = []
    available = sorted(np.flatnonzero(indegree == 0).tolist())
    while available:
        node = available.pop(0)
        order.append(node)
        for child in np.flatnonzero(A[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                available.append(int(child))
                available.sort()
    if len(order) != A.shape[0]:
        raise ValueError("O grafo verdadeiro tem ciclos: sem ordem topológica")
    return order


def oracle_priors(B: np.ndarray) -> dict:
    """r_G = r_H = true skeleton (1 on a true edge, 0 elsewhere), d = true order.

    With r in {0, 1} the existence is e = 1 on the true edges and 0 everywhere
    else, whatever alpha, q_G and q_H are, because 0 ** q = 0 and 1 ** q = 1.
    """
    from prism.stein import order_direction

    A = (np.asarray(B) != 0).astype(float)
    skeleton = np.maximum(A, A.T)
    np.fill_diagonal(skeleton, 0.0)
    d = order_direction(topological_order(B))
    return {"r_G": skeleton, "r_H": skeleton.copy(), "binary": d, "bootstrap": d.copy()}


def transform_priors(priors: dict, transform: str, *, family: str, dataset_seed: int,
                     B: np.ndarray | None = None) -> tuple[dict, dict]:
    """The transformed priors and what was done to them (for the record)."""
    if transform == "none":
        return dict(priors), {"prior_transform": "none"}
    if transform == "oracle":
        if B is None:
            raise ValueError("A condição oracle precisa do grafo verdadeiro")
        return oracle_priors(B), {"prior_transform": "oracle", "reads_true_graph": True}
    if transform == "reverse_direction":
        out = dict(priors)
        for key in DIRECTION_KEYS:
            out[key] = 1.0 - np.asarray(priors[key], dtype=float)
        return out, {"prior_transform": "reverse_direction"}

    keys = {"permute_nodes": PRIOR_KEYS, "permute_existence": EXISTENCE_KEYS,
            "permute_direction": DIRECTION_KEYS}.get(transform)
    if keys is None:
        raise ValueError(f"Transform desconhecido: {transform!r}")
    permutation = permutation_for(family, dataset_seed, np.asarray(priors["r_G"]).shape[0])
    out = dict(priors)
    for key in keys:
        out[key] = _relabel(priors[key], permutation)
    return out, {"prior_transform": transform, "permutation": permutation.tolist()}


def build_lambda(priors: dict, params: dict, *, family: str, dataset_seed: int,
                 prior_transform: str = "none", lambda_mode: str = "prior",
                 B: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Lambda of one condition, and the record of how it was built."""
    transformed, record = transform_priors(priors, prior_transform, family=family,
                                           dataset_seed=dataset_seed, B=B)
    Lambda = penalty.build_lambda(transformed, params)
    record["lambda_mode"] = lambda_mode
    if lambda_mode == "matched_constant":
        if prior_transform != "none":
            raise ValueError("matched_constant é sobre o Lambda do modelo completo: "
                             "não se combina com transforms do prior")
        off = Lambda[~np.eye(Lambda.shape[0], dtype=bool)]
        constant = float(off.mean())
        record["matched_constant_value"] = constant
        record["matched_constant_source"] = "média fora da diagonal do Lambda do vencedor"
        Lambda = np.full_like(Lambda, constant)
        np.fill_diagonal(Lambda, 0.0)
    elif lambda_mode != "prior":
        raise ValueError(f"Modo de Lambda desconhecido: {lambda_mode!r}")
    return Lambda, record
