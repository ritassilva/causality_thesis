"""The model with a set of edges forbidden outright.

Why a mask and not a large penalty
----------------------------------
The penalty is a per-edge weight in an L1 term, so however large ``Lambda_ij``
is made, the fit is still free to pay for that edge when the likelihood gains
more than the penalty costs. "Expression cannot cause a mutation" is not a
strong preference to be outweighed; it is a fact about the measurement, so it
belongs in the parametrisation rather than in the objective.

Why masking the weights is exact
--------------------------------
Everything the model knows about edge i -> j passes through

    A[i, j] = sum_k W[j, k, i]**2 ,   W = fc1.weight viewed as [target, hidden, source]

which is the only quantity read by ``h_func`` (acyclicity), by
``fc1_l1_reg_adaptive`` (the penalty) and by ``fc1_to_adj`` (the returned
graph). Multiplying ``fc1.weight`` by a 0/1 mask before each of those three
readings removes the forbidden entries from the forward pass, the acyclicity
constraint, the penalty and the output at once. Their gradient is then exactly
zero, because they only ever enter the loss multiplied by zero; ``fc1`` is
zero-initialised, and Adam's weight decay on a parameter that is zero with zero
gradient leaves it at zero. So a forbidden entry of the returned ``W`` is
exactly 0.0, not merely small, and ``verify`` asserts that rather than assuming
it.

``lib/dagma_model.py`` is a verbatim copy of the fit the HPO tuned and is not
touched here: the subclass overrides the four methods that read ``fc1.weight``
and inherits the rest, so a masked fit and an unmasked fit differ in the mask
and in nothing else. With an all-ones mask the two are the same computation.
"""

from __future__ import annotations

import numpy as np
import torch  # noqa: E402  (lib.dagma_model imports it anyway)
import torch.nn.functional as F  # noqa: E402
from prism.prism_model import PrismMLP, PrismNonlinear  # noqa: E402


def weight_mask(allowed: np.ndarray, hidden_dim: int) -> np.ndarray:
    """The 0/1 mask over ``fc1.weight``, shaped [d * hidden, d].

    ``fc1.weight`` has shape [d * hidden, d] and is read as [target j, hidden k,
    source i] by ``view(d, -1, d)``, so the mask entry for (j, k, i) is
    ``allowed[i, j]`` for every k.
    """
    allowed = np.asarray(allowed, dtype=bool)
    d = allowed.shape[0]
    if allowed.shape != (d, d):
        raise ValueError(f"allowed must be square, got {allowed.shape}")
    if allowed.diagonal().any():
        raise ValueError("allowed[i, i] must be False: the model has no self-loops")
    # [j, i] -> [j, hidden, i] -> [j * hidden, i]
    mask = np.repeat(allowed.T[:, None, :], hidden_dim, axis=1)
    return mask.reshape(d * hidden_dim, d).astype(float)


class MaskedPrismMLP(PrismMLP):
    """``PrismMLP`` with a fixed set of edges held at exactly zero."""

    def __init__(self, dims, allowed: np.ndarray, bias: bool = True, dtype=None):
        dtype = torch.double if dtype is None else dtype
        super().__init__(dims=dims, bias=bias, dtype=dtype)
        mask = weight_mask(allowed, hidden_dim=dims[1])
        if mask.shape != tuple(self.fc1.weight.shape):
            raise ValueError(
                f"Mask shape {mask.shape} does not match fc1.weight {tuple(self.fc1.weight.shape)}"
            )
        self.register_buffer("weight_mask", torch.from_numpy(mask).type(dtype))
        self.allowed = torch.from_numpy(np.asarray(allowed, dtype=bool))
        with torch.no_grad():
            # fc1 is zero-initialised, so this changes nothing; it is here so
            # the invariant holds from the first step whatever the base class
            # does in future.
            self.fc1.weight.mul_(self.weight_mask)

    # The single place the mask is applied; every reading below goes through it.
    def _masked_weight(self):
        return self.fc1.weight * self.weight_mask

    def _adjacency(self):
        weight = self._masked_weight().view(self.d, -1, self.d)
        return weight.pow(2).sum(dim=1).t()  # A[i, j]

    def forward(self, x):
        x = F.linear(x, self._masked_weight(), self.fc1.bias)
        x = x.view(-1, self.dims[0], self.dims[1])
        for fc in self.fc2:
            x = torch.sigmoid(x)
            x = fc(x)
        return x.squeeze(dim=2)

    def h_func(self, s: float = 1.0):
        A = self._adjacency()
        return -torch.slogdet(s * self.I - A)[1] + self.d * np.log(s)

    def fc1_l1_reg_adaptive(self, lambda_matrix, eps: float = 1e-8):
        A = self._adjacency()
        return torch.sum(lambda_matrix.to(self.fc1.weight.device) * torch.sqrt(A + eps))

    def fc1_to_adj(self) -> np.ndarray:
        with torch.no_grad():
            return torch.sqrt(self._adjacency()).cpu().numpy()


def build_model(n_nodes: int, hidden_dim: int, allowed: np.ndarray | None):
    """The fit, masked or not. ``None`` means no constraint at all.

    ``None`` means that no cross-type constraint is relevant (the single-omics
    datasets), and runs the untouched ``PrismMLP``.
    """
    unconstrained = allowed is None or bool(
        np.asarray(allowed, dtype=bool)[~np.eye(n_nodes, dtype=bool)].all()
    )
    if unconstrained:
        return PrismNonlinear(PrismMLP(dims=[n_nodes, hidden_dim, 1], bias=True))
    return PrismNonlinear(MaskedPrismMLP(dims=[n_nodes, hidden_dim, 1], allowed=allowed, bias=True))


def verify(W: np.ndarray, allowed: np.ndarray | None) -> dict:
    """Check the constraint held, and say by how much it was not merely small.

    Returns the count of forbidden cells and the largest absolute value found
    in one of them; the caller stores it. A non-zero maximum is a bug, not a
    tolerance, so it is raised rather than reported.
    """
    if allowed is None:
        return {"n_forbidden": 0, "max_abs_forbidden": 0.0, "enforced": False}
    W = np.asarray(W, dtype=float)
    forbidden = ~np.asarray(allowed, dtype=bool)
    np.fill_diagonal(forbidden, False)  # the diagonal is never an edge
    largest = float(np.abs(W[forbidden]).max()) if forbidden.any() else 0.0
    if largest != 0.0:
        raise AssertionError(
            f"A forbidden edge came back non-zero (max |W| = {largest:g}); the mask did not hold"
        )
    return {"n_forbidden": int(forbidden.sum()), "max_abs_forbidden": largest, "enforced": True}


__all__ = ["weight_mask", "MaskedPrismMLP", "build_model", "verify"]
