"""Model runners that accept hyperparameters and keep continuous output.

Two differences from ``evaluation_pipeline``'s ``run_*`` functions:

* every argument in the search space is passed through, rather than relying on
  library defaults;
* models that apply a threshold at the end return the matrix from *before* that
  threshold, so the threshold can be swept offline over the stored matrix
  instead of costing one fit per value.

A runner returns ``(prediction, is_continuous)``. When ``is_continuous`` is
True the caller must threshold before scoring; otherwise the prediction is
already a binary adjacency or a causal-learn graph.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np


def _as_array(matrix) -> np.ndarray:
    """gcastle returns its own Tensor subclass; scoring wants plain arrays."""
    return np.asarray(matrix, dtype=float)


# --------------------------------------------------------------------------- #
# Constraint-based                                                             #
# --------------------------------------------------------------------------- #
def run_pc(data, col_names, p):
    from causallearn.search.ConstraintBased.PC import pc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pc(
            data,
            alpha=p["alpha"],
            indep_test=p["indep_test"],
            uc_rule=p["uc_rule"],
            show_progress=False,
        )
    return result.G, False


def run_fci(data, col_names, p):
    from causallearn.search.ConstraintBased.FCI import fci

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph, _ = fci(
            data,
            independence_test_method=p["independence_test_method"],
            alpha=p["alpha"],
            depth=p["depth"],
            max_path_length=p["max_path_length"],
            show_progress=False,
        )
    return graph, False


# --------------------------------------------------------------------------- #
# Score-based and permutation-based                                            #
# --------------------------------------------------------------------------- #
# lambda_value is the BIC penalty discount; cv_lambda is the CV score's own
# kernel-ridge regulariser (search_spaces samples whichever applies). kfold
# stays at the library's 10.
CV_SCORE = "local_score_CV_general"


def _cv_parameters(p) -> dict:
    return {"kfold": 10, "lambda": p["cv_lambda"]}


def run_ges(data, col_names, p):
    from causallearn.search.ScoreBased.GES import ges

    # ges() copies a non-None lambda_value into `parameters` before choosing
    # the score, so for CV it must stay None and the CV keys go in directly.
    if p["score_func"] == CV_SCORE:
        parameters, lambda_value = _cv_parameters(p), None
    else:
        parameters, lambda_value = None, p["lambda_value"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ges(
            data,
            score_func=p["score_func"],
            maxP=p["maxP"],
            parameters=parameters,
            lambda_value=lambda_value,
        )
    return result["G"], False


def run_grasp(data, col_names, p):
    from causallearn.search.PermutationBased.GRaSP import grasp

    if p["score_func"] == CV_SCORE:
        parameters = _cv_parameters(p)
    else:
        parameters = {"lambda_value": p["lambda_value"]}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = grasp(
            data,
            score_func=p["score_func"],
            depth=p["depth"],
            parameters=parameters,
            verbose=False,
        )
    return graph, False


# --------------------------------------------------------------------------- #
# Topological order (dodiscover)                                               #
# --------------------------------------------------------------------------- #
def _run_toporder(estimator_cls, data, col_names, kwargs):
    import networkx as nx
    import pandas as pd
    from dodiscover import make_context

    frame = pd.DataFrame(data, columns=col_names)
    context = make_context().variables(observed=col_names).build()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = estimator_cls(**kwargs)
        model.learn_graph(frame, context)

    adjacency = nx.to_numpy_array(model.graph_, nodelist=col_names, dtype=int)
    np.fill_diagonal(adjacency, 0)
    return adjacency, False


def run_score(data, col_names, p):
    from dodiscover.toporder import SCORE

    return _run_toporder(
        SCORE,
        data,
        col_names,
        dict(
            alpha=p["alpha"],
            eta_G=p["eta_G"],
            eta_H=p["eta_H"],
            n_splines=p["n_splines"],
        ),
    )


def run_nogam(data, col_names, p):
    from dodiscover.toporder import NoGAM

    return _run_toporder(
        NoGAM,
        data,
        col_names,
        dict(
            alpha=p["alpha"],
            eta_G=p["eta_G"],
            eta_H=p["eta_H"],
            ridge_alpha=p["ridge_alpha"],
            n_splines=p["n_splines"],
        ),
    )


def run_das(data, col_names, p):
    from dodiscover.toporder import DAS

    return _run_toporder(
        DAS,
        data,
        col_names,
        dict(
            alpha=p["alpha"],
            das_cutoff=p["das_cutoff"],
            eta_G=p["eta_G"],
            eta_H=p["eta_H"],
            max_parents=p["max_parents"],
        ),
    )


# --------------------------------------------------------------------------- #
# Continuous optimisation -- these keep their pre-threshold matrix             #
# --------------------------------------------------------------------------- #
def run_notears(data, col_names, p):
    from castle.algorithms import Notears

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Notears(
            lambda1=p["lambda1"],
            loss_type="l2",
            max_iter=p["max_iter"],
            w_threshold=0.0,
        )
        model.learn(data)
    return _as_array(model.weight_causal_matrix), True


def run_golem(data, col_names, p):
    from castle.algorithms import GOLEM

    # The GOLEM paper initialises the non-equal-variance objective from the
    # equal-variance solution on the same data; gcastle passes B_init straight
    # through and does nothing of the sort on its own. Without this, NV would
    # be judged on a bad local optimum rather than on its merits.
    b_init = None
    init_path = p.get("b_init_path")
    if init_path:
        from pathlib import Path

        path = Path(init_path)
        if not path.exists():
            # Running NV from zeros instead would score a different procedure
            # than the one searched; fail so it shows up as an error.
            raise FileNotFoundError(f"GOLEM-NV initialisation missing: {path}")
        with np.load(path, allow_pickle=False) as stored:
            b_init = np.asarray(stored["B"], dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GOLEM(
            B_init=b_init,
            lambda_1=p["lambda_1"],
            lambda_2=p["lambda_2"],
            equal_variances=p.get("equal_variances", True),
            learning_rate=p["learning_rate"],
            num_iter=p["num_iter"],
            graph_thres=0.0,
        )
        model.learn(data)
    return _as_array(model.weight_causal_matrix), True


def run_dagma_linear(data, col_names, p):
    from dagma.linear import DagmaLinear

    model = DagmaLinear(loss_type="l2")
    W = model.fit(
        data,
        lambda1=p["lambda1"],
        w_threshold=0.0,
        T=p["T"],
        lr=p["lr"],
        mu_factor=p["mu_factor"],
        warm_iter=p.get("warm_iter", 30000),
        max_iter=p.get("max_iter", 60000),
    )
    return _as_array(W), True


def run_dagma_nonlinear(data, col_names, p):
    import torch
    from dagma.nonlinear import DagmaMLP, DagmaNonlinear

    n_vars = data.shape[1]
    scale = p.get("iter_scale", 1.0)
    eq_model = DagmaMLP(dims=[n_vars, p["hidden_dim"], 1], bias=True)
    model = DagmaNonlinear(eq_model)
    W = model.fit(
        torch.from_numpy(data),
        lambda1=p["lambda1"],
        lambda2=p["lambda2"],
        T=p["T"],
        lr=p["lr"],
        warm_iter=int(5e4 * scale),
        max_iter=int(8e4 * scale),
        w_threshold=0.0,
    )
    return _as_array(W), True


def run_gran_dag(data, col_names, p):
    from castle.algorithms import GraNDAG

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GraNDAG(
            input_dim=data.shape[1],
            hidden_dim=p["hidden_dim"],
            hidden_num=p["hidden_num"],
            lr=p["lr"],
            model_name=p["model_name"],
            use_pns=p["use_pns"],
            iterations=p.get("iterations", 10000),
        )
        model.learn(data)
    # gcastle does not expose GraNDAG's pre-threshold jacobian, so unlike the
    # other gradient methods this one can only be scored as a binary graph.
    adjacency = (_as_array(model.causal_matrix) != 0).astype(int)
    np.fill_diagonal(adjacency, 0)
    return adjacency, False


# --------------------------------------------------------------------------- #
# Others                                                                       #
# --------------------------------------------------------------------------- #
def run_directlingam(data, col_names, p):
    from causallearn.search.FCMBased.lingam import DirectLiNGAM

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DirectLiNGAM(measure=p["measure"])
        model.fit(data)
    # adjacency_matrix_[i, j] is the coefficient of j -> i, hence the transpose.
    return _as_array(model.adjacency_matrix_).T, True


_AVICI_MODEL = None


def _avici_cache_dir():
    """Locate the pretrained SCM weights in the configured model cache."""
    import os
    from pathlib import Path

    configured = os.environ.get("AVICI_CACHE_DIR")
    path = Path(configured) if configured else Path(__file__).resolve().parent.parent / "datasets/avici/pretrained"
    if not (path / "models--larslorch--avici").is_dir():
        raise FileNotFoundError(
            f"AVICI weights not found under {path}; set AVICI_CACHE_DIR to the "
            "folder that contains models--larslorch--avici"
        )
    return str(path)


def run_avici(data, col_names, p):
    import avici
    from avici_settings import avici_experimental_chunk_size

    global _AVICI_MODEL
    if _AVICI_MODEL is None:
        _AVICI_MODEL = avici.load_pretrained(download="scm-v0", cache_path=_avici_cache_dir())

    probabilities = np.asarray(
        _AVICI_MODEL(
            x=data.astype(np.float32),
            experimental_chunk_size=avici_experimental_chunk_size(),
        )
    )
    np.fill_diagonal(probabilities, 0.0)
    return probabilities, True


RUNNERS: dict[str, Any] = {
    "PC": run_pc,
    "FCI": run_fci,
    "GES": run_ges,
    "GRaSP": run_grasp,
    "SCORE": run_score,
    "NoGAM": run_nogam,
    "DAS": run_das,
    "NOTEARS": run_notears,
    "DAGMA_Linear": run_dagma_linear,
    "DAGMA_Nonlinear": run_dagma_nonlinear,
    "GOLEM": run_golem,
    "GraNDAG": run_gran_dag,
    "DirectLiNGAM": run_directlingam,
    "AVICI": run_avici,
}

# Imported before seeding so that library initialisation does not consume the
# seeded RNG stream used by the benchmark.
TORCH_MODELS = frozenset({"NOTEARS", "GraNDAG", "GOLEM", "DAGMA_Nonlinear", "DAGMA_Linear"})


def run(model: str, data: np.ndarray, col_names: list[str], params: dict) -> tuple[Any, bool]:
    return RUNNERS[model](data, col_names, params)
