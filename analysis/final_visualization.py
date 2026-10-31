"""Reusable visualisations for benchmark and model evaluation results.

The module deliberately separates quantities that already exist in the result
CSVs from descriptive summaries computed for a figure:

* graph metrics are read from the saved per-condition results;
* Wilcoxon-Holm ranks use paired dataset blocks;
* model seeds are averaged inside each independent dataset block;
* uncertainty is then computed across dataset seeds or configurations.

No public plotting function calls :func:`matplotlib.pyplot.show` and saving is
always opt-in.  Every plot returns the exact tidy data used to draw it.
"""

from __future__ import annotations

import inspect
import json
import math
import re
import sqlite3
import unicodedata
import warnings
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

_PLOT_ALPHA: ContextVar[float] = ContextVar("plot_alpha", default=1.0)
DEFAULT_FONT_SIZE: float = 14
DEFAULT_TICK_FONT_SIZE: float = 12
DEFAULT_TITLE_SIZE: float = 16
THESIS_FONT_FAMILY: str = "DejaVu Sans"
THESIS_RC_PARAMS: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": [THESIS_FONT_FAMILY, "Arial", "Liberation Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": DEFAULT_FONT_SIZE,
    "xtick.labelsize": DEFAULT_TICK_FONT_SIZE,
    "ytick.labelsize": DEFAULT_TICK_FONT_SIZE,
    "axes.titlesize": DEFAULT_TITLE_SIZE,
    "axes.labelsize": DEFAULT_FONT_SIZE,
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style() -> None:
    """Apply the typography and colour cycle used by the six-panel F1 plot."""
    plt.rcParams.update(THESIS_RC_PARAMS)


DEFAULT_MODEL_ORDER: tuple[str, ...] = (
    "PC",
    "FCI",
    "GES",
    "GRaSP",
    "DirectLiNGAM",
    "NOTEARS",
    "GOLEM",
    "DAGMA_Linear",
    "SCORE",
    "NoGAM",
    "DAS",
    "GraNDAG",
    "DAGMA_Nonlinear",
    "AVICI",
)
OKABE_ITO: dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}
DEFAULT_MODEL_COLORS: dict[str, str] = {
    "PC": OKABE_ITO["blue"],
    "FCI": OKABE_ITO["orange"],
    "GES": OKABE_ITO["bluish_green"],
    "GRaSP": OKABE_ITO["vermillion"],
    "DirectLiNGAM": OKABE_ITO["reddish_purple"],
    "NOTEARS": OKABE_ITO["yellow"],
    "GOLEM": OKABE_ITO["sky_blue"],
    "DAGMA_Linear": OKABE_ITO["black"],
    "SCORE": "#A66F00",
    "NoGAM": "#F3C45E",
    "DAS": "#00664B",
    "GraNDAG": "#8FD3F4",
    "DAGMA_Nonlinear": "#004C7F",
    "AVICI": "#E3A1C7",
    "PRISM": "#8A3F71",
    "Random": "#7A3E00",
}
THESIS_RC_PARAMS["axes.prop_cycle"] = plt.cycler(color=list(DEFAULT_MODEL_COLORS.values()))
apply_style()
OKABE_ITO_EXTENDED: tuple[str, ...] = tuple(
    dict.fromkeys((*DEFAULT_MODEL_COLORS.values(), *OKABE_ITO.values()))
)
OKABE_ITO_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "okabe_ito_sequential",
    ("#F7FBFF", OKABE_ITO["sky_blue"], OKABE_ITO["blue"], "#00304F"),
)
OKABE_ITO_SEQUENTIAL_R = OKABE_ITO_SEQUENTIAL.reversed()
OKABE_ITO_DIVERGING = LinearSegmentedColormap.from_list(
    "okabe_ito_diverging", (OKABE_ITO["vermillion"], "#F7F7F7", OKABE_ITO["blue"])
)
DEFAULT_SCALABILITY_FAMILIES: dict[str, str] = {
    "eb_gaussian_linear": "Gaussian Linear",
    "eb_nongaussian_mixed": "Non Gaussian Mixed",
}
GRAPH_TYPE_DISPLAY_LABELS: dict[str, str] = {
    "ModMultiOmics": "MultiOmics",
    "ModOncogene": "Oncogene",
}
COMMON_ROBUSTNESS_DATASET_SEEDS: tuple[int, ...] = (19, 47, 50, 52, 71, 75, 77, 102, 125, 158, 222, 237, 308, 402, 603)
DEFAULT_ROBUSTNESS_FAMILIES: tuple[str, ...] = (
    "20nodes_EB_obsconf0_latconf0_Non_Gaussian_Mixed",
    "20nodes_ModMultiOmics_obsconf0_latconf0_Laplace_Mixed",
    "20nodes_ModOncogene_obsconf0_latconf0_Gaussian_Mixed",
    "20nodes_EB_obsconf0_latconf0_Gaussian_Linear",
    "20nodes_EB_obsconf0_latconf0_Gaussian_Mixed",
    "20nodes_EB_obsconf0_latconf0_NonGaussian_Linear",
)
DEFAULT_ROBUSTNESS_MODELS: tuple[str, ...] = (*DEFAULT_MODEL_ORDER, "PRISM")
MODEL_NAME_ALIASES: dict[str, str] = {}
MODEL_DISPLAY_LABELS: dict[str, str] = {
    "DAGMA_Linear": "DAGMA Linear",
    "DAGMA_Nonlinear": "DAGMA Nonlinear",
    "PRISM": "PRISM",
    "GraNDAG": "GraN-DAG",
}
FBETA: float = 0.5
F1_LABEL: str = "$F_1$"
F05_LABEL: str = "$F_{0.5}$"
METRIC_INFO: dict[str, dict[str, Any]] = {
    "f1": {"label": F1_LABEL, "higher_is_better": True},
    "fbeta": {"label": F05_LABEL, "higher_is_better": True},
    "precision": {"label": "Precision", "higher_is_better": True},
    "recall": {"label": "Recall", "higher_is_better": True},
    "SHD": {"label": "SHD", "higher_is_better": False},
    "runtime_seconds": {"label": "Runtime (s)", "higher_is_better": False},
}
DEFAULT_ASSUMPTIONS = pd.DataFrame(
    [
        {
            "model": "DirectLiNGAM",
            "noise": "NonGaussian",
            "mechanism": "Linear",
            "confounding": ("NoConf", "ObsConf"),
            "rationale": "Linear non-Gaussian SEM.",
        },
        {
            "model": "DAGMA_Linear",
            "mechanism": "Linear",
            "confounding": ("NoConf", "ObsConf"),
            "rationale": "Linear structural equations.",
        },
        {
            "model": "NOTEARS",
            "mechanism": "Linear",
            "confounding": ("NoConf", "ObsConf"),
            "rationale": "Original linear NOTEARS formulation.",
        },
        {
            "model": "GOLEM",
            "mechanism": "Linear",
            "confounding": ("NoConf", "ObsConf"),
            "rationale": "Linear SEM likelihood.",
        },
        {
            "model": "DAGMA_Nonlinear",
            "mechanism": "Mixed",
            "confounding": ("NoConf", "ObsConf"),
            "rationale": "Mixed node mechanisms.",
        },
        {
            "model": "SCORE",
            "noise": "Gaussian",
            "mechanism": "Mixed",
            "confounding": ("NoConf", "ObsConf"),
            "rationale": "Mixed additive Gaussian noise model.",
        },
        {
            "model": "SCORE",
            "noise": "HighGaussian",
            "mechanism": "Mixed",
            "confounding": ("NoConf", "ObsConf"),
            "rationale": "Mixed additive Gaussian noise model.",
        },
        {
            "model": "NoGAM",
            "mechanism": "Mixed",
            "confounding": ("NoConf", "ObsConf"),
            "rationale": "Mixed additive mechanisms with flexible noise.",
        },
        {
            "model": "DAS",
            "mechanism": "Mixed",
            "confounding": ("NoConf", "ObsConf"),
            "rationale": "Mixed additive mechanisms.",
        },
        {
            "model": "GraNDAG",
            "mechanism": "Mixed",
            "confounding": ("NoConf", "ObsConf"),
            "rationale": "Neural mixed mechanisms.",
        },
        {
            "model": "PC",
            "confounding": ("NoConf", "ObsConf"),
            "rationale": "Causal sufficiency: no latent confounding.",
        },
        {
            "model": "GES",
            "confounding": ("NoConf", "ObsConf"),
            "rationale": "Causal sufficiency: no latent confounding.",
        },
        {
            "model": "GRaSP",
            "confounding": ("NoConf", "ObsConf"),
            "rationale": "Causal sufficiency: no latent confounding.",
        },
        {
            "model": "FCI",
            "confounding": ("NoConf", "ObsConf", "LatConf", "Obs+Lat"),
            "rationale": "Allows observed and latent confounding.",
        },
        {
            "model": "AVICI",
            "graph_type": "EB",
            "confounding": "NoConf",
            "rationale": "In-distribution random graphs without confounding.",
        },
    ]
)
FILTER_COLUMNS = (
    "dataset",
    "n_nodes",
    "graph_type",
    "noise",
    "mechanism",
    "obs_conf",
    "lat_conf",
    "model",
    "n_samples",
    "normalized",
    "execution_platform",
)


@dataclass(frozen=True)
class FinalVisualizationData:
    """Validated inputs used by the public visualisation functions."""

    metrics: pd.DataFrame
    ranks_by_dataset: pd.DataFrame
    ranks_overall: pd.DataFrame
    coverage_by_dataset: pd.DataFrame
    friedman_by_dataset: pd.DataFrame
    paths: Mapping[str, Path]


def _project_root(start=None):
    return Path(__file__).resolve().parents[1]


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.casefold().isin({"true", "1", "yes"})


def _canonical_model_name(value: Any) -> Any:
    return MODEL_NAME_ALIASES.get(value, value)


def _model_display_label(value: Any) -> str:
    """Return the publication-facing model name without internal underscores."""
    canonical = str(_canonical_model_name(value))
    return MODEL_DISPLAY_LABELS.get(canonical, canonical.replace("_", " "))


def _canonicalize_model_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "model" in out.columns:
        out["model"] = out["model"].map(_canonical_model_name)
    return out


def _despine(axes: plt.Axes | Iterable[plt.Axes], *, font_size: float = DEFAULT_FONT_SIZE) -> None:
    """Apply the shared tick size and remove decorative frame lines."""
    tick_font_size = max(1.0, font_size - 2)
    for axis in np.atleast_1d(axes).ravel():
        axis.tick_params(axis="both", labelsize=tick_font_size)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)


def _first_int(value: Any) -> int | None:
    match = re.search("\\d+", str(value))
    return int(match.group()) if match else None


def parse_dataset_name(name: str) -> dict[str, Any]:
    """Parse a canonical synthetic dataset name into filterable attributes."""
    parts = str(name).split("_")
    if len(parts) < 6:
        return {
            "n_nodes": None,
            "graph_type": None,
            "obs_conf": None,
            "lat_conf": None,
            "noise": None,
            "mechanism": None,
        }
    return {
        "n_nodes": _first_int(parts[0]),
        "graph_type": parts[1],
        "obs_conf": _first_int(parts[2]),
        "lat_conf": _first_int(parts[3]),
        "noise": "_".join(parts[4:-1]),
        "mechanism": parts[-1],
    }


def _add_dataset_attributes(df: pd.DataFrame, dataset_col: str = "dataset") -> pd.DataFrame:
    out = df.copy()
    if dataset_col not in out.columns:
        return out
    parsed = pd.DataFrame(
        [parse_dataset_name(value) for value in out[dataset_col]], index=out.index
    )
    for col in parsed.columns:
        if col not in out.columns:
            out[col] = parsed[col]
        else:
            out[col] = out[col].where(out[col].notna(), parsed[col])
    return out


def add_fbeta(
    df: pd.DataFrame,
    *,
    beta: float = FBETA,
    precision_col: str = "precision",
    recall_col: str = "recall",
    out: str = "fbeta",
) -> pd.DataFrame:
    """Add the F-beta column derived from the stored precision and recall.

    This is not a recomputation of a graph metric: F-beta is a function of two
    columns the runs already carry, with the same convention as
    tuning_study/scoring.fbeta, so the untuned benchmark and the tuning study
    are measured identically. A run whose precision or recall is missing gets
    NaN, which the coverage panels already treat as an unsuccessful run, and a
    run with zero precision and zero recall gets 0.
    """
    if precision_col not in df.columns or recall_col not in df.columns:
        return df
    precision = pd.to_numeric(df[precision_col], errors="coerce")
    recall = pd.to_numeric(df[recall_col], errors="coerce")
    b2 = beta * beta
    denominator = b2 * precision + recall
    value = (1.0 + b2) * precision * recall / denominator
    df[out] = value.where(denominator > 0, 0.0).where(precision.notna() & recall.notna())
    return df


def _require_columns(df: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def load_final_visualization_data(
    *,
    metrics_path: str | Path | None = None,
    rank_by_dataset_path: str | Path | None = None,
    rank_overall_path: str | Path | None = None,
    coverage_by_dataset_path: str | Path | None = None,
    friedman_by_dataset_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> FinalVisualizationData:
    """Load the metric, precomputed rank and coverage CSVs.

    This function validates identities but never recomputes a graph metric,
    statistical test or Wilcoxon rank.
    """
    if all(value is None for value in (metrics_path, rank_by_dataset_path, rank_overall_path, coverage_by_dataset_path, friedman_by_dataset_path)):
        from result_tables import load_aligned_sources, visualization_data
        return visualization_data(load_aligned_sources()["benchmark_with_biological"])
    if metrics_path is None:
        raise ValueError("Provide metrics_path when loading custom result tables")
    supplied = {
        "ranks_by_dataset": rank_by_dataset_path,
        "ranks_overall": rank_overall_path,
        "coverage_by_dataset": coverage_by_dataset_path,
        "friedman_by_dataset": friedman_by_dataset_path,
    }
    missing = [label for label, path in supplied.items() if path is None]
    if missing:
        raise ValueError(f"Custom metrics require explicit summary CSV paths: {missing}")
    paths = {
        "metrics": Path(metrics_path),
        **{label: Path(path) for label, path in supplied.items()},
    }
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} CSV: {path}")
    metrics = _canonicalize_model_names(pd.read_csv(paths["metrics"], low_memory=False))
    ranks = _canonicalize_model_names(pd.read_csv(paths["ranks_by_dataset"], low_memory=False))
    ranks_overall = _canonicalize_model_names(pd.read_csv(paths["ranks_overall"], low_memory=False))
    coverage = _canonicalize_model_names(
        pd.read_csv(paths["coverage_by_dataset"], low_memory=False)
    )
    friedman = _canonicalize_model_names(
        pd.read_csv(paths["friedman_by_dataset"], low_memory=False)
    )
    _require_columns(
        metrics,
        ["dataset_seed", "model_seed", "dataset", "model", "status", "timed_out"],
        "metrics CSV",
    )
    _require_columns(ranks, ["dataset_scope", "metric", "model", "rank"], "rank-by-dataset CSV")
    _require_columns(
        ranks_overall, ["dataset_scope", "metric", "model", "rank"], "overall-rank CSV"
    )
    _require_columns(
        coverage,
        ["dataset", "metric", "model", "successful_runs", "expected_runs"],
        "coverage CSV",
    )
    _require_columns(friedman, ["dataset_scope", "metric", "model", "mean", "std"], "Friedman CSV")
    duplicate_ids = metrics.duplicated(["dataset_seed", "dataset", "model_seed", "model"])
    if duplicate_ids.any():
        raise ValueError(
            f"metrics CSV contains {int(duplicate_ids.sum())} duplicate run identities."
        )
    metrics = _add_dataset_attributes(metrics)
    metrics["timed_out"] = _as_bool(metrics["timed_out"])
    metrics["finished"] = metrics["status"].eq("finished") & ~metrics["timed_out"]
    metrics = add_fbeta(metrics)
    metrics = add_fbeta(
        metrics,
        precision_col="precision",
        recall_col="recall",
        out="fbeta",
    )
    ranks = ranks.rename(columns={"dataset_scope": "dataset"})
    ranks = _add_dataset_attributes(ranks)
    friedman = friedman.rename(columns={"dataset_scope": "dataset"})
    friedman = _add_dataset_attributes(friedman)
    coverage = _add_dataset_attributes(coverage)
    metric_datasets = set(metrics["dataset"].dropna())
    for label, frame in (
        ("rank", ranks),
        ("coverage", coverage),
        ("Friedman", friedman),
    ):
        unknown = set(frame["dataset"].dropna()) - metric_datasets
        if unknown:
            raise ValueError(
                f"{label} CSV contains datasets absent from metrics CSV: {sorted(unknown)[:5]}"
            )
    numeric_metric_candidates = set(METRIC_INFO).intersection(metrics.columns)
    for col in numeric_metric_candidates:
        metrics[col] = pd.to_numeric(metrics[col], errors="coerce")
    for frame, columns in (
        (
            ranks,
            [
                "rank",
                "tier",
                "significant_wins",
                "significant_losses",
                "non_significant_ties",
                "comparisons",
                "mean_metric",
            ],
        ),
        (
            ranks_overall,
            [
                "rank",
                "tier",
                "significant_wins",
                "significant_losses",
                "non_significant_ties",
                "comparisons",
                "mean_metric",
            ],
        ),
        (coverage, ["successful_runs", "expected_runs", "coverage_fraction"]),
        (friedman, ["mean", "std", "median", "min", "max", "count", "mean_rank"]),
    ):
        for col in columns:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return FinalVisualizationData(
        metrics=metrics,
        ranks_by_dataset=ranks,
        ranks_overall=ranks_overall,
        coverage_by_dataset=coverage,
        friedman_by_dataset=friedman,
        paths=paths,
    )


def _normalise_token(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join((ch for ch in text if not unicodedata.combining(ch)))
    return re.sub("[^a-z0-9]+", "", text.casefold())


def _as_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
        return list(value)
    return [value]


def _match_values(series: pd.Series, requested: Any, column: str) -> pd.Series:
    values = _as_list(requested)
    if values is None:
        return pd.Series(True, index=series.index)
    if pd.api.types.is_numeric_dtype(series):
        wanted = pd.to_numeric(pd.Series(values), errors="coerce").dropna().tolist()
        return series.isin(wanted)
    if pd.api.types.is_bool_dtype(series):
        wanted = [_normalise_token(v) in {"true", "1", "yes"} for v in values]
        return series.isin(wanted)
    available = pd.Index(series.dropna().astype(str).unique())
    lookup: dict[str, list[str]] = {}
    for original in available:
        lookup.setdefault(_normalise_token(original), []).append(original)
    matched: set[str] = set()
    missing: list[Any] = []
    for value in values:
        key = _normalise_token(value)
        candidates = lookup.get(key, [])
        if candidates:
            matched.update(candidates)
        else:
            missing.append(value)
    if missing:
        examples = ", ".join(map(str, sorted(available)[:12]))
        raise ValueError(
            f"Unknown value(s) {missing!r} for filter '{column}'. Available values include: {examples}"
        )
    return series.astype(str).isin(matched)


def _filter_frame(
    df: pd.DataFrame,
    *,
    datasets: Any = None,
    n_nodes: Any = None,
    graph_type: Any = None,
    noise: Any = None,
    mechanism: Any = None,
    obs_conf: Any = None,
    lat_conf: Any = None,
    models: Any = None,
    filters: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    requested: dict[str, Any] = {
        "dataset": datasets,
        "n_nodes": n_nodes,
        "graph_type": graph_type,
        "noise": noise,
        "mechanism": mechanism,
        "obs_conf": obs_conf,
        "lat_conf": lat_conf,
        "model": models,
    }
    if filters:
        requested.update(filters)
    if requested.get("model") is not None:
        requested["model"] = [
            _canonical_model_name(value) for value in _as_list(requested["model"]) or []
        ]
    out = _canonicalize_model_names(_add_dataset_attributes(df))
    for column, value in requested.items():
        if value is None or (isinstance(value, str) and value.casefold() == "all"):
            continue
        if column not in out.columns:
            raise ValueError(f"Cannot filter by '{column}'; available columns: {list(out.columns)}")
        out = out[_match_values(out[column], value, column)]
    if out.empty:
        raise ValueError("No rows remain after applying the selected filters.")
    return out


def _filter_kwargs(
    datasets: Any,
    n_nodes: Any,
    graph_type: Any,
    noise: Any,
    mechanism: Any,
    obs_conf: Any,
    lat_conf: Any,
    models: Any,
    filters: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return dict(
        datasets=datasets,
        n_nodes=n_nodes,
        graph_type=graph_type,
        noise=noise,
        mechanism=mechanism,
        obs_conf=obs_conf,
        lat_conf=lat_conf,
        models=models,
        filters=filters,
    )


def build_predicted_edge_orientation_table(
    data: FinalVisualizationData,
    *,
    datasets: Any = None,
    n_nodes: Any = None,
    graph_type: Any = None,
    noise: Any = None,
    mechanism: Any = None,
    obs_conf: Any = None,
    lat_conf: Any = None,
    models: Any = ("PC", "FCI", "GES"),
    filters: Mapping[str, Any] | None = None,
    zero_precision_oriented_counts: Mapping[tuple[int, str, str], float] | None = None,
) -> pd.DataFrame:
    """Summarise fully oriented versus other predicted adjacencies.

    The legacy benchmark CSV does not store endpoint counts directly.  Its
    classic precision and recall, however, were computed only from fully
    directed predictions.  Thus ``recall * n_true_edges / precision`` recovers
    their count whenever precision is positive.  Runs with zero precision are
    intrinsically ambiguous and must be supplied explicitly through
    ``zero_precision_oriented_counts``, keyed by
    ``(dataset_seed, dataset, model)``.

    Counts are first averaged over model seeds within each independent
    dataset block and then over dataset seeds, matching the aggregation used
    by :func:`plot_metric_and_timeouts`.
    """
    kwargs = _filter_kwargs(
        datasets,
        n_nodes,
        graph_type,
        noise,
        mechanism,
        obs_conf,
        lat_conf,
        models,
        filters,
    )
    selected = _filter_frame(data.metrics, **kwargs).copy()
    required = {
        "dataset_seed",
        "dataset",
        "model",
        "precision",
        "recall",
        "n_true_edges",
        "n_pred_edges",
    }
    missing = sorted(required.difference(selected.columns))
    if missing:
        raise ValueError(f"Missing columns required for edge orientation counts: {missing}")
    for column in ("precision", "recall", "n_true_edges", "n_pred_edges"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected[selected["finished"] & selected["n_pred_edges"].notna()].copy()
    if selected.empty:
        raise ValueError("No finished runs with predicted-edge counts remain after filtering.")
    selected["oriented_edges"] = np.where(
        selected["precision"].gt(0),
        selected["recall"] * selected["n_true_edges"] / selected["precision"],
        np.nan,
    )
    selected.loc[selected["oriented_edges"].notna(), "oriented_edges"] = selected.loc[
        selected["oriented_edges"].notna(), "oriented_edges"
    ].round()
    overrides = dict(zero_precision_oriented_counts or {})
    unresolved: list[tuple[int, str, str]] = []
    for index, row in selected[selected["oriented_edges"].isna()].iterrows():
        key = (int(row["dataset_seed"]), str(row["dataset"]), str(row["model"]))
        if key in overrides:
            selected.at[index, "oriented_edges"] = float(overrides[key])
        else:
            unresolved.append(key)
    if unresolved:
        unique = list(dict.fromkeys(unresolved))
        raise ValueError(
            f"Fully oriented edge counts cannot be inferred for zero-precision runs. Supply zero_precision_oriented_counts for: {unique}"
        )
    invalid = selected["oriented_edges"].lt(0) | selected["oriented_edges"].gt(
        selected["n_pred_edges"]
    )
    if invalid.any():
        raise ValueError("Inferred oriented-edge counts fall outside [0, n_pred_edges].")
    selected["non_oriented_edges"] = selected["n_pred_edges"] - selected["oriented_edges"]
    block_columns = ["dataset_seed", "dataset", "model"]
    blocks = (
        selected.groupby(block_columns, observed=True)
        .agg(
            predicted_edges=("n_pred_edges", "mean"),
            oriented_edges=("oriented_edges", "mean"),
            non_oriented_edges=("non_oriented_edges", "mean"),
        )
        .reset_index()
    )
    summary = (
        blocks.groupby("model", observed=True)
        .agg(
            predicted_edges=("predicted_edges", "mean"),
            oriented_edges=("oriented_edges", "mean"),
            non_oriented_edges=("non_oriented_edges", "mean"),
            n_dataset_seeds=("dataset_seed", "nunique"),
        )
        .reset_index()
    )
    summary["oriented_pct"] = np.where(
        summary["predicted_edges"].gt(0),
        100 * summary["oriented_edges"] / summary["predicted_edges"],
        0.0,
    )
    summary["non_oriented_pct"] = 100.0 - summary["oriented_pct"]
    order = _ordered_models(summary["model"], model_order=_as_list(models))
    return (
        summary.set_index("model")
        .reindex(order)
        .reset_index()[
            [
                "model",
                "predicted_edges",
                "oriented_edges",
                "non_oriented_edges",
                "oriented_pct",
                "non_oriented_pct",
                "n_dataset_seeds",
            ]
        ]
    )


def _metric_meta(metric: str, higher_is_better: bool | None = None) -> tuple[str, bool]:
    info = METRIC_INFO.get(metric, {"label": metric, "higher_is_better": True})
    return (
        info["label"],
        info["higher_is_better"] if higher_is_better is None else higher_is_better,
    )


def _coverage_from_metrics(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return coverage by dataset and aggregated over the selected scope."""
    work = df.copy()
    if metric not in work.columns:
        raise ValueError(f"Metric '{metric}' is not present in all_metrics.csv.")
    valid_metric = pd.to_numeric(work[metric], errors="coerce").notna()
    work["_successful"] = work["finished"] & valid_metric
    work["_timeout"] = work["timed_out"] | work["status"].eq("timeout")
    work["_error"] = ~work["status"].isin(["finished", "timeout"]) & ~work["timed_out"]
    by_dataset = (
        work.groupby(["dataset", "model"], observed=True)
        .agg(
            successful_runs=("_successful", "sum"),
            expected_runs=("model_seed", "size"),
            timeouts=("_timeout", "sum"),
            errors=("_error", "sum"),
        )
        .reset_index()
    )
    by_dataset["coverage_fraction"] = by_dataset["successful_runs"] / by_dataset[
        "expected_runs"
    ].clip(lower=1)
    scope = (
        by_dataset.groupby("model", observed=True)
        .agg(
            successful_runs=("successful_runs", "sum"),
            expected_runs=("expected_runs", "sum"),
            timeouts=("timeouts", "sum"),
            errors=("errors", "sum"),
            configurations_present=("dataset", "nunique"),
        )
        .reset_index()
    )
    scope["coverage_fraction"] = scope["successful_runs"] / scope["expected_runs"].clip(lower=1)
    return (by_dataset, scope)


def _eligible_models(
    by_dataset: pd.DataFrame,
    *,
    coverage_policy: str,
    min_completed_runs: int | None,
    min_coverage_fraction: float | None,
) -> set[str]:
    policy = coverage_policy.casefold()
    if policy not in {"complete", "all", "min_runs", "min_coverage"}:
        raise ValueError("coverage_policy must be complete, all, min_runs, or min_coverage.")
    if policy == "all":
        return set(by_dataset["model"])
    if policy == "complete":
        complete = by_dataset["successful_runs"].eq(by_dataset["expected_runs"])
        return set(
            by_dataset.loc[complete]
            .groupby("model")
            .filter(lambda x: len(x) == by_dataset["dataset"].nunique())["model"]
        )
    scope = by_dataset.groupby("model", observed=True).agg(
        successful_runs=("successful_runs", "sum"),
        expected_runs=("expected_runs", "sum"),
    )
    if policy == "min_runs":
        if min_completed_runs is None or min_completed_runs < 1:
            raise ValueError("min_completed_runs must be provided and >= 1 for min_runs.")
        return set(scope.index[scope["successful_runs"].ge(min_completed_runs)])
    if min_coverage_fraction is None or not 0 <= min_coverage_fraction <= 1:
        raise ValueError("min_coverage_fraction must be in [0, 1] for min_coverage.")
    fraction = scope["successful_runs"] / scope["expected_runs"].clip(lower=1)
    return set(scope.index[fraction.ge(min_coverage_fraction)])


def _ordered_models(
    models: Iterable[str],
    *,
    model_order: Sequence[str] | None = None,
    values: Mapping[str, float] | None = None,
    ascending: bool = False,
) -> list[str]:
    present = list(dict.fromkeys((str(_canonical_model_name(model)) for model in models)))
    if model_order is not None:
        requested_order = [str(_canonical_model_name(model)) for model in model_order]
        ordered = [m for m in requested_order if m in present]
        return ordered + sorted(set(present) - set(ordered))
    if values is not None:
        return sorted(
            present,
            key=lambda m: (pd.isna(values.get(m, np.nan)), values.get(m, np.nan)),
            reverse=not ascending,
        )
    known = [m for m in DEFAULT_MODEL_ORDER if m in present]
    return known + sorted(set(present) - set(known))


def _colors(models: Sequence[str], model_colors: Mapping[str, str] | None) -> list[str]:
    supplied = {
        str(_canonical_model_name(model)): color
        for model, color in dict(model_colors or {}).items()
    }
    colors = {**DEFAULT_MODEL_COLORS, **supplied}
    result = []
    for model in models:
        if model in colors:
            result.append(colors[model])
            continue
        encoded = str(model).encode("utf-8")
        index = sum(((position + 1) * value for position, value in enumerate(encoded)))
        result.append(OKABE_ITO_EXTENDED[index % len(OKABE_ITO_EXTENDED)])
    return result


def _uncertainty_summary(
    values: Sequence[float],
    *,
    uncertainty: str | None = "ci",
    confidence_level: float = 0.95,
    ci_method: str = "bootstrap",
    n_bootstrap: int = 2000,
    random_state: int | None = 42,
) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    result = {
        "mean": float(np.mean(x)) if n else np.nan,
        "median": float(np.median(x)) if n else np.nan,
        "std": float(np.std(x, ddof=1)) if n > 1 else np.nan,
        "min": float(np.min(x)) if n else np.nan,
        "max": float(np.max(x)) if n else np.nan,
        "q1": float(np.quantile(x, 0.25)) if n else np.nan,
        "q3": float(np.quantile(x, 0.75)) if n else np.nan,
        "n": n,
        "lower": np.nan,
        "upper": np.nan,
    }
    if uncertainty is None or n == 0:
        return result
    kind = uncertainty.casefold().replace("_", "")
    if kind in {"sd", "std"}:
        result["lower"], result["upper"] = (
            result["mean"] - result["std"],
            result["mean"] + result["std"],
        )
    elif kind == "sem":
        sem = result["std"] / math.sqrt(n) if n > 1 else np.nan
        result["lower"], result["upper"] = (result["mean"] - sem, result["mean"] + sem)
    elif kind in {"minmax", "range"}:
        result["lower"], result["upper"] = (result["min"], result["max"])
    elif kind == "iqr":
        result["lower"], result["upper"] = (result["q1"], result["q3"])
    elif kind in {"ci", "ci95"}:
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1.")
        if n < 2:
            return result
        alpha = 1 - confidence_level
        method = ci_method.casefold()
        sem = result["std"] / math.sqrt(n)
        if method == "t":
            critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
            result["lower"], result["upper"] = (
                result["mean"] - critical * sem,
                result["mean"] + critical * sem,
            )
        elif method == "normal":
            critical = stats.norm.ppf(1 - alpha / 2)
            result["lower"], result["upper"] = (
                result["mean"] - critical * sem,
                result["mean"] + critical * sem,
            )
        elif method == "bootstrap":
            if n_bootstrap < 100:
                raise ValueError("n_bootstrap must be at least 100.")
            rng = np.random.default_rng(random_state)
            sampled = rng.choice(x, size=(n_bootstrap, n), replace=True).mean(axis=1)
            result["lower"], result["upper"] = np.quantile(sampled, [alpha / 2, 1 - alpha / 2])
        else:
            raise ValueError("ci_method must be bootstrap, t, or normal.")
    else:
        raise ValueError("uncertainty must be ci, sd, sem, minmax, iqr, or None.")
    return result


def _tukey_outlier_mask(values: Sequence[float]) -> np.ndarray:
    """Return the boxplot/Tukey outliers outside 1.5 times the IQR."""
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    mask = np.zeros(len(x), dtype=bool)
    if not finite.any():
        return mask
    q1, q3 = np.quantile(x[finite], [0.25, 0.75])
    iqr = q3 - q1
    mask[finite] = (x[finite] < q1 - 1.5 * iqr) | (x[finite] > q3 + 1.5 * iqr)
    return mask


def _block_means(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' is not present in all_metrics.csv.")
    work = df[df["finished"]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric])
    if work.empty:
        raise ValueError(f"No finished runs contain metric '{metric}' after filtering.")
    blocks = (
        work.groupby(["dataset_seed", "dataset", "model"], observed=True)[metric]
        .agg(metric_value="mean", completed_model_seeds="size")
        .reset_index()
    )
    return _add_dataset_attributes(blocks)


def _validate_plot_alpha(value: Any) -> float:
    alpha = float(value)
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be a finite number between 0 and 1.")
    return alpha


def _apply_visual_alpha(fig: plt.Figure, alpha: float) -> None:
    """Apply a reversible opacity multiplier to the data artists in a figure."""
    alpha = _validate_plot_alpha(alpha)
    previous = float(getattr(fig, "_figure_visual_alpha", 1.0))
    if math.isclose(previous, alpha, rel_tol=0, abs_tol=1e-12):
        return
    factor = alpha / previous if previous else alpha
    artists: list[Any] = []
    for axis in fig.axes:
        artists.extend(axis.lines)
        artists.extend(axis.patches)
        artists.extend(axis.collections)
        artists.extend(axis.images)
        legend = axis.get_legend()
        if legend is not None:
            artists.extend(getattr(legend, "legend_handles", ()))
    for legend in fig.legends:
        artists.extend(getattr(legend, "legend_handles", ()))
    seen: set[int] = set()
    for artist in artists:
        if id(artist) in seen or not hasattr(artist, "set_alpha"):
            continue
        seen.add(id(artist))
        current = artist.get_alpha() if hasattr(artist, "get_alpha") else None
        base = 1.0 if current is None else float(current)
        artist.set_alpha(min(1.0, max(0.0, base * factor)))
    fig._figure_visual_alpha = alpha


def _figures_in_result(value: Any) -> list[plt.Figure]:
    """Collect figures from the standard tuples and mappings returned here."""
    found: list[plt.Figure] = []
    seen: set[int] = set()

    def visit(item: Any) -> None:
        if id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, plt.Figure):
            found.append(item)
        elif isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (tuple, list)):
            for nested in item:
                visit(nested)

    visit(value)
    return found


def _alpha_controlled_plot(function):
    """Add the shared keyword-only ``alpha`` argument to a plot function."""
    missing = object()

    @wraps(function)
    def wrapper(*args, **kwargs):
        supplied = kwargs.pop("alpha", missing)
        effective = _PLOT_ALPHA.get() if supplied is missing else _validate_plot_alpha(supplied)
        token = _PLOT_ALPHA.set(effective)
        try:
            result = function(*args, **kwargs)
            for figure in _figures_in_result(result):
                _apply_visual_alpha(figure, effective)
            return result
        finally:
            _PLOT_ALPHA.reset(token)

    signature = inspect.signature(function)
    parameters = list(signature.parameters.values())
    insert_at = next(
        (
            index
            for index, parameter in enumerate(parameters)
            if parameter.kind is inspect.Parameter.VAR_KEYWORD
        ),
        len(parameters),
    )
    parameters.insert(
        insert_at,
        inspect.Parameter(
            "alpha", kind=inspect.Parameter.KEYWORD_ONLY, default=1.0, annotation=float
        ),
    )
    wrapper.__signature__ = signature.replace(parameters=parameters)
    wrapper.__doc__ = (
        (function.__doc__ or "")
        + "\n\n    alpha : float, default 1.0\n        Opacity multiplier for the plotted data marks (0 to 1)."
    )
    return wrapper


def _save(fig: plt.Figure, save_path: str | Path | None, dpi: int, transparent: bool) -> None:
    _apply_visual_alpha(fig, _PLOT_ALPHA.get())
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, transparent=transparent, bbox_inches="tight")


def load_robustness_fits(
    path: str | Path | None = None, *, project_root: str | Path | None = None
) -> pd.DataFrame:
    """Load and validate the per-fit tuning/generalisation robustness table."""
    if path is None:
        from condition_data import load_results
        return add_fbeta(_canonicalize_model_names(load_results("evaluation")))
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Robustness fits table not found: {source}")
    fits = pd.read_csv(source)
    _require_columns(
        fits,
        ["model", "arm", "dataset", "dataset_seed", "model_seed", "status"],
        str(source),
    )
    return add_fbeta(_canonicalize_model_names(fits))


def build_tuned_stage_table(
    *,
    metric: str = "fbeta",
    include_prism: bool = True,
    results_dir: str | Path | None = None,
    dagma_results_dir: str | Path | None = None,
    test_fits: pd.DataFrame | str | Path | None = None,
    stage_fits: Mapping[str, pd.DataFrame | str | Path] | None = None,
    uncertainty: str | None = "ci",
    confidence_level: float = 0.95,
    ci_method: str = "bootstrap",
    n_bootstrap: int = 5000,
    random_state: int | None = 42,
    model_order: Sequence[str] | None = None,
    digits: int = 4,
    project_root: str | Path | None = None,
) -> pd.DataFrame:
    """Summarise the tuned models across train, validation and test.

    Stored aggregate values are always reported. Seed-level uncertainty is
    added only when per-fit rows are available: model seeds are averaged first
    inside each ``dataset x dataset_seed x model`` block, dataset families are
    then macro-averaged inside each dataset seed, and uncertainty is finally
    calculated across the independent dataset seeds. This is the same
    hierarchy used by the other final-analysis summaries.

    The current repository contains per-fit rows for test, while search/train
    values are stored in Optuna SQLite databases and validation values in
    aggregate JSON files. The databases are read directly, without requiring
    Optuna to be installed. Aggregate-only cells explicitly say
    ``error unavailable`` instead of treating the error as zero. If per-fit
    CSVs become available later, pass them through
    ``stage_fits={'train': ..., 'validation': ...}``; their seed-level error is
    then included automatically. The unformatted long summary is retained in
    ``returned_table.attrs['summary']`` for audit and export.
    """
    metric_key = str(metric).strip().casefold().replace(" ", "")
    metric = {
        "f0.5": "fbeta",
        "f05": "fbeta",
        "f_0.5": "fbeta",
        "f1": "f1",
        "fbeta": "fbeta",
    }.get(metric_key, str(metric))
    root = _project_root(project_root)
    joint_dir = (
        Path(results_dir)
        if results_dir is not None
        else root / "results/prism/tuning/predictions"
    )
    dagma_dir = (
        Path(dagma_results_dir)
        if dagma_results_dir is not None
        else root / "results/prism/tuning/predictions"
    )
    families = set(DEFAULT_ROBUSTNESS_FAMILIES[:3])
    rows: list[dict[str, Any]] = []
    winner_payloads: dict[str, dict[str, Any]] = {}

    def _read_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            warnings.warn(f"Could not read {path}: {error}", RuntimeWarning)
            return None

    def _metric_value(payload: Mapping[str, Any] | None) -> float:
        if not payload:
            return float("nan")
        metrics = payload.get("metrics") or {}
        macro = metrics.get("macro") or {}
        value = macro.get(metric, payload.get(metric))
        return float(value) if value is not None and pd.notna(value) else float("nan")

    def _dataset_seed_count(payload: Mapping[str, Any] | None) -> float:
        if not payload:
            return float("nan")
        seeds = payload.get("dataset_seeds")
        if isinstance(seeds, list):
            return float(len(seeds))
        per_family = (payload.get("coverage") or {}).get("per_family") or {}
        counts = [
            value.get("dataset_seeds_planned")
            for value in per_family.values()
            if isinstance(value, Mapping)
        ]
        counts = [value for value in counts if value is not None]
        return float(max(counts)) if counts else float("nan")

    def _append_aggregate(
        model: str,
        stage: str,
        value: float,
        payload: Mapping[str, Any] | None,
        source: str,
    ) -> None:
        rows.append(
            {
                "model": _canonical_model_name(model),
                "stage": stage,
                "mean": value,
                "lower": np.nan,
                "upper": np.nan,
                "std": np.nan,
                "sem": np.nan,
                "n_dataset_seeds": _dataset_seed_count(payload),
                "error_available": False,
                "source": source,
            }
        )

    def _optuna_trials(database: Path) -> list[dict[str, Any]]:
        """Read completed Optuna trials without requiring Optuna itself."""
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            trial_rows = connection.execute(
                "\n                SELECT t.trial_id, v.value, v.value_type\n                FROM trials AS t\n                JOIN trial_values AS v ON v.trial_id = t.trial_id\n                WHERE t.state = 'COMPLETE' AND v.objective = 0\n                ORDER BY t.number\n                "
            ).fetchall()
            trials: dict[int, dict[str, Any]] = {
                int(trial_id): {
                    "value": float(value)
                    if value is not None and value_type == "FINITE"
                    else float("nan"),
                    "params": {},
                    "user_attrs": {},
                }
                for trial_id, value, value_type in trial_rows
            }
            if not trials:
                return []
            placeholders = ",".join(("?" for _ in trials))
            for trial_id, name, internal_value, distribution_text in connection.execute(
                f"\n                SELECT trial_id, param_name, param_value, distribution_json\n                FROM trial_params\n                WHERE trial_id IN ({placeholders})\n                ",
                tuple(trials),
            ):
                distribution = json.loads(distribution_text)
                attributes = distribution.get("attributes") or {}
                if distribution.get("name") == "CategoricalDistribution":
                    choices = attributes.get("choices") or []
                    index = int(round(float(internal_value)))
                    value = choices[index]
                elif distribution.get("name") == "IntDistribution":
                    value = int(round(float(internal_value)))
                else:
                    value = float(internal_value)
                trials[int(trial_id)]["params"][str(name)] = value
            for trial_id, key, value_text in connection.execute(
                f"\n                SELECT trial_id, key, value_json\n                FROM trial_user_attributes\n                WHERE trial_id IN ({placeholders})\n                ",
                tuple(trials),
            ):
                trials[int(trial_id)]["user_attrs"][str(key)] = json.loads(value_text)
            return list(trials.values())
        finally:
            connection.close()

    for path in sorted(joint_dir.glob("winner__*__joint.json")):
        payload = _read_json(path)
        if not isinstance(payload, dict) or "model" not in payload:
            continue
        model = str(_canonical_model_name(payload["model"]))
        winner_payloads[model] = payload
        _append_aggregate(model, "validation", _metric_value(payload), payload, str(path))
    for path in sorted(joint_dir.glob("test__*__joint.json")):
        payload = _read_json(path)
        if not isinstance(payload, dict) or "model" not in payload:
            continue
        _append_aggregate(str(payload["model"]), "test", _metric_value(payload), payload, str(path))
    for model, winner in winner_payloads.items():
        train_value = float("nan")
        train_payload: Mapping[str, Any] | None = None
        source = "unavailable"
        enumerated_path = joint_dir / f"enumerated__{model}__joint.json"
        if enumerated_path.exists():
            candidates = _read_json(enumerated_path)
            if isinstance(candidates, list):
                match = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate.get("params") == winner.get("params")
                    ),
                    None,
                )
                if isinstance(match, dict):
                    train_value = _metric_value(match)
                    train_payload = match
                    source = str(enumerated_path)
        if not np.isfinite(train_value):
            study_dir = root / "results/prism/tuning/predictions"
            databases = sorted(study_dir.glob(f"{model}__joint*.db"))
            if databases:
                try:
                    for database in databases:
                        winner_params = winner.get("params") or {}
                        if model == "GOLEM":
                            expects_equal_variances = "__ev" in database.stem
                            if (
                                bool(winner_params.get("equal_variances"))
                                != expects_equal_variances
                            ):
                                continue

                        def _same_parameter(value: Any, expected: Any) -> bool:
                            if isinstance(value, (int, float)) and isinstance(
                                expected, (int, float)
                            ):
                                return bool(
                                    np.isclose(float(value), float(expected), equal_nan=True)
                                )
                            return value == expected

                        for trial in _optuna_trials(database):
                            trial_params = trial["params"]
                            matches_winner = bool(trial_params) and all(
                                (
                                    key in winner_params
                                    and _same_parameter(value, winner_params[key])
                                    for key, value in trial_params.items()
                                )
                            )
                            if not np.isfinite(trial["value"]) or not matches_winner:
                                continue
                            trial_metrics = trial["user_attrs"].get("metrics") or {}
                            if metric == "fbeta":
                                train_value = float(trial["value"])
                            else:
                                train_value = float(
                                    (trial_metrics.get("macro") or {}).get(metric, float("nan"))
                                )
                            train_payload = {"coverage": trial["user_attrs"].get("coverage") or {}}
                            source = str(database)
                            break
                        if np.isfinite(train_value):
                            break
                except (OSError, ValueError, sqlite3.Error) as error:
                    warnings.warn(
                        f"Could not read search study for {model}: {error}",
                        RuntimeWarning,
                    )
        _append_aggregate(model, "train", train_value, train_payload, source)
    if include_prism:
        dagma_winner = _read_json(dagma_dir / "validation" / "winner.json")
        dagma_test = _read_json(dagma_dir / "test" / "test.json")
        if isinstance(dagma_winner, dict):
            _append_aggregate(
                "PRISM",
                "validation",
                _metric_value(dagma_winner),
                dagma_winner,
                str(dagma_dir / "validation" / "winner.json"),
            )
            search_value = dagma_winner.get("search_value") if metric == "fbeta" else None
            _append_aggregate(
                "PRISM",
                "train",
                float(search_value) if search_value is not None else float("nan"),
                None,
                str(dagma_dir / "validation" / "winner.json"),
            )
        if isinstance(dagma_test, dict):
            _append_aggregate(
                "PRISM",
                "test",
                _metric_value(dagma_test),
                dagma_test,
                str(dagma_dir / "test" / "test.json"),
            )
    per_fit_sources = dict(stage_fits or {})
    if "test" not in per_fit_sources:
        per_fit_sources["test"] = (
            test_fits
            if test_fits is not None
            else joint_dir / "generalisation_analysis_robustez" / "fits_long.csv"
        )
    for stage, raw_source in per_fit_sources.items():
        stage_key = str(stage).casefold()
        if stage_key not in {"train", "validation", "test"}:
            raise ValueError("stage_fits keys must be train, validation, or test.")
        if isinstance(raw_source, pd.DataFrame):
            frame = raw_source.copy()
            source_name = "provided dataframe"
        else:
            path = Path(raw_source)
            if not path.exists():
                continue
            frame = pd.read_csv(path, low_memory=False)
            source_name = str(path)
        frame = _canonicalize_model_names(frame)
        if "arm" in frame.columns:
            frame = frame[frame["arm"].eq("tuned")]
        if metric == "fbeta":
            frame = add_fbeta(frame)
        _require_columns(
            frame,
            ["model", "dataset", "dataset_seed", "model_seed", "status", metric],
            source_name,
        )
        frame = frame[frame["dataset"].isin(families)].copy()
        values = pd.to_numeric(frame[metric], errors="coerce")
        frame = frame[frame["status"].eq("finished") & values.notna()].copy()
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        family_blocks = (
            frame.groupby(["model", "dataset_seed", "dataset"], observed=True)[metric]
            .mean()
            .reset_index()
        )
        seed_blocks = (
            family_blocks.groupby(["model", "dataset_seed"], observed=True)[metric]
            .mean()
            .reset_index()
        )
        for model, group in seed_blocks.groupby("model", observed=True):
            summary = _uncertainty_summary(
                group[metric],
                uncertainty=uncertainty,
                confidence_level=confidence_level,
                ci_method=ci_method,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )
            rows = [
                row for row in rows if not (row["model"] == model and row["stage"] == stage_key)
            ]
            rows.append(
                {
                    "model": model,
                    "stage": stage_key,
                    "mean": summary["mean"],
                    "lower": summary["lower"],
                    "upper": summary["upper"],
                    "std": summary["std"],
                    "sem": summary["std"] / math.sqrt(summary["n"]) if summary["n"] > 1 else np.nan,
                    "n_dataset_seeds": summary["n"],
                    "error_available": uncertainty is not None and summary["n"] > 1,
                    "source": source_name,
                }
            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("No tuned train/validation/test results were found.")
    if not include_prism:
        summary = summary[summary["model"].ne("PRISM")]
    models = _ordered_models(summary["model"].unique(), model_order=model_order)
    stage_labels = {"train": "Train", "validation": "Validation", "test": "Test"}

    def _format_cell(record: pd.Series | None) -> str:
        if record is None or not np.isfinite(record["mean"]):
            return "—"
        mean = f"{record['mean']:.{digits}f}"
        if (
            uncertainty is not None
            and uncertainty.casefold().replace("_", "") in {"sd", "std"}
            and bool(record["error_available"])
            and np.isfinite(record["std"])
        ):
            return f"{mean} ± {record['std']:.{digits}f} SD (n={int(record['n_dataset_seeds'])})"
        if (
            bool(record["error_available"])
            and np.isfinite(record["lower"])
            and np.isfinite(record["upper"])
        ):
            return f"{mean} [{record['lower']:.{digits}f}, {record['upper']:.{digits}f}] (n={int(record['n_dataset_seeds'])})"
        n_value = record["n_dataset_seeds"]
        suffix = f"; n={int(n_value)}" if pd.notna(n_value) else ""
        return f"{mean} (error unavailable{suffix})"

    table_rows = []
    for model in models:
        row: dict[str, str] = {"Model": model}
        for stage, label in stage_labels.items():
            selected = summary[summary["model"].eq(model) & summary["stage"].eq(stage)]
            row[label] = _format_cell(selected.iloc[0] if not selected.empty else None)
        table_rows.append(row)
    table = pd.DataFrame(table_rows)
    table.attrs["summary"] = summary.sort_values(["stage", "model"]).reset_index(drop=True)
    table.attrs["metric"] = metric
    table.attrs["uncertainty"] = uncertainty
    table.attrs["confidence_level"] = confidence_level
    return table


def plot_family_robustness(
    fits: pd.DataFrame | str | Path,
    *,
    metric: str = "f1",
    arms: Sequence[str] | str = ("untuned", "tuned"),
    families: Sequence[str] | str | None = DEFAULT_ROBUSTNESS_FAMILIES,
    dataset_seeds: Sequence[int] | int | None = COMMON_ROBUSTNESS_DATASET_SEEDS,
    models: Sequence[str] | str | None = DEFAULT_ROBUSTNESS_MODELS,
    include_prism: bool = True,
    model_order: Sequence[str] | None = None,
    order_by: str = "mean",
    higher_is_better: bool | None = None,
    uncertainty: str | None = "iqr",
    confidence_level: float = 0.95,
    ci_method: str = "bootstrap",
    n_bootstrap: int = 2000,
    random_state: int | None = 42,
    highlight_model: str | None = "PRISM",
    model_colors: Mapping[str, str] | None = None,
    arm_colors: Mapping[str, str] | None = None,
    arm_markers: Mapping[str, str] | None = None,
    highlight_color: str = DEFAULT_MODEL_COLORS["PRISM"],
    highlight_marker: str = "D",
    family_labels: Mapping[str, str] | None = None,
    title_template: str = "{family}",
    show_title: bool = True,
    show_metric_title: bool = True,
    model_axis: str = "x",
    xlabel: str = "",
    ylabel: str | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    marker_size: float = 4.5,
    show_fliers: bool = True,
    capsize: float = 2.5,
    arm_offset: float = 0.2,
    errorbar_alpha: float = 0.45,
    errorbar_linewidth: float = 0.8,
    wrap_model_labels: bool = True,
    tick_rotation: float = 30,
    show_model_bands: bool = True,
    model_band_color: str = OKABE_ITO["black"],
    model_band_alpha: float = 0.045,
    show_legend: bool = True,
    legend_title: str | None = None,
    grid: bool = True,
    despine: bool = True,
    save_dir: str | Path | None = None,
    filename_template: str = "robustness_{family}_{metric}.png",
    dpi: int = 300,
    transparent: bool = False,
) -> dict[str, tuple[plt.Figure, plt.Axes, dict[str, pd.DataFrame]]]:
    """Compare tuned and untuned results in one robustness plot per family.

    Only ``status == 'finished'`` fits are used.  The metric is first averaged
    over model seeds inside each ``arm x dataset x dataset_seed x model`` block.
    Point centres and uncertainty are then calculated across those independent
    dataset-seed means. Tuned and untuned points are offset side by side;
    ``arm_offset`` controls their separation. Error bars can be softened with
    ``errorbar_alpha`` and ``errorbar_linewidth``. Long underscore-separated
    model names are wrapped over two lines by default, and alternating model
    bands can be disabled with ``show_model_bands=False``. With
    ``show_fliers=True``, open circles mark dataset-seed means outside the
    Tukey boxplot fences ``Q1 - 1.5 IQR`` and ``Q3 + 1.5 IQR``.
    Set ``model_axis='y'`` for a horizontal layout with models on the y-axis;
    the default ``'x'`` preserves the vertical layout. ``ylim`` always limits
    the metric axis, so it is applied as an x-limit in the horizontal layout.
    Likewise, ``xlabel`` labels the model axis and ``ylabel`` labels the metric
    axis, irrespective of the selected orientation.
    ``uncertainty`` accepts ``'ci'``, ``'sd'``, ``'sem'``, ``'minmax'``,
    ``'iqr'`` or ``None``; confidence intervals additionally use
    ``confidence_level``, ``ci_method`` and ``n_bootstrap``. The default model
    set contains the 14 benchmark algorithms plus ``PRISM`` and excludes
    the random baseline. Set ``include_prism=False`` to remove
    ``PRISM`` from every family, even when it appears in ``models``; its
    default highlight is disabled automatically.

    The returned dictionary is keyed by exact family name.  Each value contains
    its figure, axis and the plotted ``seed_means``, ``summary`` and shared
    ``model_order`` tables.
    """
    model_axis = str(model_axis).casefold()
    if model_axis not in {"x", "y"}:
        raise ValueError("model_axis must be 'x' or 'y'.")
    source_name = "robustness fits"
    if isinstance(fits, (str, Path)):
        source_name = str(fits)
        fits = load_robustness_fits(fits)
    else:
        fits = _canonicalize_model_names(fits)
    if metric == "fbeta":
        fits = add_fbeta(fits)
    _require_columns(
        fits,
        ["model", "arm", "dataset", "dataset_seed", "model_seed", "status", metric],
        source_name,
    )
    highlight_model = _canonical_model_name(highlight_model)
    if not include_prism and highlight_model == "PRISM":
        highlight_model = None
    if models is not None:
        models = [_canonical_model_name(value) for value in _as_list(models) or []]
        if not include_prism:
            models = [model for model in models if model != "PRISM"]
    selected = fits[fits["status"].astype(str).str.casefold().eq("finished")].copy()
    selected = selected[_match_values(selected["arm"], arms, "arm")]
    if families is not None:
        selected = selected[_match_values(selected["dataset"], families, "dataset")]
    if dataset_seeds is not None:
        selected = selected[_match_values(selected["dataset_seed"], dataset_seeds, "dataset_seed")]
    if models is not None:
        selected = selected[_match_values(selected["model"], models, "model")]
    if not include_prism:
        selected = selected[selected["model"].ne("PRISM")]
    selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
    selected = selected.dropna(subset=[metric])
    if selected.empty:
        raise ValueError("No finished robustness fits contain the selected metric and filters.")
    selected_families = list(
        dict.fromkeys(map(str, _as_list(families) or selected["dataset"].unique()))
    )
    present_families = set(selected["dataset"].astype(str))
    missing_families = [family for family in selected_families if family not in present_families]
    if missing_families:
        raise ValueError(f"No finished robustness fits for families: {missing_families}")
    seed_means = (
        selected.groupby(["arm", "dataset", "dataset_seed", "model"], observed=True)[metric]
        .agg(metric_value="mean", completed_model_seeds="size")
        .reset_index()
    )
    summary_rows: list[dict[str, Any]] = []
    for (arm, family, model), group in seed_means.groupby(
        ["arm", "dataset", "model"], observed=True
    ):
        summary_rows.append(
            {
                "arm": arm,
                "dataset": family,
                "model": model,
                **_uncertainty_summary(
                    group["metric_value"],
                    uncertainty=uncertainty,
                    confidence_level=confidence_level,
                    ci_method=ci_method,
                    n_bootstrap=n_bootstrap,
                    random_state=random_state,
                ),
                "min_completed_model_seeds": int(group["completed_model_seeds"].min()),
                "max_completed_model_seeds": int(group["completed_model_seeds"].max()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    outlier_frames: list[pd.DataFrame] = []
    for _, group in seed_means.groupby(["arm", "dataset", "model"], observed=True):
        mask = _tukey_outlier_mask(group["metric_value"])
        if mask.any():
            outlier_frames.append(group.loc[mask].copy())
    outliers = (
        pd.concat(outlier_frames, ignore_index=True)
        if outlier_frames
        else seed_means.iloc[0:0].copy()
    )
    _, higher = _metric_meta(metric, higher_is_better)
    overall_means = seed_means.groupby("model", observed=True)["metric_value"].mean().to_dict()
    if order_by == "mean":
        shared_order = _ordered_models(
            seed_means["model"],
            model_order=model_order,
            values=overall_means,
            ascending=not higher,
        )
    elif order_by in set(seed_means["arm"].astype(str)):
        arm_means = (
            seed_means[seed_means["arm"].astype(str).eq(order_by)]
            .groupby("model", observed=True)["metric_value"]
            .mean()
            .to_dict()
        )
        shared_order = _ordered_models(
            seed_means["model"],
            model_order=model_order,
            values=arm_means,
            ascending=not higher,
        )
    elif order_by in {"default", "manual"}:
        shared_order = _ordered_models(seed_means["model"], model_order=model_order)
    else:
        available_arms = sorted(set(seed_means["arm"].astype(str)))
        raise ValueError(
            f"order_by must be 'mean', 'default', 'manual', or one of the selected arms: {available_arms}."
        )
    if highlight_model is not None and highlight_model not in shared_order:
        raise ValueError(
            f"highlight_model '{highlight_model}' is not present after filtering. Available models: {shared_order}"
        )
    metric_label, _ = _metric_meta(metric, higher_is_better)
    labels = {
        str(key): _dataset_display_title(value) for key, value in dict(family_labels or {}).items()
    }
    selected_arms = list(dict.fromkeys(map(str, _as_list(arms) or [])))
    arm_legend_colors = {
        "untuned": "#8C8C8C",
        "tuned": OKABE_ITO["blue"],
        **dict(arm_colors or {}),
    }
    markers = {"untuned": "o", "tuned": "s", **dict(arm_markers or {})}
    outputs: dict[str, tuple[plt.Figure, plt.Axes, dict[str, pd.DataFrame]]] = {}
    for family in selected_families:
        family_summary = summary[summary["dataset"].astype(str).eq(family)].copy()
        present_models = set(family_summary["model"].astype(str))
        missing_models = [model for model in shared_order if model not in present_models]
        if missing_models:
            raise ValueError(
                f"Family '{family}' has no finished '{metric}' value in either arm for models: {missing_models}"
            )
        if figsize is None:
            family_figsize = (
                (max(10, 0.72 * len(shared_order)), 5.5)
                if model_axis == "x"
                else (7.5, max(5.5, 0.48 * len(shared_order)))
            )
        else:
            family_figsize = figsize
        fig, ax = plt.subplots(figsize=family_figsize)
        model_positions = np.arange(len(shared_order), dtype=float)
        if show_model_bands:
            for index in range(0, len(shared_order), 2):
                band = ax.axvspan if model_axis == "x" else ax.axhspan
                band(
                    index - 0.5,
                    index + 0.5,
                    color=model_band_color,
                    alpha=model_band_alpha,
                    linewidth=0,
                    zorder=0,
                )
        offsets = (
            np.linspace(-arm_offset, arm_offset, len(selected_arms))
            if len(selected_arms) > 1
            else np.array([0.0])
        )
        for arm, offset in zip(selected_arms, offsets):
            rows = (
                family_summary[family_summary["arm"].astype(str).eq(arm)]
                .set_index("model")
                .reindex(shared_order)
            )
            means = rows["mean"].to_numpy(dtype=float)
            available = np.isfinite(means)
            yerr_all = None
            if uncertainty is not None:
                lower = rows["lower"].to_numpy(dtype=float)
                upper = rows["upper"].to_numpy(dtype=float)
                yerr_all = np.vstack(
                    [np.maximum(0.0, means - lower), np.maximum(0.0, upper - means)]
                )
            for model_index, model in enumerate(shared_order):
                if not available[model_index]:
                    continue
                is_highlighted = model == highlight_model
                colour = (
                    highlight_color if is_highlighted else arm_legend_colors.get(arm, "#4D4D4D")
                )
                marker = highlight_marker if is_highlighted else markers.get(arm, "o")
                point_error = (
                    yerr_all[:, model_index : model_index + 1] if yerr_all is not None else None
                )
                model_value = model_positions[model_index] + offset
                errorbar = ax.errorbar(
                    model_value if model_axis == "x" else means[model_index],
                    means[model_index] if model_axis == "x" else model_value,
                    yerr=point_error if model_axis == "x" else None,
                    xerr=point_error if model_axis == "y" else None,
                    fmt=marker,
                    linestyle="none",
                    color=colour,
                    ecolor=colour,
                    markerfacecolor=colour,
                    markersize=marker_size + (1 if is_highlighted else 0),
                    capsize=capsize,
                    elinewidth=errorbar_linewidth,
                    markeredgewidth=0.6,
                    label=_model_display_label(highlight_model) if is_highlighted else arm,
                    zorder=4 if is_highlighted else 3,
                )
                if point_error is not None:
                    for cap in errorbar[1]:
                        cap.set_alpha(errorbar_alpha)
                    for bars in errorbar[2]:
                        bars.set_alpha(errorbar_alpha)
            if show_fliers:
                arm_outliers = outliers.loc[
                    outliers["dataset"].astype(str).eq(family) & outliers["arm"].astype(str).eq(arm)
                ]
                for model_index, model in enumerate(shared_order):
                    values = arm_outliers.loc[
                        arm_outliers["model"].astype(str).eq(model), "metric_value"
                    ].to_numpy(float)
                    if not len(values):
                        continue
                    color = (
                        highlight_color
                        if model == highlight_model
                        else arm_legend_colors.get(arm, "#4D4D4D")
                    )
                    model_position = model_positions[model_index] + offset
                    ax.scatter(
                        np.full(len(values), model_position) if model_axis == "x" else values,
                        values if model_axis == "x" else np.full(len(values), model_position),
                        s=(marker_size + 1.0) ** 2,
                        marker="o",
                        facecolors="none",
                        edgecolors=color,
                        linewidths=0.9,
                        alpha=0.85,
                        zorder=5,
                    )
        tick_labels = [
            (
                model.replace("_", " ")
                if model.startswith("DAGMA_")
                else _model_display_label(model)
            ).replace(" ", "\n", 1)
            if wrap_model_labels
            else _model_display_label(model)
            for model in shared_order
        ]
        if model_axis == "x":
            ax.set_xticks(model_positions, tick_labels, rotation=tick_rotation, ha="right")
            ax.set_xlabel(xlabel, fontsize=font_size)
            ax.set_ylabel(ylabel or metric_label, fontsize=font_size)
        else:
            ax.set_yticks(model_positions, tick_labels)
            ax.set_xlabel(ylabel or metric_label, fontsize=font_size)
            ax.set_ylabel(xlabel, fontsize=font_size)
            ax.invert_yaxis()
        if show_title and show_metric_title:
            ax.set_title(
                title_template.format(
                    family=_dataset_display_title(labels.get(family, family)),
                    dataset=_dataset_display_title(family),
                    metric=metric_label,
                ),
                fontsize=title_size,
            )
        ax.tick_params(labelsize=max(1.0, font_size - 2))
        metric_axis = "y" if model_axis == "x" else "x"
        model_grid_axis = "x" if model_axis == "x" else "y"
        ax.grid(axis=metric_axis, alpha=0.25, visible=grid)
        ax.grid(axis=model_grid_axis, visible=False)
        ax.set_axisbelow(True)
        if ylim is not None:
            (ax.set_ylim if model_axis == "x" else ax.set_xlim)(*ylim)
        if show_legend:
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker=markers.get(arm, "o"),
                    linestyle="none",
                    color=arm_legend_colors.get(arm, "#4D4D4D"),
                    markerfacecolor="white"
                    if arm == "untuned"
                    else arm_legend_colors.get(arm, "#4D4D4D"),
                    label=arm,
                )
                for arm in selected_arms
            ]
            ax.legend(
                legend_handles,
                selected_arms,
                title=legend_title,
                fontsize=max(6, font_size - 2),
                markerscale=0.9,
                handlelength=1.2,
                handletextpad=0.6,
                borderpad=0.5,
                labelspacing=0.45,
            )
        if despine:
            _despine(ax, font_size=font_size)
        fig.tight_layout()
        if save_dir is not None:
            safe_family = re.sub("[^A-Za-z0-9_.-]+", "_", family).strip("_")
            filename = filename_template.format(
                family=safe_family, dataset=safe_family, metric=metric
            )
            _save(fig, Path(save_dir) / filename, dpi, transparent)
        family_seed_means = seed_means[seed_means["dataset"].astype(str).eq(family)].copy()
        outputs[family] = (
            fig,
            ax,
            {
                "seed_means": family_seed_means.reset_index(drop=True),
                "summary": family_summary.reset_index(drop=True),
                "outliers": outliers[outliers["dataset"].astype(str).eq(family)].reset_index(
                    drop=True
                ),
                "model_order": pd.DataFrame({"model": shared_order}),
            },
        )
    return outputs


def plot_metric_and_timeouts(
    data: FinalVisualizationData,
    *,
    metric: str = "f1",
    datasets: Any = None,
    n_nodes: Any = None,
    graph_type: Any = None,
    noise: Any = None,
    mechanism: Any = None,
    obs_conf: Any = None,
    lat_conf: Any = None,
    models: Any = None,
    filters: Mapping[str, Any] | None = None,
    split_by: str | None = None,
    plot_kind: str = "violin",
    coverage_policy: str = "complete",
    min_completed_runs: int | None = None,
    min_coverage_fraction: float | None = None,
    coverage_display: str = "timeouts",
    show_coverage: bool = True,
    show_timeouts: bool | None = None,
    show_excluded_in_coverage: bool = True,
    show_mean: bool = True,
    show_fliers: bool = True,
    whiskers: str | tuple[float, float] = "1.5iqr",
    show_points: bool = False,
    show_mean_ci: bool = False,
    violin_inner: str | None = "box",
    uncertainty: str | None = "iqr",
    confidence_level: float = 0.95,
    ci_method: str = "bootstrap",
    n_bootstrap: int = 2000,
    random_state: int | None = 42,
    higher_is_better: bool | None = None,
    order_by: str = "mean",
    model_order: Sequence[str] | None = None,
    model_colors: Mapping[str, str] | None = None,
    model_axis: str = "y",
    highlight_model: str | None = "PRISM",
    highlight_color: str = DEFAULT_MODEL_COLORS["PRISM"],
    highlight_marker: str = "D",
    summary_color: str | None = None,
    summary_marker: str = "o",
    marker_size: float = 4.5,
    capsize: float = 2.5,
    errorbar_alpha: float = 0.45,
    errorbar_linewidth: float = 0.8,
    wrap_model_labels: bool = True,
    tick_rotation: float = 30,
    show_model_bands: bool = False,
    model_band_color: str = OKABE_ITO["black"],
    model_band_alpha: float = 0.045,
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
    metric_title: str | None = None,
    show_metric_title: bool = True,
    coverage_title: str | None = None,
    show_title: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend_title: str | None = None,
    legend_labels: Mapping[str, str] | None = None,
    show_legend: bool = True,
    figsize: tuple[float, float] | None = None,
    axes: Sequence[plt.Axes] | plt.Axes | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    grid: bool = True,
    despine: bool = True,
    apply_layout: bool = True,
    save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> (
    tuple[plt.Figure, np.ndarray, dict[str, pd.DataFrame]]
    | dict[Any, tuple[plt.Figure, np.ndarray, dict[str, pd.DataFrame]]]
):
    """Plot a model metric distribution and an optional timeout panel.

    Each plotted metric value is a mean over model seeds within one
    ``dataset_seed x dataset x model`` block.  With ``split_by`` the function
    returns one independent figure tuple per selected value. Set
    ``show_timeouts=False`` to return only the metric plot. ``show_coverage``
    is retained as the backwards-compatible name for the same panel. Set
    ``show_metric_title=False`` to omit the metric-panel title and
    ``despine=True`` to hide the top and right axis spines.

    ``plot_kind='point'`` (also accepted as ``'errorbar'`` or ``'summary'``)
    uses the same point-and-uncertainty visual grammar as
    :func:`plot_family_robustness`: one mean per model, uncertainty caps,
    alternating model bands and an optional highlighted model. By default,
    every model retains its colour from ``model_colors``; set
    ``summary_color`` to force a common colour. Use
    ``model_axis='y'``, ``show_model_bands=True`` and the matching marker,
    cap and error-bar arguments to make final-benchmark and tuning figures
    visually identical. With ``show_fliers=True``, open circles use the same
    ``1.5 IQR`` outlier rule as a standard boxplot. ``ylim`` always limits the metric axis, including
    when models are displayed on the y-axis.
    """
    model_axis = str(model_axis).casefold()
    if model_axis not in {"x", "y"}:
        raise ValueError("model_axis must be 'x' or 'y'.")
    highlight_model = _canonical_model_name(highlight_model)
    if show_timeouts is not None:
        if not isinstance(show_timeouts, (bool, np.bool_)):
            raise TypeError("show_timeouts must be True, False, or None.")
        show_coverage = bool(show_timeouts)
    kwargs = _filter_kwargs(
        datasets,
        n_nodes,
        graph_type,
        noise,
        mechanism,
        obs_conf,
        lat_conf,
        models,
        filters,
    )
    selected = _filter_frame(data.metrics, **kwargs)
    if split_by is not None:
        if split_by not in selected.columns:
            raise ValueError(f"Unknown split_by column '{split_by}'.")
        outputs = {}
        for value in sorted(selected[split_by].dropna().unique(), key=str):
            split_filters = dict(filters or {})
            split_filters[split_by] = value
            outputs[value] = plot_metric_and_timeouts(
                data,
                metric=metric,
                datasets=datasets,
                n_nodes=n_nodes,
                graph_type=graph_type,
                noise=noise,
                mechanism=mechanism,
                obs_conf=obs_conf,
                lat_conf=lat_conf,
                models=models,
                filters=split_filters,
                split_by=None,
                plot_kind=plot_kind,
                coverage_policy=coverage_policy,
                min_completed_runs=min_completed_runs,
                min_coverage_fraction=min_coverage_fraction,
                coverage_display=coverage_display,
                show_coverage=show_coverage,
                show_excluded_in_coverage=show_excluded_in_coverage,
                show_mean=show_mean,
                show_fliers=show_fliers,
                whiskers=whiskers,
                show_points=show_points,
                show_mean_ci=show_mean_ci,
                violin_inner=violin_inner,
                uncertainty=uncertainty,
                confidence_level=confidence_level,
                ci_method=ci_method,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
                higher_is_better=higher_is_better,
                order_by=order_by,
                model_order=model_order,
                model_colors=model_colors,
                model_axis=model_axis,
                highlight_model=highlight_model,
                highlight_color=highlight_color,
                highlight_marker=highlight_marker,
                summary_color=summary_color,
                summary_marker=summary_marker,
                marker_size=marker_size,
                capsize=capsize,
                errorbar_alpha=errorbar_alpha,
                errorbar_linewidth=errorbar_linewidth,
                wrap_model_labels=wrap_model_labels,
                tick_rotation=tick_rotation,
                show_model_bands=show_model_bands,
                model_band_color=model_band_color,
                model_band_alpha=model_band_alpha,
                ylim=ylim,
                title=title,
                metric_title=metric_title,
                show_metric_title=show_metric_title,
                coverage_title=coverage_title,
                show_title=show_title,
                xlabel=xlabel,
                ylabel=ylabel,
                legend_title=legend_title,
                legend_labels=legend_labels,
                show_legend=show_legend,
                figsize=figsize,
                font_size=font_size,
                title_size=title_size,
                grid=grid,
                despine=despine,
                dpi=dpi,
                apply_layout=apply_layout,
                transparent=transparent,
            )
        return outputs
    by_dataset_coverage, scope_coverage = _coverage_from_metrics(selected, metric)
    eligible = _eligible_models(
        by_dataset_coverage,
        coverage_policy=coverage_policy,
        min_completed_runs=min_completed_runs,
        min_coverage_fraction=min_coverage_fraction,
    )
    blocks = _block_means(selected, metric)
    plotted = blocks[blocks["model"].isin(eligible)].copy()
    if plotted.empty:
        raise ValueError("No models satisfy the selected coverage policy.")
    label, higher = _metric_meta(metric, higher_is_better)
    model_means = plotted.groupby("model", observed=True)["metric_value"].mean().to_dict()
    if order_by == "mean":
        order = _ordered_models(
            plotted["model"],
            model_order=model_order,
            values=model_means,
            ascending=not higher,
        )
    elif order_by in {"default", "manual"}:
        order = _ordered_models(plotted["model"], model_order=model_order)
    elif order_by == "coverage":
        cov_values = scope_coverage.set_index("model")["coverage_fraction"].to_dict()
        order = _ordered_models(
            plotted["model"],
            model_order=model_order,
            values=cov_values,
            ascending=False,
        )
    else:
        raise ValueError("order_by must be mean, coverage, default, or manual.")
    n_axes = 2 if show_coverage else 1
    if axes is None:
        if figsize is None:
            if model_axis == "y":
                figsize = (14 if show_coverage else 9, max(5, 0.43 * len(order) + 1.8))
            else:
                figsize = (max(9, 0.72 * len(order)), 5.5)
        fig, made_axes = plt.subplots(
            1,
            n_axes,
            figsize=figsize,
            squeeze=False,
            gridspec_kw={"width_ratios": [2.2, 1] if show_coverage else [1]},
        )
        axes_arr = made_axes.ravel()
    else:
        axes_arr = np.atleast_1d(axes)
        if len(axes_arr) != n_axes:
            raise ValueError(f"Expected {n_axes} axes, received {len(axes_arr)}.")
        fig = axes_arr[0].figure
    ax = axes_arr[0]
    arrays = [plotted.loc[plotted["model"].eq(m), "metric_value"].to_numpy() for m in order]
    colors = _colors(order, model_colors)
    positions = np.arange(1, len(order) + 1)
    tick_labels = [
        _model_display_label(model).replace(" ", "\n", 1)
        if wrap_model_labels
        else _model_display_label(model)
        for model in order
    ]
    if show_model_bands:
        band = ax.axhspan if model_axis == "y" else ax.axvspan
        for index in range(0, len(order), 2):
            centre = positions[index]
            band(
                centre - 0.5,
                centre + 0.5,
                color=model_band_color,
                alpha=model_band_alpha,
                linewidth=0,
                zorder=0,
            )
    kind = plot_kind.casefold()
    if kind == "box":
        whis = (0, 100) if whiskers == "minmax" else 1.5 if whiskers == "1.5iqr" else whiskers
        artists = ax.boxplot(
            arrays,
            orientation="horizontal" if model_axis == "y" else "vertical",
            positions=positions,
            tick_labels=tick_labels,
            patch_artist=True,
            showmeans=show_mean,
            showfliers=show_fliers,
            whis=whis,
            medianprops={"color": "#111827", "linewidth": 1.2},
            meanprops={
                "marker": "D",
                "markerfacecolor": "#111827",
                "markeredgecolor": "#111827",
                "markersize": 3,
            },
        )
        for patch, color in zip(artists["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
    elif kind == "violin":
        too_small = [model for model, values in zip(order, arrays) if len(values) < 2]
        if too_small:
            raise ValueError(
                f"A violin plot needs at least two independent dataset blocks per model. Insufficient data for: {too_small}"
            )
        inner = None if violin_inner is None else violin_inner.casefold()
        if inner not in {None, "box"}:
            raise ValueError("violin_inner must be 'box' or None.")
        artists = ax.violinplot(
            arrays,
            positions=positions,
            orientation="horizontal" if model_axis == "y" else "vertical",
            showmeans=False,
            showmedians=inner is None,
            showextrema=inner is None,
        )
        for body, color in zip(artists["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.55)
        if inner == "box":
            whis = (0, 100) if whiskers == "minmax" else 1.5 if whiskers == "1.5iqr" else whiskers
            inner_artists = ax.boxplot(
                arrays,
                orientation="horizontal" if model_axis == "y" else "vertical",
                positions=positions,
                widths=0.16,
                patch_artist=True,
                showmeans=show_mean,
                showfliers=show_fliers,
                whis=whis,
                manage_ticks=False,
                boxprops={
                    "facecolor": "white",
                    "edgecolor": "#111827",
                    "linewidth": 1.0,
                },
                whiskerprops={"color": "#111827", "linewidth": 0.9},
                capprops={"color": "#111827", "linewidth": 0.9},
                medianprops={"color": "#111827", "linewidth": 1.4},
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "#111827",
                    "markeredgecolor": "#111827",
                    "markersize": 3,
                },
                flierprops={
                    "marker": "o",
                    "markerfacecolor": "#111827",
                    "markeredgecolor": "none",
                    "markersize": 2.5,
                    "alpha": 0.45,
                },
            )
            for patch in inner_artists["boxes"]:
                patch.set_alpha(0.9)
        if model_axis == "y":
            ax.set_yticks(positions, tick_labels)
        else:
            ax.set_xticks(positions, tick_labels)
    elif kind in {"point", "errorbar", "summary"}:
        summary_rows: list[dict[str, Any]] = []
        outlier_frames: list[pd.DataFrame] = []
        for model, pos, values, model_color in zip(order, positions, arrays, colors):
            row = _uncertainty_summary(
                values,
                uncertainty=uncertainty,
                confidence_level=confidence_level,
                ci_method=ci_method,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )
            summary_rows.append({"model": model, **row})
            mean = row["mean"]
            interval = None
            if uncertainty is not None and np.isfinite(row["lower"]):
                interval = np.array(
                    [[max(0.0, mean - row["lower"]), max(0.0, row["upper"] - mean)]]
                ).T
            highlighted = model == highlight_model
            color = (
                highlight_color
                if highlighted
                else summary_color
                if summary_color is not None
                else model_color
            )
            errorbar = ax.errorbar(
                mean if model_axis == "y" else pos,
                pos if model_axis == "y" else mean,
                xerr=interval if model_axis == "y" else None,
                yerr=interval if model_axis == "x" else None,
                fmt=highlight_marker if highlighted else summary_marker,
                linestyle="none",
                color=color,
                ecolor=color,
                markersize=marker_size + (1 if highlighted else 0),
                capsize=capsize,
                elinewidth=errorbar_linewidth,
                markeredgewidth=0.6,
                zorder=4 if highlighted else 3,
            )
            if interval is not None:
                for cap in errorbar[1]:
                    cap.set_alpha(errorbar_alpha)
                for bars in errorbar[2]:
                    bars.set_alpha(errorbar_alpha)
            model_blocks = plotted.loc[plotted["model"].eq(model)].copy()
            outlier_mask = _tukey_outlier_mask(model_blocks["metric_value"])
            model_outliers = model_blocks.loc[outlier_mask]
            if not model_outliers.empty:
                outlier_frames.append(model_outliers)
                if show_fliers:
                    outlier_values = model_outliers["metric_value"].to_numpy(float)
                    ax.scatter(
                        outlier_values if model_axis == "y" else np.full(len(outlier_values), pos),
                        np.full(len(outlier_values), pos) if model_axis == "y" else outlier_values,
                        s=(marker_size + 1.0) ** 2,
                        marker="o",
                        facecolors="none",
                        edgecolors=color,
                        linewidths=0.9,
                        alpha=0.85,
                        zorder=5,
                    )
        if model_axis == "y":
            ax.set_yticks(positions, tick_labels)
        else:
            ax.set_xticks(positions, tick_labels)
    else:
        raise ValueError("plot_kind must be 'box', 'violin', 'point', 'errorbar', or 'summary'.")
    rng = np.random.default_rng(random_state)
    if show_points:
        for pos, values, color in zip(positions, arrays, colors):
            jitter = rng.normal(0, 0.045, len(values))
            ax.scatter(
                values if model_axis == "y" else pos + jitter,
                pos + jitter if model_axis == "y" else values,
                s=12,
                alpha=0.5,
                color=color,
                edgecolors="none",
            )
    if show_mean_ci and kind not in {"point", "errorbar", "summary"}:
        for pos, values in zip(positions, arrays):
            summary = _uncertainty_summary(
                values,
                uncertainty=uncertainty,
                confidence_level=confidence_level,
                ci_method=ci_method,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )
            if np.isfinite(summary["lower"]):
                ax.errorbar(
                    summary["mean"] if model_axis == "y" else pos,
                    pos if model_axis == "y" else summary["mean"],
                    xerr=[
                        [summary["mean"] - summary["lower"]],
                        [summary["upper"] - summary["mean"]],
                    ]
                    if model_axis == "y"
                    else None,
                    yerr=[
                        [summary["mean"] - summary["lower"]],
                        [summary["upper"] - summary["mean"]],
                    ]
                    if model_axis == "x"
                    else None,
                    fmt="o",
                    color=OKABE_ITO["black"],
                    markersize=4,
                    capsize=3,
                    zorder=5,
                )
    if model_axis == "y":
        ax.invert_yaxis()
        ax.set_xlabel(xlabel or label, fontsize=font_size)
        ax.set_ylabel(ylabel or "", fontsize=font_size)
        ax.tick_params(axis="y", rotation=0)
    else:
        ax.set_xlabel(xlabel or "", fontsize=font_size)
        ax.set_ylabel(ylabel or label, fontsize=font_size)
        ax.tick_params(axis="x", rotation=tick_rotation)
    if show_title and show_metric_title:
        ax.set_title(metric_title or f"{label} by model", fontsize=title_size)
    ax.tick_params(labelsize=font_size)
    metric_axis = "x" if model_axis == "y" else "y"
    model_grid_axis = "y" if model_axis == "y" else "x"
    ax.grid(axis=metric_axis, alpha=0.25, visible=grid)
    ax.grid(axis=model_grid_axis, visible=False)
    ax.set_axisbelow(True)
    if ylim is not None:
        (ax.set_xlim if model_axis == "y" else ax.set_ylim)(*ylim)
    if show_coverage:
        cov_ax = axes_arr[1]
        coverage_models = set(scope_coverage["model"]) if show_excluded_in_coverage else set(order)
        cov_order = order + [m for m in _ordered_models(coverage_models) if m not in order]
        cov = scope_coverage.set_index("model").reindex(cov_order).reset_index()
        y = np.arange(len(cov_order))
        display = coverage_display.casefold()
        labels = {
            "finished": "Completed",
            "timeout": "Timeout",
            "error": "Error",
            **dict(legend_labels or {}),
        }
        if display == "timeouts":
            cov_ax.barh(y, cov["timeouts"], color=OKABE_ITO["orange"], label=labels["timeout"])
            cov_ax.set_xlabel("Timeouts")
        elif display == "completed":
            cov_ax.barh(
                y,
                cov["successful_runs"],
                color=OKABE_ITO["bluish_green"],
                label=labels["finished"],
            )
            cov_ax.set_xlabel("Completed runs")
        elif display == "percentage":
            cov_ax.barh(
                y,
                100 * cov["coverage_fraction"],
                color=OKABE_ITO["bluish_green"],
                label=labels["finished"],
            )
            cov_ax.set_xlabel("Completed runs (%)")
            cov_ax.set_xlim(0, 103)
        elif display == "stacked":
            cov_ax.barh(
                y,
                cov["successful_runs"],
                color=OKABE_ITO["bluish_green"],
                label=labels["finished"],
            )
            cov_ax.barh(
                y,
                cov["timeouts"],
                left=cov["successful_runs"],
                color=OKABE_ITO["orange"],
                label=labels["timeout"],
            )
            cov_ax.barh(
                y,
                cov["errors"],
                left=cov["successful_runs"] + cov["timeouts"],
                color=OKABE_ITO["vermillion"],
                label=labels["error"],
            )
            cov_ax.set_xlabel("Runs")
        else:
            raise ValueError(
                "coverage_display must be timeouts, completed, percentage, or stacked."
            )
        cov_ax.set_yticks(
            y, [_model_display_label(model) for model in cov_order], fontsize=font_size
        )
        cov_ax.invert_yaxis()
        if show_title:
            cov_ax.set_title(coverage_title or "Execution coverage", fontsize=title_size)
        cov_ax.grid(axis="x", alpha=0.25, visible=grid)
        cov_ax.grid(axis="y", visible=False)
        if show_legend and display == "stacked":
            cov_ax.legend(title=legend_title, fontsize=font_size)
    if despine:
        _despine(axes_arr, font_size=font_size)
    if show_title and title:
        fig.suptitle(title, fontsize=title_size + 1)
    if apply_layout:
        fig.tight_layout()
    _save(fig, save_path, dpi, transparent)
    output = {
        "block_means": plotted.reset_index(drop=True),
        "coverage": scope_coverage.reset_index(drop=True),
        "coverage_by_dataset": by_dataset_coverage.reset_index(drop=True),
        "eligible_models": pd.DataFrame({"model": order}),
    }
    if kind in {"point", "errorbar", "summary"}:
        output["summary"] = pd.DataFrame(summary_rows)
        output["outliers"] = (
            pd.concat(outlier_frames, ignore_index=True)
            if outlier_frames
            else plotted.iloc[0:0].copy()
        )
    return (fig, axes_arr, output)


def plot_metric_boxplot_grid(
    data: FinalVisualizationData,
    panel_specs: Sequence[Mapping[str, Any]],
    *,
    metric: str = "f1",
    models: Any = None,
    coverage_policy: str = "all",
    min_completed_runs: int | None = None,
    min_coverage_fraction: float | None = None,
    plot_kind: str = "box",
    show_mean: bool = True,
    show_fliers: bool = True,
    whiskers: str | tuple[float, float] = "1.5iqr",
    uncertainty: str | None = "iqr",
    confidence_level: float = 0.95,
    ci_method: str = "bootstrap",
    n_bootstrap: int = 2000,
    random_state: int | None = 42,
    order_by: str = "mean",
    model_order: Sequence[str] | None = None,
    model_colors: Mapping[str, str] | None = None,
    highlight_model: str | None = "PRISM",
    highlight_color: str = DEFAULT_MODEL_COLORS["PRISM"],
    highlight_marker: str = "D",
    summary_color: str | None = None,
    summary_marker: str = "o",
    marker_size: float = 4.5,
    capsize: float = 2.5,
    errorbar_alpha: float = 0.45,
    errorbar_linewidth: float = 0.8,
    wrap_model_labels: bool = True,
    show_model_bands: bool = False,
    model_band_color: str = OKABE_ITO["black"],
    model_band_alpha: float = 0.045,
    ylim: tuple[float, float] | None = None,
    ncols: int = 2,
    sharex: bool = True,
    panel_labels: bool = True,
    xlabel: str | None = None,
    legend_title: str | None = None,
    legend_ncol: int = 8,
    figsize: tuple[float, float] | None = None,
    panel_left: float = 0.13,
    panel_right: float = 0.99,
    panel_bottom: float = 0.07,
    panel_top: float | None = None,
    panel_wspace: float = 0.08,
    panel_hspace: float = 0.2,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    grid: bool = True,
    despine: bool = True,
    save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> tuple[plt.Figure, np.ndarray, dict[str, dict[str, pd.DataFrame]]]:
    """Arrange horizontal metric plots in a shared figure.

    ``panel_specs`` contains one mapping per panel. A mapping may include a
    display ``title`` plus any dataset selector accepted by
    :func:`plot_metric_and_timeouts` (for example ``graph_type``, ``noise`` or
    ``mechanism``). Each panel keeps the same implementation and model-ordering
    rules as the standalone plot. ``plot_kind='point'`` gives all panels the
    point-and-uncertainty style used by the standalone tuning-like figures.
    A single model-colour legend is placed above the complete figure.
    """
    specs = [dict(spec) for spec in panel_specs]
    if not specs:
        raise ValueError("panel_specs must contain at least one panel.")
    if ncols < 1:
        raise ValueError("ncols must be at least 1.")
    if legend_ncol < 1:
        raise ValueError("legend_ncol must be at least 1.")
    allowed = {
        "datasets",
        "n_nodes",
        "graph_type",
        "noise",
        "mechanism",
        "obs_conf",
        "lat_conf",
        "filters",
    }
    invalid = sorted({key for spec in specs for key in spec if key not in allowed | {"title"}})
    if invalid:
        raise ValueError(f"Unsupported panel specification keys: {invalid}")
    nrows = math.ceil(len(specs) / ncols)
    if figsize is None:
        figsize = (8.2 * ncols, 6.2 * nrows + 1.0)
    fig, made_axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharex=sharex)
    axes = made_axes.ravel()
    outputs: dict[str, dict[str, pd.DataFrame]] = {}
    present_models: list[str] = []
    for index, (axis, spec) in enumerate(zip(axes, specs)):
        panel_title = str(spec.pop("title", f"Panel {index + 1}"))
        _, _, tables = plot_metric_and_timeouts(
            data,
            metric=metric,
            models=models,
            plot_kind=plot_kind,
            coverage_policy=coverage_policy,
            min_completed_runs=min_completed_runs,
            min_coverage_fraction=min_coverage_fraction,
            show_timeouts=False,
            show_mean=show_mean,
            show_fliers=show_fliers,
            whiskers=whiskers,
            show_mean_ci=False,
            uncertainty=uncertainty,
            confidence_level=confidence_level,
            ci_method=ci_method,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            order_by=order_by,
            model_order=model_order,
            model_colors=model_colors,
            model_axis="y",
            highlight_model=highlight_model,
            highlight_color=highlight_color,
            highlight_marker=highlight_marker,
            summary_color=summary_color,
            summary_marker=summary_marker,
            marker_size=marker_size,
            capsize=capsize,
            errorbar_alpha=errorbar_alpha,
            errorbar_linewidth=errorbar_linewidth,
            wrap_model_labels=wrap_model_labels,
            show_model_bands=show_model_bands,
            model_band_color=model_band_color,
            model_band_alpha=model_band_alpha,
            ylim=ylim,
            show_title=False,
            show_metric_title=False,
            show_legend=False,
            xlabel=xlabel,
            axes=axis,
            font_size=font_size,
            title_size=title_size,
            grid=grid,
            despine=despine,
            apply_layout=False,
            **spec,
        )
        prefix = f"({chr(97 + index)}) " if panel_labels and index < 26 else ""
        axis.set_title(f"{prefix}{panel_title}", fontsize=title_size)
        outputs[panel_title] = tables
        present_models.extend(tables["block_means"]["model"].astype(str))
    for axis in axes[len(specs) :]:
        axis.set_visible(False)
    legend_models = _ordered_models(dict.fromkeys(present_models), model_order=model_order)
    legend_colours = _colors(legend_models, model_colors)
    point_kind = str(plot_kind).casefold() in {"point", "errorbar", "summary"}
    canonical_highlight = _canonical_model_name(highlight_model)
    if point_kind:
        handles = [
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker=highlight_marker if model == canonical_highlight else summary_marker,
                color=highlight_color if model == canonical_highlight else colour,
                markersize=marker_size + (1 if model == canonical_highlight else 0),
                label=_model_display_label(model),
            )
            for model, colour in zip(legend_models, legend_colours)
        ]
    else:
        handles = [
            Patch(
                facecolor=colour,
                edgecolor="#111827",
                linewidth=0.8,
                alpha=0.55,
                label=_model_display_label(model),
            )
            for model, colour in zip(legend_models, legend_colours)
        ]
    columns = min(legend_ncol, len(handles))
    legend_rows = math.ceil(len(handles) / columns)
    fig.legend(
        handles=handles,
        title=legend_title,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=columns,
        frameon=False,
        fontsize=max(7, font_size - 2),
        title_fontsize=max(8, font_size - 1),
        handlelength=1.5,
        columnspacing=1.2,
    )
    top = panel_top if panel_top is not None else max(0.82, 0.97 - 0.02 * legend_rows)
    fig.subplots_adjust(
        left=panel_left,
        right=panel_right,
        bottom=panel_bottom,
        top=top,
        wspace=panel_wspace,
        hspace=panel_hspace,
    )
    _save(fig, save_path, dpi, transparent)
    return (fig, made_axes, outputs)


def plot_metric_polar(
    data: FinalVisualizationData,
    *,
    metric: str = "f1",
    datasets: Any = None,
    n_nodes: Any = None,
    graph_type: Any = None,
    noise: Any = None,
    mechanism: Any = None,
    obs_conf: Any = None,
    lat_conf: Any = None,
    models: Any = None,
    filters: Mapping[str, Any] | None = None,
    dataset_order: Sequence[str] | None = None,
    dataset_labels: Mapping[str, str] | None = None,
    model_order: Sequence[str] | None = None,
    model_colors: Mapping[str, str] | None = None,
    radial_limits: tuple[float, float] = (0.0, 1.0),
    radial_ticks: Sequence[float] | None = None,
    radial_scale: str = "linear",
    log_offset: float = 0.01,
    title: str | None = None,
    legend_title: str | None = "Model",
    show_legend: bool = True,
    figsize: tuple[float, float] = (11.5, 8.5),
    ax: plt.Axes | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    grid: bool = True,
    save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> tuple[plt.Figure, plt.Axes, dict[str, pd.DataFrame]]:
    """Plot one mean metric point per dataset and model on polar axes.

    Model seeds are first averaged inside each independent
    ``dataset_seed x dataset x model`` block. Each polar point is then the
    mean of those independent dataset-seed values. No uncertainty bars are
    drawn. Missing model/dataset combinations remain gaps rather than being
    imputed or connected across. ``radial_scale`` accepts ``"linear"``,
    ``"sqrt"`` (a moderate expansion of low scores), or ``"log"``. The log
    option uses ``log10(value + log_offset)`` so that a valid F1 of zero can
    still be displayed; radial tick labels always show the original values.
    """
    selected = _filter_frame(
        data.metrics,
        **_filter_kwargs(
            datasets,
            n_nodes,
            graph_type,
            noise,
            mechanism,
            obs_conf,
            lat_conf,
            models,
            filters,
        ),
    )
    blocks = _block_means(selected, metric)
    summary = (
        blocks.groupby(["dataset", "model"], observed=True)["metric_value"]
        .agg(mean_metric="mean", n_dataset_seeds="size")
        .reset_index()
    )
    present_datasets = list(dict.fromkeys(summary["dataset"].astype(str)))
    if dataset_order is None:
        ordered_datasets = sorted(present_datasets, key=str)
    else:
        requested = [str(value) for value in dataset_order]
        unknown = [value for value in requested if value not in present_datasets]
        if unknown:
            raise ValueError(f"dataset_order contains unknown datasets: {unknown}")
        ordered_datasets = requested + [
            value for value in present_datasets if value not in requested
        ]
    if len(ordered_datasets) < 3:
        raise ValueError("A polar comparison needs at least three datasets.")
    overall_means = summary.groupby("model", observed=True)["mean_metric"].mean().to_dict()
    if model_order is None:
        ordered_models = _ordered_models(summary["model"], values=overall_means, ascending=False)
    else:
        ordered_models = _ordered_models(summary["model"], model_order=model_order)
    values = summary.pivot(index="model", columns="dataset", values="mean_metric").reindex(
        index=ordered_models, columns=ordered_datasets
    )
    scale = radial_scale.casefold()
    if scale not in {"linear", "sqrt", "log"}:
        raise ValueError("radial_scale must be 'linear', 'sqrt', or 'log'.")
    lower, upper = map(float, radial_limits)
    if not np.isfinite([lower, upper]).all() or lower < 0 or lower >= upper:
        raise ValueError("radial_limits must be finite, non-negative, and increasing.")
    if scale == "log" and (not np.isfinite(log_offset) or log_offset <= 0):
        raise ValueError("log_offset must be positive and finite for log scale.")

    def transform_radius(radius: Any) -> np.ndarray:
        radius_array = np.asarray(radius, dtype=float)
        if scale == "sqrt":
            return np.sqrt(radius_array)
        if scale == "log":
            return np.log10(radius_array + log_offset) - np.log10(log_offset)
        return radius_array

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    else:
        if ax.name != "polar":
            raise ValueError("ax must use a polar projection.")
        fig = ax.figure
    angles = np.linspace(0, 2 * np.pi, len(ordered_datasets), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    colors = _colors(ordered_models, model_colors)
    for model, color in zip(ordered_models, colors):
        original_radius = values.loc[model].to_numpy(dtype=float)
        radius = transform_radius(original_radius)
        closed_radius = np.r_[radius, radius[0]]
        ax.plot(
            closed_angles,
            closed_radius,
            color=color,
            linewidth=1.35,
            alpha=0.72,
            label=_model_display_label(model),
            zorder=2,
        )
        finite = np.isfinite(original_radius)
        ax.scatter(
            angles[finite],
            radius[finite],
            color=color,
            s=28,
            edgecolor="white",
            linewidth=0.55,
            alpha=0.95,
            zorder=3,
        )
    labels = dict(dataset_labels or {})
    display_labels = [
        labels.get(dataset, _dataset_display_title(dataset)) for dataset in ordered_datasets
    ]
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(angles, display_labels)
    ax.tick_params(axis="x", labelsize=max(8, font_size - 2), pad=11)
    ax.tick_params(axis="y", labelsize=max(8, font_size - 3))
    transformed_limits = transform_radius([lower, upper])
    ax.set_ylim(*transformed_limits)
    if radial_ticks is None:
        if scale == "sqrt":
            radial_ticks = np.linspace(np.sqrt(lower), np.sqrt(upper), 6)[1:] ** 2
        elif scale == "log":
            radial_ticks = np.geomspace(lower + log_offset, upper + log_offset, 6)[1:] - log_offset
        else:
            radial_ticks = np.linspace(lower, upper, 6)[1:]
    radial_ticks_array = np.asarray(radial_ticks, dtype=float)
    if (
        radial_ticks_array.ndim != 1
        or not np.isfinite(radial_ticks_array).all()
        or np.any(radial_ticks_array < lower)
        or np.any(radial_ticks_array > upper)
    ):
        raise ValueError("radial_ticks must lie within radial_limits.")
    tick_labels = [
        f"{tick:.2f}" if 0 < abs(tick) < 0.1 else f"{tick:.1f}" for tick in radial_ticks_array
    ]
    ax.set_yticks(transform_radius(radial_ticks_array), tick_labels)
    ax.set_rlabel_position(82)
    ax.grid(grid, alpha=0.25, linewidth=0.8)
    ax.spines["polar"].set_alpha(0.35)
    label, _ = _metric_meta(metric)
    ax.set_title(title or f"{label} across datasets", fontsize=title_size, pad=24)
    if show_legend:
        ax.legend(
            title=legend_title,
            loc="center left",
            bbox_to_anchor=(1.16, 0.5),
            frameon=False,
            ncol=2 if len(ordered_models) > 9 else 1,
            fontsize=max(8, font_size - 3),
            title_fontsize=max(9, font_size - 2),
            handlelength=1.8,
            columnspacing=1.2,
            labelspacing=0.55,
        )
    fig.subplots_adjust(left=0.08, right=0.73 if show_legend else 0.95, top=0.88, bottom=0.1)
    _save(fig, save_path, dpi, transparent)
    return (
        fig,
        ax,
        {
            "block_means": blocks.reset_index(drop=True),
            "summary": summary.reset_index(drop=True),
            "values": values.reset_index(),
        },
    )


def build_execution_summary_tables(
    data: FinalVisualizationData,
    *,
    datasets: Any = None,
    n_nodes: Any = None,
    graph_type: Any = None,
    noise: Any = None,
    mechanism: Any = None,
    obs_conf: Any = None,
    lat_conf: Any = None,
    models: Any = None,
    filters: Mapping[str, Any] | None = None,
    runtime_column: str = "runtime_seconds",
    confidence_level: float = 0.95,
    ci_method: str = "t",
    n_bootstrap: int = 2000,
    random_state: int | None = 42,
    model_order: Sequence[str] | None = None,
    dataset_labels: Mapping[str, str] | None = None,
    runtime_digits: int = 2,
) -> dict[str, pd.DataFrame]:
    """Build completion/timeout and runtime-CI tables by model and dataset.

    A fit is completed when ``status == 'finished'`` and it did not time out.
    Runtime summaries use only completed fits with a finite runtime. Model-seed
    runtimes are first averaged inside each independent ``dataset_seed x
    dataset x model`` block; means and confidence intervals are then computed
    across those dataset-seed means. The matrix tables are presentation-ready;
    the long tables retain all exact numeric values for auditing.
    """
    selected = _filter_frame(
        data.metrics,
        **_filter_kwargs(
            datasets,
            n_nodes,
            graph_type,
            noise,
            mechanism,
            obs_conf,
            lat_conf,
            models,
            filters,
        ),
    ).copy()
    _require_columns(
        selected, ["dataset", "model", "status", "timed_out", runtime_column], "metrics"
    )
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")
    if runtime_digits < 0:
        raise ValueError("runtime_digits must be at least 0.")
    status = selected["status"].astype(str).str.casefold()
    timed_out = _as_bool(selected["timed_out"]) | status.eq("timeout")
    selected["_completed"] = status.eq("finished") & ~timed_out
    selected["_timeout"] = timed_out
    selected["_error"] = ~selected["_completed"] & ~selected["_timeout"]
    selected["_runtime"] = pd.to_numeric(selected[runtime_column], errors="coerce")
    counts = (
        selected.groupby(["dataset", "model"], observed=True)
        .agg(
            completed_fits=("_completed", "sum"),
            total_fits=("model", "size"),
            timeouts=("_timeout", "sum"),
            errors=("_error", "sum"),
        )
        .reset_index()
    )
    counts["completion_fraction"] = counts["completed_fits"] / counts["total_fits"].clip(lower=1)
    for column in ["completed_fits", "total_fits", "timeouts", "errors"]:
        counts[column] = counts[column].astype(int)
    _require_columns(selected, ["dataset_seed"], "metrics")
    valid_runtime = selected[selected["_completed"] & selected["_runtime"].notna()]
    runtime_seed_means = (
        valid_runtime.groupby(["dataset_seed", "dataset", "model"], observed=True)["_runtime"]
        .agg(mean_runtime_seconds="mean", completed_model_seeds="size")
        .reset_index()
    )
    runtime_rows: list[dict[str, Any]] = []
    for (dataset, model), group in runtime_seed_means.groupby(["dataset", "model"], observed=True):
        runtimes = group["mean_runtime_seconds"].to_numpy(dtype=float)
        summary = _uncertainty_summary(
            runtimes,
            uncertainty="ci",
            confidence_level=confidence_level,
            ci_method=ci_method,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        runtime_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "mean_runtime_seconds": summary["mean"],
                "ci_lower_seconds": summary["lower"],
                "ci_upper_seconds": summary["upper"],
                "std_runtime_seconds": summary["std"],
                "median_runtime_seconds": summary["median"],
                "n_dataset_seeds": int(summary["n"]),
                "n_runtime_fits": int(group["completed_model_seeds"].sum()),
                "min_completed_model_seeds": int(group["completed_model_seeds"].min()),
                "max_completed_model_seeds": int(group["completed_model_seeds"].max()),
                "confidence_level": confidence_level,
                "ci_method": ci_method,
            }
        )
    runtime_columns = [
        "dataset",
        "model",
        "mean_runtime_seconds",
        "ci_lower_seconds",
        "ci_upper_seconds",
        "std_runtime_seconds",
        "median_runtime_seconds",
        "n_dataset_seeds",
        "n_runtime_fits",
        "min_completed_model_seeds",
        "max_completed_model_seeds",
        "confidence_level",
        "ci_method",
    ]
    runtime_summary = pd.DataFrame(runtime_rows, columns=runtime_columns)
    dataset_names = list(dict.fromkeys(selected["dataset"].astype(str)))
    supplied_labels = dict(dataset_labels or {})
    label_rows: list[dict[str, str]] = []
    for dataset in dataset_names:
        row = selected[selected["dataset"].astype(str).eq(dataset)].iloc[0]
        label_rows.append(
            {
                "dataset": dataset,
                "dataset_label": supplied_labels.get(dataset, _pretty_dataset_label(row)),
            }
        )
    label_table = pd.DataFrame(label_rows)
    duplicated = label_table["dataset_label"].duplicated(keep=False)
    for index in label_table.index[duplicated]:
        dataset = label_table.at[index, "dataset"]
        row = selected[selected["dataset"].astype(str).eq(dataset)].iloc[0]
        details = []
        if pd.notna(row.get("n_nodes")):
            details.append(f"{int(row['n_nodes'])} nodes")
        if pd.notna(row.get("graph_type")):
            details.append(_dataset_display_title(row["graph_type"]))
        if details:
            label_table.at[index, "dataset_label"] += f" ({', '.join(details)})"
    label_lookup = label_table.set_index("dataset")["dataset_label"].to_dict()
    model_names = _ordered_models(counts["model"].unique(), model_order=model_order)
    counts["display"] = counts.apply(
        lambda row: (
            f"{row['completed_fits']}/{row['total_fits']} | T:{row['timeouts']} | E:{row['errors']}"
        ),
        axis=1,
    )
    completion_table = counts.pivot(index="model", columns="dataset", values="display").reindex(
        index=model_names, columns=dataset_names
    )
    completion_table = completion_table.rename(columns=label_lookup)

    def _format_runtime(row: pd.Series) -> str:
        mean = row["mean_runtime_seconds"]
        lower = row["ci_lower_seconds"]
        upper = row["ci_upper_seconds"]
        if not np.isfinite(mean):
            return "–"
        if np.isfinite(lower) and np.isfinite(upper):
            return f"{mean:.{runtime_digits}f} [{lower:.{runtime_digits}f}, {upper:.{runtime_digits}f}]"
        return f"{mean:.{runtime_digits}f} [CI n/a]"

    if runtime_summary.empty:
        runtime_table = pd.DataFrame(
            "–",
            index=model_names,
            columns=[label_lookup[dataset] for dataset in dataset_names],
        )
    else:
        runtime_summary["display"] = runtime_summary.apply(_format_runtime, axis=1)
        runtime_table = runtime_summary.pivot(
            index="model", columns="dataset", values="display"
        ).reindex(index=model_names, columns=dataset_names)
        runtime_table = runtime_table.rename(columns=label_lookup)
    return {
        "completion_table": completion_table,
        "runtime_table": runtime_table,
        "fit_counts": counts.drop(columns="display").reset_index(drop=True),
        "runtime_summary": runtime_summary.drop(columns="display", errors="ignore").reset_index(
            drop=True
        ),
        "runtime_seed_means": runtime_seed_means.reset_index(drop=True),
        "dataset_labels": label_table.reset_index(drop=True),
    }


def _rank_metric_available(frame: pd.DataFrame, metric: str) -> None:
    available = sorted(frame["metric"].dropna().unique())
    if metric not in available:
        raise ValueError(
            f"Please compute rank first. No precomputed Wilcoxon-Holm rank is available for metric '{metric}'. Available metrics: {available}"
        )


def _rank_rows_with_coverage(
    data: FinalVisualizationData,
    *,
    metric: str,
    filter_kwargs: Mapping[str, Any],
    coverage_policy: str,
    min_completed_runs: int | None,
    min_coverage_fraction: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    _rank_metric_available(data.ranks_by_dataset, metric)
    metrics_selected = _filter_frame(data.metrics, **filter_kwargs)
    ranks_selected = _filter_frame(data.ranks_by_dataset, **filter_kwargs)
    ranks_selected = ranks_selected[ranks_selected["metric"].eq(metric)].copy()
    if ranks_selected.empty:
        raise ValueError("Please compute rank first. No rank exists for the selected scope.")
    coverage_by_dataset, coverage_scope = _coverage_from_metrics(metrics_selected, metric)
    eligible = _eligible_models(
        coverage_by_dataset,
        coverage_policy=coverage_policy,
        min_completed_runs=min_completed_runs,
        min_coverage_fraction=min_coverage_fraction,
    )
    coverage_selected = coverage_by_dataset.copy()
    rank_cols = [
        c for c in ranks_selected.columns if c not in FILTER_COLUMNS or c in {"dataset", "model"}
    ]
    merged = coverage_selected.merge(
        ranks_selected[rank_cols],
        on=["dataset", "model"],
        how="left",
        validate="one_to_one",
    )
    friedman = _filter_frame(data.friedman_by_dataset, **filter_kwargs)
    friedman = friedman[friedman["metric"].eq(metric)]
    stat_cols = [
        c
        for c in [
            "dataset",
            "model",
            "mean",
            "std",
            "median",
            "min",
            "max",
            "count",
            "mean_rank",
        ]
        if c in friedman.columns
    ]
    merged = merged.merge(
        friedman[stat_cols], on=["dataset", "model"], how="left", validate="one_to_one"
    )
    merged["rank_available"] = merged["rank"].notna()
    merged["included_by_coverage_policy"] = merged["model"].isin(eligible)
    return (_add_dataset_attributes(merged), coverage_scope, eligible)


def build_ranking_table(
    data: FinalVisualizationData,
    *,
    metric: str = "f1",
    datasets: Any = None,
    n_nodes: Any = None,
    graph_type: Any = None,
    noise: Any = None,
    mechanism: Any = None,
    obs_conf: Any = None,
    lat_conf: Any = None,
    models: Any = None,
    filters: Mapping[str, Any] | None = None,
    rank_aggregation: str = "per_dataset",
    coverage_policy: str = "complete",
    min_completed_runs: int | None = None,
    min_coverage_fraction: float | None = None,
    include_excluded: bool = False,
    uncertainty: str | None = "ci",
    confidence_level: float = 0.95,
    ci_method: str = "bootstrap",
    n_bootstrap: int = 2000,
    random_state: int | None = 42,
    columns: Sequence[str] | None = None,
    column_labels: Mapping[str, str] | None = None,
    sort: bool = True,
) -> pd.DataFrame:
    """Build a rank-and-coverage table using only precomputed ranks.

    ``rank_aggregation='mean'`` averages the already-computed per-configuration
    ranks.  It does not run Wilcoxon.  ``precomputed_global`` returns the
    existing global rank and is only valid without dataset filters.
    """
    kwargs = _filter_kwargs(
        datasets,
        n_nodes,
        graph_type,
        noise,
        mechanism,
        obs_conf,
        lat_conf,
        models,
        filters,
    )
    aggregation = rank_aggregation.casefold()
    if aggregation == "precomputed_global":
        if (
            any(
                (
                    v not in (None, "all")
                    for v in [
                        datasets,
                        n_nodes,
                        graph_type,
                        noise,
                        mechanism,
                        obs_conf,
                        lat_conf,
                    ]
                )
            )
            or filters
        ):
            raise ValueError("precomputed_global cannot be combined with dataset filters.")
        _rank_metric_available(data.ranks_overall, metric)
        table = data.ranks_overall[data.ranks_overall["metric"].eq(metric)].copy()
        metrics_selected = _filter_frame(data.metrics, models=models)
        by_dataset, coverage = _coverage_from_metrics(metrics_selected, metric)
        eligible = _eligible_models(
            by_dataset,
            coverage_policy=coverage_policy,
            min_completed_runs=min_completed_runs,
            min_coverage_fraction=min_coverage_fraction,
        )
        table = table.merge(coverage, on="model", how="outer", validate="one_to_one")
        table["rank_available"] = table["rank"].notna()
        table["included_by_coverage_policy"] = table["model"].isin(eligible)
    else:
        table, coverage, eligible = _rank_rows_with_coverage(
            data,
            metric=metric,
            filter_kwargs=kwargs,
            coverage_policy=coverage_policy,
            min_completed_runs=min_completed_runs,
            min_coverage_fraction=min_coverage_fraction,
        )
        if aggregation == "mean":
            rows = []
            selected_configs = table["dataset"].nunique()
            for model, group in table.groupby("model", observed=True):
                ranks = group["rank"].dropna().to_numpy(dtype=float)
                summary = _uncertainty_summary(
                    ranks,
                    uncertainty=uncertainty,
                    confidence_level=confidence_level,
                    ci_method=ci_method,
                    n_bootstrap=n_bootstrap,
                    random_state=random_state,
                )
                cov = coverage[coverage["model"].eq(model)]
                cov_row = cov.iloc[0].to_dict() if not cov.empty else {}
                rows.append(
                    {
                        "model": model,
                        "mean_rank": summary["mean"],
                        "median_rank": summary["median"],
                        "rank_std": summary["std"],
                        "best_rank": summary["min"],
                        "worst_rank": summary["max"],
                        "rank_lower": summary["lower"],
                        "rank_upper": summary["upper"],
                        "n_configurations_ranked": summary["n"],
                        "n_configurations_selected": selected_configs,
                        "rank_available": summary["n"] > 0,
                        "included_by_coverage_policy": model in eligible,
                        **cov_row,
                    }
                )
            table = pd.DataFrame(rows)
        elif aggregation != "per_dataset":
            raise ValueError("rank_aggregation must be per_dataset, mean, or precomputed_global.")
    if not include_excluded:
        table = table[table["included_by_coverage_policy"]]
    if table.empty:
        raise ValueError("No models satisfy the selected coverage policy.")
    if sort:
        sort_cols = (
            ["dataset", "rank"]
            if aggregation == "per_dataset"
            else ["rank"]
            if aggregation == "precomputed_global"
            else ["mean_rank"]
        )
        table = table.sort_values(sort_cols, na_position="last")
    if columns is not None:
        missing = sorted(set(columns) - set(table.columns))
        if missing:
            raise ValueError(f"Requested table columns are unavailable: {missing}")
        table = table[list(columns)]
    if column_labels:
        table = table.rename(columns=dict(column_labels))
    return table.reset_index(drop=True)


def plot_rank_variation(
    data: FinalVisualizationData,
    *,
    metric: str = "f1",
    datasets: Any = None,
    n_nodes: Any = None,
    graph_type: Any = None,
    noise: Any = None,
    mechanism: Any = None,
    obs_conf: Any = None,
    lat_conf: Any = None,
    models: Any = None,
    filters: Mapping[str, Any] | None = None,
    plot_kind: str = "violin",
    vary_by: str = "dataset",
    coverage_policy: str = "complete",
    min_completed_runs: int | None = None,
    min_coverage_fraction: float | None = None,
    show_coverage: bool = False,
    coverage_display: str = "percentage",
    uncertainty: str | None = "ci",
    confidence_level: float = 0.95,
    ci_method: str = "bootstrap",
    n_bootstrap: int = 2000,
    random_state: int | None = 42,
    show_mean: bool = True,
    show_points: bool = True,
    show_fliers: bool = True,
    violin_inner: str | None = "box",
    model_order: Sequence[str] | None = None,
    model_colors: Mapping[str, str] | None = None,
    title: str | None = None,
    show_title: bool = True,
    show_metric_title: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend_title: str | None = None,
    legend_labels: Mapping[str, str] | None = None,
    show_legend: bool = True,
    figsize: tuple[float, float] | None = None,
    axes: Sequence[plt.Axes] | plt.Axes | None = None,
    invert_rank_axis: bool = True,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    grid: bool = True,
    save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> tuple[plt.Figure, np.ndarray, dict[str, pd.DataFrame]]:
    """Plot variation of precomputed Wilcoxon-Holm ranks.

    ``box`` or ``violin`` shows the distribution across selected
    configurations; ``line`` shows the trajectory over ``vary_by`` (normally
    ``n_nodes``); and ``line_and_box``/``line_and_violin`` combines both views.
    """
    kwargs = _filter_kwargs(
        datasets,
        n_nodes,
        graph_type,
        noise,
        mechanism,
        obs_conf,
        lat_conf,
        models,
        filters,
    )
    rank_rows, coverage, eligible = _rank_rows_with_coverage(
        data,
        metric=metric,
        filter_kwargs=kwargs,
        coverage_policy=coverage_policy,
        min_completed_runs=min_completed_runs,
        min_coverage_fraction=min_coverage_fraction,
    )
    rank_rows = rank_rows[rank_rows["model"].isin(eligible) & rank_rows["rank_available"]].copy()
    if rank_rows.empty:
        raise ValueError("Please compute rank first. No eligible precomputed ranks remain.")
    if vary_by not in rank_rows.columns:
        raise ValueError(f"Unknown vary_by column '{vary_by}'.")
    order = _ordered_models(rank_rows["model"].unique(), model_order=model_order)
    colors = dict(zip(order, _colors(order, model_colors)))
    kind = plot_kind.casefold()
    if kind not in {"box", "violin", "line", "line_and_box", "line_and_violin"}:
        raise ValueError("plot_kind must be box, violin, line, line_and_box, or line_and_violin.")
    main_axes = 2 if kind in {"line_and_box", "line_and_violin"} else 1
    n_axes = main_axes + int(show_coverage)
    if axes is None:
        if figsize is None:
            figsize = (
                8 * main_axes + (4 if show_coverage else 0),
                max(5, 0.42 * len(order) + 2),
            )
        widths = ([1.35, 1] if kind in {"line_and_box", "line_and_violin"} else [1]) + (
            [0.75] if show_coverage else []
        )
        fig, made_axes = plt.subplots(
            1,
            n_axes,
            figsize=figsize,
            squeeze=False,
            gridspec_kw={"width_ratios": widths},
        )
        axes_arr = made_axes.ravel()
    else:
        axes_arr = np.atleast_1d(axes)
        if len(axes_arr) != n_axes:
            raise ValueError(f"Expected {n_axes} axes, received {len(axes_arr)}.")
        fig = axes_arr[0].figure
    rng = np.random.default_rng(random_state)

    def draw_distribution(ax: plt.Axes, distribution_kind: str) -> None:
        arrays = [
            rank_rows.loc[rank_rows["model"].eq(m), "rank"].to_numpy(dtype=float) for m in order
        ]
        positions = np.arange(1, len(order) + 1)
        if distribution_kind == "box":
            artists = ax.boxplot(
                arrays,
                orientation="horizontal",
                positions=positions,
                tick_labels=[_model_display_label(model) for model in order],
                patch_artist=True,
                showmeans=show_mean,
                showfliers=show_fliers,
                medianprops={"color": "#111827", "linewidth": 1.2},
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "#111827",
                    "markeredgecolor": "#111827",
                    "markersize": 3,
                },
            )
            for patch, model in zip(artists["boxes"], order):
                patch.set_facecolor(colors[model])
                patch.set_alpha(0.55)
        else:
            too_small = [model for model, values in zip(order, arrays) if len(values) < 2]
            if too_small:
                raise ValueError(
                    f"A violin plot needs at least two rank observations per model. Insufficient data for: {too_small}"
                )
            inner = None if violin_inner is None else violin_inner.casefold()
            if inner not in {None, "box"}:
                raise ValueError("violin_inner must be 'box' or None.")
            artists = ax.violinplot(
                arrays,
                positions=positions,
                orientation="horizontal",
                showmeans=False,
                showmedians=inner is None,
                showextrema=inner is None,
            )
            for body, model in zip(artists["bodies"], order):
                body.set_facecolor(colors[model])
                body.set_edgecolor(colors[model])
                body.set_alpha(0.55)
            if inner == "box":
                inner_artists = ax.boxplot(
                    arrays,
                    orientation="horizontal",
                    positions=positions,
                    widths=0.16,
                    patch_artist=True,
                    showmeans=show_mean,
                    showfliers=show_fliers,
                    manage_ticks=False,
                    boxprops={
                        "facecolor": "white",
                        "edgecolor": "#111827",
                        "linewidth": 1.0,
                    },
                    whiskerprops={"color": "#111827", "linewidth": 0.9},
                    capprops={"color": "#111827", "linewidth": 0.9},
                    medianprops={"color": "#111827", "linewidth": 1.4},
                    meanprops={
                        "marker": "D",
                        "markerfacecolor": "#111827",
                        "markeredgecolor": "#111827",
                        "markersize": 3,
                    },
                    flierprops={
                        "marker": "o",
                        "markerfacecolor": "#111827",
                        "markeredgecolor": "none",
                        "markersize": 2.5,
                        "alpha": 0.45,
                    },
                )
                for patch in inner_artists["boxes"]:
                    patch.set_alpha(0.9)
            ax.set_yticks(positions, [_model_display_label(model) for model in order])
        if show_points:
            for pos, (model, values) in enumerate(zip(order, arrays), start=1):
                jitter = rng.normal(0, 0.045, len(values))
                ax.scatter(
                    values,
                    pos + jitter,
                    color=colors[model],
                    s=13,
                    alpha=0.55,
                    edgecolors="none",
                )
        ax.set_xlabel(xlabel or "Wilcoxon-Holm rank", fontsize=font_size)
        ax.set_ylabel(ylabel or "", fontsize=font_size)
        if show_title and show_metric_title:
            ax.set_title("Rank distribution across configurations", fontsize=title_size)
        ax.grid(axis="x", alpha=0.25, visible=grid)
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()

    line_summary_rows: list[dict[str, Any]] = []

    def draw_line(ax: plt.Axes) -> None:
        values = sorted(rank_rows[vary_by].dropna().unique())
        for model in order:
            sub = rank_rows[rank_rows["model"].eq(model)]
            means, lower, upper, xs = ([], [], [], [])
            for value in values:
                y = sub.loc[sub[vary_by].eq(value), "rank"].to_numpy(dtype=float)
                if len(y) == 0:
                    continue
                summary = _uncertainty_summary(
                    y,
                    uncertainty=uncertainty,
                    confidence_level=confidence_level,
                    ci_method=ci_method,
                    n_bootstrap=n_bootstrap,
                    random_state=random_state,
                )
                xs.append(value)
                means.append(summary["mean"])
                lower.append(summary["lower"])
                upper.append(summary["upper"])
                line_summary_rows.append({"model": model, vary_by: value, **summary})
            label = (legend_labels or {}).get(model, _model_display_label(model))
            ax.plot(xs, means, marker="o", linewidth=1.6, color=colors[model], label=label)
            if len(xs) and np.isfinite(lower).any():
                lo = np.asarray(lower, dtype=float)
                hi = np.asarray(upper, dtype=float)
                mean_arr = np.asarray(means, dtype=float)
                ax.errorbar(
                    xs,
                    mean_arr,
                    yerr=[mean_arr - lo, hi - mean_arr],
                    fmt="none",
                    color=colors[model],
                    alpha=0.65,
                    capsize=2,
                )
        ax.set_xlabel(xlabel or vary_by.replace("_", " ").title(), fontsize=font_size)
        ax.set_ylabel(ylabel or "Mean Wilcoxon-Holm rank", fontsize=font_size)
        if show_title and show_metric_title:
            ax.set_title(f"Rank by {vary_by.replace('_', ' ')}", fontsize=title_size)
        ax.grid(alpha=0.25, visible=grid)
        if invert_rank_axis:
            ax.invert_yaxis()
        if show_legend:
            ax.legend(title=legend_title, fontsize=max(7, font_size - 1), ncols=2)

    next_axis = 0
    if kind in {"line", "line_and_box", "line_and_violin"}:
        draw_line(axes_arr[next_axis])
        next_axis += 1
    if kind in {"box", "line_and_box"}:
        draw_distribution(axes_arr[next_axis], "box")
        next_axis += 1
    if kind in {"violin", "line_and_violin"}:
        draw_distribution(axes_arr[next_axis], "violin")
        next_axis += 1
    if show_coverage:
        cov_ax = axes_arr[next_axis]
        cov = coverage.set_index("model").reindex(order).reset_index()
        y = np.arange(len(order))
        if coverage_display == "percentage":
            cov_ax.barh(
                y,
                100 * cov["coverage_fraction"],
                color=[colors[m] for m in order],
                alpha=0.75,
            )
            cov_ax.set_xlabel("Completed (%)")
            cov_ax.set_xlim(0, 103)
        elif coverage_display == "timeouts":
            cov_ax.barh(y, cov["timeouts"], color=OKABE_ITO["orange"])
            cov_ax.set_xlabel("Timeouts")
        else:
            cov_ax.barh(y, cov["successful_runs"], color=OKABE_ITO["bluish_green"])
            cov_ax.set_xlabel("Completed runs")
        cov_ax.set_yticks(y, [_model_display_label(model) for model in order])
        cov_ax.invert_yaxis()
        if show_title and show_metric_title:
            cov_ax.set_title("Execution coverage", fontsize=title_size)
        cov_ax.grid(axis="x", alpha=0.25, visible=grid)
        cov_ax.grid(axis="y", visible=False)
    if show_title and title:
        fig.suptitle(title, fontsize=title_size + 1)
    _despine(axes_arr, font_size=font_size)
    fig.tight_layout()
    _save(fig, save_path, dpi, transparent)
    summary_rows = []
    for model, group in rank_rows.groupby("model", observed=True):
        summary = _uncertainty_summary(
            group["rank"],
            uncertainty=uncertainty,
            confidence_level=confidence_level,
            ci_method=ci_method,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        summary_rows.append({"model": model, **summary})
    return (
        fig,
        axes_arr,
        {
            "ranks": rank_rows.reset_index(drop=True),
            "rank_summary": pd.DataFrame(summary_rows),
            "line_summary": pd.DataFrame(line_summary_rows),
            "coverage": coverage.reset_index(drop=True),
        },
    )


def _pretty_dataset_label(row: pd.Series) -> str:
    """Turn a canonical dataset identifier into a concise legend label."""
    noise = str(row.get("noise", "")).replace("_", " ")
    noise_lookup = {
        "nongaussian": "Non Gaussian",
        "non gaussian": "Non Gaussian",
        "non-gaussian": "Non Gaussian",
        "highgaussian": "High Gaussian",
        "high gaussian": "High Gaussian",
        "gaussian": "Gaussian",
        "laplace": "Laplace",
    }
    noise_label = noise_lookup.get(noise.casefold(), noise.strip().title())
    mechanism_raw = str(row.get("mechanism", "")).replace("_", " ").strip()
    mechanism_lookup = {
        "nonlinear": "Mixed",
        "non linear": "Mixed",
        "non-linear": "Mixed",
        "linear": "Linear",
        "mixed": "Mixed",
    }
    mechanism = mechanism_lookup.get(mechanism_raw.casefold(), mechanism_raw.title())
    observed_value = pd.to_numeric(pd.Series([row.get("obs_conf", 0)]), errors="coerce").iloc[0]
    latent_value = pd.to_numeric(pd.Series([row.get("lat_conf", 0)]), errors="coerce").iloc[0]
    observed = int(observed_value) if pd.notna(observed_value) else 0
    latent = int(latent_value) if pd.notna(latent_value) else 0
    if observed == latent:
        confounding = f"{latent} latent and observed confounders"
    else:
        confounding = f"{latent} latent and {observed} observed confounders"
    return f"{noise_label}, {mechanism}, {confounding}"


def _dataset_display_title(value: Any) -> str:
    """Standardise dataset text used in plot titles and legends."""
    text = str(value).replace("_", " ").strip().title()
    replacements = (
        ("(?i)\\bmodmultiomics\\b", "MultiOmics"),
        ("(?i)\\bmultiomics\\b", "MultiOmics"),
        ("(?i)\\bmodoncogene\\b", "Oncogene"),
        ("(?i)\\bnon[ -]?gaussian\\b", "Non Gaussian"),
        ("(?i)\\bhigh[ -]?gaussian\\b", "High Gaussian"),
        ("(?i)\\bnon[ -]?linear\\b", "Mixed"),
        ("(?i)\\bgaussian\\b", "Gaussian"),
        ("(?i)\\blinear\\b", "Linear"),
        ("(?i)\\bmixed\\b", "Mixed"),
        ("(?i)\\bmodular\\b", "Modular"),
        ("(?i)\\bconfounding\\b", "Confounding"),
        ("(?i)\\bwith\\b", "With"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    text = re.sub(
        "\\b(Linear|Mixed)\\s*,?\\s+(Non Gaussian|High Gaussian|Gaussian|Laplace)\\b",
        "\\2 \\1",
        text,
    )
    return text


def plot_rank_scatter_by_dataset(
    data: FinalVisualizationData,
    *,
    metric: str = "f1",
    datasets: Any = None,
    n_nodes: Any = None,
    graph_type: Any = None,
    noise: Any = None,
    mechanism: Any = None,
    obs_conf: Any = None,
    lat_conf: Any = None,
    models: Any = None,
    filters: Mapping[str, Any] | None = None,
    coverage_policy: str = "all",
    min_completed_runs: int | None = None,
    min_coverage_fraction: float | None = None,
    model_order: Sequence[str] | None = None,
    dataset_labels: Mapping[str, str] | None = None,
    dataset_colors: Mapping[str, str] | None = None,
    title: str | None = None,
    show_title: bool = True,
    show_metric_title: bool = True,
    xlabel: str = "",
    ylabel: str = "Wilcoxon-Holm rank",
    invert_rank_axis: bool = True,
    marker_size: float = 42,
    legend_ncols: int | None = None,
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    grid: bool = True,
    save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> tuple[plt.Figure, plt.Axes, dict[str, pd.DataFrame]]:
    """Scatter one precomputed model rank per selected dataset.

    Datasets are separated by colour and a small horizontal offset. Their
    canonical identifiers are converted to readable assumption descriptions
    in a legend placed below the plotting area.
    """
    kwargs = _filter_kwargs(
        datasets,
        n_nodes,
        graph_type,
        noise,
        mechanism,
        obs_conf,
        lat_conf,
        models,
        filters,
    )
    rank_rows, coverage, eligible = _rank_rows_with_coverage(
        data,
        metric=metric,
        filter_kwargs=kwargs,
        coverage_policy=coverage_policy,
        min_completed_runs=min_completed_runs,
        min_coverage_fraction=min_coverage_fraction,
    )
    rank_rows = rank_rows[rank_rows["model"].isin(eligible) & rank_rows["rank_available"]].copy()
    if rank_rows.empty:
        raise ValueError("No eligible precomputed ranks remain for the scatter plot.")
    model_names = _ordered_models(rank_rows["model"].unique(), model_order=model_order)
    dataset_names = list(dict.fromkeys(rank_rows["dataset"].astype(str)))
    supplied_labels = dict(dataset_labels or {})
    label_rows: list[dict[str, Any]] = []
    for dataset in dataset_names:
        row = rank_rows[rank_rows["dataset"].astype(str).eq(dataset)].iloc[0]
        label_rows.append(
            {
                "dataset": dataset,
                "dataset_label": supplied_labels.get(dataset, _pretty_dataset_label(row)),
            }
        )
    label_table = pd.DataFrame(label_rows)
    duplicated = label_table["dataset_label"].duplicated(keep=False)
    for index in label_table.index[duplicated]:
        dataset = label_table.at[index, "dataset"]
        row = rank_rows[rank_rows["dataset"].astype(str).eq(dataset)].iloc[0]
        details = []
        if pd.notna(row.get("n_nodes")):
            details.append(f"{int(row['n_nodes'])} nodes")
        if pd.notna(row.get("graph_type")):
            details.append(_dataset_display_title(row["graph_type"]))
        if details:
            label_table.at[index, "dataset_label"] += f" ({', '.join(details)})"
    if ax is None:
        if figsize is None:
            figsize = (max(9, 0.72 * len(model_names)), 6)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    color_lookup = {
        dataset: (dataset_colors or {}).get(
            dataset, OKABE_ITO_EXTENDED[index % len(OKABE_ITO_EXTENDED)]
        )
        for index, dataset in enumerate(dataset_names)
    }
    label_lookup = label_table.set_index("dataset")["dataset_label"].to_dict()
    model_positions = {model: index for index, model in enumerate(model_names)}
    offsets = (
        np.linspace(-0.28, 0.28, len(dataset_names)) if len(dataset_names) > 1 else np.array([0.0])
    )
    for offset, dataset in zip(offsets, dataset_names):
        rows = rank_rows[rank_rows["dataset"].astype(str).eq(dataset)]
        x = np.array([model_positions[model] for model in rows["model"]], dtype=float)
        ax.scatter(
            x + offset,
            rows["rank"].to_numpy(dtype=float),
            s=marker_size,
            color=color_lookup[dataset],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
            label=label_lookup[dataset],
            zorder=3,
        )
    metric_label, _ = _metric_meta(metric)
    ax.set_xticks(
        np.arange(len(model_names)),
        [_model_display_label(model) for model in model_names],
        rotation=35,
        ha="right",
    )
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    if show_title and (title is not None or show_metric_title):
        ax.set_title(title or f"{metric_label} rank by dataset", fontsize=title_size)
    ax.grid(axis="y", alpha=0.25, visible=grid)
    ax.grid(axis="x", visible=False)
    ax.tick_params(labelsize=font_size)
    if invert_rank_axis:
        ax.invert_yaxis()
    columns = legend_ncols or min(3, len(dataset_names))
    ax.legend(
        title="Dataset conditions",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncols=max(1, columns),
        fontsize=max(7, font_size - 1),
        title_fontsize=font_size,
        frameon=False,
    )
    _despine(ax, font_size=font_size)
    fig.tight_layout()
    _save(fig, save_path, dpi, transparent)
    return (
        fig,
        ax,
        {
            "ranks": rank_rows.reset_index(drop=True),
            "dataset_labels": label_table.reset_index(drop=True),
            "coverage": coverage.reset_index(drop=True),
            "model_order": pd.DataFrame({"model": model_names}),
        },
    )


def _compact_dataset_label(row: pd.Series) -> str:
    """Return a short, multi-line condition label for rank heatmaps."""
    graph_type = str(row.get("graph_type", "")).strip()
    if graph_type in GRAPH_TYPE_DISPLAY_LABELS:
        return GRAPH_TYPE_DISPLAY_LABELS[graph_type]
    readable = _pretty_dataset_label(row)
    noise, mechanism, _ = readable.split(", ", maxsplit=2)
    observed_value = pd.to_numeric(pd.Series([row.get("obs_conf", 0)]), errors="coerce").iloc[0]
    latent_value = pd.to_numeric(pd.Series([row.get("lat_conf", 0)]), errors="coerce").iloc[0]
    observed = int(observed_value) if pd.notna(observed_value) else 0
    latent = int(latent_value) if pd.notna(latent_value) else 0
    return f"{noise}\n{mechanism}\nL{latent} · O{observed}"


def plot_rank_heatmap_by_dataset(
    data: FinalVisualizationData,
    *,
    metric: str = "f1",
    datasets: Any = None,
    n_nodes: Any = None,
    graph_type: Any = None,
    noise: Any = None,
    mechanism: Any = None,
    obs_conf: Any = None,
    lat_conf: Any = None,
    models: Any = None,
    filters: Mapping[str, Any] | None = None,
    coverage_policy: str = "all",
    min_completed_runs: int | None = None,
    min_coverage_fraction: float | None = None,
    model_order: Sequence[str] | None = None,
    order_by: str = "median",
    dataset_labels: Mapping[str, str] | None = None,
    include_median: bool = True,
    highlight_model: str | None = "PRISM",
    title: str | None = None,
    show_title: bool = True,
    show_metric_title: bool = True,
    xlabel: str = "Dataset conditions",
    ylabel: str = "",
    cmap: Any = OKABE_ITO_SEQUENTIAL_R,
    annotate: bool = True,
    annotation_format: str = ".3g",
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> tuple[plt.Figure, plt.Axes, dict[str, pd.DataFrame]]:
    """Heatmap of each model's precomputed rank in every selected dataset.

    Models are ordered by median rank by default. Dataset identifiers are
    replaced by concise multi-line condition labels and missing ranks are
    rendered as grey cells. Lower (better) ranks use the green end of the
    colour scale.
    """
    kwargs = _filter_kwargs(
        datasets,
        n_nodes,
        graph_type,
        noise,
        mechanism,
        obs_conf,
        lat_conf,
        models,
        filters,
    )
    rank_rows, coverage, eligible = _rank_rows_with_coverage(
        data,
        metric=metric,
        filter_kwargs=kwargs,
        coverage_policy=coverage_policy,
        min_completed_runs=min_completed_runs,
        min_coverage_fraction=min_coverage_fraction,
    )
    rank_rows = rank_rows[rank_rows["model"].isin(eligible) & rank_rows["rank_available"]].copy()
    if rank_rows.empty:
        raise ValueError("No eligible precomputed ranks remain for the heatmap.")
    dataset_names = list(dict.fromkeys(rank_rows["dataset"].astype(str)))
    supplied_labels = dict(dataset_labels or {})
    label_rows: list[dict[str, Any]] = []
    for dataset in dataset_names:
        row = rank_rows[rank_rows["dataset"].astype(str).eq(dataset)].iloc[0]
        label_rows.append(
            {
                "dataset": dataset,
                "dataset_label": supplied_labels.get(dataset, _compact_dataset_label(row)),
            }
        )
    label_table = pd.DataFrame(label_rows)
    duplicated = label_table["dataset_label"].duplicated(keep=False)
    for index in label_table.index[duplicated]:
        dataset = label_table.at[index, "dataset"]
        row = rank_rows[rank_rows["dataset"].astype(str).eq(dataset)].iloc[0]
        details = []
        if pd.notna(row.get("n_nodes")):
            details.append(f"{int(row['n_nodes'])} nodes")
        if pd.notna(row.get("graph_type")):
            details.append(_dataset_display_title(row["graph_type"]))
        if details:
            label_table.at[index, "dataset_label"] += "\n" + " · ".join(details)
    rank_table = rank_rows.pivot(index="model", columns="dataset", values="rank")
    rank_table = rank_table.reindex(columns=dataset_names)
    model_summary = pd.DataFrame(
        {
            "median_rank": rank_table.median(axis=1, skipna=True),
            "mean_rank": rank_table.mean(axis=1, skipna=True),
            "best_rank": rank_table.min(axis=1, skipna=True),
            "worst_rank": rank_table.max(axis=1, skipna=True),
            "datasets_ranked": rank_table.notna().sum(axis=1),
        }
    )
    ordering = order_by.casefold()
    if model_order is not None:
        model_names = _ordered_models(rank_table.index, model_order=model_order)
    elif ordering == "median":
        model_names = model_summary.sort_values(
            ["median_rank", "mean_rank"], kind="stable"
        ).index.tolist()
    elif ordering in {"default", "manual"}:
        model_names = _ordered_models(rank_table.index)
    else:
        raise ValueError("order_by must be median, default, or manual.")
    rank_table = rank_table.reindex(model_names)
    model_summary = model_summary.reindex(model_names)
    label_lookup = label_table.set_index("dataset")["dataset_label"].to_dict()
    heatmap_table = rank_table.rename(columns=label_lookup).copy()
    if include_median:
        heatmap_table["Median\nrank"] = model_summary["median_rank"]
    if ax is None:
        if figsize is None:
            figsize = (
                max(9, 1.45 * heatmap_table.shape[1] + 3),
                max(5, 0.48 * heatmap_table.shape[0] + 2.5),
            )
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    finite_ranks = rank_table.to_numpy(dtype=float)
    finite_ranks = finite_ranks[np.isfinite(finite_ranks)]
    maximum_rank = float(finite_ranks.max()) if finite_ranks.size else 1.0
    _draw_table_heatmap(
        ax,
        heatmap_table,
        cmap=cmap,
        colorbar_label="Rank (1 = best)",
        annotate=annotate,
        annotation_format=annotation_format,
        vmin=1.0,
        vmax=max(1.0, maximum_rank),
        font_size=font_size,
    )
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, pad=7)
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
        tick.set_ha("center")
    if include_median:
        ax.axvline(len(dataset_names) - 0.5, color="white", linewidth=3)
    canonical_highlight = _canonical_model_name(highlight_model)
    if canonical_highlight in model_names:
        ax.get_yticklabels()[model_names.index(canonical_highlight)].set_fontweight("bold")
    ax.set_xlabel(xlabel, fontsize=font_size, labelpad=10)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel(ylabel, fontsize=font_size)
    metric_label, _ = _metric_meta(metric)
    if show_title and (title is not None or show_metric_title):
        ax.set_title(title or f"{metric_label} rank by dataset", fontsize=title_size, pad=14)
    _despine(ax, font_size=font_size)
    fig.tight_layout()
    _save(fig, save_path, dpi, transparent)
    return (
        fig,
        ax,
        {
            "ranks": rank_rows.reset_index(drop=True),
            "rank_table": rank_table,
            "heatmap_table": heatmap_table,
            "model_summary": model_summary.reset_index(names="model"),
            "dataset_labels": label_table.reset_index(drop=True),
            "coverage": coverage.reset_index(drop=True),
        },
    )


def _requested_groups(
    available: Iterable[Any],
    requested: Sequence[str] | Mapping[str, Sequence[str]] | str | None,
) -> dict[str, list[str]]:
    """Resolve display groups to the exact values present in a column."""
    available_values = list(dict.fromkeys(map(str, available)))
    if requested is None:
        return {value: [value] for value in available_values}
    raw_groups = (
        dict(requested)
        if isinstance(requested, Mapping)
        else {str(value): [str(value)] for value in _as_list(requested) or []}
    )
    groups: dict[str, list[str]] = {}
    for label, candidates in raw_groups.items():
        candidate_keys = {_normalise_token(value) for value in _as_list(candidates) or []}
        matched = [value for value in available_values if _normalise_token(value) in candidate_keys]
        if not matched:
            raise ValueError(
                f"Requested group '{label}' does not match any available value. Available: {available_values}"
            )
        groups[str(label)] = matched
    return groups


def _factor_performance_table(
    selected: pd.DataFrame,
    metric: str,
    *,
    factor: str,
    groups: Mapping[str, Sequence[str]],
    coverage_policy: str,
    min_completed_runs: int | None,
    min_coverage_fraction: float | None,
    model_order: Sequence[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    coverage_rows: list[pd.DataFrame] = []
    for label, values in groups.items():
        group = selected[selected[factor].astype(str).isin(map(str, values))]
        if group.empty:
            continue
        by_dataset, coverage = _coverage_from_metrics(group, metric)
        eligible = _eligible_models(
            by_dataset,
            coverage_policy=coverage_policy,
            min_completed_runs=min_completed_runs,
            min_coverage_fraction=min_coverage_fraction,
        )
        coverage = coverage.copy()
        coverage.insert(0, "level", label)
        coverage["included"] = coverage["model"].isin(eligible)
        coverage_rows.append(coverage)
        blocks = _block_means(group, metric)
        blocks = blocks[blocks["model"].isin(eligible)]
        for model, model_blocks in blocks.groupby("model", observed=True):
            values_array = model_blocks["metric_value"].to_numpy(dtype=float)
            rows.append(
                {
                    "level": label,
                    "model": model,
                    "mean": float(np.mean(values_array)),
                    "std": float(np.std(values_array, ddof=1)) if len(values_array) > 1 else np.nan,
                    "n_blocks": len(values_array),
                }
            )
    summary = pd.DataFrame(rows, columns=["level", "model", "mean", "std", "n_blocks"])
    if summary.empty:
        raise ValueError("No models satisfy the selected filters and coverage policy.")
    table = summary.pivot(index="model", columns="level", values="mean")
    table = table.reindex(
        index=_ordered_models(table.index, model_order=model_order),
        columns=list(groups),
    )
    coverage = pd.concat(coverage_rows, ignore_index=True) if coverage_rows else pd.DataFrame()
    return (table, summary, coverage)


def _draw_table_heatmap(
    ax: plt.Axes,
    table: pd.DataFrame,
    *,
    cmap: Any,
    colorbar_label: str,
    annotate: bool,
    annotation_format: str,
    center_zero: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
):
    values = table.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if center_zero and vmin is None and (vmax is None):
        limit = float(np.max(np.abs(finite))) if finite.size else 1.0
        vmin, vmax = (-(limit or 1.0), limit or 1.0)
    colour_map = plt.get_cmap(cmap).copy()
    colour_map.set_bad("#eeeeee")
    image = ax.imshow(
        np.ma.masked_invalid(values),
        cmap=colour_map,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    display_columns = [_dataset_display_title(value) for value in table.columns]
    ax.set_xticks(range(table.shape[1]), display_columns, rotation=35, ha="right")
    ax.set_yticks(range(table.shape[0]), [_model_display_label(model) for model in table.index])
    ax.tick_params(labelsize=font_size)
    ax.set_xticks(np.arange(-0.5, table.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, table.shape[0]), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", length=0)
    if annotate:
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                if not np.isfinite(value):
                    ax.text(column, row, "–", ha="center", va="center", color="0.4")
                    continue
                rgba = image.cmap(image.norm(value))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(
                    column,
                    row,
                    format(value, annotation_format),
                    ha="center",
                    va="center",
                    fontsize=max(6, font_size - 2),
                    color="white" if luminance < 0.48 else "black",
                )
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    colorbar.set_label(colorbar_label, fontsize=font_size)
    return image


def plot_noise_heatmap(
    data: FinalVisualizationData,
    *,
    metric: str = "f1",
    noises: Sequence[str] | Mapping[str, Sequence[str]] | str | None = (
        "Gaussian",
        "HighGaussian",
        "NonGaussian",
    ),
    mechanisms: Any = ("Linear",),
    datasets: Any = None,
    n_nodes: Any = None,
    graph_type: Any = "EB",
    obs_conf: Any = 0,
    lat_conf: Any = 0,
    models: Any = None,
    filters: Mapping[str, Any] | None = None,
    coverage_policy: str = "complete",
    min_completed_runs: int | None = None,
    min_coverage_fraction: float | None = None,
    model_order: Sequence[str] | None = None,
    title: str | None = None,
    show_title: bool = True,
    show_metric_title: bool = True,
    xlabel: str = "Noise distribution",
    ylabel: str = "",
    cmap: str | None = None,
    annotate: bool = True,
    annotation_format: str = ".3f",
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> tuple[plt.Figure, plt.Axes, dict[str, pd.DataFrame]]:
    """Plot model × noise performance for any selected mechanisms/noises."""
    selected = _filter_frame(
        data.metrics,
        datasets=datasets,
        n_nodes=n_nodes,
        graph_type=graph_type,
        mechanism=mechanisms,
        obs_conf=obs_conf,
        lat_conf=lat_conf,
        models=models,
        filters=filters,
    )
    groups = _requested_groups(selected["noise"].dropna().unique(), noises)
    table, summary, coverage = _factor_performance_table(
        selected,
        metric,
        factor="noise",
        groups=groups,
        coverage_policy=coverage_policy,
        min_completed_runs=min_completed_runs,
        min_coverage_fraction=min_coverage_fraction,
        model_order=model_order,
    )
    label, higher = _metric_meta(metric)
    if ax is None:
        if figsize is None:
            figsize = (1.35 * len(table.columns) + 3.5, 0.45 * len(table.index) + 2)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    _draw_table_heatmap(
        ax,
        table,
        cmap=cmap or (OKABE_ITO_SEQUENTIAL if higher else OKABE_ITO_SEQUENTIAL_R),
        colorbar_label=label,
        annotate=annotate,
        annotation_format=annotation_format,
        font_size=font_size,
    )
    mechanism_label = ", ".join(
        (_dataset_display_title(value) for value in _as_list(mechanisms) or ["all"])
    )
    if show_title and (title is not None or show_metric_title):
        ax.set_title(
            title or f"Noise sensitivity ({mechanism_label} mechanism)",
            fontsize=title_size,
        )
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    _despine(ax, font_size=font_size)
    fig.tight_layout()
    _save(fig, save_path, dpi, transparent)
    return (fig, ax, {"table": table, "summary": summary, "coverage": coverage})


def plot_mechanism_heatmap(
    data: FinalVisualizationData,
    *,
    metric: str = "f1",
    mechanisms: Any = ("Linear", "Mixed"),
    noises: Sequence[str] | Mapping[str, Sequence[str]] | str | None = (
        "Gaussian",
        "NonGaussian",
    ),
    datasets: Any = None,
    n_nodes: Any = None,
    graph_type: Any = "EB",
    obs_conf: Any = 0,
    lat_conf: Any = 0,
    models: Any = None,
    filters: Mapping[str, Any] | None = None,
    coverage_policy: str = "complete",
    min_completed_runs: int | None = None,
    min_coverage_fraction: float | None = None,
    model_order: Sequence[str] | None = None,
    title: str | None = None,
    show_title: bool = True,
    show_metric_title: bool = True,
    xlabel: str = "Mechanism",
    ylabel: str = "",
    cmap: str | None = None,
    annotate: bool = True,
    annotation_format: str = ".3f",
    figsize: tuple[float, float] | None = None,
    axes: Sequence[plt.Axes] | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> tuple[plt.Figure, np.ndarray, dict[str, dict[str, pd.DataFrame]]]:
    """Plot model × mechanism heatmaps, with one panel per selected noise group."""
    selected = _filter_frame(
        data.metrics,
        datasets=datasets,
        n_nodes=n_nodes,
        graph_type=graph_type,
        mechanism=mechanisms,
        obs_conf=obs_conf,
        lat_conf=lat_conf,
        models=models,
        filters=filters,
    )
    noise_groups = _requested_groups(selected["noise"].dropna().unique(), noises)
    outputs: dict[str, dict[str, pd.DataFrame]] = {}
    model_union: set[str] = set()
    for noise_label, noise_values in noise_groups.items():
        subset = selected[selected["noise"].astype(str).isin(noise_values)]
        mechanism_groups = _requested_groups(subset["mechanism"].dropna().unique(), mechanisms)
        table, summary, coverage = _factor_performance_table(
            subset,
            metric,
            factor="mechanism",
            groups=mechanism_groups,
            coverage_policy=coverage_policy,
            min_completed_runs=min_completed_runs,
            min_coverage_fraction=min_coverage_fraction,
            model_order=model_order,
        )
        outputs[noise_label] = {
            "table": table,
            "summary": summary,
            "coverage": coverage,
        }
        model_union.update(map(str, table.index))
    order = _ordered_models(model_union, model_order=model_order)
    all_finite = [output["table"].to_numpy(dtype=float).ravel() for output in outputs.values()]
    finite = np.concatenate(all_finite)
    finite = finite[np.isfinite(finite)]
    common_vmin = float(finite.min()) if finite.size else None
    common_vmax = float(finite.max()) if finite.size else None
    if axes is None:
        if figsize is None:
            figsize = (4.5 * len(outputs), 0.45 * len(order) + 2)
        fig, made_axes = plt.subplots(1, len(outputs), sharey=True, figsize=figsize, squeeze=False)
        axes_array = made_axes.ravel()
    else:
        axes_array = np.atleast_1d(axes)
        if len(axes_array) != len(outputs):
            raise ValueError(f"Expected {len(outputs)} axes, received {len(axes_array)}.")
        fig = axes_array[0].figure
    label, higher = _metric_meta(metric)
    for axis, (noise_label, output) in zip(axes_array, outputs.items()):
        table = output["table"].reindex(index=order)
        output["table"] = table
        _draw_table_heatmap(
            axis,
            table,
            cmap=cmap or (OKABE_ITO_SEQUENTIAL if higher else OKABE_ITO_SEQUENTIAL_R),
            colorbar_label=label,
            annotate=annotate,
            annotation_format=annotation_format,
            vmin=common_vmin,
            vmax=common_vmax,
            font_size=font_size,
        )
        if show_title and show_metric_title:
            axis.set_title(_dataset_display_title(noise_label), fontsize=title_size)
        axis.set_xlabel(xlabel, fontsize=font_size)
    axes_array[0].set_ylabel(ylabel, fontsize=font_size)
    if show_title and (title is not None or show_metric_title):
        fig.suptitle(
            title or "Mechanism sensitivity by noise distribution",
            fontsize=title_size + 1,
        )
    _despine(axes_array, font_size=font_size)
    fig.tight_layout()
    _save(fig, save_path, dpi, transparent)
    return (fig, axes_array, outputs)


def _assign_scalability_family(df: pd.DataFrame) -> pd.DataFrame:
    out = _add_dataset_attributes(df)
    if "scalability_family" not in out.columns:
        out["scalability_family"] = pd.NA
    dataset = out["dataset"].astype(str)
    missing = out["scalability_family"].isna()
    out.loc[
        missing & dataset.str.contains("_EB_obsconf0_latconf0_Gaussian_Linear", regex=False),
        "scalability_family",
    ] = "eb_gaussian_linear"
    out.loc[
        missing & dataset.str.contains("_EB_obsconf0_latconf0_Non_Gaussian_Mixed", regex=False),
        "scalability_family",
    ] = "eb_nongaussian_mixed"
    return out


def _prepare_scalability_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = _assign_scalability_family(df)
    _require_columns(
        out,
        ["dataset_seed", "model_seed", "dataset", "model", "status"],
        "scalability data",
    )
    if "timed_out" in out:
        out["timed_out"] = _as_bool(out["timed_out"])
    else:
        out["timed_out"] = out["status"].eq("timeout")
    out["finished"] = out["status"].eq("finished") & ~out["timed_out"]
    out["n_nodes"] = pd.to_numeric(out["n_nodes"], errors="coerce")
    if "runtime_seconds" in out:
        out["runtime_seconds"] = pd.to_numeric(out["runtime_seconds"], errors="coerce")
    return out


def load_scalability_data(
    data: FinalVisualizationData | None = None,
    *,
    scalability_paths: Sequence[str | Path] | None = None,
    project_root: str | Path | None = None,
    include_main_benchmark: bool = True,
    families: Sequence[str] | None = None,
    n_nodes: Sequence[int] | int | None = None,
) -> pd.DataFrame:
    """Load and combine the 10–40 and 70–500-node scalability results.

    Explicit ``scalability_paths`` may be supplied for future experiments.
    Returned rows retain a ``result_source`` column and are de-duplicated by
    benchmark run identity.
    """
    root = _project_root(project_root)
    frames: list[pd.DataFrame] = []
    if include_main_benchmark:
        if data is None:
            data = load_final_visualization_data(project_root=root)
        base = _prepare_scalability_frame(data.metrics)
        base = base[base["scalability_family"].notna()].copy()
        base["result_source"] = "main_benchmark"
        frames.append(base)
    if scalability_paths is None:
        from condition_data import load_results
        final = _prepare_scalability_frame(load_results("scalability"))
        final["result_source"] = "results/scalability"
        frames.append(final)
        scalability_paths = []
    for raw_path in scalability_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(f"Missing scalability CSV: {path}")
        frame = pd.read_csv(path, low_memory=False)
        _require_columns(
            frame,
            [
                "dataset_seed",
                "model_seed",
                "dataset",
                "model",
                "status",
                "runtime_seconds",
            ],
            f"scalability CSV {path}",
        )
        frame = _prepare_scalability_frame(frame)
        frame["result_source"] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError("No scalability result sources were selected.")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = _prepare_scalability_frame(combined)
    identity = ["dataset_seed", "dataset", "model_seed", "model"]
    duplicates = combined.duplicated(identity, keep=False)
    if duplicates.any():
        examples = combined.loc[duplicates, identity].head().to_dict("records")
        raise ValueError(f"Duplicate scalability run identities: {examples}")
    wanted_families = list(families or DEFAULT_SCALABILITY_FAMILIES)
    unknown = sorted(set(wanted_families) - set(combined["scalability_family"].dropna()))
    if unknown:
        raise ValueError(f"Unknown scalability families: {unknown}")
    combined = combined[combined["scalability_family"].isin(wanted_families)]
    if n_nodes is not None:
        combined = combined[combined["n_nodes"].isin(_as_list(n_nodes) or [])]
    if combined.empty:
        raise ValueError("No scalability rows remain after filtering.")
    return combined.sort_values(
        ["scalability_family", "n_nodes", "dataset_seed", "model", "model_seed"],
        kind="stable",
    ).reset_index(drop=True)


def _scalability_summaries(
    frame: pd.DataFrame,
    value: str,
    *,
    uncertainty: str | None,
    confidence_level: float,
    ci_method: str,
    n_bootstrap: int,
    random_state: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    if value not in work.columns:
        raise ValueError(f"Scalability data has no '{value}' column.")
    work[value] = pd.to_numeric(work[value], errors="coerce")
    work["_valid"] = work["finished"] & work[value].notna()
    coverage = (
        work.groupby(["scalability_family", "n_nodes", "model"], observed=True)
        .agg(
            successful_runs=("_valid", "sum"),
            expected_runs=("model_seed", "size"),
            dataset_seeds=("dataset_seed", "nunique"),
            timeouts=("timed_out", "sum"),
        )
        .reset_index()
    )
    coverage["completion_fraction"] = coverage["successful_runs"] / coverage["expected_runs"].clip(
        lower=1
    )
    valid = work[work["_valid"]].copy()
    if valid.empty:
        raise ValueError(f"No finished scalability runs contain '{value}'.")
    blocks = (
        valid.groupby(
            ["scalability_family", "n_nodes", "dataset_seed", "dataset", "model"],
            observed=True,
        )[value]
        .agg(value="mean", completed_model_seeds="size")
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for (family, nodes, model), group in blocks.groupby(
        ["scalability_family", "n_nodes", "model"], observed=True
    ):
        result = _uncertainty_summary(
            group["value"],
            uncertainty=uncertainty,
            confidence_level=confidence_level,
            ci_method=ci_method,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        rows.append({"scalability_family": family, "n_nodes": nodes, "model": model, **result})
    return (blocks, pd.DataFrame(rows), coverage)


def _scalability_axes(
    family_keys: Sequence[str],
    *,
    axes: Sequence[plt.Axes] | None,
    figsize: tuple[float, float] | None,
    show_coverage: bool = False,
) -> tuple[plt.Figure, np.ndarray, np.ndarray]:
    n_rows = 2 if show_coverage else 1
    if axes is None:
        fig, made_axes = plt.subplots(
            n_rows,
            len(family_keys),
            figsize=figsize or (5.2 * len(family_keys), 10 if show_coverage else 5),
            squeeze=False,
            gridspec_kw={"height_ratios": [1.25, 1] if show_coverage else [1]},
        )
        primary = made_axes[0]
        coverage_axes = made_axes[1] if show_coverage else np.asarray([], dtype=object)
        return (fig, primary, coverage_axes)
    axes_array = np.atleast_1d(axes).ravel()
    expected = n_rows * len(family_keys)
    if len(axes_array) != expected:
        raise ValueError(f"Expected {expected} axes, received {len(axes_array)}.")
    primary = axes_array[: len(family_keys)]
    coverage_axes = (
        axes_array[len(family_keys) :] if show_coverage else np.asarray([], dtype=object)
    )
    return (axes_array[0].figure, primary, coverage_axes)


def _draw_scalability_coverage(
    axes: Sequence[plt.Axes],
    family_keys: Sequence[str],
    coverage: pd.DataFrame,
    model_names: Sequence[str],
    family_labels: Mapping[str, str],
    *,
    font_size: float,
    show_title: bool,
    show_family_titles: bool,
) -> None:
    """Draw the percentage of completed fits at every scalability point."""
    for index, (axis, family) in enumerate(zip(axes, family_keys)):
        family_coverage = coverage[coverage["scalability_family"].eq(family)].copy()
        node_values = sorted(family_coverage["n_nodes"].dropna().unique())
        fractions = family_coverage.pivot(
            index="model", columns="n_nodes", values="completion_fraction"
        ).reindex(index=model_names, columns=node_values)
        colour_map = OKABE_ITO_SEQUENTIAL.copy()
        colour_map.set_bad("#eeeeee")
        axis.imshow(
            np.ma.masked_invalid(fractions.to_numpy(dtype=float)),
            cmap=colour_map,
            vmin=0,
            vmax=1,
            aspect="auto",
        )
        lookup = family_coverage.set_index(["model", "n_nodes"])
        for row, model in enumerate(model_names):
            for column, nodes in enumerate(node_values):
                if (model, nodes) not in lookup.index:
                    axis.text(
                        column,
                        row,
                        "–",
                        ha="center",
                        va="center",
                        color="0.45",
                        fontsize=max(5, font_size - 3),
                    )
                    continue
                record = lookup.loc[model, nodes]
                fraction = float(record["completion_fraction"])
                axis.text(
                    column,
                    row,
                    f"{fraction:.0%}",
                    ha="center",
                    va="center",
                    fontsize=max(6, font_size - 3),
                    color="white" if fraction < 0.35 else "black",
                )
        axis.set_xticks(range(len(node_values)), [int(value) for value in node_values])
        axis.set_yticks(
            range(len(model_names)),
            [_model_display_label(model) for model in model_names] if index == 0 else [],
        )
        axis.set_xlabel("Number of observed nodes", fontsize=font_size)
        if index == 0:
            axis.set_ylabel("", fontsize=font_size)
        if show_title and show_family_titles:
            axis.set_title(
                _dataset_display_title(family_labels.get(family, family)),
                fontsize=font_size + 1,
            )
        axis.tick_params(labelsize=max(6, font_size - 1))
        axis.set_xticks(np.arange(-0.5, len(node_values)), minor=True)
        axis.set_yticks(np.arange(-0.5, len(model_names)), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.7)
        axis.tick_params(which="minor", length=0)


def _set_scalability_xscale(axis: plt.Axes, node_values: Sequence[float], *, x_scale: str) -> None:
    """Apply a readable node-count scale while keeping exact node ticks."""
    scale = x_scale.casefold()
    if scale not in {"linear", "log"}:
        raise ValueError("x_scale must be 'linear' or 'log'.")
    values = sorted({float(value) for value in node_values if pd.notna(value)})
    if scale == "log":
        if values and min(values) <= 0:
            raise ValueError("A logarithmic x-axis requires positive node counts.")
        axis.set_xscale("log")
        axis.xaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        axis.set_xscale("linear")
    if values:
        axis.set_xticks(values)
        labels = [
            str(int(value)) if float(value).is_integer() else f"{value:g}" for value in values
        ]
        rotate = len(values) > 6
        axis.set_xticklabels(
            labels, rotation=35 if rotate else 0, ha="right" if rotate else "center"
        )
        axis.tick_params(axis="x", which="minor", labelbottom=False)


def _independent_family_axes(
    family_keys: Sequence[str],
    *,
    axes: Sequence[plt.Axes] | None,
    figsize: tuple[float, float] | None,
) -> tuple[dict[str, plt.Figure], dict[str, plt.Axes]]:
    """Return one independent figure and axis per scalability family."""
    figures: dict[str, plt.Figure] = {}
    axes_by_family: dict[str, plt.Axes] = {}
    if axes is not None:
        axes_array = np.atleast_1d(axes).ravel()
        if len(axes_array) != len(family_keys):
            raise ValueError(f"Expected {len(family_keys)} axes, received {len(axes_array)}.")
        for family, axis in zip(family_keys, axes_array):
            figures[family] = axis.figure
            axes_by_family[family] = axis
        return (figures, axes_by_family)
    for family in family_keys:
        fig, axis = plt.subplots(figsize=figsize or (7.2, 5.2))
        figures[family] = fig
        axes_by_family[family] = axis
    return (figures, axes_by_family)


def _save_scalability_figures(
    figures: Mapping[str, plt.Figure],
    save_path: str | Path | None,
    *,
    dpi: int,
    transparent: bool,
) -> None:
    """Save a family-to-figure mapping using distinct, predictable names."""
    if save_path is None:
        return
    raw = str(save_path)
    base = Path(save_path)
    multiple = len(figures) > 1
    for family, figure in figures.items():
        safe_family = re.sub("[^A-Za-z0-9]+", "_", family).strip("_")
        if "{family}" in raw:
            destination = Path(raw.format(family=safe_family))
        elif base.suffix:
            destination = (
                base.with_name(f"{base.stem}_{safe_family}{base.suffix}") if multiple else base
            )
        else:
            destination = base / f"{safe_family}.png"
        _save(figure, destination, dpi, transparent)


def _draw_scalability_curve(
    axis: plt.Axes,
    family_summary: pd.DataFrame,
    model_names: Sequence[str],
    color_lookup: Mapping[str, str],
    *,
    uncertainty: str | None,
    family_title: str,
    show_title: bool,
    show_family_titles: bool,
    x_scale: str,
    xlabel: str,
    ylabel: str,
    font_size: float,
    title_size: float,
    y_scale: str = "linear",
    errorbar_alpha: float = 0.35,
) -> None:
    """Draw one F1 or runtime scalability panel."""
    for model in model_names:
        rows = family_summary[family_summary["model"].eq(model)].sort_values("n_nodes")
        if rows.empty:
            continue
        yerr = None
        if uncertainty is not None:
            yerr = np.vstack([rows["mean"] - rows["lower"], rows["upper"] - rows["mean"]])
        errorbar = axis.errorbar(
            rows["n_nodes"],
            rows["mean"],
            yerr=yerr,
            marker="o",
            markersize=4,
            linewidth=1.4,
            capsize=2,
            color=color_lookup[model],
            label=_model_display_label(model),
        )
        if yerr is not None:
            for cap in errorbar[1]:
                cap.set_alpha(errorbar_alpha)
            for bars in errorbar[2]:
                bars.set_alpha(errorbar_alpha)
    if y_scale != "linear":
        axis.set_yscale(y_scale)
    if show_title and show_family_titles:
        axis.set_title(_dataset_display_title(family_title), fontsize=title_size)
    _set_scalability_xscale(axis, family_summary["n_nodes"].dropna().unique(), x_scale=x_scale)
    axis.set_xlabel(xlabel, fontsize=font_size)
    axis.set_ylabel(ylabel, fontsize=font_size)
    axis.grid(alpha=0.22, which="both" if y_scale != "linear" else "major")
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=max(6, font_size - 1))


def plot_scalability_f1(
    data: FinalVisualizationData | None = None,
    *,
    scalability_data: pd.DataFrame | None = None,
    scalability_paths: Sequence[str | Path] | None = None,
    project_root: str | Path | None = None,
    metric: str = "f1",
    families: Sequence[str] | None = None,
    family_labels: Mapping[str, str] | None = None,
    n_nodes: Sequence[int] | int | None = None,
    exclude_n_nodes: Sequence[int] | int | None = None,
    include_20_nodes: bool = True,
    models: Sequence[str] | str | None = None,
    uncertainty: str | None = "ci",
    confidence_level: float = 0.95,
    ci_method: str = "bootstrap",
    n_bootstrap: int = 2000,
    random_state: int | None = 42,
    model_order: Sequence[str] | None = None,
    model_colors: Mapping[str, str] | None = None,
    separate_families: bool = True,
    show_coverage: bool = False,
    separate_coverage: bool = True,
    coverage_title: str | None = None,
    zoom_max_nodes: int | None = 70,
    title: str | None = None,
    show_title: bool = True,
    show_metric_title: bool = False,
    show_family_titles: bool = True,
    x_scale: str = "log",
    xlabel: str = "Number of observed nodes",
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = None,
    axes: Sequence[plt.Axes] | None = None,
    coverage_figsize: tuple[float, float] | None = None,
    coverage_axes: Sequence[plt.Axes] | None = None,
    zoom_figsize: tuple[float, float] | None = None,
    zoom_axes: Sequence[plt.Axes] | None = None,
    zoom_x_scale: str = "linear",
    errorbar_alpha: float = 0.35,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    save_path: str | Path | None = None,
    coverage_save_path: str | Path | None = None,
    zoom_save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> tuple[plt.Figure, np.ndarray, dict[str, Any]]:
    """Plot F1 against node count with one figure per dataset family.

    By default the full-range plots use a logarithmic x-axis and an additional
    figure per family is produced for node counts up to ``zoom_max_nodes``.
    Coverage/timeouts are also independent figures when requested. Set
    Pass one or more counts through ``exclude_n_nodes`` to omit them from all
    outputs; by default no node counts are excluded. ``include_20_nodes`` is
    retained for backwards compatibility. Set ``separate_families=False`` to
    recover the former multi-panel layout.
    """
    family_keys = list(families or DEFAULT_SCALABILITY_FAMILIES)
    labels = {
        key: _dataset_display_title(value)
        for key, value in {
            **DEFAULT_SCALABILITY_FAMILIES,
            **dict(family_labels or {}),
        }.items()
    }
    frame = (
        load_scalability_data(
            data,
            scalability_paths=scalability_paths,
            project_root=project_root,
            families=family_keys,
            n_nodes=n_nodes,
        )
        if scalability_data is None
        else _prepare_scalability_frame(scalability_data)
    )
    frame = frame[frame["scalability_family"].isin(family_keys)].copy()
    if n_nodes is not None:
        frame = frame[frame["n_nodes"].isin(_as_list(n_nodes) or [])]
    excluded_nodes = {int(value) for value in _as_list(exclude_n_nodes) or []}
    if not include_20_nodes:
        excluded_nodes.add(20)
    if excluded_nodes:
        frame = frame[~frame["n_nodes"].isin(excluded_nodes)]
    if models is not None:
        frame = frame[_match_values(frame["model"], models, "model")]
    blocks, summary, coverage = _scalability_summaries(
        frame,
        metric,
        uncertainty=uncertainty,
        confidence_level=confidence_level,
        ci_method=ci_method,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    model_names = _ordered_models(summary["model"].unique(), model_order=model_order)
    color_lookup = dict(zip(model_names, _colors(model_names, model_colors)))
    metric_label, _ = _metric_meta(metric)
    if separate_families:
        figures, axes_by_family = _independent_family_axes(family_keys, axes=axes, figsize=figsize)
        for family, axis in axes_by_family.items():
            _draw_scalability_curve(
                axis,
                summary[summary["scalability_family"].eq(family)],
                model_names,
                color_lookup,
                uncertainty=uncertainty,
                family_title=labels.get(family, family),
                show_title=show_title,
                show_family_titles=show_family_titles,
                x_scale=x_scale,
                xlabel=xlabel,
                ylabel=ylabel or metric_label,
                font_size=font_size,
                title_size=title_size,
                errorbar_alpha=errorbar_alpha,
            )
            if show_title and title is not None:
                figures[family].suptitle(title, fontsize=title_size + 1)
            elif show_title and show_metric_title:
                figures[family].suptitle(f"{metric_label} scalability", fontsize=title_size + 1)
            _despine(axis, font_size=font_size)
            figures[family].tight_layout()
        axes_array = np.asarray(list(axes_by_family.values()), dtype=object)
        fig = figures[family_keys[0]]
        _save_scalability_figures(figures, save_path, dpi=dpi, transparent=transparent)
    else:
        combined_coverage = show_coverage and (not separate_coverage)
        fig, axes_array, combined_coverage_axes = _scalability_axes(
            family_keys, axes=axes, figsize=figsize, show_coverage=combined_coverage
        )
        figures = {family: fig for family in family_keys}
        axes_by_family = dict(zip(family_keys, axes_array))
        for index, (family, axis) in enumerate(axes_by_family.items()):
            _draw_scalability_curve(
                axis,
                summary[summary["scalability_family"].eq(family)],
                model_names,
                color_lookup,
                uncertainty=uncertainty,
                family_title=labels.get(family, family),
                show_title=show_title,
                show_family_titles=show_family_titles,
                x_scale=x_scale,
                xlabel=xlabel,
                ylabel=ylabel or metric_label if index == 0 else "",
                font_size=font_size,
                title_size=title_size,
                errorbar_alpha=errorbar_alpha,
            )
            if index != len(family_keys) - 1 and axis.get_legend() is not None:
                axis.get_legend().remove()
        if show_title and title is not None:
            fig.suptitle(title, fontsize=title_size + 1)
        elif show_title and show_metric_title:
            fig.suptitle(f"{metric_label} scalability", fontsize=title_size + 1)
        _despine(axes_array, font_size=font_size)
        fig.tight_layout()
        _save(fig, save_path, dpi, transparent)
    zoom_figures: dict[str, plt.Figure] = {}
    zoom_axes_by_family: dict[str, plt.Axes] = {}
    zoom_summary = summary.iloc[0:0].copy()
    if zoom_max_nodes is not None:
        zoom_summary = summary[summary["n_nodes"].le(zoom_max_nodes)].copy()
        zoom_figures, zoom_axes_by_family = _independent_family_axes(
            family_keys, axes=zoom_axes, figsize=zoom_figsize or figsize
        )
        for family, axis in zoom_axes_by_family.items():
            family_zoom = zoom_summary[zoom_summary["scalability_family"].eq(family)]
            _draw_scalability_curve(
                axis,
                family_zoom,
                model_names,
                color_lookup,
                uncertainty=uncertainty,
                family_title=labels.get(family, family),
                show_title=show_title,
                show_family_titles=show_family_titles,
                x_scale=zoom_x_scale,
                xlabel=xlabel,
                ylabel=ylabel or metric_label,
                font_size=font_size,
                title_size=title_size,
                errorbar_alpha=errorbar_alpha,
            )
            _despine(axis, font_size=font_size)
            zoom_figures[family].tight_layout()
        _save_scalability_figures(zoom_figures, zoom_save_path, dpi=dpi, transparent=transparent)
    coverage_figures: dict[str, plt.Figure] = {}
    coverage_axes_by_family: dict[str, plt.Axes] = {}
    coverage_axes_array = np.asarray([], dtype=object)
    coverage_fig: plt.Figure | None = None
    if show_coverage:
        if not separate_families and (not separate_coverage):
            coverage_axes_array = combined_coverage_axes
            coverage_fig = fig
            coverage_figures = {family: fig for family in family_keys}
            coverage_axes_by_family = dict(zip(family_keys, coverage_axes_array))
        else:
            coverage_figures, coverage_axes_by_family = _independent_family_axes(
                family_keys,
                axes=coverage_axes,
                figsize=coverage_figsize or (7.2, max(5.5, 0.38 * len(model_names) + 1.8)),
            )
            coverage_axes_array = np.asarray(list(coverage_axes_by_family.values()), dtype=object)
            coverage_fig = coverage_figures[family_keys[0]]
        for family, axis in coverage_axes_by_family.items():
            _draw_scalability_coverage(
                [axis],
                [family],
                coverage,
                model_names,
                labels,
                font_size=font_size,
                show_title=show_title,
                show_family_titles=show_family_titles,
            )
            if show_title and coverage_title:
                coverage_figures[family].suptitle(coverage_title, fontsize=title_size + 1)
            _despine(axis, font_size=font_size)
            coverage_figures[family].tight_layout()
        if coverage_fig is None and coverage_figures:
            coverage_fig = coverage_figures[family_keys[0]]
        _save_scalability_figures(
            coverage_figures, coverage_save_path, dpi=dpi, transparent=transparent
        )
    return (
        fig,
        axes_array,
        {
            "summary": summary,
            "block_means": blocks,
            "coverage": coverage,
            "raw_results": frame,
            "figures": figures,
            "axes_by_family": axes_by_family,
            "zoom_figures": zoom_figures,
            "zoom_axes": zoom_axes_by_family,
            "zoom_summary": zoom_summary,
            "coverage_figure": coverage_fig,
            "coverage_axes": coverage_axes_array,
            "coverage_figures": coverage_figures,
            "coverage_axes_by_family": coverage_axes_by_family,
            "excluded_n_nodes": sorted(excluded_nodes),
        },
    )


def _scalability_highlight_models(
    summary: pd.DataFrame,
    family: str,
    *,
    highlight_models: Sequence[str] | str | None,
    highlight_n_models: int,
    highlight_at_nodes: int | None,
) -> list[str]:
    """Select a compact set of models for readable scalability plots."""
    family_summary = summary[summary["scalability_family"].eq(family)]
    present = set(family_summary["model"].astype(str))
    if highlight_models is not None:
        requested = [
            str(_canonical_model_name(model)) for model in _as_list(highlight_models) or []
        ]
        return [model for model in requested if model in present]
    if highlight_n_models < 1:
        raise ValueError("highlight_n_models must be at least 1.")
    node_values = sorted(family_summary["n_nodes"].dropna().unique())
    if not node_values:
        return []
    if highlight_at_nodes is None:
        target = node_values[-1]
    else:
        eligible_nodes = [value for value in node_values if value <= highlight_at_nodes]
        target = eligible_nodes[-1] if eligible_nodes else node_values[0]
    scores = (
        family_summary[family_summary["n_nodes"].eq(target)]
        .sort_values("mean", ascending=False)["model"]
        .astype(str)
        .tolist()
    )
    if len(scores) < highlight_n_models:
        fallback = (
            family_summary.groupby("model", observed=True)["mean"]
            .mean()
            .sort_values(ascending=False)
            .index.astype(str)
            .tolist()
        )
        scores.extend((model for model in fallback if model not in scores))
    return scores[:highlight_n_models]


def _draw_equal_spaced_scalability(
    axis: plt.Axes,
    family_summary: pd.DataFrame,
    *,
    highlighted_models: Sequence[str],
    color_lookup: Mapping[str, Any],
    ylabel: str,
    family_title: str,
    show_title: bool,
    show_family_titles: bool,
    show_other_models: bool,
    reference_line: float | None,
    font_size: float,
    title_size: float,
) -> None:
    """Draw discrete node counts at equal distances with subdued context."""
    node_values = sorted(family_summary["n_nodes"].dropna().unique())
    positions = {value: index for index, value in enumerate(node_values)}
    highlighted = set(highlighted_models)
    model_names = _ordered_models(family_summary["model"].unique())
    for model in model_names:
        is_highlighted = model in highlighted
        if not is_highlighted and (not show_other_models):
            continue
        rows = family_summary[family_summary["model"].eq(model)].sort_values("n_nodes")
        x = rows["n_nodes"].map(positions).to_numpy(dtype=float)
        colour = color_lookup[model]
        axis.plot(
            x,
            rows["mean"],
            marker="o",
            markersize=4.5 if is_highlighted else 3,
            linewidth=2 if is_highlighted else 0.8,
            alpha=1 if is_highlighted else 0.28,
            color=colour,
            label=_model_display_label(model) if is_highlighted else "_nolegend_",
            zorder=3 if is_highlighted else 1,
        )
        if is_highlighted and {"lower", "upper"}.issubset(rows.columns):
            lower = rows["lower"].to_numpy(dtype=float)
            upper = rows["upper"].to_numpy(dtype=float)
            if np.isfinite(lower).any() and np.isfinite(upper).any():
                axis.fill_between(x, lower, upper, color=colour, alpha=0.12, linewidth=0, zorder=2)
    if reference_line is not None:
        axis.axhline(reference_line, color="0.35", linestyle="--", linewidth=1, alpha=0.75)
    axis.set_xticks(
        range(len(node_values)),
        [str(int(value)) if float(value).is_integer() else f"{value:g}" for value in node_values],
    )
    axis.set_xlabel("Number of observed nodes", fontsize=font_size)
    axis.set_ylabel(ylabel, fontsize=font_size)
    if show_title and show_family_titles:
        axis.set_title(_dataset_display_title(family_title), fontsize=title_size)
    axis.grid(axis="y", alpha=0.2)
    if highlighted_models:
        axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=max(6, font_size - 1))


def plot_scalability_f1_highlighted(
    scalability_output: Mapping[str, Any],
    *,
    highlight_models: Sequence[str] | str | None = None,
    highlight_n_models: int = 5,
    highlight_at_nodes: int | None = 70,
    show_other_models: bool = True,
    family_labels: Mapping[str, str] | None = None,
    model_colors: Mapping[str, str] | None = None,
    show_title: bool = True,
    show_family_titles: bool = True,
    figsize: tuple[float, float] | None = None,
    axes: Sequence[plt.Axes] | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> tuple[plt.Figure, np.ndarray, dict[str, Any]]:
    """Readable F1 curves with equally spaced node counts and highlights."""
    if "summary" not in scalability_output:
        raise ValueError("scalability_output must contain a 'summary' table.")
    summary = scalability_output["summary"].copy()
    _require_columns(
        summary,
        ["scalability_family", "n_nodes", "model", "mean"],
        "scalability summary",
    )
    family_keys = list(dict.fromkeys(summary["scalability_family"].astype(str)))
    labels = {**DEFAULT_SCALABILITY_FAMILIES, **dict(family_labels or {})}
    model_names = _ordered_models(summary["model"].unique())
    colors = dict(zip(model_names, _colors(model_names, model_colors)))
    figures, axes_by_family = _independent_family_axes(family_keys, axes=axes, figsize=figsize)
    selected_rows: list[dict[str, str]] = []
    for family, axis in axes_by_family.items():
        selected = _scalability_highlight_models(
            summary,
            family,
            highlight_models=highlight_models,
            highlight_n_models=highlight_n_models,
            highlight_at_nodes=highlight_at_nodes,
        )
        selected_rows.extend(({"scalability_family": family, "model": model} for model in selected))
        _draw_equal_spaced_scalability(
            axis,
            summary[summary["scalability_family"].eq(family)],
            highlighted_models=selected,
            color_lookup=colors,
            ylabel=F1_LABEL,
            family_title=labels.get(family, family),
            show_title=show_title,
            show_family_titles=show_family_titles,
            show_other_models=show_other_models,
            reference_line=None,
            font_size=font_size,
            title_size=title_size,
        )
        _despine(axis, font_size=font_size)
        figures[family].tight_layout()
    _save_scalability_figures(figures, save_path, dpi=dpi, transparent=transparent)
    axes_array = np.asarray(list(axes_by_family.values()), dtype=object)
    return (
        figures[family_keys[0]],
        axes_array,
        {
            "figures": figures,
            "axes_by_family": axes_by_family,
            "highlighted_models": pd.DataFrame(selected_rows),
            "summary": summary,
        },
    )


def plot_scalability_f1_heatmap(
    scalability_output: Mapping[str, Any],
    *,
    family_labels: Mapping[str, str] | None = None,
    model_order: Sequence[str] | None = None,
    cmap: Any = OKABE_ITO_SEQUENTIAL,
    annotate: bool = True,
    annotation_format: str = ".2f",
    vmin: float | None = 0,
    vmax: float | None = 1,
    show_title: bool = True,
    show_family_titles: bool = True,
    figsize: tuple[float, float] | None = None,
    axes: Sequence[plt.Axes] | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> tuple[plt.Figure, np.ndarray, dict[str, Any]]:
    """Heatmaps of F1 by model and node count, one figure per family."""
    if "summary" not in scalability_output:
        raise ValueError("scalability_output must contain a 'summary' table.")
    summary = scalability_output["summary"].copy()
    family_keys = list(dict.fromkeys(summary["scalability_family"].astype(str)))
    labels = {**DEFAULT_SCALABILITY_FAMILIES, **dict(family_labels or {})}
    order = _ordered_models(summary["model"].unique(), model_order=model_order)
    default_size = (
        max(7.2, 1.05 * summary["n_nodes"].nunique() + 3.5),
        max(5.5, 0.42 * len(order) + 1.8),
    )
    figures, axes_by_family = _independent_family_axes(
        family_keys, axes=axes, figsize=figsize or default_size
    )
    tables: dict[str, pd.DataFrame] = {}
    for family, axis in axes_by_family.items():
        rows = summary[summary["scalability_family"].eq(family)]
        nodes = sorted(rows["n_nodes"].dropna().unique())
        table = rows.pivot(index="model", columns="n_nodes", values="mean")
        table = table.reindex(index=order, columns=nodes)
        tables[family] = table
        _draw_table_heatmap(
            axis,
            table,
            cmap=cmap,
            colorbar_label=F1_LABEL,
            annotate=annotate,
            annotation_format=annotation_format,
            vmin=vmin,
            vmax=vmax,
            font_size=font_size,
        )
        axis.set_xlabel("Number of observed nodes", fontsize=font_size)
        axis.set_ylabel("", fontsize=font_size)
        if show_title and show_family_titles:
            axis.set_title(_dataset_display_title(labels.get(family, family)), fontsize=title_size)
        _despine(axis, font_size=font_size)
        figures[family].tight_layout()
    _save_scalability_figures(figures, save_path, dpi=dpi, transparent=transparent)
    axes_array = np.asarray(list(axes_by_family.values()), dtype=object)
    return (
        figures[family_keys[0]],
        axes_array,
        {
            "figures": figures,
            "axes_by_family": axes_by_family,
            "tables": tables,
            "summary": summary,
        },
    )


def plot_scalability_f1_retention(
    scalability_output: Mapping[str, Any],
    *,
    baseline_n_nodes: int | None = None,
    highlight_models: Sequence[str] | str | None = None,
    highlight_n_models: int = 5,
    highlight_at_nodes: int | None = 70,
    show_other_models: bool = True,
    uncertainty: str | None = "ci",
    confidence_level: float = 0.95,
    ci_method: str = "bootstrap",
    n_bootstrap: int = 2000,
    random_state: int | None = 42,
    family_labels: Mapping[str, str] | None = None,
    model_colors: Mapping[str, str] | None = None,
    show_title: bool = True,
    show_family_titles: bool = True,
    figsize: tuple[float, float] | None = None,
    axes: Sequence[plt.Axes] | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> tuple[plt.Figure, np.ndarray, dict[str, Any]]:
    """Plot paired F1 retention relative to each seed's smallest graph."""
    if "block_means" not in scalability_output or "summary" not in scalability_output:
        raise ValueError("scalability_output must contain 'block_means' and 'summary'.")
    blocks = scalability_output["block_means"].copy()
    absolute_summary = scalability_output["summary"].copy()
    family_keys = list(dict.fromkeys(blocks["scalability_family"].astype(str)))
    retention_frames: list[pd.DataFrame] = []
    baseline_rows: list[dict[str, Any]] = []
    for family in family_keys:
        family_blocks = blocks[blocks["scalability_family"].eq(family)].copy()
        available = sorted(family_blocks["n_nodes"].dropna().unique())
        baseline = baseline_n_nodes if baseline_n_nodes is not None else available[0]
        if baseline not in available:
            raise ValueError(f"baseline_n_nodes={baseline} is unavailable for family '{family}'.")
        reference = family_blocks[family_blocks["n_nodes"].eq(baseline)][
            ["model", "dataset_seed", "value"]
        ].rename(columns={"value": "baseline_f1"})
        paired = family_blocks.merge(
            reference, on=["model", "dataset_seed"], how="inner", validate="many_to_one"
        )
        paired["retention"] = paired["value"] / paired["baseline_f1"].replace(0, np.nan)
        paired = paired.dropna(subset=["retention"])
        retention_frames.append(paired)
        baseline_rows.append({"scalability_family": family, "baseline_n_nodes": baseline})
    retention_blocks = pd.concat(retention_frames, ignore_index=True)
    rows: list[dict[str, Any]] = []
    for (family, nodes, model), group in retention_blocks.groupby(
        ["scalability_family", "n_nodes", "model"], observed=True
    ):
        stats_row = _uncertainty_summary(
            group["retention"],
            uncertainty=uncertainty,
            confidence_level=confidence_level,
            ci_method=ci_method,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        rows.append(
            {
                "scalability_family": family,
                "n_nodes": nodes,
                "model": model,
                **stats_row,
            }
        )
    retention_summary = pd.DataFrame(rows)
    labels = {**DEFAULT_SCALABILITY_FAMILIES, **dict(family_labels or {})}
    model_names = _ordered_models(retention_summary["model"].unique())
    colors = dict(zip(model_names, _colors(model_names, model_colors)))
    figures, axes_by_family = _independent_family_axes(family_keys, axes=axes, figsize=figsize)
    selected_rows: list[dict[str, str]] = []
    for family, axis in axes_by_family.items():
        selected = _scalability_highlight_models(
            absolute_summary,
            family,
            highlight_models=highlight_models,
            highlight_n_models=highlight_n_models,
            highlight_at_nodes=highlight_at_nodes,
        )
        selected_rows.extend(({"scalability_family": family, "model": model} for model in selected))
        _draw_equal_spaced_scalability(
            axis,
            retention_summary[retention_summary["scalability_family"].eq(family)],
            highlighted_models=selected,
            color_lookup=colors,
            ylabel=f"{F1_LABEL} retention relative to baseline",
            family_title=labels.get(family, family),
            show_title=show_title,
            show_family_titles=show_family_titles,
            show_other_models=show_other_models,
            reference_line=1.0,
            font_size=font_size,
            title_size=title_size,
        )
        _despine(axis, font_size=font_size)
        figures[family].tight_layout()
    _save_scalability_figures(figures, save_path, dpi=dpi, transparent=transparent)
    axes_array = np.asarray(list(axes_by_family.values()), dtype=object)
    return (
        figures[family_keys[0]],
        axes_array,
        {
            "figures": figures,
            "axes_by_family": axes_by_family,
            "highlighted_models": pd.DataFrame(selected_rows),
            "retention_summary": retention_summary,
            "retention_blocks": retention_blocks,
            "baselines": pd.DataFrame(baseline_rows),
        },
    )


def plot_scalability_runtime(
    data: FinalVisualizationData | None = None,
    *,
    scalability_data: pd.DataFrame | None = None,
    scalability_paths: Sequence[str | Path] | None = None,
    project_root: str | Path | None = None,
    families: Sequence[str] | None = None,
    family_labels: Mapping[str, str] | None = None,
    n_nodes: Sequence[int] | int | None = None,
    exclude_n_nodes: Sequence[int] | int | None = None,
    include_20_nodes: bool = True,
    models: Sequence[str] | str | None = None,
    top_n_models: int = 5,
    survival_completion_fraction: float = 0.8,
    uncertainty: str | None = "ci",
    confidence_level: float = 0.95,
    ci_method: str = "bootstrap",
    n_bootstrap: int = 2000,
    random_state: int | None = 42,
    model_order: Sequence[str] | None = None,
    model_colors: Mapping[str, str] | None = None,
    separate_families: bool = True,
    show_coverage: bool = False,
    separate_coverage: bool = True,
    zoom_max_nodes: int | None = 70,
    title: str | None = None,
    show_title: bool = True,
    show_metric_title: bool = False,
    show_family_titles: bool = True,
    xlabel: str = "Number of observed nodes",
    ylabel: str = "Runtime (seconds, log scale)",
    x_scale: str = "linear",
    zoom_x_scale: str = "linear",
    errorbar_alpha: float = 0.35,
    figsize: tuple[float, float] | None = None,
    axes: Sequence[plt.Axes] | None = None,
    zoom_figsize: tuple[float, float] | None = None,
    zoom_axes: Sequence[plt.Axes] | None = None,
    coverage_figsize: tuple[float, float] | None = None,
    coverage_axes: Sequence[plt.Axes] | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    save_path: str | Path | None = None,
    zoom_save_path: str | Path | None = None,
    coverage_save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> tuple[plt.Figure, np.ndarray, dict[str, Any]]:
    """Plot runtime growth, with one full and one <=70-node figure per family.

    Full-range plots contain the explicitly selected models or the most robust
    models chosen by ``top_n_models``. The zoom plots contain every available
    model up to ``zoom_max_nodes``. Pass one or more counts through
    ``exclude_n_nodes`` to omit them from the full plots, zoom plots and
    coverage tables. By default no counts are excluded. ``include_20_nodes``
    remains available for backwards compatibility.
    """
    if not 0 <= survival_completion_fraction <= 1:
        raise ValueError("survival_completion_fraction must be in [0, 1].")
    family_keys = list(families or DEFAULT_SCALABILITY_FAMILIES)
    labels = {
        key: _dataset_display_title(value)
        for key, value in {
            **DEFAULT_SCALABILITY_FAMILIES,
            **dict(family_labels or {}),
        }.items()
    }
    frame = (
        load_scalability_data(
            data,
            scalability_paths=scalability_paths,
            project_root=project_root,
            families=family_keys,
            n_nodes=n_nodes,
        )
        if scalability_data is None
        else _prepare_scalability_frame(scalability_data)
    )
    frame = frame[frame["scalability_family"].isin(family_keys)].copy()
    if n_nodes is not None:
        frame = frame[frame["n_nodes"].isin(_as_list(n_nodes) or [])]
    excluded_nodes = {int(value) for value in _as_list(exclude_n_nodes) or []}
    if not include_20_nodes:
        excluded_nodes.add(20)
    if excluded_nodes:
        frame = frame[~frame["n_nodes"].isin(excluded_nodes)]
    all_blocks, all_summary, coverage = _scalability_summaries(
        frame,
        "runtime_seconds",
        uncertainty=uncertainty,
        confidence_level=confidence_level,
        ci_method=ci_method,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    survival_rows: list[dict[str, Any]] = []
    for model, model_coverage in coverage.groupby("model", observed=True):
        qualifying_nodes = []
        for nodes, node_rows in model_coverage.groupby("n_nodes", observed=True):
            complete_families = set(
                node_rows.loc[
                    node_rows["completion_fraction"].ge(survival_completion_fraction),
                    "scalability_family",
                ]
            )
            if set(family_keys).issubset(complete_families):
                qualifying_nodes.append(nodes)
        survival_rows.append(
            {
                "model": model,
                "largest_robust_n_nodes": max(qualifying_nodes) if qualifying_nodes else np.nan,
                "mean_completion_fraction": model_coverage["completion_fraction"].mean(),
            }
        )
    model_selection = pd.DataFrame(survival_rows).sort_values(
        ["largest_robust_n_nodes", "mean_completion_fraction"],
        ascending=[False, False],
        na_position="last",
    )
    if models is None:
        selected_models = model_selection.head(top_n_models)["model"].tolist()
    else:
        selected_models = _ordered_models(_as_list(models) or [], model_order=model_order)
    if not selected_models:
        raise ValueError("No models were selected for the runtime plot.")
    summary = all_summary[all_summary["model"].isin(selected_models)].copy()
    blocks = all_blocks[all_blocks["model"].isin(selected_models)].copy()
    all_model_names = _ordered_models(all_summary["model"].unique(), model_order=model_order)
    model_names = _ordered_models(selected_models, model_order=model_order)
    color_lookup = dict(zip(all_model_names, _colors(all_model_names, model_colors)))
    if separate_families:
        figures, axes_by_family = _independent_family_axes(family_keys, axes=axes, figsize=figsize)
        for family, axis in axes_by_family.items():
            _draw_scalability_curve(
                axis,
                summary[summary["scalability_family"].eq(family)],
                model_names,
                color_lookup,
                uncertainty=uncertainty,
                family_title=labels.get(family, family),
                show_title=show_title,
                show_family_titles=show_family_titles,
                x_scale=x_scale,
                xlabel=xlabel,
                ylabel=ylabel,
                font_size=font_size,
                title_size=title_size,
                y_scale="log",
                errorbar_alpha=errorbar_alpha,
            )
            if show_title and title is not None:
                figures[family].suptitle(title, fontsize=title_size + 1)
            elif show_title and show_metric_title:
                figures[family].suptitle(
                    "Runtime scalability of the most robust models",
                    fontsize=title_size + 1,
                )
            _despine(axis, font_size=font_size)
            figures[family].tight_layout()
        axes_array = np.asarray(list(axes_by_family.values()), dtype=object)
        fig = figures[family_keys[0]]
        _save_scalability_figures(figures, save_path, dpi=dpi, transparent=transparent)
    else:
        combined_coverage = show_coverage and (not separate_coverage)
        fig, axes_array, combined_coverage_axes = _scalability_axes(
            family_keys, axes=axes, figsize=figsize, show_coverage=combined_coverage
        )
        figures = {family: fig for family in family_keys}
        axes_by_family = dict(zip(family_keys, axes_array))
        for index, (family, axis) in enumerate(axes_by_family.items()):
            _draw_scalability_curve(
                axis,
                summary[summary["scalability_family"].eq(family)],
                model_names,
                color_lookup,
                uncertainty=uncertainty,
                family_title=labels.get(family, family),
                show_title=show_title,
                show_family_titles=show_family_titles,
                x_scale=x_scale,
                xlabel=xlabel,
                ylabel=ylabel if index == 0 else "",
                font_size=font_size,
                title_size=title_size,
                y_scale="log",
                errorbar_alpha=errorbar_alpha,
            )
            if index != len(family_keys) - 1 and axis.get_legend() is not None:
                axis.get_legend().remove()
        if show_title and (title is not None or show_metric_title):
            fig.suptitle(
                title or "Runtime scalability of the most robust models",
                fontsize=title_size + 1,
            )
        _despine(axes_array, font_size=font_size)
        fig.tight_layout()
        _save(fig, save_path, dpi, transparent)
    zoom_figures: dict[str, plt.Figure] = {}
    zoom_axes_by_family: dict[str, plt.Axes] = {}
    zoom_summary = all_summary.iloc[0:0].copy()
    if zoom_max_nodes is not None:
        zoom_summary = all_summary[all_summary["n_nodes"].le(zoom_max_nodes)].copy()
        zoom_figures, zoom_axes_by_family = _independent_family_axes(
            family_keys, axes=zoom_axes, figsize=zoom_figsize or figsize
        )
        for family, axis in zoom_axes_by_family.items():
            _draw_scalability_curve(
                axis,
                zoom_summary[zoom_summary["scalability_family"].eq(family)],
                all_model_names,
                color_lookup,
                uncertainty=uncertainty,
                family_title=labels.get(family, family),
                show_title=show_title,
                show_family_titles=show_family_titles,
                x_scale=zoom_x_scale,
                xlabel=xlabel,
                ylabel=ylabel,
                font_size=font_size,
                title_size=title_size,
                y_scale="log",
                errorbar_alpha=errorbar_alpha,
            )
            _despine(axis, font_size=font_size)
            zoom_figures[family].tight_layout()
        _save_scalability_figures(zoom_figures, zoom_save_path, dpi=dpi, transparent=transparent)
    coverage_figures: dict[str, plt.Figure] = {}
    coverage_axes_by_family: dict[str, plt.Axes] = {}
    coverage_axes_array = np.asarray([], dtype=object)
    if show_coverage:
        selected_coverage = coverage[coverage["model"].isin(model_names)]
        if not separate_families and (not separate_coverage):
            coverage_axes_array = combined_coverage_axes
            coverage_figures = {family: fig for family in family_keys}
            coverage_axes_by_family = dict(zip(family_keys, coverage_axes_array))
        else:
            coverage_figures, coverage_axes_by_family = _independent_family_axes(
                family_keys,
                axes=coverage_axes,
                figsize=coverage_figsize or (7.2, max(5.5, 0.38 * len(model_names) + 1.8)),
            )
            coverage_axes_array = np.asarray(list(coverage_axes_by_family.values()), dtype=object)
        for family, axis in coverage_axes_by_family.items():
            _draw_scalability_coverage(
                [axis],
                [family],
                selected_coverage,
                model_names,
                labels,
                font_size=font_size,
                show_title=show_title,
                show_family_titles=show_family_titles,
            )
            _despine(axis, font_size=font_size)
            coverage_figures[family].tight_layout()
        _save_scalability_figures(
            coverage_figures, coverage_save_path, dpi=dpi, transparent=transparent
        )
    return (
        fig,
        axes_array,
        {
            "summary": summary,
            "block_means": blocks,
            "coverage": coverage,
            "model_selection": model_selection,
            "raw_results": frame,
            "figures": figures,
            "axes_by_family": axes_by_family,
            "zoom_figures": zoom_figures,
            "zoom_axes": zoom_axes_by_family,
            "zoom_summary": zoom_summary,
            "coverage_figures": coverage_figures,
            "coverage_axes": coverage_axes_array,
            "coverage_axes_by_family": coverage_axes_by_family,
            "excluded_n_nodes": sorted(excluded_nodes),
        },
    )


def _plot_seed_stability(
    data: FinalVisualizationData,
    *,
    varying_seed: str,
    fixed_seed: str,
    metric: str,
    variability: str,
    min_coverage_fraction: float,
    datasets: Any,
    n_nodes: Any,
    graph_type: Any,
    noise: Any,
    mechanism: Any,
    obs_conf: Any,
    lat_conf: Any,
    models: Any,
    filters: Mapping[str, Any] | None,
    model_order: Sequence[str] | None,
    model_colors: Mapping[str, str] | None,
    plot_kind: str,
    show_points: bool,
    title: str | None,
    show_title: bool,
    show_metric_title: bool,
    model_axis: str,
    ylabel: str | None,
    figsize: tuple[float, float] | None,
    ax: plt.Axes | None,
    font_size: float,
    title_size: float,
    save_path: str | Path | None,
    dpi: int,
    transparent: bool,
) -> tuple[plt.Figure, plt.Axes, dict[str, pd.DataFrame]]:
    model_axis = str(model_axis).casefold()
    if model_axis not in {"x", "y"}:
        raise ValueError("model_axis must be 'x' or 'y'.")
    if not 0 <= min_coverage_fraction <= 1:
        raise ValueError("min_coverage_fraction must be in [0, 1].")
    measure_columns = {"range": "range_width", "std": "std", "sd": "std", "iqr": "iqr"}
    key = variability.casefold()
    if key not in measure_columns:
        raise ValueError("variability must be 'range', 'std', or 'iqr'.")
    value_column = measure_columns[key]
    selected = _filter_frame(
        data.metrics,
        datasets=datasets,
        n_nodes=n_nodes,
        graph_type=graph_type,
        noise=noise,
        mechanism=mechanism,
        obs_conf=obs_conf,
        lat_conf=lat_conf,
        models=models,
        filters=filters,
    ).copy()
    _require_columns(selected, [varying_seed, fixed_seed, metric], "metrics")
    selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
    selected["_valid"] = selected["finished"] & selected[metric].notna()
    expected = (
        selected.groupby("dataset", observed=True)[varying_seed]
        .nunique()
        .rename("n_expected")
        .to_dict()
    )
    rows: list[dict[str, Any]] = []
    group_columns = ["dataset", "model", fixed_seed]
    for identity, group in selected.groupby(group_columns, observed=True, dropna=False):
        values = group.loc[group["_valid"], metric].to_numpy(dtype=float)
        dataset, model, fixed_value = identity
        n_expected = int(expected[dataset])
        n_completed = len(values)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                fixed_seed: fixed_value,
                "varying_seed": varying_seed,
                "fixed_seed": fixed_seed,
                "n_completed": n_completed,
                "n_expected": n_expected,
                "coverage_fraction": n_completed / n_expected if n_expected else np.nan,
                "mean": float(np.mean(values)) if n_completed else np.nan,
                "std": float(np.std(values, ddof=1)) if n_completed > 1 else np.nan,
                "minimum": float(np.min(values)) if n_completed else np.nan,
                "maximum": float(np.max(values)) if n_completed else np.nan,
                "range_width": float(np.ptp(values)) if n_completed else np.nan,
                "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
                if n_completed
                else np.nan,
            }
        )
    all_groups = pd.DataFrame(rows)
    all_groups["included"] = (
        all_groups["coverage_fraction"].ge(min_coverage_fraction) & all_groups[value_column].notna()
    )
    variability_values = all_groups[all_groups["included"]].copy()
    if variability_values.empty:
        raise ValueError("No seed groups satisfy the selected coverage threshold.")
    model_names = _ordered_models(variability_values["model"].unique(), model_order=model_order)
    if ax is None:
        default_figsize = (
            (max(9, 0.75 * len(model_names)), 5.5)
            if model_axis == "x"
            else (7.5, max(5.5, 0.48 * len(model_names)))
        )
        fig, ax = plt.subplots(figsize=figsize or default_figsize)
    else:
        fig = ax.figure
    values = [
        variability_values.loc[variability_values["model"].eq(model), value_column].to_numpy()
        for model in model_names
    ]
    colors = _colors(model_names, model_colors)
    positions = np.arange(1, len(model_names) + 1)
    kind = plot_kind.casefold()
    if kind == "box":
        artists = ax.boxplot(
            values,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
            manage_ticks=False,
            orientation="vertical" if model_axis == "x" else "horizontal",
            medianprops={"color": "#111827", "linewidth": 1.2},
            meanprops={
                "marker": "D",
                "markerfacecolor": "#111827",
                "markeredgecolor": "#111827",
                "markersize": 3,
            },
        )
        for patch, color in zip(artists["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
    elif kind == "violin":
        too_small = [
            model for model, model_values in zip(model_names, values) if len(model_values) < 2
        ]
        if too_small:
            raise ValueError(
                f"A violin plot needs at least two independent seed groups per model. Insufficient data for: {too_small}"
            )
        violin = ax.violinplot(
            values,
            positions=positions,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            orientation="vertical" if model_axis == "x" else "horizontal",
        )
        for body, color in zip(violin["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.55)
        inner_box = ax.boxplot(
            values,
            positions=positions,
            widths=0.16,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
            manage_ticks=False,
            orientation="vertical" if model_axis == "x" else "horizontal",
            boxprops={"facecolor": "white", "edgecolor": "#111827", "linewidth": 1.0},
            whiskerprops={"color": "#111827", "linewidth": 0.9},
            capprops={"color": "#111827", "linewidth": 0.9},
            medianprops={"color": "#111827", "linewidth": 1.4},
            meanprops={
                "marker": "D",
                "markerfacecolor": "#111827",
                "markeredgecolor": "#111827",
                "markersize": 3,
            },
        )
        for patch in inner_box["boxes"]:
            patch.set_alpha(0.9)
    else:
        raise ValueError("plot_kind must be 'box' or 'violin'.")
    if model_axis == "x":
        ax.set_xticks(positions, [_model_display_label(model) for model in model_names])
    else:
        ax.set_yticks(positions, [_model_display_label(model) for model in model_names])
        ax.invert_yaxis()
    if show_points:
        rng = np.random.default_rng(42)
        for position, (model_values, color) in enumerate(zip(values, colors), start=1):
            jitter = rng.uniform(-0.12, 0.12, len(model_values))
            ax.scatter(
                position + jitter if model_axis == "x" else model_values,
                model_values if model_axis == "x" else position + jitter,
                s=15,
                color=color,
                edgecolor="white",
                linewidth=0.3,
                alpha=0.7,
            )
    metric_label, _ = _metric_meta(metric)
    measure_label = {
        "range_width": "range (max − min)",
        "std": "standard deviation",
        "iqr": "IQR",
    }[value_column]
    varied_label = "dataset seed" if varying_seed == "dataset_seed" else "model seed"
    metric_axis_label = ylabel or f"{metric_label} {measure_label}"
    if model_axis == "x":
        ax.set_ylabel(metric_axis_label, fontsize=font_size)
    else:
        ax.set_xlabel(metric_axis_label, fontsize=font_size)
    if show_title and (title is not None or show_metric_title):
        ax.set_title(
            title or f"{metric_label} stability across {varied_label}s",
            fontsize=title_size,
        )
    ax.tick_params(axis=model_axis, rotation=35 if model_axis == "x" else 0, labelsize=font_size)
    ax.grid(axis="y" if model_axis == "x" else "x", alpha=0.25)
    _despine(ax, font_size=font_size)
    fig.tight_layout()
    _save(fig, save_path, dpi, transparent)
    return (
        fig,
        ax,
        {
            "variability": variability_values.reset_index(drop=True),
            "coverage": all_groups.reset_index(drop=True),
            "selected_results": selected.drop(columns=["_valid"]),
        },
    )


def plot_dataset_seed_stability(
    data: FinalVisualizationData,
    *,
    metric: str = "f1",
    variability: str = "range",
    min_coverage_fraction: float = 0.8,
    datasets: Any = None,
    n_nodes: Any = None,
    graph_type: Any = None,
    noise: Any = None,
    mechanism: Any = None,
    obs_conf: Any = None,
    lat_conf: Any = None,
    models: Any = None,
    filters: Mapping[str, Any] | None = None,
    model_order: Sequence[str] | None = None,
    model_colors: Mapping[str, str] | None = None,
    plot_kind: str = "violin",
    show_points: bool = True,
    title: str | None = None,
    show_title: bool = True,
    show_metric_title: bool = True,
    model_axis: str = "x",
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> tuple[plt.Figure, plt.Axes, dict[str, pd.DataFrame]]:
    """Plot the distribution of dataset-seed variability by model.

    ``plot_kind`` may be ``"violin"`` (with an inner boxplot) or ``"box"``.
    Each plotted observation is the selected ``variability`` measure computed
    across dataset seeds while holding one model seed fixed.
    ``model_axis`` may be ``"x"`` (default) or ``"y"``.
    """
    return _plot_seed_stability(
        data,
        varying_seed="dataset_seed",
        fixed_seed="model_seed",
        metric=metric,
        variability=variability,
        min_coverage_fraction=min_coverage_fraction,
        datasets=datasets,
        n_nodes=n_nodes,
        graph_type=graph_type,
        noise=noise,
        mechanism=mechanism,
        obs_conf=obs_conf,
        lat_conf=lat_conf,
        models=models,
        filters=filters,
        model_order=model_order,
        model_colors=model_colors,
        plot_kind=plot_kind,
        show_points=show_points,
        title=title,
        show_title=show_title,
        show_metric_title=show_metric_title,
        model_axis=model_axis,
        ylabel=ylabel,
        figsize=figsize,
        ax=ax,
        font_size=font_size,
        title_size=title_size,
        save_path=save_path,
        dpi=dpi,
        transparent=transparent,
    )


def plot_model_seed_stability(
    data: FinalVisualizationData,
    *,
    metric: str = "f1",
    variability: str = "range",
    min_coverage_fraction: float = 0.8,
    datasets: Any = None,
    n_nodes: Any = None,
    graph_type: Any = None,
    noise: Any = None,
    mechanism: Any = None,
    obs_conf: Any = None,
    lat_conf: Any = None,
    models: Any = None,
    filters: Mapping[str, Any] | None = None,
    model_order: Sequence[str] | None = None,
    model_colors: Mapping[str, str] | None = None,
    plot_kind: str = "violin",
    show_points: bool = True,
    title: str | None = None,
    show_title: bool = True,
    show_metric_title: bool = True,
    model_axis: str = "x",
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> tuple[plt.Figure, plt.Axes, dict[str, pd.DataFrame]]:
    """Plot the distribution of model-seed variability by model.

    ``plot_kind`` may be ``"violin"`` (with an inner boxplot) or ``"box"``.
    Each plotted observation is the selected ``variability`` measure computed
    across model seeds while holding one dataset seed fixed.
    ``model_axis`` may be ``"x"`` (default) or ``"y"``.
    """
    return _plot_seed_stability(
        data,
        varying_seed="model_seed",
        fixed_seed="dataset_seed",
        metric=metric,
        variability=variability,
        min_coverage_fraction=min_coverage_fraction,
        datasets=datasets,
        n_nodes=n_nodes,
        graph_type=graph_type,
        noise=noise,
        mechanism=mechanism,
        obs_conf=obs_conf,
        lat_conf=lat_conf,
        models=models,
        filters=filters,
        model_order=model_order,
        model_colors=model_colors,
        plot_kind=plot_kind,
        show_points=show_points,
        title=title,
        show_title=show_title,
        show_metric_title=show_metric_title,
        model_axis=model_axis,
        ylabel=ylabel,
        figsize=figsize,
        ax=ax,
        font_size=font_size,
        title_size=title_size,
        save_path=save_path,
        dpi=dpi,
        transparent=transparent,
    )


def plot_confounding_f1(
    data: FinalVisualizationData,
    *,
    metric: str = "f1",
    noise: Any = "Gaussian",
    mechanism: Any = "Linear",
    graph_type: Any = "EB",
    datasets: Any = None,
    n_nodes: Any = None,
    obs_conf: Any = None,
    lat_conf: Any = None,
    models: Any = None,
    filters: Mapping[str, Any] | None = None,
    coverage_policy: str = "complete",
    min_completed_runs: int | None = None,
    min_coverage_fraction: float | None = None,
    uncertainty: str | None = "sd",
    confidence_level: float = 0.95,
    ci_method: str = "bootstrap",
    n_bootstrap: int = 2000,
    random_state: int | None = 42,
    model_order: Sequence[str] | None = None,
    model_colors: Mapping[str, str] | None = None,
    bar_alpha: float = 0.55,
    title: str | None = None,
    show_title: bool = True,
    show_metric_title: bool = True,
    xlabel: str = "Observed and latent confounders",
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> tuple[plt.Figure, plt.Axes, dict[str, pd.DataFrame]]:
    """Plot model F1 for each selected observed/latent-confounder setting.

    Gaussian noise and a linear mechanism are the defaults, but every dataset
    selector is configurable. Model seeds are averaged within each independent
    dataset block before uncertainty is computed across dataset seeds.
    ``bar_alpha=0.55`` matches the model-colour transparency of the six-panel
    F1 boxplot.
    """
    if not 0 <= bar_alpha <= 1:
        raise ValueError("bar_alpha must be between 0 and 1.")
    selected = _filter_frame(
        data.metrics,
        datasets=datasets,
        n_nodes=n_nodes,
        graph_type=graph_type,
        noise=noise,
        mechanism=mechanism,
        obs_conf=obs_conf,
        lat_conf=lat_conf,
        models=models,
        filters=filters,
    )
    selected = selected.copy()
    selected["confounding_level"] = selected.apply(
        lambda row: f"Obs {int(row['obs_conf'])} · Lat {int(row['lat_conf'])}", axis=1
    )
    level_order = (
        selected[["obs_conf", "lat_conf", "confounding_level"]]
        .drop_duplicates()
        .sort_values(["obs_conf", "lat_conf"], kind="stable")["confounding_level"]
        .tolist()
    )
    summaries: list[dict[str, Any]] = []
    block_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []
    for level in level_order:
        group = selected[selected["confounding_level"].eq(level)]
        by_dataset, coverage = _coverage_from_metrics(group, metric)
        eligible = _eligible_models(
            by_dataset,
            coverage_policy=coverage_policy,
            min_completed_runs=min_completed_runs,
            min_coverage_fraction=min_coverage_fraction,
        )
        coverage = coverage.copy()
        coverage.insert(0, "confounding_level", level)
        coverage["included"] = coverage["model"].isin(eligible)
        coverage_frames.append(coverage)
        blocks = _block_means(group, metric)
        blocks = blocks[blocks["model"].isin(eligible)].copy()
        blocks.insert(0, "confounding_level", level)
        block_frames.append(blocks)
        for model, model_blocks in blocks.groupby("model", observed=True):
            stats_row = _uncertainty_summary(
                model_blocks["metric_value"],
                uncertainty=uncertainty,
                confidence_level=confidence_level,
                ci_method=ci_method,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )
            summaries.append({"confounding_level": level, "model": model, **stats_row})
    summary = pd.DataFrame(summaries)
    if summary.empty:
        raise ValueError("No models satisfy the selected filters and coverage policy.")
    block_means = pd.concat(block_frames, ignore_index=True)
    coverage = pd.concat(coverage_frames, ignore_index=True)
    model_names = _ordered_models(summary["model"].unique(), model_order=model_order)
    if ax is None:
        if figsize is None:
            figsize = (max(9, 2.2 * len(level_order)), 6)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    x = np.arange(len(level_order), dtype=float)
    width = min(0.8 / max(len(model_names), 1), 0.16)
    offsets = (np.arange(len(model_names)) - (len(model_names) - 1) / 2) * width
    for offset, model, color in zip(offsets, model_names, _colors(model_names, model_colors)):
        rows = (
            summary[summary["model"].eq(model)].set_index("confounding_level").reindex(level_order)
        )
        means = rows["mean"].to_numpy(dtype=float)
        lower = means - rows["lower"].to_numpy(dtype=float)
        upper = rows["upper"].to_numpy(dtype=float) - means
        yerr = np.vstack([lower, upper]) if uncertainty is not None else None
        ax.bar(
            x + offset,
            means,
            width=width,
            color=color,
            alpha=bar_alpha,
            label=_model_display_label(model),
            yerr=yerr,
            capsize=2,
            error_kw={"linewidth": 0.8},
        )
    label, _ = _metric_meta(metric)
    ax.set_xticks(x, level_order)
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel or label, fontsize=font_size)
    noise_label = ", ".join(map(str, _as_list(noise) or ["all noises"]))
    mechanism_label = ", ".join(map(str, _as_list(mechanism) or ["all mechanisms"]))
    if show_title and (title is not None or show_metric_title):
        ax.set_title(
            title or f"{label} by confounding setting ({noise_label}, {mechanism_label})",
            fontsize=title_size,
        )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=max(6, font_size - 1))
    _despine(ax, font_size=font_size)
    fig.tight_layout()
    _save(fig, save_path, dpi, transparent)
    return (
        fig,
        ax,
        {"summary": summary, "block_means": block_means, "coverage": coverage},
    )


def build_confounding_pair_table(
    true_adjacency: pd.DataFrame | np.ndarray,
    predicted_adjacency: pd.DataFrame | np.ndarray,
    latent_children: Mapping[Any, Sequence[Any]],
    *,
    node_names: Sequence[Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build one-row-per-unordered-pair input for the FP-rate plot.

    A pair is marked as sharing a latent confounder when both observed nodes
    occur in the child list of at least one latent variable. Any edge in either
    direction counts as an adjacency for the false-positive analysis.
    """
    if isinstance(true_adjacency, pd.DataFrame):
        names = list(true_adjacency.columns) if node_names is None else list(node_names)
        true = true_adjacency.loc[names, names].to_numpy()
    else:
        true = np.asarray(true_adjacency)
        names = list(range(true.shape[0])) if node_names is None else list(node_names)
    if isinstance(predicted_adjacency, pd.DataFrame):
        predicted = predicted_adjacency.loc[names, names].to_numpy()
    else:
        predicted = np.asarray(predicted_adjacency)
    if true.ndim != 2 or true.shape[0] != true.shape[1] or predicted.shape != true.shape:
        raise ValueError(
            "true_adjacency and predicted_adjacency must be equally sized square matrices."
        )
    if len(names) != true.shape[0]:
        raise ValueError("node_names must contain one label per adjacency node.")
    child_sets = [set(map(str, children)) for children in latent_children.values()]
    rows: list[dict[str, Any]] = []
    identity = dict(metadata or {})
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            left_name, right_name = (names[left], names[right])
            true_edge = bool(true[left, right] != 0 or true[right, left] != 0)
            predicted_edge = bool(predicted[left, right] != 0 or predicted[right, left] != 0)
            shared = any(
                (
                    str(left_name) in children and str(right_name) in children
                    for children in child_sets
                )
            )
            rows.append(
                {
                    **identity,
                    "node_1": left_name,
                    "node_2": right_name,
                    "true_edge": true_edge,
                    "predicted_edge": predicted_edge,
                    "false_positive": predicted_edge and (not true_edge),
                    "shares_latent_confounder": shared,
                }
            )
    return pd.DataFrame(rows)


def plot_confounding_false_positive_rates(
    pair_results: pd.DataFrame,
    *,
    noise: Any = "Gaussian",
    mechanism: Any = "Linear",
    graph_type: Any = "EB",
    datasets: Any = None,
    n_nodes: Any = None,
    obs_conf: Any = None,
    lat_conf: Any = None,
    models: Any = None,
    filters: Mapping[str, Any] | None = None,
    include_no_latent: bool = False,
    model_order: Sequence[str] | None = None,
    uncertainty: str | None = "sd",
    confidence_level: float = 0.95,
    ci_method: str = "bootstrap",
    n_bootstrap: int = 2000,
    random_state: int | None = 42,
    title: str | None = None,
    show_title: bool = True,
    show_metric_title: bool = True,
    xlabel: str = "",
    ylabel: str = "False-positive rate",
    class_labels: Mapping[bool, str] | None = None,
    class_colors: Mapping[bool, str] | None = None,
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> tuple[plt.Figure, plt.Axes, dict[str, pd.DataFrame]]:
    """Compare FP rates for pairs with and without a shared latent parent.

    ``pair_results`` must contain one row per *unordered* observed-node pair and
    run. Rates use the number of absent true adjacencies in each class as the
    denominator. Model seeds are averaged before dataset seeds/configurations.
    """
    _require_columns(
        pair_results,
        ["dataset_seed", "dataset", "model", "shares_latent_confounder"],
        "pair_results",
    )
    selected = _filter_frame(
        pair_results,
        datasets=datasets,
        n_nodes=n_nodes,
        graph_type=graph_type,
        noise=noise,
        mechanism=mechanism,
        obs_conf=obs_conf,
        lat_conf=lat_conf,
        models=models,
        filters=filters,
    ).copy()
    if not include_no_latent:
        selected = selected[pd.to_numeric(selected["lat_conf"], errors="coerce").gt(0)]
    if selected.empty:
        raise ValueError("No latent-confounding pair rows remain after filtering.")
    if "false_positive" not in selected.columns:
        _require_columns(selected, ["true_edge", "predicted_edge"], "pair_results")
        selected["false_positive"] = _as_bool(selected["predicted_edge"]) & ~_as_bool(
            selected["true_edge"]
        )
    if "true_edge" not in selected.columns:
        raise ValueError("pair_results needs true_edge to define possible false-positive pairs.")
    selected["shares_latent_confounder"] = _as_bool(selected["shares_latent_confounder"])
    selected["false_positive"] = _as_bool(selected["false_positive"])
    selected["true_edge"] = _as_bool(selected["true_edge"])
    run_keys = ["dataset_seed", "dataset", "model"]
    if "model_seed" in selected.columns:
        run_keys.insert(2, "model_seed")
    rows: list[dict[str, Any]] = []
    for keys, group in selected.groupby(run_keys, observed=True, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(run_keys, key_values))
        for shared in (True, False):
            eligible = group[group["shares_latent_confounder"].eq(shared) & ~group["true_edge"]]
            denominator = len(eligible)
            numerator = int(eligible["false_positive"].sum())
            rows.append(
                {
                    **identity,
                    "shares_latent_confounder": shared,
                    "false_positives": numerator,
                    "possible_pairs": denominator,
                    "false_positive_rate": numerator / denominator if denominator else np.nan,
                }
            )
    run_rates = pd.DataFrame(rows)
    seed_keys = ["dataset_seed", "dataset", "model", "shares_latent_confounder"]
    seed_rates = (
        run_rates.groupby(seed_keys, observed=True, dropna=False)
        .agg(
            false_positive_rate=("false_positive_rate", "mean"),
            false_positives=("false_positives", "mean"),
            possible_pairs=("possible_pairs", "mean"),
            completed_model_seeds=("false_positive_rate", "count"),
        )
        .reset_index()
    )
    summary_rows: list[dict[str, Any]] = []
    for (model, shared), group in seed_rates.groupby(
        ["model", "shares_latent_confounder"], observed=True
    ):
        stats_row = _uncertainty_summary(
            group["false_positive_rate"],
            uncertainty=uncertainty,
            confidence_level=confidence_level,
            ci_method=ci_method,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        summary_rows.append(
            {
                "model": model,
                "shares_latent_confounder": shared,
                **stats_row,
                "mean_false_positives": group["false_positives"].mean(),
                "mean_possible_pairs": group["possible_pairs"].mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    model_names = _ordered_models(summary["model"].unique(), model_order=model_order)
    labels = {
        True: "Pairs sharing a latent confounder",
        False: "Other pairs",
        **dict(class_labels or {}),
    }
    colors = {
        True: OKABE_ITO["reddish_purple"],
        False: OKABE_ITO["blue"],
        **dict(class_colors or {}),
    }
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (max(8, 0.75 * len(model_names)), 5.5))
    else:
        fig = ax.figure
    x = np.arange(len(model_names), dtype=float)
    width = 0.38
    for offset, shared in ((-width / 2, True), (width / 2, False)):
        rows = (
            summary[summary["shares_latent_confounder"].eq(shared)]
            .set_index("model")
            .reindex(model_names)
        )
        means = rows["mean"].to_numpy(dtype=float)
        yerr = None
        if uncertainty is not None:
            yerr = np.vstack(
                [
                    means - rows["lower"].to_numpy(dtype=float),
                    rows["upper"].to_numpy(dtype=float) - means,
                ]
            )
        ax.bar(
            x + offset,
            means,
            width=width,
            label=labels[shared],
            color=colors[shared],
            yerr=yerr,
            capsize=2,
        )
    ax.set_xticks(
        x,
        [_model_display_label(model) for model in model_names],
        rotation=35,
        ha="right",
    )
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    if show_title and (title is not None or show_metric_title):
        ax.set_title(
            title or "False positives by latent-confounding pair class",
            fontsize=title_size,
        )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=max(6, font_size - 1))
    _despine(ax, font_size=font_size)
    fig.tight_layout()
    _save(fig, save_path, dpi, transparent)
    return (
        fig,
        ax,
        {"summary": summary, "seed_rates": seed_rates, "run_rates": run_rates},
    )


def plot_varsortability_vs_f1(
    data: FinalVisualizationData,
    *,
    metric: str = "f1",
    varsortability_values: pd.DataFrame | None = None,
    include_r2_sort_regress: bool = False,
    r2_sort_regress_values: pd.DataFrame | None = None,
    r2_sort_regress_label: str = "R²-SortnRegress",
    sortability: str = "r2_sortability",
    project_root: str | Path | None = None,
    dataset_roots: Iterable[str | Path] | None = None,
    datasets: Any = None,
    n_nodes: Any = None,
    graph_type: Any = None,
    noise: Any = None,
    mechanism: Any = None,
    obs_conf: Any = 0,
    lat_conf: Any = 0,
    models: Any = None,
    filters: Mapping[str, Any] | None = None,
    coverage_policy: str = "complete",
    min_completed_runs: int | None = None,
    min_coverage_fraction: float | None = None,
    model_order: Sequence[str] | None = None,
    model_colors: Mapping[str, str] | None = None,
    show_regression: bool = True,
    columns: int = 4,
    title: str | None = None,
    show_title: bool = True,
    show_metric_title: bool = True,
    show_model_titles: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = None,
    axes: Sequence[plt.Axes] | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> tuple[plt.Figure, np.ndarray, dict[str, pd.DataFrame]]:
    """Plot R²-sortability or raw-data varsortability against model F1.

    If ``varsortability_values`` is omitted, this function imports and calls
    :func:`calculate_varsortability_table` from the separate varsortability
    module. The default x-axis is scale-invariant R²-sortability; the returned
    dataset table also contains varsortability calculated before z-scoring.
    Set ``include_r2_sort_regress=True`` to fit CausalDisco's
    ``r2_sort_regress`` to each raw observational dataset and add its directed
    F1 as another model panel. Precomputed results can instead be supplied via
    ``r2_sort_regress_values``. Values include the plotted points and per-model
    correlations and fit R².
    ``show_metric_title`` controls only the overall ``versus`` title;
    ``show_model_titles`` independently controls the model and fit R² title
    above every panel.
    """
    selected = _filter_frame(
        data.metrics,
        datasets=datasets,
        n_nodes=n_nodes,
        graph_type=graph_type,
        noise=noise,
        mechanism=mechanism,
        obs_conf=obs_conf,
        lat_conf=lat_conf,
        models=models,
        filters=filters,
    )
    by_dataset, _ = _coverage_from_metrics(selected, metric)
    eligible = _eligible_models(
        by_dataset,
        coverage_policy=coverage_policy,
        min_completed_runs=min_completed_runs,
        min_coverage_fraction=min_coverage_fraction,
    )
    blocks = _block_means(selected, metric)
    blocks = blocks[blocks["model"].isin(eligible)].copy()
    if varsortability_values is None:
        try:
            from .varsortability import calculate_varsortability_table
        except ImportError:
            from varsortability import calculate_varsortability_table
        varsortability_values = calculate_varsortability_table(
            blocks[["dataset_seed", "dataset"]],
            project_root=project_root,
            dataset_roots=dataset_roots,
        )
    else:
        varsortability_values = varsortability_values.copy()
    baseline_values = pd.DataFrame()
    if include_r2_sort_regress or r2_sort_regress_values is not None:
        if metric not in {"f1", "f1"}:
            raise ValueError(
                "R²-SortnRegress currently supports metric='f1' or metric='f1'."
            )
        if r2_sort_regress_values is None:
            try:
                from .varsortability import calculate_r2_sort_regress_table
            except ImportError:
                from varsortability import calculate_r2_sort_regress_table
            baseline_values = calculate_r2_sort_regress_table(
                blocks[["dataset_seed", "dataset"]],
                project_root=project_root,
                dataset_roots=dataset_roots,
            )
        else:
            baseline_values = r2_sort_regress_values.copy()
        _require_columns(baseline_values, ["dataset_seed", "dataset"], "r2_sort_regress_values")
        if baseline_values.duplicated(["dataset_seed", "dataset"]).any():
            raise ValueError("r2_sort_regress_values has duplicate dataset_seed/dataset rows.")
        baseline_metric = (
            metric
            if metric in baseline_values.columns
            else "f1"
            if "f1" in baseline_values.columns
            else "metric_value"
            if "metric_value" in baseline_values.columns
            else None
        )
        if baseline_metric is None:
            raise ValueError(
                "r2_sort_regress_values must contain the selected metric, 'f1', or 'metric_value'."
            )
        identities = blocks[["dataset_seed", "dataset"]].drop_duplicates()
        baseline_blocks = identities.merge(
            baseline_values,
            on=["dataset_seed", "dataset"],
            how="left",
            validate="one_to_one",
        )
        baseline_blocks["metric_value"] = pd.to_numeric(
            baseline_blocks[baseline_metric], errors="coerce"
        )
        baseline_blocks = baseline_blocks.dropna(subset=["metric_value"])
        baseline_blocks["model"] = str(r2_sort_regress_label)
        baseline_blocks["completed_model_seeds"] = 1
        baseline_blocks = _add_dataset_attributes(baseline_blocks)
        blocks = pd.concat([blocks, baseline_blocks], ignore_index=True)
    if sortability not in {"r2_sortability", "varsortability"}:
        raise ValueError("sortability must be 'r2_sortability' or 'varsortability'.")
    _require_columns(
        varsortability_values,
        ["dataset_seed", "dataset", sortability],
        "varsortability_values",
    )
    duplicate_values = varsortability_values.duplicated(["dataset_seed", "dataset"])
    if duplicate_values.any():
        raise ValueError("varsortability_values has duplicate dataset_seed/dataset rows.")
    points = blocks.merge(
        varsortability_values,
        on=["dataset_seed", "dataset"],
        how="left",
        validate="many_to_one",
    )
    points[sortability] = pd.to_numeric(points[sortability], errors="coerce")
    points = points.dropna(subset=[sortability, "metric_value"])
    if points.empty:
        raise ValueError("No F1 blocks could be matched to varsortability values.")
    summary_rows: list[dict[str, Any]] = []
    for model, group in points.groupby("model", observed=True):
        x = group[sortability].to_numpy(dtype=float)
        y = group["metric_value"].to_numpy(dtype=float)
        valid_regression = len(x) >= 2 and np.unique(x).size >= 2
        regression = stats.linregress(x, y) if valid_regression else None
        spearman = stats.spearmanr(x, y) if valid_regression else None
        summary_rows.append(
            {
                "model": model,
                "n_datasets": len(group),
                "pearson_r": regression.rvalue if regression else np.nan,
                "fit_r_squared": regression.rvalue**2 if regression else np.nan,
                "slope": regression.slope if regression else np.nan,
                "intercept": regression.intercept if regression else np.nan,
                "p_value": regression.pvalue if regression else np.nan,
                "spearman_rho": spearman.statistic if spearman else np.nan,
                "spearman_p_value": spearman.pvalue if spearman else np.nan,
            }
        )
    model_summary = pd.DataFrame(summary_rows)
    model_names = _ordered_models(points["model"].unique(), model_order=model_order)
    columns = max(1, min(int(columns), len(model_names)))
    rows = math.ceil(len(model_names) / columns)
    if axes is None:
        if figsize is None:
            figsize = (4 * columns, 3.3 * rows)
        fig, made_axes = plt.subplots(
            rows, columns, figsize=figsize, squeeze=False, sharex=True, sharey=True
        )
        axes_array = made_axes.ravel()
    else:
        axes_array = np.atleast_1d(axes).ravel()
        if len(axes_array) < len(model_names):
            raise ValueError(
                f"Expected at least {len(model_names)} axes, received {len(axes_array)}."
            )
        fig = axes_array[0].figure
    color_lookup = dict(zip(model_names, _colors(model_names, model_colors)))
    label, _ = _metric_meta(metric)
    summary_lookup = model_summary.set_index("model")
    for axis, model in zip(axes_array, model_names):
        group = points[points["model"].eq(model)]
        x = group[sortability].to_numpy(dtype=float)
        y = group["metric_value"].to_numpy(dtype=float)
        axis.scatter(x, y, color=color_lookup[model], alpha=0.75, s=24)
        row = summary_lookup.loc[model]
        if show_regression and np.isfinite(row["slope"]):
            line_x = np.linspace(x.min(), x.max(), 100)
            axis.plot(
                line_x,
                row["intercept"] + row["slope"] * line_x,
                color=color_lookup[model],
                linewidth=1.5,
            )
        fit_text = (
            f"fit R² = {row['fit_r_squared']:.3f}"
            if np.isfinite(row["fit_r_squared"])
            else "fit R² = n/a"
        )
        if show_title and show_model_titles:
            axis.set_title(f"{_model_display_label(model)}\n{fit_text}", fontsize=font_size + 1)
        axis.grid(alpha=0.2)
    for axis in axes_array[len(model_names) :]:
        axis.set_visible(False)
    for index, axis in enumerate(axes_array[: len(model_names)]):
        if index // columns == rows - 1:
            default_xlabel = (
                "R²-sortability" if sortability == "r2_sortability" else "Varsortability (raw data)"
            )
            axis.set_xlabel(xlabel or default_xlabel, fontsize=font_size)
        if index % columns == 0:
            axis.set_ylabel(ylabel or label, fontsize=font_size)
    sortability_label = "R²-sortability" if sortability == "r2_sortability" else "Varsortability"
    if show_title and show_metric_title:
        fig.suptitle(title or f"{sortability_label} versus {label}", fontsize=title_size)
    _despine(axes_array[: len(model_names)], font_size=font_size)
    fig.tight_layout()
    _save(fig, save_path, dpi, transparent)
    return (
        fig,
        axes_array[: len(model_names)],
        {
            "values": points.reset_index(drop=True),
            "varsortability": varsortability_values.reset_index(drop=True),
            "r2_sort_regress": baseline_values.reset_index(drop=True),
            "model_summary": model_summary,
        },
    )


def _confounding_label(row: pd.Series) -> str:
    observed = float(row.get("obs_conf", 0) or 0) > 0
    latent = float(row.get("lat_conf", 0) or 0) > 0
    if observed and latent:
        return "Obs+Lat"
    if observed:
        return "ObsConf"
    if latent:
        return "LatConf"
    return "NoConf"


def _add_analysis_scenario(df: pd.DataFrame) -> pd.DataFrame:
    out = _add_dataset_attributes(df)
    out["confounding"] = out.apply(_confounding_label, axis=1)
    noise_labels = {
        "Gaussian": "Gaussian",
        "HighGaussian": "High Gaussian",
        "NonGaussian": "Non Gaussian",
        "Non_Gaussian": "Non Gaussian",
    }
    out["scenario"] = (
        out["graph_type"].astype(str)
        + "·"
        + out["confounding"]
        + "·"
        + out["noise"].map(noise_labels).fillna(out["noise"].astype(str))
        + "·"
        + out["mechanism"].astype(str)
    )
    return out


def _assumption_scenarios(
    catalog: pd.DataFrame, model: str, assumptions: pd.DataFrame
) -> tuple[list[str], str]:
    rows = assumptions[assumptions["model"].astype(str).eq(str(model))]
    if rows.empty:
        return (
            catalog["scenario"].tolist(),
            "No model-specific assumption; all selected scenarios.",
        )
    selector_columns = ["scenario", "graph_type", "noise", "mechanism", "confounding"]
    combined = pd.Series(False, index=catalog.index)
    rationales: list[str] = []
    for _, assumption in rows.iterrows():
        mask = pd.Series(True, index=catalog.index)
        for column in selector_columns:
            if column not in assumptions.columns:
                continue
            value = assumption.get(column)
            if value is None or (not isinstance(value, (list, tuple, set)) and pd.isna(value)):
                continue
            if isinstance(value, str) and _normalise_token(value) in {"", "all", "any"}:
                continue
            wanted = {_normalise_token(item) for item in _as_list(value) or []}
            mask &= catalog[column].astype(str).map(_normalise_token).isin(wanted)
        combined |= mask
        rationale = assumption.get("rationale")
        if isinstance(rationale, str) and rationale and (rationale not in rationales):
            rationales.append(rationale)
    return (catalog.loc[combined, "scenario"].tolist(), " ".join(rationales))


def plot_assumption_match_heatmap(
    data: FinalVisualizationData,
    *,
    metric: str = "f1",
    assumptions: pd.DataFrame | None = None,
    noises: Any = None,
    mechanisms: Any = None,
    datasets: Any = None,
    n_nodes: Any = None,
    graph_type: Any = None,
    obs_conf: Any = None,
    lat_conf: Any = None,
    models: Any = None,
    filters: Mapping[str, Any] | None = None,
    coverage_policy: str = "complete",
    min_completed_runs: int | None = None,
    min_coverage_fraction: float | None = None,
    model_order: Sequence[str] | None = None,
    higher_is_better: bool | None = None,
    title: str | None = None,
    show_title: bool = True,
    show_metric_title: bool = True,
    xlabel: str = "Evaluated scenario",
    ylabel: str = "",
    cmap: Any = OKABE_ITO_DIVERGING,
    annotate: bool = True,
    annotation_format: str = "+.3f",
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> tuple[plt.Figure, plt.Axes, dict[str, pd.DataFrame]]:
    """Plot scenario deltas relative to each model's assumed scenario mean.

    ``assumptions`` uses long format with a required ``model`` column and any
    of ``scenario``, ``graph_type``, ``noise``, ``mechanism`` or
    ``confounding``. Empty cells are wildcards and rows for one model are OR'ed.
    """
    assumption_table = DEFAULT_ASSUMPTIONS.copy() if assumptions is None else assumptions.copy()
    _require_columns(assumption_table, ["model"], "assumptions DataFrame")
    selected = _filter_frame(
        data.metrics,
        datasets=datasets,
        n_nodes=n_nodes,
        graph_type=graph_type,
        noise=noises,
        mechanism=mechanisms,
        obs_conf=obs_conf,
        lat_conf=lat_conf,
        models=models,
        filters=filters,
    )
    selected = _add_analysis_scenario(selected)
    scenario_order = list(dict.fromkeys(selected["scenario"]))
    scenario_groups = {scenario: [scenario] for scenario in scenario_order}
    table, summary, coverage = _factor_performance_table(
        selected,
        metric,
        factor="scenario",
        groups=scenario_groups,
        coverage_policy=coverage_policy,
        min_completed_runs=min_completed_runs,
        min_coverage_fraction=min_coverage_fraction,
        model_order=model_order,
    )
    catalog = (
        selected[["scenario", "graph_type", "noise", "mechanism", "confounding"]]
        .drop_duplicates("scenario")
        .set_index("scenario", drop=False)
        .reindex(table.columns)
        .reset_index(drop=True)
    )
    _, higher = _metric_meta(metric, higher_is_better)
    delta = table.copy()
    reference_rows: list[dict[str, Any]] = []
    for model in table.index:
        assumed, rationale = _assumption_scenarios(catalog, str(model), assumption_table)
        available = [scenario for scenario in assumed if pd.notna(table.loc[model, scenario])]
        reference = float(table.loc[model, available].mean()) if available else np.nan
        difference = table.loc[model] - reference
        delta.loc[model] = difference if higher else -difference
        reference_rows.append(
            {
                "model": model,
                "reference_mean": reference,
                "assumed_scenarios": tuple(assumed),
                "available_assumed_scenarios": tuple(available),
                "complete_reference": len(available) == len(assumed) and bool(assumed),
                "rationale": rationale,
            }
        )
    references = pd.DataFrame(reference_rows).set_index("model")
    label, _ = _metric_meta(metric, higher_is_better)
    if ax is None:
        if figsize is None:
            figsize = (1.65 * len(delta.columns) + 3, 0.47 * len(delta.index) + 2)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    _draw_table_heatmap(
        ax,
        delta,
        cmap=cmap,
        colorbar_label=f"Δ {label}",
        annotate=annotate,
        annotation_format=annotation_format,
        center_zero=True,
        font_size=font_size,
    )
    if show_title and (title is not None or show_metric_title):
        ax.set_title(
            title or f"Assumption match — Δ {label} from assumed-scenario mean",
            fontsize=title_size,
        )
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    _despine(ax, font_size=font_size)
    fig.tight_layout()
    _save(fig, save_path, dpi, transparent)
    return (
        fig,
        ax,
        {
            "performance_table": table,
            "delta_table": delta,
            "reference_table": references,
            "summary": summary,
            "coverage": coverage,
            "assumptions": assumption_table,
            "scenario_catalog": catalog,
        },
    )


ASSUMPTION_CONTRAST_ORDER: tuple[str, ...] = (
    "Noise Assumption",
    "Mechanism Assumption",
    "Observed Confounding",
    "Latent Confounding",
    "Observed + Latent Confounding",
)


def _assumed_dataset_ids(
    catalog: pd.DataFrame,
    model: str,
    assumptions: pd.DataFrame,
    contrast: str | None = None,
) -> set[str]:
    """Return datasets satisfying a model's declared assumption.

    When ``contrast`` is supplied, only the dimension being tested is used.
    This isolates, for example, the latent-confounding assumption from a
    model's separate noise or mechanism assumptions.
    """
    model_rows = assumptions[assumptions["model"].astype(str).eq(str(model))]
    if model_rows.empty:
        return set()
    contrast_columns = {
        "Noise Assumption": ["noise"],
        "Mechanism Assumption": ["mechanism"],
        "Observed Confounding": ["confounding"],
        "Latent Confounding": ["confounding"],
        "Observed + Latent Confounding": ["confounding"],
    }
    selector_columns = contrast_columns.get(
        contrast,
        ["dataset", "scenario", "graph_type", "noise", "mechanism", "confounding"],
    )
    if contrast is not None and (
        not any(
            (
                column in assumptions.columns
                and model_rows[column]
                .map(
                    lambda value: (
                        not (
                            value is None
                            or (not isinstance(value, (list, tuple, set)) and pd.isna(value))
                            or (
                                isinstance(value, str)
                                and _normalise_token(value) in {"", "all", "any"}
                            )
                        )
                    )
                )
                .any()
                for column in selector_columns
            )
        )
    ):
        return set()
    combined = pd.Series(False, index=catalog.index)
    for _, assumption in model_rows.iterrows():
        mask = pd.Series(True, index=catalog.index)
        for column in selector_columns:
            if column not in assumptions.columns or column not in catalog.columns:
                continue
            value = assumption.get(column)
            if value is None or (not isinstance(value, (list, tuple, set)) and pd.isna(value)):
                continue
            if isinstance(value, str) and _normalise_token(value) in {"", "all", "any"}:
                continue
            wanted = {_normalise_token(item) for item in _as_list(value) or []}
            mask &= catalog[column].astype(str).map(_normalise_token).isin(wanted)
        combined |= mask
    return set(catalog.loc[combined, "dataset"].astype(str))


def _assumption_contrast_pairs(catalog: pd.DataFrame) -> pd.DataFrame:
    """Build clean dataset pairs that differ in exactly one assumption."""
    work = catalog.copy()
    for column in ["graph_type", "noise", "mechanism", "confounding"]:
        work[f"_{column}_key"] = work[column].astype(str).map(_normalise_token)
    specifications = (
        (
            "Noise Assumption",
            "_noise_key",
            "gaussian",
            "nongaussian",
            ["n_nodes", "_graph_type_key", "_mechanism_key", "_confounding_key"],
        ),
        (
            "Mechanism Assumption",
            "_mechanism_key",
            "linear",
            "mixed",
            ["n_nodes", "_graph_type_key", "_noise_key", "_confounding_key"],
        ),
        (
            "Observed Confounding",
            "_confounding_key",
            "noconf",
            "obsconf",
            ["n_nodes", "_graph_type_key", "_noise_key", "_mechanism_key"],
        ),
        (
            "Latent Confounding",
            "_confounding_key",
            "noconf",
            "latconf",
            ["n_nodes", "_graph_type_key", "_noise_key", "_mechanism_key"],
        ),
        (
            "Observed + Latent Confounding",
            "_confounding_key",
            "noconf",
            "obslat",
            ["n_nodes", "_graph_type_key", "_noise_key", "_mechanism_key"],
        ),
    )
    pair_frames: list[pd.DataFrame] = []
    for (
        assumption,
        varied_column,
        first_value,
        second_value,
        controls,
    ) in specifications:
        first = work[work[varied_column].eq(first_value)][controls + ["dataset"]]
        second = work[work[varied_column].eq(second_value)][controls + ["dataset"]]
        pairs = first.merge(second, on=controls, suffixes=("_a", "_b"))
        if pairs.empty:
            continue
        pairs.insert(0, "assumption", assumption)
        pair_frames.append(pairs[["assumption", "dataset_a", "dataset_b", *controls]])
    if not pair_frames:
        return pd.DataFrame(columns=["assumption", "dataset_a", "dataset_b"])
    pairs = pd.concat(pair_frames, ignore_index=True).drop_duplicates(
        ["assumption", "dataset_a", "dataset_b"]
    )
    pairs["pair_id"] = np.arange(len(pairs))
    return pairs


def _assumption_contrast_level(catalog_lookup: pd.DataFrame, dataset: str, assumption: str) -> str:
    """Return the human-readable level changed by an assumption contrast."""
    row = catalog_lookup.loc[dataset]
    if assumption == "Noise Assumption":
        return _dataset_display_title(row["noise"])
    if assumption == "Mechanism Assumption":
        return _dataset_display_title(row["mechanism"])
    confounding_labels = {
        "NoConf": "No Confounding",
        "ObsConf": "Observed Confounding",
        "LatConf": "Latent Confounding",
        "Obs+Lat": "Observed + Latent Confounding",
    }
    return confounding_labels.get(
        str(row["confounding"]), _dataset_display_title(row["confounding"])
    )


def build_assumption_degradation_data(
    data: FinalVisualizationData,
    *,
    metric: str = "f1",
    assumptions: pd.DataFrame | None = None,
    datasets: Any = None,
    n_nodes: Any = None,
    graph_type: Any = None,
    models: Any = None,
    filters: Mapping[str, Any] | None = None,
    uncertainty: str = "sd",
    confidence_level: float = 0.95,
    ci_method: str = "bootstrap",
    n_bootstrap: int = 2000,
    random_state: int | None = 42,
    higher_is_better: bool | None = None,
) -> dict[str, pd.DataFrame]:
    """Calculate paired performance change when assumptions are violated.

    Model seeds are averaged first. For each clean scenario pair, exactly one
    side must satisfy *all* assumptions declared for that model; that side
    becomes the matched reference. The paired side differs only in the tested
    dimension, so every contrast violates one assumption while keeping the
    remaining assumptions satisfied. Pair differences are macro-averaged
    within each dataset seed before mean ± sample SD is calculated (or the requested uncertainty).
    The sign is oriented so negative values always mean worse performance
    under the violation, for both higher-is-better and lower-is-better metrics.
    """
    assumption_table = DEFAULT_ASSUMPTIONS.copy() if assumptions is None else assumptions.copy()
    _require_columns(assumption_table, ["model"], "assumptions DataFrame")
    assumption_table["model"] = assumption_table["model"].map(_canonical_model_name)
    selected = _filter_frame(
        data.metrics,
        datasets=datasets,
        n_nodes=n_nodes,
        graph_type=graph_type,
        models=models,
        filters=filters,
    )
    selected = _add_analysis_scenario(selected)
    blocks = _block_means(selected, metric)
    blocks = _add_analysis_scenario(blocks)
    catalog_columns = [
        "dataset",
        "n_nodes",
        "graph_type",
        "noise",
        "mechanism",
        "obs_conf",
        "lat_conf",
        "confounding",
        "scenario",
    ]
    catalog = selected[catalog_columns].drop_duplicates("dataset").reset_index(drop=True)
    contrast_pairs = _assumption_contrast_pairs(catalog)
    catalog_lookup = catalog.set_index("dataset")
    _, higher = _metric_meta(metric, higher_is_better)
    difference_rows: list[dict[str, Any]] = []
    applicable_rows: list[dict[str, Any]] = []
    for model in _ordered_models(blocks["model"].unique()):
        model_blocks = blocks[blocks["model"].eq(model)]
        for pair in contrast_pairs.itertuples(index=False):
            assumed = _assumed_dataset_ids(catalog, model, assumption_table, contrast=None)
            if not assumed:
                continue
            a_assumed = pair.dataset_a in assumed
            b_assumed = pair.dataset_b in assumed
            if a_assumed == b_assumed:
                continue
            matched_dataset = pair.dataset_a if a_assumed else pair.dataset_b
            violated_dataset = pair.dataset_b if a_assumed else pair.dataset_a
            matched_level = _assumption_contrast_level(
                catalog_lookup, matched_dataset, pair.assumption
            )
            violated_level = _assumption_contrast_level(
                catalog_lookup, violated_dataset, pair.assumption
            )
            comparison = f"{matched_level} → {violated_level}"
            matched = model_blocks[model_blocks["dataset"].eq(matched_dataset)][
                ["dataset_seed", "metric_value"]
            ].rename(columns={"metric_value": "matched_value"})
            violated = model_blocks[model_blocks["dataset"].eq(violated_dataset)][
                ["dataset_seed", "metric_value"]
            ].rename(columns={"metric_value": "violated_value"})
            paired = matched.merge(violated, on="dataset_seed", how="inner")
            if paired.empty:
                continue
            applicable_rows.append(
                {
                    "pair_id": pair.pair_id,
                    "assumption": pair.assumption,
                    "model": model,
                    "matched_dataset": matched_dataset,
                    "violated_dataset": violated_dataset,
                    "matched_level": matched_level,
                    "violated_level": violated_level,
                    "comparison": comparison,
                    "paired_dataset_seeds": int(paired["dataset_seed"].nunique()),
                }
            )
            raw_change = paired["violated_value"] - paired["matched_value"]
            paired["degradation"] = raw_change if higher else -raw_change
            for row in paired.itertuples(index=False):
                difference_rows.append(
                    {
                        "pair_id": pair.pair_id,
                        "assumption": pair.assumption,
                        "model": model,
                        "dataset_seed": row.dataset_seed,
                        "matched_dataset": matched_dataset,
                        "violated_dataset": violated_dataset,
                        "matched_level": matched_level,
                        "violated_level": violated_level,
                        "comparison": comparison,
                        "matched_value": row.matched_value,
                        "violated_value": row.violated_value,
                        "degradation": row.degradation,
                    }
                )
    pair_differences = pd.DataFrame(difference_rows)
    if pair_differences.empty:
        raise ValueError(
            "No paired matched-versus-violated scenarios are available for the selected models, filters and assumptions."
        )
    seed_degradation = (
        pair_differences.groupby(["model", "assumption", "dataset_seed"], observed=True)
        .agg(
            degradation=("degradation", "mean"),
            matched_value=("matched_value", "mean"),
            violated_value=("violated_value", "mean"),
            paired_scenarios=("pair_id", "nunique"),
        )
        .reset_index()
    )
    summary_rows: list[dict[str, Any]] = []
    for (model, assumption), group in seed_degradation.groupby(
        ["model", "assumption"], observed=True
    ):
        stats_row = _uncertainty_summary(
            group["degradation"],
            uncertainty=uncertainty,
            confidence_level=confidence_level,
            ci_method=ci_method,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        summary_rows.append(
            {
                "model": model,
                "assumption": assumption,
                **stats_row,
                "mean_matched": float(group["matched_value"].mean()),
                "mean_violated": float(group["violated_value"].mean()),
                "min_paired_scenarios": int(group["paired_scenarios"].min()),
                "max_paired_scenarios": int(group["paired_scenarios"].max()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    applicable_pairs = pd.DataFrame(applicable_rows)
    comparison_lookup = (
        applicable_pairs.groupby(["model", "assumption"], observed=True)["comparison"]
        .agg(lambda values: " / ".join(dict.fromkeys(map(str, values))))
        .rename("comparison")
        .reset_index()
    )
    if "rationale" in assumption_table.columns:
        rationale_lookup = (
            assumption_table.groupby("model", observed=True)["rationale"]
            .agg(
                lambda values: " ".join(
                    dict.fromkeys((str(value) for value in values if pd.notna(value)))
                )
            )
            .rename("assumption_rationale")
            .reset_index()
        )
    else:
        rationale_lookup = pd.DataFrame(
            {
                "model": assumption_table["model"].drop_duplicates(),
                "assumption_rationale": "",
            }
        )
    summary = summary.merge(comparison_lookup, on=["model", "assumption"], how="left").merge(
        rationale_lookup, on="model", how="left"
    )
    summary["assumption"] = pd.Categorical(
        summary["assumption"], categories=ASSUMPTION_CONTRAST_ORDER, ordered=True
    )
    summary = summary.sort_values(["assumption", "mean"], ascending=[True, True])
    return {
        "summary": summary.reset_index(drop=True),
        "seed_degradation": seed_degradation,
        "pair_differences": pair_differences,
        "applicable_pairs": applicable_pairs,
        "contrast_pairs": contrast_pairs,
        "scenario_catalog": catalog,
        "assumptions": assumption_table,
    }


def plot_assumption_degradation_forest(
    data: FinalVisualizationData,
    *,
    metric: str = "f1",
    assumptions: pd.DataFrame | None = None,
    datasets: Any = None,
    n_nodes: Any = None,
    graph_type: Any = None,
    models: Any = None,
    filters: Mapping[str, Any] | None = None,
    uncertainty: str = "sd",
    confidence_level: float = 0.95,
    ci_method: str = "bootstrap",
    n_bootstrap: int = 2000,
    random_state: int | None = 42,
    higher_is_better: bool | None = None,
    columns: int = 3,
    annotate: bool = True,
    title: str | None = None,
    show_title: bool = True,
    show_metric_title: bool = True,
    xlabel: str | None = None,
    figsize: tuple[float, float] | None = None,
    axes: Sequence[plt.Axes] | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> tuple[plt.Figure, np.ndarray, dict[str, pd.DataFrame]]:
    """Forest plots of paired degradation for each violated assumption."""
    output = build_assumption_degradation_data(
        data,
        metric=metric,
        assumptions=assumptions,
        datasets=datasets,
        n_nodes=n_nodes,
        graph_type=graph_type,
        models=models,
        filters=filters,
        uncertainty=uncertainty,
        confidence_level=confidence_level,
        ci_method=ci_method,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        higher_is_better=higher_is_better,
    )
    summary = output["summary"]
    assumption_names = [
        value
        for value in ASSUMPTION_CONTRAST_ORDER
        if value in set(summary["assumption"].astype(str))
    ]
    columns = max(1, min(columns, len(assumption_names)))
    rows = math.ceil(len(assumption_names) / columns)
    if axes is None:
        fig, made_axes = plt.subplots(
            rows, columns, figsize=figsize or (5.2 * columns, 4.2 * rows), squeeze=False
        )
        axes_array = made_axes.ravel()
    else:
        axes_array = np.atleast_1d(axes).ravel()
        if len(axes_array) < len(assumption_names):
            raise ValueError(
                f"Expected at least {len(assumption_names)} axes, received {len(axes_array)}."
            )
        fig = axes_array[0].figure
    finite_limits = summary[["lower", "upper"]].to_numpy(dtype=float)
    finite_limits = finite_limits[np.isfinite(finite_limits)]
    limit = max(0.05, float(np.max(np.abs(finite_limits))) if finite_limits.size else 1.0)
    limit *= 1.18
    metric_label, _ = _metric_meta(metric, higher_is_better)
    x_label = xlabel or f"Δ {metric_label} (negative = degradation)"
    for axis, assumption in zip(axes_array, assumption_names):
        rows_for_assumption = (
            summary[summary["assumption"].astype(str).eq(assumption)]
            .sort_values("mean", ascending=True)
            .reset_index(drop=True)
        )
        y = np.arange(len(rows_for_assumption))
        axis.axvline(0, color=OKABE_ITO["black"], linewidth=1, linestyle="--", alpha=0.65)
        for position, row in rows_for_assumption.iterrows():
            colour = OKABE_ITO["vermillion"] if row["mean"] < 0 else OKABE_ITO["blue"]
            axis.errorbar(
                row["mean"],
                position,
                xerr=np.asarray([[row["mean"] - row["lower"]], [row["upper"] - row["mean"]]]),
                fmt="o",
                color=colour,
                capsize=3,
                markersize=5,
            )
            if annotate:
                offset = 0.025 * limit
                axis.text(
                    row["lower"] - offset if row["mean"] < 0 else row["upper"] + offset,
                    position,
                    f"{row['mean']:+.3f}",
                    va="center",
                    ha="right" if row["mean"] < 0 else "left",
                    fontsize=max(6, font_size - 1),
                )
        row_labels = [
            f"{_model_display_label(row.model)}\n{row.comparison}"
            for row in rows_for_assumption.itertuples(index=False)
        ]
        axis.set_yticks(y, row_labels)
        axis.tick_params(axis="y", labelsize=max(6, font_size - 1))
        axis.invert_yaxis()
        axis.set_xlim(-limit, limit)
        axis.set_title(assumption, fontsize=title_size)
        axis.set_xlabel(x_label, fontsize=font_size)
        axis.set_ylabel("", fontsize=font_size)
        axis.grid(axis="x", alpha=0.2)
    for axis in axes_array[len(assumption_names) :]:
        axis.set_visible(False)
    if show_title and (title is not None or show_metric_title):
        fig.suptitle(
            title or "Performance degradation when assumptions are violated",
            fontsize=title_size + 1,
        )
    _despine(axes_array[: len(assumption_names)], font_size=font_size)
    fig.tight_layout()
    _save(fig, save_path, dpi, transparent)
    return (fig, axes_array[: len(assumption_names)], output)


def build_model_assumption_table(
    *,
    assumptions: pd.DataFrame | None = None,
    models: Sequence[str] | str | None = None,
    model_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return one concise declared-assumption row per model.

    A dash means that the model has no restriction for that contrast, the
    assumption was not declared, or both compared levels are compatible.
    """
    assumption_table = DEFAULT_ASSUMPTIONS.copy() if assumptions is None else assumptions.copy()
    _require_columns(assumption_table, ["model"], "assumptions DataFrame")
    assumption_table["model"] = assumption_table["model"].map(_canonical_model_name)
    if models is None:
        selected_models = assumption_table["model"].drop_duplicates().tolist()
    else:
        selected_models = [_canonical_model_name(value) for value in _as_list(models) or []]
    ordered_models = _ordered_models(selected_models, model_order=model_order)
    specifications = {
        "Noise Assumption": (
            "noise",
            {"gaussian": "Gaussian", "nongaussian": "Non Gaussian"},
        ),
        "Mechanism Assumption": ("mechanism", {"linear": "Linear", "mixed": "Mixed"}),
        "Observed Confounding": (
            "confounding",
            {"noconf": "No Confounding", "obsconf": "Observed Confounding"},
        ),
        "Latent Confounding": (
            "confounding",
            {"noconf": "No Confounding", "latconf": "Latent Confounding"},
        ),
        "Observed + Latent Confounding": (
            "confounding",
            {"noconf": "No Confounding", "obslat": "Observed + Latent Confounding"},
        ),
    }
    rows: list[dict[str, str]] = []
    for model in ordered_models:
        model_rows = assumption_table[assumption_table["model"].astype(str).eq(str(model))]
        row: dict[str, str] = {"Model": str(model)}
        for label, (column, level_labels) in specifications.items():
            declared: set[str] = set()
            if column in model_rows.columns:
                for value in model_rows[column]:
                    if value is None or (
                        not isinstance(value, (list, tuple, set)) and pd.isna(value)
                    ):
                        continue
                    declared.update((_normalise_token(item) for item in _as_list(value) or []))
            relevant = [key for key in level_labels if key in declared]
            row[label] = level_labels[relevant[0]] if len(relevant) == 1 else "–"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


def plot_model_assumption_table(
    *,
    assumptions: pd.DataFrame | None = None,
    models: Sequence[str] | str | None = None,
    model_order: Sequence[str] | None = None,
    title: str | None = "Model assumptions",
    show_title: bool = True,
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Render and return the concise model-by-assumption table."""
    table = build_model_assumption_table(
        assumptions=assumptions, models=models, model_order=model_order
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (16, max(4.5, 0.48 * len(table.index) + 1.8)))
    else:
        fig = ax.figure
    ax.axis("off")
    artist = ax.table(
        cellText=table.to_numpy(dtype=object),
        rowLabels=table.index.tolist(),
        colLabels=table.columns.tolist(),
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    artist.auto_set_font_size(False)
    artist.set_fontsize(max(7, font_size - 2))
    artist.scale(1.0, 1.45)
    for (row, column), cell in artist.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor(OKABE_ITO["blue"])
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif column == -1:
            cell.set_facecolor("#E5E7EB")
            cell.get_text().set_weight("bold")
        elif cell.get_text().get_text() == "–":
            cell.set_facecolor("#F3F4F6")
            cell.get_text().set_color("#6B7280")
        else:
            cell.set_facecolor("#E8F1F8" if row % 2 else "#DCEAF4")
    if show_title and title:
        ax.set_title(title, fontsize=title_size, pad=14)
    fig.tight_layout()
    _save(fig, save_path, dpi, transparent)
    return (fig, ax, table)


def plot_assumption_degradation_heatmap(
    data: FinalVisualizationData,
    *,
    metric: str = "f1",
    assumptions: pd.DataFrame | None = None,
    datasets: Any = None,
    n_nodes: Any = None,
    graph_type: Any = None,
    models: Any = None,
    filters: Mapping[str, Any] | None = None,
    uncertainty: str = "sd",
    confidence_level: float = 0.95,
    ci_method: str = "bootstrap",
    n_bootstrap: int = 2000,
    random_state: int | None = 42,
    higher_is_better: bool | None = None,
    cmap: Any = OKABE_ITO_DIVERGING,
    annotate: bool = True,
    show_comparisons: bool | None = None,
    annotation_format: str = "+.3f",
    title: str | None = None,
    show_title: bool = True,
    show_metric_title: bool = True,
    xlabel: str = "Violated assumption",
    ylabel: str = "",
    figsize: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> tuple[plt.Figure, plt.Axes, dict[str, pd.DataFrame]]:
    """Compact heatmap of signed change; negative values are degradation.

    ``show_comparisons`` is accepted for backwards compatibility with older
    notebook cells but is intentionally ignored: cell annotations always show
    only the numeric difference.
    """
    output = build_assumption_degradation_data(
        data,
        metric=metric,
        assumptions=assumptions,
        datasets=datasets,
        n_nodes=n_nodes,
        graph_type=graph_type,
        models=models,
        filters=filters,
        uncertainty=uncertainty,
        confidence_level=confidence_level,
        ci_method=ci_method,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        higher_is_better=higher_is_better,
    )
    summary = output["summary"]
    table = summary.pivot(index="model", columns="assumption", values="mean")
    present_columns = [name for name in ASSUMPTION_CONTRAST_ORDER if name in table.columns]
    table = table.reindex(columns=present_columns)
    assumption_models = set(output["assumptions"]["model"].astype(str))
    available_models = set(data.metrics["model"].map(_canonical_model_name))
    declared_models = assumption_models & available_models
    if models is not None:
        requested_models = {_canonical_model_name(value) for value in _as_list(models) or []}
        declared_models &= requested_models
    model_order = list(table.mean(axis=1, skipna=True).sort_values(ascending=True).index)
    model_order.extend(
        (model for model in _ordered_models(declared_models) if model not in model_order)
    )
    table = table.reindex(index=model_order)
    output["degradation_table"] = table
    output["comparison_table"] = pd.DataFrame("–", index=table.index, columns=table.columns)
    metric_label, _ = _metric_meta(metric, higher_is_better)
    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize or (2.25 * len(table.columns) + 3.5, 0.48 * len(table.index) + 2.2)
        )
    else:
        fig = ax.figure
    _draw_table_heatmap(
        ax,
        table,
        cmap=cmap,
        colorbar_label=f"Δ {metric_label} (negative = degradation)",
        annotate=annotate,
        annotation_format=annotation_format,
        center_zero=True,
        font_size=font_size,
    )
    if show_title and (title is not None or show_metric_title):
        ax.set_title(title or "Degradation under violated model assumptions", fontsize=title_size)
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    _despine(ax, font_size=font_size)
    fig.tight_layout()
    _save(fig, save_path, dpi, transparent)
    return (fig, ax, output)


def build_compact_rank_table(rank_output: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Condense a rank-heatmap output into one auditable row per model."""
    if "rank_table" not in rank_output:
        raise ValueError("rank_output must contain a 'rank_table'.")
    ranks = rank_output["rank_table"].apply(pd.to_numeric, errors="coerce")
    table = pd.DataFrame(
        {
            "median_rank": ranks.median(axis=1, skipna=True),
            "mean_rank": ranks.mean(axis=1, skipna=True),
            "best_rank": ranks.min(axis=1, skipna=True),
            "worst_rank": ranks.max(axis=1, skipna=True),
            "top_tier_datasets": ranks.eq(1).sum(axis=1),
            "datasets_ranked": ranks.notna().sum(axis=1),
        }
    )
    table.index.name = "model"
    return table.sort_values(
        ["median_rank", "mean_rank", "top_tier_datasets"],
        ascending=[True, True, False],
        kind="stable",
    ).reset_index()


def build_noise_mechanism_table(
    mechanism_output: Mapping[str, Mapping[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Combine mechanism heatmap tables into one model x scenario DataFrame."""
    tables: list[pd.DataFrame] = []
    for noise, output in mechanism_output.items():
        if "table" not in output:
            raise ValueError(f"Mechanism output for '{noise}' has no 'table'.")
        table = output["table"].copy()
        table.columns = [
            f"{_dataset_display_title(noise)} {_dataset_display_title(mechanism)}"
            for mechanism in table.columns
        ]
        tables.append(table)
    if not tables:
        return pd.DataFrame()
    combined = pd.concat(tables, axis=1)
    combined.index.name = "model"
    return combined.reset_index()


def build_scalability_compact_table(
    scalability_output: Mapping[str, Any], *, node_counts: Sequence[int] = (10, 70)
) -> pd.DataFrame:
    """Return selected F1 values and the largest evaluated size per model/family."""
    if "summary" not in scalability_output:
        raise ValueError("scalability_output must contain a 'summary' table.")
    summary = scalability_output["summary"].copy()
    _require_columns(
        summary,
        ["scalability_family", "model", "n_nodes", "mean"],
        "scalability summary",
    )
    selected = summary[summary["n_nodes"].isin(list(node_counts))]
    values = selected.pivot_table(
        index=["scalability_family", "model"],
        columns="n_nodes",
        values="mean",
        aggfunc="first",
    )
    values = values.rename(columns={value: f"f1_at_{int(value)}_nodes" for value in values.columns})
    maximum = (
        summary.dropna(subset=["mean"])
        .groupby(["scalability_family", "model"], observed=True)["n_nodes"]
        .max()
        .rename("largest_evaluated_n_nodes")
    )
    return values.join(maximum, how="outer").reset_index()


def plot_noise_mechanism_slope(
    mechanism_output: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    highlight_models: Sequence[str] | str | None = None,
    highlight_n_models: int = 5,
    show_other_models: bool = True,
    model_colors: Mapping[str, str] | None = None,
    ylabel: str = F1_LABEL,
    show_title: bool = True,
    figsize: tuple[float, float] | None = None,
    axes: Sequence[plt.Axes] | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
) -> tuple[plt.Figure, np.ndarray, dict[str, Any]]:
    """Show Linear-to-Mixed changes as one slope panel per noise family."""
    if not mechanism_output:
        raise ValueError("mechanism_output is empty.")
    tables = {
        str(noise): output["table"].apply(pd.to_numeric, errors="coerce")
        for noise, output in mechanism_output.items()
    }
    model_names = _ordered_models(
        set().union(*(set(table.index.astype(str)) for table in tables.values()))
    )
    overall = pd.concat(
        [table.reindex(model_names).mean(axis=1) for table in tables.values()], axis=1
    ).mean(axis=1)
    if highlight_models is None:
        highlighted = overall.nlargest(max(1, highlight_n_models)).index.tolist()
    else:
        highlighted = [_canonical_model_name(value) for value in _as_list(highlight_models) or []]
    colour_lookup = dict(zip(model_names, _colors(model_names, model_colors)))
    if axes is None:
        fig, made_axes = plt.subplots(
            1,
            len(tables),
            sharey=True,
            figsize=figsize or (5.6 * len(tables), 5.5),
            squeeze=False,
        )
        axes_array = made_axes.ravel()
    else:
        axes_array = np.atleast_1d(axes).ravel()
        if len(axes_array) != len(tables):
            raise ValueError(f"Expected {len(tables)} axes, received {len(axes_array)}.")
        fig = axes_array[0].figure
    long_rows: list[dict[str, Any]] = []
    for axis, (noise, table) in zip(axes_array, tables.items()):
        mechanisms = list(map(str, table.columns))
        x = np.arange(len(mechanisms))
        for model in model_names:
            values = table.reindex(index=[model]).iloc[0].to_numpy(dtype=float)
            if not np.isfinite(values).any():
                continue
            is_highlighted = model in highlighted
            if not is_highlighted and (not show_other_models):
                continue
            colour = colour_lookup[model]
            axis.plot(
                x,
                values,
                marker="o",
                color=colour,
                linewidth=2 if is_highlighted else 0.8,
                markersize=5 if is_highlighted else 3,
                alpha=1 if is_highlighted else 0.3,
                label=_model_display_label(model) if is_highlighted else "_nolegend_",
            )
            long_rows.extend(
                (
                    {
                        "noise": noise,
                        "mechanism": mechanism,
                        "model": model,
                        "value": value,
                    }
                    for mechanism, value in zip(mechanisms, values)
                )
            )
        axis.set_xticks(x, [_dataset_display_title(value) for value in mechanisms])
        axis.set_xlabel("Mechanism", fontsize=font_size)
        axis.set_ylabel(ylabel if axis is axes_array[0] else "", fontsize=font_size)
        if show_title:
            axis.set_title(_dataset_display_title(noise), fontsize=title_size)
        axis.grid(axis="y", alpha=0.22)
    handles, labels = axes_array[-1].get_legend_handles_labels()
    if handles:
        axes_array[-1].legend(
            handles,
            labels,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=max(7, font_size - 1),
        )
    _despine(axes_array, font_size=font_size)
    fig.tight_layout()
    return (
        fig,
        axes_array,
        {
            "values": pd.DataFrame(long_rows),
            "highlighted_models": pd.DataFrame({"model": highlighted}),
            "tables": tables,
        },
    )


def plot_mechanism_change_bars(
    mechanism_output: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    baseline: str = "Linear",
    comparison: str = "Mixed",
    order_by: str = "comparison",
    model_order: Sequence[str] | None = None,
    xlim: tuple[float, float] = (0.0, 1.0),
    gain_color: str = OKABE_ITO["blue"],
    loss_color: str = OKABE_ITO["vermillion"],
    neutral_color: str = "#7A7A7A",
    baseline_color: str = "#D9D9D9",
    annotate_delta: bool = False,
    delta_threshold: float = 0.005,
    xlabel: str = F1_LABEL,
    title: str | None = "Mechanism sensitivity",
    show_title: bool = True,
    figsize: tuple[float, float] | None = None,
    axes: Sequence[plt.Axes] | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> tuple[plt.Figure, np.ndarray, dict[str, Any]]:
    """Show a baseline score and the absolute change to another mechanism.

    The pale horizontal bar ends at the baseline score. The coloured segment
    then connects that absolute score to the comparison score: blue denotes a
    gain and vermillion a loss. This deliberately avoids stacked bars, which
    would incorrectly imply that the two F1 values are additive.
    """
    if not mechanism_output:
        raise ValueError("mechanism_output is empty.")

    def resolve_column(table: pd.DataFrame, requested: str) -> str:
        matches = [
            str(column)
            for column in table.columns
            if _normalise_token(column) == _normalise_token(requested)
        ]
        if not matches:
            raise ValueError(
                f"Mechanism '{requested}' is unavailable; found {list(table.columns)}."
            )
        return matches[0]

    tables: dict[str, pd.DataFrame] = {}
    rows: list[pd.DataFrame] = []
    for noise, output in mechanism_output.items():
        table = output["table"].apply(pd.to_numeric, errors="coerce")
        baseline_column = resolve_column(table, baseline)
        comparison_column = resolve_column(table, comparison)
        paired = table[[baseline_column, comparison_column]].rename(
            columns={baseline_column: "baseline", comparison_column: "comparison"}
        )
        paired.index = paired.index.astype(str)
        tables[str(noise)] = paired
        long = paired.rename_axis("model").reset_index()
        long.insert(0, "noise", str(noise))
        rows.append(long)
    values = pd.concat(rows, ignore_index=True)
    values["delta"] = values["comparison"] - values["baseline"]
    present_models = set(values["model"])
    if model_order is not None:
        order = _ordered_models(present_models, model_order=model_order)
    else:
        aggregate = values.groupby("model", observed=True).agg(
            baseline=("baseline", "mean"),
            comparison=("comparison", "mean"),
            delta=("delta", "mean"),
        )
        if order_by == "baseline":
            sort_values = aggregate["baseline"]
        elif order_by == "comparison":
            sort_values = aggregate["comparison"]
        elif order_by == "delta":
            sort_values = aggregate["delta"]
        elif order_by == "mean":
            sort_values = aggregate[["baseline", "comparison"]].mean(axis=1)
        else:
            raise ValueError("order_by must be 'baseline', 'comparison', 'delta', or 'mean'.")
        order = sort_values.sort_values(ascending=False, na_position="last").index.tolist()
    if axes is None:
        fig, made_axes = plt.subplots(
            1,
            len(tables),
            sharex=True,
            sharey=True,
            squeeze=False,
            figsize=figsize or (5.2 * len(tables), max(6.0, 0.43 * len(order) + 1.8)),
        )
        axes_array = made_axes.ravel()
    else:
        axes_array = np.atleast_1d(axes).ravel()
        if len(axes_array) != len(tables):
            raise ValueError(f"Expected {len(tables)} axes, received {len(axes_array)}.")
        fig = axes_array[0].figure
    y = np.arange(len(order))
    for axis, (noise, table) in zip(axes_array, tables.items()):
        panel = table.reindex(order)
        baseline_values = panel["baseline"].to_numpy(dtype=float)
        comparison_values = panel["comparison"].to_numpy(dtype=float)
        deltas = comparison_values - baseline_values
        finite = np.isfinite(baseline_values) & np.isfinite(comparison_values)
        colours = np.where(
            deltas > delta_threshold,
            gain_color,
            np.where(deltas < -delta_threshold, loss_color, neutral_color),
        )
        axis.barh(
            y[finite],
            baseline_values[finite],
            height=0.34,
            color=baseline_color,
            edgecolor="none",
            zorder=1,
        )
        for position in np.flatnonzero(finite):
            axis.plot(
                [baseline_values[position], comparison_values[position]],
                [position, position],
                color=colours[position],
                linewidth=3.0,
                solid_capstyle="round",
                zorder=2,
            )
        axis.scatter(
            baseline_values[finite],
            y[finite],
            s=28,
            marker="o",
            facecolor="white",
            edgecolor="#555555",
            linewidth=1.0,
            zorder=3,
        )
        axis.scatter(
            comparison_values[finite],
            y[finite],
            s=32,
            marker="D",
            color=colours[finite],
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        if annotate_delta:
            span = xlim[1] - xlim[0]
            for position in np.flatnonzero(finite):
                value = comparison_values[position]
                align_right = value > xlim[1] - 0.09 * span
                axis.text(
                    value + (-0.015 if align_right else 0.015) * span,
                    position,
                    f"{deltas[position]:+.2f}",
                    ha="right" if align_right else "left",
                    va="center",
                    fontsize=max(7, font_size - 3),
                    color=colours[position],
                )
        axis.set_title(_dataset_display_title(noise), fontsize=title_size)
        axis.set_xlim(*xlim)
        axis.set_xlabel(xlabel, fontsize=font_size)
        axis.set_yticks(y, [_model_display_label(model) for model in order])
        axis.grid(axis="x", alpha=0.18, linewidth=0.8)
        axis.grid(axis="y", visible=False)
        _despine(axis, font_size=font_size)
    axes_array[0].invert_yaxis()
    legend_handles = [
        Patch(facecolor=baseline_color, edgecolor="none", label=f"{baseline} {F1_LABEL}"),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            color=gain_color,
            markeredgecolor="white",
            label=f"{comparison} {F1_LABEL} Gain",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            color=loss_color,
            markeredgecolor="white",
            label=f"{comparison} {F1_LABEL} Loss",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=max(8, font_size - 2),
    )
    if show_title and title:
        fig.suptitle(title, fontsize=title_size + 1, y=1.035)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, save_path, dpi, transparent)
    return (
        fig,
        axes_array,
        {
            "values": values,
            "model_order": pd.DataFrame({"model": order}),
            "tables": tables,
        },
    )


def plot_noise_change_bars(
    noise_output: Mapping[str, Any] | pd.DataFrame,
    *,
    baseline: str = "Gaussian",
    comparisons: Sequence[str] = ("HighGaussian", "NonGaussian"),
    panel_specs: Sequence[Mapping[str, Any]] | None = None,
    order_by: str = "comparison_mean",
    model_order: Sequence[str] | None = None,
    xlim: tuple[float, float] = (0.0, 1.0),
    gain_color: str = OKABE_ITO["blue"],
    loss_color: str = OKABE_ITO["vermillion"],
    neutral_color: str = "#7A7A7A",
    baseline_color: str = "#D9D9D9",
    delta_threshold: float = 0.005,
    xlabel: str = f"{F1_LABEL} score",
    title: str | None = None,
    show_title: bool = False,
    figsize: tuple[float, float] = (11.0, 7.5),
    axes: Sequence[plt.Axes] | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> tuple[plt.Figure, np.ndarray, dict[str, Any]]:
    """Compare one or more noise conditions with a Gaussian reference.

    By default, ``noise_output`` supplies two panels (High Gaussian and Non
    Gaussian).  ``panel_specs`` can add or replace panels whose values come
    from different mechanism-specific noise tables.  Each specification must
    provide ``noise_output`` and ``comparison``; optional ``title`` and
    ``mechanism`` fields are carried into the returned tidy data.  This is
    useful when, for example, a third panel compares Gaussian with High
    Gaussian under a Mixed rather than Linear mechanism.

    Every panel uses the pale bar and open circle for its Gaussian reference.
    The coloured segment ends at the comparison noise: blue denotes a gain
    and vermillion a loss.
    """

    def extract_table(output: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame:
        table = output["table"] if isinstance(output, Mapping) and "table" in output else output
        if not isinstance(table, pd.DataFrame) or table.empty:
            raise ValueError("Each noise_output must contain a non-empty 'table' DataFrame.")
        return table.apply(pd.to_numeric, errors="coerce").copy()

    def resolve_column(table: pd.DataFrame, requested: str) -> str:
        matches = [
            str(column)
            for column in table.columns
            if _normalise_token(column) == _normalise_token(requested)
        ]
        if not matches:
            raise ValueError(f"Noise '{requested}' is unavailable; found {list(table.columns)}.")
        return matches[0]

    if panel_specs is None:
        comparison_names = [str(value) for value in comparisons]
        if not comparison_names:
            raise ValueError("comparisons must contain at least one noise condition.")
        raw_specs = [
            {
                "title": _dataset_display_title(comparison_name),
                "noise_output": noise_output,
                "comparison": comparison_name,
                "mechanism": None,
            }
            for comparison_name in comparison_names
        ]
    else:
        if not panel_specs:
            raise ValueError("panel_specs must contain at least one panel.")
        raw_specs = []
        for spec in panel_specs:
            missing = {"noise_output", "comparison"} - set(spec)
            if missing:
                raise ValueError(f"Each panel specification requires {sorted(missing)}.")
            comparison_name = str(spec["comparison"])
            raw_specs.append(
                {
                    "title": str(spec.get("title", _dataset_display_title(comparison_name))),
                    "noise_output": spec["noise_output"],
                    "comparison": comparison_name,
                    "mechanism": spec.get("mechanism"),
                }
            )
    panels: list[dict[str, Any]] = []
    order_rows: list[pd.DataFrame] = []
    for spec in raw_specs:
        table = extract_table(spec["noise_output"])
        baseline_column = resolve_column(table, baseline)
        comparison_column = resolve_column(table, spec["comparison"])
        paired = table[[baseline_column, comparison_column]].copy()
        paired.columns = ["baseline", "comparison"]
        paired.index = paired.index.astype(str)
        panels.append({**spec, "table": paired})
        order_part = paired.rename_axis("model").reset_index()
        order_part["panel_title"] = spec["title"]
        order_rows.append(order_part)
    ordering_values = pd.concat(order_rows, ignore_index=True)
    present_models = set(ordering_values["model"])
    if model_order is not None:
        order = _ordered_models(present_models, model_order=model_order)
    elif order_by == "baseline":
        order = (
            ordering_values.groupby("model", observed=True)["baseline"]
            .mean()
            .sort_values(ascending=False, na_position="last")
            .index.tolist()
        )
    elif order_by == "comparison_mean":
        order = (
            ordering_values.groupby("model", observed=True)["comparison"]
            .mean()
            .sort_values(ascending=False, na_position="last")
            .index.tolist()
        )
    else:
        raise ValueError("order_by must be 'baseline' or 'comparison_mean'.")
    if axes is None:
        fig, made_axes = plt.subplots(
            1, len(panels), sharex=True, sharey=True, squeeze=False, figsize=figsize
        )
        axes_array = made_axes.ravel()
    else:
        axes_array = np.atleast_1d(axes).ravel()
        if len(axes_array) != len(panels):
            raise ValueError(f"Expected {len(panels)} axes, received {len(axes_array)}.")
        fig = axes_array[0].figure
    y = np.arange(len(order), dtype=float)
    markers = ("D", "s", "^", "v", "P", "X")
    long_rows: list[dict[str, Any]] = []
    for panel_index, (axis, panel_spec) in enumerate(zip(axes_array, panels)):
        panel = panel_spec["table"].reindex(order)
        comparison_name = panel_spec["comparison"]
        marker = markers[panel_index % len(markers)]
        baseline_values = panel["baseline"].to_numpy(dtype=float)
        comparison_values = panel["comparison"].to_numpy(dtype=float)
        deltas = comparison_values - baseline_values
        finite = np.isfinite(baseline_values) & np.isfinite(comparison_values)
        colours = np.where(
            deltas > delta_threshold,
            gain_color,
            np.where(deltas < -delta_threshold, loss_color, neutral_color),
        )
        axis.barh(
            y[finite],
            baseline_values[finite],
            height=0.34,
            color=baseline_color,
            edgecolor="none",
            zorder=1,
        )
        for position in np.flatnonzero(finite):
            axis.plot(
                [baseline_values[position], comparison_values[position]],
                [y[position], y[position]],
                color=colours[position],
                linewidth=3.0,
                solid_capstyle="round",
                zorder=2,
            )
        axis.scatter(
            baseline_values[finite],
            y[finite],
            s=28,
            marker="o",
            facecolor="white",
            edgecolor="#555555",
            linewidth=1.0,
            zorder=3,
        )
        axis.scatter(
            comparison_values[finite],
            y[finite],
            s=32,
            marker=marker,
            color=colours[finite],
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        long_rows.extend(
            (
                {
                    "model": model,
                    "baseline": baseline_values[position],
                    "comparison_name": comparison_name,
                    "panel_title": panel_spec["title"],
                    "mechanism": panel_spec["mechanism"],
                    "comparison": comparison_values[position],
                    "delta": deltas[position],
                }
                for position, model in enumerate(order)
            )
        )
        axis.set_title(panel_spec["title"], fontsize=title_size)
        axis.set_xlim(*xlim)
        axis.set_xlabel(xlabel, fontsize=font_size)
        axis.set_yticks(y, [_model_display_label(model) for model in order])
        axis.grid(axis="x", alpha=0.18, linewidth=0.8)
        axis.grid(axis="y", visible=False)
        _despine(axis, font_size=font_size)
    axes_array[0].invert_yaxis()
    if show_title and title:
        fig.suptitle(title, fontsize=title_size + 1, y=1.035)
    baseline_handle = Patch(
        facecolor=baseline_color,
        edgecolor="none",
        label=f"{_dataset_display_title(baseline)} {F1_LABEL}",
    )
    comparison_handles: list[tuple[Any, Any]] = []
    for panel_index, panel_spec in enumerate(panels):
        marker = markers[panel_index % len(markers)]
        display_name = str(panel_spec["title"]).replace("\n", " ")
        comparison_handles.append(
            (
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    linestyle="none",
                    color=gain_color,
                    markeredgecolor="white",
                    label=f"{display_name} {F1_LABEL} Gain",
                ),
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    linestyle="none",
                    color=loss_color,
                    markeredgecolor="white",
                    label=f"{display_name} {F1_LABEL} Loss",
                ),
            )
        )
    legend_handles = [baseline_handle]
    for gain_handle, loss_handle in comparison_handles:
        legend_handles.extend([gain_handle, loss_handle])
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=max(3, len(panels)),
        frameon=False,
        fontsize=max(8, font_size - 2),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, save_path, dpi, transparent)
    legacy_table: pd.DataFrame | None = None
    if panel_specs is None:
        source_table = extract_table(noise_output)
        baseline_column = resolve_column(source_table, baseline)
        comparison_columns = [resolve_column(source_table, name) for name in comparison_names]
        legacy_table = source_table[[baseline_column, *comparison_columns]].copy()
        legacy_table.columns = ["baseline", *comparison_names]
        legacy_table.index = legacy_table.index.astype(str)
    return (
        fig,
        axes_array,
        {
            "values": pd.DataFrame(long_rows),
            "model_order": pd.DataFrame({"model": order}),
            "table": legacy_table,
            "tables": {str(panel["title"]): panel["table"] for panel in panels},
        },
    )


def plot_noise_change_by_mechanism(
    mechanism_outputs: Mapping[str, Mapping[str, Any] | pd.DataFrame],
    *,
    baseline: str = "Gaussian",
    comparisons: Sequence[str] = ("HighGaussian", "NonGaussian"),
    comparison_labels: Mapping[str, str] | None = None,
    order_by: str = "comparison_mean",
    model_order: Sequence[str] | None = None,
    xlim: tuple[float, float] = (0.0, 1.0),
    gain_color: str = OKABE_ITO["blue"],
    loss_color: str = OKABE_ITO["vermillion"],
    neutral_color: str = "#7A7A7A",
    baseline_color: str = "#D9D9D9",
    delta_threshold: float = 0.005,
    xlabel: str = f"{F1_LABEL} score",
    title: str | None = None,
    show_title: bool = False,
    figsize: tuple[float, float] | None = None,
    axes: Sequence[plt.Axes] | None = None,
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
    save_path: str | Path | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> tuple[plt.Figure, np.ndarray, dict[str, Any]]:
    """Compare noise changes in one panel per fixed mechanism.

    The Gaussian score is the common reference in every panel.  Multiple
    alternative noises are drawn on slightly offset rows so neither change is
    hidden.  Marker shape identifies the noise condition, while colour only
    identifies the direction of change.  This makes the layout the direct
    counterpart of :func:`plot_mechanism_change_bars`: that function fixes
    noise by panel; this one fixes mechanism by panel.
    """
    if not mechanism_outputs:
        raise ValueError("mechanism_outputs must contain at least one mechanism.")
    comparison_names = [str(value) for value in comparisons]
    if not comparison_names:
        raise ValueError("comparisons must contain at least one noise condition.")
    labels = {str(key): str(value) for key, value in (comparison_labels or {}).items()}

    def extract_table(output: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame:
        table = output["table"] if isinstance(output, Mapping) and "table" in output else output
        if not isinstance(table, pd.DataFrame) or table.empty:
            raise ValueError("Each mechanism output must contain a non-empty 'table' DataFrame.")
        return table.apply(pd.to_numeric, errors="coerce").copy()

    def resolve_column(table: pd.DataFrame, requested: str) -> str:
        matches = [
            str(column)
            for column in table.columns
            if _normalise_token(column) == _normalise_token(requested)
        ]
        if not matches:
            raise ValueError(f"Noise '{requested}' is unavailable; found {list(table.columns)}.")
        return matches[0]

    tables: dict[str, pd.DataFrame] = {}
    order_rows: list[pd.DataFrame] = []
    for mechanism, output in mechanism_outputs.items():
        source = extract_table(output)
        baseline_column = resolve_column(source, baseline)
        comparison_columns = [resolve_column(source, value) for value in comparison_names]
        table = source[[baseline_column, *comparison_columns]].copy()
        table.columns = ["baseline", *comparison_names]
        table.index = table.index.astype(str)
        mechanism_name = str(mechanism)
        tables[mechanism_name] = table
        ordering = table.rename_axis("model").reset_index()
        ordering["mechanism"] = mechanism_name
        order_rows.append(ordering)
    ordering_values = pd.concat(order_rows, ignore_index=True)
    present_models = set(ordering_values["model"])
    if model_order is not None:
        order = _ordered_models(present_models, model_order=model_order)
    elif order_by == "baseline":
        order = (
            ordering_values.groupby("model", observed=True)["baseline"]
            .mean()
            .sort_values(ascending=False, na_position="last")
            .index.tolist()
        )
    elif order_by == "comparison_mean":
        comparison_mean = (
            ordering_values.groupby("model", observed=True)[comparison_names].mean().mean(axis=1)
        )
        order = comparison_mean.sort_values(ascending=False, na_position="last").index.tolist()
    else:
        raise ValueError("order_by must be 'baseline' or 'comparison_mean'.")
    if axes is None:
        fig, made_axes = plt.subplots(
            1,
            len(tables),
            sharex=True,
            sharey=True,
            squeeze=False,
            figsize=figsize or (5.2 * len(tables), max(6.0, 0.43 * len(order) + 1.8)),
        )
        axes_array = made_axes.ravel()
    else:
        axes_array = np.atleast_1d(axes).ravel()
        if len(axes_array) != len(tables):
            raise ValueError(f"Expected {len(tables)} axes, received {len(axes_array)}.")
        fig = axes_array[0].figure
    markers = ("D", "s", "^", "v", "P", "X")
    if len(comparison_names) == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-0.11, 0.11, len(comparison_names))
    y = np.arange(len(order), dtype=float)
    long_rows: list[dict[str, Any]] = []
    for axis, (mechanism, table) in zip(axes_array, tables.items()):
        panel = table.reindex(order)
        baseline_values = panel["baseline"].to_numpy(dtype=float)
        baseline_finite = np.isfinite(baseline_values)
        axis.barh(
            y[baseline_finite],
            baseline_values[baseline_finite],
            height=0.42,
            color=baseline_color,
            edgecolor="none",
            zorder=1,
        )
        for comparison_index, (comparison_name, offset) in enumerate(
            zip(comparison_names, offsets)
        ):
            comparison_values = panel[comparison_name].to_numpy(dtype=float)
            deltas = comparison_values - baseline_values
            finite = baseline_finite & np.isfinite(comparison_values)
            colours = np.where(
                deltas > delta_threshold,
                gain_color,
                np.where(deltas < -delta_threshold, loss_color, neutral_color),
            )
            positions = y + offset
            for position in np.flatnonzero(finite):
                axis.plot(
                    [baseline_values[position], comparison_values[position]],
                    [positions[position], positions[position]],
                    color=colours[position],
                    linewidth=2.4,
                    solid_capstyle="round",
                    zorder=2,
                )
            axis.scatter(
                baseline_values[finite],
                positions[finite],
                s=23,
                marker="o",
                facecolor="white",
                edgecolor="#555555",
                linewidth=0.9,
                zorder=3,
            )
            axis.scatter(
                comparison_values[finite],
                positions[finite],
                s=31,
                marker=markers[comparison_index % len(markers)],
                color=colours[finite],
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )
            long_rows.extend(
                (
                    {
                        "model": model,
                        "mechanism": mechanism,
                        "baseline_name": baseline,
                        "baseline": baseline_values[position],
                        "comparison_name": comparison_name,
                        "comparison": comparison_values[position],
                        "delta": deltas[position],
                    }
                    for position, model in enumerate(order)
                )
            )
        axis.set_title(_dataset_display_title(mechanism), fontsize=title_size)
        axis.set_xlim(*xlim)
        axis.set_xlabel(xlabel, fontsize=font_size)
        axis.set_yticks(y, [_model_display_label(model) for model in order])
        axis.grid(axis="x", alpha=0.18, linewidth=0.8)
        axis.grid(axis="y", visible=False)
        _despine(axis, font_size=font_size)
    axes_array[0].invert_yaxis()
    if show_title and title:
        fig.suptitle(title, fontsize=title_size + 1, y=1.035)
    legend_handles: list[Any] = [
        Patch(
            facecolor=baseline_color,
            edgecolor="none",
            label=f"{_dataset_display_title(baseline)} {F1_LABEL}",
        )
    ]
    if len(comparison_names) == 1:
        comparison_name = comparison_names[0]
        display_name = labels.get(comparison_name, _dataset_display_title(comparison_name))
        legend_handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker=markers[0],
                    linestyle="none",
                    color=gain_color,
                    markeredgecolor="white",
                    label=f"{display_name} {F1_LABEL} Gain",
                ),
                Line2D(
                    [0],
                    [0],
                    marker=markers[0],
                    linestyle="none",
                    color=loss_color,
                    markeredgecolor="white",
                    label=f"{display_name} {F1_LABEL} Loss",
                ),
            ]
        )
    else:
        for comparison_index, comparison_name in enumerate(comparison_names):
            display_name = labels.get(comparison_name, _dataset_display_title(comparison_name))
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=markers[comparison_index % len(markers)],
                    linestyle="none",
                    color=neutral_color,
                    markeredgecolor="white",
                    label=display_name,
                )
            )
        legend_handles.extend(
            [
                Line2D([0, 1], [0, 0], color=gain_color, linewidth=2.4, label="Gain"),
                Line2D([0, 1], [0, 0], color=loss_color, linewidth=2.4, label="Loss"),
            ]
        )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=max(8, font_size - 2),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, save_path, dpi, transparent)
    return (
        fig,
        axes_array,
        {
            "values": pd.DataFrame(long_rows),
            "model_order": pd.DataFrame({"model": order}),
            "tables": tables,
        },
    )


def _robustness_seed_table(
    robustness_output: Mapping[str, tuple[plt.Figure, plt.Axes, Mapping[str, pd.DataFrame]]],
) -> pd.DataFrame:
    frames = []
    for family, (_, _, tables) in robustness_output.items():
        if "seed_means" not in tables:
            raise ValueError(f"Robustness output for '{family}' has no 'seed_means'.")
        frame = tables["seed_means"].copy()
        if "dataset" not in frame.columns:
            frame["dataset"] = family
        frames.append(frame)
    if not frames:
        raise ValueError("robustness_output is empty.")
    return pd.concat(frames, ignore_index=True)


def plot_family_robustness_panel(
    robustness_output: Mapping[str, tuple[plt.Figure, plt.Axes, Mapping[str, pd.DataFrame]]],
    *,
    metric: str = "fbeta",
    families: Sequence[str] | None = None,
    family_labels: Mapping[str, str] | None = None,
    model_order: Sequence[str] | None = None,
    model_colors: Mapping[str, str] | None = None,
    arms: Sequence[str] = ("untuned", "tuned"),
    highlight_model: str = "PRISM",
    ncols: int = 3,
    xlim: tuple[float, float] = (0.0, 1.0),
    figsize: tuple[float, float] = (14.5, 10.2),
    font_size: float = 10.0,
    title_size: float = 11.0,
    marker_size: float = 4.2,
    capsize: float = 2.0,
    arm_offset: float = 0.16,
    errorbar_alpha: float = 0.48,
    errorbar_linewidth: float = 0.8,
    show_model_bands: bool = True,
    show_fliers: bool = True,
) -> tuple[plt.Figure, np.ndarray, dict[str, pd.DataFrame]]:
    """Combine the per-family robustness plots into one publication panel.

    This function consumes the output of :func:`plot_family_robustness`, so it
    reuses exactly the same dataset-seed summaries and uncertainty intervals.
    Models have one fixed order across panels; only the first column repeats
    their labels.  A model available in a single arm, such as ``PRISM``, is
    centred on its row instead of being offset as if a second point existed.
    """
    if not robustness_output:
        raise ValueError("robustness_output is empty.")
    if ncols < 1:
        raise ValueError("ncols must be at least 1.")
    available_families = list(robustness_output)
    selected_families = list(families or available_families)
    missing = [family for family in selected_families if family not in robustness_output]
    if missing:
        raise KeyError(f"Unknown robustness families: {missing}")
    if model_order is None:
        first_tables = robustness_output[selected_families[0]][2]
        if "model_order" not in first_tables:
            raise ValueError("Robustness output has no shared 'model_order' table.")
        shared_order = first_tables["model_order"]["model"].astype(str).tolist()
    else:
        shared_order = [_canonical_model_name(value) for value in model_order]
    selected_arms = list(dict.fromkeys(map(str, arms)))
    if not selected_arms:
        raise ValueError("At least one arm is required.")
    arm_colors = {"untuned": "#8C8C8C", "tuned": OKABE_ITO["blue"]}
    arm_markers = {"untuned": "o", "tuned": "s"}
    highlight_model = _canonical_model_name(highlight_model)
    highlight_color = DEFAULT_MODEL_COLORS["PRISM"]
    nrows = math.ceil(len(selected_families) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    flat_axes = axes.ravel()
    panel_letters = "abcdefghijklmnopqrstuvwxyz"
    labels = dict(family_labels or {})
    summary_frames: list[pd.DataFrame] = []
    outlier_frames: list[pd.DataFrame] = []
    model_positions = np.arange(len(shared_order), dtype=float)
    offsets = (
        dict(zip(selected_arms, np.linspace(-arm_offset, arm_offset, len(selected_arms))))
        if len(selected_arms) > 1
        else {selected_arms[0]: 0.0}
    )
    for panel_index, (family, axis) in enumerate(zip(selected_families, flat_axes)):
        tables = robustness_output[family][2]
        if "summary" not in tables:
            raise ValueError(f"Robustness output for '{family}' has no 'summary'.")
        family_summary = tables["summary"].copy()
        summary_frames.append(family_summary)
        family_outliers = tables.get("outliers", pd.DataFrame()).copy()
        if not family_outliers.empty:
            outlier_frames.append(family_outliers)
        if show_model_bands:
            for row_index in range(0, len(shared_order), 2):
                axis.axhspan(
                    row_index - 0.5,
                    row_index + 0.5,
                    color=OKABE_ITO["black"],
                    alpha=0.045,
                    linewidth=0,
                    zorder=0,
                )
        arm_presence = {
            arm: set(
                family_summary.loc[
                    family_summary["arm"].astype(str).eq(arm)
                    & np.isfinite(pd.to_numeric(family_summary["mean"], errors="coerce")),
                    "model",
                ].astype(str)
            )
            for arm in selected_arms
        }
        for arm in selected_arms:
            rows = (
                family_summary[family_summary["arm"].astype(str).eq(arm)]
                .set_index("model")
                .reindex(shared_order)
            )
            means = pd.to_numeric(rows["mean"], errors="coerce").to_numpy(float)
            lower = pd.to_numeric(rows["lower"], errors="coerce").to_numpy(float)
            upper = pd.to_numeric(rows["upper"], errors="coerce").to_numpy(float)
            available = np.isfinite(means)
            row_offsets = np.full(len(shared_order), offsets[arm], dtype=float)
            for row_index, model in enumerate(shared_order):
                present_arms = sum((model in arm_presence[value] for value in selected_arms))
                if present_arms == 1:
                    row_offsets[row_index] = 0.0
            for row_index, model in enumerate(shared_order):
                if not available[row_index]:
                    continue
                is_highlighted = model == highlight_model
                colour = highlight_color if is_highlighted else arm_colors.get(arm, "#4D4D4D")
                marker = "D" if is_highlighted else arm_markers.get(arm, "o")
                xerr = np.asarray(
                    [
                        [max(0.0, means[row_index] - lower[row_index])],
                        [max(0.0, upper[row_index] - means[row_index])],
                    ]
                )
                errorbar = axis.errorbar(
                    means[row_index],
                    model_positions[row_index] + row_offsets[row_index],
                    xerr=xerr,
                    fmt=marker,
                    linestyle="none",
                    color=colour,
                    ecolor=colour,
                    markerfacecolor=colour,
                    markersize=marker_size + (1.2 if is_highlighted else 0),
                    capsize=capsize,
                    elinewidth=errorbar_linewidth,
                    markeredgewidth=0.6,
                    zorder=4 if is_highlighted else 3,
                )
                for cap in errorbar[1]:
                    cap.set_alpha(errorbar_alpha)
                for bars in errorbar[2]:
                    bars.set_alpha(errorbar_alpha)
            if show_fliers and (not family_outliers.empty):
                arm_outliers = family_outliers.loc[family_outliers["arm"].astype(str).eq(arm)]
                for row_index, model in enumerate(shared_order):
                    values = arm_outliers.loc[
                        arm_outliers["model"].astype(str).eq(model), "metric_value"
                    ].to_numpy(float)
                    if not len(values):
                        continue
                    color = (
                        highlight_color
                        if model == highlight_model
                        else arm_colors.get(arm, "#4D4D4D")
                    )
                    axis.scatter(
                        values,
                        np.full(
                            len(values),
                            model_positions[row_index] + row_offsets[row_index],
                        ),
                        s=(marker_size + 1.0) ** 2,
                        marker="o",
                        facecolors="none",
                        edgecolors=color,
                        linewidths=0.9,
                        alpha=0.85,
                        zorder=5,
                    )
        title = str(labels.get(family, _dataset_display_title(family)))
        axis.set_title(f"({panel_letters[panel_index]}) {title}", fontsize=title_size, pad=7)
        axis.set_xlim(*xlim)
        axis.set_ylim(len(shared_order) - 0.5, -0.5)
        axis.set_yticks(
            model_positions,
            [
                model.replace("_", "\n")
                if model.startswith("DAGMA_")
                else _model_display_label(model)
                for model in shared_order
            ],
        )
        axis.tick_params(
            axis="both",
            labelsize=max(7.0, font_size - 1),
            labelleft=panel_index % ncols == 0,
        )
        axis.grid(axis="x", alpha=0.2, linewidth=0.7)
        axis.grid(axis="y", visible=False)
        axis.set_axisbelow(True)
        _despine(axis, font_size=font_size)
    for axis in flat_axes[len(selected_families) :]:
        axis.set_visible(False)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=arm_colors["untuned"],
            markersize=5,
            label="Untuned",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            color=arm_colors["tuned"],
            markersize=5,
            label="Tuned",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            color=highlight_color,
            markersize=5.5,
            label="PRISM",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=font_size,
        handlelength=1.0,
        handletextpad=0.5,
        columnspacing=1.5,
    )
    fig.supxlabel("$F_{0.5}$ score", fontsize=font_size + 1, y=0.018)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.07, top=0.94, wspace=0.08, hspace=0.2)
    return (
        fig,
        axes,
        {
            "summary": pd.concat(summary_frames, ignore_index=True),
            "outliers": pd.concat(outlier_frames, ignore_index=True)
            if outlier_frames
            else pd.DataFrame(),
            "model_order": pd.DataFrame({"model": shared_order}),
        },
    )


def plot_robustness_dumbbell(
    robustness_output: Mapping[str, tuple[plt.Figure, plt.Axes, Mapping[str, pd.DataFrame]]],
    *,
    metric: str = "fbeta",
    family_labels: Mapping[str, str] | None = None,
    order_by: str = "tuned",
    highlight_model: str | None = "PRISM",
    model_colors: Mapping[str, str] | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (7.2, 6.5),
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
) -> tuple[plt.Figure, np.ndarray, dict[str, Any]]:
    """Alternative tuned-versus-untuned robustness view using dumbbells."""
    seed_means = _robustness_seed_table(robustness_output)
    means = seed_means.pivot_table(
        index=["dataset", "model"], columns="arm", values="metric_value", aggfunc="mean"
    ).reset_index()
    families = list(dict.fromkeys(seed_means["dataset"].astype(str)))
    figures: dict[str, plt.Figure] = {}
    axes_by_family: dict[str, plt.Axes] = {}
    labels = dict(family_labels or {})
    highlight_model = _canonical_model_name(highlight_model)
    for family in families:
        rows = means[means["dataset"].astype(str).eq(family)].copy()
        if order_by == "delta" and {"tuned", "untuned"}.issubset(rows.columns):
            rows["_order"] = rows["tuned"] - rows["untuned"]
        elif order_by in rows.columns:
            rows["_order"] = rows[order_by]
        else:
            rows["_order"] = rows[[c for c in ("tuned", "untuned") if c in rows]].mean(axis=1)
        rows = rows.sort_values("_order", ascending=False, na_position="last").reset_index(
            drop=True
        )
        row_colours = _colors(rows["model"].astype(str).tolist(), model_colors)
        fig, axis = plt.subplots(figsize=figsize)
        y = np.arange(len(rows))
        if {"tuned", "untuned"}.issubset(rows.columns):
            paired = rows[["tuned", "untuned"]].notna().all(axis=1)
            for position in np.flatnonzero(paired.to_numpy()):
                axis.hlines(
                    y[position],
                    rows.loc[position, "untuned"],
                    rows.loc[position, "tuned"],
                    color=row_colours[position],
                    alpha=0.45,
                    linewidth=1.5,
                    zorder=1,
                )
        if "untuned" in rows:
            axis.scatter(
                rows["untuned"],
                y,
                facecolors="white",
                edgecolors=row_colours,
                marker="o",
                label="untuned",
                zorder=2,
            )
        if "tuned" in rows:
            normal = rows["model"].ne(highlight_model)
            normal_positions = np.flatnonzero(normal.to_numpy())
            axis.scatter(
                rows.loc[normal, "tuned"],
                y[normal_positions],
                color=[row_colours[position] for position in normal_positions],
                marker="s",
                label="tuned",
                zorder=3,
            )
            special = ~normal & rows["tuned"].notna()
            if special.any():
                axis.scatter(
                    rows.loc[special, "tuned"],
                    y[special],
                    color=DEFAULT_MODEL_COLORS["PRISM"],
                    marker="D",
                    label=_model_display_label(highlight_model),
                    zorder=4,
                )
        axis.set_yticks(y, [_model_display_label(model) for model in rows["model"]])
        axis.invert_yaxis()
        axis.set_xlabel(_metric_meta(metric)[0], fontsize=font_size)
        axis.set_ylabel("")
        axis.set_title(_dataset_display_title(labels.get(family, family)), fontsize=title_size)
        axis.grid(axis="x", alpha=0.22)
        if ylim is not None:
            axis.set_xlim(*ylim)
        handles, legend_labels = axis.get_legend_handles_labels()
        unique = dict(zip(legend_labels, handles))
        axis.legend(unique.values(), unique.keys(), fontsize=max(7, font_size - 1))
        _despine(axis, font_size=font_size)
        fig.tight_layout()
        figures[family] = fig
        axes_by_family[family] = axis
    return (
        figures[families[0]],
        np.asarray(list(axes_by_family.values()), dtype=object),
        {"figures": figures, "axes_by_family": axes_by_family, "table": means},
    )


def plot_robustness_delta_forest(
    robustness_output: Mapping[str, tuple[plt.Figure, plt.Axes, Mapping[str, pd.DataFrame]]],
    *,
    metric: str = "fbeta",
    family_labels: Mapping[str, str] | None = None,
    model_colors: Mapping[str, str] | None = None,
    confidence_level: float = 0.95,
    ci_method: str = "bootstrap",
    n_bootstrap: int = 5000,
    random_state: int | None = 42,
    figsize: tuple[float, float] = (7.2, 6.5),
    font_size: float = DEFAULT_FONT_SIZE,
    title_size: float = DEFAULT_TITLE_SIZE,
) -> tuple[plt.Figure, np.ndarray, dict[str, Any]]:
    """Forest plots of paired tuned-minus-untuned robustness differences."""
    seed_means = _robustness_seed_table(robustness_output)
    paired = seed_means.pivot_table(
        index=["dataset", "dataset_seed", "model"],
        columns="arm",
        values="metric_value",
        aggfunc="first",
    ).reset_index()
    if not {"tuned", "untuned"}.issubset(paired.columns):
        raise ValueError("Both tuned and untuned arms are required.")
    paired = paired.dropna(subset=["tuned", "untuned"]).copy()
    paired["delta"] = paired["tuned"] - paired["untuned"]
    if paired.empty:
        raise ValueError("No paired tuned/untuned dataset seeds are available.")
    summary_rows = []
    for (family, model), group in paired.groupby(["dataset", "model"], observed=True):
        summary_rows.append(
            {
                "dataset": family,
                "model": model,
                **_uncertainty_summary(
                    group["delta"],
                    uncertainty="ci",
                    confidence_level=confidence_level,
                    ci_method=ci_method,
                    n_bootstrap=n_bootstrap,
                    random_state=random_state,
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    families = list(dict.fromkeys(seed_means["dataset"].astype(str)))
    labels = dict(family_labels or {})
    figures: dict[str, plt.Figure] = {}
    axes_by_family: dict[str, plt.Axes] = {}
    for family in families:
        rows = (
            summary[summary["dataset"].astype(str).eq(family)]
            .sort_values("mean", ascending=False)
            .reset_index(drop=True)
        )
        if rows.empty:
            continue
        fig, axis = plt.subplots(figsize=figsize)
        y = np.arange(len(rows))
        row_colours = _colors(rows["model"].astype(str).tolist(), model_colors)
        axis.axvline(0, color="0.3", linestyle="--", linewidth=1)
        for position, row in rows.iterrows():
            axis.errorbar(
                row["mean"],
                position,
                xerr=np.asarray([[row["mean"] - row["lower"]], [row["upper"] - row["mean"]]]),
                fmt="o",
                color=row_colours[position],
                capsize=3,
                markersize=5,
            )
        axis.set_yticks(y, [_model_display_label(model) for model in rows["model"]])
        axis.invert_yaxis()
        axis.set_xlabel(f"Δ {_metric_meta(metric)[0]} (tuned - untuned)", fontsize=font_size)
        axis.set_ylabel("")
        axis.set_title(_dataset_display_title(labels.get(family, family)), fontsize=title_size)
        axis.grid(axis="x", alpha=0.22)
        _despine(axis, font_size=font_size)
        fig.tight_layout()
        figures[family] = fig
        axes_by_family[family] = axis
    if not figures:
        raise ValueError("No family has paired tuned/untuned values.")
    first_family = next(iter(figures))
    return (
        figures[first_family],
        np.asarray(list(axes_by_family.values()), dtype=object),
        {
            "figures": figures,
            "axes_by_family": axes_by_family,
            "paired_seed_differences": paired,
            "summary": summary,
        },
    )


for _plot_name, _plot_function in list(globals().items()):
    if _plot_name.startswith("plot_") and callable(_plot_function):
        globals()[_plot_name] = _alpha_controlled_plot(_plot_function)
__all__ = [
    "FinalVisualizationData",
    "OKABE_ITO",
    "OKABE_ITO_EXTENDED",
    "OKABE_ITO_SEQUENTIAL",
    "OKABE_ITO_SEQUENTIAL_R",
    "OKABE_ITO_DIVERGING",
    "DEFAULT_MODEL_ORDER",
    "DEFAULT_MODEL_COLORS",
    "DEFAULT_SCALABILITY_FAMILIES",
    "DEFAULT_ASSUMPTIONS",
    "METRIC_INFO",
    "parse_dataset_name",
    "load_final_visualization_data",
    "plot_family_robustness",
    "build_tuned_stage_table",
    "plot_metric_and_timeouts",
    "plot_metric_boxplot_grid",
    "plot_metric_polar",
    "build_execution_summary_tables",
    "build_ranking_table",
    "plot_rank_variation",
    "plot_rank_scatter_by_dataset",
    "plot_rank_heatmap_by_dataset",
    "plot_noise_heatmap",
    "plot_mechanism_heatmap",
    "plot_assumption_match_heatmap",
    "build_assumption_degradation_data",
    "plot_assumption_degradation_forest",
    "build_model_assumption_table",
    "plot_model_assumption_table",
    "plot_assumption_degradation_heatmap",
    "build_compact_rank_table",
    "build_noise_mechanism_table",
    "build_scalability_compact_table",
    "plot_noise_mechanism_slope",
    "plot_mechanism_change_bars",
    "plot_noise_change_bars",
    "plot_noise_change_by_mechanism",
    "plot_robustness_dumbbell",
    "plot_robustness_delta_forest",
    "load_scalability_data",
    "plot_scalability_f1",
    "plot_scalability_f1_highlighted",
    "plot_scalability_f1_heatmap",
    "plot_scalability_f1_retention",
    "plot_scalability_runtime",
    "plot_dataset_seed_stability",
    "plot_model_seed_stability",
    "plot_confounding_f1",
    "build_confounding_pair_table",
    "plot_confounding_false_positive_rates",
    "plot_varsortability_vs_f1",
]
