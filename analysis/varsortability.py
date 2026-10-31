"""Compute var- and R²-sortability from observational data and a known DAG.

Varsortability is the fraction of directed causal paths whose source has a
smaller marginal variance than their destination.  The calculation in this
module intentionally computes varsortability from the raw observational CSV,
before z-score normalisation.  Each source/target pair is counted once per path
length, following the reference matrix-power definition; ties contribute half.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _path_sortability(
    adjacency: np.ndarray, scores: np.ndarray, *, tie_tolerance: float
) -> dict[str, float | int]:
    """Measure score-order agreement along all directed path lengths."""
    binary = adjacency != 0
    path_mask = binary.copy()
    n_paths = 0
    ordered_paths = 0
    tied_paths = 0
    difference = scores[None, :] - scores[:, None]
    for _ in range(1, binary.shape[0]):
        if np.any(np.diag(path_mask)):
            raise ValueError("adjacency must represent a DAG; a directed cycle was found.")
        n_paths += int(path_mask.sum())
        ordered_paths += int((path_mask & (difference > tie_tolerance)).sum())
        tied_paths += int((path_mask & (np.abs(difference) <= tie_tolerance)).sum())
        path_mask = path_mask.astype(np.int64) @ binary.astype(np.int64) != 0
    score = (ordered_paths + 0.5 * tied_paths) / n_paths if n_paths else np.nan
    return {
        "score": float(score),
        "n_paths": n_paths,
        "n_ordered_paths": ordered_paths,
        "n_tied_paths": tied_paths,
    }


def _r2_coefficients(values: np.ndarray) -> np.ndarray:
    """R² of each variable regressed on all remaining variables."""
    correlation = np.corrcoef(values, rowvar=False)
    try:
        precision = np.linalg.inv(correlation)
        coefficients = 1 - 1 / np.diag(precision)
        if np.all(np.isfinite(coefficients)):
            return coefficients
    except np.linalg.LinAlgError:
        pass
    coefficients = np.empty(values.shape[1], dtype=float)
    for target in range(values.shape[1]):
        predictors = np.arange(values.shape[1]) != target
        design = np.column_stack([np.ones(values.shape[0]), values[:, predictors]])
        response = values[:, target]
        beta, *_ = np.linalg.lstsq(design, response, rcond=None)
        residual = response - design @ beta
        total = response - response.mean()
        denominator = float(total @ total)
        coefficients[target] = (
            1 - float(residual @ residual) / denominator if denominator else np.nan
        )
    return coefficients


def calculate_varsortability(
    observations: pd.DataFrame | np.ndarray,
    adjacency: pd.DataFrame | np.ndarray,
    *,
    ddof: int = 0,
    tie_tolerance: float = 1e-12,
) -> dict[str, float | int]:
    """Calculate varsortability for one dataset.

    ``adjacency[i, j] != 0`` must mean ``i -> j``.  If both inputs are
    DataFrames, the observational columns are aligned to the adjacency labels.
    NaNs are ignored independently in each variable's variance.
    """
    if isinstance(adjacency, pd.DataFrame):
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be square.")
        if list(adjacency.index.astype(str)) != list(adjacency.columns.astype(str)):
            raise ValueError("adjacency row and column labels must have the same order.")
        labels = list(adjacency.columns.astype(str))
        matrix = adjacency.to_numpy(dtype=float)
    else:
        matrix = np.asarray(adjacency, dtype=float)
        labels = None
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be a square two-dimensional matrix.")
    if np.any(np.diag(matrix) != 0):
        raise ValueError("adjacency must not contain self-loops.")
    if isinstance(observations, pd.DataFrame):
        numeric = observations.select_dtypes(include=[np.number])
        if labels is not None:
            missing = [label for label in labels if label not in numeric.columns]
            if missing:
                raise ValueError(f"observations are missing adjacency variables: {missing[:8]}")
            numeric = numeric[labels]
        values = numeric.to_numpy(dtype=float)
    else:
        values = np.asarray(observations, dtype=float)
    if values.ndim != 2 or values.shape[1] != matrix.shape[0]:
        raise ValueError(
            "observations must be samples x variables with one column per adjacency node."
        )
    if values.shape[0] <= ddof:
        raise ValueError("Not enough observations for the selected ddof.")
    variances = np.nanvar(values, axis=0, ddof=ddof)
    if not np.all(np.isfinite(variances)):
        raise ValueError("Every variable must have a finite variance.")
    var_result = _path_sortability(matrix, variances, tie_tolerance=tie_tolerance)
    complete = values[~np.isnan(values).any(axis=1)]
    if len(complete) < 2:
        raise ValueError("At least two complete observations are needed for R²-sortability.")
    r2_coefficients = _r2_coefficients(complete)
    if not np.all(np.isfinite(r2_coefficients)):
        raise ValueError("Every variable must have a finite R² coefficient.")
    r2_result = _path_sortability(matrix, r2_coefficients, tie_tolerance=tie_tolerance)
    return {
        "varsortability": var_result["score"],
        "r2_sortability": r2_result["score"],
        "n_paths": var_result["n_paths"],
        "n_ordered_variance_paths": var_result["n_ordered_paths"],
        "n_tied_variance_paths": var_result["n_tied_paths"],
        "n_ordered_r2_paths": r2_result["n_ordered_paths"],
        "n_tied_r2_paths": r2_result["n_tied_paths"],
        "n_variables": int(values.shape[1]),
        "n_observations": int(values.shape[0]),
    }


def _find_project_root(start: str | Path | None = None) -> Path:
    return Path(__file__).resolve().parents[1]


def _dataset_directory(
    project_root: Path,
    dataset_seed: int | str,
    dataset: str,
    dataset_roots: Iterable[str | Path] | None,
) -> Path:
    roots = (
        [Path(root) for root in dataset_roots]
        if dataset_roots
        else [project_root / f"synthetic_datasets_seed_{dataset_seed}"]
    )
    candidates = [root / dataset for root in roots]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find raw dataset '{dataset}' for seed {dataset_seed}. Checked: {[str(path) for path in candidates]}"
    )


def calculate_varsortability_table(
    datasets: pd.DataFrame,
    *,
    project_root: str | Path | None = None,
    dataset_roots: Iterable[str | Path] | None = None,
    ddof: int = 0,
    tie_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Calculate raw-data varsortability for unique generation instance pairs.

    ``datasets`` needs ``dataset_seed`` and ``dataset`` columns.  The returned
    DataFrame is suitable for merging with benchmark metrics and saving as CSV.
    """
    required = {"dataset_seed", "dataset"}
    missing = required - set(datasets.columns)
    if missing:
        raise ValueError(f"datasets is missing required columns: {sorted(missing)}")
    root = _find_project_root(project_root)
    identities = (
        datasets[["dataset_seed", "dataset"]]
        .drop_duplicates()
        .sort_values(["dataset_seed", "dataset"], kind="stable")
    )
    rows: list[dict[str, object]] = []
    for identity in identities.itertuples(index=False):
        if dataset_roots is None:
            from condition_data import load_sample, sample_path
            sample = load_sample("benchmark", str(identity.dataset), identity.dataset_seed)
            if "X_raw" not in sample:
                raise ValueError("Raw observations are unavailable for this condition; raw varsortability cannot use standardized observations.")
            labels = sample["col_names"].astype(str)
            observations = pd.DataFrame(sample["X_raw"], columns=labels)
            adjacency = pd.DataFrame(sample["true_adj"], index=labels, columns=labels)
            data_path = adjacency_path = sample_path("benchmark", str(identity.dataset))
        else:
            directory = _dataset_directory(
                root, identity.dataset_seed, str(identity.dataset), dataset_roots
            )
            stem = str(identity.dataset)
            data_path = directory / f"{stem}_observational.csv"
            adjacency_path = directory / f"{stem}_adjacency.csv"
            if not data_path.exists() or not adjacency_path.exists():
                raise FileNotFoundError(
                    f"Raw observational or adjacency CSV is missing in {directory}."
                )
            observations = pd.read_csv(data_path)
            adjacency = pd.read_csv(adjacency_path, index_col=0)
        result = calculate_varsortability(
            observations, adjacency, ddof=ddof, tie_tolerance=tie_tolerance
        )
        rows.append(
            {
                "dataset_seed": identity.dataset_seed,
                "dataset": str(identity.dataset),
                **result,
                "data_path": str(data_path),
                "adjacency_path": str(adjacency_path),
            }
        )
    return pd.DataFrame(rows)


def calculate_r2_sort_regress_table(
    datasets: pd.DataFrame,
    *,
    project_root: str | Path | None = None,
    dataset_roots: Iterable[str | Path] | None = None,
) -> pd.DataFrame:
    """Fit CausalDisco's R²-SortnRegress baseline and calculate directed F1.

    The CausalDisco weight convention is ``weights[source, target]``. Every
    finite non-zero weight is therefore binarised as the directed edge
    ``source -> target`` and compared with the benchmark adjacency matrix.
    Scores use the directed precision, recall and F1 definitions.
    """
    try:
        from CausalDisco.baselines import r2_sort_regress
    except ImportError as exc:
        raise ImportError(
            "R²-SortnRegress requires CausalDisco. Install it with `python -m pip install CausalDisco`."
        ) from exc
    required = {"dataset_seed", "dataset"}
    missing = required - set(datasets.columns)
    if missing:
        raise ValueError(f"datasets is missing required columns: {sorted(missing)}")
    root = _find_project_root(project_root)
    identities = (
        datasets[["dataset_seed", "dataset"]]
        .drop_duplicates()
        .sort_values(["dataset_seed", "dataset"], kind="stable")
    )
    rows: list[dict[str, object]] = []
    for identity in identities.itertuples(index=False):
        if dataset_roots is None:
            from condition_data import load_sample, sample_path
            sample = load_sample("benchmark", str(identity.dataset), identity.dataset_seed)
            if "X_raw" not in sample:
                raise ValueError("Raw observations are unavailable for this condition; raw varsortability cannot use standardized observations.")
            labels = sample["col_names"].astype(str)
            observations = pd.DataFrame(sample["X_raw"], columns=labels)
            adjacency = pd.DataFrame(sample["true_adj"], index=labels, columns=labels)
            data_path = adjacency_path = sample_path("benchmark", str(identity.dataset))
        else:
    
            directory = _dataset_directory(
                root, identity.dataset_seed, str(identity.dataset), dataset_roots
            )
            stem = str(identity.dataset)
            data_path = directory / f"{stem}_observational.csv"
            adjacency_path = directory / f"{stem}_adjacency.csv"
            if not data_path.exists() or not adjacency_path.exists():
                raise FileNotFoundError(
                    f"Raw observational or adjacency CSV is missing in {directory}."
                )
            observations = pd.read_csv(data_path)
            adjacency = pd.read_csv(adjacency_path, index_col=0)
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(f"Adjacency matrix is not square: {adjacency_path}")
        if list(adjacency.index.astype(str)) != list(adjacency.columns.astype(str)):
            raise ValueError(f"Adjacency row/column labels do not match in {adjacency_path}.")
        labels = list(adjacency.columns.astype(str))
        missing_variables = [label for label in labels if label not in observations]
        if missing_variables:
            raise ValueError(
                f"Observations are missing adjacency variables in {data_path}: {missing_variables[:8]}"
            )
        values = observations[labels].apply(pd.to_numeric, errors="coerce")
        values = values.dropna(axis=0, how="any").to_numpy(dtype=float)
        if len(values) < 2:
            raise ValueError(f"At least two complete rows are required in {data_path}.")
        weights = np.asarray(r2_sort_regress(values), dtype=float)
        true_edges = adjacency.to_numpy(dtype=float) != 0
        if weights.shape != true_edges.shape:
            raise ValueError(
                f"R²-SortnRegress returned shape {weights.shape}; expected {true_edges.shape} for {identity.dataset}."
            )
        predicted_edges = np.isfinite(weights) & (weights != 0)
        np.fill_diagonal(predicted_edges, False)
        tp = int(np.sum(true_edges & predicted_edges))
        fp = int(np.sum(~true_edges & predicted_edges))
        fn = int(np.sum(true_edges & ~predicted_edges))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "dataset_seed": identity.dataset_seed,
                "dataset": str(identity.dataset),
                "model": "R²-SortnRegress",
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "calculate_varsortability",
    "calculate_varsortability_table",
    "calculate_r2_sort_regress_table",
]
