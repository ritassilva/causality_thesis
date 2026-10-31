"""Paired Wilcoxon tests and Holm-corrected competition ranks."""

import re
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def paired_wilcoxon(x, y=None, **kwargs):
    """Use SciPy's automatic rule, including its normal fallback for zero differences."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Exact p-value calculation does not work if there are zeros.*", category=UserWarning)
        warnings.filterwarnings("ignore", message="Sample size too small for normal approximation.*", category=UserWarning)
        return wilcoxon(x, y, **kwargs)

DEFAULT_HIGHER_IS_BETTER = {
    "f1": True,
    "precision": True,
    "recall": True,
    "fbeta": True,
    "auprc": True,
    "SHD": False,
    "runtime_seconds": False,
}


def find_csv_files(
    csv_folder: Optional[Union[str, Path]] = None,
    csv_files: Optional[Iterable[Union[str, Path]]] = None,
    csv_pattern: str = "*.csv",
    recursive: bool = True,
) -> list[Path]:
    """Return CSV paths from an explicit list, a folder, or both."""
    paths = []

    if csv_files is not None:
        paths.extend(Path(csv_file) for csv_file in csv_files)

    if csv_folder is not None:
        folder = Path(csv_folder)
        if not folder.exists():
            raise FileNotFoundError(f"CSV folder not found: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"csv_folder is not a folder: {folder}")

        finder = folder.rglob if recursive else folder.glob
        paths.extend(finder(csv_pattern))

    paths = sorted(set(paths))
    if not paths:
        raise ValueError("No CSV files found. Set csv_folder, csv_files, or adjust csv_pattern.")

    return paths


def load_csv_results(csv_files: Iterable[Union[str, Path]]) -> pd.DataFrame:
    """Load selected CSV files and keep the source path for pairing rows."""
    frames = []

    for csv_file in csv_files:
        path = Path(csv_file)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        df = pd.read_csv(path)
        df["_csv_path"] = str(path)
        frames.append(df)

    if not frames:
        raise ValueError("csv_files is empty. Add at least one CSV path.")

    return pd.concat(frames, ignore_index=True)


def parse_dataset_attributes(dataset_name: str) -> dict[str, Any]:
    """
    Parse synthetic dataset names into filterable attributes.

    Expected format:
        {n_nodes}nodes_{graph_type}_obsconf{obs}_latconf{lat}_{noise}_{mechanism}
    """
    parts = str(dataset_name).split("_")
    if len(parts) < 6:
        return {
            "n_nodes_parsed": None,
            "graph_type": None,
            "obs_conf": None,
            "lat_conf": None,
            "noise": None,
            "mechanism": None,
        }

    return {
        "n_nodes_parsed": _first_int(parts[0]),
        "graph_type": parts[1],
        "obs_conf": _first_int(parts[2]),
        "lat_conf": _first_int(parts[3]),
        "noise": "_".join(parts[4:-1]),
        "mechanism": parts[-1],
    }


def _first_int(value: Any) -> Optional[int]:
    """Return the first integer in a string, or None when there is none."""
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def add_filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add parsed dataset attributes when they are missing from the CSV."""
    df = df.copy()

    if "dataset" not in df.columns:
        return df

    parsed = df["dataset"].apply(parse_dataset_attributes).apply(pd.Series)
    for col in parsed.columns:
        if col not in df.columns:
            df[col] = parsed[col]
        else:
            df[col] = df[col].combine_first(parsed[col])

    return df


def _as_list(value: Any) -> Optional[list[Any]]:
    """Allow filters to be written as either one value or a list of values."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def apply_filters(df: pd.DataFrame, filters: Optional[dict[str, Any]]) -> pd.DataFrame:
    """
    Filter results before running Wilcoxon tests.

    Supported filter keys include n_nodes, n_samples, graph_type, noise,
    mechanism, obs_conf, lat_conf, normalized, status, and timed_out.
    I can also filter any exact column present in the dataframe.
    """
    if not filters:
        return df

    df = add_filter_columns(df)
    filtered = df.copy()

    for key, value in filters.items():
        values = _as_list(value)
        if values is None:
            continue

        if key == "n_nodes" and "n_nodes" not in filtered.columns:
            col = "n_nodes_parsed"
        else:
            col = key

        if col not in filtered.columns:
            raise ValueError(
                f"Cannot filter by '{key}' because column '{col}' is not available. "
                f"Available columns: {list(filtered.columns)}"
            )

        filtered = filtered[filtered[col].isin(values)]

    return filtered


def _default_pairing_columns(df: pd.DataFrame) -> list[str]:
    """Choose repeated-experiment columns used to align model scores."""
    if "dataset" in df.columns:
        # Deucalion stores several independently sampled versions of each
        # dataset configuration in one CSV.  Keep those dataset seeds as
        # separate statistical blocks while averaging repeated model seeds
        # within a block through pivot_table below.
        if "dataset_seed" in df.columns:
            return ["_csv_path", "dataset_seed", "dataset"]
        return ["_csv_path", "dataset"]

    fallback_cols = [
        col
        for col in ["n_nodes", "n_samples", "graph_type", "noise", "mechanism"]
        if col in df.columns
    ]
    if fallback_cols:
        return fallback_cols

    raise ValueError(
        "Could not infer the repeated experiment setting. Use result CSVs "
        "that include a dataset column or common setup columns such as "
        "n_nodes and n_samples."
    )


def holm_correct(p_values: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """
    Apply Holm-Bonferroni correction.

    Returns adjusted p-values and reject decisions in the original order.
    """
    n_tests = len(p_values)
    if n_tests == 0:
        return [], []

    order = np.argsort(p_values)
    sorted_p = np.array(p_values, dtype=float)[order]
    sorted_adjusted = np.empty(n_tests, dtype=float)

    running_max = 0.0
    for i, p_value in enumerate(sorted_p):
        adjusted = min((n_tests - i) * p_value, 1.0)
        running_max = max(running_max, adjusted)
        sorted_adjusted[i] = running_max

    sorted_reject = sorted_adjusted < alpha

    adjusted_p = np.empty(n_tests, dtype=float)
    reject = np.empty(n_tests, dtype=bool)
    adjusted_p[order] = sorted_adjusted
    reject[order] = sorted_reject

    return adjusted_p.tolist(), reject.tolist()


def build_wilcoxon_rank(
    pairwise_results: pd.DataFrame,
    models: Optional[list[str]] = None,
    model_means: Optional[dict[str, float]] = None,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """
    Build a final model ranking from pairwise Wilcoxon results.

    A model gets one win only when it has the better mean and the Holm-corrected
    Wilcoxon test is significant. Non-significant comparisons are counted as
    ties, because the test does not provide enough evidence that one model beats
    the other. Models in the same rank do not differ significantly from every
    other model in that rank.
    """
    if pairwise_results.empty:
        raise ValueError("pairwise_results is empty.")

    if models is None:
        models = sorted(set(pairwise_results["model_1"]).union(pairwise_results["model_2"]))

    model_means = model_means or {}
    scores = {
        model: {
            "model": model,
            "significant_wins": 0,
            "significant_losses": 0,
            "non_significant_ties": 0,
            "comparisons": 0,
            "mean_metric": model_means.get(model, np.nan),
        }
        for model in models
    }

    for _, row in pairwise_results.iterrows():
        model_1 = row["model_1"]
        model_2 = row["model_2"]
        if model_1 not in scores or model_2 not in scores:
            continue

        scores[model_1]["comparisons"] += 1
        scores[model_2]["comparisons"] += 1

        if bool(row["significant_holm"]):
            winner = row["better_model_by_mean"]
            loser = model_2 if winner == model_1 else model_1
            scores[winner]["significant_wins"] += 1
            scores[loser]["significant_losses"] += 1
        else:
            scores[model_1]["non_significant_ties"] += 1
            scores[model_2]["non_significant_ties"] += 1

    rank_df = pd.DataFrame(scores.values())
    rank_df["net_wins"] = rank_df["significant_wins"] - rank_df["significant_losses"]
    rank_df["win_rate"] = np.where(
        rank_df["comparisons"] > 0,
        rank_df["significant_wins"] / rank_df["comparisons"],
        np.nan,
    )

    rank_df = rank_df.sort_values(
        [
            "net_wins",
            "significant_wins",
            "significant_losses",
            "mean_metric",
            "model",
        ],
        ascending=[False, False, True, not higher_is_better, True],
    ).reset_index(drop=True)

    significant_pairs = {
        frozenset((row["model_1"], row["model_2"])): bool(row["significant_holm"])
        for _, row in pairwise_results.iterrows()
    }

    rank_values = []
    tier_values = []
    current_group = []
    current_tier = 0
    current_rank = 1

    for idx, model in enumerate(rank_df["model"]):
        tied_with_current_group = bool(current_group) and all(
            significant_pairs.get(frozenset((model, other_model))) is False
            for other_model in current_group
        )

        if not current_group or not tied_with_current_group:
            current_group = [model]
            current_tier += 1
            current_rank = idx + 1
        else:
            current_group.append(model)

        rank_values.append(current_rank)
        tier_values.append(current_tier)

    rank_df.insert(0, "tier", tier_values)
    rank_df.insert(0, "rank", rank_values)

    numeric_cols = ["mean_metric", "win_rate"]
    rank_df[numeric_cols] = rank_df[numeric_cols].round(6)
    return rank_df


def run_pairwise_wilcoxon_tests(
    metric: str,
    csv_folder: Optional[Union[str, Path]] = None,
    csv_files: Optional[Iterable[Union[str, Path]]] = None,
    csv_pattern: str = "*.csv",
    recursive: bool = True,
    output_path: Optional[Union[str, Path]] = None,
    result_df_name: Optional[str] = None,
    rank_output_path: Optional[Union[str, Path]] = None,
    rank_df_name: Optional[str] = None,
    models: Optional[list[str]] = None,
    filters: Optional[dict[str, Any]] = None,
    higher_is_better: Optional[bool] = None,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run Wilcoxon signed-rank tests between all model pairs.

    Holm correction is applied across all model-pair p-values.
    """
    selected_csv_files = find_csv_files(
        csv_folder=csv_folder,
        csv_files=csv_files,
        csv_pattern=csv_pattern,
        recursive=recursive,
    )

    df = add_filter_columns(load_csv_results(selected_csv_files))
    df = apply_filters(df, filters)

    required_cols = {"model", metric}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    pairing_cols = _default_pairing_columns(df)

    df = df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric, "model", *pairing_cols])

    if models is not None:
        df = df[df["model"].isin(models)]

    if df.empty:
        raise ValueError("No usable rows remain after filtering and numeric conversion.")

    pivot = df.pivot_table(
        index=pairing_cols,
        columns="model",
        values=metric,
        aggfunc="mean",
    )

    if models is None:
        models = sorted(pivot.columns.tolist())
    else:
        missing_models = [model for model in models if model not in pivot.columns]
        if missing_models:
            raise ValueError(f"Model(s) not found after filtering: {missing_models}")

    if len(models) < 2:
        raise ValueError("Pairwise Wilcoxon tests need at least two models.")

    if higher_is_better is None:
        higher_is_better = DEFAULT_HIGHER_IS_BETTER.get(metric, True)

    rows = []
    for model_1, model_2 in combinations(models, 2):
        paired = pivot[[model_1, model_2]].dropna()
        if len(paired) < 2:
            continue

        values_1 = paired[model_1].to_numpy(dtype=float)
        values_2 = paired[model_2].to_numpy(dtype=float)
        differences = values_1 - values_2

        if np.allclose(differences, 0):
            statistic = 0.0
            p_value = 1.0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                statistic, p_value = wilcoxon(
                    values_1,
                    values_2,
                    alternative=alternative,
                    zero_method="wilcox",
                )

        mean_1 = float(np.mean(values_1))
        mean_2 = float(np.mean(values_2))
        mean_difference = mean_1 - mean_2

        if higher_is_better:
            better_model = model_1 if mean_1 > mean_2 else model_2
        else:
            better_model = model_1 if mean_1 < mean_2 else model_2

        rows.append(
            {
                "metric": metric,
                "model_1": model_1,
                "model_2": model_2,
                "mean_1": mean_1,
                "mean_2": mean_2,
                "mean_difference": mean_difference,
                "better_model_by_mean": better_model,
                "wilcoxon_statistic": float(statistic),
                "raw_p_value": float(p_value),
                "n_paired": len(paired),
                "alternative": alternative,
                "higher_is_better": higher_is_better,
                "n_csv_files": len(selected_csv_files),
                "n_rows_after_filters": len(df),
                "filters": str(filters or {}),
            }
        )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        raise ValueError("No pairwise Wilcoxon tests could be run.")

    adjusted_p, reject = holm_correct(result_df["raw_p_value"].tolist(), alpha=alpha)
    result_df["holm_p_value"] = adjusted_p
    result_df["significant_holm"] = reject
    result_df["alpha"] = alpha

    result_df["significance"] = np.select(
        [
            result_df["holm_p_value"] < 0.001,
            result_df["holm_p_value"] < 0.01,
            result_df["holm_p_value"] < 0.05,
        ],
        ["***", "**", "*"],
        default="ns",
    )

    numeric_cols = [
        "mean_1",
        "mean_2",
        "mean_difference",
        "wilcoxon_statistic",
    ]
    result_df[numeric_cols] = result_df[numeric_cols].round(6)
    result_df = result_df.sort_values(["holm_p_value", "raw_p_value"]).reset_index(drop=True)

    if result_df_name:
        globals()[result_df_name] = result_df

    model_means = pivot[models].mean(axis=0).to_dict()
    rank_df = build_wilcoxon_rank(
        result_df,
        models=models,
        model_means={model: float(mean) for model, mean in model_means.items()},
        higher_is_better=higher_is_better,
    )

    if rank_df_name:
        globals()[rank_df_name] = rank_df

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)
        print(f"Saved Wilcoxon results to: {output_path}")

    if rank_output_path:
        rank_output_path = Path(rank_output_path)
        rank_output_path.parent.mkdir(parents=True, exist_ok=True)
        rank_df.to_csv(rank_output_path, index=False)
        print(f"Saved Wilcoxon rank to: {rank_output_path}")

    return result_df


