"""Friedman comparisons of methods across paired dataset blocks."""

import re
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import pandas as pd
from scipy.stats import friedmanchisquare

# Define this for the results to appear in a defined order
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
    """
    Return CSV paths from an explicit list, a folder, or both.

    If csv_folder is given, every CSV matching csv_pattern inside that folder
    is used. With recursive=True, subfolders are searched too.
    """
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
    """
    Load the CSV files selected for the Friedman test.

    A _csv_path column is added so repeated dataset names in different files
    can still be treated as different paired test blocks.
    """
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

    Example:
        10nodes_DenseER_obsconf0_latconf0_Gaussian_Linear
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
    """
    Add dataset attributes used by filters when they are missing from the CSV.
    """
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
    Filter results before running the Friedman test.

    Supported filter keys:
        n_nodes, n_samples, graph_type, noise, mechanism, obs_conf, lat_conf,
        normalized, status, timed_out

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
    """Choose sensible repeated-experiment columns for common result files."""
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
        "Could not infer the repeated experiment setting for the Friedman test. "
        "Use result CSVs that include a dataset column or common setup columns "
        "such as n_nodes and n_samples."
    )


def run_friedman_test(
    metric: str,
    csv_folder: Optional[Union[str, Path]] = None,
    csv_files: Optional[Iterable[Union[str, Path]]] = None,
    csv_pattern: str = "*.csv",
    recursive: bool = True,
    output_path: Optional[Union[str, Path]] = None,
    result_df_name: Optional[str] = None,
    models: Optional[list[str]] = None,
    filters: Optional[dict[str, Any]] = None,
    higher_is_better: Optional[bool] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run a Friedman test for one metric across models.

    Parameters
    ----------
    metric:
        Metric column to test, for example "f1", "SHD", or "runtime_seconds".
    csv_folder:
        Optional folder/repo containing CSVs to include.
    csv_files:
        Optional explicit list of CSV paths to include.
    csv_pattern:
        File pattern used inside csv_folder. Defaults to all CSV files.
    recursive:
        If True, search csv_folder and all subfolders.
    output_path:
        Optional path where the result CSV should be saved.
    result_df_name:
        Optional variable name to create in globals(), useful in notebooks or
        interactive runs.
    models:
        Optional list of models to compare. If None, all models in the CSVs are used.
    filters:
        Optional dataset/result filters. Examples:
        {"n_nodes": [10], "noise": ["Gaussian"], "mechanism": ["Linear"]}.
    higher_is_better:
        Controls ranking direction. If None, uses DEFAULT_HIGHER_IS_BETTER.
    alpha:
        Significance threshold.

    Returns
    -------
    pd.DataFrame
        One row per model with the Friedman statistic, p-value, mean rank, and
        summary statistics for the selected metric.
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

    missing_blocks = [col for col in pairing_cols if col not in df.columns]
    if missing_blocks:
        raise ValueError(f"Repeated experiment column(s) not found: {missing_blocks}")

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

    paired = pivot[models].dropna()

    if len(models) < 3:
        raise ValueError("Friedman test needs at least three models.")
    if len(paired) < 2:
        raise ValueError(
            "Friedman test needs at least two complete paired observations. "
            f"Only found {len(paired)} after aligning repeated experiment settings."
        )

    statistic, p_value = friedmanchisquare(
        *[paired[model].to_numpy(dtype=float) for model in models]
    )

    if higher_is_better is None:
        higher_is_better = DEFAULT_HIGHER_IS_BETTER.get(metric, True)

    ranks = paired.rank(axis=1, ascending=not higher_is_better)
    mean_ranks = ranks.mean().sort_values()

    summary = (
        paired.agg(["mean", "std", "median", "min", "max", "count"])
        .T.rename_axis("model")
        .reset_index()
    )

    result_df = summary.merge(
        mean_ranks.rename("mean_rank").reset_index(),
        on="model",
        how="left",
    ).sort_values("mean_rank")

    result_df.insert(0, "metric", metric)
    result_df["friedman_statistic"] = float(statistic)
    result_df["p_value"] = float(p_value)
    result_df["significant"] = bool(p_value < alpha)
    result_df["alpha"] = alpha
    result_df["n_blocks"] = len(paired)
    result_df["n_models"] = len(models)
    result_df["n_rows_after_filters"] = len(df)
    result_df["n_csv_files"] = len(selected_csv_files)
    result_df["filters"] = str(filters or {})
    result_df["higher_is_better"] = higher_is_better

    numeric_cols = [
        "mean",
        "std",
        "median",
        "min",
        "max",
        "mean_rank",
        "friedman_statistic",
    ]
    result_df[numeric_cols] = result_df[numeric_cols].round(6)

    if result_df_name:
        globals()[result_df_name] = result_df

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)
        print(f"Saved Friedman results to: {output_path}")

    return result_df


