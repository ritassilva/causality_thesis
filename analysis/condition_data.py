"""Read observation arrays from datasets/ and result tables from results/."""

from pathlib import Path

import pandas as pd
import re

def require_allowed_condition(condition):
    if re.search(r"mod(?:ular)?[_-]?subtype", str(condition), flags=re.I):
        raise ValueError("ModSubtype is excluded from this experiment collection")
    return condition


ROOT = Path(__file__).resolve().parents[1]


STUDY_FOLDERS = {
    "benchmark": "benchmark",
    "benchmark_control": "benchmark/control",
    "evaluation": "tuned",
    "model_selection": "prism/tuning",
    "ablation": "prism/ablation",
    "scalability": "scalability",
    "sachs": "sachs",
    "tcga": "tcga",
}

RESULT_COLUMNS = "dataset model arm dataset_seed model_seed condition_id status timed_out n_nodes n_samples f1 fbeta precision recall SHD runtime_seconds".split()


def result_table(frame):
    """Return per-fit results in their shared column order."""
    out = frame.copy()
    if "timed_out" not in out and "status" in out:
        out["timed_out"] = out["status"].eq("timeout")
    if "stage" in out:
        out["condition_id"] = out["condition_id"].fillna(out["stage"]) if "condition_id" in out else out["stage"]
    return out.reindex(columns=RESULT_COLUMNS)


def load_results(study, *, n_variables=None):
    """Read the result CSVs directly from the requested experiment folder."""
    study = {"tuned": "evaluation", "tuning": "model_selection"}.get(study, study)
    folder = ROOT / "results" / STUDY_FOLDERS[study]
    frames = []
    for path in sorted(folder.glob("*.csv")):
        require_allowed_condition(path.stem)
        frame = pd.read_csv(path, low_memory=False, float_precision="round_trip")
        for condition in frame["dataset"].dropna().unique():
            require_allowed_condition(condition)
        if n_variables is not None:
            frame = frame.loc[frame.n_nodes.eq(n_variables)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError(f"No results for {study!r}")
    return pd.concat(frames, ignore_index=True, sort=False)


class Sample(dict):
    """An in-memory sample with NumPy-style context-manager access."""

    @property
    def files(self):
        return list(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def sample_path(study, condition):
    require_allowed_condition(condition)
    study = {"tuned_untuned": "evaluation", "tuned": "evaluation", "tuning": "model_selection"}.get(study, study)
    return ROOT / "datasets" / STUDY_FOLDERS[study] / f"{condition}.npz"


def load_sample_file(path, dataset_seed=None):
    """Read a standalone sample or select a seed from a dataset bundle."""
    import numpy as np

    path = Path(path)
    require_allowed_condition(path.stem)
    with np.load(path, allow_pickle=False) as package:
        if dataset_seed is None and "X" not in package.files:
            available = package["dataset_seeds"].astype(int).tolist()
            if len(available) != 1:
                raise ValueError(f"Select a dataset_seed in {path.name}: {available}")
            dataset_seed = available[0]
        prefix = f"seed_{int(dataset_seed)}__" if dataset_seed is not None else ""
        keys = [key for key in package.files if key.startswith(prefix) and key != "dataset_seeds"]
        if prefix and not keys and "X" in package.files:
            prefix = ""
            keys = [key for key in package.files if key != "dataset_seeds"]
        if not keys:
            raise KeyError(f"Dataset seed {dataset_seed} not found in {path}")
        values = Sample({key.removeprefix(prefix): package[key] for key in keys})
    return values



def load_sample(study, condition, dataset_seed=None):
    return load_sample_file(sample_path(study, condition), dataset_seed)
