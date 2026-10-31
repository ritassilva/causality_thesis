"""Utilities for the TCGA glioma analysis notebook.

The module deliberately loads results into memory and does not write derived
tables.  It combines the three complete-cohort datasets in ``arms_analysis``
with the nine cancer-type datasets in ``arms_analysis_subtypes``.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

MODEL_ALIASES = {
    "PC": "PC__tuned", "GES": "GES__tuned", "GRASP": "GRaSP__tuned",
    "GOLEM": "GOLEM__tuned", "AVICI": "AVICI__tuned", "PRISM": "PRISM__tuned",
}

MODELS = (
    "PC__tuned",
    "GES__tuned",
    "GRaSP__tuned",
    "GOLEM__tuned",
    "AVICI__tuned",
    "PRISM__tuned",
)
BASE_DATASETS = ("tcga_mut35", "tcga_expr35", "tcga_mut_expr70")
SUBTYPES = ("astro", "gbm", "oligo")
DATASETS = BASE_DATASETS + tuple(
    (f"{dataset}__{subtype}" for dataset in BASE_DATASETS for subtype in SUBTYPES)
)
MUTATION_SUFFIX = "_mut"
EXPRESSION_SUFFIX = "_expr_log2tpm"


def find_project_root(start=None):
    return Path(__file__).resolve().parents[1]


def base_dataset(dataset: str) -> str:
    return dataset.split("__", 1)[0]


def default_view(dataset: str) -> str:
    """Remove forbidden/unresolved cross-type edges without reversing arrows."""
    return "pruned" if base_dataset(dataset) == "tcga_mut_expr70" else "raw"


def source_arm_for(dataset: str, model: str) -> str:
    """Use PC background knowledge where mutation and expression coexist.

    The analysis keeps its existing PC key across all datasets; source_arm
    records the actual saved run used, including the __bk suffix.
    """
    if model == "PC__tuned" and base_dataset(dataset) == "tcga_mut_expr70":
        return "PC__tuned__bk"
    return model


def gene_of(label: str) -> str:
    if label.endswith(MUTATION_SUFFIX):
        return label[: -len(MUTATION_SUFFIX)]
    if label.endswith(EXPRESSION_SUFFIX):
        return label[: -len(EXPRESSION_SUFFIX)]
    raise ValueError(f"Unknown TCGA variable label: {label!r}")


def variable_kind(label: str) -> str:
    if label.endswith(MUTATION_SUFFIX):
        return "mut"
    if label.endswith(EXPRESSION_SUFFIX):
        return "rna"
    raise ValueError(f"Unknown TCGA variable label: {label!r}")


def resolve_models(models: Iterable[str] | None = None) -> tuple[str, ...]:
    resolved = tuple((MODEL_ALIASES.get(model.upper(), model) for model in models or MODELS))
    unknown = set(resolved) - set(MODELS)
    if unknown:
        raise ValueError(f"Unknown model(s): {sorted(unknown)}; expected {list(MODELS)}")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Repeated model after resolving aliases: {resolved}")
    return resolved


def _analysis_dir(tcga_root: Path, dataset: str) -> Path:
    return tcga_root / ("arms_analysis_subtypes" if "__" in dataset else "arms_analysis")


def _load_labels() -> dict[str, list[str]]:
    from condition_data import load_sample

    labels = {}
    for base in BASE_DATASETS:
        ordered = []
        for subtype in SUBTYPES:
            dataset = f"{base}__{subtype}"
            with load_sample("tcga", dataset) as observations:
                labels[dataset] = list(map(str, observations["col_names"]))
            for label in labels[dataset]:
                if label not in ordered:
                    ordered.append(label)
        labels[base] = ordered
    return labels


@dataclass
class GliomaBundle:
    project_root: Path
    tcga_root: Path
    models: tuple[str, ...]
    datasets: tuple[str, ...]
    labels: dict[str, list[str]]
    occurrences: dict[str, dict[str, pd.DataFrame]]
    run_status: pd.DataFrame
    availability: pd.DataFrame

    def result(self, dataset: str, model: str) -> pd.DataFrame:
        return self.occurrences[dataset][MODEL_ALIASES.get(model.upper(), model)]


def edge_occurrences(analysis: Path) -> pd.DataFrame:
    """Aggregate individual saved graph edges using the measured fit counts."""
    from condition_data import load_results

    metrics = load_results("tcga")
    metrics["arm"] = metrics.model + "__" + metrics.arm
    knowledge = metrics.condition_id.astype(str).str.endswith("__background_knowledge")
    metrics.loc[knowledge, "arm"] += "__bk"
    edges = pd.read_csv(analysis / "edges_by_run.csv")
    pairs = np.sort(edges[["source", "target"]].to_numpy(str), axis=1)
    edges["left"], edges["right"] = pairs[:, 0], pairs[:, 1]
    keys = ["dataset", "arm", "view"]
    pair_keys = keys + ["left", "right"]
    skeleton = (edges.drop_duplicates(pair_keys + ["model_seed"])
                .groupby(pair_keys).size().rename("skeleton_count").reset_index())
    unoriented = (edges.loc[~edges.orientation.eq("directed")]
                  .drop_duplicates(pair_keys + ["model_seed"])
                  .groupby(pair_keys).size().rename("unoriented_count").reset_index())
    directed = (edges.loc[edges.orientation.eq("directed")]
                .groupby(keys + ["source", "target"]).size()
                .rename("directed_count").reset_index())
    rates = pd.concat([
        skeleton.assign(source=skeleton.left, target=skeleton.right),
        skeleton.assign(source=skeleton.right, target=skeleton.left),
    ], ignore_index=True)
    rates = (rates.merge(unoriented, on=pair_keys, how="left")
             .merge(directed, on=keys + ["source", "target"], how="left")
             .fillna({"unoriented_count": 0, "directed_count": 0}))
    denominators = metrics.groupby(["dataset", "arm"]).size().rename("n_runs").reset_index()
    finished = (metrics.loc[metrics.status.eq("finished")].groupby(["dataset", "arm"])
                .size().rename("n_graphs").reset_index())
    rates = (rates.merge(denominators, on=["dataset", "arm"])
             .merge(finished, on=["dataset", "arm"], how="left"))
    for kind in ("directed", "skeleton", "unoriented"):
        rates[f"{kind}_rate"] = rates[f"{kind}_count"] / rates.n_runs
    return rates[keys + ["source", "target", "n_runs", "n_graphs",
                         "directed_rate", "skeleton_rate", "unoriented_rate"]]


def load_glioma_results(
    project_root: Path | str | None = None, models: Iterable[str] | None = None
) -> GliomaBundle:
    """Load the selected tuned models for all 12 datasets without displaying them."""
    root = find_project_root(project_root)
    tcga_root = root / "results/tcga"
    selected = resolve_models(models)
    labels = _load_labels()
    occurrences: dict[str, dict[str, pd.DataFrame]] = {}
    status_parts = []
    availability_rows = []
    for analysis_name, datasets in (
        ("arms_analysis", BASE_DATASETS),
        ("arms_analysis_subtypes", tuple((d for d in DATASETS if "__" in d))),
    ):
        analysis = tcga_root / analysis_name
        rates = edge_occurrences(analysis)
        from condition_data import load_results
        metrics = load_results("tcga")
        names = metrics["model"]
        metrics["arm"] = names + "__" + metrics["arm"]
        knowledge = metrics["condition_id"].astype(str).str.endswith("__background_knowledge")
        metrics.loc[knowledge & metrics.model.eq("PC"), "arm"] += "__bk"
        for dataset in datasets:
            view = default_view(dataset)
            occurrences[dataset] = {}
            for model in selected:
                source_arm = source_arm_for(dataset, model)
                frame = rates[
                    (rates["dataset"] == dataset)
                    & (rates["arm"] == source_arm)
                    & (rates["view"] == view)
                ].copy()
                frame["source_arm"] = source_arm
                frame["arm"] = model
                frame.attrs.update(dataset=dataset, model=model, view=view, source_arm=source_arm)
                occurrences[dataset][model] = frame.reset_index(drop=True)
                runs = metrics[
                    (metrics["dataset"] == dataset) & (metrics["arm"] == source_arm)
                ].copy()
                runs["source_arm"] = source_arm
                runs["arm"] = model
                status_parts.append(runs)
                finished = int((runs["status"] == "finished").sum())
                availability_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "source_arm": source_arm,
                        "view": view,
                        "n_planned_runs": int(len(runs)),
                        "n_finished_runs": finished,
                        "n_timeout_runs": int((runs["status"] == "timeout").sum()),
                        "available": finished > 0,
                        "n_occurrence_rows": int(len(frame)),
                    }
                )
    status = pd.concat(status_parts, ignore_index=True, sort=False)
    availability = pd.DataFrame(availability_rows)
    return GliomaBundle(
        root, tcga_root, selected, DATASETS, labels, occurrences, status, availability
    )


def _rates_for_pair(frame: pd.DataFrame, a: str, b: str) -> tuple[float, float, float]:
    """Return forward, backward and skeleton occurrence for canonical a < b."""
    ab = frame[(frame["source"] == a) & (frame["target"] == b)]
    ba = frame[(frame["source"] == b) & (frame["target"] == a)]
    forward = float(ab["directed_rate"].max()) if not ab.empty else 0.0
    backward = float(ba["directed_rate"].max()) if not ba.empty else 0.0
    skeleton_values = pd.concat(
        [
            ab.get("skeleton_rate", pd.Series(dtype=float)),
            ba.get("skeleton_rate", pd.Series(dtype=float)),
        ]
    )
    skeleton = float(skeleton_values.max()) if not skeleton_values.empty else 0.0
    unoriented_values = pd.concat(
        [
            ab.get("unoriented_rate", pd.Series(dtype=float)),
            ba.get("unoriented_rate", pd.Series(dtype=float)),
        ]
    )
    unoriented = float(unoriented_values.max()) if not unoriented_values.empty else 0.0
    return (forward + 0.5 * unoriented, backward + 0.5 * unoriented, skeleton)


def edge_set(
    frame: pd.DataFrame, min_rate: float = 0.8, mode: str = "directed"
) -> set[tuple[str, str] | frozenset[str]]:
    """Convert an occurrence frame to directed edges or skeleton adjacencies."""
    if mode not in {"directed", "skeleton"}:
        raise ValueError("mode must be 'directed' or 'skeleton'")
    if not 0 <= min_rate <= 1:
        raise ValueError("min_rate must be between 0 and 1")
    if mode == "directed":
        chosen = frame[frame["directed_rate"] >= min_rate]
        return set(zip(chosen["source"], chosen["target"]))
    chosen = frame[frame["skeleton_rate"] >= min_rate]
    return {frozenset((row.source, row.target)) for row in chosen.itertuples(index=False)}


def common_edges(
    bundle: GliomaBundle,
    dataset: str,
    models: Iterable[str] | None = None,
    min_rate: float = 0.8,
    min_models: int | None = None,
    mode: str = "directed",
    allow_unavailable: bool = False,
) -> pd.DataFrame:
    """Edges shared by a configurable number of models.

    ``min_models=None`` means intersection of every requested model.  Set it to
    1 for the union, or any intermediate integer for k-of-n agreement.
    """
    selected = resolve_models(models if models is not None else bundle.models)
    required = len(selected) if min_models is None else int(min_models)
    if not 1 <= required <= len(selected):
        raise ValueError("min_models must be between 1 and the number of models")
    if not allow_unavailable:
        available = bundle.availability.set_index(["dataset", "model"])["available"]
        missing = [m for m in selected if not bool(available.get((dataset, m), False))]
        if missing:
            raise ValueError(
                f"No finished graph for {dataset}: {missing}. Pass allow_unavailable=True only if treating them as empty is intentional."
            )
    supports: dict[object, list[str]] = {}
    for model in selected:
        for edge in edge_set(bundle.result(dataset, model), min_rate, mode):
            supports.setdefault(edge, []).append(model)
    rows = []
    for edge, found in supports.items():
        if len(found) < required:
            continue
        if mode == "directed":
            source, target = edge
        else:
            source, target = sorted(edge)
        rows.append(
            {
                "dataset": dataset,
                "source": source,
                "target": target,
                "mode": mode,
                "n_models": len(found),
                "models": tuple(found),
                "min_rate": min_rate,
            }
        )
    columns = ["dataset", "source", "target", "mode", "n_models", "models", "min_rate"]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["n_models", "source", "target"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def same_gene_mut_rna_edges(
    frame: pd.DataFrame, min_rate: float = 0.8, direction: str = "mut_to_rna"
) -> pd.DataFrame:
    """Find same-gene mutation/RNA links in a 70-feature result frame."""
    if direction not in {"mut_to_rna", "rna_to_mut", "either"}:
        raise ValueError("direction must be mut_to_rna, rna_to_mut or either")
    rows = []
    for row in frame.itertuples(index=False):
        try:
            source_kind, target_kind = (
                variable_kind(row.source),
                variable_kind(row.target),
            )
        except ValueError:
            continue
        if source_kind == target_kind or gene_of(row.source) != gene_of(row.target):
            continue
        edge_direction = f"{source_kind}_to_{target_kind}"
        if direction != "either" and edge_direction != direction:
            continue
        if float(row.directed_rate) < min_rate:
            continue
        record = row._asdict()
        record.update(gene=gene_of(row.source), edge_direction=edge_direction)
        rows.append(record)
    return pd.DataFrame(rows)


def all_same_gene_links(
    bundle: GliomaBundle,
    models: Iterable[str] | None = None,
    min_rate: float = 0.8,
    direction: str = "mut_to_rna",
) -> pd.DataFrame:
    selected = resolve_models(models if models is not None else bundle.models)
    parts = []
    for dataset in bundle.datasets:
        if base_dataset(dataset) != "tcga_mut_expr70":
            continue
        for model in selected:
            found = same_gene_mut_rna_edges(bundle.result(dataset, model), min_rate, direction)
            if not found.empty:
                parts.append(found.assign(dataset=dataset, model=model))
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def _soft_votes(bundle: GliomaBundle, dataset: str, models: Sequence[str]):
    labels = sorted(bundle.labels[dataset])
    pairs = list(combinations(labels, 2))
    votes = np.empty((len(pairs), len(models), 3), dtype=float)
    lookups = []
    for model in models:
        frame = bundle.result(dataset, model)
        if frame.empty:
            lookups.append({})
            continue
        compact = frame.groupby(["source", "target"], as_index=False)[
            ["directed_rate", "skeleton_rate", "unoriented_rate"]
        ].max()
        lookups.append(
            {
                (row.source, row.target): (
                    float(row.directed_rate),
                    float(row.skeleton_rate),
                    float(row.unoriented_rate),
                )
                for row in compact.itertuples(index=False)
            }
        )
    for task, (a, b) in enumerate(pairs):
        for worker, lookup in enumerate(lookups):
            ab = lookup.get((a, b), (0.0, 0.0, 0.0))
            ba = lookup.get((b, a), (0.0, 0.0, 0.0))
            unoriented = max(ab[2], ba[2])
            forward = ab[0] + 0.5 * unoriented
            backward = ba[0] + 0.5 * unoriented
            skeleton = max(ab[1], ba[1])
            raw = np.array([forward, backward, max(0.0, 1.0 - skeleton)])
            votes[task, worker] = raw / raw.sum() if raw.sum() else (0.0, 0.0, 1.0)
    return (pairs, votes)


def _em_onecoin(votes: np.ndarray, max_iter: int = 300, tol: float = 1e-09):
    """One-coin Dawid-Skene EM with soft votes from repeated model runs."""
    n_tasks, n_workers, n_states = votes.shape
    prior = np.clip(votes.mean(axis=(0, 1)), 1e-06, None)
    prior /= prior.sum()
    skills = np.full(n_workers, 0.7)
    posterior = np.full((n_tasks, n_states), 1 / n_states)
    for _ in range(max_iter):
        old = np.r_[prior, skills]
        logp = np.broadcast_to(np.log(prior), (n_tasks, n_states)).copy()
        for worker in range(n_workers):
            q = votes[:, worker, :]
            likelihood = skills[worker] * q + (1 - skills[worker]) * (1 - q) / (n_states - 1)
            logp += np.log(np.clip(likelihood, 1e-12, None))
        logp -= logp.max(axis=1, keepdims=True)
        posterior = np.exp(logp)
        posterior /= posterior.sum(axis=1, keepdims=True)
        prior = np.clip(posterior.mean(axis=0), 1e-06, None)
        prior /= prior.sum()
        skills = np.einsum("tk,tmk->m", posterior, votes) / n_tasks
        skills = np.clip(skills, 1 / n_states + 0.0001, 1 - 1e-06)
        if np.max(np.abs(np.r_[prior, skills] - old)) < tol:
            break
    return (prior, skills, posterior)


def _posterior_for_workers(votes: np.ndarray, prior: np.ndarray, skills: np.ndarray) -> np.ndarray:
    n_tasks, n_workers, n_states = votes.shape
    logp = np.broadcast_to(np.log(prior), (n_tasks, n_states)).copy()
    for worker in range(n_workers):
        q = votes[:, worker, :]
        likelihood = skills[worker] * q + (1 - skills[worker]) * (1 - q) / (n_states - 1)
        logp += np.log(np.clip(likelihood, 1e-12, None))
    logp -= logp.max(axis=1, keepdims=True)
    posterior = np.exp(logp)
    return posterior / posterior.sum(axis=1, keepdims=True)


def _reaches(children: Mapping[str, set[str]], start: str, goal: str) -> bool:
    stack, seen = ([start], set())
    while stack:
        node = stack.pop()
        if node == goal:
            return True
        if node not in seen:
            seen.add(node)
            stack.extend(children.get(node, ()))
    return False


def _dagbag(
    probabilities: Mapping[tuple[str, str], float],
    alpha: float = 1.0,
    threshold: float = 0.5,
) -> pd.DataFrame:
    scores = {
        (u, v): p + (1 - alpha / 2) * probabilities.get((v, u), 0.0)
        for (u, v), p in probabilities.items()
    }
    children: dict[str, set[str]] = {}
    rows = []
    for (u, v), score in sorted(scores.items(), key=lambda item: -item[1]):
        if score <= threshold:
            break
        status = "cyclic_skipped" if _reaches(children, v, u) else "kept"
        if status == "kept":
            children.setdefault(u, set()).add(v)
        rows.append(
            {
                "source": u,
                "target": v,
                "status": status,
                "edge_probability": probabilities[u, v],
                "reverse_probability": probabilities.get((v, u), 0.0),
                "generalised_frequency": score,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "source",
            "target",
            "status",
            "edge_probability",
            "reverse_probability",
            "generalised_frequency",
        ],
    )


def aggregate_model_pair(
    bundle: GliomaBundle,
    dataset: str,
    models: Sequence[str],
    calibration_models: Iterable[str] | None = None,
    alpha: float = 1.0,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DAGBag pair aggregation with globally calibrated one-coin DS skills.

    Skills are fitted using every available calibration model in the dataset,
    then held fixed while the requested two models vote.  This avoids fitting
    Dawid-Skene from only two raters.
    """
    pair_models = resolve_models(models)
    if len(pair_models) != 2:
        raise ValueError("models must contain exactly two models")
    available = bundle.availability.set_index(["dataset", "model"])["available"]
    missing = [m for m in pair_models if not bool(available.get((dataset, m), False))]
    if missing:
        raise ValueError(f"No finished graph for {dataset}: {missing}")
    candidates = resolve_models(
        calibration_models if calibration_models is not None else bundle.models
    )
    calibration = tuple((m for m in candidates if bool(available.get((dataset, m), False))))
    if len(calibration) < 3:
        raise ValueError(f"Dawid-Skene calibration needs >=3 available models; got {calibration}")
    tasks, all_votes = _soft_votes(bundle, dataset, calibration)
    prior, skills, _ = _em_onecoin(all_votes)
    indices = [calibration.index(model) for model in pair_models]
    pair_posterior = _posterior_for_workers(all_votes[:, indices, :], prior, skills[indices])
    probabilities = {}
    for (a, b), posterior in zip(tasks, pair_posterior):
        probabilities[a, b] = float(posterior[0])
        probabilities[b, a] = float(posterior[1])
    edges = _dagbag(probabilities, alpha=alpha, threshold=threshold)
    edges = edges.assign(dataset=dataset, model_a=pair_models[0], model_b=pair_models[1])
    competence = pd.DataFrame(
        {
            "dataset": dataset,
            "model": calibration,
            "accuracy": skills,
            "used_in_pair": [m in pair_models for m in calibration],
        }
    )
    return (edges, competence)


def aggregate_model_combination(
    bundle: GliomaBundle,
    dataset: str,
    models: Sequence[str],
    calibration_models: Iterable[str] | None = None,
    alpha: float = 1.0,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DAGBag aggregation for a combination of two or three tuned models.

    Dawid–Skene is calibrated with every available model in ``calibration_models``
    and its fitted competences are then restricted to the requested voters.
    """
    selected = resolve_models(models)
    if len(selected) not in {2, 3}:
        raise ValueError("models must contain exactly two or three models")
    available = bundle.availability.set_index(["dataset", "model"])["available"]
    missing = [m for m in selected if not bool(available.get((dataset, m), False))]
    if missing:
        raise ValueError(f"No finished graph for {dataset}: {missing}")
    candidates = resolve_models(
        calibration_models if calibration_models is not None else bundle.models
    )
    calibration = tuple(
        (model for model in candidates if bool(available.get((dataset, model), False)))
    )
    if len(calibration) < 3:
        raise ValueError(f"Dawid-Skene calibration needs >=3 available models; got {calibration}")
    tasks, all_votes = _soft_votes(bundle, dataset, calibration)
    prior, skills, _ = _em_onecoin(all_votes)
    indices = [calibration.index(model) for model in selected]
    posterior = _posterior_for_workers(all_votes[:, indices, :], prior, skills[indices])
    probabilities = {}
    for (a, b), task_posterior in zip(tasks, posterior):
        probabilities[a, b] = float(task_posterior[0])
        probabilities[b, a] = float(task_posterior[1])
    edges = _dagbag(probabilities, alpha=alpha, threshold=threshold)
    edges = edges.assign(dataset=dataset, combination=" + ".join(selected))
    competence = pd.DataFrame(
        {
            "dataset": dataset,
            "model": calibration,
            "accuracy": skills,
            "used_in_combination": [m in selected for m in calibration],
        }
    )
    return (edges, competence)


def aggregate_all_pairs(
    bundle: GliomaBundle,
    models: Iterable[str] | None = None,
    alpha: float = 1.0,
    threshold: float = 0.5,
) -> tuple[
    dict[str, dict[tuple[str, str], pd.DataFrame]],
    pd.DataFrame,
    dict[str, pd.DataFrame],
]:
    """Prepare every 2-model combination; unavailable pairs are recorded, not faked."""
    selected = resolve_models(models if models is not None else bundle.models)
    ensembles: dict[str, dict[tuple[str, str], pd.DataFrame]] = {}
    competence_by_dataset: dict[str, pd.DataFrame] = {}
    audit = []
    available_index = bundle.availability.set_index(["dataset", "model"])["available"]
    for dataset in bundle.datasets:
        ensembles[dataset] = {}
        calibration = tuple(
            (model for model in selected if bool(available_index.get((dataset, model), False)))
        )
        if len(calibration) >= 3:
            tasks, all_votes = _soft_votes(bundle, dataset, calibration)
            prior, skills, _ = _em_onecoin(all_votes)
            competence_by_dataset[dataset] = pd.DataFrame(
                {"dataset": dataset, "model": calibration, "accuracy": skills}
            )
        for model_a, model_b in combinations(selected, 2):
            missing = [model for model in (model_a, model_b) if model not in calibration]
            if missing or len(calibration) < 3:
                reason = (
                    f"No finished graph for {dataset}: {missing}"
                    if missing
                    else f"Dawid-Skene calibration needs >=3 models; got {calibration}"
                )
                ensembles[dataset][model_a, model_b] = pd.DataFrame()
                audit.append(
                    {
                        "dataset": dataset,
                        "model_a": model_a,
                        "model_b": model_b,
                        "status": "unavailable",
                        "reason": reason,
                    }
                )
                continue
            indices = [calibration.index(model_a), calibration.index(model_b)]
            pair_posterior = _posterior_for_workers(
                all_votes[:, indices, :], prior, skills[indices]
            )
            probabilities = {}
            for (a, b), posterior in zip(tasks, pair_posterior):
                probabilities[a, b] = float(posterior[0])
                probabilities[b, a] = float(posterior[1])
            edges = _dagbag(probabilities, alpha=alpha, threshold=threshold)
            ensembles[dataset][model_a, model_b] = edges.assign(
                dataset=dataset, model_a=model_a, model_b=model_b
            )
            audit.append(
                {
                    "dataset": dataset,
                    "model_a": model_a,
                    "model_b": model_b,
                    "status": "ready",
                    "reason": "",
                }
            )
    return (ensembles, pd.DataFrame(audit), competence_by_dataset)


def kept_ensemble_edges(ensembles, dataset: str, model_a: str, model_b: str) -> pd.DataFrame:
    a, b = resolve_models((model_a, model_b))
    key = (a, b) if (a, b) in ensembles[dataset] else (b, a)
    frame = ensembles[dataset][key]
    return frame[frame.get("status", pd.Series(dtype=str)) == "kept"].copy()


def import_existing_analyses(bundle: GliomaBundle):
    """Import the existing hypermutation, literature and scoring scripts."""
    literature_dir = bundle.project_root / "analysis/tcga"
    for path in (literature_dir,):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return {
        "hypermutation": importlib.import_module("hypermutation_diagnostic"),
        "literature": importlib.import_module("compare_literature"),
        "model_scoring": importlib.import_module("score_model"),
        "phased_scoring": importlib.import_module("score_model_phased"),
    }


def score_edges_against_literature(
    bundle: GliomaBundle, edges: pd.DataFrame, dataset: str, max_path: int = 3
):
    """Apply the existing real-world relation scorer to any result DataFrame."""
    modules = import_existing_analyses(bundle)
    scorer = modules["model_scoring"]
    kept = edges[edges["status"] == "kept"] if "status" in edges else edges
    skeleton = {frozenset((row.source, row.target)) for row in kept.itertuples(index=False)}
    arrows = {
        (row.source, row.target)
        for row in kept.itertuples(index=False)
        if getattr(row, "orientation", "directed") == "directed"
    }
    graph = scorer.ModelGraph(skeleton, arrows)
    variables = bundle.labels[dataset]
    genes = sorted({gene_of(label) for label in bundle.labels[base_dataset(dataset)]})
    if hasattr(scorer, "all_relations"):
        relations = scorer.all_relations(genes)
    elif hasattr(scorer, "load_relations"):
        relations = scorer.load_relations(genes)
    else:
        raise AttributeError(
            "The literature scorer exposes neither all_relations nor load_relations"
        )
    return scorer.score(graph, variables, relations, max_path)


def score_edges_against_phased_relations(
    bundle: GliomaBundle, edges: pd.DataFrame, dataset: str, max_path: int = 3
):
    """Score one graph against ``phased_relations_glioma_rna_final.csv``.

    Phase 1 contains effects expected at RNA level and is suitable for the
    main evaluation. Phase 2 contains protein-level mechanisms and is kept as
    contextual support: failing to recover it is not treated as model error.
    """
    modules = import_existing_analyses(bundle)
    scorer = modules["phased_scoring"]
    kept = edges[edges["status"] == "kept"] if "status" in edges else edges
    skeleton = {frozenset((row.source, row.target)) for row in kept.itertuples(index=False)}
    arrows = {
        (row.source, row.target)
        for row in kept.itertuples(index=False)
        if getattr(row, "orientation", "directed") == "directed"
    }
    graph = scorer.ModelGraph(skeleton, arrows)
    variables = bundle.labels[dataset]
    genes = sorted({gene_of(label) for label in bundle.labels[base_dataset(dataset)]})
    relations = scorer.load_relations(genes)
    return scorer.score(graph, variables, relations, max_path)


def phased_relation_recovery_summary(
    bundle: GliomaBundle,
    model: str = "PRISM",
    min_rate: float = 0.8,
    graphs: Mapping[str, pd.DataFrame] | None = None,
    max_path: int = 3,
) -> pd.DataFrame:
    """Recovery of phased relations, preserving phase 1 and 2 separately.

    The primary metric is exact directed recovery. Short directed paths are
    retained for exploration, but do not replace the exact-edge metric.
    """
    arm = resolve_models((model,))[0]
    rows = []
    for dataset in bundle.datasets:
        predicted = (
            graphs[dataset].copy()
            if graphs is not None
            else _consensus_frame(bundle, dataset, arm, min_rate)
        )
        detail, _, _ = score_edges_against_phased_relations(
            bundle, predicted, dataset, max_path=max_path
        )
        for phase in (1, 2):
            part = detail[detail["phase"] == phase] if not detail.empty else detail
            total = int(len(part))
            n_direct = int((part["status"] == "direta_sentido_certo").sum()) if total else 0
            n_path = (
                int(part["status"].isin({"direta_sentido_certo", "indireta_sentido_certo"}).sum())
                if total
                else 0
            )
            rows.append(
                {
                    "dataset": dataset,
                    "phase": phase,
                    "phase_role": "avaliacao_mrna" if phase == 1 else "contexto_proteina",
                    "is_evaluable": bool(total),
                    "n_reference_applicable": total,
                    "n_recovered_direct": n_direct,
                    "pct_recovered_direct": 100 * n_direct / total if total else np.nan,
                    "n_recovered_direct_or_path": n_path,
                    "pct_recovered_direct_or_path": 100 * n_path / total if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def phased_reference_precision_summary(
    bundle: GliomaBundle,
    model: str = "PRISM",
    min_rate: float = 0.8,
    graphs: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Exact directed matches to each phase as a fraction of predicted edges.

    This is reference-limited support, not exhaustive biological precision.
    Phase 2 matches are reported as context and must not be interpreted as a
    penalty when absent from RNA-based graphs.
    """
    arm = resolve_models((model,))[0]
    modules = import_existing_analyses(bundle)
    scorer = modules["phased_scoring"]
    genes = sorted({gene_of(label) for label in bundle.labels["tcga_mut_expr70"]})
    relations = scorer.load_relations(genes)
    references = {
        phase: {(r["source"], r["target"]) for r in relations if r["phase"] == phase}
        for phase in (1, 2)
    }
    rows = []
    for dataset in bundle.datasets:
        predicted = (
            graphs[dataset].copy()
            if graphs is not None
            else _consensus_frame(bundle, dataset, arm, min_rate)
        )
        if "status" in predicted:
            predicted = predicted[predicted["status"] == "kept"]
        predicted_edges = {
            (row.source, row.target)
            for row in predicted.itertuples(index=False)
            if getattr(row, "orientation", "directed") == "directed"
        }
        variables = set(bundle.labels[dataset])
        for phase in (1, 2):
            applicable = {
                edge for edge in references[phase] if edge[0] in variables and edge[1] in variables
            }
            confirmed = predicted_edges & applicable
            total = len(predicted_edges)
            rows.append(
                {
                    "dataset": dataset,
                    "phase": phase,
                    "phase_role": "avaliacao_mrna" if phase == 1 else "contexto_proteina",
                    "is_evaluable": bool(applicable),
                    "n_reference_applicable": len(applicable),
                    "n_predicted_edges": total,
                    "n_reference_confirmed": len(confirmed),
                    "n_unmatched_predicted_edges": total - len(confirmed),
                    "pct_reference_confirmed": 100 * len(confirmed) / total
                    if applicable and total
                    else 0.0
                    if applicable
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def phased_literature_agreement(
    bundle: GliomaBundle,
    dataset: str,
    model: str = "PRISM",
    min_rate: float = 0.8,
    phases: Sequence[int] = (1, 2),
    edges: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Exact, correctly oriented reference edges recovered in one dataset."""
    invalid = set(phases) - {1, 2}
    if invalid:
        raise ValueError(f"Unknown phased-relation phase(s): {sorted(invalid)}")
    arm = resolve_models((model,))[0]
    predicted = (
        edges.copy() if edges is not None else _consensus_frame(bundle, dataset, arm, min_rate)
    )
    detail, _, _ = score_edges_against_phased_relations(bundle, predicted, dataset)
    if detail.empty:
        return detail
    return detail[detail["phase"].isin(phases) & detail["status"].eq("direta_sentido_certo")].copy()




def _short_variable(label: str) -> str:
    kind = "M" if variable_kind(label) == "mut" else "R"
    return f"{gene_of(label)} [{kind}]"


def build_analysis_graphs(
    bundle: GliomaBundle,
    min_rate: float = 0.8,
    model: str = "PRISM",
    ensembles=None,
    model_pair: Sequence[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], str]:
    """Select either one model or a two-model DAGBag ensemble for all datasets."""
    if model_pair is None:
        arm = resolve_models((model,))[0]
        return (
            {
                dataset: _consensus_frame(bundle, dataset, arm, min_rate)
                for dataset in bundle.datasets
            },
            arm,
        )
    if ensembles is None:
        raise ValueError("model_pair requires the pair_ensembles object")
    pair_models = resolve_models(model_pair)
    if len(pair_models) != 2:
        raise ValueError("model_pair must contain exactly two models")
    model_a, model_b = pair_models
    graphs = {}
    for dataset in bundle.datasets:
        frame = kept_ensemble_edges(ensembles, dataset, model_a, model_b).copy()
        if not frame.empty:
            frame["orientation"] = "directed"
            frame["rate"] = frame["edge_probability"]
        graphs[dataset] = frame
    return (graphs, f"DAGBag: {model_a} + {model_b}")


def _consensus_frame(
    bundle: GliomaBundle, dataset: str, model: str, min_rate: float
) -> pd.DataFrame:
    """One row per consensus edge, using skeletons for mutation-only data."""
    frame = bundle.result(dataset, model)
    if base_dataset(dataset) != "tcga_mut35":
        out = frame[frame["directed_rate"] >= min_rate].copy()
        out["orientation"] = "directed"
        out["rate"] = out["directed_rate"]
        return out.drop_duplicates(["source", "target"])
    chosen = frame[frame["skeleton_rate"] >= min_rate].copy()
    rows = {}
    for row in chosen.itertuples(index=False):
        source, target = sorted((row.source, row.target))
        key = (source, target)
        value = float(row.skeleton_rate)
        if key not in rows or value > rows[key]["rate"]:
            rows[key] = {
                "source": source,
                "target": target,
                "orientation": "unoriented",
                "rate": value,
                "skeleton_rate": value,
            }
    return pd.DataFrame(rows.values())


def edge_agreement_matrix(
    bundle: GliomaBundle,
    dataset_family: str,
    model: str = "PRISM",
    min_rate: float = 0.8,
    min_datasets: int = 2,
    graphs: Mapping[str, pd.DataFrame] | None = None,
    comparison_mode: str = "auto",
) -> pd.DataFrame:
    """Edge-frequency matrix for the full cohort and three cancer subtypes."""
    base = base_dataset(dataset_family)
    if base not in BASE_DATASETS:
        raise ValueError(f"Unknown dataset family: {dataset_family}")
    if comparison_mode not in {"auto", "directed", "skeleton"}:
        raise ValueError("comparison_mode must be auto, directed or skeleton")
    arm = resolve_models((model,))[0]
    datasets = (base,) + tuple((f"{base}__{subtype}" for subtype in SUBTYPES))
    rates = {}
    for dataset in datasets:
        consensus = (
            graphs[dataset].copy()
            if graphs is not None
            else _consensus_frame(bundle, dataset, arm, min_rate)
        )
        use_skeleton = comparison_mode == "skeleton" or (
            comparison_mode == "auto" and base == "tcga_mut35"
        )
        if use_skeleton and (not consensus.empty):
            canonical = {}
            for row in consensus.itertuples(index=False):
                source, target = sorted((row.source, row.target))
                rate = float(getattr(row, "rate", getattr(row, "edge_probability", 1.0)))
                canonical[source, target] = max(canonical.get((source, target), 0.0), rate)
            consensus = pd.DataFrame(
                [
                    {"source": source, "target": target, "rate": rate}
                    for (source, target), rate in canonical.items()
                ]
            )
        for row in consensus.itertuples(index=False):
            edge = (row.source, row.target)
            rate = float(getattr(row, "rate", getattr(row, "edge_probability", 1.0)))
            rates.setdefault(edge, {})[dataset] = rate
    kept = {edge: values for edge, values in rates.items() if len(values) >= min_datasets}
    index = sorted(kept, key=lambda edge: (-len(kept[edge]), edge))
    matrix = pd.DataFrame(
        [[kept[edge].get(dataset, 0.0) for dataset in datasets] for edge in index],
        index=pd.MultiIndex.from_tuples(index, names=["source", "target"]),
        columns=datasets,
        dtype=float,
    )
    return matrix


def plot_edge_agreement(
    bundle: GliomaBundle,
    dataset_family: str,
    model: str = "PRISM",
    min_rate: float = 0.8,
    min_datasets: int = 2,
    graphs: Mapping[str, pd.DataFrame] | None = None,
    analysis_label: str | None = None,
    comparison_mode: str = "auto",
):
    """Heatmap naming the DAGMA edges reproduced across cohort/subtypes."""
    import matplotlib.pyplot as plt

    matrix = edge_agreement_matrix(
        bundle,
        dataset_family,
        model,
        min_rate,
        min_datasets,
        graphs=graphs,
        comparison_mode=comparison_mode,
    )
    height = max(3.0, min(24.0, 0.28 * max(len(matrix), 1) + 1.8))
    fig, ax = plt.subplots(figsize=(9, height), constrained_layout=True)
    if matrix.empty:
        ax.text(
            0.5,
            0.5,
            "Nenhuma edge cumpre este critério",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        return (fig, ax)
    image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto")
    use_skeleton = comparison_mode == "skeleton" or (
        comparison_mode == "auto" and base_dataset(dataset_family) == "tcga_mut35"
    )
    mode = "—" if use_skeleton else "→"
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(
        [
            f"{_short_variable(source)} {mode} {_short_variable(target)}"
            for source, target in matrix.index
        ]
    )
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(["completo", "astro", "gbm", "oligo"])
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Edge reproduzida em pelo menos " + str(min_datasets) + " datasets")
    label = analysis_label or resolve_models((model,))[0]
    ax.set_title(f"{label} — concordância em {base_dataset(dataset_family)}")
    fig.colorbar(image, ax=ax, label="Frequência da edge")
    return (fig, ax)


def literature_agreement(
    bundle: GliomaBundle,
    dataset: str,
    model: str = "PRISM",
    min_rate: float = 0.8,
    include_reversed: bool = False,
    edges: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Known direct relations recovered by the model in one dataset."""
    arm = resolve_models((model,))[0]
    predicted = (
        edges.copy() if edges is not None else _consensus_frame(bundle, dataset, arm, min_rate)
    )
    detail, _, _ = score_edges_against_literature(bundle, predicted, dataset)
    accepted = {
        "direta",
        "direta_sentido_certo",
        "direta_sem_orientacao",
        "direta_outro_sentido_na_referencia",
    }
    if include_reversed:
        accepted.add("direta_sentido_trocado")
    if detail.empty or "status" not in detail:
        return pd.DataFrame(columns=getattr(detail, "columns", []))
    return detail[detail["status"].isin(accepted)].copy()


def literature_recovery_summary(
    bundle: GliomaBundle,
    model: str = "PRISM",
    min_rate: float = 0.8,
    graphs: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Recall of applicable known relations, strict and allowing short paths.

    Reversed directed relations never count as recovered. ``direct`` requires
    an adjacency in the model; ``direct_or_path`` also accepts a correctly
    oriented path (or a skeleton path for undirected mutation relations), as
    defined by the existing literature scorer.
    """
    arm = resolve_models((model,))[0]
    direct_ok = {"direta", "direta_sentido_certo", "direta_sem_orientacao"}
    path_ok = direct_ok | {"indireta", "indireta_sentido_certo"}
    rows = []
    for dataset in bundle.datasets:
        predicted = (
            graphs[dataset].copy()
            if graphs is not None
            else _consensus_frame(bundle, dataset, arm, min_rate)
        )
        detail, _, _ = score_edges_against_literature(bundle, predicted, dataset)
        total = int(len(detail))
        if detail.empty or "status" not in detail:
            n_direct = n_path = 0
        else:
            n_direct = int(detail["status"].isin(direct_ok).sum())
            n_path = int(detail["status"].isin(path_ok).sum())
        rows.append(
            {
                "dataset": dataset,
                "n_known_applicable": total,
                "n_recovered_direct": n_direct,
                "pct_recovered_direct": 100 * n_direct / total if total else np.nan,
                "n_recovered_direct_or_path": n_path,
                "pct_recovered_direct_or_path": 100 * n_path / total if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_literature_recovery(
    bundle: GliomaBundle,
    model: str = "PRISM",
    min_rate: float = 0.8,
    graphs: Mapping[str, pd.DataFrame] | None = None,
    analysis_label: str | None = None,
):
    """Percent of applicable known relations recovered in every dataset."""
    import matplotlib.pyplot as plt

    summary = literature_recovery_summary(bundle, model=model, min_rate=min_rate, graphs=graphs)
    labels = [
        dataset.replace("tcga_", "")
        .replace("mut_expr70", "mut+rna")
        .replace("mut35", "mut")
        .replace("expr35", "rna")
        .replace("__", " · ")
        for dataset in summary["dataset"]
    ]
    y = np.arange(len(summary))
    height = 0.36
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    direct = ax.barh(
        y - height / 2,
        summary["pct_recovered_direct"],
        height,
        label="edge direta",
        color="tab:green",
    )
    paths = ax.barh(
        y + height / 2,
        summary["pct_recovered_direct_or_path"],
        height,
        label="edge direta ou caminho correto",
        color="tab:blue",
        alpha=0.75,
    )
    for bars, count_column in (
        (direct, "n_recovered_direct"),
        (paths, "n_recovered_direct_or_path"),
    ):
        for bar, recovered, total in zip(
            bars, summary[count_column], summary["n_known_applicable"]
        ):
            ax.text(
                bar.get_width() + 0.6,
                bar.get_y() + bar.get_height() / 2,
                f"{int(recovered)}/{int(total)} ({bar.get_width():.1f}%)",
                va="center",
                ha="left",
                fontsize=8,
            )
    maximum = float(summary["pct_recovered_direct_or_path"].max())
    ax.set_xlim(0, max(20, min(100, maximum + 18)))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Relações conhecidas aplicáveis recuperadas (%)")
    ax.set_ylabel("Dataset")
    label = analysis_label or resolve_models((model,))[0]
    ax.set_title(f"{label} — recuperação das relações conhecidas")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    return (fig, ax, summary)


def reference_precision_summary(
    bundle: GliomaBundle,
    model: str = "PRISM",
    min_rate: float = 0.8,
    graphs: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Fraction of predicted edges directly supported by a project reference.

    This is a conservative, reference-limited precision: an unmatched edge is
    unknown rather than proven false because the biological references are not
    exhaustive.
    """
    arm = resolve_models((model,))[0]
    rows = []
    for dataset in bundle.datasets:
        predicted = (
            graphs[dataset].copy()
            if graphs is not None
            else _consensus_frame(bundle, dataset, arm, min_rate)
        )
        mutation_only = base_dataset(dataset) == "tcga_mut35"
        if mutation_only:
            predicted_keys = {
                frozenset((row.source, row.target)) for row in predicted.itertuples(index=False)
            }
        else:
            predicted_keys = {(row.source, row.target) for row in predicted.itertuples(index=False)}
        detail, _, _ = score_edges_against_literature(bundle, predicted, dataset)
        confirmed = set()
        for row in detail.itertuples(index=False):
            if row.status not in {
                "direta",
                "direta_sentido_certo",
                "direta_sem_orientacao",
            }:
                continue
            if mutation_only:
                key = frozenset((row.source, row.target))
                if key in predicted_keys:
                    confirmed.add(key)
            elif row.status == "direta_sentido_certo":
                key = (row.source, row.target)
                if key in predicted_keys:
                    confirmed.add(key)
            else:
                for key in ((row.source, row.target), (row.target, row.source)):
                    if key in predicted_keys:
                        confirmed.add(key)
        total = len(predicted_keys)
        n_confirmed = len(confirmed)
        rows.append(
            {
                "dataset": dataset,
                "n_predicted_edges": total,
                "n_reference_confirmed": n_confirmed,
                "pct_reference_confirmed": 100 * n_confirmed / total if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def strict_reference_precision_summary(
    bundle: GliomaBundle,
    model: str = "PRISM",
    min_rate: float = 0.8,
    graphs: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Exact directed-edge precision using only ``strict_relations.csv``.

    Only a predicted arrow with the same source and target as a strict
    reference counts as confirmed. Mutation-only datasets are not evaluable,
    because the strict file contains no mutation--mutation relations.
    """
    arm = resolve_models((model,))[0]
    path = bundle.project_root / "datasets/tcga/references/strict_relations.csv"
    strict = pd.read_csv(path)
    suffix = {"mut": MUTATION_SUFFIX, "expr": EXPRESSION_SUFFIX}
    all_reference_edges = {
        (
            f"{row.source_gene}{suffix[row.source_kind]}",
            f"{row.target_gene}{suffix[row.target_kind]}",
        )
        for row in strict.itertuples(index=False)
    }
    rows = []
    for dataset in bundle.datasets:
        predicted = (
            graphs[dataset].copy()
            if graphs is not None
            else _consensus_frame(bundle, dataset, arm, min_rate)
        )
        if "status" in predicted:
            predicted = predicted[predicted["status"] == "kept"]
        predicted_edges = {
            (row.source, row.target)
            for row in predicted.itertuples(index=False)
            if getattr(row, "orientation", "directed") == "directed"
        }
        variables = set(bundle.labels[dataset])
        applicable = {
            edge for edge in all_reference_edges if edge[0] in variables and edge[1] in variables
        }
        confirmed = predicted_edges & applicable
        evaluable = bool(applicable)
        total = len(predicted_edges)
        rows.append(
            {
                "dataset": dataset,
                "is_evaluable": evaluable,
                "n_strict_applicable": len(applicable),
                "n_predicted_edges": total,
                "n_reference_confirmed": len(confirmed),
                "pct_reference_confirmed": 100 * len(confirmed) / total
                if evaluable and total
                else 0.0
                if evaluable
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def strict_reference_recovery_summary(
    bundle: GliomaBundle, graphs: Mapping[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exact directed recall against ``strict_relations.csv``.

    The strict reference has no phase labels.  A relation is applicable to a
    dataset only when both variables are present, and it is recovered only
    when the graph contains the arrow in the same direction.
    """
    path = bundle.project_root / "datasets/tcga/references/strict_relations.csv"
    strict = pd.read_csv(path)
    suffix = {"mut": MUTATION_SUFFIX, "expr": EXPRESSION_SUFFIX}
    reference = [
        {
            "source": f"{row.source_gene}{suffix[row.source_kind]}",
            "target": f"{row.target_gene}{suffix[row.target_kind]}",
            "group": row.group,
        }
        for row in strict.itertuples(index=False)
    ]
    summary_rows, detail_rows = ([], [])
    for dataset, graph in graphs.items():
        selected = graph.copy()
        if "status" in selected:
            selected = selected[selected["status"] == "kept"]
        predicted = {
            (row.source, row.target)
            for row in selected.itertuples(index=False)
            if getattr(row, "orientation", "directed") == "directed"
        }
        variables = set(bundle.labels[dataset])
        applicable = [
            row for row in reference if row["source"] in variables and row["target"] in variables
        ]
        recovered = 0
        for row in applicable:
            is_recovered = (row["source"], row["target"]) in predicted
            recovered += int(is_recovered)
            detail_rows.append(
                {
                    "dataset": dataset,
                    **row,
                    "status": "recovered" if is_recovered else "not_recovered",
                }
            )
        summary_rows.append(
            {
                "dataset": dataset,
                "n_reference_applicable": len(applicable),
                "n_reference_recovered_direct": recovered,
                "pct_reference_recovered_direct": 100 * recovered / len(applicable)
                if applicable
                else np.nan,
            }
        )
    return (pd.DataFrame(summary_rows), pd.DataFrame(detail_rows))


def combination_label(models: Sequence[str], method: str | None = None) -> str:
    """Readable label for a model combination and its aggregation method."""
    display = {
        "PC__tuned": "PC",
        "GES__tuned": "GES",
        "GRaSP__tuned": "GRASP",
        "GOLEM__tuned": "GOLEM",
        "AVICI__tuned": "AVICI",
        "PRISM__tuned": "PRISM",
    }
    resolved = resolve_models(models)
    prefixes = {
        None: "Modelo" if len(resolved) == 1 else "DAGBag",
        "single": "Modelo",
        "intersection": "Interseção",
        "dagbag_uniform": "DAGBag uniforme",
        "dagbag_dawid_skene": "DAGBag + Dawid–Skene",
    }
    if method not in prefixes:
        raise ValueError(f"Unknown aggregation method: {method!r}")
    prefix = prefixes[method]
    return f"{prefix}: " + " + ".join((display.get(model, model) for model in resolved))


def _empty_aggregated_graph() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source",
            "target",
            "status",
            "orientation",
            "rate",
            "edge_probability",
            "reverse_probability",
            "generalised_frequency",
        ]
    )


def _intersection_graph(
    bundle: GliomaBundle,
    dataset: str,
    models: Sequence[str],
    min_rate: float,
    require_unanimous_direction: bool = False,
) -> pd.DataFrame:
    """Strict adjacency intersection, oriented only under unanimity."""
    selected = resolve_models(models)
    skeletons = []
    directions = []
    frames = []
    for model in selected:
        frame = bundle.result(dataset, model)
        frames.append(frame)
        skeletons.append(edge_set(frame, min_rate=min_rate, mode="skeleton"))
        directions.append(edge_set(frame, min_rate=min_rate, mode="directed"))
    common = set.intersection(*skeletons) if skeletons else set()
    rows = []
    for adjacency in sorted(common, key=lambda edge: tuple(sorted(edge))):
        a, b = sorted(adjacency)
        unanimous_ab = all(((a, b) in model_edges for model_edges in directions))
        unanimous_ba = all(((b, a) in model_edges for model_edges in directions))
        if unanimous_ab != unanimous_ba:
            source, target = (a, b) if unanimous_ab else (b, a)
            orientation = "directed"
            rates = []
            for frame in frames:
                values = frame.loc[
                    (frame["source"] == source) & (frame["target"] == target),
                    "directed_rate",
                ]
                rates.append(float(values.max()))
        elif require_unanimous_direction and base_dataset(dataset) != "tcga_mut35":
            continue
        else:
            source, target = (a, b)
            orientation = "unoriented"
            rates = []
            for frame in frames:
                values = frame.loc[
                    (frame["source"] == a) & (frame["target"] == b)
                    | (frame["source"] == b) & (frame["target"] == a),
                    "skeleton_rate",
                ]
                rates.append(float(values.max()))
        rows.append(
            {
                "source": source,
                "target": target,
                "status": "kept",
                "orientation": orientation,
                "rate": min(rates),
                "edge_probability": min(rates),
                "reverse_probability": 0.0,
                "generalised_frequency": min(rates),
                "dataset": dataset,
                "combination": " + ".join(selected),
            }
        )
    return pd.DataFrame(rows) if rows else _empty_aggregated_graph()


def rank_model_combinations(
    bundle: GliomaBundle,
    models: Iterable[str] | None = None,
    min_rate: float = 0.8,
    max_size: int = 3,
    alpha: float = 1.0,
    threshold: float = 0.5,
    aggregation_methods: Sequence[str] = (
        "intersection",
        "dagbag_uniform",
        "dagbag_dawid_skene",
    ),
    require_unanimous_intersection_direction: bool = False,
    precision_summary_function=None,
) -> tuple[
    pd.DataFrame,
    dict[tuple[str, tuple[str, ...]], dict[str, pd.DataFrame]],
    dict[tuple[str, tuple[str, ...]], pd.DataFrame],
]:
    """Rank singles and all pair/trio aggregation strategies by precision.

    Every dataset has equal weight. A dataset with no predicted edges receives
    zero precision so that an empty/sparse solution cannot improve its ranking
    merely by omitting datasets. Multi-model candidates can use strict
    intersection, uniformly weighted DAGBag, or DAGBag with one-coin
    Dawid–Skene calibrated from all selected available models.
    """
    selected = resolve_models(models if models is not None else bundle.models)
    if max_size not in {1, 2, 3}:
        raise ValueError("max_size must be 1, 2 or 3")
    if max_size >= 2 and len(selected) < 3:
        raise ValueError("Dawid-Skene calibration requires at least three models")
    allowed_methods = {"intersection", "dagbag_uniform", "dagbag_dawid_skene"}
    methods = tuple(aggregation_methods)
    unknown_methods = set(methods) - allowed_methods
    if unknown_methods:
        raise ValueError(f"Unknown aggregation method(s): {sorted(unknown_methods)}")
    if len(set(methods)) != len(methods):
        raise ValueError(f"Repeated aggregation method: {methods}")
    combinations_to_test = tuple(
        (
            combo
            for size in range(1, min(max_size, len(selected)) + 1)
            for combo in combinations(selected, size)
        )
    )
    candidates = tuple(
        [("single", combo) for combo in combinations_to_test if len(combo) == 1]
        + [
            (method, combo)
            for combo in combinations_to_test
            if len(combo) > 1
            for method in methods
        ]
    )
    graphs_by_combination = {candidate: {} for candidate in candidates}
    available = bundle.availability.set_index(["dataset", "model"])["available"]
    for dataset in bundle.datasets:
        calibration = tuple(
            (model for model in selected if bool(available.get((dataset, model), False)))
        )
        for method, combo in (item for item in candidates if item[0] == "single"):
            model = combo[0]
            if model in calibration:
                graph = _consensus_frame(bundle, dataset, model, min_rate)
            else:
                graph = _empty_aggregated_graph()
            graphs_by_combination[method, combo][dataset] = graph
        multi = [candidate for candidate in candidates if candidate[0] != "single"]
        if not multi:
            continue
        tasks, all_votes = (
            _soft_votes(bundle, dataset, calibration) if calibration else ([], np.empty((0, 0, 3)))
        )
        if len(calibration) >= 3:
            prior, skills, _ = _em_onecoin(all_votes)
        else:
            prior, skills = (None, None)
        for method, combo in multi:
            if any((model not in calibration for model in combo)):
                graph = _empty_aggregated_graph()
            elif method == "intersection":
                graph = _intersection_graph(
                    bundle,
                    dataset,
                    combo,
                    min_rate,
                    require_unanimous_direction=require_unanimous_intersection_direction,
                )
            else:
                indices = [calibration.index(model) for model in combo]
                if method == "dagbag_uniform":
                    posterior = all_votes[:, indices, :].mean(axis=1)
                elif prior is not None and skills is not None:
                    posterior = _posterior_for_workers(
                        all_votes[:, indices, :], prior, skills[indices]
                    )
                else:
                    graph = _empty_aggregated_graph()
                    graphs_by_combination[method, combo][dataset] = graph
                    continue
                probabilities = {}
                for (a, b), task_posterior in zip(tasks, posterior):
                    probabilities[a, b] = float(task_posterior[0])
                    probabilities[b, a] = float(task_posterior[1])
                graph = _dagbag(probabilities, alpha=alpha, threshold=threshold)
                graph = graph[graph["status"] == "kept"].copy()
                graph["orientation"] = "directed"
                graph["rate"] = graph["edge_probability"]
                graph["dataset"] = dataset
                graph["combination"] = " + ".join(combo)
            graphs_by_combination[method, combo][dataset] = graph
    details = {}
    rows = []
    summarise_precision = precision_summary_function or reference_precision_summary
    for candidate, graphs in graphs_by_combination.items():
        method, combo = candidate
        summary = summarise_precision(bundle, graphs=graphs)
        details[candidate] = summary
        evaluable = (
            summary["is_evaluable"].astype(bool)
            if "is_evaluable" in summary
            else pd.Series(True, index=summary.index)
        )
        evaluated = summary.loc[evaluable]
        values = evaluated["pct_reference_confirmed"].fillna(0.0)
        total_predicted = int(evaluated["n_predicted_edges"].sum())
        total_confirmed = int(evaluated["n_reference_confirmed"].sum())
        rows.append(
            {
                "candidate": candidate,
                "combination": combo,
                "method": method,
                "label": combination_label(combo, method=method),
                "n_models": len(combo),
                "mean_precision_pct": float(values.mean()),
                "pooled_precision_pct": 100 * total_confirmed / total_predicted
                if total_predicted
                else 0.0,
                "n_datasets_with_predictions": int((evaluated["n_predicted_edges"] > 0).sum()),
                "n_datasets_evaluated": int(evaluable.sum()),
                "n_predicted_edges": total_predicted,
                "n_reference_confirmed": total_confirmed,
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["mean_precision_pct", "pooled_precision_pct", "n_reference_confirmed"],
        ascending=[False, False, False],
        ignore_index=True,
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return (ranking, graphs_by_combination, details)


def rank_model_combinations_strict(
    bundle: GliomaBundle,
    models: Iterable[str] | None = None,
    min_rate: float = 0.8,
    max_size: int = 3,
    alpha: float = 1.0,
    threshold: float = 0.5,
    aggregation_methods: Sequence[str] = (
        "intersection",
        "dagbag_uniform",
        "dagbag_dawid_skene",
    ),
):
    """Rank the 65 candidates using oriented intersections and strict refs."""
    return rank_model_combinations(
        bundle,
        models=models,
        min_rate=min_rate,
        max_size=max_size,
        alpha=alpha,
        threshold=threshold,
        aggregation_methods=aggregation_methods,
        require_unanimous_intersection_direction=True,
        precision_summary_function=strict_reference_precision_summary,
    )


def plot_top_precision_combinations(
    ranking: pd.DataFrame, top_n: int = 5, title: str | None = None
):
    """Compare the best combinations by mean precision over all datasets."""
    import matplotlib.pyplot as plt

    top = ranking.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.8 * len(top) + 1.8)), constrained_layout=True)
    bars = ax.barh(top["label"], top["mean_precision_pct"], color="tab:purple")
    for bar, mean, confirmed, predicted in zip(
        bars,
        top["mean_precision_pct"],
        top["n_reference_confirmed"],
        top["n_predicted_edges"],
    ):
        ax.text(
            bar.get_width() + 0.25,
            bar.get_y() + bar.get_height() / 2,
            f"média {mean:.1f}% · {int(confirmed)}/{int(predicted)} no total",
            va="center",
            ha="left",
            fontsize=9,
        )
    maximum = float(top["mean_precision_pct"].max()) if len(top) else 0.0
    ax.set_xlim(0, max(10, min(100, maximum + 18)))
    evaluated_counts = top.get("n_datasets_evaluated", pd.Series(dtype=int)).unique()
    if len(evaluated_counts) == 1:
        axis_label = f"Precisão média nos {int(evaluated_counts[0])} datasets avaliáveis (%)"
    else:
        axis_label = "Precisão média nos datasets avaliáveis (%)"
    ax.set_xlabel(axis_label)
    ax.set_ylabel("Modelo ou combinação")
    ax.set_title(title or f"Top {len(top)} por precisão média limitada às referências")
    ax.grid(axis="x", alpha=0.2)
    return (fig, ax)


def plot_reference_precision(
    bundle: GliomaBundle,
    model: str = "PRISM",
    min_rate: float = 0.8,
    graphs: Mapping[str, pd.DataFrame] | None = None,
    analysis_label: str | None = None,
):
    """Plot directly reference-confirmed edges as a fraction of all predictions."""
    import matplotlib.pyplot as plt

    summary = reference_precision_summary(bundle, model=model, min_rate=min_rate, graphs=graphs)
    labels = [
        dataset.replace("tcga_", "")
        .replace("mut_expr70", "mut+rna")
        .replace("mut35", "mut")
        .replace("expr35", "rna")
        .replace("__", " · ")
        for dataset in summary["dataset"]
    ]
    y = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    bars = ax.barh(y, summary["pct_reference_confirmed"], color="tab:purple")
    for bar, confirmed, total in zip(
        bars, summary["n_reference_confirmed"], summary["n_predicted_edges"]
    ):
        percentage = bar.get_width()
        ax.text(
            percentage + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{int(confirmed)}/{int(total)} ({percentage:.1f}%)",
            va="center",
            ha="left",
            fontsize=8,
        )
    finite = summary["pct_reference_confirmed"].dropna()
    maximum = float(finite.max()) if len(finite) else 0.0
    ax.set_xlim(0, max(10, min(100, maximum + 15)))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Edges previstas diretamente confirmadas pelas referências (%)")
    ax.set_ylabel("Dataset")
    label = analysis_label or resolve_models((model,))[0]
    ax.set_title(f"{label} — fração das edges previstas com suporte conhecido")
    ax.grid(axis="x", alpha=0.2)
    return (fig, ax, summary)


def plot_strict_reference_precision(
    bundle: GliomaBundle,
    model: str = "PRISM",
    min_rate: float = 0.8,
    graphs: Mapping[str, pd.DataFrame] | None = None,
    analysis_label: str | None = None,
):
    """Plot exact oriented matches against ``strict_relations.csv``."""
    import matplotlib.pyplot as plt

    summary = strict_reference_precision_summary(
        bundle, model=model, min_rate=min_rate, graphs=graphs
    )
    labels = [
        dataset.replace("tcga_", "")
        .replace("mut_expr70", "mut+rna")
        .replace("mut35", "mut")
        .replace("expr35", "rna")
        .replace("__", " · ")
        for dataset in summary["dataset"]
    ]
    y = np.arange(len(summary))
    values = summary["pct_reference_confirmed"].fillna(0.0)
    colors = ["tab:purple" if value else "#d9d9d9" for value in summary["is_evaluable"]]
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    bars = ax.barh(y, values, color=colors)
    for bar, row in zip(bars, summary.itertuples(index=False)):
        if not row.is_evaluable:
            text = "n/a — sem relações estritas aplicáveis"
        else:
            text = f"{int(row.n_reference_confirmed)}/{int(row.n_predicted_edges)} ({row.pct_reference_confirmed:.1f}%)"
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            text,
            va="center",
            ha="left",
            fontsize=8,
        )
    finite = summary["pct_reference_confirmed"].dropna()
    maximum = float(finite.max()) if len(finite) else 0.0
    ax.set_xlim(0, max(10, min(100, maximum + 20)))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Setas previstas exatamente confirmadas pelas referências estritas (%)")
    ax.set_ylabel("Dataset")
    label = analysis_label or resolve_models((model,))[0]
    ax.set_title(f"{label} — precisão orientada (strict_relations.csv)")
    ax.grid(axis="x", alpha=0.2)
    return (fig, ax, summary)


def plot_literature_agreement(
    bundle: GliomaBundle,
    dataset: str,
    model: str = "PRISM",
    min_rate: float = 0.8,
    include_reversed: bool = False,
    edges: pd.DataFrame | None = None,
    analysis_label: str | None = None,
):
    """Network containing only direct model edges represented in the references."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    matches = literature_agreement(bundle, dataset, model, min_rate, include_reversed, edges=edges)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    label = analysis_label or resolve_models((model,))[0]
    ax.set_title(f"{label} — relações conhecidas em {dataset}")
    if matches.empty:
        ax.text(
            0.5,
            0.5,
            "Nenhuma relação conhecida direta recuperada",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        return (fig, ax)
    edges = []
    for row in matches.itertuples(index=False):
        if row.status in {
            "direta_sentido_trocado",
            "direta_outro_sentido_na_referencia",
        }:
            source, target = (row.target, row.source)
        else:
            source, target = (row.source, row.target)
        edges.append(
            (
                source,
                target,
                row.group,
                row.status,
                bool(getattr(row, "directed", True)),
            )
        )
    nodes = sorted({node for edge in edges for node in edge[:2]})
    angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False) + np.pi / 2
    pos = {node: np.array([np.cos(angle), np.sin(angle)]) for node, angle in zip(nodes, angles)}
    directed_pairs = {(source, target) for source, target, _, _, _ in edges}
    for source, target, group, status, directed in edges:
        reverse_exists = (target, source) in directed_pairs
        radial = 0.18 if reverse_exists and source < target else -0.18 if reverse_exists else 0.0
        color = "tab:red" if status == "direta_sentido_trocado" else "tab:green"
        patch = FancyArrowPatch(
            pos[source],
            pos[target],
            arrowstyle="-|>" if directed else "-",
            connectionstyle=f"arc3,rad={radial}",
            mutation_scale=16,
            linewidth=2,
            linestyle="--" if status == "direta_sentido_trocado" else "-",
            color=color,
            shrinkA=22,
            shrinkB=22,
        )
        ax.add_patch(patch)
        midpoint = (pos[source] + pos[target]) / 2
        ax.text(
            midpoint[0],
            midpoint[1],
            group,
            fontsize=7,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.5},
        )
    for node in nodes:
        color = "tab:orange" if variable_kind(node) == "mut" else "tab:blue"
        ax.scatter(*pos[node], s=1250, color=color, alpha=0.88, zorder=3)
        ax.text(
            *pos[node],
            _short_variable(node),
            fontsize=8,
            ha="center",
            va="center",
            zorder=4,
        )
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="tab:orange",
            label="mutação",
            markersize=9,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="tab:blue",
            label="RNA",
            markersize=9,
        ),
        Line2D([0], [0], color="tab:green", lw=2, label="sentido compatível"),
    ]
    if include_reversed:
        legend.append(
            Line2D(
                [0],
                [0],
                color="tab:red",
                lw=2,
                linestyle="--",
                label="sentido contrário",
            )
        )
    ax.legend(handles=legend, loc="best", frameon=False)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    return (fig, ax)


def plot_phased_literature_agreement(
    bundle: GliomaBundle,
    dataset: str,
    model: str = "PRISM",
    min_rate: float = 0.8,
    phases: Sequence[int] = (1, 2),
    edges: pd.DataFrame | None = None,
    analysis_label: str | None = None,
):
    """Network of exact matches, distinguishing RNA and protein phases."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    matches = phased_literature_agreement(
        bundle, dataset, model=model, min_rate=min_rate, phases=phases, edges=edges
    )
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    label = analysis_label or resolve_models((model,))[0]
    ax.set_title(f"{label} — relações faseadas recuperadas em {dataset}")
    if matches.empty:
        ax.text(
            0.5,
            0.5,
            "Nenhuma relação faseada direta recuperada",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        return (fig, ax)
    found = [
        (row.source, row.target, row.group, int(row.phase))
        for row in matches.itertuples(index=False)
    ]
    nodes = sorted({node for edge in found for node in edge[:2]})
    angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False) + np.pi / 2
    pos = {node: np.array([np.cos(angle), np.sin(angle)]) for node, angle in zip(nodes, angles)}
    directed_pairs = {(source, target) for source, target, _, _ in found}
    phase_colors = {1: "tab:green", 2: "tab:purple"}
    for source, target, group, phase in found:
        reverse_exists = (target, source) in directed_pairs
        radial = 0.18 if reverse_exists and source < target else -0.18 if reverse_exists else 0.0
        patch = FancyArrowPatch(
            pos[source],
            pos[target],
            arrowstyle="-|>",
            connectionstyle=f"arc3,rad={radial}",
            mutation_scale=16,
            linewidth=2,
            linestyle="-" if phase == 1 else "--",
            color=phase_colors[phase],
            shrinkA=22,
            shrinkB=22,
        )
        ax.add_patch(patch)
        midpoint = (pos[source] + pos[target]) / 2
        ax.text(
            midpoint[0],
            midpoint[1],
            group,
            fontsize=7,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.5},
        )
    for node in nodes:
        color = "tab:orange" if variable_kind(node) == "mut" else "tab:blue"
        ax.scatter(*pos[node], s=1250, color=color, alpha=0.88, zorder=3)
        ax.text(
            *pos[node],
            _short_variable(node),
            fontsize=8,
            ha="center",
            va="center",
            zorder=4,
        )
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="tab:orange",
            label="mutação",
            markersize=9,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="tab:blue",
            label="RNA",
            markersize=9,
        ),
    ]
    if 1 in phases:
        legend.append(Line2D([0], [0], color=phase_colors[1], lw=2, label="fase 1 — efeito no RNA"))
    if 2 in phases:
        legend.append(
            Line2D(
                [0],
                [0],
                color=phase_colors[2],
                lw=2,
                linestyle="--",
                label="fase 2 — contexto proteico",
            )
        )
    ax.legend(handles=legend, loc="best", frameon=False)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    return (fig, ax)


def same_gene_rate_matrix(
    bundle: GliomaBundle,
    model: str = "PRISM",
    graphs: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Mutation -> matching RNA frequency for every gene and 70-feature dataset."""
    arm = resolve_models((model,))[0]
    datasets = ("tcga_mut_expr70",) + tuple((f"tcga_mut_expr70__{subtype}" for subtype in SUBTYPES))
    genes = sorted({gene_of(label) for label in bundle.labels["tcga_mut_expr70"]})
    matrix = pd.DataFrame(0.0, index=genes, columns=datasets)
    for dataset in datasets:
        frame = graphs[dataset] if graphs is not None else bundle.result(dataset, arm)
        for gene in genes:
            source, target = (gene + MUTATION_SUFFIX, gene + EXPRESSION_SUFFIX)
            rows = frame[(frame["source"] == source) & (frame["target"] == target)]
            if not rows.empty:
                rate_column = (
                    "rate"
                    if "rate" in rows
                    else "edge_probability"
                    if "edge_probability" in rows
                    else "directed_rate"
                )
                matrix.loc[gene, dataset] = float(rows[rate_column].max())
    return matrix


def same_gene_mut_rna_catalog(
    bundle: GliomaBundle, graphs: Mapping[str, pd.DataFrame], min_rate: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Catalogue and count every predicted cis mutation -> RNA edge.

    These edges are selected only by variable type, matching gene and exact
    direction. They do not need to occur in the biological reference file.
    The summary denominator is the number of genes for which both mutation and
    RNA variables are present in each dataset.
    """
    datasets = ("tcga_mut_expr70",) + tuple((f"tcga_mut_expr70__{subtype}" for subtype in SUBTYPES))
    edge_rows, summary_rows = ([], [])
    for dataset in datasets:
        labels = bundle.labels[dataset]
        mutation_genes = {gene_of(label) for label in labels if variable_kind(label) == "mut"}
        expression_genes = {gene_of(label) for label in labels if variable_kind(label) == "rna"}
        possible_genes = mutation_genes & expression_genes
        graph = graphs.get(dataset, pd.DataFrame())
        found = {}
        for row in graph.itertuples(index=False):
            try:
                source_kind = variable_kind(row.source)
                target_kind = variable_kind(row.target)
            except ValueError:
                continue
            gene = gene_of(row.source)
            if (
                source_kind != "mut"
                or target_kind != "rna"
                or gene != gene_of(row.target)
                or (gene not in possible_genes)
            ):
                continue
            rate = float(
                getattr(
                    row,
                    "rate",
                    getattr(row, "edge_probability", getattr(row, "directed_rate", 1.0)),
                )
            )
            if rate >= min_rate:
                found[row.source, row.target] = max(found.get((row.source, row.target), 0.0), rate)
        for (source, target), rate in sorted(found.items()):
            edge_rows.append(
                {
                    "dataset": dataset,
                    "gene": gene_of(source),
                    "source": source,
                    "target": target,
                    "rate": rate,
                }
            )
        n_possible = len(possible_genes)
        n_predicted = len(found)
        summary_rows.append(
            {
                "dataset": dataset,
                "n_same_gene_predicted": n_predicted,
                "n_same_gene_possible": n_possible,
                "pct_same_gene_predicted": 100 * n_predicted / n_possible if n_possible else np.nan,
            }
        )
    catalog_columns = ["dataset", "gene", "source", "target", "rate"]
    return (
        pd.DataFrame(edge_rows, columns=catalog_columns),
        pd.DataFrame(summary_rows),
    )


def plot_same_gene_mut_rna(
    bundle: GliomaBundle,
    model: str = "PRISM",
    min_rate: float = 0.8,
    show_all_genes: bool = False,
    graphs: Mapping[str, pd.DataFrame] | None = None,
    analysis_label: str | None = None,
):
    """Heatmap of cis mutation -> RNA links, highlighting the chosen threshold."""
    import matplotlib.pyplot as plt

    matrix = same_gene_rate_matrix(bundle, model, graphs=graphs)
    shown = matrix if show_all_genes else matrix[(matrix >= min_rate).any(axis=1)]
    height = max(3.0, 0.42 * max(len(shown), 1) + 1.5)
    fig, ax = plt.subplots(figsize=(8, height), constrained_layout=True)
    if shown.empty:
        ax.text(
            0.5,
            0.5,
            "Nenhuma ligação mutação → RNA passou o limiar",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        return (fig, ax)
    image = ax.imshow(shown.to_numpy(), vmin=0, vmax=1, cmap="YlGn", aspect="auto")
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels(shown.index)
    ax.set_xticks(range(len(shown.columns)))
    ax.set_xticklabels(["completo", "astro", "gbm", "oligo"])
    for row in range(len(shown)):
        for column in range(len(shown.columns)):
            value = shown.iat[row, column]
            if value > 0:
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="black" if value < 0.75 else "white",
                    fontsize=8,
                )
    ax.set_xlabel("Dataset mutação + RNA")
    ax.set_ylabel("Gene")
    label = analysis_label or resolve_models((model,))[0]
    ax.set_title(f"{label} — mutação → RNA do próprio gene")
    fig.colorbar(image, ax=ax, label="Frequência da edge")
    return (fig, ax)


MODEL_DISPLAY_NAMES = {
    "PC__tuned": "PC",
    "GES__tuned": "GES",
    "GRaSP__tuned": "GRaSP",
    "GOLEM__tuned": "GOLEM",
    "AVICI__tuned": "AVICI",
    "PRISM__tuned": "PRISM",
}


def display_model_name(model: str) -> str:
    """Return the publication-facing name of a configured model."""
    resolved = resolve_models((model,))[0]
    return MODEL_DISPLAY_NAMES.get(resolved, resolved)


def dataset_display_name(dataset: str) -> str:
    """Compact, stable label used by all final TCGA plots."""
    return (
        dataset.replace("tcga_", "")
        .replace("mut_expr70", "mut+rna")
        .replace("mut35", "mut")
        .replace("expr35", "rna")
        .replace("__", " · ")
    )


def build_model_graphs(
    bundle: GliomaBundle,
    models: Iterable[str] | None = None,
    min_rate: float = 0.8,
    edge_mode: str = "directed",
) -> dict[str, dict[str, pd.DataFrame]]:
    """Consensus graphs for every requested individual model and dataset.

    ``edge_mode='directed'`` applies the same exact-arrow rule to all twelve
    datasets, including mutation-only data. ``legacy`` preserves the older
    mutation-skeleton convention used by the exploratory notebook.
    """
    if edge_mode not in {"directed", "legacy"}:
        raise ValueError("edge_mode must be 'directed' or 'legacy'")
    selected = resolve_models(models if models is not None else bundle.models)

    def graph_for(dataset: str, model: str) -> pd.DataFrame:
        if edge_mode == "legacy":
            return _consensus_frame(bundle, dataset, model, min_rate)
        frame = bundle.result(dataset, model)
        out = frame[frame["directed_rate"] >= min_rate].copy()
        out["orientation"] = "directed"
        out["rate"] = out["directed_rate"]
        return out.drop_duplicates(["source", "target"])

    return {
        display_model_name(model): {
            dataset: graph_for(dataset, model) for dataset in bundle.datasets
        }
        for model in selected
    }


def build_internal_threshold_graphs(
    bundle: GliomaBundle,
    thresholds: Sequence[float] = (0.1, 0.125, 0.15, 0.2),
    consensus_rate: float = 0.8,
    model: str = "PRISM",
    matrix_roots: Sequence[str | Path] | None = None,
) -> tuple[dict[float, dict[str, pd.DataFrame]], pd.DataFrame]:
    """Re-threshold stored continuous matrices without refitting the model.

    ``thresholds`` act on ``abs(W)`` inside each fitted seed. ``consensus_rate``
    is the separate across-seed frequency required for an edge to be retained.
    Missing matrices are audited and never converted into empty predictions.
    """
    arm = resolve_models((model,))[0]
    roots = tuple(
        (
            Path(path)
            for path in matrix_roots
            or (
                bundle.tcga_root / "arms_results",
                bundle.tcga_root / "arms_results_subtypes",
            )
        )
    )
    clean_thresholds = tuple(dict.fromkeys((float(value) for value in thresholds)))
    if not clean_thresholds or any((value < 0 for value in clean_thresholds)):
        raise ValueError("thresholds must contain non-negative values")
    if not 0 < consensus_rate <= 1:
        raise ValueError("consensus_rate must be in (0, 1]")
    graphs = {threshold: {} for threshold in clean_thresholds}
    audit_rows = []
    matrix_name = arm + ".npz"
    availability = bundle.availability.loc[bundle.availability["model"].eq(arm)]
    planned_by_dataset = dict(zip(availability["dataset"], availability["n_planned_runs"]))
    for dataset in bundle.datasets:
        dataset_dir = next((root / dataset for root in roots if (root / dataset).is_dir()), None)
        matrix_files = (
            sorted(dataset_dir.glob(f"model_seed_*/matrices/{matrix_name}"))
            if dataset_dir is not None
            else []
        )
        matrices, labels = ([], None)
        error = ""
        try:
            for path in matrix_files:
                with np.load(path, allow_pickle=False) as stored:
                    key = "matrix" if "matrix" in stored else "W"
                    matrix = np.asarray(stored[key], dtype=float)
                    current_labels = [str(value) for value in stored["col_names"].tolist()]
                if matrix.shape != (len(current_labels), len(current_labels)):
                    raise ValueError(f"invalid matrix shape in {path}")
                if labels is not None and current_labels != labels:
                    raise ValueError(f"inconsistent labels in {path}")
                matrices.append(matrix)
                labels = current_labels
        except (OSError, KeyError, ValueError) as exc:
            matrices, labels, error = ([], None, str(exc))
        planned = int(planned_by_dataset.get(dataset, len(matrix_files)))
        status = "ready" if matrices and planned > 0 else "unavailable"
        if status == "unavailable" and (not error):
            error = "continuous DAGMA matrices not found"
        audit_rows.append(
            {
                "dataset": dataset,
                "model": arm,
                "status": status,
                "matrix_root": str(dataset_dir) if dataset_dir is not None else "",
                "n_matrices_found": len(matrices),
                "n_seeds_planned": planned,
                "reason": error,
            }
        )
        if status != "ready":
            continue
        stack = np.stack(matrices)
        for threshold in clean_thresholds:
            present = np.abs(stack) > threshold
            for index in range(len(present)):
                np.fill_diagonal(present[index], False)
            counts = present.sum(axis=0)
            source_indices, target_indices = np.nonzero(counts / planned >= consensus_rate)
            frame = pd.DataFrame(
                {
                    "source": [labels[index] for index in source_indices],
                    "target": [labels[index] for index in target_indices],
                    "orientation": "directed",
                    "rate": [
                        float(counts[source, target] / planned)
                        for source, target in zip(source_indices, target_indices)
                    ],
                    "internal_threshold": threshold,
                    "n_seeds_present": [
                        int(counts[source, target])
                        for source, target in zip(source_indices, target_indices)
                    ],
                    "n_seeds_planned": planned,
                }
            )
            graphs[threshold][dataset] = frame
    return (graphs, pd.DataFrame(audit_rows))


def internal_threshold_sensitivity_summary(
    bundle: GliomaBundle,
    threshold_graphs: Mapping[float, Mapping[str, pd.DataFrame]],
    phase: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise edge retention and phased-reference recovery by threshold."""
    summary_rows, detail_rows = ([], [])
    for threshold, graphs in sorted(threshold_graphs.items()):
        n_applicable = n_direct = n_edges = 0
        for dataset, graph in graphs.items():
            n_edges += len(graph_edge_set(graph, mode="directed"))
            detail, _, _ = score_edges_against_phased_relations(bundle, graph, dataset)
            if detail.empty or "phase" not in detail:
                continue
            part = detail.loc[detail["phase"].eq(phase)].copy()
            n_applicable += len(part)
            n_direct += int(part["status"].eq("direta_sentido_certo").sum())
            if not part.empty:
                part.insert(0, "dataset", dataset)
                part.insert(0, "internal_threshold", float(threshold))
                detail_rows.append(part)
        summary_rows.append(
            {
                "internal_threshold": float(threshold),
                "n_datasets_available": len(graphs),
                "n_consensus_edges": n_edges,
                "n_reference_applicable": n_applicable,
                "n_reference_recovered_direct": n_direct,
                "pct_reference_recovered_direct": 100 * n_direct / n_applicable
                if n_applicable
                else np.nan,
            }
        )
    details = (
        pd.DataFrame([row for frame in detail_rows for row in frame.to_dict("records")])
        if detail_rows
        else pd.DataFrame()
    )
    return (pd.DataFrame(summary_rows), details)


def internal_threshold_strict_sensitivity_summary(
    bundle: GliomaBundle, threshold_graphs: Mapping[float, Mapping[str, pd.DataFrame]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise retention and strict-reference recall by internal threshold."""
    summary_rows, detail_rows = ([], [])
    for threshold, graphs in sorted(threshold_graphs.items()):
        recovery, detail = strict_reference_recovery_summary(bundle, graphs)
        n_applicable = int(recovery["n_reference_applicable"].sum())
        n_recovered = int(recovery["n_reference_recovered_direct"].sum())
        n_edges = sum((len(graph_edge_set(graph, mode="directed")) for graph in graphs.values()))
        summary_rows.append(
            {
                "internal_threshold": float(threshold),
                "n_datasets_available": len(graphs),
                "n_consensus_edges": n_edges,
                "n_reference_applicable": n_applicable,
                "n_reference_recovered_direct": n_recovered,
                "pct_reference_recovered_direct": 100 * n_recovered / n_applicable
                if n_applicable
                else np.nan,
            }
        )
        if not detail.empty:
            detail = detail.copy()
            detail.insert(0, "internal_threshold", float(threshold))
            detail_rows.append(detail)
    details = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    return (pd.DataFrame(summary_rows), details)


def internal_threshold_stable_edges(
    threshold_graphs: Mapping[float, Mapping[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Edges retained at every internal threshold available for each dataset."""
    thresholds = tuple(sorted(threshold_graphs))
    if not thresholds:
        return pd.DataFrame(columns=["dataset", "source", "target"])
    common_datasets = set.intersection(
        *(set(threshold_graphs[threshold]) for threshold in thresholds)
    )
    rows = []
    for dataset in sorted(common_datasets):
        edge_sets = [
            graph_edge_set(threshold_graphs[threshold][dataset], mode="directed")
            for threshold in thresholds
        ]
        for source, target in sorted(set.intersection(*edge_sets)):
            rows.append(
                {
                    "dataset": dataset,
                    "source": source,
                    "target": target,
                    "min_internal_threshold": max(thresholds),
                }
            )
    return pd.DataFrame(rows)


def build_configured_ensembles(
    bundle: GliomaBundle,
    ensemble_specs: Mapping[str, Mapping[str, object]],
    *,
    min_rate: float = 0.8,
    calibration_models: Iterable[str] | None = None,
    alpha: float = 1.0,
    threshold: float = 0.5,
) -> tuple[dict[str, dict[str, pd.DataFrame]], pd.DataFrame, pd.DataFrame]:
    """Build named, configurable ensembles for all TCGA datasets.

    Each specification needs ``models`` and ``method``. Supported methods are
    ``intersection``, ``dagbag_uniform`` and ``dagbag_dawid_skene``.  The
    intersection is strictly oriented: every member must contain the same
    arrow. Dawid--Skene competence is calibrated using all available models,
    then restricted to the requested voters.
    """
    allowed_methods = {"intersection", "dagbag_uniform", "dagbag_dawid_skene"}
    calibration_requested = resolve_models(
        calibration_models if calibration_models is not None else bundle.models
    )
    availability = bundle.availability.set_index(["dataset", "model"])["available"]
    graphs: dict[str, dict[str, pd.DataFrame]] = {label: {} for label in ensemble_specs}
    audit_rows: list[dict] = []
    competence_rows: list[dict] = []
    parsed_specs = {}
    for label, spec in ensemble_specs.items():
        if "models" not in spec or "method" not in spec:
            raise ValueError(f"Ensemble {label!r} needs 'models' and 'method'.")
        members = resolve_models(spec["models"])
        method = str(spec["method"])
        if len(members) < 2:
            raise ValueError(f"Ensemble {label!r} needs at least two models.")
        if method not in allowed_methods:
            raise ValueError(
                f"Unknown method {method!r} for {label!r}; choose from {sorted(allowed_methods)}."
            )
        parsed_specs[label] = (members, method)
    for dataset in bundle.datasets:
        calibration = tuple(
            (
                model
                for model in calibration_requested
                if bool(availability.get((dataset, model), False))
            )
        )
        tasks, all_votes = (
            _soft_votes(bundle, dataset, calibration) if calibration else ([], np.empty((0, 0, 3)))
        )
        prior = skills = None
        if len(calibration) >= 3:
            prior, skills, _ = _em_onecoin(all_votes)
            competence_rows.extend(
                (
                    {"dataset": dataset, "model": model, "accuracy": float(skill)}
                    for model, skill in zip(calibration, skills)
                )
            )
        for label, (members, method) in parsed_specs.items():
            missing = [model for model in members if model not in calibration]
            status, reason = ("ready", "")
            if missing:
                graph = _empty_aggregated_graph()
                status, reason = ("unavailable", f"missing finished models: {missing}")
            elif method == "intersection":
                graph = _intersection_graph(
                    bundle, dataset, members, min_rate, require_unanimous_direction=True
                )
            elif method == "dagbag_dawid_skene" and prior is None:
                graph = _empty_aggregated_graph()
                status, reason = (
                    "unavailable",
                    "Dawid-Skene needs >=3 calibration models",
                )
            else:
                indices = [calibration.index(model) for model in members]
                if method == "dagbag_uniform":
                    posterior = all_votes[:, indices, :].mean(axis=1)
                else:
                    posterior = _posterior_for_workers(
                        all_votes[:, indices, :], prior, skills[indices]
                    )
                probabilities = {}
                for (a, b), task_posterior in zip(tasks, posterior):
                    probabilities[a, b] = float(task_posterior[0])
                    probabilities[b, a] = float(task_posterior[1])
                graph = _dagbag(probabilities, alpha=alpha, threshold=threshold)
                graph = graph[graph["status"].eq("kept")].copy()
                graph["orientation"] = "directed"
                graph["rate"] = graph["edge_probability"]
                graph["dataset"] = dataset
                graph["combination"] = " + ".join(members)
            graphs[label][dataset] = graph
            audit_rows.append(
                {
                    "ensemble": label,
                    "dataset": dataset,
                    "method": method,
                    "models": members,
                    "status": status,
                    "reason": reason,
                    "n_edges": int(len(graph)),
                }
            )
    return (graphs, pd.DataFrame(audit_rows), pd.DataFrame(competence_rows))


def graph_edge_set(
    graph: pd.DataFrame, mode: str = "directed"
) -> set[tuple[str, str] | frozenset[str]]:
    """Canonical edge set from any individual-model or ensemble graph."""
    if mode not in {"directed", "skeleton"}:
        raise ValueError("mode must be 'directed' or 'skeleton'")
    if graph.empty:
        return set()
    frame = graph
    if "status" in frame:
        frame = frame[frame["status"].eq("kept")]
    if mode == "directed":
        if "orientation" in frame:
            frame = frame[frame["orientation"].eq("directed")]
        return set(zip(frame["source"], frame["target"]))
    return {frozenset((row.source, row.target)) for row in frame.itertuples(index=False)}


def compare_graph_collections(
    anchor_graphs: Mapping[str, pd.DataFrame],
    candidate_graphs: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    datasets: Iterable[str] = DATASETS,
    mode: str = "directed",
) -> pd.DataFrame:
    """Agreement of several graph collections against one anchor collection."""
    rows = []
    for dataset in datasets:
        anchor = graph_edge_set(anchor_graphs[dataset], mode=mode)
        for candidate, graphs in candidate_graphs.items():
            other = graph_edge_set(graphs[dataset], mode=mode)
            common = anchor & other
            union = anchor | other
            reversed_edges = set()
            if mode == "directed":
                reversed_edges = {
                    (source, target)
                    for source, target in anchor - common
                    if (target, source) in other
                }
            rows.append(
                {
                    "dataset": dataset,
                    "candidate": candidate,
                    "mode": mode,
                    "n_anchor_edges": len(anchor),
                    "n_candidate_edges": len(other),
                    "n_common_edges": len(common),
                    "n_reversed_edges": len(reversed_edges),
                    "anchor_supported_pct": 100 * len(common) / len(anchor) if anchor else np.nan,
                    "candidate_shared_pct": 100 * len(common) / len(other) if other else np.nan,
                    "jaccard_pct": 100 * len(common) / len(union) if union else np.nan,
                }
            )
    return pd.DataFrame(rows)


def anchor_edge_support_catalog(
    anchor_graphs: Mapping[str, pd.DataFrame],
    supporter_graphs: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    datasets: Iterable[str] = DATASETS,
    mode: str = "directed",
) -> pd.DataFrame:
    """One row per anchor edge, with exact and reversed model support."""
    rows = []
    for dataset in datasets:
        anchor = graph_edge_set(anchor_graphs[dataset], mode=mode)
        supporter_sets = {
            label: graph_edge_set(graphs[dataset], mode=mode)
            for label, graphs in supporter_graphs.items()
        }
        for edge in sorted(anchor, key=lambda item: tuple(sorted(item))):
            if mode == "directed":
                source, target = edge
                exact = tuple((label for label, edges in supporter_sets.items() if edge in edges))
                reversed_support = tuple(
                    (label for label, edges in supporter_sets.items() if (target, source) in edges)
                )
            else:
                source, target = sorted(edge)
                exact = tuple((label for label, edges in supporter_sets.items() if edge in edges))
                reversed_support = tuple()
            rows.append(
                {
                    "dataset": dataset,
                    "source": source,
                    "target": target,
                    "mode": mode,
                    "n_supporting_models": len(exact),
                    "supporting_models": exact,
                    "n_reversed_models": len(reversed_support),
                    "reversed_models": reversed_support,
                }
            )
    return pd.DataFrame(rows)


def plot_overlap_heatmap(
    agreement: pd.DataFrame,
    metric: str = "anchor_supported_pct",
    title: str | None = None,
):
    """Dataset-by-candidate heatmap for a configurable overlap metric."""
    import matplotlib.pyplot as plt

    allowed = {"anchor_supported_pct", "candidate_shared_pct", "jaccard_pct"}
    if metric not in allowed:
        raise ValueError(f"metric must be one of {sorted(allowed)}")
    matrix = agreement.pivot(index="dataset", columns="candidate", values=metric)
    order = [dataset for dataset in DATASETS if dataset in matrix.index]
    matrix = matrix.reindex(order)
    height = max(5.0, 0.45 * len(matrix) + 1.7)
    width = max(7.0, 1.25 * len(matrix.columns) + 3.0)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=100, cmap="Purples", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels([dataset_display_name(dataset) for dataset in matrix.index])
    for row in range(len(matrix)):
        for column in range(len(matrix.columns)):
            value = matrix.iat[row, column]
            text = "—" if pd.isna(value) else f"{value:.0f}%"
            ax.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if pd.notna(value) and value >= 60 else "black",
            )
    labels = {
        "anchor_supported_pct": "Edges do PRISM apoiadas (%)",
        "candidate_shared_pct": "Edges do candidato partilhadas (%)",
        "jaccard_pct": "Jaccard orientado (%)",
    }
    ax.set_xlabel("Modelo ou ensemble")
    ax.set_ylabel("Dataset")
    ax.set_title(title or labels[metric])
    fig.colorbar(image, ax=ax, label=labels[metric])
    return (fig, ax, matrix)


def plot_support_distribution(
    catalog: pd.DataFrame, *, title: str = "Suporte externo das edges do PRISM"
):
    """Stacked counts of anchor edges supported by zero or more models."""
    import matplotlib.pyplot as plt

    if catalog.empty:
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.text(0.5, 0.5, "Nenhuma edge disponível", ha="center", va="center")
        ax.axis("off")
        return (fig, ax, pd.DataFrame())
    counts = (
        catalog.groupby(["dataset", "n_supporting_models"])
        .size()
        .unstack(fill_value=0)
        .reindex([dataset for dataset in DATASETS if dataset in set(catalog["dataset"])])
    )
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    bottom = np.zeros(len(counts))
    palette = plt.cm.Purples(np.linspace(0.25, 0.95, max(len(counts.columns), 2)))
    for color, support in zip(palette, counts.columns):
        values = counts[support].to_numpy()
        ax.barh(
            range(len(counts)),
            values,
            left=bottom,
            color=color,
            label=f"{support} modelos",
        )
        bottom += values
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels([dataset_display_name(dataset) for dataset in counts.index])
    ax.invert_yaxis()
    ax.set_xlabel("Número de edges do PRISM")
    ax.set_ylabel("Dataset")
    ax.set_title(title)
    ax.legend(title="Suporte exato", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    return (fig, ax, counts)
