"""Data preparation, statistical summaries and plotting inputs for the notebooks."""

from __future__ import annotations

import json
from functools import lru_cache
import sys
from pathlib import Path
from typing import Iterable

import final_visualization as fv
from condition_data import load_results, load_sample
import numpy as np
import pandas as pd
from Wilcoxon_test.wilcoxon_test import build_wilcoxon_rank, paired_wilcoxon as wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from metrics import order_fnr
publication = ROOT
OUTPUT = ROOT / "results/benchmark/figures"
ALIGNED_BIOLOGICAL = ROOT / "results/benchmark"
ALIGNED_PRIORS = ROOT / "results/prism/priors/aligned"
DATASET_SEEDS = (19, 47, 50, 52, 71, 75, 77, 102, 125, 158, 222, 237, 308, 402, 603)
TEST_FAMILIES = (
    "20nodes_EB_obsconf0_latconf0_Non_Gaussian_Mixed",
    "20nodes_ModMultiOmics_obsconf0_latconf0_Laplace_Mixed",
    "20nodes_ModOncogene_obsconf0_latconf0_Gaussian_Mixed",
)
OOD_FAMILIES = (
    "20nodes_EB_obsconf0_latconf0_Gaussian_Linear",
    "20nodes_EB_obsconf0_latconf0_Gaussian_Mixed",
    "20nodes_EB_obsconf0_latconf0_NonGaussian_Linear",
)
SIX_FAMILIES = (*OOD_FAMILIES, *TEST_FAMILIES)
FAMILY_LABELS = {
    TEST_FAMILIES[0]: "ER, non Gaussian mixed",
    TEST_FAMILIES[1]: "MultiOmics, Laplace mixed",
    TEST_FAMILIES[2]: "Oncogene, Laplace mixed",
    OOD_FAMILIES[0]: "ER, Gaussian linear",
    OOD_FAMILIES[1]: "ER, Gaussian mixed",
    OOD_FAMILIES[2]: "ER, non Gaussian linear",
}
MODEL_LABELS = {
    "DAGMA_Linear": "DAGMA Linear",
    "DAGMA_Nonlinear": "DAGMA Nonlinear",
    "PRISM": "PRISM",
    "GraNDAG": "GraN-DAG",
}


def load_historical_benchmark():
    """Load archived graph sizes; aligned sources supersede historical 20-node fits."""
    frame = load_results("benchmark")
    return frame.loc[frame.n_nodes.ne(20)].copy()


def model_label(value: object) -> str:
    return MODEL_LABELS.get(str(value), str(value).replace("_", " "))


def ensure_output_dirs(topic=None) -> None:
    """Keep exported figures beside their experiment results."""
    global OUTPUT
    if topic is not None:
        OUTPUT = ROOT / "results" / topic / "figures"
    OUTPUT.mkdir(parents=True, exist_ok=True)


def result_view(frame: pd.DataFrame) -> pd.DataFrame:
    """Return per-fit results in the public schema."""
    from condition_data import result_table
    return result_table(frame)


def prepare_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "timed_out" not in out:
        out["timed_out"] = out["status"].eq("timeout")
    else:
        out["timed_out"] = out["timed_out"].astype(str).str.lower().isin({"true", "1", "yes"})
    out["finished"] = out["status"].eq("finished") & ~out["timed_out"]
    out = fv._add_dataset_attributes(out)
    out = fv.add_fbeta(out)
    for column in set(fv.METRIC_INFO) & set(out):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def visualization_data(frame: pd.DataFrame) -> fv.FinalVisualizationData:
    empty_rank = pd.DataFrame(columns=["dataset", "metric", "model", "rank"])
    empty_coverage = pd.DataFrame(
        columns=["dataset", "metric", "model", "successful_runs", "expected_runs"]
    )
    empty_friedman = pd.DataFrame(columns=["dataset", "metric", "model", "mean", "std"])
    return fv.FinalVisualizationData(
        metrics=frame.copy(),
        ranks_by_dataset=empty_rank,
        ranks_overall=empty_rank.copy(),
        coverage_by_dataset=empty_coverage,
        friedman_by_dataset=empty_friedman,
        paths={},
    )


def load_aligned_sources() -> dict[str, object]:
    benchmark_all = prepare_metrics(load_results("benchmark", n_variables=20))
    benchmark_biological = benchmark_all.loc[
        benchmark_all.dataset.isin(TEST_FAMILIES[1:])
    ].copy()
    benchmark = benchmark_all.loc[
        ~benchmark_all.dataset.isin(TEST_FAMILIES[1:])
    ].copy()
    complete = prepare_metrics(load_results("evaluation"))
    six_untuned = complete.loc[
        complete["arm"].eq("untuned") & complete["dataset"].isin(SIX_FAMILIES)
    ].copy()
    return {
        "benchmark": benchmark,
        "complete": complete,
        "six_untuned": six_untuned,
        "benchmark_biological": benchmark_biological,
        "benchmark_biological_source": ALIGNED_BIOLOGICAL,
        "benchmark_data": visualization_data(benchmark),
        "benchmark_with_biological": benchmark_all,
        "benchmark_with_biological_data": visualization_data(benchmark_all),
        "six_data": visualization_data(six_untuned),
    }


def save_pdf(fig, filename: str) -> Path:
    ensure_output_dirs()
    target = OUTPUT / filename
    fig.savefig(target, format="pdf", bbox_inches="tight")
    return target


def seed_means(
    frame: pd.DataFrame, metric: str, *, arms: Iterable[str] | None = None
) -> pd.DataFrame:
    selected = frame.loc[frame["finished"]].copy()
    if arms is not None:
        selected = selected[selected["arm"].isin(list(arms))]
    return selected.groupby(
        [column for column in ("arm", "dataset", "dataset_seed", "model") if column in selected],
        observed=True,
        as_index=False,
        dropna=False,
    )[metric].mean()


def mean_sd(values: pd.Series, digits: int = 3) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return f"{numeric.mean():.{digits}f} ± {numeric.std(ddof=1):.{digits}f}"


def prism_performance(complete: pd.DataFrame) -> pd.DataFrame:
    selected = seed_means(complete.loc[complete["model"].eq("PRISM")], "fbeta", arms=("tuned",))
    rows = []
    for family in (*TEST_FAMILIES, *OOD_FAMILIES):
        scores = selected.loc[selected["dataset"].eq(family), "fbeta"]
        order_values = []
        for seed in DATASET_SEEDS:
            prior_path = ALIGNED_PRIORS / f"{family}__seed_{seed}.npz"
            with load_sample("evaluation", family, seed) as data:
                B = np.asarray(data["true_adj"]) != 0
            with np.load(prior_path, allow_pickle=False) as prior:
                order = np.asarray(prior["order"], dtype=int)
            order_values.append(order_fnr(B, order))
        rows.append(
            {
                "Split": "Test" if family in TEST_FAMILIES else "OOD",
                "Dataset family": FAMILY_LABELS[family],
                "F0.5": mean_sd(scores),
                "Order FNR": mean_sd(pd.Series(order_values)),
                "f05_mean": float(scores.mean()),
                "f05_sd": float(scores.std(ddof=1)),
                "order_fnr_mean": float(np.mean(order_values)),
                "order_fnr_sd": float(np.std(order_values, ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def tuned_test_ood(complete: pd.DataFrame) -> pd.DataFrame:
    cells = seed_means(complete, "fbeta", arms=("tuned",))
    rows = []
    for model, model_rows in cells.groupby("model", observed=True):
        row = {"Method": model_label(model)}
        for label, families in (("Test", TEST_FAMILIES), ("OOD", OOD_FAMILIES)):
            family_cells = model_rows[model_rows["dataset"].isin(families)]
            macro = family_cells.groupby("dataset_seed")["fbeta"].agg(["mean", "size"])
            macro = macro.loc[macro["size"].eq(len(families)), "mean"]
            row[f"{label} F0.5"] = mean_sd(macro)
            row[f"_{label.lower()}_mean"] = float(macro.mean())
            row[f"_{label.lower()}_sd"] = float(macro.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("_test_mean", ascending=False).reset_index(drop=True)


def _holm(raw: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    ordered = sorted(raw, key=raw.get)
    adjusted: dict[tuple[str, str], float] = {}
    running = 0.0
    total = len(ordered)
    for index, key in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * raw[key]))
        adjusted[key] = running
    return adjusted


def significant_win_ranks(blocks: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Competition ranks from paired Wilcoxon tests with Holm correction."""
    pivot = blocks.pivot(index="dataset_seed", columns="model", values=metric)
    models = sorted(pivot.columns)
    raw: dict[tuple[str, str], float] = {}
    direction: dict[tuple[str, str], int] = {}
    for left_index, left in enumerate(models):
        for right in models[left_index + 1 :]:
            paired = pivot[[left, right]].dropna()
            delta = paired[left] - paired[right]
            try:
                pvalue = float(wilcoxon(delta).pvalue)
            except ValueError:
                pvalue = 1.0
            key = (left, right)
            raw[key] = pvalue
            direction[key] = int(np.sign(delta.mean()))
    adjusted = _holm(raw)
    pairwise = pd.DataFrame(
        [
            {
                "model_1": left,
                "model_2": right,
                "significant_holm": adjusted[left, right] < 0.05,
                "better_model_by_mean": left if direction[left, right] >= 0 else right,
            }
            for left, right in raw
        ]
    )
    means = {model: float(pivot[model].mean()) for model in models}
    ranked = build_wilcoxon_rank(pairwise, models=models, model_means=means, higher_is_better=True)
    return ranked.rename(columns={"mean_metric": "mean"})


def ranks_by_family(complete: pd.DataFrame, metric: str = "fbeta") -> pd.DataFrame:
    blocks = seed_means(complete, metric, arms=("tuned",))
    tables = []
    for family in (*TEST_FAMILIES, *OOD_FAMILIES):
        ranked = significant_win_ranks(blocks[blocks["dataset"].eq(family)], metric)
        ranked["dataset"] = family
        tables.append(ranked)
    long = pd.concat(tables, ignore_index=True)
    wide = long.pivot(index="model", columns="dataset", values="rank")
    wide = wide[[*TEST_FAMILIES, *OOD_FAMILIES]]
    wide["Median"] = wide.median(axis=1)
    wide["Mean"] = wide.mean(axis=1)
    wide = wide.sort_values(["Median", "Mean"]).drop(columns="Mean")
    wide.index = [model_label(model) for model in wide.index]
    return wide.reset_index(names="Method")


def ranks_by_scope(complete: pd.DataFrame, metric: str = "fbeta") -> pd.DataFrame:
    """Wilcoxon-Holm ranks over Test, OOD, and their combined blocks."""
    blocks = seed_means(complete, metric, arms=("tuned",))
    scope_frames = []
    for scope, families in (
        ("OOD", OOD_FAMILIES),
        ("Test", TEST_FAMILIES),
        ("Test + OOD", SIX_FAMILIES),
    ):
        selected = blocks.loc[blocks["dataset"].isin(families)].copy()
        selected["dataset_seed"] = (
            selected["dataset"].astype(str) + "__" + selected["dataset_seed"].astype(str)
        )
        ranked = significant_win_ranks(selected[["dataset_seed", "model", metric]], metric)[
            ["model", "rank"]
        ]
        ranked["scope"] = scope
        scope_frames.append(ranked)
    long = pd.concat(scope_frames, ignore_index=True)
    wide = long.pivot(index="model", columns="scope", values="rank")
    wide = wide[["OOD", "Test", "Test + OOD"]]
    wide = wide.sort_values(["Test + OOD", "Test", "OOD"])
    wide.index = [model_label(model) for model in wide.index]
    return wide.reset_index(names="Method")


def benchmark_ranks(benchmark: pd.DataFrame, biological: pd.DataFrame) -> pd.DataFrame:
    """Per-condition ranks for the corrected benchmark; HighGaussian Linear is absent."""
    bio = biological[biological["dataset"].isin(TEST_FAMILIES[1:])]
    combined = pd.concat([benchmark, bio], ignore_index=True, sort=False)
    blocks = (
        combined.loc[combined["finished"]]
        .groupby(["dataset", "dataset_seed", "model"], observed=True, as_index=False)["f1"]
        .mean()
    )
    tables = []
    for family, family_rows in blocks.groupby("dataset", observed=True):
        raw_family = combined.loc[combined["dataset"].eq(family)]
        eligible = raw_family.groupby("model")["finished"].all()
        family_rows = family_rows.loc[family_rows["model"].isin(eligible[eligible].index)]
        ranked = significant_win_ranks(family_rows, "f1")
        ranked["dataset"] = family
        tables.append(ranked)
    long = pd.concat(tables, ignore_index=True)
    wide = long.pivot(index="model", columns="dataset", values="rank")
    wide["Median"] = wide.median(axis=1)
    wide["Mean"] = wide.mean(axis=1)
    wide = wide.sort_values(["Median", "Mean"]).drop(columns="Mean")
    wide.index = [model_label(model) for model in wide.index]
    return wide.reset_index(names="Method")


def prior_quality(scores: pd.DataFrame, *, include_extra=False) -> pd.DataFrame:
    """AUPRC across search seeds, with equal weight for each family."""
    definitions = [
        ("GLasso", "correlation", None, "GLasso"),
        ("Hessian", "raw", None, "Hessian, raw"),
        ("Hessian", "standardized", None, "Hessian, standardized"),
        ("Combined", "raw", .5, "Combined, raw"),
        ("Combined", "standardized", .5, "Combined, standardized"),
        ("Binary", "raw", None, "Binary, raw"),
        ("Binary", "standardized", None, "Binary, standardized"),
        ("Subsampling", "raw", None, "Subsampling, raw"),
        ("Subsampling", "standardized", None, "Subsampling, standardized"),
    ]
    if include_extra:
        definitions += [("Combined", rep, alpha, f"Combined, {rep}, alpha={alpha:.2f}")
                        for rep in ("raw", "standardized") for alpha in (.25, .75)]
    columns = dict(zip(TEST_FAMILIES, ["ER, non-Gaussian mixed", "MultiOmics", "Oncogene"]))
    rows = []
    for score, representation, alpha, label in definitions:
        selected = scores.loc[scores.score.eq(score) & scores.representation.eq(representation)]
        selected = selected.loc[selected.alpha.isna() if alpha is None else selected.alpha.eq(alpha)]
        row = {"Prior": label}
        for family, column in columns.items():
            row[column] = mean_sd(selected.loc[selected.dataset.eq(family), "auprc"])
        macro = selected.pivot(index="dataset_seed", columns="dataset", values="auprc").mean(axis=1)
        row["Macro"] = mean_sd(macro)
        rows.append(row)
    return pd.DataFrame(rows)


def aligned_ablation_table() -> pd.DataFrame:
    """Ablation summary using completed fits only; unsuccessful fits are excluded."""
    fits = load_results("ablation").assign(family=lambda frame: frame.dataset)
    completed = fits.loc[fits["status"].eq("finished")].copy()
    completed["fbeta"] = pd.to_numeric(completed["fbeta"], errors="coerce")
    completed = completed.dropna(subset=["fbeta"])
    cell = (
        completed.groupby(["condition_id", "family", "dataset_seed"], observed=True)["fbeta"]
        .mean()
        .reset_index()
    )
    family_count = cell.groupby(["condition_id", "dataset_seed"], observed=True)["family"].nunique()
    macro = (
        cell.groupby(["condition_id", "dataset_seed"], observed=True)["fbeta"]
        .mean()
        .loc[family_count.eq(len(TEST_FAMILIES))]
        .rename("fbeta")
        .reset_index()
    )
    summary = macro.groupby("condition_id").fbeta.agg(fbeta="mean", fbeta_sd="std")
    macro_by_condition = {
        condition: group.set_index("dataset_seed")["fbeta"]
        for condition, group in macro.groupby("condition_id", observed=True)
    }
    order = (
        "full",
        "glasso_only",
        "hessian_only",
        "glasso_no_direction",
        "hessian_no_direction",
        "permuted_existence",
        "no_existence",
        "no_direction",
        "permuted_direction",
        "reversed_direction",
        "no_prior",
        "matched_constant",
        "permuted_prior",
        "oracle",
    )
    labels = {
        "full": "Full model",
        "glasso_only": "GLasso only",
        "hessian_only": "Hessian only",
        "glasso_no_direction": "GLasso without direction",
        "hessian_no_direction": "Hessian without direction",
        "permuted_existence": "Permuted existence",
        "no_existence": "No existence prior",
        "no_direction": "No directional prior",
        "permuted_direction": "Permuted direction",
        "reversed_direction": "Reversed direction",
        "no_prior": "No prior",
        "matched_constant": "Matched constant penalty",
        "permuted_prior": "Permuted prior",
        "oracle": "Oracle prior",
    }

    def pvalue(value: float) -> str:
        return "$<0.001$" if value < 0.001 else f"${value:.3f}$"

    rng = np.random.default_rng(20260920)
    comparisons: dict[str, dict[str, float]] = {}
    raw_pvalues: dict[str, float] = {}
    full_values = macro_by_condition["full"]
    for condition in order[1:]:
        paired = pd.concat(
            [
                full_values.rename("full"),
                macro_by_condition[condition].rename("condition"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        delta = (paired["full"] - paired["condition"]).to_numpy(dtype=float)
        draws = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
        try:
            raw_pvalue = float(wilcoxon(delta, alternative="two-sided").pvalue)
        except ValueError:
            raw_pvalue = 1.0
        raw_pvalues[condition] = raw_pvalue
        comparisons[condition] = {
            "loss": float(delta.mean()),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
            "n": int(len(delta)),
        }
    adjusted: dict[str, float] = {}
    running = 0.0
    ordered_pvalues = sorted(raw_pvalues, key=raw_pvalues.get)
    for index, condition in enumerate(ordered_pvalues):
        running = max(running, min(1.0, (len(ordered_pvalues) - index) * raw_pvalues[condition]))
        adjusted[condition] = running
    rows = []
    for condition in order:
        condition_summary = summary.loc[condition]
        row = {
            "Condition": labels[condition],
            "$F_{0.5}$": f"${condition_summary['fbeta']:.3f} \\pm {condition_summary['fbeta_sd']:.3f}$",
        }
        if condition == "full":
            row.update({"Loss": "--", "95\\% CI": "--", "Holm $p$": "--"})
        else:
            difference = comparisons[condition]
            row.update(
                {
                    "Loss": f"${difference['loss']:.3f}$",
                    "95\\% CI": f"$[{difference['ci_low']:.3f},\\,{difference['ci_high']:.3f}]$",
                    "Holm $p$": pvalue(adjusted[condition]),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)




@lru_cache(maxsize=1)
def raw_sortability() -> pd.DataFrame:
    """Calculate sortability on the benchmark's 2000-row Gaussian-linear samples."""
    from varsortability import calculate_varsortability

    family = "20nodes_EB_obsconf0_latconf0_Gaussian_Linear"
    rows = []
    measurements = load_results("benchmark")
    seeds = sorted(measurements.loc[measurements.dataset.eq(family), "dataset_seed"].unique())
    for seed in seeds:
        with load_sample("benchmark", family, int(seed)) as sample:
            observations = sample["X_raw"]
            values = calculate_varsortability(observations, sample["true_adj"])
            rows.append(dict(dataset_seed=seed, dataset=family, **values))
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def r2_sort_regress() -> pd.DataFrame:
    """Fit the sorting baseline to the same 2000-row benchmark instances."""
    from varsortability import calculate_r2_sort_regress_table

    return calculate_r2_sort_regress_table(raw_sortability(), project_root=ROOT)
