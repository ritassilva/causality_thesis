"""Create the publication convergence figure for PRISM from saved traces."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from final_visualization import DEFAULT_FONT_SIZE, OKABE_ITO, apply_style

ALIGNED_FITS = ROOT / "results/prism/convergence/fits"
FIGURES = ROOT / "results/prism/convergence/figures"
FAMILY_LABELS = {
    "20nodes_EB_obsconf0_latconf0_Non_Gaussian_Mixed": "ER, non Gaussian mixed",
    "20nodes_ModMultiOmics_obsconf0_latconf0_Laplace_Mixed": "MultiOmics",
    "20nodes_ModOncogene_obsconf0_latconf0_Gaussian_Mixed": "Oncogene",
}
FAMILY_COLORS = {
    "20nodes_EB_obsconf0_latconf0_Non_Gaussian_Mixed": OKABE_ITO["blue"],
    "20nodes_ModMultiOmics_obsconf0_latconf0_Laplace_Mixed": OKABE_ITO["orange"],
    "20nodes_ModOncogene_obsconf0_latconf0_Gaussian_Mixed": OKABE_ITO["bluish_green"],
}


def load_aligned_convergence() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the 225 saved PRISM traces from the aligned test evaluation."""
    fit_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    for path in sorted(ALIGNED_FITS.rglob("*.json")):
        payload = json.loads(path.read_text())
        family = payload.get("family")
        if family not in FAMILY_LABELS:
            continue
        convergence = payload.get("convergence") or {}
        steps = convergence.get("steps") or []
        trace = convergence.get("trace") or []
        if not steps or not trace:
            raise ValueError(f"Missing convergence history in {path}")
        dataset_seed = int(payload["dataset_seed"])
        model_seed = int(payload["model_seed"])
        fit = f"{family}__ds{dataset_seed}__ms{model_seed}"
        last_step = steps[-1]
        fit_rows.append(
            {
                "source": str(path.relative_to(ROOT)),
                "fit": fit,
                "family": family,
                "dataset_seed": dataset_seed,
                "model_seed": model_seed,
                "status": payload.get("status"),
                "runtime_s": payload.get("runtime_seconds"),
                "retries_h_negative": int(payload.get("retries_h_negative", 0)),
                "outer_steps": len(steps),
                "steps_on_budget": sum((step.get("stop") == "budget" for step in steps)),
                "steps_on_tolerance": sum((step.get("stop") == "tolerance" for step in steps)),
                "steps_h_negative": sum((step.get("stop") == "h_negative" for step in steps)),
                "last_step_stop": convergence.get("last_step_stop"),
                "iterations_last_step": last_step.get("iterations"),
                "budget_last_step": last_step.get("budget"),
                "relative_change_last": last_step.get("relative_change"),
                "objective_final": convergence.get("last_step_objective"),
                "h_final": convergence.get("last_step_h"),
                "h_at_s1": convergence.get("h_at_s1"),
                "w_l1": convergence.get("w_l1"),
                "w_frobenius": convergence.get("w_frobenius"),
                "w_max": convergence.get("w_max"),
                "w_above_1e-6": convergence.get("w_above_1e-6"),
            }
        )
        for point in trace:
            trace_rows.append(
                {
                    "fit": fit,
                    "family": family,
                    "dataset_seed": dataset_seed,
                    "model_seed": model_seed,
                    **point,
                }
            )
    fits = pd.DataFrame(fit_rows)
    traces = pd.DataFrame(trace_rows)
    expected = len(FAMILY_LABELS) * 15 * 5
    if len(fits) != expected:
        raise ValueError(f"Expected {expected} aligned test fits, found {len(fits)}")
    if fits.duplicated(["family", "dataset_seed", "model_seed"]).any():
        raise ValueError("Duplicate aligned convergence fits")
    if not fits["status"].eq("finished").all():
        raise ValueError("The aligned convergence collection contains unfinished fits")
    return (fits, traces)


def _prepare_traces(traces: pd.DataFrame) -> pd.DataFrame:
    prepared = traces.copy()
    budgets = np.where(prepared["outer_step"].eq(5), 5000.0, 3125.0)
    prepared["outer_progress"] = prepared["outer_step"] + prepared["iteration"] / budgets
    prepared["objective_start"] = prepared.groupby(["fit", "outer_step"], sort=False)[
        "objective"
    ].transform("first")
    prepared["relative_objective"] = prepared["objective"] / prepared["objective_start"]
    return prepared


def _trajectory_summary(traces: pd.DataFrame, value: str) -> pd.DataFrame:
    return (
        traces.groupby(["family", "outer_step", "iteration"], as_index=False)
        .agg(
            x=("outer_progress", "first"),
            median=(value, "median"),
            q25=(value, lambda series: series.quantile(0.25)),
            q75=(value, lambda series: series.quantile(0.75)),
        )
        .sort_values(["family", "x"])
    )


def convergence_statistics(fits: pd.DataFrame, traces: pd.DataFrame) -> dict[str, float]:
    ordered = traces.sort_values(["fit", "outer_step", "iteration"])
    last_step = ordered.loc[ordered["outer_step"].eq(5)]
    first = last_step.groupby("fit", sort=False).first()
    last = last_step.groupby("fit", sort=False).last()
    objective_change = 100.0 * np.abs(last["objective"] / first["objective"] - 1.0)
    h_reduction = first["h"] / last["h"]
    return {
        "fits": float(len(fits)),
        "finished": float(fits["status"].eq("finished").sum()),
        "tolerance_stops": float(fits["last_step_stop"].eq("tolerance").sum()),
        "budget_stops": float(fits["last_step_stop"].eq("budget").sum()),
        "retries": float(fits["retries_h_negative"].sum()),
        "h_final_median": float(fits["h_final"].median()),
        "h_final_max": float(fits["h_final"].max()),
        "relative_change_median": float(fits["relative_change_last"].median()),
        "relative_change_q95": float(fits["relative_change_last"].quantile(0.95)),
        "relative_change_max": float(fits["relative_change_last"].max()),
        "last_step_objective_change_median_pct": float(objective_change.median()),
        "last_step_h_reduction_median": float(h_reduction.median()),
    }


def plot_prism_convergence(
    traces: pd.DataFrame | None = None,
    fits: pd.DataFrame | None = None,
    *,
    save: bool = True,
) -> tuple[plt.Figure, np.ndarray, dict[str, float]]:
    """Plot objective, acyclicity and final numerical change for aligned test fits."""
    if traces is None or fits is None:
        aligned_fits, aligned_traces = load_aligned_convergence()
        fits = aligned_fits if fits is None else fits
        traces = aligned_traces if traces is None else traces
    apply_style()
    prepared = _prepare_traces(traces)
    objective = _trajectory_summary(prepared, "relative_objective")
    acyclicity = _trajectory_summary(prepared, "h")
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.0, 4.8),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.08]},
        constrained_layout=True,
    )
    for family in FAMILY_LABELS:
        color = FAMILY_COLORS[family]
        block = objective.loc[objective["family"].eq(family)]
        axes[0].plot(block["x"], block["median"], color=color, linewidth=1.9)
        axes[0].fill_between(
            block["x"], block["q25"], block["q75"], color=color, alpha=0.18, linewidth=0
        )
        block = acyclicity.loc[acyclicity["family"].eq(family)].copy()
        for column in ["median", "q25", "q75"]:
            block[column] = block[column].clip(lower=1e-12)
        axes[1].plot(block["x"], block["median"], color=color, linewidth=1.9)
        axes[1].fill_between(
            block["x"], block["q25"], block["q75"], color=color, alpha=0.18, linewidth=0
        )
    for ax in axes[:2]:
        for boundary in range(1, 6):
            ax.axvline(boundary, color="#9CA3AF", linewidth=0.8, linestyle="--")
        ax.set_xlim(0, 6)
        ax.set_xticks(np.arange(0.5, 6.0, 1.0), [str(step) for step in range(1, 7)])
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.65)
    axes[0].axhline(1.0, color="#6B7280", linewidth=0.8, linestyle=":")
    axes[0].set_ylabel("Objective relative to step start")
    axes[0].set_xlabel("Outer step")
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-12, 1)
    axes[1].set_ylabel("Acyclicity constraint $h(W)$")
    axes[1].set_xlabel("Outer step")
    ordered_families = list(FAMILY_LABELS)
    values = [
        fits.loc[fits["family"].eq(family), "relative_change_last"].dropna().to_numpy()
        for family in ordered_families
    ]
    positions = np.arange(1, len(ordered_families) + 1)
    box = axes[2].boxplot(
        values,
        positions=positions,
        widths=0.54,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111827", "linewidth": 1.4},
        whiskerprops={"color": "#4B5563", "linewidth": 1.0},
        capprops={"color": "#4B5563", "linewidth": 1.0},
    )
    random = np.random.default_rng(42)
    for position, family, series, patch in zip(positions, ordered_families, values, box["boxes"]):
        color = FAMILY_COLORS[family]
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.42)
        jitter = random.uniform(-0.16, 0.16, size=len(series))
        axes[2].scatter(
            position + jitter,
            series,
            s=11,
            color=color,
            alpha=0.42,
            edgecolors="none",
            zorder=2,
        )
    axes[2].axhline(
        1e-06,
        color="#111827",
        linewidth=1.0,
        linestyle="--",
        label="Stopping tolerance ($10^{-6}$)",
    )
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Final relative objective change")
    axes[2].set_xticks(
        positions,
        [
            "ER\nnon Gaussian mixed",
            "MultiOmics",
            "Oncogene",
        ],
    )
    axes[2].grid(axis="x", visible=False)
    axes[2].grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.65)
    axes[2].legend(loc="upper right", fontsize=DEFAULT_FONT_SIZE)
    for label, ax in zip(["(a)", "(b)", "(c)"], axes):
        ax.text(
            0.012,
            0.97,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=DEFAULT_FONT_SIZE,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=FAMILY_COLORS[family],
            linewidth=2.2,
            label=FAMILY_LABELS[family],
        )
        for family in ordered_families
    ]
    fig.legend(
        handles=legend_handles,
        loc="outside upper center",
        ncol=3,
        fontsize=DEFAULT_FONT_SIZE,
        handlelength=2.8,
        columnspacing=2.0,
    )
    stats = convergence_statistics(fits, prepared)
    if save:
        FIGURES.mkdir(parents=True, exist_ok=True)
        for suffix in ["png", "pdf"]:
            fig.savefig(
                FIGURES / f"prism_convergence.{suffix}",
                dpi=400,
                bbox_inches="tight",
            )
    return (fig, axes, stats)


