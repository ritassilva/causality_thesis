"""Sachs (cd3cd28, 853 rows, 17-arc gold standard): one row per model, untuned and tuned.

Inputs are the individual measurements in results/sachs/sachs_cd3cd28.csv. Each
arm is 15 model seeds on the same data, so the ± is the standard deviation across
model seeds, not data uncertainty.

F0.5 is computed from the stored precision and recall.

    python analysis/sachs_table.py
"""

from __future__ import annotations

import pandas as pd

METRICS = {
    "fbeta": "F0.5",
    "precision": "Precisão",
    "recall": "Recall",
    "f1": "F1",
    "SHD": "SHD",
}


def load():
    from condition_data import load_results
    frame = load_results("sachs")
    frame["base"] = frame["model"]
    return frame


def build(digits: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = load()
    stats = results.groupby(["model", "arm"])[list(METRICS)].agg(["mean", "std", "size"])
    rows = []
    for (base, arm), row in stats.iterrows():
        record = {"Modelo": base, "Braço": arm, "N": int(row["fbeta", "size"])}
        for metric, label in METRICS.items():
            mean, std = (row[metric, "mean"], row[metric, "std"])
            d = 1 if metric in ("SHD", "n_pred_edges") else digits
            record[label] = f"{mean:.{d}f} ± {std:.{d}f}"
            record[f"_{metric}"] = mean
        rows.append(record)
    long = pd.DataFrame(rows)
    order = (
        long[long["Braço"].eq("tuned")]
        .set_index("Modelo")["_fbeta"]
        .reindex(long["Modelo"].unique())
        .fillna(-1)
        .sort_values(ascending=False)
        .index
    )
    long["Modelo"] = pd.Categorical(long["Modelo"], categories=list(order), ordered=True)
    long = long.sort_values(["Modelo", "Braço"], ascending=[True, False]).reset_index(drop=True)
    wide = long.pivot(index="Modelo", columns="Braço", values="F0.5").reindex(order)
    wide = wide.rename(columns={"untuned": "F0.5 untuned", "tuned": "F0.5 tuned"}).fillna("—")
    return (long, wide)


