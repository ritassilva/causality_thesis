"""Split TCGA modality datasets by WHO 2021 subtype."""

import csv
import os
from pathlib import Path

SOURCE_DIR = None
SOURCE = None
ANNOTATIONS = None
OUT = None

DATASETS = {
    "mut_expr70": lambda c: c.endswith("_mut") or c.endswith("_expr_log2tpm"),
    "mut35": lambda c: c.endswith("_mut"),
    "expr35": lambda c: c.endswith("_expr_log2tpm"),
}
SUBTYPES = ("glioblastoma", "astrocytoma", "oligodendroglioma")


def read_tsv(path):
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    header = rows[0][1:] if rows[0][0] == "" else rows[0]
    return header, {row[0]: row[1:] for row in rows[1:]}


def subtype_of(gbm, astro):
    if gbm == "1" and astro == "0":
        return "glioblastoma"
    if gbm == "0" and astro == "1":
        return "astrocytoma"
    if gbm == "0" and astro == "0":
        return "oligodendroglioma"
    raise ValueError(f"invalid GBM/ASTRO labels: {gbm}/{astro}")


def main():
    global SOURCE_DIR, SOURCE, ANNOTATIONS, OUT
    if not os.environ.get("TCGA_COHORT_DIR") or not os.environ.get("TCGA_SUBTYPE_DIR"):
        raise ValueError("Set TCGA_COHORT_DIR and TCGA_SUBTYPE_DIR, or use runs/prepare_tcga/prepare.py --work-dir /path/to/tcga-work")
    SOURCE_DIR = Path(os.environ["TCGA_COHORT_DIR"])
    SOURCE = SOURCE_DIR / "06_causal_input_mutation_plus_log2_tpm.tsv"
    ANNOTATIONS = SOURCE_DIR / "07_participant_annotations.tsv"
    OUT = Path(os.environ["TCGA_SUBTYPE_DIR"])
    columns, data = read_tsv(SOURCE)
    ann_columns, annotations = read_tsv(ANNOTATIONS)
    gbm, astro = ann_columns.index("GBM"), ann_columns.index("ASTRO")

    missing = set(data) ^ set(annotations)
    if missing:
        raise ValueError(f"participants not in both files: {sorted(missing)[:10]}")
    subtype = {pid: subtype_of(a[gbm], a[astro]) for pid, a in annotations.items()}

    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, keep in DATASETS.items():
        idx = [i for i, c in enumerate(columns) if keep(c)]
        selected = [columns[i] for i in idx]
        groups = {"all": list(data)}
        groups.update({s: [p for p in data if subtype[p] == s] for s in SUBTYPES})
        for group, participants in groups.items():
            path = OUT / f"{name}_{group}.tsv"
            with open(path, "w", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow([""] + selected)
                for pid in participants:
                    writer.writerow([pid] + [data[pid][i] for i in idx])
            constant = [
                c for c, i in zip(selected, idx) if len({data[p][i] for p in participants}) == 1
            ]
            summary.append(
                [
                    path.name,
                    name,
                    group,
                    len(participants),
                    len(selected),
                    len(constant),
                    ",".join(constant),
                ]
            )

    with open(OUT / "summary.tsv", "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "file",
                "dataset",
                "subtype",
                "n_participants",
                "n_columns",
                "n_constant_columns",
                "constant_columns",
            ]
        )
        writer.writerows(summary)
    for row in summary:
        print("\t".join(map(str, row[:6])), row[6])


if __name__ == "__main__":
    main()
