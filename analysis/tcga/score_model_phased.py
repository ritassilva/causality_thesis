"""Relações causais conhecidas, separadas pelo nível em que o modelo as pode ver.

Versão do ``score_model.py`` que lê
``references/phased_relations_glioma_rna_final.csv`` e
reporta duas fases em separado:

| fase | nível | grupos | o quê |
|---|---|---|---|
| 1 | mRNA | ``mut_expr`` | IDH1_mut -> PDGFRA_expr (Flavahan et al. 2016) |
| 1 | mRNA | ``cis`` | ATRX, NF1, RB1: a mutação baixa a expressão do próprio gene |
| 1 | mRNA | ``transcricao`` | TP53 -> EGFR, TP53 -> PTEN, NOTCH1 -> EGFR, NOTCH1 -> TCF12 |
| 2 | proteína | ``proteina`` | as outras 7 setas do SIGNOR |

A fase 1 é a avaliação: relações cujo efeito cai no mRNA, que é o que os
dados medem, e que um bom modelo devia poder encontrar. A fase 2 é
exploratória: as setas actuam na actividade ou na estabilidade da proteína,
não se espera que deixem marca no mRNA, e falhá-las não é um erro do modelo.

A forma de avaliar cada seta é a do ``score_model.py`` (directa ou por
caminho dirigido de até ``--max-path`` arestas; setas nos dois sentidos
avaliadas cada uma; errado = sentido contrário, ou aresta expressão ->
mutação), e as funções vêm de lá.

Exemplos (a partir da raiz do repositório, com o ambiente de análise ativo)::

    python analysis/tcga/score_model_phased.py --arm PRISM__tuned
    python analysis/tcga/score_model_phased.py --arm all
    python analysis/tcga/score_model_phased.py --edges grafo.csv --dataset tcga_mut_expr70

Escreve em ``results/tcga/literature_comparison/model_scores_phased/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from compare_literature import (
    DEFAULT_VIEW,
    EXPR,
    FINAL_MODEL,
    MUT,
    REFERENCES,
    RESULTS,
    default_source_file,
    gene_of,
    kind_of,
    load_arm_skeletons,
)
from score_model import (
    DATASETS,
    STATUS_TEXT,
    STATUSES,
    WRONG,
    ModelGraph,
    chance_levels,
    dataset_variables,
    edges_from_file,
    label,
    short,
    status_of,
)

PHASED_FILE = REFERENCES / "phased_relations_glioma_rna_final.csv"
OUT = RESULTS / "model_scores_phased"
PHASES = {
    1: "fase 1 -- efeito no mRNA (avaliação)",
    2: "fase 2 -- efeito na proteína (exploratória)",
}
GROUP_ORDER = ("mut_expr", "cis", "transcricao", "proteina")


def load_relations(genes: list[str]) -> list[dict]:
    table = pd.read_csv(PHASED_FILE)
    unknown = (set(table.source_gene) | set(table.target_gene)) - set(genes)
    if unknown:
        raise ValueError(f"{PHASED_FILE.name} tem genes fora do painel: {sorted(unknown)}")
    bad_phase = set(table.phase) - set(PHASES)
    if bad_phase:
        raise ValueError(f"Fases desconhecidas em {PHASED_FILE.name}: {sorted(bad_phase)}")
    return [
        {
            "phase": int(r.phase),
            "level": r.level,
            "group": r.group,
            "source": label(r.source_gene, r.source_kind),
            "target": label(r.target_gene, r.target_kind),
            "effect": r.effect,
            "line": "" if pd.isna(r.line) else r.line,
            "evidence": r.evidence,
            "citation": r.citation,
            "note": "" if pd.isna(r.note) else r.note,
        }
        for r in table.itertuples(index=False)
    ]


def impossible_edges(graph: ModelGraph) -> pd.DataFrame:
    """Arestas expressão -> mutação: biologicamente impossíveis."""
    return pd.DataFrame(
        [
            {"source": a, "target": b}
            for a, b in sorted(graph.arrows)
            if kind_of(a) == EXPR and kind_of(b) == MUT
        ],
        columns=["source", "target"],
    )


def score(
    graph: ModelGraph, variables: list[str], relations: list[dict], max_path: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    present = set(variables)
    applicable = [r for r in relations if r["source"] in present and r["target"] in present]
    # As setas nos dois sentidos só se desculpam dentro da mesma fase.
    reference_by_phase = {}
    for r in applicable:
        reference_by_phase.setdefault(r["phase"], set()).add((r["source"], r["target"]))
    chance_cache, rows = {}, []
    for rel in applicable:
        status, hops = status_of(rel, graph, reference_by_phase[rel["phase"]], max_path)
        key = (rel["group"], kind_of(rel["source"]), kind_of(rel["target"]))
        if key not in chance_cache:
            chance_cache[key] = chance_levels(graph, variables, rel, max_path)
        chance_direct, chance_path = chance_cache[key]
        rows.append(
            {
                "phase": rel["phase"],
                "level": rel["level"],
                "group": rel["group"],
                "source": rel["source"],
                "target": rel["target"],
                "effect": rel["effect"],
                "line": rel["line"],
                "status": status,
                "model_path_length": hops,
                "wrong": status in WRONG,
                "chance_direta": chance_direct,
                "chance_caminho": chance_path,
                "evidence": rel["evidence"],
                "citation": rel["citation"],
                "note": rel["note"],
            }
        )
    detail = pd.DataFrame(rows)

    summary = []
    if not detail.empty:
        for (phase, group), part in detail.groupby(["phase", "group"], sort=False):
            counts = part.status.value_counts()
            summary.append(
                {
                    "phase": phase,
                    "group": group,
                    "n_applicable": len(part),
                    "n_found": int((part.status != "nao_encontrada").sum()),
                    **{s: int(counts.get(s, 0)) for s in STATUSES},
                    "n_wrong": int(part.wrong.sum()),
                    "chance_direta": part.chance_direta.mean(),
                    "chance_caminho": part.chance_caminho.mean(),
                }
            )
    return detail, pd.DataFrame(summary), impossible_edges(graph)


def report(
    name: str,
    dataset: str,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    impossible: pd.DataFrame,
) -> None:
    print(f"\n=== {name} | {dataset} ===")
    if detail.empty:
        print("nenhuma relação de referência se aplica a este dataset")
    for phase, title in PHASES.items():
        in_phase = detail[detail.phase == phase] if not detail.empty else detail
        if in_phase.empty:
            continue
        found = int((in_phase.status != "nao_encontrada").sum())
        wrong = int(in_phase.wrong.sum())
        print(f"\n## {title}: {found}/{len(in_phase)} encontradas, {wrong} ao contrário")
        for group in GROUP_ORDER:
            part = in_phase[in_phase.group == group]
            if part.empty:
                continue
            row = summary[(summary.phase == phase) & (summary.group == group)].iloc[0]
            print(
                f"  [{group}] {int(row.n_found)}/{len(part)} "
                f"(acaso: directa {row.chance_direta:.0%}, "
                f"caminho {row.chance_caminho:.0%})"
            )
            for r in part.itertuples(index=False):
                arrow = "-|" if r.effect == "down-regulates" else "->"
                text = STATUS_TEXT.get(r.status, "não encontrada")
                hops = (
                    f" ({int(r.model_path_length)} arestas)"
                    if r.status.startswith("indireta")
                    else ""
                )
                print(f"    {short(r.source)} {arrow} {short(r.target)}: {text}{hops}")
    if not impossible.empty:
        print(
            f"\n[impossíveis] {len(impossible)} arestas expressão -> mutação: "
            + ", ".join(f"{short(a)} -> {short(b)}" for a, b in impossible.itertuples(index=False))
        )


