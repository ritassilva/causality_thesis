"""Que relações causais conhecidas encontrou um modelo, e quais encontrou ao contrário.

Para um grafo (o consenso de um braço do estudo TCGA, ou um CSV de arestas
qualquer), percorre as relações de ``references/strict_relations.csv`` e diz,
uma a uma, o que o modelo fez com cada uma. Só entram relações com direcção e
bem suportadas:

| grupo | o quê | variáveis |
|---|---|---|
| ``signor`` | todas as setas do SIGNOR entre os 35 genes, incluindo as tracejadas (indirectas) | expressão -> expressão |
| ``cis`` | NF1, ATRX e RB1: a mutação baixa a expressão do próprio gene | mutação -> expressão |

As setas que o SIGNOR dá nos dois sentidos (PIK3R1 <-> PIK3CA, NOTCH1 <->
ZMIZ1) estão como duas linhas, A -> B e B -> A. Cada seta é avaliada à parte,
e encontrar uma delas no sentido B -> A não é um erro, porque B -> A também
está na referência.

**Directas e indirectas.** Uma seta A -> B conta como encontrada se o modelo
tiver a aresta A - B (directa) *ou* um caminho dirigido A => B com no máximo
``--max-path`` arestas (indirecta). As setas tracejadas do SIGNOR passam por
moléculas que não estão no painel (o PIP3, no caso PTEN -| PIK3CA), por isso
o modelo pode representá-las de qualquer das duas formas.

Estados de cada seta:

* ``direta_sentido_certo`` / ``indireta_sentido_certo``;
* ``direta_sentido_trocado`` / ``indireta_sentido_trocado`` -- **errada**;
* ``direta_outro_sentido_na_referencia`` -- o modelo tem B -> A e B -> A
  também é uma seta da referência: não é erro;
* ``direta_sem_orientacao`` -- a aresta existe mas o modelo não a orientou;
* ``nao_encontrada``.

**Errado** quer dizer só duas coisas: uma seta encontrada ao contrário (e o
contrário não está na referência), e arestas expressão -> mutação, que são
biologicamente impossíveis (aparecem na vista ``raw`` dos braços que não
impõem a restrição). Uma aresta do modelo que não está na referência não é
contada como erro: a referência é incompleta.

**Acaso.** Num grafo denso quase tudo está ligado por algum caminho, por isso
cada seta traz a probabilidade de um par ao acaso do mesmo tipo estar ligado
directamente (``chance_direta``) ou por caminho dirigido (``chance_caminho``).

Exemplos (a partir da raiz do repositório, com o ambiente de análise ativo)::

    # o modelo final, nos três datasets
    python analysis/tcga/score_model.py --arm PRISM__tuned

    # todos os braços, só as tabelas
    python analysis/tcga/score_model.py --arm all

    # um grafo qualquer: CSV com colunas source,target[,orientation]
    python analysis/tcga/score_model.py --edges grafo.csv --dataset tcga_mut_expr70

Escreve em ``results/tcga/literature_comparison/model_scores/``.
"""

from __future__ import annotations

import argparse
from itertools import product
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
    expr_label,
    gene_of,
    kind_of,
    load_arm_skeletons,
    mut_label,
    pair,
)

STRICT_FILE = REFERENCES / "strict_relations.csv"
OUT = RESULTS / "model_scores"
DATASETS = ("tcga_mut35", "tcga_expr35", "tcga_mut_expr70")
GROUPS = ("signor", "cis")
STATUSES = (
    "direta_sentido_certo",
    "direta_outro_sentido_na_referencia",
    "direta_sem_orientacao",
    "direta_sentido_trocado",
    "indireta_sentido_certo",
    "indireta_sentido_trocado",
    "nao_encontrada",
)
WRONG = {"direta_sentido_trocado", "indireta_sentido_trocado"}


def label(gene: str, kind: str) -> str:
    if kind == "mut":
        return mut_label(gene)
    if kind == "expr":
        return expr_label(gene)
    raise ValueError(f"Tipo de variável desconhecido em {STRICT_FILE.name}: {kind!r}")


def load_relations(genes: list[str]) -> list[dict]:
    table = pd.read_csv(STRICT_FILE)
    unknown = (set(table.source_gene) | set(table.target_gene)) - set(genes)
    if unknown:
        raise ValueError(f"{STRICT_FILE.name} tem genes fora do painel: {sorted(unknown)}")
    relations = []
    for r in table.itertuples(index=False):
        if r.group not in GROUPS:
            raise ValueError(f"Grupo desconhecido em {STRICT_FILE.name}: {r.group!r}")
        relations.append(
            {
                "group": r.group,
                "source": label(r.source_gene, r.source_kind),
                "target": label(r.target_gene, r.target_kind),
                "effect": r.effect,
                "line": "" if pd.isna(r.line) else r.line,
                "evidence": r.evidence,
                "citation": r.citation,
                "note": "" if pd.isna(r.note) else r.note,
            }
        )
    return relations


# Grafo do modelo ------------------------------------------------------------- #


class ModelGraph:
    def __init__(self, skeleton: set, arrows: set):
        self.skeleton = set(skeleton)
        self.arrows = set(arrows)
        self.children: dict[str, set] = {}
        for a, b in self.arrows:
            self.children.setdefault(a, set()).add(b)

    def directed_distance(self, start, goal, limit):
        """Comprimento do caminho dirigido mais curto, ou None se > limit."""
        seen, frontier = {start}, [start]
        for depth in range(1, limit + 1):
            nxt = []
            for node in frontier:
                for other in self.children.get(node, ()):
                    if other == goal:
                        return depth
                    if other not in seen:
                        seen.add(other)
                        nxt.append(other)
            frontier = nxt
        return None


def edges_from_file(path: Path) -> ModelGraph:
    table = pd.read_csv(path)
    if not {"source", "target"} <= set(table.columns):
        raise ValueError(f"{path} precisa das colunas source e target")
    if "orientation" not in table.columns:
        table["orientation"] = "directed"
    skeleton = {pair(r.source, r.target) for r in table.itertuples(index=False)}
    arrows = {
        (r.source, r.target) for r in table.itertuples(index=False) if r.orientation == "directed"
    }
    return ModelGraph(skeleton, arrows)


# Pontuação -------------------------------------------------------------------- #


def status_of(
    rel: dict, graph: ModelGraph, reference_arrows: set, max_path: int
) -> tuple[str, int | None]:
    a, b = rel["source"], rel["target"]
    if pair(a, b) in graph.skeleton:
        if (a, b) in graph.arrows:
            return "direta_sentido_certo", 1
        if (b, a) in graph.arrows:
            if (b, a) in reference_arrows:
                return "direta_outro_sentido_na_referencia", 1
            return "direta_sentido_trocado", 1
        return "direta_sem_orientacao", 1
    forward = graph.directed_distance(a, b, max_path)
    if forward:
        return "indireta_sentido_certo", forward
    if (b, a) not in reference_arrows:
        backward = graph.directed_distance(b, a, max_path)
        if backward:
            return "indireta_sentido_trocado", backward
    return "nao_encontrada", None


def chance_levels(
    graph: ModelGraph, variables: list[str], rel: dict, max_path: int
) -> tuple[float, float]:
    """Probabilidade de um par ao acaso do mesmo tipo estar ligado."""
    left = [v for v in variables if kind_of(v) == kind_of(rel["source"])]
    right = [v for v in variables if kind_of(v) == kind_of(rel["target"])]
    pairs = [(x, y) for x, y in product(left, right) if gene_of(x) != gene_of(y)]
    if rel["group"] == "cis":
        # o acaso de um par mut -> expr qualquer, não só do mesmo gene
        pairs = [(x, y) for x, y in product(left, right)]
    if not pairs:
        return float("nan"), float("nan")
    direct = sum(pair(x, y) in graph.skeleton for x, y in pairs) / len(pairs)
    reach = sum(graph.directed_distance(x, y, max_path) is not None for x, y in pairs) / len(pairs)
    return direct, reach


def score(
    graph: ModelGraph, variables: list[str], relations: list[dict], max_path: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    present = set(variables)
    applicable = [r for r in relations if r["source"] in present and r["target"] in present]
    reference_arrows = {(r["source"], r["target"]) for r in applicable}
    chance_cache, rows = {}, []
    for rel in applicable:
        status, hops = status_of(rel, graph, reference_arrows, max_path)
        key = (rel["group"], kind_of(rel["source"]), kind_of(rel["target"]))
        if key not in chance_cache:
            chance_cache[key] = chance_levels(graph, variables, rel, max_path)
        chance_direct, chance_path = chance_cache[key]
        rows.append(
            {
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
    for group in GROUPS:
        part = detail[detail.group == group] if not detail.empty else detail
        if part.empty:
            continue
        counts = part.status.value_counts()
        summary.append(
            {
                "group": group,
                "n_applicable": len(part),
                "n_found": int((part.status != "nao_encontrada").sum()),
                **{s: int(counts.get(s, 0)) for s in STATUSES},
                "n_wrong": int(part.wrong.sum()),
                "chance_direta": part.chance_direta.mean(),
                "chance_caminho": part.chance_caminho.mean(),
            }
        )
    impossible = pd.DataFrame(
        [
            {"source": a, "target": b}
            for a, b in sorted(graph.arrows)
            if kind_of(a) == EXPR and kind_of(b) == MUT
        ],
        columns=["source", "target"],
    )
    return detail, pd.DataFrame(summary), impossible


# Relatório ------------------------------------------------------------------- #

STATUS_TEXT = {
    "direta_sentido_certo": "encontrada, aresta directa no sentido certo",
    "direta_outro_sentido_na_referencia": "encontrada no sentido B -> A, que também está no SIGNOR",
    "direta_sem_orientacao": "encontrada, aresta directa sem orientação",
    "direta_sentido_trocado": "ERRADA: aresta directa no sentido contrário",
    "indireta_sentido_certo": "encontrada por caminho no sentido certo",
    "indireta_sentido_trocado": "ERRADA: caminho no sentido contrário",
}


def short(variable: str) -> str:
    return f"{gene_of(variable)}_{'mut' if kind_of(variable) == MUT else 'expr'}"


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
    for group in GROUPS:
        part = detail[detail.group == group] if not detail.empty else detail
        if part.empty:
            continue
        row = summary[summary.group == group].iloc[0]
        print(
            f"\n[{group}] {int(row.n_found)}/{len(part)} encontradas, "
            f"{int(row.n_wrong)} ao contrário "
            f"(acaso: directa {row.chance_direta:.0%}, caminho {row.chance_caminho:.0%})"
        )
        for r in part.itertuples(index=False):
            arrow = "-|" if r.effect == "down-regulates" else "->"
            text = STATUS_TEXT.get(r.status, "não encontrada")
            hops = (
                f" ({int(r.model_path_length)} arestas)" if r.status.startswith("indireta") else ""
            )
            print(f"  {short(r.source)} {arrow} {short(r.target)}: {text}{hops}")
    if not impossible.empty:
        print(
            f"\n[impossíveis] {len(impossible)} arestas expressão -> mutação: "
            + ", ".join(f"{short(a)} -> {short(b)}" for a, b in impossible.itertuples(index=False))
        )


def dataset_variables(header, dataset: str) -> list[str]:
    if dataset == "tcga_mut35":
        return [c for c in header if kind_of(c) == MUT]
    if dataset == "tcga_expr35":
        return [c for c in header if kind_of(c) == EXPR]
    return list(header)


