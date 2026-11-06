"""Compare TCGA graph relationships with selected biological reference sources.

The comparisons distinguish mutation, expression and cross-modality variable
pairs. Statistical enrichment can be evaluated against uniform and degree-
preserving nulls and against marginal-association and partial-correlation
baselines. Curated reference files are under datasets/tcga/references/.
Optional STRING and OmniPath queries are used only by the explicit data-fetch
command; opening the presentation notebooks does not access these services.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.parse
import urllib.request
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
STUDY_ROOT = Path(__file__).resolve().parents[2] / "results/tcga"
sys.path.insert(0, str(HERE))

from tcga_config import (
    EXPRESSION_SUFFIX,
    MUTATION_SUFFIX,  # noqa: E402
    default_source_file,
    gene_of,
)

ARMS_ANALYSIS = STUDY_ROOT / "edges"
REFERENCES = Path(__file__).resolve().parents[2] / "datasets/tcga/references"
RESULTS = Path(__file__).resolve().parents[2] / "results/tcga/literature_comparison"
THESIS_FILE = REFERENCES / "phased_relations_glioma_rna_final.csv"
STRING_FILE = REFERENCES / "string_network.tsv"
STRING_IDS_FILE = REFERENCES / "string_ids.tsv"
OMNIPATH_FILE = REFERENCES / "omnipath_interactions.tsv"

FINAL_MODEL = "PRISM__tuned"

# STRING ---------------------------------------------------------------------- #
STRING_API = "https://string-db.org/api"
STRING_SPECIES = 9606
STRING_CALLER = "prism_tcga_analysis"
# 150 é o limiar "low confidence" do STRING; o corte a sério é aplicado depois,
# sobre os canais escolhidos, por isso só tem de não deitar nada fora antes.
STRING_FETCH_MIN_SCORE = 150
# Prior de interacção ao acaso que o STRING desconta de cada canal antes de os
# combinar (FAQ do STRING, "How are the scores computed?").
STRING_PRIOR = 0.041
STRING_EXPDB_CHANNELS = ("escore", "dscore")

# OmniPath ---------------------------------------------------------------------- #
OMNIPATH_API = "https://omnipathdb.org/interactions"
# ``omnipath`` é o núcleo de sinalização curada da literatura; ``collectri`` é
# factor de transcrição -> alvo. Os outros datasets (pathwayextra,
# kinaseextra, ...) têm menos curadoria e ficam de fora.
OMNIPATH_DATASETS = "omnipath,collectri"

MUT, EXPR = "mut", "expr"
BLOCKS = {"mut-mut": (MUT, MUT), "mut-expr": (MUT, EXPR), "expr-expr": (EXPR, EXPR)}

# Que referência se lê em que bloco de que dataset.
PLAN = {
    "tcga_mut35": [
        ("literature_all", "mut-mut"),
        ("literature_independent", "mut-mut"),
    ],
    "tcga_expr35": [
        ("string_expdb", "expr-expr"),
        ("string_combined", "expr-expr"),
        ("omnipath_directed", "expr-expr"),
        ("thesis_expression", "expr-expr"),
        ("collectri", "expr-expr"),
    ],
    "tcga_mut_expr70": [
        ("cis", "mut-expr"),
        ("literature_all", "mut-mut"),
        ("literature_independent", "mut-mut"),
        ("string_expdb", "expr-expr"),
        ("string_combined", "expr-expr"),
        ("omnipath_directed", "expr-expr"),
        ("thesis_expression", "expr-expr"),
        ("collectri", "expr-expr"),
    ],
}
DEFAULT_VIEW = {
    "tcga_mut35": "raw",
    "tcga_expr35": "raw",
    "tcga_mut_expr70": "oriented",
}


def kind_of(label: str) -> str:
    if label.endswith(MUTATION_SUFFIX):
        return MUT
    if label.endswith(EXPRESSION_SUFFIX):
        return EXPR
    raise ValueError(f"Variável sem tipo reconhecido: {label!r}")


def mut_label(gene: str) -> str:
    return gene + MUTATION_SUFFIX


def expr_label(gene: str) -> str:
    return gene + EXPRESSION_SUFFIX


def pair(a: str, b: str) -> frozenset:
    return frozenset((a, b))


def block_of(a: str, b: str) -> str:
    kinds = sorted((kind_of(a), kind_of(b)))
    return {(MUT, MUT): "mut-mut", (EXPR, MUT): "mut-expr", (EXPR, EXPR): "expr-expr"}[tuple(kinds)]


def block_universe(labels: list[str], block: str) -> list[frozenset]:
    """Todos os pares não ordenados possíveis no bloco."""
    first, second = BLOCKS[block]
    if first == second:
        nodes = [x for x in labels if kind_of(x) == first]
        return [pair(a, b) for a, b in combinations(nodes, 2)]
    left = [x for x in labels if kind_of(x) == first]
    right = [x for x in labels if kind_of(x) == second]
    return [pair(a, b) for a in left for b in right]


# Referências ----------------------------------------------------------------- #


def cis_reference(genes: list[str]) -> dict[frozenset, dict]:
    table = pd.read_csv(THESIS_FILE)
    return {
        pair(mut_label(r.source_gene), expr_label(r.target_gene)): {
            "gene_a": r.source_gene, "gene_b": r.target_gene,
            "relation": "cis", "sources": r.citation,
        }
        for r in table.itertuples(index=False)
        if r.group == "cis" and r.source_gene in genes and r.target_gene in genes
    }


def literature_reference(
    genes: set[str], independent_only: bool, label_of
) -> dict[frozenset, dict]:
    """The eight retained thesis references contain no mutation-pair controls."""
    return {}


def _string_post(endpoint: str, **params) -> pd.DataFrame:
    params.setdefault("species", STRING_SPECIES)
    params.setdefault("caller_identity", STRING_CALLER)
    data = urllib.parse.urlencode(params).encode()
    with urllib.request.urlopen(f"{STRING_API}/tsv/{endpoint}", data=data, timeout=120) as response:
        text = response.read().decode()
    return pd.read_csv(io.StringIO(text), sep="\t")


def fetch_string(genes: list[str]) -> None:
    """Descarrega a rede STRING entre os genes do painel para ``references/``."""
    REFERENCES.mkdir(parents=True, exist_ok=True)
    ids = _string_post("get_string_ids", identifiers="\r".join(genes), limit=1, echo_query=1)
    ids = ids.rename(columns={"queryItem": "gene"})[["gene", "stringId", "preferredName"]]
    ids.to_csv(STRING_IDS_FILE, sep="\t", index=False)
    missing = sorted(set(genes) - set(ids.gene))
    if missing:
        print(f"[string] sem identificador no STRING: {missing}")
    gene_by_id = dict(zip(ids.stringId, ids.gene))
    network = _string_post(
        "network",
        identifiers="\r".join(ids.stringId),
        required_score=STRING_FETCH_MIN_SCORE,
    )
    network["gene_a"] = network.stringId_A.map(gene_by_id)
    network["gene_b"] = network.stringId_B.map(gene_by_id)
    network.to_csv(STRING_FILE, sep="\t", index=False)
    print(f"[string] {len(network)} interacções entre {len(ids)} genes -> {STRING_FILE}")


def combine_channels(frame: pd.DataFrame, channels) -> pd.Series:
    """Score combinado só sobre ``channels``, à maneira do STRING."""
    without_prior = 1.0
    for channel in channels:
        s = ((frame[channel] - STRING_PRIOR) / (1 - STRING_PRIOR)).clip(lower=0)
        without_prior = without_prior * (1 - s)
    total = 1 - without_prior
    return total + STRING_PRIOR * (1 - total)


def string_reference(threshold: float, channels: str) -> dict[frozenset, dict]:
    if not STRING_FILE.is_file():
        raise FileNotFoundError(f"{STRING_FILE} não existe: corre uma vez com --fetch-string")
    network = pd.read_csv(STRING_FILE, sep="\t").dropna(subset=["gene_a", "gene_b"])
    if channels == "expdb":
        network["used_score"] = combine_channels(network, STRING_EXPDB_CHANNELS)
    else:
        network["used_score"] = network["score"]
    kept = network[network.used_score >= threshold]
    reference = {}
    for row in kept.itertuples(index=False):
        if row.gene_a == row.gene_b:
            continue
        key = pair(expr_label(row.gene_a), expr_label(row.gene_b))
        previous = reference.get(key)
        if previous is None or row.used_score > previous["score"]:
            reference[key] = {
                "gene_a": row.gene_a,
                "gene_b": row.gene_b,
                "relation": f"string_{channels}",
                "score": round(float(row.used_score), 3),
                "sources": "STRING",
            }
    return reference


def fetch_omnipath(genes: list[str]) -> None:
    """Descarrega do OmniPath as interacções que tocam os genes do painel."""
    REFERENCES.mkdir(parents=True, exist_ok=True)
    query = urllib.parse.urlencode(
        {
            "partners": ",".join(genes),
            "genesymbols": "yes",
            "datasets": OMNIPATH_DATASETS,
            "organisms": 9606,
            "fields": "sources,references,datasets,curation_effort",
        }
    )
    with urllib.request.urlopen(f"{OMNIPATH_API}?{query}", timeout=180) as response:
        text = response.read().decode()
    network = pd.read_csv(io.StringIO(text), sep="\t")
    panel = set(genes)
    inside = network[network.source_genesymbol.isin(panel) & network.target_genesymbol.isin(panel)]
    inside.to_csv(OMNIPATH_FILE, sep="\t", index=False)
    print(f"[omnipath] {len(inside)} interacções dentro do painel -> {OMNIPATH_FILE}")


def _true(column: pd.Series) -> pd.Series:
    return column.astype(str).str.strip().str.lower() == "true"


def omnipath_reference(subset: str) -> dict[frozenset, dict]:
    """Pares dirigidos do OmniPath; ``direction`` é None se vier nos dois sentidos."""
    if not OMNIPATH_FILE.is_file():
        raise FileNotFoundError(f"{OMNIPATH_FILE} não existe: corre uma vez com --fetch")
    network = pd.read_csv(OMNIPATH_FILE, sep="\t")
    network = network[
        _true(network.is_directed) & (network.source_genesymbol != network.target_genesymbol)
    ]
    if subset == "thesis_expression":
        network = network[
            network.sources.fillna("").str.split(";").apply(lambda resources: "SIGNOR" in resources)
        ]
    elif subset == "collectri":
        network = network[_true(network.collectri)]
    triples = []
    for row in network.itertuples(index=False):
        sign = ("+" if str(row.is_stimulation).lower() == "true" else "") + (
            "-" if str(row.is_inhibition).lower() == "true" else ""
        )
        triples.append((row.source_genesymbol, row.target_genesymbol, sign))
    return directed_reference(triples, f"OmniPath ({subset})")


def thesis_expression_reference() -> dict[frozenset, dict]:
    """Expression relationships from the eight final thesis references."""
    table = pd.read_csv(THESIS_FILE)
    triples = [
        (r.source_gene, r.target_gene, r.effect)
        for r in table.itertuples(index=False)
        if r.source_kind == "expr" and r.target_kind == "expr"
    ]
    return directed_reference(triples, "Thesis references")


def directed_reference(triples, label: str) -> dict[frozenset, dict]:
    """(origem, destino, sinal) -> referência de expressão com direcção.

    ``direction`` fica None quando o par vem nos dois sentidos.
    """
    directions: dict[frozenset, set] = {}
    effects: dict[frozenset, set] = {}
    for source_gene, target_gene, sign in triples:
        source = expr_label(source_gene)
        target = expr_label(target_gene)
        key = pair(source, target)
        directions.setdefault(key, set()).add((source, target))
        effects.setdefault(key, set()).add(f"{source_gene}->{target_gene}{sign}")
    reference = {}
    for key, found in directions.items():
        ((source, target),) = found if len(found) == 1 else ((None, None),)
        reference[key] = {
            "gene_a": gene_of(sorted(key)[0]),
            "gene_b": gene_of(sorted(key)[1]),
            "relation": "; ".join(sorted(effects[key])),
            "direction": (source, target) if source else None,
            "sources": label,
        }
    return reference


# Grafos dos braços ----------------------------------------------------------- #


def load_arm_skeletons(
    within: float, views: dict
) -> tuple[dict[tuple[str, str], set], dict[tuple[str, str], set]]:
    """(dataset, braço) -> esqueleto do consenso, e as arestas orientadas.

    Execution records identify every arm, including empty graphs. Consensus is
    calculated in memory from the current per-run edge tables, with the same
    measured run counts used by the notebook analyses. Skeleton support counts
    each unordered pair once per model seed; arrows require support for the
    exact direction. No intermediate consensus CSVs are required or written.
    """
    if not 0 < within <= 1:
        raise ValueError("within must be greater than 0 and at most 1")
    if str(HERE.parent) not in sys.path:
        sys.path.insert(0, str(HERE.parent))
    from glioma_analysis import edge_occurrences

    skeletons, oriented = {}, {}
    occurrences = {}
    for dataset, view in views.items():
        base = dataset.split("__", 1)[0]
        valid_views = {"raw", "oriented", "pruned"} if base == "tcga_mut_expr70" else {"raw"}
        if view not in valid_views:
            raise ValueError(f"Invalid view {view!r} for {dataset}")
        metrics = pd.read_csv(STUDY_ROOT / "metrics" / f"{dataset}.csv")
        metrics = metrics.loc[metrics.dataset.eq(dataset)].copy()
        if metrics.empty:
            raise ValueError(f"No execution records for {dataset}")
        metrics["arm"] = metrics.model + "__" + metrics.arm
        background = metrics.condition_id.astype(str).str.endswith("__background_knowledge")
        metrics.loc[background, "arm"] += "__bk"
        for arm in metrics.arm.unique():
            skeletons[(dataset, arm)] = set()
            oriented[(dataset, arm)] = set()

        edge_file = ARMS_ANALYSIS / ("subtypes.csv" if "__" in dataset else "full_cohort.csv")
        if edge_file not in occurrences:
            occurrences[edge_file] = edge_occurrences(edge_file)
        rates = occurrences[edge_file]
        chosen = rates.loc[rates.dataset.eq(dataset) & rates.view.eq(view)]
        for row in chosen.itertuples(index=False):
            key = (dataset, row.arm)
            if row.skeleton_rate >= within:
                skeletons[key].add(pair(row.source, row.target))
            if row.directed_rate >= within:
                oriented[key].add((row.source, row.target))
    return skeletons, oriented


def direction_counts(reference: dict, edges: set, arrows: set) -> dict:
    """Dos pares recuperados com direcção única na referência, como o braço os orientou."""
    same = reversed_ = unoriented = 0
    for key, meta in reference.items():
        direction = meta.get("direction")
        if direction is None or key not in edges:
            continue
        if direction in arrows:
            same += 1
        elif direction[::-1] in arrows:
            reversed_ += 1
        else:
            unoriented += 1
    oriented_total = same + reversed_
    return {
        "n_same": same,
        "n_reversed": reversed_,
        "n_unoriented": unoriented,
        "p_direction": (
            stats.binomtest(same, oriented_total, 0.5, alternative="greater").pvalue
            if oriented_total
            else np.nan
        ),
    }


def arm_orientation(meta: dict, key: frozenset, edges: set, arrows: set) -> str:
    if key not in edges:
        return ""
    direction = meta.get("direction")
    a, b = sorted(key)
    if (a, b) in arrows:
        drawn = (a, b)
    elif (b, a) in arrows:
        drawn = (b, a)
    else:
        return "unoriented"
    if direction is None:
        return f"{gene_of(drawn[0])}->{gene_of(drawn[1])}"
    return "same" if drawn == direction else "reversed"


# Baselines ------------------------------------------------------------------- #


def marginal_ranking(data: pd.DataFrame, block: str) -> list[frozenset]:
    """Pares do bloco do mais para o menos significativo, teste par-a-par."""
    labels = list(data.columns)
    scored = []
    if block == "expr-expr":
        cols = [x for x in labels if kind_of(x) == EXPR]
        rho, p = stats.spearmanr(data[cols].to_numpy())
        for i, j in combinations(range(len(cols)), 2):
            scored.append((p[i, j], -abs(rho[i, j]), pair(cols[i], cols[j])))
    elif block == "mut-mut":
        cols = [x for x in labels if kind_of(x) == MUT]
        values = data[cols].to_numpy().astype(int)
        for i, j in combinations(range(len(cols)), 2):
            a, b = values[:, i], values[:, j]
            table = [
                [int(np.sum((a == 1) & (b == 1))), int(np.sum((a == 1) & (b == 0)))],
                [int(np.sum((a == 0) & (b == 1))), int(np.sum((a == 0) & (b == 0)))],
            ]
            _, p = stats.fisher_exact(table)
            log_or = np.log(
                ((table[0][0] + 0.5) * (table[1][1] + 0.5))
                / ((table[0][1] + 0.5) * (table[1][0] + 0.5))
            )
            scored.append((p, -abs(log_or), pair(cols[i], cols[j])))
    else:
        muts = [x for x in labels if kind_of(x) == MUT]
        exprs = [x for x in labels if kind_of(x) == EXPR]
        for m in muts:
            carriers = data[m].to_numpy() == 1
            for e in exprs:
                values = data[e].to_numpy()
                if carriers.all() or not carriers.any():
                    scored.append((1.0, 0.0, pair(m, e)))
                    continue
                u, p = stats.mannwhitneyu(
                    values[carriers], values[~carriers], alternative="two-sided"
                )
                auc = u / (carriers.sum() * (~carriers).sum())
                scored.append((p, -abs(auc - 0.5), pair(m, e)))
    scored.sort(key=lambda item: (item[0], item[1], sorted(item[2])))
    return [item[2] for item in scored]


def partial_ranking(data: pd.DataFrame, block: str) -> list[frozenset]:
    """Pares do bloco por |correlação parcial| no dataset inteiro."""
    labels = list(data.columns)
    precision = np.linalg.pinv(np.corrcoef(data.to_numpy(dtype=float), rowvar=False))
    scale = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    partial = -precision / scale
    index = {x: i for i, x in enumerate(labels)}
    scored = []
    for key in block_universe(labels, block):
        a, b = sorted(key)
        scored.append((-abs(partial[index[a], index[b]]), sorted(key), key))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in scored]


# Nulo que preserva graus ------------------------------------------------------ #


def degree_preserving_null(
    edges: set, reference: set, bipartite: bool, n_perm: int, rng: np.random.Generator
) -> np.ndarray:
    """Hits na referência de ``n_perm`` grafos com os mesmos graus.

    Cadeia de trocas de arestas (a-b, c-d -> a-d, c-b): 10k trocas de
    aquecimento e 2k entre amostras. No bloco mut-expr as arestas são
    orientadas (mut, expr) e a troca mantém o grafo bipartido. Um grafo quase
    completo aceita poucas trocas e o nulo fica perto do observado, o que torna
    o p-valor conservador, não optimista.
    """
    k = len(edges)
    if k < 2:
        return np.array([])
    if bipartite:
        current = [tuple(sorted(e, key=lambda x: kind_of(x) != MUT)) for e in edges]
    else:
        current = [tuple(sorted(e)) for e in edges]
    present = {pair(a, b) for a, b in current}

    def run(n_steps: int) -> None:
        picks = rng.integers(k, size=(n_steps, 2))
        flips = rng.random(n_steps) < 0.5
        for (i, j), flip in zip(picks, flips):
            if i == j:
                continue
            a, b = current[i]
            c, d = current[j]
            if flip and not bipartite:
                c, d = d, c
            if len({a, b, c, d}) < 4:
                continue
            new_one, new_two = pair(a, d), pair(c, b)
            if new_one in present or new_two in present:
                continue
            present.discard(pair(a, b))
            present.discard(pair(c, d))
            present.add(new_one)
            present.add(new_two)
            current[i], current[j] = (a, d), (c, b)

    run(10 * k)
    hits = np.empty(n_perm, dtype=int)
    for n in range(n_perm):
        run(2 * k)
        hits[n] = len(present & reference)
    return hits


# Comparação ----------------------------------------------------------------- #


def compare(args) -> None:
    source = Path(args.data) if args.data else default_source_file()
    data_all = pd.read_csv(source, sep="\t", index_col=0)
    if data_all.isna().any().any():
        raise ValueError(f"{source} tem valores em falta")
    genes = sorted({gene_of(x) for x in data_all.columns})

    references = {
        "cis": cis_reference(genes),
        "literature_all": literature_reference(set(genes), False, mut_label),
        "literature_independent": literature_reference(set(genes), True, mut_label),
        "string_expdb": string_reference(args.string_threshold, "expdb"),
        "string_combined": string_reference(args.string_threshold, "combined"),
        "omnipath_directed": omnipath_reference("all"),
        "thesis_expression": thesis_expression_reference(),
        "collectri": omnipath_reference("collectri"),
    }
    directional = {"omnipath_directed", "thesis_expression", "collectri"}

    views = {d: (args.view_70 if d == "tcga_mut_expr70" else DEFAULT_VIEW[d]) for d in PLAN}
    skeletons, oriented = load_arm_skeletons(args.within, views)
    rng = np.random.default_rng(args.seed)

    columns_of = {
        "tcga_mut35": [x for x in data_all.columns if kind_of(x) == MUT],
        "tcga_expr35": [x for x in data_all.columns if kind_of(x) == EXPR],
        "tcga_mut_expr70": list(data_all.columns),
    }
    carriers = {gene_of(x): int(data_all[x].sum()) for x in data_all.columns if kind_of(x) == MUT}

    rows, detail = [], []
    for dataset, plan in PLAN.items():
        data = data_all[columns_of[dataset]]
        labels = list(data.columns)
        rankings = {}
        for block in sorted({b for _, b in plan}):
            rankings[block] = (
                marginal_ranking(data, block),
                partial_ranking(data, block),
            )
        arms = sorted(arm for (d, arm) in skeletons if d == dataset)
        if args.arms:
            arms = [a for a in arms if a in args.arms]
        for reference_name, block in plan:
            universe = set(block_universe(labels, block))
            reference = {
                key: meta for key, meta in references[reference_name].items() if key in universe
            }
            reference_keys = set(reference)
            marginal, partial = rankings[block]
            for arm in arms:
                edges = {e for e in skeletons[(dataset, arm)] if block_of(*sorted(e)) == block}
                arrows = oriented[(dataset, arm)]
                direction = (
                    direction_counts(reference, edges, arrows)
                    if reference_name in directional
                    else dict.fromkeys(
                        ("n_same", "n_reversed", "n_unoriented", "p_direction"), np.nan
                    )
                )
                k = len(edges)
                hits = len(edges & reference_keys)
                top_marginal = set(marginal[:k])
                top_partial = set(partial[:k])
                null = degree_preserving_null(
                    edges, reference_keys, block == "mut-expr", args.n_perm, rng
                )
                expected = k * len(reference_keys) / len(universe)
                rows.append(
                    {
                        "dataset": dataset,
                        "view": views[dataset],
                        "within_threshold": args.within,
                        "arm": arm,
                        "is_final_model": arm == FINAL_MODEL,
                        "reference": reference_name,
                        "block": block,
                        "universe": len(universe),
                        "n_reference": len(reference_keys),
                        "k": k,
                        "hits": hits,
                        "precision": hits / k if k else np.nan,
                        "recall": hits / len(reference_keys) if reference_keys else np.nan,
                        "expected_uniform": expected,
                        "fold_enrichment": hits / expected if expected else np.nan,
                        "p_hypergeom": (
                            stats.hypergeom.sf(hits - 1, len(universe), len(reference_keys), k)
                            if k
                            else np.nan
                        ),
                        "null_degree_mean": null.mean() if null.size else np.nan,
                        "p_degree_preserving": (
                            (1 + np.sum(null >= hits)) / (1 + null.size) if null.size else np.nan
                        ),
                        "baseline_marginal_hits": len(top_marginal & reference_keys),
                        "baseline_partial_hits": len(top_partial & reference_keys),
                        **direction,
                    }
                )
                for key, meta in reference.items():
                    detail.append(
                        {
                            "dataset": dataset,
                            "arm": arm,
                            "reference": reference_name,
                            "gene_a": meta["gene_a"],
                            "gene_b": meta["gene_b"],
                            "relation": meta["relation"],
                            "sources": meta["sources"],
                            "string_score": meta.get("score", np.nan),
                            "carriers_a": carriers.get(meta["gene_a"])
                            if block != "expr-expr"
                            else np.nan,
                            "carriers_b": carriers.get(meta["gene_b"])
                            if block == "mut-mut"
                            else np.nan,
                            "reference_direction": (
                                f"{gene_of(meta['direction'][0])}->{gene_of(meta['direction'][1])}"
                                if meta.get("direction")
                                else ""
                            ),
                            "found_by_arm": key in edges,
                            "arm_orientation": arm_orientation(meta, key, edges, arrows),
                            "found_by_marginal_at_k": key in top_marginal,
                            "found_by_partial_at_k": key in top_partial,
                        }
                    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "comparison_by_arm.csv", index=False)
    pd.DataFrame(detail).to_csv(RESULTS / "reference_hits.csv", index=False)
    summary = {
        "source_file": str(source),
        "within_threshold": args.within,
        "views": views,
        "n_perm": args.n_perm,
        "seed": args.seed,
        "string_threshold": args.string_threshold,
        "reference_sizes": {name: len(ref) for name, ref in references.items()},
        "final_model": FINAL_MODEL,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))

    final = table[table.is_final_model]
    show = [
        "dataset",
        "reference",
        "k",
        "hits",
        "n_reference",
        "expected_uniform",
        "p_degree_preserving",
        "baseline_marginal_hits",
        "baseline_partial_hits",
        "n_same",
        "n_reversed",
        "n_unoriented",
    ]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(f"\n{FINAL_MODEL} (within={args.within}):")
        print(final[show].round(3).to_string(index=False))
    print(f"\nTabelas em {RESULTS}")


