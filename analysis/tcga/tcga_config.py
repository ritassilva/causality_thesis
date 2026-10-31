from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MUTATION_SUFFIX = "_mut"
EXPRESSION_SUFFIX = "_expr_log2tpm"


def default_source_file():
    return ROOT / "datasets/tcga/preparation/cohort/06_causal_input_mutation_plus_log2_tpm.tsv"


def gene_of(label):
    for suffix in (MUTATION_SUFFIX, EXPRESSION_SUFFIX):
        if label.endswith(suffix):
            return label[: -len(suffix)]
    raise ValueError(label)
