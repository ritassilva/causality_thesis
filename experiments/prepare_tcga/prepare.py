"""Recover the exact TCGA source files and optionally rebuild the cohort.

Only Python's standard library and curl are needed for downloads. Processing
requires Rscript with data.table, dplyr and stringr. The RNA stage uses the
archived file selection, never a new query that could change the cohort.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")


def obtain(source, raw_dir, cache_dir):
    destination = raw_dir / source["filename"]
    if destination.exists():
        require_hash(destination, source["sha256"])
        print(f"Verified existing {destination.name}", flush=True)
        return
    cached = cache_dir / source["download_sha256"]
    if not cached.exists():
        # Temporary files live beside the destination for atomic publication.
        with tempfile.TemporaryDirectory(dir=cache_dir) as temp:
            downloaded = Path(temp) / "download"
            subprocess.run(
                [
                    "curl",
                    "-fL",
                    "--retry",
                    "3",
                    "--connect-timeout",
                    "30",
                    "--max-time",
                    "900",
                    source["url"],
                    "-o",
                    str(downloaded),
                ],
                check=True,
            )
            require_hash(downloaded, source["download_sha256"])
            downloaded.replace(cached)
    require_hash(cached, source["download_sha256"])
    with tempfile.TemporaryDirectory(dir=raw_dir) as temp:
        extracted = Path(temp) / source["filename"]
        if source["archive_member"]:
            # Read just the pinned regular file; never extract arbitrary paths.
            with tarfile.open(cached, "r:gz") as archive:
                member = archive.getmember(source["archive_member"])
                if not member.isfile():
                    raise ValueError("Expected a regular file in the clinical archive")
                with archive.extractfile(member) as src, extracted.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        else:
            shutil.copyfile(cached, extracted)
        require_hash(extracted, source["sha256"])
        extracted.replace(destination)
    print(f"Recovered and verified {destination.name}", flush=True)


def run_r(script, env, rscript):
    subprocess.run([rscript, "--vanilla", str(HERE / script)], env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["raw", "preprocess", "rna"], default="raw")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(os.environ.get("TCGA_WORK_DIR", ROOT / "results/new_runs/tcga_preparation")),
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument(
        "--max-rna-files",
        type=int,
        help="Bound a live download check; partial runs do not produce a final cohort",
    )
    args = parser.parse_args()
    if args.max_rna_files is not None and args.max_rna_files < 1:
        parser.error("--max-rna-files must be positive")
    work = args.work_dir.resolve()
    raw = work / "data_raw"
    cache = (args.cache_dir or work / "downloads").resolve()
    raw.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/tcga_sources.json").read_text())
    for source in config["sources"]:
        obtain(source, raw, cache)
    if args.stage == "raw":
        return
    env = dict(os.environ, TCGA_WORK_DIR=str(work))
    for script in [
        "scripts/01_0_create_firebrowse_gbmlgg_matrices.R",
        "scripts/01_1_select_TCGA_genes.R",
        "scripts/01_2_dowload_TCGA_who2021_classification.R",
        "scripts/02_0_prepare_firebrowse_GLIOMA.R",
    ]:
        run_r(script, env, args.rscript)
    for expected in config["expected_preprocessed"]:
        require_hash(work / expected["path"], expected["sha256"])
    metadata_dir = work / "tcga_rnaseq_selected_genes"
    metadata_dir.mkdir(exist_ok=True)
    for expected in config["metadata"]:
        source = ROOT / "datasets/tcga/preparation/source_metadata" / expected["filename"]
        require_hash(source, expected["sha256"])
        shutil.copyfile(source, metadata_dir / source.name)
    run_r("validate_tcga_dna_rna_sample_match.R", env, args.rscript)
    if args.stage == "rna":
        env["TCGA_MAX_FILES"] = str(args.max_rna_files or "Inf")
        env["TCGA_RUN_DOWNLOAD"] = "true"
    else:
        env["TCGA_RUN_DOWNLOAD"] = "false"
    run_r("stream_extract_tcga_rnaseq_selected_genes.R", env, args.rscript)
    if args.stage == "rna":
        cohort = work / "tcga_rnaseq_selected_genes_streamed"
        if (cohort / "06_causal_input_mutation_plus_log2_tpm.tsv").exists():
            for expected in config["expected_cohort"]:
                require_hash(work / expected["path"], expected["sha256"])
            env["TCGA_COHORT_DIR"] = str(cohort)
            env["TCGA_SUBTYPE_DIR"] = str(work / "subtype_splits")
            subprocess.run(
                [sys.executable, str(HERE / "scripts/03_0_split_datasets_by_who2021_subtype.py")],
                env=env,
                check=True,
            )
            for expected in config["expected_subtypes"]:
                require_hash(work / expected["path"], expected["sha256"])
            print(
                "All seven cohort tables match the archived publication inputs byte for byte.",
                flush=True,
            )
        else:
            print(
                "Partial RNA run: no final cohort produced; resume without --max-rna-files.",
                flush=True,
            )
    print(f"TCGA {args.stage} stage finished; outputs: {work}", flush=True)


if __name__ == "__main__":
    main()
