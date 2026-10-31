# Benchmark and PRISM experiments

Datasets, experiment scripts and executed notebooks for PRISM and baseline causal
discovery methods. PRISM is imported from the
[`prism` package](https://github.com/ritassilva/PRISM) developed on this thesis.

## Analyses

Open a notebook to see its saved figures and tables, or run all cells to
recalculate the analysis from the included data and model results.

| Analysis | Notebook |
|---|---|
| Benchmark performance and rankings | [Overview](notebooks/benchmark/01_overview.ipynb) |
| Noise, mechanisms and confounding | [Sensitivity analysis](notebooks/benchmark/02_noise_mechanisms_confounding.ipynb) |
| Seed variability and assumptions | [Stability analysis](notebooks/benchmark/03_stability_sortability_assumptions.ipynb) |
| Graph sizes and numerical summaries | [10, 20 and 30 variables](notebooks/benchmark/05_graph_sizes.ipynb) |
| Additional metrics across all conditions | [Metrics](notebooks/benchmark/06_metrics_by_condition.ipynb) |
| Sortability across all conditions | [Var- and R²-sortability](notebooks/benchmark/sortability/01_all_conditions.ipynb) |
| PRISM selection, performance and causal order | [Tuning](notebooks/prism/tuning.ipynb) |
| PRISM numerical behavior | [Convergence](notebooks/prism/convergence.ipynb) |
| Existence and direction estimates | [Prior quality](notebooks/prism/priors.ipynb) |
| PRISM component contributions | [Ablation](notebooks/prism/ablation.ipynb) |
| Tuned and untuned methods on test/OOD data | [Comparison](notebooks/tuned/01_comparison.ipynb) |
| Search spaces and selected parameters | [Configurations](notebooks/tuned/02_parameters.ipynb) |
| Computational scalability | [Accuracy and runtime](notebooks/scalability/01_accuracy_runtime.ipynb) |
| Sachs | [Biological reference results](notebooks/sachs/01_comparison.ipynb) |
| TCGA glioma | [Biological relationships](notebooks/tcga/01_biological.ipynb) |
| Statistical comparisons | [Paired tests and rankings](notebooks/statistics/01_paired_comparisons.ipynb) |

Per-family notebooks and additional analyses are beside the corresponding main
notebooks. The benchmark includes 36 comparison conditions plus a separate
[Gaussian linear control](notebooks/benchmark/08_gaussian_linear_control.ipynb).

## Run the notebooks

Use Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m ipykernel install --user --name prism-analysis --display-name "PRISM (Python 3.11)"
jupyter lab notebooks/
```

Select the **PRISM (Python 3.11)** kernel. Tables and statistical summaries are
calculated and displayed in the notebooks. Exported figures are written under
the corresponding experiment in `results/`.

## Repository layout

```text
datasets/                 Observation arrays and biological references
results/
├── benchmark/            Individual benchmark measurements and figures
├── tuned/                Individual tuned/untuned measurements and figures
├── prism/
│   ├── tuning/           Selected-model scores and predictions
│   ├── priors/           Estimated prior matrices
│   ├── ablation/         Component ablations
│   └── convergence/      Optimization traces and figures
├── scalability/
├── sachs/
└── tcga/
notebooks/                Analyses grouped by the same experiment names
experiments/              Benchmark, tuning and dataset preparation scripts
analysis/                 Data readers, statistical analysis and plotting functions
configs/                  Study settings, selected parameters and package pins
```

Within an experiment, observation files and per-fit result tables use the same
condition name: `datasets/<experiment>/<condition>.npz` and
`results/<experiment>/<condition>.csv`. 

Per-fit CSV columns are:

```text
dataset,model,arm,dataset_seed,model_seed,condition_id,status,timed_out,n_nodes,n_samples,f1,fbeta,precision,recall,SHD,runtime_seconds
```

`arm` distinguishes tuned and untuned configurations. `condition_id` identifies
ablation variants, search/validation stages or TCGA constraints. AUPRC is retained
separately for the prior analyses.

`requirements-analysis.txt` installs the notebook dependencies, including
CausalDisco for R²-SortnRegress. `requirements-experiments.txt` adds the model
implementations and the PRISM/causalgen commits in
`configs/package-requirements.txt`. AVICI uses `requirements-avici.txt`.
Full dependency snapshots for Python 3.11 on macOS ARM64 are in `configs/locks/`.

## Run new experiments

```bash
python -m pip install -r requirements-experiments.txt
python experiments/benchmark.py --study benchmark --nodes 20 --output results/new_runs/benchmark
python experiments/benchmark.py --study tuned --output results/new_runs/tuned
python experiments/tuning.py --model PRISM --output results/new_runs/prism_tuning
```

These commands show or save the execution plan. Add `--execute` to run it and
`--resume` to continue an interrupted run. Omitting `--nodes` includes all stored
benchmark graph sizes. Use `--model` to restrict a benchmark run to one method.
Each script documents its options through `--help`. An individual fit can also select a
sample directly from a dataset file:

```bash
python experiments/run_model.py --sample datasets/tuned/20nodes_EB_obsconf0_latconf0_Non_Gaussian_Mixed.npz --dataset-seed 19 --model PRISM --output results/new_runs/prism_seed19
```

The PRISM search uses one 400-trial budget over the final parameter space.
`configs/prism_winner.json` stores the selected parameters used in the included
evaluations; `configs/frozen/frozen_winners.json` stores the baseline selections.
Running tuning starts a new search. The notebooks reproduce the included final
measurements without rerunning that search.

| Setting | Model seeds per method |
|---|---|
| Benchmark, its control, Sachs and TCGA | 0–14 |
| Search, validation, held-out evaluation and ablation | 0–4 |
| Dedicated scalability experiments | 0–2 |



AVICI requires its pretrained SCM weights and an optional JAX environment. The
original dependency combination is:

```bash
python3.11 -m venv .venv-avici
source .venv-avici/bin/activate
python -m pip install -r requirements-avici.txt
python -c 'import avici; avici.load_pretrained(download="scm-v0", cache_path="datasets/avici/pretrained")'
```

Use `AVICI_CACHE_DIR` for an existing cache containing `models--larslorch--avici`.
Run the AVICI experiment commands with the Python environment holding those
dependencies.

## TCGA preparation

The source selection and processing scripts are in `experiments/prepare_tcga/`:

```bash
python experiments/prepare_tcga/prepare.py --stage raw
python experiments/prepare_tcga/prepare.py --stage preprocess
python experiments/prepare_tcga/prepare.py --stage rna
```

Install the R dependencies with:

```bash
Rscript experiments/prepare_tcga/install_dependencies.R
```

The optional `download_validate_tcga_rnaseq.R` GDC query workflow also needs
`TCGAbiolinks` and `SummarizedExperiment`; enable their installation with
`TCGA_INSTALL_BIOCONDUCTOR=true Rscript experiments/prepare_tcga/install_dependencies.R`.
These Bioconductor packages are not needed by the streaming preparation workflow.
R dependencies are installed from their repositories and are not version locked.

The workflow uses
pinned source files and RNA identifiers and writes to
`results/new_runs/tcga_preparation/` by default. `--work-dir` selects another
location. The prepared cohorts and reference relationships are already included,
so the notebooks require neither downloads nor R.

Large NumPy arrays are stored with Git LFS. After cloning, run `git lfs pull`
if the arrays have not been downloaded automatically.
