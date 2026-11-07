# Causal discovery results

Model measurements, preparation code and notebook analyses for the PRISM thesis study. The PRISM implementation is installed from the commit recorded in `requirements-models.txt`.

## Analyses

The 12 notebooks load prepared numerical results and display tables and inline figures.

| Analysis | Notebook |
|---|---|
| Benchmark performance, paired rankings, coverage, runtime and metrics | [Performance](notebooks/benchmark/01_performance.ipynb) |
| Noise, mechanisms, confounding, assumptions, seed stability, sortability and family details | [Sensitivity and stability](notebooks/benchmark/02_sensitivity_stability.ipynb) |
| Graph sizes, scalability, completion and runtime | [Graph sizes and scalability](notebooks/benchmark/03_graph_sizes_scalability.ipynb) |
| Tuned/untuned Test/OOD comparison, parameters and family details | [Final comparison](notebooks/tuned/01_comparison.ipynb) |
| PRISM selection scores, held-out performance and causal order | [Tuning](notebooks/prism/tuning.ipynb) |
| PRISM numerical convergence and trajectories by family | [Convergence](notebooks/prism/convergence.ipynb) |
| Existence and direction priors, representations and mixtures | [Prior quality](notebooks/prism/priors.ipynb) |
| PRISM component contributions | [Ablation](notebooks/prism/ablation.ipynb) |
| Sachs performance, metrics and reference network | [Sachs](notebooks/sachs/01_comparison.ipynb) |
| TCGA reference relationships, candidate edges and threshold sensitivity | [Biological relationships](notebooks/tcga/01_biological.ipynb) |
| TCGA agreement across methods, cohorts and subtypes | [Agreement](notebooks/tcga/02_agreement.ipynb) |
| Paired model rankings and comparison coverage | [Rankings](notebooks/statistics/01_paired_comparisons.ipynb) |

Detailed family sections display all available families by default. Set `family` to one of the printed names to focus on a single family. The comparative 20-variable benchmark covers ten families, and the tuned/untuned evaluation covers six. The graph-size analyses cover 10, 15, 20 and 30 variables; dedicated scalability results extend to 500 variables.

## Open the notebooks

Use Python 3.11 and the analysis dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m ipykernel install --user --name prism-analysis --display-name "PRISM (Python 3.11)"
jupyter lab notebooks/
```

Select the **PRISM (Python 3.11)** kernel. The saved numerical tables are included, so opening and running the notebooks requires neither model training nor data downloads.

## Repository layout

```text
references/tcga/          Eight thesis literature relationships
results/
├── analysis/              Prepared numerical notebook tables and arrays
├── benchmark/             Per-condition benchmark measurements
├── tuned/                 Per-condition tuned/untuned measurements
├── prism/                 Selection, priors, ablation and convergence results
├── scalability/           Dedicated scalability measurements
├── sachs/                 Sachs model measurements
└── tcga/
    ├── metrics/           Per-dataset execution records
    ├── edges/             Full-cohort and subtype edge tables
    ├── fits/              PRISM matrices and graphs by cohort
    └── priors/            Prior matrices and settings
notebooks/                 Presentation of the reported analyses
analysis/                  Numerical calculations and visualization functions
runs/                      Model fitting, numerical preparation and data processing
configs/                   Study protocols, selected parameters and environments
└── tcga/file_selection/   Fixed GDC file-selection manifests used in preparation
requirements-*.txt         Python dependency specifications
```

## Data inputs

Observational datasets, downloaded source data and intermediate cohort tables are not distributed in this repository. Synthetic datasets were generated with [CausalGen](https://github.com/ritassilva/causalgen), whose model-environment version is recorded in `requirements-models.txt`. Use CausalGen to generate inputs for new experiments. 

Sachs observations originate from the CD3/CD28 dataset; its 17-arc reference network and variable names are defined in `runs/sachs_ground_truth.py`. TCGA source acquisition and processing are retained in `runs/prepare_tcga/`, with pinned sources in `configs/tcga_sources.json` and the fixed GDC file selection in `configs/tcga/file_selection/`. Only the eight selected thesis relationships are retained in `references/tcga/`. TCGA variable names are read from the existing prior results rather than from observations.

The comparative measurements are stored once per condition in `results/benchmark/` and `results/tuned/`. The shared per-fit result fields are listed below. `dataset_seed` applies only to synthetic datasets and is omitted from Sachs and TCGA results; `model_seed` identifies algorithm runs on each fixed observational dataset.

```text
dataset,model,arm,dataset_seed,model_seed,condition_id,status,timed_out,n_nodes,n_samples,f1,fbeta,precision,recall,SHD,runtime_seconds
```

`arm` distinguishes tuned and untuned configurations; `condition_id` identifies an analysis condition or stage. `fbeta` denotes F0.5. Prior AUPRC evaluations are stored in the prepared prior-analysis tables.

## Numerical preparation and figures

`analysis/calculations.py` contains the numerical preparation functions. `runs/calculate_results.py` executes them and saves the outputs under `results/analysis/`. The `.pkl` bundles contain tables and arrays, preserving their types and indices; they contain no figures. They are included for notebook presentation. Running notebooks does not recalculate them. Sections based on saved measurements can be regenerated independently, for example:

```bash
python runs/calculate_results.py --section scalability
```

Recalculating sortability, prior quality or PRISM selection/order scores requires the corresponding observational inputs and true graphs, supplied explicitly in a local manifest:

```bash
python runs/calculate_results.py --section sortability --samples /path/to/experiment.samples.json
```

Calling the command without `--section` recalculates all sections and therefore requires `--samples`.

Numerical preparation includes graph-score evaluation, uncertainty summaries, paired statistical comparisons and the sorting baseline. The analysis dependency versions used for this preparation and notebook execution are pinned in `requirements-analysis.txt`.

Notebooks load per-fit measurements directly from the source CSVs; no additional source-data bundle is stored. Calculated Wilcoxon and Friedman results are kept in `statistics.pkl`.  Visualization functions are in `analysis/final_visualization.py`, `analysis/glioma_analysis.py` and `analysis/prism_convergence_figure.py`. Study-specific layouts also appear directly in notebook.

## Statistical comparison workbook

`results/analysis/pvalues.xlsx` organizes the recorded Wilcoxon and Friedman results by dataset and configuration. It includes benchmark, tuned and untuned comparisons, pooled Test/OOD scopes, tuned-versus-untuned comparisons and PRISM ablations. The sheets retain the original p-values and the available Holm-adjusted p-values, with the number of paired blocks and test statistic. The index identifies every sheet and the recorded exclusions.

The workbook is a static export of `results/analysis/statistics.pkl`, containing 8,778 between-method Wilcoxon comparisons, 84 Friedman results, 252 tuned-versus-untuned comparisons and 13 ablation comparisons. Exporting these tables does not rerun the statistical tests. The statistics notebook displays rankings and coverage rather than p-value tables.

## Model runs and configurations

`runs/run_model.py` fits one method on an explicit NPZ input and writes predictions and numerical measurements. `runs/run_tcga.py` applies the joint-data TCGA constraint. Model fitting needs the additional dependencies in `requirements-runs.txt`; AVICI also needs `requirements-avici.txt` and downloaded pretrained SCM weights located via `AVICI_CACHE_DIR`.

```bash
python runs/run_model.py --sample /path/to/dataset.npz --study benchmark \
  --condition my_condition --dataset-seed 13 --model PC --model-seed 0 \
  --arm untuned --output results/new_runs/example
```

Paths above are placeholders supplied at execution time. NPZ inputs contain a two-dimensional `X` array, optional `col_names`, and `true_adj` when graph-score evaluation is required. PRISM additionally requires unstandardized `X_raw`. Sachs inputs use `--study sachs` and do not require `--dataset-seed`. TCGA inputs also contain `kinds` for the constrained runner:

```bash
python runs/run_tcga.py --sample /path/to/tcga_mut_expr70.npz \
  --dataset tcga_mut_expr70 --model-seed 0 --output results/new_runs/tcga_example
```

Single fits recompute PRISM priors by default. `--priors stored` explicitly requests the retained prior for the supplied condition and dataset identifier; it must correspond to that exact input. GOLEM-NV requires `--golem-init /path/to/ev_initialization.npz` (array `B`) or `b_init_path` in an explicit parameter JSON, fitted on the same observations. Benchmark replay uses the retained initialization when available for the recorded configuration.

For batch benchmark and tuning runs, `--samples` supplies a caller-owned JSON list with one explicit file per study/dataset/identifier. It may contain absolute paths or paths relative to that JSON file. Keep it outside version control; `*.samples.json` is ignored. For example:

```json
[
  {"study": "benchmark", "dataset": "my_condition", "dataset_seed": 13, "sample": "/path/to/benchmark_dataset.npz"},
  {"study": "model_selection", "dataset": "my_condition", "dataset_seed": 13, "sample": "/path/to/search_dataset.npz"},
  {"study": "evaluation", "dataset": "my_condition", "dataset_seed": 19, "sample": "/path/to/evaluation_dataset.npz"}
]
```

The list must cover the selected configurations and identifiers; these entries illustrate the format. 

```bash
python runs/benchmark.py --study benchmark --samples /path/to/experiment.samples.json --output results/new_runs/benchmark
python runs/tuning.py --model PRISM --samples /path/to/experiment.samples.json --output results/new_runs/tuning
```

Both commands plan by default; add `--execute` to run the methods. They retain the full configured model-seed ranges.

`configs/baseline_study.json` defines the baseline search spaces, parameter settings, trial budgets, thresholds and search/validation/test seeds. `configs/prism_study.json` defines the PRISM search with its 400-trial budget. The selected PRISM parameters are in `configs/prism_winner.json`; baseline selections and untuned settings are in `configs/frozen/`. Package inventories for model-fitting environments are under `configs/models/`.

| Study | Model-seed range |
|---|---|
| Benchmark, Sachs and TCGA | 0–14 |
| Search, validation, held-out evaluation and ablation | 0–4 |
| Dedicated scalability study | 0–2 |

## Obtain and prepare TCGA inputs

The complete acquisition and preparation pipeline remains in `runs/prepare_tcga/`: source downloads and checksum verification, mutation and clinical processing, gene selection, WHO 2021 classification, RNA acquisition and transformation, DNA/RNA matching, subtype splitting and model-input export. Install the model-run Python dependencies and the R dependencies using `runs/prepare_tcga/install_dependencies.R`.

Choose a working location explicitly:

```bash
python runs/prepare_tcga/prepare.py --stage raw --work-dir /path/to/tcga-work
python runs/prepare_tcga/prepare.py --stage preprocess --work-dir /path/to/tcga-work
python runs/prepare_tcga/prepare.py --stage rna --work-dir /path/to/tcga-work
```

Stages verify pinned source and cohort checksums. The RNA stage downloads the fixed selected files and writes the full cohort, subtype TSVs and 12 model-ready NPZ files under the chosen work location (`model_inputs/`). 

Result matrices use Git LFS. After cloning, run `git lfs pull` if the retained result arrays were not downloaded automatically.
