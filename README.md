# Causal discovery results

Final datasets, model measurements and notebook analyses for the PRISM thesis study. The PRISM implementation is installed from the commit recorded in `requirements-models.txt`.

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
datasets/
├── synthetic/             <condition>__seed_<seed>.npz
├── sachs/                 Observations and reference graph
└── tcga/                  Cohorts, subtypes and biological references
results/
├── final/                 Final comparative benchmark and evaluation CSVs
├── analysis/              Prepared numerical notebook tables and arrays
├── benchmark/             Per-condition benchmark measurements and control
├── tuned/                 Per-condition tuned/untuned measurements
├── prism/                 Selection, priors, ablation and convergence results
├── scalability/           Dedicated scalability measurements
├── sachs/                 Sachs model measurements
└── tcga/                  TCGA model outputs and numerical summaries
notebooks/                 Presentation of the reported analyses
analysis/                  Numerical calculations and visualization functions
runs/                      Model fitting, numerical preparation and data processing
configs/                   Study protocols, selected parameters and environments
requirements-*.txt         Python dependency specifications
```

Synthetic files are stored once per condition and dataset seed. They contain `X`, `X_raw`, `true_adj`, `latent_nodes` and `observed_conf_nodes` where available. Search, validation and evaluation samples share this folder; the configurations determine which seeds each stage uses. Dataset seeds identify data instances, while model seeds identify algorithm runs on those instances.

The final comparative measurements are in `results/final/benchmark.csv`, `results/final/biological_benchmark.csv` and `results/final/evaluation.csv`. The evaluation contains 5,940 completed fits across six families and 15 dataset seeds. Per-condition CSVs under `results/benchmark/` and `results/tuned/` contain the corresponding measurements. Dataset-instance averages are the comparison units used in the reported paired analyses.

The shared per-fit result fields are:

```text
dataset,model,arm,dataset_seed,model_seed,condition_id,status,timed_out,n_nodes,n_samples,f1,fbeta,precision,recall,SHD,runtime_seconds
```

The three aggregate source CSVs also retain measurement metadata. `arm` distinguishes tuned and untuned configurations; `condition_id` identifies an analysis condition or stage. `fbeta` denotes F0.5. Prior AUPRC evaluations are stored in the prepared prior-analysis tables.

## Numerical preparation and figures

`analysis/calculations.py` contains the numerical preparation functions. `runs/calculate_results.py` executes them and saves the outputs under `results/analysis/`. The `.pkl` bundles contain tables and arrays, preserving their types and indices; they contain no figures. They are included for notebook presentation and can be regenerated explicitly when inputs or analysis definitions change:

```bash
python runs/calculate_results.py
```

Numerical preparation includes graph-score evaluation, uncertainty summaries, paired statistical comparisons and the sorting baseline. The analysis dependency versions used for this preparation and notebook execution are pinned in `requirements-analysis.txt`.

Notebooks load per-fit measurements directly from the source CSVs; no additional source-data bundle is stored. Calculated Wilcoxon and Friedman results are kept in `statistics.pkl`; benchmark and ablation presentation tables do not duplicate those tests. The Excel workbook is their consultation export.

Visualization functions are in `analysis/final_visualization.py`, `analysis/glioma_analysis.py` and `analysis/prism_convergence_figure.py`. Study-specific layouts also appear directly in notebook.

## Statistical comparison workbook

`results/analysis/pvalues.xlsx` organizes the recorded Wilcoxon and Friedman results by dataset and configuration. It includes benchmark, tuned and untuned comparisons, pooled Test/OOD scopes, tuned-versus-untuned comparisons and PRISM ablations. The sheets retain the original p-values and the available Holm-adjusted p-values, with the number of paired blocks and test statistic. The index identifies every sheet and the recorded exclusions.

The workbook is a static export of `results/analysis/statistics.pkl`, containing 8,778 between-method Wilcoxon comparisons, 84 Friedman results, 252 tuned-versus-untuned comparisons and 13 ablation comparisons. Exporting these tables does not rerun the statistical tests. The statistics notebook displays rankings and coverage rather than p-value tables.

## Model runs and configurations

`runs/run_model.py` fits one method on a stored sample and writes predictions and numerical measurements. It is called by `runs/benchmark.py` and `runs/tuning.py`. `runs/run_tcga.py` applies the joint-data TCGA constraint. Model fitting needs the additional dependencies in `requirements-runs.txt`; AVICI also needs `requirements-avici.txt` and its pretrained SCM weights.

`configs/baseline_study.json` defines the baseline search spaces, parameter settings, trial budgets, thresholds and search/validation/test seeds. `configs/prism_study.json` defines the PRISM search with its 400-trial budget. The selected PRISM parameters are in `configs/prism_winner.json`; baseline selections and untuned settings are in `configs/frozen/`. Package inventories for model-fitting environments are under `configs/models/`.

| Study | Model-seed range |
|---|---|
| Benchmark, its control, Sachs and TCGA | 0–14 |
| Search, validation, held-out evaluation and ablation | 0–4 |
| Dedicated scalability study | 0–2 |

`runs/prepare_tcga/` contains the cohort-processing code, and `configs/tcga_sources.json` records the data sources. Prepared cohorts and references are included under `datasets/tcga/`. 

Large NumPy files use Git LFS. After cloning, run `git lfs pull` if the arrays were not downloaded automatically.
