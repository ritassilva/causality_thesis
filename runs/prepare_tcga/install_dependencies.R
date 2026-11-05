# Run from the repository root: Rscript runs/prepare_tcga/install_dependencies.R
# Dependencies used by the final streaming preparation workflow.
packages <- c("data.table", "dplyr", "stringr")
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
