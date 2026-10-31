# Run from the repository root: Rscript experiments/prepare_tcga/install_dependencies.R
# The streaming workflow uses CRAN packages. The optional GDC query workflow also
# needs Bioconductor; enable it with TCGA_INSTALL_BIOCONDUCTOR=true.
packages <- c("data.table", "dplyr", "stringr")
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
if (tolower(Sys.getenv("TCGA_INSTALL_BIOCONDUCTOR", "false")) == "true") {
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
  }
  BiocManager::install(c("TCGAbiolinks", "SummarizedExperiment"), ask = FALSE, update = FALSE)
}
