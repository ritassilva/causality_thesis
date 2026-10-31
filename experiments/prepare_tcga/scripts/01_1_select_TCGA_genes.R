############################################################
# Filter TCGA GLIOMA / GBMLGG mutation binary matrix
#
# Inputs:
# 1) processed_matrices/sig_genes_mutation_binary_matrix.txt
# 2) sig_genes.txt
#
# Output:
# 1) processed_matrices/filtered_sig_genes_mutation_binary_matrix.txt
#
# Filtering:
# keep genes with:
# q <= filter_q_max
# AND mutation_frequency >= min_mutation_frequency
#
# Matrix format:
# attrib_name | TCGA.XX.XXXX | TCGA.XX.XXXX | ...
############################################################

library(data.table)
library(dplyr)

############################################################
# 1. User settings
############################################################

base_dir <- file.path(Sys.getenv("TCGA_WORK_DIR", file.path(getwd(), "results", "new_runs", "tcga_preparation")), "data_raw")

processed_dir <- file.path(base_dir, "processed_matrices")

# The original sig_genes mutation matrix here:
input_mutation_file <- file.path(
  processed_dir,
  "sig_genes_mutation_binary_matrix.txt"
)

# This will be the new filtered matrix:
output_mutation_file <- file.path(
  processed_dir,
  "filtered_sig_genes_mutation_binary_matrix.txt"
)

sig_file <- file.path(base_dir, "sig_genes.txt")

############################################################
# 2. Filtering settings
############################################################

# Keep genes with q-value below or equal to this value
filter_q_max <- 0.10

# Keep genes mutated in at least this fraction of patients
# 0.05 = at least 5% of patients
min_mutation_frequency <- 0.01

# Optional absolute minimum number of mutated patients.
# Set to NULL if you do not want to use it.
# Example: min_mutated_patients <- 10
min_mutated_patients <- NULL

############################################################
# 3. Read unfiltered mutation binary matrix
############################################################

mutation_df <- fread(
  input_mutation_file,
  data.table = FALSE,
  sep = "\t",
  header = TRUE,
  check.names = FALSE
)

if (!"attrib_name" %in% colnames(mutation_df)) {
  stop("Expected first column named 'attrib_name' in mutation matrix.")
}

gene_names <- mutation_df$attrib_name
patient_cols <- colnames(mutation_df)[-1]

cat("Unfiltered mutation matrix dimensions:\n")
print(dim(mutation_df))

cat("Number of genes in unfiltered matrix:", length(gene_names), "\n")
cat("Number of patients:", length(patient_cols), "\n")

############################################################
# 4. Compute mutation frequency per gene
############################################################

mutation_values <- mutation_df[, patient_cols, drop = FALSE]

mutation_values[] <- lapply(mutation_values, function(x) {
  as.numeric(as.character(x))
})

valid_counts <- rowSums(!is.na(mutation_values))
mutation_counts <- rowSums(mutation_values == 1, na.rm = TRUE)

mutation_frequency <- mutation_counts / valid_counts

mutation_frequency_df <- data.frame(
  gene = gene_names,
  mutated_patients = mutation_counts,
  valid_patients = valid_counts,
  total_patients = length(patient_cols),
  missing_patients = length(patient_cols) - valid_counts,
  mutation_frequency = mutation_counts / valid_counts,
  mutation_percentage = 100 * mutation_counts / valid_counts,
  stringsAsFactors = FALSE
)

cat("\nMutation frequency summary:\n")
print(summary(mutation_frequency_df$mutation_frequency))

cat("\nTop mutated genes before filtering:\n")
print(
  mutation_frequency_df %>%
    arrange(desc(mutated_patients), gene) %>%
    head(20)
)

############################################################
# 5. Read MutSig2CV q-values
############################################################

sig_genes <- fread(
  sig_file,
  data.table = FALSE,
  sep = "\t",
  header = TRUE,
  check.names = FALSE
)

if (!all(c("gene", "q") %in% colnames(sig_genes))) {
  stop("Expected columns 'gene' and 'q' in sig_genes.txt.")
}

sig_genes <- sig_genes %>%
  mutate(
    gene = as.character(gene),
    q = as.numeric(q)
  ) %>%
  filter(
    !is.na(gene),
    gene != "",
    !is.na(q)
  )

cat("\nsig_genes dimensions after cleaning:\n")
print(dim(sig_genes))

cat("\nQ-value summary:\n")
print(summary(sig_genes$q))

############################################################
# 6. Combine q-values with mutation frequencies
############################################################

gene_selection_df <- mutation_frequency_df %>%
  inner_join(
    sig_genes %>% select(gene, q),
    by = "gene"
  ) %>%
  arrange(q, desc(mutated_patients), gene)

cat("\nGenes present in both mutation matrix and sig_genes.txt:", nrow(gene_selection_df), "\n")

############################################################
# 7. Apply filters
############################################################

kept_df <- gene_selection_df %>%
  filter(
    q <= filter_q_max,
    mutation_frequency >= min_mutation_frequency
  )

if (!is.null(min_mutated_patients)) {
  kept_df <- kept_df %>%
    filter(mutated_patients >= min_mutated_patients)
}

kept_genes <- kept_df %>%
  pull(gene) %>%
  unique()

if (length(kept_genes) == 0) {
  stop(
    "No genes were kept. Try increasing filter_q_max, ",
    "decreasing min_mutation_frequency, or decreasing min_mutated_patients."
  )
}

cat("\nFiltering criteria:\n")
cat("q <=", filter_q_max, "\n")
cat("mutation_frequency >=", min_mutation_frequency, "\n")

if (!is.null(min_mutated_patients)) {
  cat("mutated_patients >=", min_mutated_patients, "\n")
}

cat("\nNumber of kept genes:", length(kept_genes), "\n")

cat("\nKept genes:\n")
print(
  kept_df %>%
    select(
  gene,
  q,
  mutated_patients,
  valid_patients,
  total_patients,
  missing_patients,
  mutation_frequency,
  mutation_percentage
)
)

############################################################
# 8. Create filtered mutation binary matrix
############################################################

filtered_mutation_df <- mutation_df %>%
  filter(attrib_name %in% kept_genes)

# Order genes by q-value, then by mutation count
filtered_mutation_df <- filtered_mutation_df %>%
  left_join(
    kept_df %>%
      select(attrib_name = gene, q, mutated_patients),
    by = "attrib_name"
  ) %>%
  arrange(q, desc(mutated_patients), attrib_name) %>%
  select(-q, -mutated_patients)

cat("\nFiltered mutation matrix dimensions:\n")
print(dim(filtered_mutation_df))

############################################################
# 9. Save filtered matrix as mutation_binary_matrix.txt
############################################################

write.table(
  filtered_mutation_df,
  file = output_mutation_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = "NA"
)

############################################################
# 10. Summary
############################################################

cat("\nDone.\n")
cat("Original/unfiltered genes:", nrow(mutation_df), "\n")
cat("Filtered genes:", nrow(filtered_mutation_df), "\n")
cat("Patients:", length(patient_cols), "\n")

cat("\nInput file:\n")
cat(input_mutation_file, "\n")

cat("\nOutput file:\n")
cat(output_mutation_file, "\n")