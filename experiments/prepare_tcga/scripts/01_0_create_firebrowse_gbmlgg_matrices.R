############################################################
# Create TCGA GBMLGG Broad FireBrowse matrices
#
# Somatic mutations were obtained from a Mutation Annotation Format (MAF) file.
# Source: GBMLGG-TP.final_analysis_set.maf.txt. The gene symbol, variant
# classification and tumour sample barcode are used to build the binary matrix.
#
# Outputs:
# 1) mutation_binary_matrix.txt
# 2) clinical_matrix.txt
# 3) clinical_plus_mutation_matrix.txt
#
# Format:
# attrib_name | TCGA.XX.XXXX | TCGA.XX.XXXX | ...
############################################################

library(data.table)
library(dplyr)
library(stringr)

############################################################
# 1. File paths
############################################################

base_dir <- file.path(Sys.getenv("TCGA_WORK_DIR", file.path(getwd(), "results", "new_runs", "tcga_preparation")), "data_raw")

clinical_file <- file.path(base_dir, "GBMLGG.clin.merged.picked.txt")
sig_file      <- file.path(base_dir, "sig_genes.txt")
maf_file      <- file.path(base_dir, "GBMLGG-TP.final_analysis_set.maf.txt")

out_dir <- file.path(base_dir, "processed_matrices")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

q_threshold <- 0.10

############################################################
# 2. Helper functions
############################################################

tcga_to_dot <- function(x) {
  x <- toupper(x)
  x <- str_replace_all(x, "-", ".")
  x
}

tcga_patient_from_sample <- function(x) {
  # Converts TCGA-06-0744-01A-... to TCGA.06.0744
  x <- toupper(x)
  x <- str_sub(x, 1, 12)
  x <- str_replace_all(x, "-", ".")
  x
}

clean_attrib_name <- function(x) {
  x <- as.character(x)
  x <- str_trim(x)
  x <- str_replace_all(x, "\\s+", "_")
  x
}

############################################################
# 3. Read significant MutSig2CV genes
############################################################

sig_genes <- fread(sig_file, data.table = FALSE)

cat("sig_genes columns:\n")
print(colnames(sig_genes))

if (!all(c("gene", "q") %in% colnames(sig_genes))) {
  stop("Expected columns 'gene' and 'q' in sig_genes.txt.")
}

selected_genes <- sig_genes %>%
  mutate(q = as.numeric(q)) %>%
  filter(q <= q_threshold) %>%
  pull(gene) %>%
  unique() %>%
  sort()

cat("Selected MutSig2CV genes with q <=", q_threshold, ":", length(selected_genes), "\n")

writeLines(
  selected_genes,
  file.path(out_dir, paste0("selected_MutSig2CV_genes_q", q_threshold, ".txt"))
)

############################################################
# 4. Read and clean clinical matrix
############################################################

clinical_raw <- fread(
  clinical_file,
  data.table = FALSE,
  sep = "\t",
  header = TRUE,
  quote = "",
  fill = TRUE,
  check.names = FALSE
)

cat("\nClinical dimensions before cleaning:\n")
print(dim(clinical_raw))

clinical_df <- clinical_raw
colnames(clinical_df)[1] <- "attrib_name"

clinical_df$attrib_name <- clean_attrib_name(clinical_df$attrib_name)

clinical_df <- clinical_df %>%
  filter(
    !is.na(attrib_name),
    attrib_name != "",
    attrib_name != "Composite_Element_REF"
  )

old_patient_cols <- colnames(clinical_df)[-1]
new_patient_cols <- tcga_to_dot(old_patient_cols)

colnames(clinical_df) <- c("attrib_name", new_patient_cols)

if (any(duplicated(colnames(clinical_df)))) {
  duplicated_cols <- duplicated(colnames(clinical_df))
  clinical_df <- clinical_df[, !duplicated_cols, drop = FALSE]
}

############################################################
# 5. Add derived survival variables
############################################################

clinical_mat <- clinical_df
rownames(clinical_mat) <- clinical_mat$attrib_name

get_clin_row <- function(row_name) {
  if (row_name %in% clinical_mat$attrib_name) {
    as.character(clinical_mat[clinical_mat$attrib_name == row_name, -1])
  } else {
    rep(NA_character_, ncol(clinical_mat) - 1)
  }
}

patient_cols <- colnames(clinical_mat)[-1]

days_to_death <- suppressWarnings(
  as.numeric(get_clin_row("days_to_death"))
)

days_to_last_followup <- suppressWarnings(
  as.numeric(get_clin_row("days_to_last_followup"))
)

# Some FireBrowse files may use this alternative spelling
days_to_last_follow_up <- suppressWarnings(
  as.numeric(get_clin_row("days_to_last_follow_up"))
)

followup_time <- ifelse(
  !is.na(days_to_last_followup),
  days_to_last_followup,
  days_to_last_follow_up
)

vital_status <- get_clin_row("vital_status")

overall_survival <- ifelse(
  !is.na(days_to_death),
  days_to_death,
  followup_time
)

status <- case_when(
  vital_status %in% c("1", "dead", "Dead", "DEAD", "deceased", "DECEASED") ~ "1",
  vital_status %in% c("0", "alive", "Alive", "ALIVE", "living", "LIVING") ~ "0",
  TRUE ~ as.character(vital_status)
)

overall_survival_chr <- ifelse(
  is.na(overall_survival),
  "NA",
  as.character(overall_survival)
)

status_chr <- ifelse(
  is.na(status) | status == "",
  "NA",
  as.character(status)
)

overallsurvival_combined <- ifelse(
  overall_survival_chr == "NA" | status_chr == "NA",
  "NA",
  paste0(overall_survival_chr, ",", status_chr)
)

derived_survival_df <- data.frame(
  attrib_name = c("overall_survival", "status", "overallsurvival"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)

derived_survival_values <- rbind(
  overall_survival_chr,
  status_chr,
  overallsurvival_combined
)

derived_survival_values <- as.data.frame(
  derived_survival_values,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

colnames(derived_survival_values) <- patient_cols

derived_survival_df <- cbind(
  derived_survival_df["attrib_name"],
  derived_survival_values
)

clinical_df_no_derived_duplicates <- clinical_df %>%
  filter(!attrib_name %in% c("overall_survival", "status", "overallsurvival"))

clinical_final <- bind_rows(
  clinical_df_no_derived_duplicates,
  derived_survival_df
)

write.table(
  clinical_final,
  file = file.path(out_dir, "clinical_matrix.txt"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = "NA"
)

############################################################
# 6. Read MAF and create binary mutation matrix
############################################################

cat("\nSomatic mutations were obtained from a Mutation Annotation Format (MAF) file.\n")
cat("MAF source:", maf_file, "\n")

maf <- fread(
  maf_file,
  data.table = FALSE,
  sep = "\t",
  quote = "",
  fill = TRUE,
  showProgress = TRUE
)

cat("\nMAF rows:", nrow(maf), "\n")
cat("MAF columns:", ncol(maf), "\n")

required_maf_cols <- c(
  "Hugo_Symbol",
  "Variant_Classification",
  "Tumor_Sample_Barcode"
)

missing_maf_cols <- setdiff(required_maf_cols, colnames(maf))

if (length(missing_maf_cols) > 0) {
  stop(
    "Missing required MAF columns: ",
    paste(missing_maf_cols, collapse = ", ")
  )
}

############################################################
# 6.1 Mutation classes counted as non-silent
############################################################

nonsynonymous_classes <- c(
  "Missense_Mutation",
  "Nonsense_Mutation",
  "Frame_Shift_Del",
  "Frame_Shift_Ins",
  "Splice_Site",
  "In_Frame_Del",
  "In_Frame_Ins",
  "Nonstop_Mutation",
  "Start_Codon_Del",
  "Start_Codon_Ins",
  "De_novo_Start_OutOfFrame",
  "De_novo_Start_InFrame"
)

cat("\nVariant_Classification classes present in the MAF:\n")
print(sort(table(maf$Variant_Classification, useNA = "ifany"), decreasing = TRUE))

cat("\nClasses counted as non-silent:\n")
print(intersect(sort(unique(as.character(maf$Variant_Classification))), nonsynonymous_classes))

cat("\nClasses present but not counted as non-silent:\n")
print(setdiff(sort(unique(as.character(maf$Variant_Classification))), nonsynonymous_classes))

############################################################
# 6.2 Filter MAF to selected genes and non-silent classes
############################################################

maf_filtered <- maf %>%
  mutate(
    PATIENT_ID = tcga_patient_from_sample(Tumor_Sample_Barcode),
    GENE = as.character(Hugo_Symbol),
    VARIANT_CLASS = as.character(Variant_Classification)
  ) %>%
  filter(
    GENE %in% selected_genes,
    VARIANT_CLASS %in% nonsynonymous_classes
  ) %>%
  distinct(PATIENT_ID, GENE)

cat("\nFiltered MAF rows:", nrow(maf_filtered), "\n")
cat(
  "Patients with at least one selected non-silent mutation:",
  length(unique(maf_filtered$PATIENT_ID)),
  "\n"
)

############################################################
# 6.3 Create gene × patient binary mutation matrix
############################################################

# Important:
# use all clinical patients, not only mutated patients.
# Patients with no selected mutations are kept with all 0s.
all_patients <- sort(patient_cols)

mutation_matrix <- matrix(
  0L,
  nrow = length(selected_genes),
  ncol = length(all_patients),
  dimnames = list(selected_genes, all_patients)
)

if (nrow(maf_filtered) > 0) {
  maf_filtered <- maf_filtered %>%
    filter(PATIENT_ID %in% all_patients)

  for (i in seq_len(nrow(maf_filtered))) {
    g <- maf_filtered$GENE[i]
    p <- maf_filtered$PATIENT_ID[i]
    mutation_matrix[g, p] <- 1L
  }
}

mutation_df <- as.data.frame(mutation_matrix, check.names = FALSE)
mutation_df <- cbind(
  attrib_name = rownames(mutation_df),
  mutation_df
)
rownames(mutation_df) <- NULL

write.table(
  mutation_df,
  file = file.path(out_dir, "sig_genes_mutation_binary_matrix.txt"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = "NA"
)

############################################################
# 7. Create clinical + mutation matrix
############################################################

clinical_patients <- colnames(clinical_final)[-1]
mutation_patients <- colnames(mutation_df)[-1]

common_patients <- intersect(clinical_patients, mutation_patients)
common_patients <- sort(common_patients)

cat("\nClinical patients:", length(clinical_patients), "\n")
cat("Mutation patients:", length(mutation_patients), "\n")
cat("Common patients:", length(common_patients), "\n")

############################################################
# 8. Summary
############################################################

cat("\nDone.\n")
cat("Selected MutSig2CV genes:", length(selected_genes), "\n")
cat("Mutation matrix dimensions:", dim(mutation_df), "\n")
cat("Clinical matrix dimensions:", dim(clinical_final), "\n")
#cat("Clinical + mutation matrix dimensions:", dim(clinical_plus_mutation), "\n")

cat("\nFiles written to:\n")
cat(out_dir, "\n")
