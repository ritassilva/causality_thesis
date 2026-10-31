############################################################
# Create final glioma datasets from Broad FireBrowse matrices
# + WHO 2021 classification
#
# Outputs:
# 1) binary_dataset_glioma.txt
#    patients x mutation genes + binary clinical variables
#
# 2) clinical_dataset_glioma.txt
#    patients x clinical variables for survival analysis
#
# WHO logic:
# - Create GBM, ASTRO, OLIGO
# - Keep only patients where exactly one of these is 1
# - Remove OLIGO column afterward
# - Therefore OLIGO is the reference class:
#       GBM = 0, ASTRO = 0
############################################################

library(data.table)
library(dplyr)
library(stringr)

############################################################
# 1. Paths
############################################################

project_dir <- Sys.getenv("TCGA_WORK_DIR", file.path(getwd(), "results", "new_runs", "tcga_preparation"))

input_dir <- file.path(
  project_dir,
  "data_raw/processed_matrices"
)

clinical_matrix_file <- file.path(input_dir, "clinical_matrix.txt")
mutation_matrix_file <- file.path(input_dir, "filtered_sig_genes_mutation_binary_matrix.txt")

classification_file <- file.path(
  project_dir,
  "data_raw/SIMPLIFIED_CLASSIFICATION_TCGA_2016_2021_clean.csv"
)

out_dir <- file.path(
  project_dir,
  "glioma_firebrowse_who21"
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

############################################################
# 2. Helper functions
############################################################

standardize_tcga_id <- function(x) {
  x <- as.character(x)
  x <- toupper(x)
  x <- str_replace_all(x, "-", ".")
  x <- str_sub(x, 1, 12)
  x
}

clean_text <- function(x) {
  x <- as.character(x)
  x <- str_trim(x)
  x <- str_to_lower(x)
  x <- str_replace_all(x, "\\s+", "")
  x
}

matrix_attrib_to_patients <- function(file) {
  df <- fread(
    file,
    data.table = FALSE,
    sep = "\t",
    header = TRUE,
    check.names = FALSE,
    quote = "",
    fill = TRUE
  )

  colnames(df)[1] <- "attrib_name"

  attrib_names <- make.unique(as.character(df$attrib_name))

  mat <- as.data.frame(
    t(df[, -1, drop = FALSE]),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  colnames(mat) <- attrib_names
  rownames(mat) <- standardize_tcga_id(rownames(mat))

  mat <- mat[!duplicated(rownames(mat)), , drop = FALSE]

  mat
}

safe_numeric <- function(x) {
  suppressWarnings(as.numeric(as.character(x)))
}

############################################################
# 3. Read clinical and mutation matrices
############################################################

clin_clean <- matrix_attrib_to_patients(clinical_matrix_file)
mut_clean  <- matrix_attrib_to_patients(mutation_matrix_file)

# Convert mutation matrix to numeric 0/1
mut_clean[] <- lapply(mut_clean, function(x) as.integer(as.character(x)))

cat("Clinical dimensions before WHO filtering:\n")
print(dim(clin_clean))

cat("Mutation dimensions before WHO filtering:\n")
print(dim(mut_clean))

############################################################
# 4. Read WHO 2021 classification
############################################################

classification_df <- read.csv(
  classification_file,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

cat("Classification columns:\n")
print(colnames(classification_df))

possible_id_cols <- c(
  "Patient_ID_clin_glioma",
  "Patient_ID",
  "patient_id",
  "PATIENT_ID",
  "Patient.ID"
)

id_col <- intersect(possible_id_cols, colnames(classification_df))[1]

if (is.na(id_col)) {
  stop("Could not find patient ID column in classification file.")
}

if (!"classification.2021" %in% colnames(classification_df)) {
  stop("Could not find column 'classification.2021' in classification file.")
}

classification_df <- classification_df %>%
  mutate(
    PATIENT_ID = standardize_tcga_id(.data[[id_col]]),
    who2021_class = clean_text(classification.2021)
  ) %>%
  select(PATIENT_ID, who2021_class) %>%
  distinct(PATIENT_ID, .keep_all = TRUE)

cat("WHO 2021 class counts in classification file:\n")
print(table(classification_df$who2021_class, useNA = "ifany"))

############################################################
# 5. Add WHO 2021 classification to clinical data
############################################################

clin_clean$who2021_class <- classification_df$who2021_class[
  match(rownames(clin_clean), classification_df$PATIENT_ID)
]

cat("WHO 2021 class counts after matching to clinical data:\n")
print(table(clin_clean$who2021_class, useNA = "ifany"))

############################################################
# 6. Keep patients common to clinical and mutation data
############################################################

common_ids <- intersect(
  rownames(clin_clean),
  rownames(mut_clean)
)

common_ids <- sort(common_ids)

clin_clean <- clin_clean[common_ids, , drop = FALSE]
mut_clean  <- mut_clean[common_ids, , drop = FALSE]

cat("Patients common to clinical and mutation matrices:", length(common_ids), "\n")

############################################################
# 7. Create GBM, ASTRO, and OLIGO columns
############################################################

who_clean <- clean_text(clin_clean$who2021_class)

who_binary_all <- data.frame(
  GBM = ifelse(!is.na(who_clean) & who_clean == "glioblastoma", 1L, 0L),
  ASTRO = ifelse(!is.na(who_clean) & who_clean == "astrocytoma", 1L, 0L),
  OLIGO = ifelse(!is.na(who_clean) & who_clean == "oligodendroglioma", 1L, 0L),
  stringsAsFactors = FALSE
)

rownames(who_binary_all) <- rownames(clin_clean)

cat("WHO binary counts before filtering:\n")
print(colSums(who_binary_all))

# Keep only patients that are classified as exactly one of:
# glioblastoma, astrocytoma, or oligodendroglioma.
who_row_sum <- rowSums(who_binary_all[, c("GBM", "ASTRO", "OLIGO")])

valid_who_patient <- who_row_sum == 1

cat("Patients with valid WHO class:", sum(valid_who_patient), "\n")
cat("Patients removed due to missing/invalid WHO class:", sum(!valid_who_patient), "\n")

clin_clean <- clin_clean[valid_who_patient, , drop = FALSE]
mut_clean  <- mut_clean[valid_who_patient, , drop = FALSE]
who_binary_all <- who_binary_all[valid_who_patient, , drop = FALSE]

# Safety check: after filtering, every patient must belong to exactly one WHO class.
stopifnot(
  all(rowSums(who_binary_all[, c("GBM", "ASTRO", "OLIGO")]) == 1)
)

cat("WHO binary counts after filtering:\n")
print(colSums(who_binary_all))

cat("Patients after WHO filtering:", nrow(clin_clean), "\n")

############################################################
# 8. Remove OLIGO column
############################################################

# OLIGO is now the reference group:
# glioblastoma      -> GBM = 1, ASTRO = 0
# astrocytoma       -> GBM = 0, ASTRO = 1
# oligodendroglioma -> GBM = 0, ASTRO = 0
who_binary <- who_binary_all[, c("GBM", "ASTRO"), drop = FALSE]

############################################################
# 9. Derive survival variables
############################################################

# Derive overall survival if needed.
# If days_to_death is available, use it.
# Otherwise use days_to_last_followup.

if (!"overall_survival" %in% colnames(clin_clean)) {
  days_to_death <- safe_numeric(clin_clean$days_to_death)
  days_to_last_followup <- safe_numeric(clin_clean$days_to_last_followup)

  clin_clean$overall_survival <- ifelse(
    !is.na(days_to_death),
    days_to_death,
    days_to_last_followup
  )
}

# Derive status if needed.
# status = 1 means death/event
# status = 0 means alive/censored

if (!"status" %in% colnames(clin_clean)) {
  vital_status <- clean_text(clin_clean$vital_status)

  clin_clean$status <- case_when(
    vital_status %in% c("1", "dead", "deceased") ~ 1,
    vital_status %in% c("0", "alive", "living") ~ 0,
    TRUE ~ NA_real_
  )
}

############################################################
# 10. Prepare required clinical variables
############################################################

age_numeric <- safe_numeric(clin_clean$years_to_birth)

# Some TCGA files encode age as negative.
# If median is negative, convert age to positive.
if (median(age_numeric, na.rm = TRUE) < 0) {
  age_numeric <- abs(age_numeric)
}

gender_clean <- clean_text(clin_clean$gender)

overall_survival_numeric <- safe_numeric(clin_clean$overall_survival)
status_numeric <- safe_numeric(clin_clean$status)

# Keep only patients with survival time > 0
overall_survival_numeric[overall_survival_numeric <= 0] <- NA
                      
############################################################
# 11. Remove patients with missing required clinical/survival data
############################################################

required_complete <- !is.na(age_numeric) &
  !is.na(gender_clean) &
  gender_clean %in% c("male", "female") &
  !is.na(overall_survival_numeric) &
  !is.na(status_numeric)

cat("Patients removed due to missing age/sex/survival/status:",
    sum(!required_complete), "\n")

clin_clean <- clin_clean[required_complete, , drop = FALSE]
mut_clean <- mut_clean[required_complete, , drop = FALSE]
who_binary <- who_binary[required_complete, , drop = FALSE]

age_numeric <- age_numeric[required_complete]
gender_clean <- gender_clean[required_complete]
overall_survival_numeric <- overall_survival_numeric[required_complete]
status_numeric <- status_numeric[required_complete]

############################################################
# 12. Create binary clinical variables
############################################################

# Median age is calculated only after removing patients
# with missing required clinical/survival information.
med_age <- median(age_numeric, na.rm = TRUE)

cat("Median age after filtering:", med_age, "\n")

clin_clean_bin <- data.frame(
  GBM = who_binary$GBM,
  ASTRO = who_binary$ASTRO,
  sex = ifelse(gender_clean == "male", 1L, 0L),
  age = ifelse(age_numeric >= med_age, 1L, 0L),
  stringsAsFactors = FALSE
)

rownames(clin_clean_bin) <- rownames(clin_clean)

cat("Binary clinical variable summary:\n")
print(colSums(clin_clean_bin, na.rm = TRUE))

############################################################
# 13. Create final binary dataset
############################################################

# Ensure mutation rows are ordered exactly like clinical rows
mut_clean <- mut_clean[rownames(clin_clean_bin), , drop = FALSE]

binary_dataset_glioma <- cbind(
  mut_clean,
  clin_clean_bin[rownames(mut_clean), , drop = FALSE]
)

############################################################
# 14. Create final clinical survival dataset
############################################################

clinical_dataset_glioma <- data.frame(
  GBM = clin_clean_bin$GBM,
  ASTRO = clin_clean_bin$ASTRO,
  sex = clin_clean_bin$sex,
  age = clin_clean_bin$age, # age_numeric, #   ############### change if you do not want clinical age binary
  overall_survival = overall_survival_numeric,
  status = status_numeric,
  stringsAsFactors = FALSE
)

rownames(clinical_dataset_glioma) <- rownames(clin_clean)

############################################################
# 15. Final NA removal safety check
############################################################

complete_binary <- complete.cases(binary_dataset_glioma)
complete_clinical <- complete.cases(clinical_dataset_glioma)

names(complete_binary) <- rownames(binary_dataset_glioma)
names(complete_clinical) <- rownames(clinical_dataset_glioma)

complete_patients <- intersect(
  names(complete_binary)[complete_binary],
  names(complete_clinical)[complete_clinical]
)

cat("Patients removed due to remaining NAs:",
    nrow(clinical_dataset_glioma) - length(complete_patients), "\n")

clinical_dataset_glioma <- clinical_dataset_glioma[
  complete_patients,
  ,
  drop = FALSE
]

binary_dataset_glioma <- binary_dataset_glioma[
  complete_patients,
  ,
  drop = FALSE
]

# Reorder binary dataset to match clinical dataset exactly
binary_dataset_glioma <- binary_dataset_glioma[
  rownames(clinical_dataset_glioma),
  ,
  drop = FALSE
]

# Safety check: datasets must contain exactly the same patients in the same order
stopifnot(
  identical(
    rownames(binary_dataset_glioma),
    rownames(clinical_dataset_glioma)
  )
)

cat("Patients after final NA removal:", nrow(clinical_dataset_glioma), "\n")

############################################################
# 16. Save final datasets
############################################################

write.table(
  binary_dataset_glioma,
  file = file.path(out_dir, "binary_dataset_glioma.txt"),
  sep = "\t",
  quote = FALSE,
  row.names = TRUE,
  col.names = TRUE,
  na = "NA"
)

write.table(
  clinical_dataset_glioma,
  file = file.path(out_dir, "clinical_dataset_glioma.txt"),
  sep = "\t",
  quote = FALSE,
  row.names = TRUE,
  col.names = TRUE,
  na = "NA"
)

############################################################
# 17. Final summary
############################################################

cat("\nDone.\n")
cat("Output folder:\n")
cat(out_dir, "\n\n")

cat("Final binary dataset dimensions:\n")
print(dim(binary_dataset_glioma))

cat("Final clinical dataset dimensions:\n")
print(dim(clinical_dataset_glioma))

cat("\nFiles created:\n")
cat(file.path(out_dir, "binary_dataset_glioma.txt"), "\n")
cat(file.path(out_dir, "clinical_dataset_glioma.txt"), "\n")