#!/usr/bin/env Rscript

###############################################################################
# Compare TCGA DNA mutation sample barcodes (MAF) with candidate RNA-seq sample
# barcodes. This script reads only Tumor_Sample_Barcode from the large MAF.
###############################################################################

options(stringsAsFactors = FALSE)

project_dir <- Sys.getenv("TCGA_WORK_DIR", file.path(getwd(), "results", "new_runs", "tcga_preparation"))

maf_file <- file.path(
  project_dir,
  "data_raw",
  "GBMLGG-TP.final_analysis_set.maf.txt"
)

rna_metadata_file <- file.path(
  project_dir,
  "tcga_rnaseq_selected_genes",
  "gdc_query_matching_participants.tsv"
)

output_dir <- file.path(
  project_dir,
  "tcga_rnaseq_selected_genes",
  "dna_rna_sample_validation"
)

if (!requireNamespace("data.table", quietly = TRUE)) {
  stop("Install data.table first with: install.packages('data.table')")
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

normalize_barcode <- function(x) {
  x <- toupper(trimws(as.character(x)))
  gsub("\\.", "-", x)
}

participant_from_barcode <- function(x) {
  x <- normalize_barcode(x)
  ifelse(nchar(x) >= 12, substr(x, 1, 12), NA_character_)
}

sample_from_barcode <- function(x) {
  x <- normalize_barcode(x)
  ifelse(nchar(x) >= 16, substr(x, 1, 16), NA_character_)
}

sample_type_code <- function(x) {
  x <- sample_from_barcode(x)
  ifelse(!is.na(x), substr(x, 14, 15), NA_character_)
}

collapse_unique <- function(x) {
  x <- sort(unique(x[!is.na(x) & x != ""]))
  if (length(x) == 0) "" else paste(x, collapse = ";")
}

choose_latest_row <- function(df) {
  created <- ifelse(is.na(df$created_datetime), "", df$created_datetime)
  updated <- ifelse(is.na(df$updated_datetime), "", df$updated_datetime)
  file_id <- ifelse(is.na(df$file_id), "", df$file_id)
  df[order(created, updated, file_id, decreasing = TRUE)[1], , drop = FALSE]
}

if (!file.exists(maf_file)) {
  stop("MAF file does not exist: ", maf_file)
}

if (!file.exists(rna_metadata_file)) {
  stop("RNA metadata file does not exist: ", rna_metadata_file)
}

cat("Reading RNA metadata...\n")
rna <- data.table::fread(
  rna_metadata_file,
  data.table = FALSE,
  sep = "\t",
  check.names = FALSE
)

required_rna_columns <- c(
  "participant_id",
  "sample.submitter_id",
  "cases",
  "file_id",
  "created_datetime",
  "updated_datetime"
)

missing_rna_columns <- setdiff(required_rna_columns, colnames(rna))
if (length(missing_rna_columns) > 0) {
  stop(
    "RNA metadata is missing columns: ",
    paste(missing_rna_columns, collapse = ", ")
  )
}

rna$participant_id <- participant_from_barcode(rna$participant_id)
rna$rna_sample_id <- sample_from_barcode(rna$sample.submitter_id)
rna$rna_sample_type_code <- sample_type_code(rna$rna_sample_id)

cat("Reading only Tumor_Sample_Barcode from the MAF...\n")
maf_barcodes <- data.table::fread(
  maf_file,
  select = "Tumor_Sample_Barcode",
  data.table = FALSE,
  sep = "\t",
  check.names = FALSE,
  showProgress = TRUE
)

if (!"Tumor_Sample_Barcode" %in% colnames(maf_barcodes)) {
  stop("MAF does not contain Tumor_Sample_Barcode.")
}

dna <- data.frame(
  dna_aliquot_barcode = normalize_barcode(
    maf_barcodes$Tumor_Sample_Barcode
  ),
  stringsAsFactors = FALSE
)

dna$dna_sample_id <- sample_from_barcode(dna$dna_aliquot_barcode)
dna$participant_id <- participant_from_barcode(dna$dna_aliquot_barcode)
dna$dna_sample_type_code <- sample_type_code(dna$dna_sample_id)
dna <- unique(dna)

wanted_participants <- sort(unique(rna$participant_id))
dna <- dna[dna$participant_id %in% wanted_participants, , drop = FALSE]

cat("Building participant-level comparison...\n")
summary_rows <- lapply(wanted_participants, function(participant) {
  r <- rna[rna$participant_id == participant, , drop = FALSE]
  d <- dna[dna$participant_id == participant, , drop = FALSE]

  rna_samples <- sort(unique(r$rna_sample_id))
  dna_samples <- sort(unique(d$dna_sample_id))
  exact <- intersect(rna_samples, dna_samples)

  rna_types <- sort(unique(r$rna_sample_type_code))
  dna_types <- sort(unique(d$dna_sample_type_code))

  match_category <- if (nrow(d) == 0) {
    "no_maf_sample"
  } else if (length(exact) > 0) {
    "exact_sample_match"
  } else if ("01" %in% rna_types && "01" %in% dna_types) {
    "same_primary_type_different_vial"
  } else {
    "different_sample_type_or_unresolved"
  }

  data.frame(
    participant_id = participant,
    dna_sample_ids = collapse_unique(dna_samples),
    rna_sample_ids = collapse_unique(rna_samples),
    exact_matching_sample_ids = collapse_unique(exact),
    dna_sample_type_codes = collapse_unique(dna_types),
    rna_sample_type_codes = collapse_unique(rna_types),
    number_dna_samples = length(dna_samples),
    number_rna_samples = length(rna_samples),
    match_category = match_category,
    stringsAsFactors = FALSE
  )
})

comparison <- do.call(rbind, summary_rows)

write.table(
  comparison,
  file.path(output_dir, "dna_rna_sample_match_summary.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = "NA"
)

cat("Selecting RNA files only when an exact DNA-RNA sample match exists...\n")
selected_rows <- list()
review_participants <- character(0)

for (participant in wanted_participants) {
  participant_summary <- comparison[
    comparison$participant_id == participant,
    ,
    drop = FALSE
  ]

  if (participant_summary$match_category != "exact_sample_match") {
    review_participants <- c(review_participants, participant)
    next
  }

  exact_samples <- strsplit(
    participant_summary$exact_matching_sample_ids,
    ";",
    fixed = TRUE
  )[[1]]

  candidates <- rna[
    rna$participant_id == participant &
      rna$rna_sample_id %in% exact_samples,
    ,
    drop = FALSE
  ]

  selected <- choose_latest_row(candidates)
  selected$selection_rule <- if (nrow(candidates) == 1) {
    "exact_dna_rna_sample_match"
  } else {
    "exact_dna_rna_sample_match_then_latest_created"
  }
  selected$matching_dna_sample_id <- selected$rna_sample_id
  selected_rows[[participant]] <- selected
}

if (length(selected_rows) > 0) {
  selected_rna <- do.call(rbind, selected_rows)
  rownames(selected_rna) <- NULL
} else {
  selected_rna <- rna[0, , drop = FALSE]
  selected_rna$selection_rule <- character(0)
  selected_rna$matching_dna_sample_id <- character(0)
}

write.table(
  selected_rna,
  file.path(output_dir, "rna_files_selected_after_dna_match.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = "NA"
)

writeLines(
  sort(unique(review_participants)),
  file.path(output_dir, "participants_requiring_sample_review.txt")
)

write.table(
  dna,
  file.path(output_dir, "unique_maf_tumor_sample_barcodes.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = "NA"
)

cat("\nDNA-RNA sample matching summary:\n")
print(table(comparison$match_category, useNA = "ifany"))
cat("\nParticipants represented in RNA metadata:", length(wanted_participants), "\n")
cat("Participants with an exact DNA-RNA sample match:", nrow(selected_rna), "\n")
cat("Participants requiring review:", length(unique(review_participants)), "\n")
cat("Participants with multiple DNA samples:", sum(comparison$number_dna_samples > 1), "\n")
cat("Output folder:", output_dir, "\n")
