#!/usr/bin/env Rscript

###############################################################################
# Stream exact, DNA-matched TCGA RNA-seq STAR-Counts files one at a time.
#
# The script:
#   1. validates the 665 selected GDC files and exact DNA-RNA sample matches;
#   2. downloads one exact file_id at a time;
#   3. verifies file size and MD5;
#   4. extracts only the mutation-selected genes;
#   5. deletes the temporary full-transcriptome file;
#   6. creates compact raw-count, TPM, log2(TPM+1), and FPKM-UQ matrices;
#   7. creates a mutation + expression matrix for causal-network algorithms;
#   8. writes QC and methods reports.
#
# Full STAR-Counts files are never retained. Small per-participant RDS
# checkpoints allow safe resume after interruption.
###############################################################################

options(stringsAsFactors = FALSE)

###############################################################################
# 1. User settings
###############################################################################

project_dir <- Sys.getenv("TCGA_WORK_DIR", file.path(getwd(), "results", "new_runs", "tcga_preparation"))

binary_file <- file.path(
  project_dir,
  "glioma_firebrowse_who21",
  "binary_dataset_glioma.txt"
)

clinical_file <- file.path(
  project_dir,
  "glioma_firebrowse_who21",
  "clinical_dataset_glioma.txt"
)

selected_metadata_file <- file.path(
  project_dir,
  "tcga_rnaseq_selected_genes",
  "gdc_query_selected_one_file_per_participant.tsv"
)

output_dir <- file.path(
  project_dir,
  "tcga_rnaseq_selected_genes_streamed"
)

checkpoint_dir <- file.path(output_dir, "checkpoints")

# First run with FALSE: validates inputs and downloads nothing.
RUN_DOWNLOAD <- tolower(Sys.getenv("TCGA_RUN_DOWNLOAD", "true")) == "true"

# For a small end-to-end test, use RUN_DOWNLOAD <- TRUE and MAX_FILES <- 3.
# Then use Inf for the full run; the first 3 are reused.
MAX_FILES <- as.numeric(Sys.getenv("TCGA_MAX_FILES", "Inf"))
if (is.na(MAX_FILES) || MAX_FILES < 1) stop("TCGA_MAX_FILES must be positive or Inf")

# Keep checkpoints for reproducibility and resume. They contain only selected
# genes, not the full STAR-Counts files, and are small.
KEEP_CHECKPOINTS <- TRUE

# Explicit, documented mappings from historical mutation symbols to the
# current symbols used in GDC STAR-Counts. Never add aliases here without
# verifying that they identify the same human gene.
gene_symbol_aliases <- c(
  MYST4 = "KAT6B"
)

# Included in checkpoint names so checkpoints made with an older extraction
# definition are not silently reused. Older checkpoint files can remain; they
# are harmless and are ignored by this version.
CHECKPOINT_SCHEMA <- "v2_symbol_aliases"

###############################################################################
# 2. Package and directory checks
###############################################################################

if (!requireNamespace("data.table", quietly = TRUE)) {
  stop("Install data.table first with: install.packages('data.table')")
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)

###############################################################################
# 3. Helper functions
###############################################################################

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

read_patient_table <- function(file) {
  if (!file.exists(file)) {
    stop("File does not exist: ", file)
  }

  df <- read.delim(
    file,
    header = TRUE,
    row.names = 1,
    check.names = FALSE,
    quote = "",
    comment.char = ""
  )

  ids <- participant_from_barcode(rownames(df))

  if (anyNA(ids)) {
    stop("Invalid TCGA participant row names in: ", file)
  }

  if (anyDuplicated(ids)) {
    stop("Duplicated TCGA participants in: ", file)
  }

  rownames(df) <- ids
  df
}

write_tsv <- function(df, file, row.names = FALSE) {
  write.table(
    as.data.frame(df, check.names = FALSE),
    file = file,
    sep = "\t",
    quote = FALSE,
    row.names = row.names,
    col.names = if (row.names) NA else TRUE,
    na = "NA"
  )
}

find_exact_column <- function(columns, candidates, label) {
  hit <- candidates[candidates %in% columns]
  if (length(hit) == 0) {
    stop(
      "Could not find ", label, ". Available columns: ",
      paste(columns, collapse = ", ")
    )
  }
  hit[1]
}

download_with_retries <- function(url, destination, attempts = 4L) {
  last_error <- NULL

  for (attempt in seq_len(attempts)) {
    result <- tryCatch(
      {
        utils::download.file(
          url = url,
          destfile = destination,
          method = "libcurl",
          mode = "wb",
          quiet = TRUE
        )
        TRUE
      },
      error = function(e) {
        last_error <<- conditionMessage(e)
        FALSE
      },
      warning = function(w) {
        last_error <<- conditionMessage(w)
        FALSE
      }
    )

    if (isTRUE(result) && file.exists(destination) &&
        file.info(destination)$size > 0) {
      return(invisible(TRUE))
    }

    if (file.exists(destination)) {
      unlink(destination)
    }

    if (attempt < attempts) {
      Sys.sleep(c(1, 3, 8)[min(attempt, 3)])
    }
  }

  stop("Download failed after ", attempts, " attempts: ", last_error)
}

checkpoint_path <- function(participant, file_id) {
  file.path(
    checkpoint_dir,
    paste0(participant, "__", file_id, "__", CHECKPOINT_SCHEMA, ".rds")
  )
}

validate_checkpoint <- function(checkpoint, metadata_row, selected_genes) {
  required <- c(
    "participant_id", "file_id", "expected_md5", "observed_md5",
    "genes", "counts", "tpm", "fpkm_uq", "missing_genes"
  )

  if (!all(required %in% names(checkpoint))) {
    return(FALSE)
  }

  identical(checkpoint$participant_id, metadata_row$participant_id) &&
    identical(checkpoint$file_id, metadata_row$file_id) &&
    identical(tolower(checkpoint$expected_md5),
              tolower(metadata_row$md5sum)) &&
    identical(tolower(checkpoint$observed_md5),
              tolower(metadata_row$md5sum)) &&
    identical(checkpoint$genes, selected_genes)
}

process_star_counts_file <- function(metadata_row, selected_genes) {
  participant <- metadata_row$participant_id
  file_id <- metadata_row$file_id
  expected_md5 <- tolower(metadata_row$md5sum)
  expected_size <- suppressWarnings(as.numeric(metadata_row$file_size))

  checkpoint_file <- checkpoint_path(participant, file_id)

  if (file.exists(checkpoint_file)) {
    checkpoint <- readRDS(checkpoint_file)
    if (validate_checkpoint(checkpoint, metadata_row, selected_genes)) {
      return(checkpoint)
    }

    stop(
      "An incompatible checkpoint already exists: ", checkpoint_file,
      ". Move it aside and rerun after inspecting why metadata changed."
    )
  }

  temporary_file <- tempfile(
    pattern = paste0(file_id, "_"),
    fileext = ".star_counts.tsv"
  )
  on.exit(unlink(temporary_file), add = TRUE)

  data_url <- paste0("https://api.gdc.cancer.gov/data/", file_id)
  download_with_retries(data_url, temporary_file)

  observed_size <- as.numeric(file.info(temporary_file)$size)
  if (!is.na(expected_size) && observed_size != expected_size) {
    stop(
      "File-size mismatch for ", participant, ": expected ",
      expected_size, ", observed ", observed_size
    )
  }

  observed_md5 <- tolower(unname(tools::md5sum(temporary_file)))
  if (!identical(observed_md5, expected_md5)) {
    stop(
      "MD5 mismatch for ", participant, ": expected ", expected_md5,
      ", observed ", observed_md5
    )
  }

  star <- data.table::fread(
    temporary_file,
    data.table = FALSE,
    sep = "\t",
    check.names = FALSE,
    showProgress = FALSE
  )

  columns <- colnames(star)
  gene_column <- find_exact_column(
    columns,
    c("gene_name", "gene_symbol"),
    "gene-symbol column"
  )
  gene_id_column <- find_exact_column(
    columns,
    c("gene_id", "ensembl_gene_id"),
    "gene-ID column"
  )
  count_column <- find_exact_column(
    columns,
    c("unstranded"),
    "unstranded raw-count column"
  )
  tpm_column <- find_exact_column(
    columns,
    c("tpm_unstranded", "tpm_unstrand"),
    "unstranded TPM column"
  )
  fpkm_uq_column <- find_exact_column(
    columns,
    c("fpkm_uq_unstranded", "fpkm_uq_unstrand"),
    "unstranded FPKM-UQ column"
  )

  symbols <- as.character(star[[gene_column]])
  requested_counts <- table(symbols[symbols %in% selected_genes])
  duplicated_requested <- names(requested_counts)[requested_counts > 1]

  if (length(duplicated_requested) > 0) {
    stop(
      "Selected gene symbols occur more than once in ", participant, ": ",
      paste(duplicated_requested, collapse = ", "),
      ". Inspect Ensembl IDs before combining duplicated symbols."
    )
  }

  row_index <- match(selected_genes, symbols)
  present <- !is.na(row_index)

  counts <- rep(NA_real_, length(selected_genes))
  tpm <- rep(NA_real_, length(selected_genes))
  fpkm_uq <- rep(NA_real_, length(selected_genes))
  gene_ids <- rep(NA_character_, length(selected_genes))

  counts[present] <- as.numeric(star[[count_column]][row_index[present]])
  tpm[present] <- as.numeric(star[[tpm_column]][row_index[present]])
  fpkm_uq[present] <- as.numeric(
    star[[fpkm_uq_column]][row_index[present]]
  )
  gene_ids[present] <- as.character(
    star[[gene_id_column]][row_index[present]]
  )

  if (any(counts[present] < 0, na.rm = TRUE) ||
      any(tpm[present] < 0, na.rm = TRUE) ||
      any(fpkm_uq[present] < 0, na.rm = TRUE)) {
    stop("Negative expression value detected for ", participant)
  }

  checkpoint <- list(
    participant_id = participant,
    sample_id = metadata_row$rna_sample_id,
    project = metadata_row$requested_project,
    file_id = file_id,
    file_name = metadata_row$file_name,
    expected_md5 = expected_md5,
    observed_md5 = observed_md5,
    expected_size = expected_size,
    observed_size = observed_size,
    genes = selected_genes,
    gene_ids = gene_ids,
    counts = counts,
    tpm = tpm,
    fpkm_uq = fpkm_uq,
    missing_genes = selected_genes[!present],
    source_columns = c(
      gene_id = gene_id_column,
      gene_symbol = gene_column,
      counts = count_column,
      tpm = tpm_column,
      fpkm_uq = fpkm_uq_column
    ),
    retrieved_at = format(Sys.time(), tz = "UTC", usetz = TRUE)
  )

  saveRDS(checkpoint, checkpoint_file)
  checkpoint
}

###############################################################################
# 4. Read and validate existing mutation, clinical, and GDC selection inputs
###############################################################################

binary <- read_patient_table(binary_file)
clinical <- read_patient_table(clinical_file)

if (!setequal(rownames(binary), rownames(clinical))) {
  stop("Binary and clinical datasets do not contain the same participants.")
}

if (!file.exists(selected_metadata_file)) {
  stop("Selected GDC metadata does not exist: ", selected_metadata_file)
}

metadata <- data.table::fread(
  selected_metadata_file,
  data.table = FALSE,
  sep = "\t",
  check.names = FALSE
)

required_metadata_columns <- c(
  "participant_id", "file_id", "file_name", "file_size", "md5sum",
  "requested_project", "sample_type", "access", "state",
  "data_type", "experimental_strategy", "analysis_workflow_type",
  "rna_sample_id", "matching_dna_sample_id", "cases"
)

missing_metadata_columns <- setdiff(
  required_metadata_columns,
  colnames(metadata)
)

if (length(missing_metadata_columns) > 0) {
  stop(
    "Selected metadata is missing columns: ",
    paste(missing_metadata_columns, collapse = ", ")
  )
}

metadata$participant_id <- participant_from_barcode(metadata$participant_id)
metadata$rna_sample_id <- sample_from_barcode(metadata$rna_sample_id)
metadata$matching_dna_sample_id <- sample_from_barcode(
  metadata$matching_dna_sample_id
)

if (anyNA(metadata$participant_id) || anyNA(metadata$rna_sample_id)) {
  stop("Selected metadata contains invalid TCGA barcodes.")
}

if (anyDuplicated(metadata$participant_id)) {
  stop("Selected metadata contains duplicated participants.")
}

if (anyDuplicated(metadata$file_id)) {
  stop("Selected metadata contains duplicated file IDs.")
}

if (any(metadata$rna_sample_id != metadata$matching_dna_sample_id)) {
  stop("Selected metadata contains a non-exact DNA-RNA sample match.")
}

if (any(substr(metadata$rna_sample_id, 14, 15) != "01")) {
  stop("A selected RNA sample is not TCGA primary solid tumor code 01.")
}

expected_single_values <- list(
  sample_type = "Primary Tumor",
  access = "open",
  state = "released",
  data_type = "Gene Expression Quantification",
  experimental_strategy = "RNA-Seq",
  analysis_workflow_type = "STAR - Counts"
)

for (column in names(expected_single_values)) {
  observed <- unique(tolower(as.character(metadata[[column]])))
  expected <- tolower(expected_single_values[[column]])
  if (!identical(observed, expected)) {
    stop(
      "Unexpected metadata value in ", column, ": ",
      paste(observed, collapse = ", ")
    )
  }
}

if (!all(metadata$participant_id %in% rownames(binary))) {
  stop("Some selected RNA participants are absent from the mutation dataset.")
}

clinical_columns <- c(
  "GBM", "ASTRO", "OLIGO", "sex", "age",
  "overall_survival", "status"
)

mutation_columns <- colnames(binary)[
  !colnames(binary) %in% clinical_columns
]

if (anyDuplicated(mutation_columns)) {
  stop(
    "The binary dataset contains duplicated mutation-gene columns: ",
    paste(unique(mutation_columns[duplicated(mutation_columns)]),
          collapse = ", ")
  )
}

mutation_genes <- mutation_columns
selected_genes <- mutation_genes
has_alias <- mutation_genes %in% names(gene_symbol_aliases)
selected_genes[has_alias] <- unname(gene_symbol_aliases[mutation_genes[has_alias]])

if (length(selected_genes) == 0) {
  stop("No mutation-gene columns were found.")
}

if (anyDuplicated(selected_genes)) {
  stop(
    "Gene-symbol mapping produced duplicated current symbols: ",
    paste(unique(selected_genes[duplicated(selected_genes)]), collapse = ", ")
  )
}

mutation_values <- binary[metadata$participant_id, mutation_genes, drop = FALSE]
mutation_numeric <- as.data.frame(
  lapply(mutation_values, function(x) suppressWarnings(as.numeric(x))),
  check.names = FALSE
)
rownames(mutation_numeric) <- metadata$participant_id

invalid_mutation <- vapply(
  mutation_numeric,
  function(x) any(!is.na(x) & !x %in% c(0, 1)),
  logical(1)
)

if (any(invalid_mutation)) {
  stop(
    "Mutation columns contain values other than 0/1: ",
    paste(names(invalid_mutation)[invalid_mutation], collapse = ", ")
  )
}

if (anyNA(mutation_numeric)) {
  stop("Mutation matrix contains missing values among selected participants.")
}

# Preserve the clinical dataset order, which is also the intended final order.
participant_order <- rownames(clinical)[
  rownames(clinical) %in% metadata$participant_id
]
metadata <- metadata[match(participant_order, metadata$participant_id), , drop = FALSE]
mutation_numeric <- mutation_numeric[participant_order, , drop = FALSE]

validation_summary <- data.frame(
  check = c(
    "selected_participants",
    "unique_participants",
    "unique_file_ids",
    "exact_dna_rna_matches",
    "primary_tumor_code_01",
    "selected_mutation_genes"
  ),
  value = c(
    nrow(metadata),
    length(unique(metadata$participant_id)),
    length(unique(metadata$file_id)),
    sum(metadata$rna_sample_id == metadata$matching_dna_sample_id),
    sum(substr(metadata$rna_sample_id, 14, 15) == "01"),
    length(selected_genes)
  ),
  stringsAsFactors = FALSE
)

write_tsv(
  validation_summary,
  file.path(output_dir, "00_pre_download_validation_summary.tsv")
)

writeLines(
  selected_genes,
  file.path(output_dir, "00_selected_genes.txt")
)

gene_symbol_map <- data.frame(
  mutation_gene_original = mutation_genes,
  expression_gene_gdc = selected_genes,
  symbol_changed = mutation_genes != selected_genes,
  stringsAsFactors = FALSE
)
write_tsv(
  gene_symbol_map,
  file.path(output_dir, "00_mutation_to_expression_gene_symbol_map.tsv")
)

cat("Pre-download validation passed.\n")
cat("Participants:", nrow(metadata), "\n")
cat("Selected mutation genes:", length(selected_genes), "\n")
cat("Exact DNA-RNA sample matches:",
    sum(metadata$rna_sample_id == metadata$matching_dna_sample_id), "\n")

###############################################################################
# 5. Dry-run exit or streaming download
###############################################################################

if (!RUN_DOWNLOAD) {
  cat(
    "\nDry run only: no RNA-seq files were downloaded.\n",
    "For a 3-file test set RUN_DOWNLOAD <- TRUE and MAX_FILES <- 3.\n",
    "For the full resumable run use RUN_DOWNLOAD <- TRUE and MAX_FILES <- Inf.\n",
    sep = ""
  )
} else {
  number_to_process <- min(nrow(metadata), MAX_FILES)
  rows_to_process <- seq_len(number_to_process)

  cat("\nStreaming", number_to_process, "of", nrow(metadata), "files.\n")

  for (i in rows_to_process) {
    participant <- metadata$participant_id[i]
    checkpoint_file <- checkpoint_path(participant, metadata$file_id[i])
    action <- if (file.exists(checkpoint_file)) "Reusing" else "Downloading"

    cat(
      sprintf(
        "[%d/%d] %s %s\n",
        i, number_to_process, action, participant
      )
    )

    process_star_counts_file(
      metadata_row = metadata[i, , drop = FALSE],
      selected_genes = selected_genes
    )
  }

  expected_checkpoint_files <- file.path(
    checkpoint_dir,
    paste0(
      metadata$participant_id, "__", metadata$file_id, "__",
      CHECKPOINT_SCHEMA, ".rds"
    )
  )
  completed <- file.exists(expected_checkpoint_files)

  progress <- data.frame(
    participant_id = metadata$participant_id,
    file_id = metadata$file_id,
    completed = completed,
    stringsAsFactors = FALSE
  )
  write_tsv(progress, file.path(output_dir, "01_download_progress.tsv"))

  completed_checkpoint_files <- expected_checkpoint_files[completed]
  completed_checkpoints <- lapply(completed_checkpoint_files, readRDS)

  if (length(completed_checkpoints) > 0) {
    partial_gene_qc <- data.frame(
      mutation_gene_original = mutation_genes,
      expression_gene_gdc = selected_genes,
      completed_files = length(completed_checkpoints),
      present_files = vapply(
        selected_genes,
        function(gene) sum(vapply(
          completed_checkpoints,
          function(x) !gene %in% x$missing_genes,
          logical(1)
        )),
        integer(1)
      ),
      missing_files = vapply(
        selected_genes,
        function(gene) sum(vapply(
          completed_checkpoints,
          function(x) gene %in% x$missing_genes,
          logical(1)
        )),
        integer(1)
      ),
      ensembl_gene_ids = vapply(
        seq_along(selected_genes),
        function(index) {
          ids <- unique(vapply(
            completed_checkpoints,
            function(x) x$gene_ids[index],
            character(1)
          ))
          ids <- sort(ids[!is.na(ids) & ids != ""])
          paste(ids, collapse = ";")
        },
        character(1)
      ),
      stringsAsFactors = FALSE
    )

    write_tsv(
      partial_gene_qc,
      file.path(output_dir, "01_partial_gene_presence_qc.tsv")
    )
  }

  cat("\nValid checkpoints:", sum(completed), "of", nrow(metadata), "\n")

  if (!all(completed)) {
    cat(
      "Partial run complete. Rerun with MAX_FILES <- Inf; valid checkpoints ",
      "will be reused and final matrices will be created only after all files ",
      "are complete.\n",
      sep = ""
    )
  } else {
    checkpoints <- lapply(expected_checkpoint_files, readRDS)

    valid_checkpoints <- vapply(
      seq_along(checkpoints),
      function(i) validate_checkpoint(
        checkpoints[[i]],
        metadata[i, , drop = FALSE],
        selected_genes
      ),
      logical(1)
    )

    if (!all(valid_checkpoints)) {
      stop("One or more completed checkpoints failed final validation.")
    }

    counts_matrix <- do.call(rbind, lapply(checkpoints, `[[`, "counts"))
    tpm_matrix <- do.call(rbind, lapply(checkpoints, `[[`, "tpm"))
    fpkm_uq_matrix <- do.call(rbind, lapply(checkpoints, `[[`, "fpkm_uq"))

    rownames(counts_matrix) <- metadata$participant_id
    rownames(tpm_matrix) <- metadata$participant_id
    rownames(fpkm_uq_matrix) <- metadata$participant_id
    colnames(counts_matrix) <- selected_genes
    colnames(tpm_matrix) <- selected_genes
    colnames(fpkm_uq_matrix) <- selected_genes

    if (!identical(dim(counts_matrix), dim(tpm_matrix)) ||
        !identical(dim(tpm_matrix), dim(fpkm_uq_matrix))) {
      stop("Final expression matrices do not have identical dimensions.")
    }

    if (any(tpm_matrix < 0, na.rm = TRUE) ||
        any(!is.finite(tpm_matrix), na.rm = TRUE)) {
      stop("Final TPM matrix contains invalid values.")
    }

    log2_tpm_matrix <- log2(tpm_matrix + 1)

    counts_df <- as.data.frame(counts_matrix, check.names = FALSE)
    tpm_df <- as.data.frame(tpm_matrix, check.names = FALSE)
    log2_tpm_df <- as.data.frame(log2_tpm_matrix, check.names = FALSE)
    fpkm_uq_df <- as.data.frame(fpkm_uq_matrix, check.names = FALSE)

    colnames(counts_df) <- paste0(colnames(counts_df), "_expr_count")
    colnames(tpm_df) <- paste0(colnames(tpm_df), "_expr_tpm")
    colnames(log2_tpm_df) <- paste0(
      colnames(log2_tpm_df), "_expr_log2tpm"
    )
    colnames(fpkm_uq_df) <- paste0(
      colnames(fpkm_uq_df), "_expr_fpkm_uq"
    )

    write_tsv(
      counts_df,
      file.path(output_dir, "02_rnaseq_raw_counts_selected_genes.tsv"),
      row.names = TRUE
    )
    write_tsv(
      tpm_df,
      file.path(output_dir, "03_rnaseq_tpm_selected_genes.tsv"),
      row.names = TRUE
    )
    write_tsv(
      log2_tpm_df,
      file.path(output_dir, "04_rnaseq_log2_tpm_selected_genes.tsv"),
      row.names = TRUE
    )
    write_tsv(
      fpkm_uq_df,
      file.path(output_dir, "05_rnaseq_fpkm_uq_selected_genes.tsv"),
      row.names = TRUE
    )

    mutation_named <- mutation_numeric
    colnames(mutation_named) <- paste0(
      selected_genes, "_mut"
    )

    causal_input <- cbind(
      mutation_named[metadata$participant_id, , drop = FALSE],
      log2_tpm_df[metadata$participant_id, , drop = FALSE]
    )

    write_tsv(
      causal_input,
      file.path(
        output_dir,
        "06_causal_input_mutation_plus_log2_tpm.tsv"
      ),
      row.names = TRUE
    )

    annotation_columns <- intersect(
      c("GBM", "ASTRO", "sex", "age", "overall_survival", "status"),
      unique(c(colnames(binary), colnames(clinical)))
    )

    annotations <- data.frame(
      project = metadata$requested_project,
      sample_id = metadata$rna_sample_id,
      stringsAsFactors = FALSE,
      row.names = metadata$participant_id
    )

    for (column in annotation_columns) {
      if (column %in% colnames(clinical)) {
        annotations[[column]] <- clinical[metadata$participant_id, column]
      } else if (column %in% colnames(binary)) {
        annotations[[column]] <- binary[metadata$participant_id, column]
      }
    }

    write_tsv(
      annotations,
      file.path(output_dir, "07_participant_annotations.tsv"),
      row.names = TRUE
    )

    missing_by_gene <- vapply(
      selected_genes,
      function(gene) sum(vapply(
        checkpoints,
        function(x) gene %in% x$missing_genes,
        logical(1)
      )),
      integer(1)
    )

    expression_qc <- data.frame(
      mutation_gene_original = mutation_genes,
      expression_gene_gdc = selected_genes,
      missing_participants = missing_by_gene,
      non_missing_participants = nrow(metadata) - missing_by_gene,
      mean_log2_tpm = colMeans(log2_tpm_matrix, na.rm = TRUE),
      sd_log2_tpm = apply(log2_tpm_matrix, 2, stats::sd, na.rm = TRUE),
      zero_variance = apply(
        log2_tpm_matrix,
        2,
        function(x) length(unique(x[!is.na(x)])) <= 1
      ),
      stringsAsFactors = FALSE
    )

    write_tsv(
      expression_qc,
      file.path(output_dir, "08_expression_gene_qc.tsv")
    )

    provenance <- data.frame(
      participant_id = metadata$participant_id,
      project = metadata$requested_project,
      sample_id = metadata$rna_sample_id,
      file_id = metadata$file_id,
      file_name = metadata$file_name,
      md5 = metadata$md5sum,
      md5_verified = vapply(
        checkpoints,
        function(x) identical(
          tolower(x$expected_md5),
          tolower(x$observed_md5)
        ),
        logical(1)
      ),
      retrieved_at = vapply(
        checkpoints,
        `[[`,
        character(1),
        "retrieved_at"
      ),
      stringsAsFactors = FALSE
    )

    write_tsv(
      provenance,
      file.path(output_dir, "09_file_provenance_and_md5.tsv")
    )

    methods_text <- c(
      "TCGA RNA-seq selected-gene extraction methods",
      "================================================",
      paste("Run completed (UTC):", format(Sys.time(), tz = "UTC", usetz = TRUE)),
      paste("Participants:", nrow(metadata)),
      paste("Mutation-selected genes:", length(selected_genes)),
      paste("TCGA-GBM participants:", sum(metadata$requested_project == "TCGA-GBM")),
      paste("TCGA-LGG participants:", sum(metadata$requested_project == "TCGA-LGG")),
      "",
      "Inclusion criteria:",
      "- GDC Transcriptome Profiling / Gene Expression Quantification",
      "- RNA-Seq, STAR - Counts, open-access, released files",
      "- Primary Tumor (TCGA sample type code 01)",
      "- One unique participant and one unique file_id",
      "- Exact first-16-character TCGA DNA-RNA sample barcode match",
      "",
      "Integrity and extraction:",
      "- Files were downloaded individually from the GDC data endpoint.",
      "- Downloaded byte size and MD5 were checked against GDC metadata.",
      "- Only selected gene rows were retained; full files were deleted.",
      "- Raw unstranded counts, GDC TPM, and GDC FPKM-UQ were retained.",
      "- Network input uses log2(GDC TPM + 1).",
      "- TPM was calculated by GDC on the full transcriptome before gene filtering.",
      "- Historical mutation symbol MYST4 was mapped explicitly to current GDC symbol KAT6B.",
      "- The original-to-current symbol mapping is recorded in 00_mutation_to_expression_gene_symbol_map.tsv.",
      "",
      "Important limitations:",
      "- This is cross-sectional observational tumor data, not perturbation data.",
      "- Mutation genes were preselected using MutSig2CV q-value and frequency filters.",
      "- Participants without RNA or without an exact MAF sample match were excluded.",
      "- TCGA-GBM and TCGA-LGG project membership may introduce cohort/batch structure.",
      "- WHO subtype labels are molecularly defined and should not be ordinary gene-network nodes.",
      "- Survival time/status require survival methods and are not ordinary continuous nodes.",
      "- Causal directions require biological constraints, stability analysis, and external validation."
    )

    writeLines(
      methods_text,
      file.path(output_dir, "10_METHODS_AND_LIMITATIONS.txt")
    )

    if (!KEEP_CHECKPOINTS) {
      unlink(expected_checkpoint_files)
    }

    cat("\nStreaming extraction complete and fully validated.\n")
    cat("Final participants:", nrow(causal_input), "\n")
    cat("Mutation variables:", ncol(mutation_named), "\n")
    cat("Expression variables:", ncol(log2_tpm_df), "\n")
    cat("Genes missing in any participant:", sum(missing_by_gene > 0), "\n")
    cat("Genes with zero expression variance:",
        sum(expression_qc$zero_variance), "\n")
    cat("Output folder:", output_dir, "\n")
  }
}
