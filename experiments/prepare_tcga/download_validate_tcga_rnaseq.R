#!/usr/bin/env Rscript

###############################################################################
# Validate and download TCGA-GBM/TCGA-LGG RNA-seq for the participants already
# present in the processed glioma datasets.
#
# Phase 1 (default): query GDC, validate metadata, report overlap and duplicates.
# Phase 2: set RUN_DOWNLOAD <- TRUE, then download and create selected-gene
#          raw-count and log2(TPM + 1) matrices.
###############################################################################

options(stringsAsFactors = FALSE)

###############################################################################
# 1. User settings
###############################################################################

project_dir <- Sys.getenv("TCGA_WORK_DIR", file.path(getwd(), "results", "new_runs", "tcga_preparation"))

clinical_file <- file.path(
  project_dir,
  "glioma_firebrowse_who21",
  "clinical_dataset_glioma.txt"
)

binary_file <- file.path(
  project_dir,
  "glioma_firebrowse_who21",
  "binary_dataset_glioma.txt"
)

output_dir <- file.path(project_dir, "tcga_rnaseq_selected_genes")
download_dir <- file.path(output_dir, "GDCdata")

dna_matched_metadata_file <- file.path(
  output_dir,
  "dna_rna_sample_validation",
  "rna_files_selected_after_dna_match.tsv"
)

# Keep FALSE for the first run. Inspect the validation reports, then change to
# TRUE only when the query and participant overlap are correct.
RUN_DOWNLOAD <- FALSE

# Duplicate resolution rule: when GDC contains more than one released
# STAR-Counts result for the same participant, retain the file with the most
# recent created_datetime (then updated_datetime as a tie-breaker). The full
# decision table is saved before download.
SELECT_LATEST_CREATED_FILE <- TRUE

# For the integrated mutation-expression network, require RNA and DNA to come
# from the same TCGA sample (first 16 barcode characters).
REQUIRE_EXACT_DNA_RNA_SAMPLE_MATCH <- TRUE

###############################################################################
# 2. Package checks
###############################################################################

required_packages <- c("TCGAbiolinks", "SummarizedExperiment")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    "Missing packages: ", paste(missing_packages, collapse = ", "), "\n",
    "Install them with:\n",
    "if (!requireNamespace('BiocManager', quietly = TRUE)) ",
    "install.packages('BiocManager')\n",
    "BiocManager::install(c('TCGAbiolinks', 'SummarizedExperiment'))"
  )
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

###############################################################################
# 3. Helper functions
###############################################################################

normalize_participant_id <- function(x) {
  x <- toupper(as.character(x))
  x <- gsub("\\.", "-", x)

  match_position <- regexpr(
    "TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}",
    x,
    perl = TRUE
  )

  out <- rep(NA_character_, length(x))
  has_match <- match_position > 0
  out[has_match] <- regmatches(x, match_position)
  out
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

  patient_ids <- normalize_participant_id(rownames(df))

  if (anyNA(patient_ids)) {
    stop(
      "Could not interpret these row names as TCGA participant IDs in ",
      basename(file), ": ",
      paste(head(rownames(df)[is.na(patient_ids)], 10), collapse = ", ")
    )
  }

  if (anyDuplicated(patient_ids)) {
    stop("Duplicated participant IDs in ", basename(file))
  }

  rownames(df) <- patient_ids
  df
}

write_tsv_safe <- function(df, file, row.names = FALSE) {
  df <- as.data.frame(df, check.names = FALSE)

  # Convert possible list columns returned by the API to readable strings.
  list_columns <- vapply(df, is.list, logical(1))
  if (any(list_columns)) {
    df[list_columns] <- lapply(
      df[list_columns],
      function(x) vapply(x, paste, character(1), collapse = ";")
    )
  }

  write.table(
    df,
    file = file,
    sep = "\t",
    quote = FALSE,
    row.names = row.names,
    col.names = if (row.names) NA else TRUE,
    na = "NA"
  )
}

make_expression_query <- function(project, barcodes = NULL) {
  args <- list(
    project = project,
    data.category = "Transcriptome Profiling",
    data.type = "Gene Expression Quantification",
    experimental.strategy = "RNA-Seq",
    workflow.type = "STAR - Counts",
    sample.type = "Primary Tumor"
  )

  if (!is.null(barcodes)) {
    args$barcode <- unique(barcodes)
  }

  do.call(TCGAbiolinks::GDCquery, args)
}

query_metadata <- function(project, query) {
  metadata <- TCGAbiolinks::getResults(query)
  metadata$requested_project <- project

  if (!"cases" %in% colnames(metadata)) {
    stop(
      "The GDC result does not contain a 'cases' column. Available columns: ",
      paste(colnames(metadata), collapse = ", ")
    )
  }

  metadata$participant_id <- normalize_participant_id(metadata$cases)

  if (anyNA(metadata$participant_id)) {
    stop(
      "Could not extract participant IDs from some GDC 'cases' values in ",
      project
    )
  }

  metadata
}

print_query_summary <- function(metadata) {
  summary_columns <- intersect(
    c(
      "requested_project",
      "project",
      "sample_type",
      "data_type",
      "experimental_strategy",
      "workflow_type",
      "access"
    ),
    colnames(metadata)
  )

  for (column in summary_columns) {
    cat("\n", column, ":\n", sep = "")
    print(table(metadata[[column]], useNA = "ifany"))
  }
}

select_one_file_per_participant <- function(metadata) {
  required <- c("participant_id", "created_datetime", "updated_datetime")
  missing <- setdiff(required, colnames(metadata))

  if (length(missing) > 0) {
    stop(
      "Cannot resolve duplicate files because metadata columns are missing: ",
      paste(missing, collapse = ", ")
    )
  }

  groups <- split(metadata, metadata$participant_id)

  selected <- lapply(groups, function(x) {
    created <- ifelse(is.na(x$created_datetime), "", x$created_datetime)
    updated <- ifelse(is.na(x$updated_datetime), "", x$updated_datetime)
    file_id <- if ("file_id" %in% colnames(x)) {
      ifelse(is.na(x$file_id), "", x$file_id)
    } else {
      rep("", nrow(x))
    }

    # ISO-8601 timestamps sort chronologically as strings here. file_id is only
    # used to make a hypothetical exact tie deterministic and reproducible.
    order_index <- order(created, updated, file_id, decreasing = TRUE)
    chosen <- x[order_index[1], , drop = FALSE]
    chosen$files_available_for_participant <- nrow(x)
    chosen$selection_rule <- if (nrow(x) == 1) {
      "only_available_file"
    } else {
      "latest_created_datetime"
    }
    chosen
  })

  selected <- do.call(rbind, selected)
  rownames(selected) <- NULL
  selected
}

pick_assay <- function(se, kind = c("counts", "tpm")) {
  kind <- match.arg(kind)
  available <- SummarizedExperiment::assayNames(se)

  if (kind == "counts") {
    exact <- available[tolower(available) == "unstranded"]
    if (length(exact) > 0) {
      return(exact[1])
    }

    candidates <- available[
      grepl("count|unstranded", available, ignore.case = TRUE) &
        !grepl("tpm|fpkm", available, ignore.case = TRUE)
    ]
  } else {
    candidates <- available[grepl("tpm", available, ignore.case = TRUE)]
  }

  if (length(candidates) == 0) {
    stop(
      "Could not find a ", kind, " assay. Available assays: ",
      paste(available, collapse = ", ")
    )
  }

  candidates[1]
}

extract_selected_assay <- function(se, selected_genes, kind) {
  assay_name <- pick_assay(se, kind)
  message("Using assay '", assay_name, "' for ", kind)

  row_metadata <- as.data.frame(SummarizedExperiment::rowData(se))
  gene_column <- intersect(
    c("gene_name", "gene_symbol", "external_gene_name"),
    colnames(row_metadata)
  )[1]

  if (is.na(gene_column)) {
    stop(
      "Could not find a gene-symbol column in rowData. Available columns: ",
      paste(colnames(row_metadata), collapse = ", ")
    )
  }

  gene_symbols <- as.character(row_metadata[[gene_column]])
  selected_counts <- table(gene_symbols[gene_symbols %in% selected_genes])
  duplicated_selected <- names(selected_counts)[selected_counts > 1]

  if (length(duplicated_selected) > 0) {
    stop(
      "Some selected gene symbols occur more than once in RNA-seq rowData: ",
      paste(duplicated_selected, collapse = ", "),
      ". Inspect Ensembl IDs before choosing how to combine them."
    )
  }

  row_index <- match(selected_genes, gene_symbols)
  present <- !is.na(row_index)

  expression_matrix <- SummarizedExperiment::assay(se, assay_name)[
    row_index[present],
    ,
    drop = FALSE
  ]
  rownames(expression_matrix) <- selected_genes[present]

  list(
    matrix = expression_matrix,
    missing_genes = selected_genes[!present],
    assay_name = assay_name
  )
}

combine_project_assays <- function(gbm, lgg, selected_genes) {
  common_genes <- selected_genes[
    selected_genes %in% rownames(gbm) & selected_genes %in% rownames(lgg)
  ]

  if (length(common_genes) == 0) {
    stop("No selected genes are shared by the GBM and LGG RNA-seq objects.")
  }

  combined <- cbind(
    gbm[common_genes, , drop = FALSE],
    lgg[common_genes, , drop = FALSE]
  )

  participant_ids <- normalize_participant_id(colnames(combined))

  if (anyNA(participant_ids)) {
    stop("Could not extract participant IDs from prepared RNA-seq columns.")
  }

  if (anyDuplicated(participant_ids)) {
    duplicated_ids <- unique(participant_ids[duplicated(participant_ids)])
    stop(
      "Prepared RNA-seq still has multiple samples for participants: ",
      paste(duplicated_ids, collapse = ", ")
    )
  }

  colnames(combined) <- participant_ids
  combined
}

###############################################################################
# 4. Read the current processed datasets and identify mutation genes
###############################################################################

clinical_data <- read_patient_table(clinical_file)
binary_data <- read_patient_table(binary_file)

if (!setequal(rownames(clinical_data), rownames(binary_data))) {
  stop("Clinical and binary datasets do not contain the same participants.")
}

clinical_columns <- c(
  "GBM", "ASTRO", "OLIGO", "sex", "age",
  "overall_survival", "status"
)

mutation_genes <- setdiff(colnames(binary_data), clinical_columns)
mutation_genes <- unique(mutation_genes)

if (length(mutation_genes) == 0) {
  stop("No mutation-gene columns were found in the binary dataset.")
}

wanted_participants <- rownames(clinical_data)

writeLines(
  mutation_genes,
  file.path(output_dir, "selected_mutation_genes.txt")
)

cat("Current clinical participants:", length(wanted_participants), "\n")
cat("Selected mutation genes:", length(mutation_genes), "\n")

###############################################################################
# 5. Phase 1: query and validate metadata without downloading
###############################################################################

projects <- c("TCGA-GBM", "TCGA-LGG")
initial_queries <- setNames(vector("list", length(projects)), projects)
initial_metadata <- setNames(vector("list", length(projects)), projects)

for (project in projects) {
  cat("\nQuerying ", project, " metadata...\n", sep = "")
  initial_queries[[project]] <- make_expression_query(project)
  initial_metadata[[project]] <- query_metadata(
    project,
    initial_queries[[project]]
  )
}

all_metadata <- do.call(rbind, initial_metadata)
rownames(all_metadata) <- NULL

write_tsv_safe(
  all_metadata,
  file.path(output_dir, "gdc_query_all_primary_tumor_star_counts.tsv")
)

print_query_summary(all_metadata)

available_participants <- unique(all_metadata$participant_id)
matching_participants <- intersect(wanted_participants, available_participants)
missing_participants <- setdiff(wanted_participants, available_participants)

overlap_report <- data.frame(
  participant_id = wanted_participants,
  has_primary_tumor_rnaseq = wanted_participants %in% matching_participants,
  stringsAsFactors = FALSE
)

write_tsv_safe(
  overlap_report,
  file.path(output_dir, "participant_rnaseq_overlap.tsv")
)

writeLines(
  missing_participants,
  file.path(output_dir, "participants_without_primary_tumor_rnaseq.txt")
)

matching_metadata <- all_metadata[
  all_metadata$participant_id %in% matching_participants,
  ,
  drop = FALSE
]

write_tsv_safe(
  matching_metadata,
  file.path(output_dir, "gdc_query_matching_participants.tsv")
)

participant_file_counts <- table(matching_metadata$participant_id)
duplicate_participants <- names(participant_file_counts)[
  participant_file_counts > 1
]

writeLines(
  duplicate_participants,
  file.path(output_dir, "participants_with_multiple_rnaseq_files.txt")
)

cat("\nParticipants requested:", length(wanted_participants), "\n")
cat("Participants with primary-tumor RNA-seq:", length(matching_participants), "\n")
cat("Participants without primary-tumor RNA-seq:", length(missing_participants), "\n")
cat("Participants with multiple matching files:", length(duplicate_participants), "\n")
cat("Validation reports written to:", output_dir, "\n")

if (!SELECT_LATEST_CREATED_FILE && length(duplicate_participants) > 0) {
  stop(
    "Multiple primary-tumor RNA-seq files were found for some participants. ",
    "Either inspect them manually or set SELECT_LATEST_CREATED_FILE <- TRUE."
  )
}

latest_metadata <- select_one_file_per_participant(matching_metadata)

if (REQUIRE_EXACT_DNA_RNA_SAMPLE_MATCH) {
  if (!file.exists(dna_matched_metadata_file)) {
    stop(
      "Exact DNA-RNA sample matching is required, but this file is missing: ",
      dna_matched_metadata_file,
      "\nRun validate_tcga_dna_rna_sample_match.R first."
    )
  }

  selected_metadata <- read.delim(
    dna_matched_metadata_file,
    header = TRUE,
    check.names = FALSE,
    quote = "",
    comment.char = ""
  )

  required_matched_columns <- c(
    "participant_id",
    "cases",
    "file_id",
    "requested_project",
    "rna_sample_id",
    "matching_dna_sample_id"
  )

  missing_matched_columns <- setdiff(
    required_matched_columns,
    colnames(selected_metadata)
  )

  if (length(missing_matched_columns) > 0) {
    stop(
      "DNA-matched RNA metadata is missing columns: ",
      paste(missing_matched_columns, collapse = ", ")
    )
  }

  selected_metadata$participant_id <- normalize_participant_id(
    selected_metadata$participant_id
  )

  if (any(selected_metadata$rna_sample_id !=
          selected_metadata$matching_dna_sample_id)) {
    stop("DNA-matched metadata contains a non-exact RNA-DNA sample pair.")
  }

  known_file_ids <- unique(matching_metadata$file_id)
  unknown_file_ids <- setdiff(selected_metadata$file_id, known_file_ids)

  if (length(unknown_file_ids) > 0) {
    stop(
      "DNA-matched metadata contains file IDs absent from the current GDC query: ",
      paste(unknown_file_ids, collapse = ", ")
    )
  }
} else {
  selected_metadata <- latest_metadata
}

if (anyDuplicated(selected_metadata$participant_id)) {
  stop("Duplicate resolution failed: selected metadata still has repeated participants.")
}

write_tsv_safe(
  selected_metadata,
  file.path(output_dir, "gdc_query_selected_one_file_per_participant.tsv")
)

excluded_no_exact_match <- setdiff(
  wanted_participants,
  selected_metadata$participant_id
)

writeLines(
  excluded_no_exact_match,
  file.path(output_dir, "participants_excluded_without_exact_dna_rna_match.txt")
)

cat(
  "Files selected after duplicate resolution:",
  nrow(selected_metadata),
  "\n"
)
cat(
  "Participants excluded without an exact DNA-RNA sample match:",
  length(excluded_no_exact_match),
  "\n"
)

###############################################################################
# 6. Phase 2: download, prepare and save selected-gene matrices
###############################################################################

if (!RUN_DOWNLOAD) {
  cat(
    "\nValidation only: no files were downloaded.\n",
    "Inspect the reports above. If they are correct, set ",
    "RUN_DOWNLOAD <- TRUE and run this script again.\n",
    sep = ""
  )
} else {
  if (length(matching_participants) == 0) {
    stop("No current participants have matching primary-tumor RNA-seq files.")
  }

  dir.create(download_dir, recursive = TRUE, showWarnings = FALSE)

  selected_queries <- setNames(vector("list", length(projects)), projects)
  prepared_objects <- setNames(vector("list", length(projects)), projects)

  for (project in projects) {
    project_metadata <- selected_metadata[
        selected_metadata$requested_project == project
      ,
      drop = FALSE
    ]

    project_participants <- unique(project_metadata$participant_id)
    project_barcodes <- unique(project_metadata$cases)

    if (length(project_participants) == 0) {
      stop("No matching participants were found for ", project)
    }

    cat("\nCreating selected query for ", project, "...\n", sep = "")
    selected_queries[[project]] <- make_expression_query(
      project,
      barcodes = project_barcodes
    )

    selected_query_metadata <- query_metadata(
      project,
      selected_queries[[project]]
    )

    unexpected <- setdiff(
      unique(selected_query_metadata$participant_id),
      project_participants
    )

    if (length(unexpected) > 0) {
      stop(
        "Selected GDC query returned unexpected participants for ", project,
        ": ", paste(unexpected, collapse = ", ")
      )
    }

    repeated_selected <- names(which(
      table(selected_query_metadata$participant_id) > 1
    ))

    if (length(repeated_selected) > 0) {
      stop(
        "The full-barcode query still returned multiple files for: ",
        paste(repeated_selected, collapse = ", ")
      )
    }

    if ("file_id" %in% colnames(selected_query_metadata) &&
        "file_id" %in% colnames(project_metadata)) {
      unexpected_file_ids <- setdiff(
        selected_query_metadata$file_id,
        project_metadata$file_id
      )
      missing_file_ids <- setdiff(
        project_metadata$file_id,
        selected_query_metadata$file_id
      )

      if (length(unexpected_file_ids) > 0 || length(missing_file_ids) > 0) {
        stop(
          "The selected GDC query did not reproduce the expected file IDs for ",
          project,
          ". Inspect the selected metadata before downloading."
        )
      }
    }

    cat("Downloading ", project, " RNA-seq...\n", sep = "")
    TCGAbiolinks::GDCdownload(
      selected_queries[[project]],
      directory = download_dir,
      method = "api",
      files.per.chunk = 20
    )

    cat("Preparing ", project, " RNA-seq...\n", sep = "")
    prepared_objects[[project]] <- TCGAbiolinks::GDCprepare(
      selected_queries[[project]],
      directory = download_dir,
      summarizedExperiment = TRUE
    )

    saveRDS(
      prepared_objects[[project]],
      file.path(output_dir, paste0(project, "_STAR_Counts.rds"))
    )
  }

  counts_gbm <- extract_selected_assay(
    prepared_objects[["TCGA-GBM"]], mutation_genes, "counts"
  )
  counts_lgg <- extract_selected_assay(
    prepared_objects[["TCGA-LGG"]], mutation_genes, "counts"
  )
  tpm_gbm <- extract_selected_assay(
    prepared_objects[["TCGA-GBM"]], mutation_genes, "tpm"
  )
  tpm_lgg <- extract_selected_assay(
    prepared_objects[["TCGA-LGG"]], mutation_genes, "tpm"
  )

  missing_gene_report <- data.frame(
    project = c(
      rep("TCGA-GBM", length(tpm_gbm$missing_genes)),
      rep("TCGA-LGG", length(tpm_lgg$missing_genes))
    ),
    gene = c(tpm_gbm$missing_genes, tpm_lgg$missing_genes),
    stringsAsFactors = FALSE
  )

  write_tsv_safe(
    missing_gene_report,
    file.path(output_dir, "selected_genes_missing_from_rnaseq.tsv")
  )

  combined_counts <- combine_project_assays(
    counts_gbm$matrix,
    counts_lgg$matrix,
    mutation_genes
  )
  combined_tpm <- combine_project_assays(
    tpm_gbm$matrix,
    tpm_lgg$matrix,
    mutation_genes
  )

  final_participants <- wanted_participants[
    wanted_participants %in% colnames(combined_counts) &
      wanted_participants %in% colnames(combined_tpm)
  ]

  combined_counts <- combined_counts[, final_participants, drop = FALSE]
  combined_tpm <- combined_tpm[, final_participants, drop = FALSE]

  counts_by_patient <- as.data.frame(
    t(combined_counts),
    check.names = FALSE
  )
  log2_tpm_by_patient <- as.data.frame(
    t(log2(combined_tpm + 1)),
    check.names = FALSE
  )

  colnames(counts_by_patient) <- paste0(
    colnames(counts_by_patient), "_expr_count"
  )
  colnames(log2_tpm_by_patient) <- paste0(
    colnames(log2_tpm_by_patient), "_expr_log2tpm"
  )

  write_tsv_safe(
    counts_by_patient,
    file.path(output_dir, "rnaseq_raw_counts_selected_genes.tsv"),
    row.names = TRUE
  )

  write_tsv_safe(
    log2_tpm_by_patient,
    file.path(output_dir, "rnaseq_log2_tpm_selected_genes.tsv"),
    row.names = TRUE
  )

  cat("\nDownload and preparation complete.\n")
  cat("Final participants:", length(final_participants), "\n")
  cat("Final selected expression genes:", ncol(log2_tpm_by_patient), "\n")
  cat("Output folder:", output_dir, "\n")
}
