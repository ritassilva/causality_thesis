# 00_download_TCGA_who2021_classification.R

# ============================================================
# Download and clean MONET classification dataset
# Removes rows with "unclassified" in classification.2021
# Creates IDs matching clin_glioma rownames format:
# TCGA-06-6391 -> TCGA.06.6391
# ============================================================

# Create data directory if it doesn't exist
#dir.create("Analysis/data/data_raw/", recursive = TRUE, showWarnings = FALSE)

# Define file paths
url <- "https://raw.githubusercontent.com/sysbiomed/MONET/4361ed6780c1f40d6a36d68085c5d8c68d2fe024/Final-outputs/SIMPLIFIED_CLASSIFICATION_TCGA_2016_2021.csv"

raw_file <- file.path(Sys.getenv("TCGA_WORK_DIR", file.path(getwd(), "results", "new_runs", "tcga_preparation")), "data_raw/SIMPLIFIED_CLASSIFICATION_TCGA_2016_2021.csv")

clean_file <- file.path(Sys.getenv("TCGA_WORK_DIR", file.path(getwd(), "results", "new_runs", "tcga_preparation")), "data_raw/SIMPLIFIED_CLASSIFICATION_TCGA_2016_2021_clean.csv")

# ------------------------------------------------------------
# Step 1 — Download dataset (only if not already present)
# ------------------------------------------------------------

dir.create(dirname(raw_file), recursive = TRUE, showWarnings = FALSE)

if (!file.exists(raw_file)) {
  message("Downloading dataset...")
  download.file(
    url = url,
    destfile = raw_file,
    method = "auto",
    mode = "wb"
  )
} else {
  message("Raw dataset already exists. Skipping download.")
}

# ------------------------------------------------------------
# Step 2 — Load dataset
# ------------------------------------------------------------

classification <- read.csv(
  raw_file,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

# ------------------------------------------------------------
# Step 3 — Remove unclassified samples
# ------------------------------------------------------------

classification_clean <- classification[
  !is.na(classification$classification.2021) &
    tolower(classification$classification.2021) != "unclassified",
]

# ------------------------------------------------------------
# Step 4 — Create ID matching clin_glioma rownames
# ------------------------------------------------------------

classification_clean$Patient_ID_clin_glioma <- gsub(
  "-",
  ".",
  classification_clean$Patient_ID
)

# Optional: move new ID column to the front
classification_clean <- classification_clean[
  ,
  c("Patient_ID_clin_glioma", setdiff(names(classification_clean), "Patient_ID_clin_glioma"))
]

# ------------------------------------------------------------
# Step 5 — Save cleaned dataset
# ------------------------------------------------------------

write.csv(
  classification_clean,
  clean_file,
  row.names = FALSE
)

# ------------------------------------------------------------
# Step 6 — Report summary
# ------------------------------------------------------------

n_removed <- nrow(classification) - nrow(classification_clean)

message("Original rows: ", nrow(classification))
message("Removed rows: ", n_removed)
message("Remaining rows: ", nrow(classification_clean))
message("Clean dataset saved to: ", clean_file)
message("Added matching ID column: Patient_ID_clin_glioma")