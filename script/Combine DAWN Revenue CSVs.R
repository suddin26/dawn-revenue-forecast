# Combine DAWN Revenue CSVs
# -------------------------
# This script combines the per-budget-account files into one master CSV.
#
# Input files:
#   dawn_revenue_accounts_9010_history.csv
#   dawn_revenue_accounts_9040_history.csv
#   etc.
#
# It intentionally does NOT include all_revenue_accounts_history.csv because that
# file overlaps with BA 9130 and would duplicate transactions.
#
# Output file:
#   data/dawn_revenue_accounts_combined_history.csv

required_packages <- c("data.table")
missing_packages <- required_packages[!required_packages %in% rownames(installed.packages())]

if (length(missing_packages) > 0) {
  stop(
    "Missing required package(s): ",
    paste(missing_packages, collapse = ", "),
    ". Install them first, then rerun this script.",
    call. = FALSE
  )
}

library(data.table)

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", script_arg[1])))
} else if (dir.exists("data") && dir.exists("script")) {
  normalizePath(file.path(getwd(), "script"))
} else if (dir.exists(file.path("..", "data"))) {
  normalizePath(getwd())
} else {
  stop("Could not find the project data folder. Run this script from the project root or script folder.", call. = FALSE)
}
project_dir <- if (basename(script_dir) == "script") {
  normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
} else {
  script_dir
}
data_dir <- file.path(project_dir, "data")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

input_pattern <- "^dawn_revenue_accounts_[0-9]+_history\\.csv$"
output_file <- file.path(data_dir, "dawn_revenue_accounts_combined_history.csv")

expected_columns <- c(
  "Doc Number",
  "Date",
  "Amount",
  "level_1_id",
  "level_2_id",
  "GL Number",
  "description",
  "Type",
  "BA Number",
  "fy"
)

input_files <- list.files(data_dir, pattern = input_pattern, full.names = TRUE)

if (length(input_files) == 0) {
  stop("No per-budget-account CSV files found.", call. = FALSE)
}

message("Found ", length(input_files), " files to combine:")
message(paste(" -", basename(input_files), collapse = "\n"))

read_revenue_file <- function(file) {
  data <- fread(file)

  # Compatibility with older outputs, in case a file still has the old names.
  if ("ba" %in% names(data) && !"BA Number" %in% names(data)) {
    setnames(data, "ba", "BA Number")
  }

  if ("revenue_account" %in% names(data) && !"GL Number" %in% names(data)) {
    setnames(data, "revenue_account", "GL Number")
  }

  missing_columns <- setdiff(expected_columns, names(data))

  if (length(missing_columns) > 0) {
    stop(
      "File ",
      file,
      " is missing expected column(s): ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }

  data[, ..expected_columns]
}

combined_data <- rbindlist(
  lapply(input_files, read_revenue_file),
  use.names = TRUE,
  fill = FALSE
)

# Drop exact duplicate rows, if any file was accidentally copied or rerun into
# another matching filename.
combined_data <- unique(combined_data)

setorder(combined_data, `BA Number`, fy, `GL Number`, Date, `Doc Number`)
fwrite(combined_data, output_file)

message("Saved ", nrow(combined_data), " rows to ", output_file)
