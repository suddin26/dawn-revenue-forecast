script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", script_arg[1])))
} else if (file.exists("DAWN Revenue Download Helper.R")) {
  normalizePath(getwd())
} else if (file.exists(file.path("script", "DAWN Revenue Download Helper.R"))) {
  normalizePath(file.path(getwd(), "script"))
} else {
  stop("Could not find DAWN Revenue Download Helper.R. Run this script from the project root or script folder.", call. = FALSE)
}
project_dir <- if (basename(script_dir) == "script") {
  normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
} else {
  script_dir
}
data_dir <- file.path(project_dir, "data")

source(file.path(script_dir, "DAWN Revenue Download Helper.R"))

# Edit these values if needed:
# - budget_account_list: budget account(s) to download.
# - fy_list: fiscal years to include.
# - output_file: CSV file that will be written.
run_dawn_revenue_download(
  budget_account_list = c(9050),
  fy_list = c(2000:current_fiscal_year()),
  output_file = file.path(data_dir, "dawn_revenue_accounts_9050_history.csv"),
  type_file = file.path(project_dir, "unique_gl_desc_type2.xlsx")
)
