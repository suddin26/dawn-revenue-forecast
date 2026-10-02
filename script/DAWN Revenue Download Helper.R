# DAWN Revenue Download Helper
# ----------------------------
# This file contains the reusable functions used by each budget-account script.
# Usually you do not need to edit this helper. To change which budget account,
# fiscal years, or output file to run, edit one of the small scripts named:
#   Get DAWN Revenues BA ####.R
#
# Output columns:
#   Doc Number, Date, Amount, level_1_id, level_2_id, GL Number,
#   description, Type, BA Number, fy
#
# Notes:
# - "description" comes from the DAWN Receipts/Funding summary page.
# - "Type" comes from unique_gl_desc_type2.xlsx by matching GL Number.
# - Spreadsheet "GL Number" maps to internal column "revenue_account" and final
#   output column "GL Number".
# - Rows with GL Numbers not found in the spreadsheet will have Type = NA.

required_packages <- c("rvest", "data.table", "tidyverse", "purrr", "readxl")
missing_packages <- required_packages[!required_packages %in% rownames(installed.packages())]

# The script checks for packages but does not install them automatically.
# This avoids failures on computers where the R library folder is not writable.
if (length(missing_packages) > 0) {
  stop(
    "Missing required package(s): ",
    paste(missing_packages, collapse = ", "),
    ". Install them first, then rerun this script.",
    call. = FALSE
  )
}

library(rvest)
library(data.table)
library(tidyverse)
library(purrr)
library(readxl)

current_fiscal_year <- function(today = Sys.Date(), fiscal_year_start_month = 7L) {
  calendar_year <- as.integer(format(today, "%Y"))
  calendar_month <- as.integer(format(today, "%m"))

  if (calendar_month >= fiscal_year_start_month) {
    calendar_year + 1L
  } else {
    calendar_year
  }
}

#==============================================================#
# Extract the GL/Type lookup table from the Excel file.
#==============================================================#

# The spreadsheet must include at least these two columns:
# - GL Number
# - Type
#
# The description column can exist in the spreadsheet, but DAWN remains the
# source for the output "description" column.
read_revenue_tracking_types <- function(type_file = "unique_gl_desc_type2.xlsx") {
  if (!file.exists(type_file)) {
    alternate_type_file <- file.path("..", type_file)
    if (file.exists(alternate_type_file)) {
      type_file <- alternate_type_file
    }
  }

  if (!file.exists(type_file)) {
    message("Revenue Type lookup file not found: ", type_file)
    return(tibble(revenue_account = integer(0), Type = character(0)))
  }

  type_lookup <- read_excel(type_file)
  required_columns <- c("GL Number", "Type")
  missing_columns <- setdiff(required_columns, names(type_lookup))

  if (length(missing_columns) > 0) {
    stop(
      "Type lookup file ",
      type_file,
      " is missing expected column(s): ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }

  type_lookup |>
    transmute(
      revenue_account = as.integer(`GL Number`),
      Type = str_squish(Type)
    ) |>
    filter(!is.na(revenue_account)) |>
    distinct(revenue_account, Type) |>
    # Some GL Numbers appear more than once because descriptions can differ.
    # If Type is repeated, keep one value. If the spreadsheet ever has
    # conflicting Types for the same GL Number, keep both separated
    # by "; " so the issue is visible in the output.
    group_by(revenue_account) |>
    summarize(
      Type = {
        type_values <- sort(unique(na.omit(Type)))
        if (length(type_values) == 0) {
          NA_character_
        } else {
          paste(type_values, collapse = "; ")
        }
      },
      .groups = "drop"
    )
}

#==============================================================#
# Step 1: for a budget account and fiscal year, find DAWN's level_1_id.
##==============================================================#

# DAWN uses internal level IDs in the links. The budget account and fiscal year
# are human-friendly inputs; level_1_id is needed to get the revenue account list.
get_level_1_ids <- function(budget_account, fiscal_year) {
  url <- paste0(
    "http://dawn12.state.nv.us:7778/pls/prodsw/bsr_gen_bbls_report?",
    "input_budget_account=", budget_account,
    "&input_fund=101",
    "&input_fiscal_year=", fiscal_year
  )

  webpage <- read_html(url)
  links <- webpage |>
    html_nodes("a") |>
    html_attr("href")

  target_link <- links[str_detect(links, "bsr_rec_fund_sum")]

  if (length(target_link) == 0) {
    message("No revenue link found for BA ", budget_account, " FY ", fiscal_year)
    return(NULL)
  }

  tibble(
    level_1_id = str_match(target_link[1], "level1_id=(\\d+)")[, 2],
    ba = as.integer(budget_account),
    fy = as.integer(fiscal_year)
  )
}

#==============================================================#
# Step 2: for one level_1_id, collect all revenue accounts shown by DAWN.
##==============================================================#

# This returns:
# - level_2_id: DAWN's internal ID used to download transaction detail.
# - revenue_account: the GL/revenue account code, such as 3001.
# - description: DAWN's account description, such as SALES AND USE TAXES.
get_level_2_ids <- function(level_1_id, ba, fy) {
  url <- paste0(
    "http://dawn12.state.nv.us:7778/pls/prodsw/bsr_rec_fund_sum?",
    "pm_level1_id=", level_1_id
  )

  parsed_html <- read_html(url)

  # DAWN's visible summary table has Code and Description columns.
  revenue_summary <- parsed_html |>
    html_table(fill = TRUE) |>
    keep(~ all(c("Code", "Description") %in% names(.x))) |>
    first() |>
    transmute(
      revenue_account = as.character(Code),
      description = str_squish(Description)
    )

  # The actual transaction detail links contain pm_level2_id.
  # We join this ID back to the visible account code and description.
  level_2_links <- parsed_html |>
    html_nodes("td a") |>
    map_dfr(function(node) {
      href <- html_attr(node, "href")
      level_2_id <- str_match(href, "pm_level2_id=(\\d+)")[, 2]
      revenue_account <- str_trim(html_text(node))

      if (is.na(level_2_id) || !str_detect(revenue_account, "^\\d+$")) {
        return(NULL)
      }

      tibble(
        level_2_id = level_2_id,
        revenue_account = revenue_account
      )
    })

  level_2_links |>
    left_join(revenue_summary, by = "revenue_account") |>
    mutate(
      level_1_id = as.character(level_1_id),
      ba = as.integer(ba),
      fy = as.integer(fy)
    )
}

#==============================================================#
# Step 3: download transaction detail for one revenue account/year record.
##==============================================================#

# DAWN may return Amount as character or numeric depending on the page. The
# columns are converted to character here so all tables can be combined safely;
# Amount is converted to numeric later after all rows are bound together.
download_transaction_data <- function(level_2_id, revenue_account, description, level_1_id, ba, fy, to_date) {
  url <- paste0(
    "http://dawn12.state.nv.us:7778/pls/prodsw/bsr_rev_det?",
    "pm_level1_id=", level_1_id,
    "&pm_level2_id=", level_2_id,
    "&pm_level3_id=",
    "&pm_from_date=01/01/2000",
    "&pm_to_date=", to_date
  )

  data <- tryCatch(
    {
      tables <- read_html(url) |>
        html_table(fill = TRUE)

      detail_tables <- keep(
        tables,
        ~ all(c("Doc Number", "Date", "Amount") %in% names(.x))
      )

      if (length(detail_tables) == 0) {
        stop("No transaction detail table found")
      }

      detail_tables[[1]] |>
        mutate(
          `Doc Number` = as.character(`Doc Number`),
          Date = as.character(Date),
          Amount = as.character(Amount)
        )
    },
    error = function(e) {
      message(
        "Error downloading BA ", ba,
        " FY ", fy,
        " revenue_account ", revenue_account,
        " description ", description,
        " level1_id=", level_1_id,
        " level2_id=", level_2_id,
        ": ", conditionMessage(e)
      )
      return(NULL)
    }
  )

  if (is.null(data)) {
    return(NULL)
  }

  data |>
    mutate(
      level_1_id = level_1_id,
      level_2_id = level_2_id,
      revenue_account = revenue_account,
      description = description,
      ba = ba,
      fy = fy
    )
}

#==============================================================#
# Main function used by the small per-budget-account scripts.
##==============================================================#

# Arguments you are most likely to change:
# - budget_account_list: one or more budget accounts, e.g. c(9130) or c(9010, 9130).
# - fy_list: fiscal years to include, e.g. c(2000:current_fiscal_year()).
# - output_file: CSV file to write.
# - revenue_account_min / revenue_account_max: optional filters for GL range.
#   Leave both as NULL to download all revenue accounts listed by DAWN.
# - type_file: Excel file containing GL Number and Type.
# - to_date: defaults to today's date; set manually for reproducible historical runs.

run_dawn_revenue_download <- function(
    budget_account_list,
    fy_list = c(2000:current_fiscal_year()),
    output_file,
    revenue_account_min = NULL,
    revenue_account_max = NULL,
    type_file = "unique_gl_desc_type2.xlsx",
    to_date = format(Sys.Date(), "%m/%d/%Y")) {
  # Build the Type lookup once at the start. It is joined after transaction rows
  # are downloaded and cleaned.
  type_lookup <- read_revenue_tracking_types(type_file)

  # Get DAWN level_1_id for each budget account/year combination.
  level_1_ids <- expand_grid(
    budget_account = budget_account_list,
    fiscal_year = fy_list
  ) |>
    pmap_dfr(~ get_level_1_ids(..1, ..2))

  if (nrow(level_1_ids) == 0) {
    stop("No level 1 revenue links found.", call. = FALSE)
  }

  # Get every DAWN revenue account listed under each level_1_id.
  level_2_ids <- level_1_ids |>
    pmap_dfr(~ get_level_2_ids(..1, ..2, ..3)) |>
    mutate(revenue_account_number = as.integer(revenue_account))

  # Optional GL filters. For example, set min = 3001 and max = 3271 to keep only
  # revenue accounts in that range.
  if (!is.null(revenue_account_min)) {
    level_2_ids <- level_2_ids |>
      filter(revenue_account_number >= revenue_account_min)
  }

  if (!is.null(revenue_account_max)) {
    level_2_ids <- level_2_ids |>
      filter(revenue_account_number <= revenue_account_max)
  }

  level_2_ids <- level_2_ids |>
    select(-revenue_account_number)

  message("Found ", nrow(level_2_ids), " revenue account/year records to download.")

  # Download each revenue account detail table and combine them into one data set.
  result_df <- level_2_ids |>
    pmap_dfr(~ download_transaction_data(..1, ..2, ..3, ..4, ..5, ..6, to_date))

  # Remove DAWN total rows, parse dates and amounts, join Type from the Excel
  # lookup by GL Number, rename user-facing columns, and enforce a stable column
  # order for every output CSV.
  revenue_data <- result_df |>
    filter(
      !is.na(`Doc Number`),
      str_trim(`Doc Number`) != "",
      !str_detect(Date, "^Total")
    ) |>
    mutate(
      Date = lubridate::mdy(Date),
      Amount = readr::parse_number(Amount),
      revenue_account = as.integer(revenue_account),
      level_1_id = as.integer(level_1_id),
      level_2_id = as.integer(level_2_id)
    ) |>
    left_join(type_lookup, by = "revenue_account") |>
    select(
      `Doc Number`,
      Date,
      Amount,
      level_1_id,
      level_2_id,
      `GL Number` = revenue_account,
      description,
      Type,
      `BA Number` = ba,
      fy
    ) |>
    arrange(`BA Number`, fy, `GL Number`, Date, `Doc Number`)

  dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
  fwrite(revenue_data, file = output_file)
  message("Saved ", nrow(revenue_data), " rows to ", output_file)

  invisible(revenue_data)
}
