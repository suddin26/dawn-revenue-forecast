required_packages <- c("rvest", "data.table", "tidyverse", "purrr", "zoo")
missing_packages <- required_packages[!required_packages %in% rownames(installed.packages())]

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
library(zoo)

#setwd("S:/ra/RADUMP/Chief Economist/R Projects/DAWN Data Download and Manipulation")

# Example Bank excise URL: http://dawn12.state.nv.us:7778/pls/prodsw/bsr_date_range?pm_detail_option=2.2.9&pm_level1_id=31706&pm_level2_id=96253
# Budget Account 9130 // GL 3068 // FY 2026
# BSR Summary: http://dawn12.state.nv.us:7778/pls/prodsw/bsr_gen_bbls_report?input_budget_account=9130&input_fund=&input_fiscal_year=2026

budget_account_list <- c(9010)
fy_list <- c(2002:2026)

# Step 1: Get Level 1 IDs

# Loop on year and budget account to pull up this table:
# http://dawn12.state.nv.us:7778/pls/prodsw/bsr_gen_bbls_report?input_budget_account=3265&input_fund=101&input_fiscal_year=2000

level_1_ids <- tibble(
  level_1 = character(0),
  ba = integer(0),
  fy = integer(0)
)
z <- 1

for(i in 1:length(budget_account_list)) {

  for(j in 1:length(fy_list)){

    # Replace 'your_url_here' with the URL of the webpage you want to scrape
    url <- paste0("http://dawn12.state.nv.us:7778/pls/prodsw/bsr_gen_bbls_report?input_budget_account=",budget_account_list[i],"&input_fund=101&input_fiscal_year=",fy_list[j])

    # Read the HTML content of the webpage
    webpage <- read_html(url)

    # Select all anchor tags and extract the href attribute
    links <- webpage %>%
      html_nodes("a") %>%
      html_attr("href")

    # Filter the links to find the one containing 'bsr_gen_bcls_report' - Expenditures, or 'bsr_rec_fund_sum' - Revenues
    # target_link <- links[grep("bsr_gen_bcls_report", links)]
    target_link <- links[grep("bsr_rec_fund_sum", links)]

    # Extract the numeric characters following 'level1_id='
    level_1_ids[z,] <- list(
      sub(".+level1_id=(\\d+).*", "\\1", target_link),
      budget_account_list[i],
      fy_list[j]
    )

    z <- z+1

  }
}

# # Step 2: Given a level 1 ID, get a list of the corresponding level 2 IDs.
# # URL looks like this: http://dawn12.state.nv.us:7778/pls/prodsw/bsr_gen_bcls_report?pm_level1_id=1659 - Expenditures
# # URL looks like this: http://dawn12.state.nv.us:7778/pls/prodsw/bsr_rec_fund_sum?pm_level1_id=31706 - Revenues
 
 
 
 level_2_ids <- list()

 for(i in 1:nrow(level_1_ids)){

   url <- paste0("http://dawn12.state.nv.us:7778/pls/prodsw/bsr_rec_fund_sum?pm_level1_id=",level_1_ids[i,1])

   # Read HTML content from the URL
   parsed_html <- read_html(url)

   # Extract level2_id and text values
   table_rows <- parsed_html %>%
     html_nodes("td a") %>%
     lapply(function(node) {
       level2_id <- sub(".+pm_level2_id=(\\d+).*", "\\1", html_attr(node, "href"))
       revenue_account <- html_text(node)
       if (grepl("^[0-9]+$", level2_id)) {  # Check if level2_id consists of only numeric characters
         data.frame(level2_id = level2_id, revenue_account = revenue_account)
       } else {
         NULL  # Skip rows with non-numeric level2_id
       }
     })

   # Combine the extracted data frames into one
   level_2_ids[[i]] <- result_df <- do.call(rbind, table_rows) %>%
     mutate(level_1_id = as.character(level_1_ids[i,1]),
            ba = as.integer(level_1_ids[i,2]),
            fy = as.integer(level_1_ids[i,3]))

 }

 level_2_ids <- bind_rows(level_2_ids) |> 
   filter(suppressWarnings(!is.na(as.numeric(revenue_account))))



 # Step 3: For all Level 1 and Level 2 records, get transaction detail.
 # Set to-date to Sys.Date.
 # URL example: http://dawn12.state.nv.us:7778/pls/prodsw/bsr_rev_det?pm_level1_id=31706&pm_level2_id=96253&pm_level3_id=&pm_from_date=01/01/2000&pm_to_date=12/02/2025

 to_date <- format(Sys.Date(), "%m/%d/%Y")

 download_transaction_data <- function(level1_id, level2_id, category, ba, fy) {
   url <- paste0(
     "http://dawn12.state.nv.us:7778/pls/prodsw/bsr_rev_det?",
     "pm_level1_id=", level1_id,
     "&pm_level2_id=", level2_id,
     "&pm_level3_id=&pm_from_date=01/01/2000&pm_to_date=",
     to_date
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

       detail_tables[[1]]
     },
     error = function(e) {
       message(
         "Error downloading data for level1_id=", level1_id,
         " level2_id=", level2_id,
         ": ", conditionMessage(e)
       )
       return(NULL)
     }
   )
   if (!is.null(data)) {
     data$level1_id <- level1_id
     data$level2_id <- level2_id
     data$category <- category
     data$ba <- ba
     data$fy <- fy
   }
   return(data)
 }
 
 
 filtered_level_2 <- level_2_ids |> 
   filter(revenue_account == "3068")
 
 result_df <- filtered_level_2 |> 
   pmap_df(~ download_transaction_data(..3, ..1, ..2, ..4, ..5))
 
 revenue_data <- result_df |> 
   filter(
     !is.na(`Doc Number`),
     str_trim(`Doc Number`) != "",
     !str_detect(Date, "^Total")
   ) |> 
   mutate(
     Date = lubridate::mdy(Date),
     Amount = readr::parse_number(Amount)
   )


fwrite(revenue_data, file = "bank_exise_tax_history.csv")
