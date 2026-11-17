# =============================================================================================
# Script: 06_cronbachs_alpha_measure.R
# Purpose: Internal Consistency & Reliability Check for Survey Constructs (Research_Question_3)
#
# This script uses the `psych` package to calculate Cronbach's Alpha for the two 
# primary psychological constructs evaluated in the user study:
#   1. Composite User Trust (Survey Items 1-4)
#   2. Perceived Explainability (Survey Items 5-7)
#
# It computes the reliability coefficients both globally (across the entire dataset) 
# and locally (isolated within each of the 4 experimental RAG conditions) to ensure 
# the survey instrument reliably measures the user's perception before conducting 
# any inferential statistics (ANOVAs).
# ============================================================================================

# Step 1: Load required packages (Uncomment installation lines if running on a fresh environment)
if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")
if (!requireNamespace("tidyverse", quietly = TRUE)) install.packages("tidyverse")
if (!requireNamespace("psych", quietly = TRUE)) install.packages("psych")

library(jsonlite)  # For parsing nested survey data
library(dplyr)     # For data manipulation
library(tidyr)     # For data shaping (pivot_longer, unnest_wider)
library(purrr)     # For mapping functions over lists
library(psych)     # For psychometric analysis (Cronbach's Alpha)

# ==============================================================================
# PHASE 1: DATA INGESTION & JSON PARSING
# ==============================================================================
# 1. Load the cleaned dataset interactively 
# (Load 'Trust_Data_Cleaned_No_Outliers.csv' from the data cleaning step)
df <- read.csv(file.choose(), stringsAsFactors = FALSE)

# 2. Define the exact column headers containing the raw survey JSON strings
survey_cols <- c("block_survey_Semantic_Low_Explainable",
                 "block_survey_Semantic_High_Explainable",
                 "block_survey_Graph_Low_Explainable",
                 "block_survey_Graph_High_Explainable")

# 3. Pivot to long format, unnest the JSON, and extract specific survey items
items_df <- df %>%
  select(participant_id, all_of(survey_cols)) %>%
  pivot_longer(cols = all_of(survey_cols),
               names_to = "condition",
               values_to = "survey_json") %>%
  filter(!is.na(survey_json) & survey_json != "") %>%
  # Parse the nested JSON structure into readable lists
  mutate(parsed_items = map(survey_json, ~fromJSON(.x)$survey_responses)) %>%
  # Widen the list so item1 through item7 become distinct columns for the psych package
  unnest_wider(parsed_items)

# ==============================================================================
# PHASE 2: OUTPUT ROUTING
# ==============================================================================
# 4. Set up output directory and file routing
save_dir <- "E:/THSI/Code/statistical_analyses _R/outputs"
if(!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
output_file <- paste0(save_dir, "/Cronbach_Alpha_Reliability_Results.txt")

# Start redirecting console output to the text file for the thesis appendix
sink(output_file)

# =========================================================================
# PART A: OVERALL RELIABILITY (Pooled across all conditions)
# =========================================================================

cat("====================================================\n")
cat("                OVERALL RELIABILITY                 \n")
cat("           (Pooled across all conditions)           \n")
cat("====================================================\n\n")

# A1. Isolate and test the four survey questions measuring Composite User Trust
trust_items_overall <- items_df %>% select(item1, item2, item3, item4)
trust_alpha_overall <- psych::alpha(trust_items_overall, check.keys = TRUE)

cat("--- COMPOSITE USER TRUST (Items 1-4) ---\n")
print(trust_alpha_overall)
cat("\n\n")

# A2. Isolate and test the three survey questions measuring Perceived Explainability
explainability_items_overall <- items_df %>% select(item5, item6, item7)
explainability_alpha_overall <- psych::alpha(explainability_items_overall, check.keys = TRUE)

cat("--- PERCEIVED EXPLAINABILITY (Items 5-7) ---\n")
print(explainability_alpha_overall)
cat("\n\n")

# =========================================================================
# PART B: PER-BLOCK RELIABILITY (Split by experimental condition)
# =========================================================================

cat("====================================================\n")
cat("               PER-BLOCK RELIABILITY                \n")
cat("              (Split by RAG condition)              \n")
cat("====================================================\n\n")

# Split the master dataframe into a list of 4 dataframes based on the experimental condition
condition_list <- split(items_df, items_df$condition)

# Loop through each condition, extract the items, and calculate localized Alphas
for (cond_name in names(condition_list)) {
  
  cat("----------------------------------------------------\n")
  cat("CONDITION: ", cond_name, "\n")
  cat("----------------------------------------------------\n\n")
  
  cond_data <- condition_list[[cond_name]]
  
  # B1. Trust localized alpha
  trust_items_block <- cond_data %>% select(item1, item2, item3, item4)
  trust_alpha_block <- psych::alpha(trust_items_block, check.keys = TRUE)
  
  cat("--- COMPOSITE USER TRUST (Items 1-4) ---\n")
  print(trust_alpha_block)
  cat("\n")
  
  # B2. Explainability localized alpha
  explainability_items_block <- cond_data %>% select(item5, item6, item7)
  explainability_alpha_block <- psych::alpha(explainability_items_block, check.keys = TRUE)
  
  cat("--- PERCEIVED EXPLAINABILITY (Items 5-7) ---\n")
  print(explainability_alpha_block)
  cat("\n\n")
}

# Close the file connection and return output to the RStudio console
sink()

cat("Overall and per-block reliability results successfully saved to:\n", output_file, "\n")