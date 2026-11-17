# ================================================================================================
# Script: 10_spearman_correlation_trust_explainability.R
# Purpose: Correlation Analysis between Subjective Explainability and Trust (Research_Question_3)
#
# This script evaluates whether a user's perceived explainability score correlates 
# with their composite trust score within each experimental condition. 
#
# Because previous assumption checks (Shapiro-Wilk) revealed non-normal distributions, 
# this script correctly utilizes the non-parametric Spearman's rank correlation (Rho) 
# instead of Pearson's r to measure the monotonic relationship between the two constructs.
# ===============================================================================================

# Step 1: Load required packages (Uncomment installation lines if running on a fresh environment)
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("broom", quietly = TRUE)) install.packages("broom")
if (!requireNamespace("readr", quietly = TRUE)) install.packages("readr")

library(dplyr) # For data manipulation and grouping
library(broom) # For tidying the output of statistical tests into clean dataframes
library(readr) # For robust CSV reading and writing

# ==============================================================================
# PHASE 1: DATA INGESTION
# ==============================================================================
# 1. INTERACTIVELY SELECT YOUR INPUT CSV FILE
cat("\nSelect your cleaned Trust Data file...\n")
file_path <- file.choose()
my_data <- read_csv(file_path, show_col_types = FALSE)

# ==============================================================================
# PHASE 2: OUTPUT ROUTING
# ==============================================================================
# 2. SET OUTPUT DIRECTORY
output_dir <- "E:/THSI/Code/statistical_analyses _R/outputs"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ==============================================================================
# PHASE 3: NON-PARAMETRIC CORRELATION ANALYSIS (SPEARMAN)
# ==============================================================================
# 3. RUN CONDITION-SPECIFIC SPEARMAN CORRELATIONS
# Group the dataset by the 4 experimental blocks to see if the correlation 
# strength changes depending on the RAG architecture or transparency level.
correlation_results <- my_data %>%
  group_by(Architecture, Explainablity_Condition) %>%
  # cor.test calculates the correlation; tidy() extracts the Rho and p-value cleanly
  # exact = FALSE prevents warnings caused by tied ranks (common in Likert-scale composites)
  do(tidy(cor.test(.$explanation_score, .$trust_score, method = "spearman", exact = FALSE))) %>%
  ungroup()

# ==============================================================================
# PHASE 4: FORMATTING & EXPORT
# ==============================================================================
# 4. FORMAT THE RESULTS TABLE
# Select only the relevant statistical outputs and rename them for reporting
formatted_results <- correlation_results %>%
  select(
    Architecture = Architecture,
    Condition = Explainablity_Condition,
    Rho = estimate,
    P_Value = p.value
  )

# Print the table to the RStudio console for immediate review
cat("\n--- SPEARMAN CORRELATION RESULTS (RHO) ---\n")
print(formatted_results)

# 5. SAVE THE RESULTS AS A CSV FILE
output_file <- file.path(output_dir, "spearman_correlation_results.csv")
write_csv(formatted_results, output_file)

cat("\nCorrelation results successfully saved to:", output_file, "\n")