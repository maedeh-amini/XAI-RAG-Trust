# =======================================================================================
# Script: 01_evaluator_bias_summary_table.R
# Purpose: Generate Table Comparing LLM vs. Human Judges (Research_Question_2)
#
# This script uses the `gt` package to construct a detailed summary table that 
# quantifies the "Measurement Bias" (Mean Delta) between the automated DeepEval 
# pipeline (LLM Judge) and the participants from the Streamlit study (Human Judge).
# 
# It visually stacks the metrics within the Semantic and Graph architectures, 
# highlighting statistically significant discrepancies in red to demonstrate where 
# LLM-as-a-judge overestimates or underestimates RAG performance.
# =======================================================================================

# Step 1: Install required packages (Uncomment if running on a fresh environment)
install.packages(c("gt", "dplyr", "webshot2"))
library(gt)       # For building data tables
library(dplyr)    # For data manipulation
library(webshot2) # Required by 'gt' to export tables as PNG images

# ==============================================================================
# PHASE 1: DATA PREPARATION & STRUCTURE
# ==============================================================================
# Prepare data frame with alternating blank strings for the 'Metric' and 
# 'Mean_Delta' columns. This creates a clean "stacked" layout when rendered, 
# preventing repetitive text from cluttering the table.
df_table <- data.frame(
  Architecture = rep(c("Semantic", "Graph"), each = 8),
  Metric = c(
    "Contextual Recall", "", 
    "Contextual Precision", "", 
    "Faithfulness", "", 
    "Answer Relevancy", "",
    "Contextual Recall", "", 
    "Contextual Precision", "", 
    "Faithfulness", "", 
    "Answer Relevancy", ""
  ),
  Evaluator = rep(c("LLM Judge", "Human Judge"), 8),
  Mean_SD = c(
    "62.00 (49.00)", "66.70 (36.10)",
    "58.90 (34.60)", "80.51 (28.53)",
    "56.60 (17.80)", "78.85 (29.47)",
    "77.80 (29.60)", "84.39 (25.40)",
    "78.00 (41.80)", "77.55 (28.86)",
    "71.60 (20.90)", "83.21 (26.20)",
    "84.50 (19.30)", "82.73 (27.10)",
    "75.40 (31.50)", "85.75 (25.46)"
  ),
  # Mean Delta (Δ) represents the measurement bias. Positive values indicate 
  # the Human Judge scored the system higher than the LLM Judge.
  Mean_Delta = c(
    "+4.70", "", "+21.61", "", "+22.25", "", "+6.59", "",
    "-0.45", "", "+11.61", "", "-1.77", "", "+10.35", ""
  ),
  Median_Range = c(
    "100.00 (0.00 - 100.00)", "75.00 (0.00 - 100.00)",
    "54.20 (0.00 - 100.00)", "100.00 (0.00 - 100.00)",
    "64.50 (0.00 - 64.50)", "100.00 (0.00 - 100.00)",
    "95.00 (0.00 - 100.00)", "100.00 (0.00 - 100.00)",
    "100.00 (0.00 - 100.00)", "100.00 (0.00 - 100.00)",
    "70.00 (33.30 - 100.00)", "100.00 (0.00 - 100.00)",
    "91.50 (0.00 - 91.50)", "100.00 (0.00 - 100.00)",
    "84.40 (0.00 - 100.00)", "100.00 (0.00 - 100.00)"
  )
)

# ==============================================================================
# PHASE 2: GT TABLE CONSTRUCTION & STYLING
# ==============================================================================
# 2. Build the gt table
formatted_table <- df_table %>%
  select(Architecture, Metric, Evaluator, Mean_SD, Mean_Delta, Median_Range) %>%
  # Groups the rows by Architecture (Semantic vs. Graph) for clear sectioning
  gt(groupname_col = "Architecture") %>%
  
  # Standardize academic column headers
  cols_label(
    Metric = "Evaluation Metric",
    Evaluator = "Evaluator",
    Mean_SD = "Mean (SD)",
    Mean_Delta = "Mean Bias (Δ)",
    Median_Range = "Median (Min - Max)"
  ) %>%
  
  # Bold the 4 metrics in the Metric column for readability
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_body(columns = Metric)
  ) %>%
  
  # Bold the Architecture group headers (Semantic, Graph)
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_row_groups()
  ) %>%
  
  # Remove the inner border between the LLM and Human rows for each metric group
  # to visually link the two evaluator scores together.
  tab_style(
    style = cell_borders(sides = "bottom", color = "transparent"),
    locations = cells_body(
      rows = c(1, 3, 5, 7, 9, 11, 13, 15)
    )
  ) %>%
  
  # Highlight significant bias metrics in bold red.
  # This visually emphasizes where the LLM Judge significantly underestimated performance.
  tab_style(
    style = cell_text(color = "#D90429", weight = "bold"),
    locations = cells_body(
      columns = Mean_Delta,
      rows = Mean_Delta %in% c("+21.61", "+22.25", "+11.61", "+10.35")
    )
  ) %>%
  
  # Center align all numerical columns
  cols_align(
    align = "center",
    columns = c(Evaluator, Mean_SD, Mean_Delta, Median_Range)
  ) %>%
  
  # Apply strict academic formatting (Times New Roman, black-and-white print style)
  opt_table_font(font = "Times New Roman") %>%
  tab_options(
    heading.title.font.size = 16,
    heading.subtitle.font.size = 12,
    table.border.top.style = "solid",
    table.border.bottom.style = "solid",
    table.background.color = "white",         # Sets overall table background to white
    table.font.color = "black",               # Sets overall text color to black
    column_labels.background.color = "white", # Sets header background to white
    column_labels.font.weight = "bold"
  ) %>%
  tab_style(
    style = cell_text(color = "black", weight = "bold"), # Sets header text to black
    locations = cells_column_labels()
  )
  
# Display the table in RStudio viewer
print(formatted_table)

# ==============================================================================
# PHASE 3: EXPORT TO PNG
# ==============================================================================
# Ensures target directory exist and saves the high-resolution image there
output_dir <- "E:/THSI/Code/statistical_analyses _R/outputs"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

output_path <- file.path(output_dir, "evaluation_summary.png")
# Zoom = 2 increases the DPI/resolution of the output PNG 
gtsave(formatted_table, filename = output_path, zoom = 2, expand = 10)