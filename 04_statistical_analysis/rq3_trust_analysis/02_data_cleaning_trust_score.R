# ==============================================================================================
# Script: 02_data_cleaning_trust_score.R
# Purpose: Outlier Detection and Data Cleaning for User Trust Evaluation (Research_Question_3)
#
# This script screens the raw composite user trust scores for univariate outliers 
# using the standard Interquartile Range (IQR) method (1.5 * IQR). 
#
# Because the study utilizes a repeated-measures design, any participant flagged as an 
# outlier in ONE condition is entirely removed across ALL conditions. This strict 
# removal strategy is mathematically required to maintain a balanced dataset for the 
# subsequent Aligned Rank Transform (ART) ANOVAs. It also generates visualisatoins.
# ============================================================================================

library(dplyr)
library(ggplot2)

# ==============================================================================
# PHASE 1: DATA INGESTION & OUTLIER DETECTION
# ==============================================================================
# 1. Upload the raw dataset
cat("\nSelect your raw Trust Data file...\n")
df <- read.csv(file.choose(), stringsAsFactors = FALSE)

# 2. Flag Outliers Row-by-Row
# Create an intermediate dataframe that holds the outlier flags for every observation
df_with_outlier_flags <- df %>%
  group_by(Architecture, Explainablity_Condition) %>%
  mutate(
    Q1 = quantile(trust_score, 0.25, na.rm = TRUE),
    Q3 = quantile(trust_score, 0.75, na.rm = TRUE),
    IQR = Q3 - Q1,
    Lower_Bound = Q1 - 1.5 * IQR,
    Upper_Bound = Q3 + 1.5 * IQR,
    Is_Outlier = trust_score < Lower_Bound | trust_score > Upper_Bound
  ) %>%
  ungroup()

# Summarize the counts per condition for reporting
outlier_summary <- df_with_outlier_flags %>%
  group_by(Architecture, Explainablity_Condition) %>%
  summarise(Outliers_Detected = sum(Is_Outlier, na.rm = TRUE), .groups = 'drop')

cat("\n--- OUTLIER SUMMARY ---\n")
print(outlier_summary)

# ==============================================================================
# PHASE 2: BALANCED REMOVAL & CLEAN DATASET GENERATION
# ==============================================================================
# 3. Extract IDs and Create Cleaned Dataset
# Pull the exact participant IDs that triggered the outlier flag anywhere in their trials
outlier_ids <- df_with_outlier_flags %>%
  filter(Is_Outlier == TRUE) %>%
  pull(participant_id) %>%
  unique()

cat("\n--- OUTLIER PARTICIPANT IDs (Slated for full removal) ---\n")
print(outlier_ids)

# Filter the original dataset to exclude these participants entirely
df_clean <- df %>%
  filter(!participant_id %in% outlier_ids)

# Convert factor back just in case to drop unused levels 
df_clean$participant_id <- as.factor(as.character(df_clean$participant_id))

total_outliers_removed <- length(outlier_ids)
cat(sprintf("\nOriginal dataset rows: %d\n", nrow(df)))
cat(sprintf("Cleaned dataset rows (after removing %d participant(s) across all 4 blocks): %d\n", 
            total_outliers_removed, nrow(df_clean)))

cat("\nSUGGESTED APPENDIX TEXT:\n")
cat(sprintf("'Prior to the main analysis, the data was screened for univariate outliers using the IQR method (1.5 * IQR). This procedure identified %d participant(s) with anomalous scores, who were subsequently removed from all experimental conditions to maintain a balanced repeated-measures design. The pre-removal distributions are highlighted in the appendix visualization.'\n\n", total_outliers_removed))

# ==============================================================================
# PHASE 3: VISUALIZATIONS
# ==============================================================================
# 4. Generate the raw data plot 
# (Using the ORIGINAL 'df' to physically show the red dot outliers before removal)
raw_boxplot <- ggplot(df, aes(x = Explainablity_Condition, y = trust_score, fill = Architecture)) +
  geom_boxplot(width = 0.6, alpha = 0.8, outlier.colour = "red", outlier.size = 3, outlier.shape = 16) +
  scale_fill_manual(values = c("Graph" = "#4C72B0", "Semantic" = "#C44E52")) +
  labs(
    x = "Explainability Condition",
    y = "Composite User Trust Score"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom" 
  )

print(raw_boxplot)

# ==============================================================================
# PHASE 4: SILENT STORAGE TO TARGET DIRECTORY
# ==============================================================================
# 5. Save the plot and the cleaned dataset for downstream analysis scripts
output_dir <- "E:/THSI/Code/statistical_analyses _R/outputs/"

# Save the plot
ggsave(paste0(output_dir, "appendix_raw_outlier_plot.png"), plot = raw_boxplot, width = 7, height = 5, dpi = 300)

# Save the new cleaned dataset (This is the file the subsequent covariate scripts should load)
write.csv(df_clean, paste0(output_dir, "Trust_Data_Cleaned_No_Outliers.csv"), row.names = FALSE)

cat("\nSuccess: Appendix plot and cleaned dataset saved to", output_dir, "\n")