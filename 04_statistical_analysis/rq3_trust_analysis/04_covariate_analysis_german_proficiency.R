# =================================================================================================
# Script: 04_covariate_analysis_german_proficiency.R
# Purpose: Analysis of User Trust and Explainability Controlling for Language (Research_Question_3)
#
# This script evaluates whether the intended explainability condition (High vs. Low) 
# and the underlying RAG Architecture influence the user's aggregated evaluation score 
# (a proxy for user trust and perceived effectiveness). 
# 
# Crucially, it introduces 'German Proficiency' extracted from the demographics JSON as a 
# third fixed effect. This 3-way Mixed-Effects Aligned Rank Transform (ART) ANOVA 
# determines if language barriers modulate user trust in AI explanations, while 
# accounting for repeated measures across individual participants.
# ================================================================================================

# Step 1: Load required packages (Uncomment installation lines if running on a fresh environment)
if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")
if (!requireNamespace("tidyverse", quietly = TRUE)) install.packages("tidyverse")
if (!requireNamespace("ARTool", quietly = TRUE)) install.packages("ARTool")
if (!requireNamespace("emmeans", quietly = TRUE)) install.packages("emmeans")
if (!requireNamespace("lme4", quietly = TRUE)) install.packages("lme4")

library(jsonlite)  # For parsing nested JSON strings in the raw Excel data
library(tidyverse) # For data manipulation and string extraction
library(ARTool)    # For Non-parametric mixed factorial ANOVA
library(emmeans)   # For post-hoc interaction contrasts
library(ggplot2)   # For publication-ready visualization
library(lme4)      # Mixed-effects modeling support

# ==============================================================================
# PHASE 0: DATA EXTRACTION & AGGREGATION
# ==============================================================================
cat("\nSelect your raw experimental data file (ExperimentData.xlsx converted to CSV)...\n")
raw_data <- read.csv(file.choose(), stringsAsFactors = FALSE)

# Data Wrangling Pipeline: Unpack demographics and parse evaluation scores
trust_data <- raw_data %>%
  # 1. Parse demographics JSON to extract German Proficiency
  mutate(
    demographics_list = map(demographics, ~ tryCatch(fromJSON(.x), error = function(e) list())),
    German_Proficiency = map_chr(demographics_list, ~ as.character(.x$german_proficiency %||% NA))
  ) %>%
  select(participant_id, German_Proficiency, starts_with("single_question_rating_")) %>%
  
  # 2. Pivot the 4 experimental rating columns from Wide format into Long format 
  # (Required for ART ANOVA statistical testing)
  pivot_longer(
    cols = starts_with("single_question_rating_"),
    names_to = "Condition_String",
    values_to = "json_responses"
  ) %>%
  filter(!is.na(json_responses) & json_responses != "") %>%
  
  # 3. Use RegEx to cleanly extract the Architecture and Explainability Condition from the column headers
  mutate(
    Architecture = ifelse(grepl("Graph", Condition_String), "Graph", "Semantic"),
    Explainablity_Condition = ifelse(grepl("High", Condition_String), "High", "Low")
  ) %>%
  
  # 4. Parse the 12 evaluation metrics and calculate the aggregate trust/effectiveness score per condition
  mutate(
    human_evaluation_score = map_dbl(json_responses, function(j_str) {
      # Parse the JSON string for this specific participant's condition block
      parsed <- tryCatch(fromJSON(j_str), error = function(e) NULL)
      if(is.null(parsed) || !"scores" %in% names(parsed)) return(NA)
      
      # Extract the nested scores (3 questions * 4 metrics = 12 total evaluation items)
      s_df <- parsed$scores
      
      # Unlist into a single vector of strings (e.g., "100%", "75%")
      vals_str <- unlist(s_df)
      
      # Clean the strings by removing '%' and converting directly to numeric floats
      vals_num <- as.numeric(gsub("%", "", vals_str))
      
      # Average the 12 sub-metrics into a single overarching evaluation/trust score (0-100 scale)
      mean(vals_num, na.rm = TRUE)
    })
  ) %>%
  select(participant_id, Architecture, Explainablity_Condition, German_Proficiency, human_evaluation_score) %>%
  drop_na(human_evaluation_score)

# Convert all independent variables and the random effect identifier to categorical factors 
trust_data$Architecture <- as.factor(trust_data$Architecture)
trust_data$Explainablity_Condition <- as.factor(trust_data$Explainablity_Condition)
trust_data$German_Proficiency <- as.factor(trust_data$German_Proficiency)
trust_data$participant_id <- as.factor(trust_data$participant_id)

print("Distribution of German Proficiency:")
print(table(trust_data$German_Proficiency))

# ==============================================================================
# PHASE 1: EXPLORATORY DATA ANALYSIS (GROUPED BY GERMAN LEVEL)
# ==============================================================================
ger_demo_summary <- trust_data %>%
  group_by(Architecture, Explainablity_Condition, German_Proficiency) %>%
  dplyr::summarise(
    N = dplyr::n(),
    Mean = mean(human_evaluation_score, na.rm = TRUE),
    Median = median(human_evaluation_score, na.rm = TRUE),
    SD = sd(human_evaluation_score, na.rm = TRUE),
    .groups = 'drop'
  )
print("--- DESCRIPTIVE SUMMARY ---")
print(ger_demo_summary)

# ==============================================================================
# PHASE 2: 3-WAY NON-PARAMETRIC MIXED MODEL (ART ANOVA)
# ==============================================================================
# Fit the 3-Way ART model testing Architecture, Explainability, and Demographics.
# (1|participant_id) handles the repeated measures design (each user saw all 4 combinations).
art_model_ger <- art(human_evaluation_score ~ Architecture * Explainablity_Condition * German_Proficiency + (1 | participant_id), 
                     data = trust_data)

# Run the ANOVA on the aligned ranks to determine significant main effects and interactions
art_anova_ger_results <- anova(art_model_ger)
print("--- 3-Way ART ANOVA Results (German Proficiency) ---")
print(art_anova_ger_results)

# ==============================================================================
# PHASE 3: POST-HOC CONTRASTS (IF SIGNIFICANT)
# ==============================================================================
# 1. Main Effect of German Proficiency
# Assesses if baseline trust fundamentally differs depending on language skills
ger_main_effect <- art.con(art_model_ger, "German_Proficiency", adjust = "tukey")
print("--- Main Effect: German Proficiency ---")
summary(ger_main_effect)

# 2. 2-Way Interaction: Explainability Condition x German Proficiency
# Assesses if the *impact* of XAI (High vs Low) depends on the user's language barrier
exp_ger_interaction <- art.con(art_model_ger, "Explainablity_Condition:German_Proficiency", adjust = "tukey")
print("--- Interaction: Explainability x German Proficiency ---")
summary(exp_ger_interaction)

# ==============================================================================
# PHASE 4: VISUALIZATIONS
# ==============================================================================
# Generate a Faceted Boxplot breaking down aggregated scores by Architecture and Explainability, 
# explicitly split across the different German Proficiency tiers.

plot_ger_boxplot <- ggplot(trust_data, aes(x = Architecture, y = human_evaluation_score, fill = Explainablity_Condition)) +
  geom_boxplot(alpha = 0.8, position = position_dodge(width = 0.8), outlier.shape = 21) +
  facet_wrap(~ German_Proficiency) +
  scale_fill_viridis_d(option = "mako", end = 0.8) +
  labs(
       x = "RAG Architecture", 
       y = "Aggregated Human Evaluation Score (0-100)",
       fill = "Explainability Condition") +
  theme_classic(base_size = 12) +
  theme(legend.position = "top", strip.background = element_rect(fill="lightgray"))

print(plot_ger_boxplot)

# ==============================================================================
# PHASE 5: SILENT STORAGE TO TARGET DIRECTORY
# ==============================================================================
output_dir <- "E:/THSI/Code/statistical_analyses _R/outputs"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Save descriptive summary tables
write.csv(ger_demo_summary, file.path(output_dir, "Explanation_German_Prof_Summary.csv"), row.names = FALSE)

# Sink raw statistical structures directly to the text file report 
sink(file.path(output_dir, "Explanation_German_Prof_Statistical_Report.txt"))
print("--- DESCRIPTIVE SUMMARY ---")
print(ger_demo_summary)
print("\n--- 3-WAY ART ANOVA RESULTS ---")
print(art_anova_ger_results)
print("\n--- EXPLAINABILITY x GERMAN PROFICIENCY INTERACTION ---")
print(summary(exp_ger_interaction))
sink()

print(paste("German Proficiency analysis complete! Results saved to:", output_dir))