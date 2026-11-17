# =====================================================================================================
# Script: 03_covariate_analysis_academic_status.R
# Purpose: Analysis of User Trust and Explainability Controlling for Demographics (Research_Question_3)
#
# This script evaluates whether the intended explainability condition (High vs. Low) 
# and the underlying RAG Architecture influence the user's aggregated evaluation score 
# (a proxy for user trust and perceived effectiveness). 
# 
# Crucially, it introduces 'Academic Status' extracted from the demographics JSON as a 
# third fixed effect. This 3-way Mixed-Effects Aligned Rank Transform (ART) ANOVA 
# determines if user background modulates their trust in AI explanations, while 
# accounting for repeated measures across individual participants.
# ====================================================================================================

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
cat("\nSelect your raw experimental data file...\n")
raw_data <- read.csv(file.choose(), stringsAsFactors = FALSE)

# Data Wrangling Pipeline: Unpack demographics and parse evaluation scores
trust_data <- raw_data %>%
  # 1. Parse demographics JSON to extract Academic Status (highest_education)
  mutate(
    demographics_list = map(demographics, ~ tryCatch(fromJSON(.x), error = function(e) list())),
    Academic_Status = map_chr(demographics_list, ~ as.character(.x$highest_education %||% NA))
  ) %>%
  select(participant_id, Academic_Status, starts_with("single_question_rating_")) %>%
  
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
  select(participant_id, Architecture, Explainablity_Condition, Academic_Status, human_evaluation_score) %>%
  drop_na(human_evaluation_score)

# Convert all independent variables and the random effect identifier to categorical factors 
trust_data$Architecture <- as.factor(trust_data$Architecture)
trust_data$Explainablity_Condition <- as.factor(trust_data$Explainablity_Condition)
trust_data$Academic_Status <- as.factor(trust_data$Academic_Status)
trust_data$participant_id <- as.factor(trust_data$participant_id)

print("Distribution of Academic Status:")
print(table(trust_data$Academic_Status))

# ==============================================================================
# PHASE 1: EXPLORATORY DATA ANALYSIS (GROUPED BY ACADEMIC STATUS)
# ==============================================================================
acad_demo_summary <- trust_data %>%
  group_by(Architecture, Explainablity_Condition, Academic_Status) %>%
  dplyr::summarise(
    N = dplyr::n(),
    Mean = mean(human_evaluation_score, na.rm = TRUE),
    Median = median(human_evaluation_score, na.rm = TRUE),
    SD = sd(human_evaluation_score, na.rm = TRUE),
    .groups = 'drop'
  )
print("--- DESCRIPTIVE SUMMARY ---")
print(acad_demo_summary)

# ==============================================================================
# PHASE 2: 3-WAY NON-PARAMETRIC MIXED MODEL (ART ANOVA)
# ==============================================================================
# Fit the 3-Way ART model testing Architecture, Explainability, and Demographics.
# (1|participant_id) handles the repeated measures design (each user saw all 4 combinations).
art_model_acad <- art(human_evaluation_score ~ Architecture * Explainablity_Condition * Academic_Status + (1 | participant_id), 
                      data = trust_data)

# Run the ANOVA on the aligned ranks to determine significant main effects and interactions
art_anova_acad_results <- anova(art_model_acad)
print("--- 3-Way ART ANOVA Results (Academic Status) ---")
print(art_anova_acad_results)

# ==============================================================================
# PHASE 3: POST-HOC CONTRASTS (IF SIGNIFICANT)
# ==============================================================================
# 1. Main Effect of Academic Status
# Assesses if baseline trust fundamentally differs across educational backgrounds
acad_main_effect <- art.con(art_model_acad, "Academic_Status", adjust = "tukey")
print("--- Main Effect: Academic Status ---")
summary(acad_main_effect)

# 2. 2-Way Interaction: Explainability Condition x Academic Status
# Assesses if the *impact* of XAI (High vs Low) depends on the user's educational background
exp_acad_interaction <- art.con(art_model_acad, "Explainablity_Condition:Academic_Status", adjust = "tukey")
print("--- Interaction: Explainability x Academic Status ---")
summary(exp_acad_interaction)

# ==============================================================================
# PHASE 4: VISUALIZATIONS
# ==============================================================================
# Generate a Faceted Boxplot breaking down aggregated scores by Architecture and Explainability, 
# explicitly split across the different Academic Status tiers.

plot_acad_boxplot <- ggplot(trust_data, aes(x = Architecture, y = human_evaluation_score, fill = Explainablity_Condition)) +
  geom_boxplot(alpha = 0.8, position = position_dodge(width = 0.8), outlier.shape = 21) +
  # Using label_wrap_gen so long demographic labels (e.g., "Master's Degree") fit nicely in the facet panels
  facet_wrap(~ Academic_Status, labeller = label_wrap_gen(width = 25)) +
  # Academic color scaling
  scale_fill_viridis_d(option = "mako", end = 0.8) +
  labs(
    x = "RAG Architecture", 
    y = "Aggregated Human Evaluation Score (0-100)",
    fill = "Explainability Condition") +
  theme_classic(base_size = 12) +
  theme(legend.position = "top", strip.background = element_rect(fill="lightgray"))

print(plot_acad_boxplot)

# ==============================================================================
# PHASE 5: SILENT STORAGE TO TARGET DIRECTORY
# ==============================================================================
output_dir <- "E:/THSI/Code/statistical_analyses _R/outputs"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Save descriptive summary tables
write.csv(acad_demo_summary, file.path(output_dir, "Explanation_Academic_Status_Summary.csv"), row.names = FALSE)

# Sink raw statistical structures directly to the text file report 
sink(file.path(output_dir, "Explanation_Academic_Status_Statistical_Report.txt"))
print("--- DESCRIPTIVE SUMMARY ---")
print(acad_demo_summary)
print("\n--- 3-WAY ART ANOVA RESULTS ---")
print(art_anova_acad_results)
print("\n--- EXPLAINABILITY x ACADEMIC STATUS INTERACTION ---")
print(summary(exp_acad_interaction))
sink()

print(paste("Academic Status analysis complete! Results saved to:", output_dir))