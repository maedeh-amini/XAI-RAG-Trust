# =======================================================================================================
# Script: 05_covariate_analysis_ai_experience.R
# Purpose: Analysis of User Trust and Explainability Controlling for AI Familiarity (Research_Question_3)
#
# This script evaluates whether the intended explainability condition (High vs. Low) 
# and the underlying RAG Architecture influence the subjective User Trust score. 
# 
# Crucially, it introduces 'AI Experience' extracted from the demographics JSON as a 
# third fixed effect. This 3-way Mixed-Effects Aligned Rank Transform (ART) ANOVA 
# determines if prior familiarity with AI tools moderates user trust in explanations, 
# while accounting for repeated measures across individual participants.
# ======================================================================================================

# Step 1: Load required packages
if (!requireNamespace("ARTool", quietly = TRUE)) install.packages("ARTool")
if (!requireNamespace("tidyverse", quietly = TRUE)) install.packages("tidyverse")
if (!requireNamespace("emmeans", quietly = TRUE)) install.packages("emmeans")
if (!requireNamespace("ggdist", quietly = TRUE)) install.packages("ggdist")

library(ARTool)      # For Non-parametric mixed factorial ANOVA
library(dplyr)       # For data manipulation
library(emmeans)     # For post-hoc interaction contrasts
library(ggplot2)     # For publication-ready visualization
library(lme4)        # Mixed-effects modeling support
library(car)         # For assumption checks
library(effectsize)  # For effect sizes
library(ggdist)      # For distribution plots
library(tidyr)       # For data shaping

# ==============================================================================
# PHASE 0: DATA PREPARATION & FACTORIZATION
# ==============================================================================
cat("\nSelect your pre-processed demographic data file (Trust_Explanation_Scores_With_Demographics.csv)...\n")
trust_data <- read.csv(choose.files(), header=T)

# Convert all independent variables and the random effect identifier to categorical factors 
trust_data$Architecture <- as.factor(trust_data$Architecture)
trust_data$Explainablity_Condition <- as.factor(trust_data$Explainablity_Condition)
trust_data$AI_Experience <- as.factor(trust_data$AI_Experience)
trust_data$participant_id <- as.factor(trust_data$participant_id)

# Verify the structural mapping
str(trust_data)
print("Distribution of AI Experience:")
print(table(trust_data$AI_Experience))

# ==============================================================================
# PHASE 1: EXPLORATORY DATA ANALYSIS (GROUPED BY AI EXPERIENCE)
# ==============================================================================
trust_demo_summary <- trust_data %>%
  group_by(Architecture, Explainablity_Condition, AI_Experience) %>%
  dplyr::summarise(
    N = dplyr::n(),
    Mean = mean(trust_score, na.rm = TRUE),
    Median = median(trust_score, na.rm = TRUE),
    SD = sd(trust_score, na.rm = TRUE),
    Min = min(trust_score, na.rm = TRUE),
    Max = max(trust_score, na.rm = TRUE),
    .groups = 'drop'
  )
print("--- DESCRIPTIVE SUMMARY ---")
print(trust_demo_summary)


# ==============================================================================
# PHASE 2: 3-WAY NON-PARAMETRIC MIXED MODEL (ART ANOVA)
# ==============================================================================
# Fit the 3-Way ART model testing Architecture, Explainability, and AI Experience.
# (1|participant_id) handles the repeated measures design (each user saw all 4 combinations).
art_model_3way <- art(trust_score ~ Architecture * Explainablity_Condition * AI_Experience + (1 | participant_id), 
                      data = trust_data)

# Run the ANOVA on the aligned ranks to determine significant main effects and interactions
art_anova_3way_results <- anova(art_model_3way)
print("--- 3-Way ART ANOVA Results (AI Experience) ---")
print(art_anova_3way_results)


# ==============================================================================
# PHASE 3: POST-HOC CONTRASTS (IF SIGNIFICANT)
# ==============================================================================
# 1. Main Effect of AI Experience 
# Assesses if frequent users are generally more trusting of AI systems overall
ai_main_effect <- art.con(art_model_3way, "AI_Experience", adjust = "tukey")
print("--- Main Effect: AI Experience ---")
summary(ai_main_effect)

# 2. 2-Way Interaction: Architecture * AI_Experience
# Assesses if power users prefer the Graph structure over Semantic, or vice versa
arch_ai_interaction <- art.con(art_model_3way, "Architecture:AI_Experience", adjust = "tukey")
print("--- Interaction: Architecture x AI Experience ---")
summary(arch_ai_interaction)

# 3. 3-Way Interaction Contrasts (Full model breakdown)
way3_contrasts <- art.con(art_model_3way, "Architecture:Explainablity_Condition:AI_Experience", adjust = "tukey")


# ==============================================================================
# PHASE 4: VISUALIZATIONS
# ==============================================================================
# Generate a Faceted Boxplot breaking down Trust Scores by Architecture and Explainability, 
# explicitly split across the different AI Experience tiers.
plot_3way_boxplot <- ggplot(trust_data, aes(x = Architecture, y = trust_score, fill = Explainablity_Condition)) +
  geom_boxplot(alpha = 0.8, position = position_dodge(width = 0.8), outlier.shape = 21) +
  facet_wrap(~ AI_Experience, labeller = label_wrap_gen(width = 20)) +
  scale_fill_viridis_d(option = "mako", end = 0.8) +
  labs(
       x = "RAG Architecture", 
       y = "Subjective Trust Score",
       fill = "Explainability Condition") +
  theme_classic(base_size = 14) +
  theme(legend.position = "top", strip.background = element_rect(fill="lightgray"))

print(plot_3way_boxplot)


# ==============================================================================
# PHASE 5: SILENT STORAGE TO TARGET DIRECTORY
# ==============================================================================
output_dir <- "E:/THSI/Code/statistical_analyses _R/outputs"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Save descriptive summary and contrast tables
write.csv(trust_demo_summary, file.path(output_dir, "Trust_Score_AI_Experience_Summary.csv"), row.names = FALSE)
write.csv(as.data.frame(summary(way3_contrasts)), file.path(output_dir, "Trust_3Way_Contrasts_AI_Experience.csv"), row.names = FALSE)

# Sink raw statistical structures directly to the text file report 
sink(file.path(output_dir, "Trust_AI_Experience_Full_Statistical_Report.txt"))
print("--- DESCRIPTIVE SUMMARY ---")
print(trust_demo_summary)
print("\n--- 3-WAY ART ANOVA RESULTS ---")
print(art_anova_3way_results)
print("\n--- ARCHITECTURE x AI EXPERIENCE INTERACTION ---")
print(summary(arch_ai_interaction))
sink()

print(paste("3-Way Analysis complete! Results and visual saved to:", output_dir))