# =======================================================================================
# Script: 04_art_mixed_anova_evaluator_alignment.R
# Purpose: Comprehensive Analysis Pipeline for Evaluator Comparison (Research_Question_2)
#
# This script evaluates whether the automated LLM-as-a-Judge framework systematically 
# overrates or underrates RAG pipelines compared to Human Annotators. 
#
# It utilizes a Mixed-Effects Aligned Rank Transform (ART) ANOVA on the metric score. 
# The model includes a random intercept for `Question_ID` to strictly control for the 
# repeated-measures evaluation of the same prompts by both humans and the LLM.
#
# Note: This is a generalized script. Change the `TARGET_METRIC` variable below 
# to run the analysis for Faithfulness, Answer Relevancy, or Contextual metrics.
# =======================================================================================

# Step 1: Install required packages (Uncomment if running on a fresh environment)
install.packages("ARTool")
install.packages("dplyr")
install.packages("emmeans")
install.packages("ggplot2")
install.packages("lme4") 
install.packages("car") 
install.packages("effectsize") 
install.packages("ggdist")     
install.packages("tidyr")      

# Step 2: Load the primary packages
library(ARTool)      # Non-parametric mixed factorial ANOVA
library(dplyr)       # Data manipulation
library(emmeans)     # Post-hoc contrasts
library(ggplot2)     # Visualization
library(lme4)        # Mixed-effects modeling
library(car)         # Assumption checks
library(effectsize)  # Effect sizes
library(ggdist)      # Raincloud plots
library(tidyr)       # Data shaping

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Change this variable to match the metric you are currently analyzing.
# This will automatically update all plot titles and output file names.
TARGET_METRIC <- "Faithfulness" 
# Example alternatives: "Answer_Relevancy", "Contextual_Precision", "Contextual_Recall"

# ==============================================================================
# PHASE 0: DATA PREPARATION & FACTORIZATION
# ==============================================================================
# Step 3: Load the data
cat(sprintf("\nSelect your cleaned Evaluator Data file for %s...\n", TARGET_METRIC))
data_metric <- read.csv(choose.files(), header=T)

# Convert categorical variables to factors
data_metric$Architecture <- as.factor(data_metric$Architecture)
data_metric$EvaluatorType <- as.factor(data_metric$EvaluatorType)
data_metric$Question_ID <- as.factor(data_metric$Question_ID)
str(data_metric)

# ==============================================================================
# PHASE 1: EXPLORATORY DATA ANALYSIS (SCORE)
# ==============================================================================

# Calculate descriptive statistics using dplyr
score_summary <- data_metric %>%
  group_by(Architecture, EvaluatorType) %>%
  dplyr::summarise(
    N = dplyr::n(),
    Mean = mean(Score, na.rm = TRUE),
    Median = median(Score, na.rm = TRUE),
    SD = sd(Score, na.rm = TRUE),
    Min = min(Score, na.rm = TRUE),
    Max = max(Score, na.rm = TRUE),
    .groups = 'drop'
  )
print(score_summary)

# Visualize the Score Data
# A. Histogram with Density Curves to visually check skewness
plot_score_hist_curve <- ggplot(data_metric, aes(x = Score, fill = EvaluatorType, color = EvaluatorType)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 0.05, position = "identity", alpha = 0.4) +
  geom_density(alpha = 0.2, linewidth = 1) +
  facet_wrap(~ Architecture) +
  labs(title = sprintf("Distribution of %s Scores with Density Curves", TARGET_METRIC),
       x = "Score",
       y = "Density") +
  theme_minimal()
print(plot_score_hist_curve)

# B. Boxplot to visually check for equal variance and outliers
plot_score_box <- ggplot(data_metric, aes(x = Architecture, y = Score, fill = EvaluatorType)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = sprintf("Boxplot of %s Scores", TARGET_METRIC), x = "Architecture", y = "Score") +
  theme_minimal()
print(plot_score_box)

# ==============================================================================
# PHASE 1.1: QUANTIFYING RAW MEAN DIFFERENCES
# ==============================================================================

# 1. Raw differences between Evaluator Types within each Architecture
raw_diffs_evaluatortype <- score_summary %>%
  select(Architecture, EvaluatorType, Mean) %>%
  tidyr::pivot_wider(names_from = EvaluatorType, values_from = Mean) %>%
  mutate(Raw_Difference = Human_Annotator - LLM_Judge) 
print(raw_diffs_evaluatortype)

# 2. Raw differences between Architectures within each Evaluator Type
raw_diffs_architecture <- score_summary %>%
  select(Architecture, EvaluatorType, Mean) %>%
  tidyr::pivot_wider(names_from = Architecture, values_from = Mean) %>%
  mutate(Raw_Difference_Graph_minus_Semantic = Graph - Semantic)
print(raw_diffs_architecture)

# ==============================================================================
# PHASE 2: ASSUMPTION CHECKS FOR ANOVA
# ==============================================================================

# 1. Normality Check (Shapiro-Wilk Test on model residuals)
basic_lm <- lm(Score ~ Architecture * EvaluatorType, data = data_metric)

# Q-Q Plot for Visual Normality Check 
plot_qq <- ggplot(data.frame(residuals = residuals(basic_lm)), aes(sample = residuals)) +
  stat_qq() +
  stat_qq_line(color = "red", linewidth = 1) +
  labs(title = sprintf("Q-Q Plot of Model Residuals (%s)", TARGET_METRIC),
       x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_minimal()
print(plot_qq)

shapiro_result <- shapiro.test(residuals(basic_lm))
print("1. Shapiro-Wilk Test for Normality:")
print(shapiro_result)
if(shapiro_result$p.value < 0.05) {
  print("-> CONCLUSION: Data violates normality assumption (p < 0.05). Non-parametric test recommended.")
} else {
  print("-> CONCLUSION: Data meets normality assumption (p > 0.05).")
}

# 2. Homogeneity of Variance Check (Levene's Test)
levene_result <- leveneTest(Score ~ Architecture * EvaluatorType, data = data_metric)
print("2. Levene's Test for Homogeneity of Variance:")
print(levene_result)
if(levene_result$`Pr(>F)`[1] < 0.05) {
  print("-> CONCLUSION: Data violates equal variance assumption (heteroscedasticity detected).")
} else {
  print("-> CONCLUSION: Data meets equal variance assumption.")
}

print("FINAL DECISION: If either assumption is severely violated, proceed to Phase 3 (ART ANOVA).")
cat("\nPress [Enter] to continue to ART ANOVA and Error Rate Analysis...\n")
readline()

# ==============================================================================
# PHASE 3: NON-PARAMETRIC ANALYSIS (ART ANOVA)
# ==============================================================================

# Fit the Aligned Rank Transform ANOVA model.
art_model <- art(Score ~ Architecture * EvaluatorType + (1|Question_ID), data = data_metric)
art_anova_results <- anova(art_model)
print(art_anova_results)

interaction_contrasts <- art.con(art_model, "Architecture:EvaluatorType", adjust = "tukey")
contrast_summary_df <- as.data.frame(summary(interaction_contrasts))

print("--- Tukey Post-Hoc Contrasts ---")
summary(interaction_contrasts)

plot_score_bar <- ggplot(data_metric, aes(x = Architecture, y = Score, fill = EvaluatorType)) +
  stat_summary(fun = mean, geom = "bar", position = "dodge") +
  labs(title = sprintf("Mean %s by Architecture and Evaluator Type", TARGET_METRIC),
       x = "RAG Architecture", y = sprintf("Mean %s Score", TARGET_METRIC)) +
  theme_minimal()
print(plot_score_bar)

# ==============================================================================
# PHASE 3.1: EFFECT SIZES (CORRECTED FOR S4 OBJECTS)
# ==============================================================================

eta_squared_results <- F_to_eta2(f = art_anova_results$F, 
                                 df = art_anova_results$Df, 
                                 df_error = art_anova_results$Df.res)

eta_table <- data.frame(Effect = rownames(art_anova_results), eta_squared_results)

hedges_g_results <- t_to_d(t = contrast_summary_df$t.ratio, 
                           df = contrast_summary_df$df)

hedges_g_results$Hedges_g <- hedges_g_results$d * (1 - (3 / (4 * contrast_summary_df$df - 1)))

g_table <- data.frame(Contrast = contrast_summary_df$contrast, 
                      Hedges_g = hedges_g_results$Hedges_g)
print(g_table)

# ==============================================================================
# PHASE 3.2: QUANTIFYING PERFORMANCE DIFFERENCES (PAIRWISE CONTRASTS)
# ==============================================================================

# 1. Main Effect of EvaluatorType (Averaged across architectures)
evaluatortype_diffs <- emmeans(art_model, ~ EvaluatorType)
evaluatortype_contrasts <- contrast(evaluatortype_diffs, method = "pairwise", infer = TRUE)
evaluatortype_diff_table <- as.data.frame(evaluatortype_contrasts)

# 2. Main Effect of Architecture (Averaged across evaluator types)
arch_diffs <- emmeans(art_model, ~ Architecture)
arch_contrasts <- contrast(arch_diffs, method = "pairwise", infer = TRUE)
arch_diff_table <- as.data.frame(arch_contrasts)

# 3. Simple Effects / Interaction Contrasts
interaction_diff_table <- as.data.frame(summary(interaction_contrasts))

# ==============================================================================
# PHASE 4: ERROR RATE EXPLORATION & VISUALIZATIONS
# ==============================================================================

# Calculate Error Rate as the inverse of the Score.
data_metric$Error_Rate <- 1 - data_metric$Score

# Calculate summary statistics for the Error Rate across architectures and evaluator types
error_summary <- data_metric %>%
  group_by(Architecture, EvaluatorType) %>%
  dplyr::summarise(
    N = dplyr::n(),
    Mean_Error = mean(Error_Rate, na.rm = TRUE),
    SD_Error = sd(Error_Rate, na.rm = TRUE),
    SE_Error = SD_Error / sqrt(N),
    CI_Error = qt(0.975, df = N - 1) * SE_Error, 
    Max_Error = max(Error_Rate, na.rm = TRUE),
    .groups = 'drop'
  )

# 1. Histogram & Density Curve: Shows the shape and spread of the error distribution
plot_error_hist_curve <- ggplot(data_metric, aes(x = Error_Rate, fill = EvaluatorType, color = EvaluatorType)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 0.05, position = "identity", alpha = 0.4) +
  geom_density(alpha = 0.2, linewidth = 1) +
  facet_wrap(~ Architecture) +
  labs(title = sprintf("Distribution of %s Error Rates with Density Curves", TARGET_METRIC), 
       x = "Error Rate", 
       y = "Density") +
  theme_minimal()
print(plot_error_hist_curve)

# 2. Basic Boxplot: Useful for quickly spotting medians, quartiles, and outliers
plot_error_box <- ggplot(data_metric, aes(x = Architecture, y = Error_Rate, fill = EvaluatorType)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = sprintf("Box Plot of %s Error Rates", TARGET_METRIC), x = "RAG Architecture", y = "Error Rate") +
  theme_minimal()
print(plot_error_box)

# 3. Raincloud Plot: Advanced visualization showing data distribution (half-violin) + summary (boxplot)
plot_raincloud <- ggplot(data_metric, aes(x = Architecture, y = Error_Rate, fill = EvaluatorType)) +
  stat_halfeye(adjust = 0.5, width = 0.5, .width = 0, justification = -0.2, alpha = 0.6, position = position_dodge(width = 0.8)) +
  geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.8, position = position_dodge(width = 0.8)) +
  scale_fill_viridis_d(option = "mako", end = 0.8) +
  labs(title = sprintf("%s Error Rates: Raincloud Distribution", TARGET_METRIC),
       x = "RAG Architecture", y = "Error Rate") +
  theme_classic(base_size = 14) +
  theme(legend.position = "top")
print(plot_raincloud)

# 4. Violin & Boxplot Combo: Displays the full, mirrored density shape alongside standard quartiles
plot_violin_box <- ggplot(data_metric, aes(x = Architecture, y = Error_Rate, fill = EvaluatorType)) +
  geom_violin(alpha = 0.5, trim = FALSE, position = position_dodge(width = 0.8)) +
  geom_boxplot(width = 0.2, color = "black", alpha = 0.9, position = position_dodge(width = 0.8)) +
  scale_fill_viridis_d(option = "mako", end = 0.8) +
  labs(title = sprintf("%s Error Rates: Violin & Boxplot", TARGET_METRIC),
       x = "RAG Architecture", y = "Error Rate") +
  theme_classic(base_size = 14)
print(plot_violin_box)

# 5. Interaction Line Plot with 95% CIs
pd <- position_dodge(0.1) 

plot_interaction_ci <- ggplot(error_summary, aes(x = EvaluatorType, y = Mean_Error, group = Architecture, color = Architecture)) +
  geom_line(linewidth = 1.2, position = pd) +
  geom_errorbar(aes(ymin = Mean_Error - CI_Error, ymax = Mean_Error + CI_Error), 
                width = 0.15, linewidth = 1, position = pd) +
  geom_point(size = 4, position = pd) +
  scale_color_viridis_d(option = "cividis", end = 0.8) +
  labs(title = sprintf("Interaction Effect: %s Evaluation Shift", TARGET_METRIC),
       subtitle = "Error bars represent 95% Confidence Intervals",
       x = "Evaluator Type", y = "Mean Error Rate") +
  theme_classic(base_size = 14)
print(plot_interaction_ci)

# ==============================================================================
# PHASE 5: SILENT STORAGE TO TARGET DIRECTORY
# ==============================================================================

output_dir <- "E:/THSI/Code/statistical_analyses _R/outputs"

# 1. Save core summaries and effect sizes as CSV files (Dynamic naming based on TARGET_METRIC)
write.csv(score_summary, file.path(output_dir, sprintf("%s_Score_Summary.csv", TARGET_METRIC)), row.names = FALSE)
write.csv(eta_table, file.path(output_dir, sprintf("%s_ANOVA_EtaSquared.csv", TARGET_METRIC)), row.names = FALSE)
write.csv(g_table, file.path(output_dir, sprintf("%s_Hedges_G.csv", TARGET_METRIC)), row.names = FALSE)
write.csv(error_summary, file.path(output_dir, sprintf("%s_Error_Summary_CI.csv", TARGET_METRIC)), row.names = FALSE)

# 2. Save quantified model-based differences as CSV files
write.csv(evaluatortype_diff_table, file.path(output_dir, sprintf("%s_EvaluatorType_Differences.csv", TARGET_METRIC)), row.names = FALSE)
write.csv(arch_diff_table, file.path(output_dir, sprintf("%s_Architecture_Differences.csv", TARGET_METRIC)), row.names = FALSE)
write.csv(interaction_diff_table, file.path(output_dir, sprintf("%s_Interaction_Contrasts_Differences.csv", TARGET_METRIC)), row.names = FALSE)

# 3. Save raw mean differences as CSV files
write.csv(raw_diffs_evaluatortype, file.path(output_dir, sprintf("%s_Raw_Diffs_EvaluatorType.csv", TARGET_METRIC)), row.names = FALSE)
write.csv(raw_diffs_architecture, file.path(output_dir, sprintf("%s_Raw_Diffs_Architecture.csv", TARGET_METRIC)), row.names = FALSE)

# 4. Sink raw statistical structures directly to the text file report
sink(file.path(output_dir, sprintf("%s_Full_Statistical_Report.txt", TARGET_METRIC)))
print("--- SHAPIRO-WILK TEST ---")
print(shapiro_result)
print("--- LEVENE'S TEST ---")
print(levene_result)
print("--- ART ANOVA RESULTS ---")
print(art_anova_results)
print("--- INTERACTION CONTRASTS ---")
print(summary(interaction_contrasts))
print("--- EVALUATOR TYPE CONTRASTS ---")
print(evaluatortype_contrasts)
print("--- ARCHITECTURE CONTRASTS ---")
print(arch_contrasts)
sink()