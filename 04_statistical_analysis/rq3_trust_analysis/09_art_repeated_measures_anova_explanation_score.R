# =======================================================================================
# Script: 09_art_repeated_measures_anova_explanation_score.R
# Purpose: Main Analysis of Subjective Explanation Scores (Research_Question_3)
#
# This script addresses the RQ3 Sub-Question: How does the intended explainability 
# condition (High vs. Low) align with the explainability effectiveness perceived by users?
#
# It utilizes a 2x2 Mixed-Effects Aligned Rank Transform (ART) ANOVA on the Subjective 
# Explanation Score (averaged from Survey Items 5-7). The model includes a random 
# intercept for `participant_id` to strictly control for the repeated-measures design.
# =======================================================================================

# Step 1: Install required packages (uncomment if needed)
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
library(ARTool)      # For Non-parametric mixed factorial ANOVA
library(dplyr)       # For data manipulation
library(emmeans)     # For post-hoc interaction contrasts
library(ggplot2)     # For publication-ready visualization
library(lme4)        # Mixed-effects modeling support
library(car)         # For assumption checks
library(effectsize)  # For effect sizes (Eta-squared, Cohen's d)
library(ggdist)      # Required for Raincloud plots
library(tidyr)       # For data shaping

# ==============================================================================
# PHASE 0: DATA PREPARATION & FACTORIZATION
# ==============================================================================
# Step 3: Load the data
cat("\nSelect your cleaned Trust Data file (Trust_Data_Cleaned_No_Outliers.csv)...\n")
trust_data <- read.csv(choose.files(), header=T)

# Convert all independent variables and the random effect identifier to categorical factors 
trust_data$Architecture <- as.factor(trust_data$Architecture)
trust_data$Explainablity_Condition <- as.factor(trust_data$Explainablity_Condition)
trust_data$participant_id <- as.factor(trust_data$participant_id)
str(trust_data)

# ==============================================================================
# PHASE 1: EXPLORATORY DATA ANALYSIS (EXPLANATION SCORE)
# ==============================================================================

# Calculate descriptive statistics using dplyr
explanation_summary <- trust_data %>%
  group_by(Architecture, Explainablity_Condition) %>%
  dplyr::summarise(
    N = dplyr::n(),
    Mean = mean(explanation_score, na.rm = TRUE),
    Median = median(explanation_score, na.rm = TRUE),
    SD = sd(explanation_score, na.rm = TRUE),
    Min = min(explanation_score, na.rm = TRUE),
    Max = max(explanation_score, na.rm = TRUE),
    .groups = 'drop'
  )
print(explanation_summary)

# Visualize the Explanation Score Data
# A. Histogram with Density Curves to visually check skewness
plot_exp_hist_curve <- ggplot(trust_data, aes(x = explanation_score, fill = Explainablity_Condition, color = Explainablity_Condition)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 0.25, position = "identity", alpha = 0.4) +
  geom_density(alpha = 0.2, linewidth = 1) +
  facet_wrap(~ Architecture) +
  labs(title = "Distribution of Subjective Explanation Scores with Density Curves",
       x = "Explanation Score (1-5)",
       y = "Density") +
  theme_minimal()
print(plot_exp_hist_curve)

# B. Boxplot to visually check for equal variance and outliers
plot_exp_box <- ggplot(trust_data, aes(x = Architecture, y = explanation_score, fill = Explainablity_Condition)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Boxplot of Subjective Explanation Scores", x = "Architecture", y = "Explanation Score (1-5)") +
  theme_minimal()
print(plot_exp_box)

# ==============================================================================
# PHASE 1.1: QUANTIFYING RAW MEAN DIFFERENCES
# ==============================================================================

# 1. Raw differences between Explainability Conditions within each Architecture
raw_diffs_explainability_exp <- explanation_summary %>%
  select(Architecture, Explainablity_Condition, Mean) %>%
  tidyr::pivot_wider(names_from = Explainablity_Condition, values_from = Mean) %>%
  mutate(Raw_Difference_High_minus_Low = High - Low)
print(raw_diffs_explainability_exp)

# 2. Raw differences between Architectures within each Explainability Condition
raw_diffs_architecture_exp <- explanation_summary %>%
  select(Architecture, Explainablity_Condition, Mean) %>%
  tidyr::pivot_wider(names_from = Architecture, values_from = Mean) %>%
  mutate(Raw_Difference_Graph_minus_Semantic = Graph - Semantic)
print(raw_diffs_architecture_exp)

# ==============================================================================
# PHASE 2: ASSUMPTION CHECKS FOR ANOVA
# ==============================================================================
# Applying Shapiro-Wilk and Levene's to the new dependent variable.

# 1. Normality Check (Shapiro-Wilk Test on model residuals)
basic_lm_exp <- lm(explanation_score ~ Architecture * Explainablity_Condition, data = trust_data)

# Q-Q Plot for Visual Normality Check 
plot_qq_exp <- ggplot(data.frame(residuals = residuals(basic_lm_exp)), aes(sample = residuals)) +
  stat_qq() +
  stat_qq_line(color = "red", linewidth = 1) +
  labs(title = "Q-Q Plot of Model Residuals (Explanation Scores)",
       x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_minimal()
print(plot_qq_exp)

shapiro_result_exp <- shapiro.test(residuals(basic_lm_exp))
print("1. Shapiro-Wilk Test for Normality:")
print(shapiro_result_exp)
if(shapiro_result_exp$p.value < 0.05) {
  print("-> CONCLUSION: Data violates normality assumption (p < 0.05). Non-parametric ART ANOVA recommended.")
} else {
  print("-> CONCLUSION: Data meets normality assumption (p > 0.05).")
}

# 2. Homogeneity of Variance Check (Levene's Test)
levene_result_exp <- leveneTest(explanation_score ~ Architecture * Explainablity_Condition, data = trust_data)
print("2. Levene's Test for Homogeneity of Variance:")
print(levene_result_exp)
if(levene_result_exp$`Pr(>F)`[1] < 0.05) {
  print("-> CONCLUSION: Data violates equal variance assumption (heteroscedasticity detected).")
} else {
  print("-> CONCLUSION: Data meets equal variance assumption.")
}

# ==============================================================================
# PHASE 3: NON-PARAMETRIC REPEATED MEASURES ANALYSIS (ART ANOVA)
# ==============================================================================

# Fit the Aligned Rank Transform ANOVA model.
# (1|participant_id) handles the repeated measures design.
art_model_exp <- art(explanation_score ~ Architecture * Explainablity_Condition + (1 | participant_id), data = trust_data)
art_anova_results_exp <- anova(art_model_exp)
print(art_anova_results_exp)

# Post-hoc Analysis using Tukey's correction for multiple comparisons
interaction_contrasts_exp <- art.con(art_model_exp, "Architecture:Explainablity_Condition", adjust = "tukey")
contrast_summary_df_exp <- as.data.frame(summary(interaction_contrasts_exp))

print("--- Tukey Post-Hoc Contrasts ---")
summary(interaction_contrasts_exp)

plot_score_bar_exp <- ggplot(trust_data, aes(x = Architecture, y = explanation_score, fill = Explainablity_Condition)) +
  stat_summary(fun = mean, geom = "bar", position = "dodge") +
  labs(title = "Mean Explanation Score by Architecture and Explainability Condition",
       x = "RAG Architecture", y = "Mean Explanation Score") +
  theme_minimal()
print(plot_score_bar_exp)

# ==============================================================================
# PHASE 3.1: EFFECT SIZES (CORRECTED FOR S4 OBJECTS)
# ==============================================================================

# Calculate Partial Eta-Squared using the F-value and degrees of freedom
eta_squared_results_exp <- F_to_eta2(f = art_anova_results_exp$F, 
                                     df = art_anova_results_exp$Df, 
                                     df_error = art_anova_results_exp$Df.res)

eta_table_exp <- data.frame(Effect = rownames(art_anova_results_exp), eta_squared_results_exp)

# Convert Cohen's d to Hedges' g (applying the correction factor for small sample bias)
hedges_g_results_exp <- t_to_d(t = contrast_summary_df_exp$t.ratio, 
                               df = contrast_summary_df_exp$df)

hedges_g_results_exp$Hedges_g <- hedges_g_results_exp$d * (1 - (3 / (4 * contrast_summary_df_exp$df - 1)))

g_table_exp <- data.frame(Contrast = contrast_summary_df_exp$contrast, 
                          Hedges_g = hedges_g_results_exp$Hedges_g)
print(g_table_exp)

# ==============================================================================
# PHASE 3.2: QUANTIFYING PERFORMANCE DIFFERENCES (PAIRWISE CONTRASTS)
# ==============================================================================

# 1. Main Effect of Explainablity_Condition (Averaged across architectures)
explainability_diffs_exp <- emmeans(art_model_exp, ~ Explainablity_Condition)
explainability_contrasts_exp <- contrast(explainability_diffs_exp, method = "pairwise", infer = TRUE)
explainability_diff_table_exp <- as.data.frame(explainability_contrasts_exp)

# 2. Main Effect of Architecture (Averaged across explainability conditions)
arch_diffs_exp <- emmeans(art_model_exp, ~ Architecture)
arch_contrasts_exp <- contrast(arch_diffs_exp, method = "pairwise", infer = TRUE)
arch_diff_table_exp <- as.data.frame(arch_contrasts_exp)

# 3. Simple Effects / Interaction Contrasts
interaction_diff_table_exp <- as.data.frame(summary(interaction_contrasts_exp))

# ==============================================================================
# PHASE 4: VISUALS (RAINCLOUD & INTERACTION PLOTS)
# ==============================================================================

plot_raincloud_exp <- ggplot(trust_data, aes(x = Architecture, y = explanation_score, fill = Explainablity_Condition)) +
  stat_halfeye(adjust = 0.5, width = 0.5, .width = 0, justification = -0.2, alpha = 0.6, position = position_dodge(width = 0.8)) +
  geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.8, position = position_dodge(width = 0.8)) +
  scale_fill_viridis_d(option = "mako", end = 0.8) +
  labs(title = "Subjective Explanation Scores: Raincloud Distribution",
       x = "RAG Architecture", y = "Explanation Score (1-5)") +
  theme_classic(base_size = 14) +
  theme(legend.position = "top")
print(plot_raincloud_exp)

plot_violin_box_exp <- ggplot(trust_data, aes(x = Architecture, y = explanation_score, fill = Explainablity_Condition)) +
  geom_violin(alpha = 0.5, trim = FALSE, position = position_dodge(width = 0.8)) +
  geom_boxplot(width = 0.2, color = "black", alpha = 0.9, position = position_dodge(width = 0.8)) +
  scale_fill_viridis_d(option = "mako", end = 0.8) +
  labs(title = "Subjective Explanation Scores: Violin & Boxplot",
       x = "RAG Architecture", y = "Explanation Score (1-5)") +
  theme_classic(base_size = 14)
print(plot_violin_box_exp)

# Summary for 95% confidence intervals
exp_error_summary <- trust_data %>%
  group_by(Architecture, Explainablity_Condition) %>%
  dplyr::summarise(
    N = dplyr::n(),
    Mean_Exp = mean(explanation_score, na.rm = TRUE),
    SD_Exp = sd(explanation_score, na.rm = TRUE),
    SE_Exp = SD_Exp / sqrt(N),
    CI_Exp = qt(0.975, df = N - 1) * SE_Exp, 
    .groups = 'drop'
  )

pd <- position_dodge(0.1)

# Interaction Line Plot with 95% CIs: Highlights how Architecture and Explainability interact
plot_interaction_ci_exp <- ggplot(exp_error_summary, aes(x = Explainablity_Condition, y = Mean_Exp, group = Architecture, color = Architecture)) +
  geom_line(linewidth = 1.2, position = pd) +
  geom_errorbar(aes(ymin = Mean_Exp - CI_Exp, ymax = Mean_Exp + CI_Exp), 
                width = 0.15, linewidth = 1, position = pd) +
  geom_point(size = 4, position = pd) +
  scale_color_viridis_d(option = "cividis", end = 0.8) +
  labs(title = "Interaction Effect: Explanation Scores across Transparency Levels",
       subtitle = "Error bars represent 95% Confidence Intervals",
       x = "Explainability Condition", y = "Mean Explanation Score") +
  theme_classic(base_size = 14)
print(plot_interaction_ci_exp)

# ==============================================================================
# PHASE 5: SILENT STORAGE TO TARGET DIRECTORY
# ==============================================================================

output_dir <- "E:/THSI/Code/statistical_analyses _R/outputs"

# 1. Save core summaries and effect sizes as CSV files
write.csv(explanation_summary, file.path(output_dir, "Explanation_Score_Summary.csv"), row.names = FALSE)
write.csv(eta_table_exp, file.path(output_dir, "Explanation_ANOVA_EtaSquared.csv"), row.names = FALSE)
write.csv(g_table_exp, file.path(output_dir, "Explanation_Hedges_G.csv"), row.names = FALSE)
write.csv(exp_error_summary, file.path(output_dir, "Explanation_Error_Summary_CI.csv"), row.names = FALSE)

# 2. Save quantified model-based differences as CSV files
write.csv(explainability_diff_table_exp, file.path(output_dir, "Explanation_Explainability_Differences.csv"), row.names = FALSE)
write.csv(arch_diff_table_exp, file.path(output_dir, "Explanation_Architecture_Differences.csv"), row.names = FALSE)
write.csv(interaction_diff_table_exp, file.path(output_dir, "Explanation_Interaction_Contrasts_Differences.csv"), row.names = FALSE)

# 3. Save raw mean differences as CSV files
write.csv(raw_diffs_explainability_exp, file.path(output_dir, "Explanation_Raw_Diffs_Explainability.csv"), row.names = FALSE)
write.csv(raw_diffs_architecture_exp, file.path(output_dir, "Explanation_Raw_Diffs_Architecture.csv"), row.names = FALSE)

# 4. Sink raw statistical structures directly to the text file report
sink(file.path(output_dir, "Explanation_Full_Statistical_Report.txt"))
shapiro_result_exp
levene_result_exp
art_anova_results_exp
summary(interaction_contrasts_exp)
explainability_contrasts_exp
arch_contrasts_exp
sink()