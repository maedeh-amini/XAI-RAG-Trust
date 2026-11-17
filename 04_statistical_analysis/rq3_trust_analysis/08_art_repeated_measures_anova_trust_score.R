# =======================================================================================
# Script: 08_art_repeated_measures_anova_trust_score.R
# Purpose: Main Analysis of Composite User Trust Scores (Research_Question_3)
#
# This script executes the primary inferential analysis to answer RQ3: How do the choice 
# of RAG architecture and output explainability condition influence user trust?
#
# It utilizes a 2x2 Mixed-Effects Aligned Rank Transform (ART) ANOVA on the composite 
# User Trust Score (averaged from Survey Items 1-4). The model includes a random intercept 
# for `participant_id` to strictly control for the repeated-measures experimental design.
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
library(ARTool)      # For Non-parametric mixed factorial ANOVA
library(dplyr)       # For data manipulation
library(emmeans)     # For post-hoc interaction contrasts
library(ggplot2)     # For publication-ready visualization
library(lme4)        # Mixed-effects modeling support
library(car)         # For assumption checks
library(effectsize)  # For effect sizes (Eta-squared, Cohen's d)
library(ggdist)      # Required for Raincloud plots
library(tidyr)       # For data shaping (pivoting raw differences)

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
# PHASE 1: EXPLORATORY DATA ANALYSIS (TRUST SCORE)
# ==============================================================================

# Calculate descriptive statistics using dplyr
trust_summary <- trust_data %>%
  group_by(Architecture, Explainablity_Condition) %>%
  dplyr::summarise(
    N = dplyr::n(),
    Mean = mean(trust_score, na.rm = TRUE),
    Median = median(trust_score, na.rm = TRUE),
    SD = sd(trust_score, na.rm = TRUE),
    Min = min(trust_score, na.rm = TRUE),
    Max = max(trust_score, na.rm = TRUE),
    .groups = 'drop'
  )
print(trust_summary)

# Visualize the Trust Score Data
# A. Histogram with Density Curves to visually check skewness
plot_trust_hist_curve <- ggplot(trust_data, aes(x = trust_score, fill = Explainablity_Condition, color = Explainablity_Condition)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 0.25, position = "identity", alpha = 0.4) +
  geom_density(alpha = 0.2, linewidth = 1) +
  facet_wrap(~ Architecture) +
  labs(
       x = "Trust Score (1-5)",
       y = "Density") +
  theme_minimal()
print(plot_trust_hist_curve)


# B. Boxplot to visually check for equal variance and outliers
plot_trust_box <- ggplot(trust_data, aes(x = Architecture, y = trust_score, fill = Explainablity_Condition)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Boxplot of User Trust Scores", x = "Architecture", y = "Trust Score (1-5)") +
  theme_minimal()
print(plot_trust_box)

# C. Raincloud Plot for raw trust_score (Density + Boxplot + Data Points)
plot_score_raincloud <- ggplot(trust_data, aes(x = Architecture, y = trust_score, fill = Explainablity_Condition)) +
  # 1. The "Cloud" (Half-Violin Density)
  stat_halfeye(adjust = 0.5, width = 0.5, .width = 0, justification = -0.25, 
               alpha = 0.6, position = position_dodge(width = 0.8)) +
  # 2. The Boxplot (Summary statistics)
  geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.8, 
               position = position_dodge(width = 0.8)) +
  # 3. The "Rain" (Individual jittered data points)
  geom_point(aes(color = Explainablity_Condition), size = 1.2, alpha = 0.4, 
             position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.8)) +
  # Color scaling 
  scale_fill_viridis_d(option = "mako", end = 0.8) +
  scale_color_viridis_d(option = "mako", end = 0.8) +
  # Clean axis labels
  labs(x = "RAG Architecture", 
       y = " User Trust Score (1-5)") +
  theme_classic(base_size = 12) +
  theme(legend.position = "top")

print(plot_score_raincloud) 


# ==============================================================================
# PHASE 1.1: QUANTIFYING RAW MEAN DIFFERENCES
# ==============================================================================

# 1. Raw differences between Explainability Conditions within each Architecture
raw_diffs_explainability <- trust_summary %>%
  select(Architecture, Explainablity_Condition, Mean) %>%
  tidyr::pivot_wider(names_from = Explainablity_Condition, values_from = Mean) %>%
  mutate(Raw_Difference_High_minus_Low = High - Low)
print(raw_diffs_explainability)

# 2. Raw differences between Architectures within each Explainability Condition
raw_diffs_architecture <- trust_summary %>%
  select(Architecture, Explainablity_Condition, Mean) %>%
  tidyr::pivot_wider(names_from = Architecture, values_from = Mean) %>%
  mutate(Raw_Difference_Graph_minus_Semantic = Graph - Semantic)
print(raw_diffs_architecture)


# ==============================================================================
# PHASE 2: ASSUMPTION CHECKS FOR ANOVA
# ==============================================================================

# 1. Normality Check (Shapiro-Wilk Test on model residuals)
basic_lm <- lm(trust_score ~ Architecture * Explainablity_Condition, data = trust_data)

# Q-Q Plot for Visual Normality Check 
plot_qq <- ggplot(data.frame(residuals = residuals(basic_lm)), aes(sample = residuals)) +
  stat_qq() +
  stat_qq_line(color = "red", linewidth = 1) +
  labs(
       x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_minimal()
print(plot_qq)

shapiro_result <- shapiro.test(residuals(basic_lm))
print("1. Shapiro-Wilk Test for Normality:")
print(shapiro_result)
if(shapiro_result$p.value < 0.05) {
  print("-> CONCLUSION: Data violates normality assumption (p < 0.05). Non-parametric ART ANOVA recommended.")
} else {
  print("-> CONCLUSION: Data meets normality assumption (p > 0.05).")
}

# 2. Homogeneity of Variance Check (Levene's Test)
levene_result <- leveneTest(trust_score ~ Architecture * Explainablity_Condition, data = trust_data)
print("2. Levene's Test for Homogeneity of Variance:")
print(levene_result)
if(levene_result$`Pr(>F)`[1] < 0.05) {
  print("-> CONCLUSION: Data violates equal variance assumption (heteroscedasticity detected).")
} else {
  print("-> CONCLUSION: Data meets equal variance assumption.")
}


# ==============================================================================
# PHASE 3: NON-PARAMETRIC REPEATED MEASURES ANALYSIS (ART ANOVA)
# ==============================================================================

# Fit the Aligned Rank Transform ANOVA model.
# (1|participant_id) is the random effect accounting for the within-subjects repeated measures.
art_model_trust <- art(trust_score ~ Architecture * Explainablity_Condition + (1 | participant_id), data = trust_data)
art_anova_results <- anova(art_model_trust)
print(art_anova_results)

# Post-hoc Analysis using Tukey's correction for multiple comparisons
interaction_contrasts <- art.con(art_model_trust, "Architecture:Explainablity_Condition", adjust = "tukey")
contrast_summary_df <- as.data.frame(summary(interaction_contrasts))

print("--- Tukey Post-Hoc Contrasts ---")
summary(interaction_contrasts)

plot_score_bar <- ggplot(trust_data, aes(x = Architecture, y = trust_score, fill = Explainablity_Condition)) +
  stat_summary(fun = mean, geom = "bar", position = "dodge") +
  labs(title = "Mean Trust Score by Architecture and Explainability Condition",
       x = "RAG Architecture", y = "Mean Trust Score") +
  theme_minimal()
print(plot_score_bar)


# ==============================================================================
# PHASE 3.1: EFFECT SIZES (CORRECTED FOR S4 OBJECTS)
# ==============================================================================

# Calculate Partial Eta-Squared using the F-value and degrees of freedom
eta_squared_results <- F_to_eta2(f = art_anova_results$F, 
                                 df = art_anova_results$Df, 
                                 df_error = art_anova_results$Df.res)

eta_table <- data.frame(Effect = rownames(art_anova_results), eta_squared_results)

# Convert Cohen's d to Hedges' g (applying the correction factor for small sample bias)
hedges_g_results <- t_to_d(t = contrast_summary_df$t.ratio, 
                           df = contrast_summary_df$df)

hedges_g_results$Hedges_g <- hedges_g_results$d * (1 - (3 / (4 * contrast_summary_df$df - 1)))

g_table <- data.frame(Contrast = contrast_summary_df$contrast, 
                      Hedges_g = hedges_g_results$Hedges_g)
print(g_table)


# ==============================================================================
# PHASE 3.2: QUANTIFYING PERFORMANCE DIFFERENCES (PAIRWISE CONTRASTS)
# ==============================================================================

# 1. Main Effect of Explainablity_Condition (Averaged across architectures)
explainability_diffs <- emmeans(art_model_trust, ~ Explainablity_Condition)
explainability_contrasts <- contrast(explainability_diffs, method = "pairwise", infer = TRUE)
explainability_diff_table <- as.data.frame(explainability_contrasts)

# 2. Main Effect of Architecture (Averaged across explainability conditions)
arch_diffs <- emmeans(art_model_trust, ~ Architecture)
arch_contrasts <- contrast(arch_diffs, method = "pairwise", infer = TRUE)
arch_diff_table <- as.data.frame(arch_contrasts)

# 3. Simple Effects / Interaction Contrasts
interaction_diff_table <- as.data.frame(summary(interaction_contrasts))


# ==============================================================================
# PHASE 4: VISUALS (ADDITIONAL PLOTS)
# ==============================================================================

# Violin & Boxplot Combo: Displays the full density shape alongside standard quartiles
plot_violin_box <- ggplot(trust_data, aes(x = Architecture, y = trust_score, fill = Explainablity_Condition)) +
  geom_violin(alpha = 0.5, trim = FALSE, position = position_dodge(width = 0.8)) +
  geom_boxplot(width = 0.2, color = "black", alpha = 0.9, position = position_dodge(width = 0.8)) +
  scale_fill_viridis_d(option = "mako", end = 0.8) +
  labs(title = "User Trust Scores: Violin & Boxplot",
       x = "RAG Architecture", y = "Trust Score (1-5)") +
  theme_classic(base_size = 14)
print(plot_violin_box)

# Summary for 95% confidence intervals
trust_error_summary <- trust_data %>%
  group_by(Architecture, Explainablity_Condition) %>%
  dplyr::summarise(
    N = dplyr::n(),
    Mean_Trust = mean(trust_score, na.rm = TRUE),
    SD_Trust = sd(trust_score, na.rm = TRUE),
    SE_Trust = SD_Trust / sqrt(N),
    CI_Trust = qt(0.975, df = N - 1) * SE_Trust, 
    .groups = 'drop'
  )

pd <- position_dodge(0.1) # Prevents overlapping error bars

# Interaction Line Plot with 95% CIs: Highlights how Architecture and Explainability interact
plot_interaction_ci <- ggplot(trust_error_summary, aes(x = Explainablity_Condition, y = Mean_Trust, group = Architecture, color = Architecture)) +
  geom_line(linewidth = 1.2, position = pd) +
  geom_errorbar(aes(ymin = Mean_Trust - CI_Trust, ymax = Mean_Trust + CI_Trust), 
                width = 0.15, linewidth = 1, position = pd) +
  geom_point(size = 4, position = pd) +
  scale_color_viridis_d(option = "cividis", end = 0.8) +
  labs(
       x = "Explainability Condition", y = "Mean Trust Score") +
  theme_classic(base_size = 12)
print(plot_interaction_ci)


# ==============================================================================
# PHASE 5: SILENT STORAGE TO TARGET DIRECTORY
# ==============================================================================

output_dir <- "E:/THSI/Code/statistical_analyses _R/outputs"

# 1. Save core summaries and effect sizes as CSV files
write.csv(trust_summary, file.path(output_dir, "Trust_Score_Summary.csv"), row.names = FALSE)
write.csv(eta_table, file.path(output_dir, "Trust_ANOVA_EtaSquared.csv"), row.names = FALSE)
write.csv(g_table, file.path(output_dir, "Trust_Hedges_G.csv"), row.names = FALSE)
write.csv(trust_error_summary, file.path(output_dir, "Trust_Error_Summary_CI.csv"), row.names = FALSE)

# 2. Save quantified model-based differences as CSV files
write.csv(explainability_diff_table, file.path(output_dir, "Trust_Explainability_Differences.csv"), row.names = FALSE)
write.csv(arch_diff_table, file.path(output_dir, "Trust_Architecture_Differences.csv"), row.names = FALSE)
write.csv(interaction_diff_table, file.path(output_dir, "Trust_Interaction_Contrasts_Differences.csv"), row.names = FALSE)

# 3. Save raw mean differences as CSV files
write.csv(raw_diffs_explainability, file.path(output_dir, "Trust_Raw_Diffs_Explainability.csv"), row.names = FALSE)
write.csv(raw_diffs_architecture, file.path(output_dir, "Trust_Raw_Diffs_Architecture.csv"), row.names = FALSE)

# 4. Sink raw statistical structures directly to the text file report
sink(file.path(output_dir, "Trust_Full_Statistical_Report.txt"))
shapiro_result
levene_result
art_anova_results
summary(interaction_contrasts)
explainability_contrasts
arch_contrasts
sink()