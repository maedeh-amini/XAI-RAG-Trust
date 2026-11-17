# ============================================================================================
# Script: 02_art_mixed_anova_answer_relevancy.R
# Purpose: Comprehensive Analysis Pipeline for DeepEval Answer Relevancy (Research_Question_1)
#
# This script evaluates whether the generated chatbot responses directly address 
# the user's question without adding extraneous or irrelevant information. 
# It compares the Semantic and Graph RAG architectures across different dataset 
# complexities (DataType), applying a Mixed-Effects Aligned Rank Transform (ART) 
# ANOVA to account for the repeated-measures experimental design.
# ==========================================================================================

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
library(car)         # For assumption checks
library(effectsize)  # For effect sizes
library(ggdist)      # For Raincloud plots
library(tidyr)       # For data shaping

# Step 3: Load the data
# Prompts the user to select the aggregated Answer Relevancy CSV output
data_AnswerRelevancy <- read.csv(choose.files(), header=T)

# Convert independent variables to categorical factors.
# Question_ID is crucial here as it acts as our random effect grouping variable.
data_AnswerRelevancy$Architecture <- as.factor(data_AnswerRelevancy$Architecture)
data_AnswerRelevancy$DataType <- as.factor(data_AnswerRelevancy$DataType)
data_AnswerRelevancy$Question_ID <- as.factor(data_AnswerRelevancy$Question_ID)
str(data_AnswerRelevancy)

# ==============================================================================
# PHASE 1: EXPLORATORY DATA ANALYSIS (SCORE)
# ==============================================================================

print("--- PHASE 1: Descriptive Statistics for Answer Relevancy Score ---")
# Calculate descriptive statistics using dplyr
score_summary <- data_AnswerRelevancy %>%
  group_by(Architecture, DataType) %>%
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
plot_score_hist_curve <- ggplot(data_AnswerRelevancy, aes(x = Score, fill = DataType, color = DataType)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 0.05, position = "identity", alpha = 0.4) +
  geom_density(alpha = 0.2, linewidth = 1) +
  facet_wrap(~ Architecture) +
  labs(
       x = "Score", 
       y = "Density") +
  theme_minimal()
print(plot_score_hist_curve)

# B. Boxplot to visually check for equal variance and outliers
plot_score_box <- ggplot(data_AnswerRelevancy, aes(x = Architecture, y = Score, fill = DataType)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Boxplot of Answer Relevancy Scores", x = "Architecture", y = "Score") +
  theme_minimal()
print(plot_score_box)

# C. Raincloud Plot for raw Answer Relevancy (Density + Boxplot + Data Points)
plot_score_raincloud <- ggplot(data_AnswerRelevancy, aes(x = Architecture, y = Score, fill = DataType)) +
  # 1. The "Cloud" (Half-Violin Density)
  stat_halfeye(adjust = 0.5, width = 0.5, .width = 0, justification = -0.25, 
               alpha = 0.6, position = position_dodge(width = 0.8)) +
  # 2. The Boxplot (Summary statistics)
  geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.8, 
               position = position_dodge(width = 0.8)) +
  # 3. The "Rain" (Individual jittered data points)
  geom_point(aes(color = DataType), size = 1.2, alpha = 0.4, 
             position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.8)) +
  # Color scaling optimized 
  scale_fill_viridis_d(option = "mako", end = 0.8) +
  scale_color_viridis_d(option = "mako", end = 0.8) +
  labs(x = "RAG Architecture", 
       y = "Answer Relevancy") +
  theme_classic(base_size = 12) + 
  theme(legend.position = "top")

print(plot_score_raincloud)

# ==============================================================================
# PHASE 1.1: QUANTIFYING RAW MEAN DIFFERENCES
# ==============================================================================

# 1. Raw differences between Data Types within each Architecture
raw_diffs_datatype <- score_summary %>%
  select(Architecture, DataType, Mean) %>%
  tidyr::pivot_wider(names_from = DataType, values_from = Mean) %>%
  mutate(Raw_Difference_Benchmark_minus_Academic = Benchmark - Academic)
print("--- Raw Mean Differences by Architecture (Benchmark vs Academic) ---")
print(raw_diffs_datatype)

# 2. Raw differences between Architectures within each Data Type
raw_diffs_architecture <- score_summary %>%
  select(Architecture, DataType, Mean) %>%
  tidyr::pivot_wider(names_from = Architecture, values_from = Mean) %>%
  mutate(Raw_Difference_Graph_minus_Semantic = Graph - Semantic)
print("--- Raw Mean Differences by Data Type (Graph vs Semantic) ---")
print(raw_diffs_architecture)

# ==============================================================================
# PHASE 2: ASSUMPTION CHECKS FOR ANOVA
# ==============================================================================

print("--- PHASE 2: ANOVA Assumption Checks ---")

# 1. Normality Check (Shapiro-Wilk Test on model residuals)
basic_lm <- lm(Score ~ Architecture * DataType, data = data_AnswerRelevancy)

# --- Q-Q Plot for Visual Normality Check ---
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
  print("-> CONCLUSION: Data violates normality assumption (p < 0.05). Non-parametric test recommended.")
} else {
  print("-> CONCLUSION: Data meets normality assumption (p > 0.05).")
}

# 2. Homogeneity of Variance Check (Levene's Test)
levene_result <- leveneTest(Score ~ Architecture * DataType, data = data_AnswerRelevancy)
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
# PHASE 3: NON-PARAMETRIC ANALYSIS (MIXED ART ANOVA)
# ==============================================================================

print("--- PHASE 3: Mixed ART ANOVA Results ---")
# Perform Mixed-Effects ART ANOVA. 
# (1|Question_ID) isolates the random variance of specific questions.
art_model <- art(Score ~ Architecture * DataType + (1|Question_ID), data = data_AnswerRelevancy)
art_anova_results <- anova(art_model)
print(art_anova_results)

# Post-hoc Analysis using Tukey's correction for multiple comparisons
interaction_contrasts <- art.con(art_model, "Architecture:DataType", adjust = "tukey")
print("--- Tukey Post-Hoc Contrasts ---")
summary(interaction_contrasts)

contrast_summary_df <- as.data.frame(summary(interaction_contrasts))

# Visualize the final central tendencies
plot_score_bar <- ggplot(data_AnswerRelevancy, aes(x = Architecture, y = Score, fill = DataType)) +
  stat_summary(fun = mean, geom = "bar", position = "dodge") +
  labs(title = "Mean Answer Relevancy by Architecture and Data Type",
       x = "RAG Architecture", y = "Mean Answer Relevancy Score") +
  theme_minimal()
print(plot_score_bar)


# ==============================================================================
# PHASE 3.1: EFFECT SIZES (CORRECTED FOR S4 OBJECTS)
# ==============================================================================

print("--- PHASE 3.5: Effect Sizes ---")
print("--- Partial Eta-Squared (ANOVA Effect Size) ---")
# Calculate Partial Eta-Squared using the F-value and degrees of freedom
eta_squared_results <- F_to_eta2(f = art_anova_results$F, 
                                 df = art_anova_results$Df, 
                                 df_error = art_anova_results$Df.res)

eta_table <- data.frame(Effect = rownames(art_anova_results), eta_squared_results)
print(eta_table)

print("--- Hedges' g (Post-Hoc Effect Size) ---")
hedges_g_results <- t_to_d(t = contrast_summary_df$t.ratio, 
                           df = contrast_summary_df$df)

# Convert Cohen's d to Hedges' g (applying the correction factor for small sample bias)
hedges_g_results$Hedges_g <- hedges_g_results$d * (1 - (3 / (4 * contrast_summary_df$df - 1)))

g_table <- data.frame(Contrast = contrast_summary_df$contrast, 
                      Hedges_g = hedges_g_results$Hedges_g)
print(g_table)


# ==============================================================================
# PHASE 3.2: QUANTIFYING PERFORMANCE DIFFERENCES (PAIRWISE CONTRASTS)
# ==============================================================================

# 1. Main Effect of DataType (Averaged across architectures)
datatype_diffs <- emmeans(art_model, ~ DataType)
datatype_contrasts <- contrast(datatype_diffs, method = "pairwise", infer = TRUE)
datatype_diff_table <- as.data.frame(datatype_contrasts)

# 2. Main Effect of Architecture (Averaged across data types)
arch_diffs <- emmeans(art_model, ~ Architecture)
arch_contrasts <- contrast(arch_diffs, method = "pairwise", infer = TRUE)
arch_diff_table <- as.data.frame(arch_contrasts)

# 3. Simple Effects / Interaction Contrasts
interaction_diff_table <- as.data.frame(summary(interaction_contrasts))


# ==============================================================================
# PHASE 4: ERROR RATE EXPLORATION & PUBLICATION VISUALS
# ==============================================================================

# Transform raw metric into an Error Rate to emphasize system failures
data_AnswerRelevancy$Error_Rate <- 1 - data_AnswerRelevancy$Score

error_summary <- data_AnswerRelevancy %>%
  group_by(Architecture, DataType) %>%
  dplyr::summarise(
    N = dplyr::n(),
    Mean_Error = mean(Error_Rate, na.rm = TRUE),
    SD_Error = sd(Error_Rate, na.rm = TRUE),
    SE_Error = SD_Error / sqrt(N),
    CI_Error = qt(0.975, df = N - 1) * SE_Error, 
    Max_Error = max(Error_Rate, na.rm = TRUE),
    .groups = 'drop'
  )

print("--- PHASE 4: Answer Relevancy Error Rate Summary with Confidence Intervals ---")
print(error_summary)

plot_error_hist_curve <- ggplot(data_AnswerRelevancy, aes(x = Error_Rate, fill = DataType, color = DataType)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 0.05, position = "identity", alpha = 0.4) +
  geom_density(alpha = 0.2, linewidth = 1) +
  facet_wrap(~ Architecture) +
  labs(title = "Distribution of Answer Relevancy Error Rates with Density Curves", 
       x = "Error Rate", 
       y = "Density") +
  theme_minimal()
print(plot_error_hist_curve)

plot_error_box <- ggplot(data_AnswerRelevancy, aes(x = Architecture, y = Error_Rate, fill = DataType)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Box Plot of Answer Relevancy Error Rates", x = "RAG Architecture", y = "Error Rate") +
  theme_minimal()
print(plot_error_box)

plot_raincloud <- ggplot(data_AnswerRelevancy, aes(x = Architecture, y = Error_Rate, fill = DataType)) +
  stat_halfeye(adjust = 0.5, width = 0.5, .width = 0, justification = -0.2, alpha = 0.6, position = position_dodge(width = 0.8)) +
  geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.8, position = position_dodge(width = 0.8)) +
  scale_fill_viridis_d(option = "mako", end = 0.8) +
  labs(title = "Answer Relevancy Error Rates: Raincloud Distribution",
       x = "RAG Architecture", y = "Error Rate") +
  theme_classic(base_size = 14) +
  theme(legend.position = "top")
print(plot_raincloud)

plot_violin_box <- ggplot(data_AnswerRelevancy, aes(x = Architecture, y = Error_Rate, fill = DataType)) +
  geom_violin(alpha = 0.5, trim = FALSE, position = position_dodge(width = 0.8)) +
  geom_boxplot(width = 0.2, color = "black", alpha = 0.9, position = position_dodge(width = 0.8)) +
  scale_fill_viridis_d(option = "mako", end = 0.8) +
  labs(title = "Answer Relevancy Error Rates: Violin & Boxplot",
       x = "RAG Architecture", y = "Error Rate") +
  theme_classic(base_size = 14)
print(plot_violin_box)

# Interaction Plot with 95% Confidence Interval Error Bars
pd <- position_dodge(0.1) 

plot_interaction_ci <- ggplot(error_summary, aes(x = DataType, y = Mean_Error, group = Architecture, color = Architecture)) +
  geom_line(linewidth = 1.2, position = pd) +
  geom_errorbar(aes(ymin = Mean_Error - CI_Error, ymax = Mean_Error + CI_Error), 
                width = 0.15, linewidth = 1, position = pd) +
  geom_point(size = 4, position = pd) +
  scale_color_viridis_d(option = "cividis", end = 0.8) +
  labs(x = "Data Type", y = "Mean Error Rate") +
  theme_classic(base_size = 12)

print(plot_interaction_ci)

# ==============================================================================
# PHASE 5: SILENT STORAGE TO TARGET DIRECTORY
# ==============================================================================

# Ensure this directory exists on your local machine before running
output_dir <- "E:/THSI/Code/statistical_analyses _R/data and results/converted_data/answer_relevancy"

# 1. Save core summaries and effect sizes as CSV files
write.csv(score_summary, file.path(output_dir, "AnswerRelevancy_Score_Summary.csv"), row.names = FALSE)
write.csv(eta_table, file.path(output_dir, "AnswerRelevancy_ANOVA_EtaSquared.csv"), row.names = FALSE)
write.csv(g_table, file.path(output_dir, "AnswerRelevancy_Hedges_G.csv"), row.names = FALSE)
write.csv(error_summary, file.path(output_dir, "AnswerRelevancy_Error_Summary_CI.csv"), row.names = FALSE)

# 2. Save quantified model-based differences as CSV files
write.csv(datatype_diff_table, file.path(output_dir, "AnswerRelevancy_DataType_Differences.csv"), row.names = FALSE)
write.csv(arch_diff_table, file.path(output_dir, "AnswerRelevancy_Architecture_Differences.csv"), row.names = FALSE)
write.csv(interaction_diff_table, file.path(output_dir, "AnswerRelevancy_Interaction_Contrasts_Differences.csv"), row.names = FALSE)

# 3. Save raw mean differences as CSV files
write.csv(raw_diffs_datatype, file.path(output_dir, "AnswerRelevancy_Raw_Diffs_DataType.csv"), row.names = FALSE)
write.csv(raw_diffs_architecture, file.path(output_dir, "AnswerRelevancy_Raw_Diffs_Architecture.csv"), row.names = FALSE)

# 4. Sink raw statistical structures directly to the text file report
sink(file.path(output_dir, "AnswerRelevancy_Full_Statistical_Report.txt"))
shapiro_result
levene_result
art_anova_results
summary(interaction_contrasts)
datatype_contrasts
arch_contrasts
sink()