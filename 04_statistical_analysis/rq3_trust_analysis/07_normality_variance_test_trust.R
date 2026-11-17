# =======================================================================================
# Script: 07_normality_variance_test_trust.R
# Purpose: ANOVA Assumption Checks for Composite User Trust Scores (Research_Question_3)
#
# This script evaluates whether the `trust_score` dependent variable meets the parametric 
# assumptions required for a standard Repeated-Measures ANOVA. 
#
# 1. Normality (Shapiro-Wilk Test): Checks if the residuals of the linear model are 
#    normally distributed.
# 2. Homoscedasticity (Levene's Test): Checks if the variance is equal across the 
#    Architecture and Explainability experimental groups.
#
# violations in these tests justify the use of the non-parametric Aligned Rank 
# Transform (ART) ANOVA in subsequent analysis scripts.
# =======================================================================================

# Step 1: Load required packages (Uncomment installation lines if running on a fresh environment)
if (!requireNamespace("car", quietly = TRUE)) install.packages("car")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")

library(car)      # Required for Levene's Test
library(dplyr)    # For data handling

# ==============================================================================
# PHASE 1: DATA INGESTION & FACTORIZATION
# ==============================================================================
# Load your cleaned trust data 
# (Load 'Trust_Data_Cleaned_No_Outliers.csv' from the data cleaning step)
cat("\nSelect your cleaned Trust Data file...\n")
trust_data <- read.csv(choose.files(), header = TRUE)

# Ensure categorical independent variables are treated as factors for linear modeling
trust_data$Architecture <- as.factor(trust_data$Architecture)
trust_data$Explainablity_Condition <- as.factor(trust_data$Explainablity_Condition)
trust_data$participant_id <- as.factor(trust_data$participant_id)

# ==============================================================================
# PHASE 2: NORMALITY TEST (Shapiro-Wilk Test on Model Residuals)
# ==============================================================================
# To test normality, we first construct a standard linear model of the main effects
model_trust <- lm(trust_score ~ Architecture * Explainablity_Condition, data = trust_data)

# Run the Shapiro-Wilk test specifically on the residuals of that model
shapiro_trust <- shapiro.test(residuals(model_trust))
print("--- 1. Shapiro-Wilk Test for Normality ---")
print(shapiro_trust)

if(shapiro_trust$p.value < 0.05) {
  print("-> CONCLUSION: Trust score residuals violate normality (p < 0.05). Non-parametric ART ANOVA recommended.")
} else {
  print("-> CONCLUSION: Trust score residuals meet the normality assumption (p >= 0.05).")
}

# Optional: Q-Q Plot for visual confirmation of normality
# (Heavy tails or deviation from the red line visually confirms the Shapiro-Wilk result)
qqnorm(residuals(model_trust), main = "Q-Q Plot for Trust Score Residuals")
qqline(residuals(model_trust), col = "red", lwd = 2)

# ==============================================================================
# PHASE 3: HOMOGENEITY OF VARIANCE (Levene's Test)
# ==============================================================================
# Test if the variance (spread) of trust scores is significantly different between groups
levene_trust <- leveneTest(trust_score ~ Architecture * Explainablity_Condition, data = trust_data)
print("--- 2. Levene's Test for Homogeneity of Variance ---")
print(levene_trust)

if(levene_trust$`Pr(>F)`[1] < 0.05) {
  print("-> CONCLUSION: Variances are significantly different across groups (Heteroscedasticity detected, p < 0.05).")
} else {
  print("-> CONCLUSION: Variances are homogeneous across groups (Assumption met, p >= 0.05).")
}

# ==============================================================================
# PHASE 4: SILENT STORAGE TO TARGET DIRECTORY
# ==============================================================================
output_dir <- "E:/THSI/Code/statistical_analyses _R/outputs"
if(!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# Export raw test outputs directly into a structured text file report 
sink(file.path(output_dir, "Trust_Assumption_Checks_Report.txt"))
cat("=== 1. SHAPIRO-WILK TEST FOR NORMALITY ===\n")
print(shapiro_trust)
cat("\n=== 2. LEVENE'S TEST FOR HOMOGENEITY OF VARIANCE ===\n")
print(levene_trust)
sink()

print(paste("Assumption check results successfully exported to:", output_dir))