# =======================================================================================
# Script: 11_correlation_density_scatter_plot.R
# Purpose: Visualization of Trust vs. Explainability Correlation (Research_Question_3)
#
# This script loads the empirical user evaluation data and generates a 2D density scatter plot.
# The plot is faceted by Architecture and Explainability Condition to visually demonstrate 
# the relationship between Perceived Explainability and User Trust.
# =======================================================================================

# Step 1: Install required packages if missing
packages <- c("dplyr", "ggplot2", "readr")
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

# Step 2: Load packages
library(dplyr)   # For data manipulation 
library(ggplot2) # For plotting
library(readr)   # For robust CSV I/O

# ==============================================================================
# PHASE 1: DATA INGESTION
# ==============================================================================
# Load your empirical data
cat("\nSelect your cleaned Trust Data file...\n")
my_data <- read_csv(choose.files(), show_col_types = FALSE)

# ==============================================================================
# PHASE 2: GENERATE THE DENSITY SCATTER PLOT
# ==============================================================================
my_plot <- ggplot(my_data, aes(x = explanation_score, y = trust_score)) +
  
  # Add the raw data points (semi-transparent to visualize dense clusters)
  geom_point(alpha = 0.5, color = "#2c3e50", size = 2) +
  
  # Add the 2D density contour rings (topographical visualization of vote clustering)
  geom_density_2d(color = "#3498db", alpha = 0.8) +
  
  # Add the dashed trend line (visually proves the positive monotonic correlation)
  geom_smooth(method = "lm", color = "#e74c3c", se = TRUE, linetype = "dashed") +
  
  # Split into a 2x2 grid mapped to the experimental conditions
  facet_grid(Architecture ~ Explainablity_Condition) +
  
  # Apply clean, publication-ready academic formatting
  theme_bw() +
  theme(
    strip.background = element_rect(fill = "#f8f9fa", color = "#cccccc"),
    strip.text = element_text(face = "bold", size = 11),
    plot.title = element_text(face = "bold", hjust = 0.5),
    panel.grid.minor = element_blank()
  ) +
  labs(
    x = "Perceived Explainability Effectiveness Score (Likert 1-5)",
    y = "User Trust Score (Likert 1-5)"
  )

# Print directly to RStudio Plots Pane for immediate review
print(my_plot)

# ==============================================================================
# PHASE 3: SILENT STORAGE TO TARGET DIRECTORY
# ==============================================================================
output_dir <- "E:/THSI/Code/statistical_analyses _R/outputs"
if(!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# Save the high-resolution visualization 
ggsave(file.path(output_dir, "Correlation_Density_Scatter_Plot.png"), plot = my_plot, width = 8, height = 6, dpi = 300)

cat(sprintf("\nPlotting complete! Visualization saved to: %s\n", output_dir))