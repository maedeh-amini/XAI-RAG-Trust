# =======================================================================================
# Script: 03_gender_distribution_pie_chart.R
# Purpose: Generate Categorical Gender Distribution Pie Chart for Demographics
#
# This script calculates the exact frequencies and percentages for the 'Gender' 
# variable from the cleaned demographics dataset.
# 
# It generates a percentage pie chart using polar coordinates,
# utilizing the `ggrepel` package to ensure the percentage labels dynamically push 
# away from each other to prevent text overlap in dense clusters.
#
# Prerequisite: Execute 01_participant_demographics_summary.R first to load `demo_clean`.
# =======================================================================================

# Step 1: Install and load the ggrepel package for non-overlapping labels
if(!require(ggrepel)) install.packages("ggrepel")
library(ggrepel)

# ==============================================================================
# PHASE 1: DATA AGGREGATION
# ==============================================================================
# Calculate frequencies and percentages for Gender
gender_summary <- demo_clean %>%
  filter(!is.na(Gender)) %>%
  count(Gender) %>%
  # Calculate proportion relative to the total N
  mutate(Percentage = n / sum(n) * 100)

# ==============================================================================
# PHASE 2: GENERATE THE PIE CHART
# ==============================================================================
# Create the pie chart by plotting a stacked bar chart and wrapping it into a solid circle
plot_gender <- ggplot(gender_summary, aes(x = 1, y = Percentage, fill = Gender)) + 
  geom_col(color = "black", width = 1) + 
  
  # coord_polar mathematically bends the y-axis into a solid circle (pie)
  coord_polar(theta = "y") +
  
  # Add explicitly styled white rectangles around the labels for high contrast
  geom_label_repel(aes(label = paste0(round(Percentage), "%")), 
                   position = position_stack(vjust = 0.5), 
                   size = 5, 
                   fontface = "bold", 
                   fill = "white",       # Forces a white background for the rectangle
                   color = "black",      # Black border and text for the rectangle
                   label.padding = unit(0.3, "lines"), # Space between text and border
                   label.size = 0.5,     # Thickness of the border
                   show.legend = FALSE) +
  
  # Apply a custom color palette tailored 
  scale_fill_manual(values = c("Female" = "#BE7B79", 
                               "Male" = "#4F62A1", 
                               "Other" = "#18BC9C", 
                               "Prefer not to say" = "#A1A6A7")) +
  labs(
       fill = "") +
  theme_void() + 
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 16, margin = margin(b = 15)),
    legend.position = "right",
    legend.text = element_text(size = 12),
    legend.key = element_rect(color = "black")
  )

# Print to the RStudio viewer for immediate visual confirmation
print(plot_gender)

# ==============================================================================
# PHASE 3: SILENT STORAGE TO TARGET DIRECTORY
# ==============================================================================
# Save the high-resolution plot to the designated outputs folder
ggsave(file.path(output_dir, "Plot_2_Gender_Pie.png"), plot_gender, width = 8, height = 6, dpi = 300)