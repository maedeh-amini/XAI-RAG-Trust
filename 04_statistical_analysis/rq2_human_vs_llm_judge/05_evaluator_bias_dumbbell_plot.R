# =======================================================================================
# Script: 02_evaluator_bias_dumbbell_plot.R
# Purpose: Visualizing Measurement Bias between LLM and Human Judges (Research_Question_2)
#
# This script uses `ggplot2` to construct a "Dumbbell Plot" (connected dot plot). 
# This visualization explicitly highlights the magnitude of disagreement (the gap/delta) 
# between automated DeepEval scoring (LLM Judge) and participant scoring (Human Judge) 
# across the four core RAG metrics, separated by underlying architecture.
# =======================================================================================

# Step 1: Install packages if necessary (Uncomment if running on a fresh environment)
install.packages(c("ggplot2", "dplyr"))
library(ggplot2) # For advanced data visualization
library(dplyr)   # For data manipulation

# ==============================================================================
# PHASE 1: DATA PREPARATION (WIDE FORMAT)
# ==============================================================================
# Prepare the dataset in wide format. 
# Unlike the long format used for ANOVA, wide format (where LLM and Human are 
# separate columns) is required for ggplot's geom_segment to easily map the 
# start (x) and end (xend) points of the connecting lines.
df <- data.frame(
  Metric = factor(rep(c("Contextual Recall", "Contextual Precision", "Faithfulness", "Answer Relevancy"), 2),
                  levels = c("Contextual Recall", "Contextual Precision", "Faithfulness", "Answer Relevancy")),
  Architecture = factor(rep(c("Semantic", "Graph"), each = 4), levels = c("Semantic", "Graph")),
  LLM = c(62.00, 58.90, 56.60, 77.80, 78.00, 71.60, 84.50, 75.40),
  Human = c(66.70, 80.51, 78.85, 84.39, 77.55, 83.21, 82.73, 85.75)
)

# ==============================================================================
# PHASE 2: AESTHETICS & COLOR PALETTE
# ==============================================================================
# Define a colorblind-friendly and academically distinct color palette
llm_color <- "#0E4C92"     # Dark Blue for LLM Judge (Automated baseline)
human_color <- "#E8A024"   # Yellow/Gold for Human Judge (Participant baseline)
segment_color <- "#ADB5BD" # Medium-gray connecting line to emphasize the delta gap

# ==============================================================================
# PHASE 3: PLOT CONSTRUCTION
# ==============================================================================
# Generate the Dumbbell Plot 
p <- ggplot(df, aes(y = Metric)) +
  
  # 1. The Connector: Draws the physical line representing the measurement bias (Delta)
  geom_segment(aes(x = LLM, xend = Human, y = Metric, yend = Metric), 
               color = segment_color, size = 2) +
  
  # 2. The Nodes: Plots the specific mean scores for both evaluator types
  geom_point(aes(x = LLM, color = "LLM Judge"), size = 4.5) +
  geom_point(aes(x = Human, color = "Human Judge"), size = 4.5) +
  
  # 3. Faceting: Splits the plot into two distinct panels for Semantic vs. Graph
  facet_wrap(~ Architecture) +
  
  # Map the defined colors to the legend
  scale_color_manual(
    name = "Evaluator",
    values = c("LLM Judge" = llm_color, "Human Judge" = human_color)
  ) +
  
  labs(
    x = "Score (50 - 90)",
    y = NULL
  ) +
  
  # Zoomed-in scale constraint: Restricting the X-axis to 50-90 instead of 0-100.
  # Rationale: All means fall within this range. A 0-100 scale would visually compress 
  # the deltas, masking the severity of the bias (e.g., the massive 20+ point gaps 
  # in Contextual Precision and Faithfulness).
  coord_cartesian(xlim = c(50, 90)) +
  
  # Apply clean theme adjustments
  theme_minimal(base_size = 13) + 
  theme(
    legend.position = "top",
    legend.title = element_blank(),
    legend.text = element_text(size = 11, face = "bold"),
    panel.grid.major.x = element_line(color = "#E9ECEF", size = 0.5), # Keep vertical gridlines for easy score reading
    panel.grid.major.y = element_blank(),                             # Remove horizontal gridlines to avoid cluttering the dumbbells
    panel.border = element_rect(color = "#DEE2E6", fill = NA, size = 0.8),
    strip.text = element_text(face = "bold", size = 13, margin = margin(b = 10)),
    plot.title = element_text(face = "bold", size = 14, margin = margin(b = 15), hjust = 0),
    axis.text.y = element_text(face = "bold", color = "#343A40")
  )

# Display the plot in the RStudio viewer
print(p)

# Optional: Adding ggsave() here if wishing to export the plot directly to the outputs folder
# ggsave("E:/THSI/Code/statistical_analyses _R/outputs/evaluator_bias_dumbbell.png", plot = p, width = 10, height = 6, dpi = 300)