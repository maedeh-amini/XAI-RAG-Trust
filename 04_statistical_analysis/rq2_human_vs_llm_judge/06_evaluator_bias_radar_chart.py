"""
===================================================================================================
Script: 05_evaluator_bias_radar_chart.py
Purpose: Visualizing Human-AI Evaluative Alignment and Transparency Impact (Research_Question_2)

This script uses `matplotlib` to generate a dual-panel Radar Chart (Spider Plot). 
It addresses the RQ2 Sub-Question by mapping how system transparency 
(Low-Explainable vs. High-Explainable conditions) influences Human evaluators 
compared to the automated LLM Judge baseline across four core RAG metrics.

The radar chart provides a holistic, multivariate view of system performance, 
visually demonstrating whether High-Explainability expands or shrinks the 
"evaluation footprint" of the system in the eyes of human users.
=================================================================================================
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# ==============================================================================
# PHASE 1: DATA PREPARATION & GEOMETRY
# ==============================================================================
# 1. Define Categories and Angles for the polar plot
categories = ['Contextual Recall', 'Contextual Precision', 'Faithfulness', 'Answer Relevancy']
N = len(categories)

# Calculate the angle for each category on the radar chart (dividing a full 360 degree circle)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1] # Append the first angle to the end to close the circular plot

# 2. Input Data (Extracted from the descriptive statistics pipeline)
# To create a closed polygon on the radar chart, the first value must be appended to the end of each list.

# --- Semantic Architecture Means ---
sem_high_hum = [67.57, 82.44, 82.08, 82.71]
sem_high_hum += sem_high_hum[:1]

sem_llm = [62.00, 58.90, 56.60, 77.80]
sem_llm += sem_llm[:1]

sem_low_hum = [66.70, 80.51, 78.85, 84.39]
sem_low_hum += sem_low_hum[:1]

# --- Graph Architecture Means ---
gra_high_hum = [79.27, 86.22, 83.94, 83.79]
gra_high_hum += gra_high_hum[:1]

gra_llm = [78.00, 71.60, 84.50, 75.40]
gra_llm += gra_llm[:1]

gra_low_hum = [77.55, 83.21, 82.73, 85.75]
gra_low_hum += gra_low_hum[:1]

# ==============================================================================
# PHASE 2: AESTHETICS & COLOR PALETTE
# ==============================================================================
# 3. Styling Configuration
c_high_hum = '#2ca02c' # Green: Represents the augmented transparency condition
c_llm = '#d62728'      # Red: Represents the automated LLM baseline
c_low_hum = '#1f77b4'  # Blue: Represents the standard/low transparency human baseline

# ==============================================================================
# PHASE 3: PLOT CONSTRUCTION
# ==============================================================================
# 4. Set up the Figure
# Width expanded to 18 to give the circular plots sufficient breathing room for labels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), subplot_kw=dict(polar=True))

def plot_radar(ax, d_high, d_llm, d_low, title):
    """
    Renders a single radar chart onto the provided matplotlib axes.
    """
    # Rotate plot so the first category ('Contextual Recall') is fixed at the top (12 o'clock)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1) # Draw clockwise
    
    # Configure radial axes and gridlines
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    
    # Lock the Y-axis limits from 50 to 100 to emphasize the actual variation range
    ax.set_ylim(50, 100)
    ax.set_yticks([50, 60, 70, 80, 90, 100])
    ax.set_yticklabels(['50', '60', '70', '80', '90', '100'], color="grey", size=10)
    
    # Plot High-Explainability (Human) - Solid Green
    ax.plot(angles, d_high, linewidth=2.5, linestyle='solid', marker='o', markersize=6, label='High-Exp. (Human)', color=c_high_hum)
    ax.fill(angles, d_high, c_high_hum, alpha=0.08) # Light alpha fill for readability
    
    # Plot Low-Explainability (LLM Judge) - Dashed Red
    # Dashed line visually separates the automated baseline from the human evaluations
    ax.plot(angles, d_llm, linewidth=2.5, linestyle='dashed', marker='o', markersize=6, label='Low-Exp. (LLM Judge)', color=c_llm)
    ax.fill(angles, d_llm, c_llm, alpha=0.03)
    
    # Plot Low-Explainability (Human) - Solid Blue
    ax.plot(angles, d_low, linewidth=2.5, linestyle='solid', marker='o', markersize=6, label='Low-Exp. (Human)', color=c_low_hum)
    ax.fill(angles, d_low, c_low_hum, alpha=0.08)
    
    ax.set_title(title, size=16, y=1.12)

# ==============================================================================
# PHASE 4: RENDERING & EXPORT
# ==============================================================================
# 5. Render Plots onto the two subplots
plot_radar(ax1, sem_high_hum, sem_llm, sem_low_hum, 'Semantic Architecture')
plot_radar(ax2, gra_high_hum, gra_llm, gra_low_hum, 'Graph Architecture')

# 6. Global Legend and Layout adjustments
# Extract handles from the first axis to create a single, unified legend at the bottom
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=14, frameon=False)

# Added w_pad to force significant horizontal distance between the two radar circles,
# preventing overlapping category text labels.
plt.tight_layout(w_pad=8.0) 
plt.subplots_adjust(bottom=0.15)

# 7. Setup save path and export
save_dir = r"E:\THSI\Code\statistical_analyses _R\RQ_2_data_results"
os.makedirs(save_dir, exist_ok=True) 
save_path = os.path.join(save_dir, "Final_Spider_Plot_Python.png")

# Export at high resolution (300 DPI) 
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()