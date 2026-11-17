# =======================================================================================
# Script: 01_participant_demographics_summary.R
# Purpose: Generate Demographic Summaries and Visualizations
#
# This script extracts the nested JSON demographic data captured from the Streamlit 
# user study. It deduplicates the rows (ensuring each participant is only counted once, 
# despite the repeated-measures design) and generates two outputs:
#   1. Summary table (.docx) for the "Participants" section.
#   2. High-resolution, Lollipop charts for individual demographic 
#      distributions (Age, Gender, Academic Status, Language Proficiency, AI Usage).
# =======================================================================================

# 1. Install and load required packages
required_packages <- c("tidyverse", "jsonlite", "gtsummary", "flextable")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

library(tidyverse) # For data manipulation and ggplot2
library(jsonlite)  # For parsing nested JSON strings
library(gtsummary) # For generating academic summary tables
library(flextable) # For exporting gtsummary tables to Word (.docx)

# ==============================================================================
# PHASE 1: DATA INGESTION & WRANGLING
# ==============================================================================
# 2. Setup Output Directory
output_dir <- "E:/THSI/Code/statistical_analyses _R/outputs"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# 3. Load the data
cat("\nSelect the raw experimental data file (ExperimentData_csv.csv)...\n")
raw_data <- read.csv(file.choose(), stringsAsFactors = FALSE)

# 4. Parse the JSON demographics column and extract variables
demo_clean <- raw_data %>%
  filter(!is.na(demographics) & demographics != "") %>%
  mutate(
    # Unpack the JSON dictionary
    demo_list = map(demographics, ~ tryCatch(fromJSON(.x), error = function(e) list())),
    Age = map_dbl(demo_list, ~ as.numeric(.x$age %||% NA)),
    Gender = map_chr(demo_list, ~ as.character(.x$gender %||% NA)),
    Academic_Status = map_chr(demo_list, ~ as.character(.x$highest_education %||% NA)),
    English_Proficiency = map_chr(demo_list, ~ as.character(.x$english_proficiency %||% NA)),
    German_Proficiency = map_chr(demo_list, ~ as.character(.x$german_proficiency %||% NA)),
    AI_Usage = map_chr(demo_list, ~ as.character(.x$chatbot_usage_frequency %||% NA)),
    Info_Seeking = map_chr(demo_list, ~ as.character(.x$info_seeking_habit %||% NA))
  ) %>%
  select(participant_id, Age, Gender, Academic_Status, English_Proficiency, German_Proficiency, AI_Usage, Info_Seeking) %>%
  # Crucial: Deduplicate the dataset so each participant is only counted once
  distinct(participant_id, .keep_all = TRUE) 

# 5. Set correct factor levels for ordinal and categorical variables
lang_levels <- c("Beginner", "Intermediate", "Advanced", "Native")
ai_levels <- c("Never", "Rarely / Less than once a week", "Around once a week", "Multiple times a week", "Daily")

demo_clean <- demo_clean %>%
  mutate(
    English_Proficiency = factor(English_Proficiency, levels = lang_levels, ordered = TRUE),
    German_Proficiency = factor(German_Proficiency, levels = lang_levels, ordered = TRUE),
    AI_Usage = factor(AI_Usage, levels = ai_levels, ordered = TRUE),
    Gender = as.factor(Gender),
    Academic_Status = as.factor(Academic_Status),
    Info_Seeking = as.factor(Info_Seeking)
  )

# ==============================================================================
# PHASE 2: SUMMARY TABLE 
# ==============================================================================
# Generates a comprehensive descriptive statistics table 
demo_table <- demo_clean %>%
  select(-participant_id) %>%
  tbl_summary(
    statistic = list(
      all_continuous() ~ "{mean} ({sd}) [Min: {min}, Max: {max}]",
      all_categorical() ~ "{n} ({p}%)"
    ),
    label = list(
      Age ~ "Age (Years)",
      Gender ~ "Gender",
      Academic_Status ~ "Current Academic Status",
      English_Proficiency ~ "English Proficiency",
      German_Proficiency ~ "German Proficiency",
      AI_Usage ~ "AI Chatbot Usage Frequency",
      Info_Seeking ~ "Information Seeking Habit"
    )
  ) %>%
  bold_labels()

# Export the table as a Word document 
demo_table %>%
  as_flex_table() %>%
  save_as_docx(path = file.path(output_dir, "Table_4_X_Demographic_Summary.docx"))

# ==============================================================================
# PHASE 3: VISUALIZATIONS (INDIVIDUAL PLOTS)
# ==============================================================================
# Set a clean base theme for all plots
theme_set(theme_minimal(base_size = 14) + theme(plot.title = element_text(face = "bold")))

# 1. Age Distribution (Histogram remains standard for continuous numerical data)
plot_age <- ggplot(demo_clean, aes(x = Age)) +
  geom_histogram(binwidth = 2, fill = "#2C3E50", color = "white", alpha = 0.9) +
  labs(title = "Age Distribution", x = "Age", y = "Count")
ggsave(file.path(output_dir, "Plot_1_Age.png"), plot_age, width = 8, height = 6, dpi = 300)

# 2. Gender (Vertical Lollipop Chart)
gender_sum <- demo_clean %>% filter(!is.na(Gender)) %>% count(Gender) %>% mutate(pct = n/sum(n)*100)
plot_gender <- ggplot(gender_sum, aes(x = fct_reorder(Gender, -pct), y = pct)) +
  geom_segment(aes(xend = fct_reorder(Gender, -pct), y = 0, yend = pct), linetype = "dotted", color = "#8E44AD", linewidth = 1.2) +
  geom_point(color = "#8E44AD", size = 14) +
  geom_text(aes(label = paste0(round(pct), "%")), color = "white", fontface = "bold", size = 4.5) +
  labs(title = "Gender Distribution", x = "Gender", y = "Percentage (%)") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  theme(panel.grid.major.x = element_blank())
ggsave(file.path(output_dir, "Plot_2_Gender.png"), plot_gender, width = 8, height = 6, dpi = 300)

# 3. Academic Status (Horizontal Lollipop for long labels)
acad_sum <- demo_clean %>% filter(!is.na(Academic_Status)) %>% count(Academic_Status) %>% mutate(pct = n/sum(n)*100)
plot_acad <- ggplot(acad_sum, aes(x = pct, y = fct_reorder(Academic_Status, pct))) +
  geom_segment(aes(x = 0, xend = pct, yend = fct_reorder(Academic_Status, pct)), linetype = "dotted", color = "#18BC9C", linewidth = 1.2) +
  geom_point(color = "#18BC9C", size = 14) +
  geom_text(aes(label = paste0(round(pct), "%")), color = "white", fontface = "bold", size = 4.5) +
  labs(title = "Academic Status", x = "Percentage (%)", y = "") +
  scale_y_discrete(labels = function(x) stringr::str_wrap(x, width = 40)) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
  theme(panel.grid.major.y = element_blank())
ggsave(file.path(output_dir, "Plot_3_Academic_Status.png"), plot_acad, width = 10, height = 6, dpi = 300)

# 4. English Proficiency (Vertical Lollipop - respects ordered factors)
eng_sum <- demo_clean %>% filter(!is.na(English_Proficiency)) %>% count(English_Proficiency) %>% mutate(pct = n/sum(n)*100)
plot_eng <- ggplot(eng_sum, aes(x = English_Proficiency, y = pct)) +
  geom_segment(aes(xend = English_Proficiency, y = 0, yend = pct), linetype = "dotted", color = "#3498DB", linewidth = 1.2) +
  geom_point(color = "#3498DB", size = 14) +
  geom_text(aes(label = paste0(round(pct), "%")), color = "white", fontface = "bold", size = 4.5) +
  labs(title = "English Proficiency", x = "Proficiency Level", y = "Percentage (%)") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  theme(panel.grid.major.x = element_blank())
ggsave(file.path(output_dir, "Plot_4_English_Proficiency.png"), plot_eng, width = 8, height = 6, dpi = 300)

# 5. German Proficiency (Vertical Lollipop - respects ordered factors)
ger_sum <- demo_clean %>% filter(!is.na(German_Proficiency)) %>% count(German_Proficiency) %>% mutate(pct = n/sum(n)*100)
plot_ger <- ggplot(ger_sum, aes(x = German_Proficiency, y = pct)) +
  geom_segment(aes(xend = German_Proficiency, y = 0, yend = pct), linetype = "dotted", color = "#E74C3C", linewidth = 1.2) +
  geom_point(color = "#E74C3C", size = 14) +
  geom_text(aes(label = paste0(round(pct), "%")), color = "white", fontface = "bold", size = 4.5) +
  labs(title = "German Proficiency", x = "Proficiency Level", y = "Percentage (%)") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  theme(panel.grid.major.x = element_blank())
ggsave(file.path(output_dir, "Plot_5_German_Proficiency.png"), plot_ger, width = 8, height = 6, dpi = 300)

# 6. AI Chatbot Usage Frequency (Vertical Lollipop - respects ordered factors)
ai_sum <- demo_clean %>% filter(!is.na(AI_Usage)) %>% count(AI_Usage) %>% mutate(pct = n/sum(n)*100)
plot_ai <- ggplot(ai_sum, aes(x = AI_Usage, y = pct)) +
  geom_segment(aes(xend = AI_Usage, y = 0, yend = pct), linetype = "dotted", color = "#9B59B6", linewidth = 1.2) +
  geom_point(color = "#9B59B6", size = 14) +
  geom_text(aes(label = paste0(round(pct), "%")), color = "white", fontface = "bold", size = 4.5) +
  labs(title = "AI Chatbot Usage Frequency", x = "", y = "Percentage (%)") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  scale_x_discrete(labels = function(x) stringr::str_wrap(x, width = 15)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), panel.grid.major.x = element_blank())
ggsave(file.path(output_dir, "Plot_6_AI_Usage.png"), plot_ai, width = 8, height = 6, dpi = 300)

# 7. Information Seeking Habits (Horizontal Lollipop for long labels)
info_sum <- demo_clean %>% filter(!is.na(Info_Seeking)) %>% count(Info_Seeking) %>% mutate(pct = n/sum(n)*100)
plot_info <- ggplot(info_sum, aes(x = pct, y = fct_reorder(Info_Seeking, pct))) +
  geom_segment(aes(x = 0, xend = pct, yend = fct_reorder(Info_Seeking, pct)), linetype = "dotted", color = "#F39C12", linewidth = 1.2) +
  geom_point(color = "#F39C12", size = 14) +
  geom_text(aes(label = paste0(round(pct), "%")), color = "white", fontface = "bold", size = 4.5) +
  labs(title = "Information Seeking Habits", x = "Percentage (%)", y = "") +
  scale_y_discrete(labels = function(x) stringr::str_wrap(x, width = 40)) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
  theme(panel.grid.major.y = element_blank())
ggsave(file.path(output_dir, "Plot_7_Info_Seeking.png"), plot_info, width = 10, height = 6, dpi = 300)

cat("\nAll independent demographic plots and the summary table have been successfully saved to:", output_dir, "\n")