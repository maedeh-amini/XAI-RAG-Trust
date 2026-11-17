"""
==============================================================================================
Script: 02_metric_per_human_extraction.py
Purpose: Data Extraction Pipeline for Human Evaluator Mixed ART ANOVA (Research_Question_2)

This script processes the raw experimental data exported from the Streamlit application 
(ExperimentData.xlsx). It is designed as a generalized template to reformat raw Human 
evaluation scores for ANY metric (Faithfulness, Answer Relevancy, etc.) into a strict 
long-format CSV structure. 

This specific formatting (Scoring_ID, Evaluator_Type, Architecture, Score) is 
mandatory for the data to be ingested by the R scripts for Mixed-Effects ART ANOVA 
non-parametric statistical testing.
==============================================================================================
"""

import pandas as pd
import json

def extract_metric_data(file_path, output_path, metric_name):
    """
    Parses JSON arrays from Excel and structures the output for Mixed ART ANOVA.
    
    Args:
        file_path (str): The absolute path to the Streamlit output Excel file.
        output_path (str): The destination path for the transformed CSV.
        metric_name (str): The exact JSON key for the metric being extracted 
                           (e.g., 'faithfulness', 'contextual_recall').
        
    Returns:
        pd.DataFrame: The long-format dataframe containing the cleaned scores.
    """
    # Load the raw experimental data
    df = pd.read_excel(file_path)
    data_extracted = []

    # Iterate through each participant (row) to preserve the repeated-measures grouping
    for idx, row in df.iterrows():
        participant_id = row['participant_id']
        
        # 1. Process Semantic Low Explainable Condition
        semantic_data = row['single_question_rating_Semantic_Low_Explainable']
        if pd.notna(semantic_data):
            try:
                sem_json = json.loads(semantic_data)
                sem_scores = []
                for item in sem_json:
                    # Dynamically retrieve the specified metric score
                    score_str = item['scores'].get(metric_name, '')
                    
                    # Data Cleaning: Exclude "I do not know" edge-cases and convert percentages to decimals
                    if score_str != 'I do not know' and '%' in score_str:
                        sem_scores.append(float(score_str.strip('%')) / 100)
                
                # If valid scores exist, calculate the participant's mean score for this condition
                if sem_scores:
                    sem_avg = sum(sem_scores) / len(sem_scores)
                    data_extracted.append({
                        'Scoring_ID': participant_id,           # Random Effect grouping variable
                        'Evaluator_Type': 'Human_Annotator',    # Fixed Effect 1
                        'Architecture': 'Semantic',             # Fixed Effect 2
                        'Score': round(sem_avg, 4)              # Continuous Dependent Variable
                    })
            except Exception as e:
                print(f"Skipping Semantic row for {participant_id} due to format error: {e}")
                
        # 2. Process Graph Low Explainable Condition
        graph_data = row['single_question_rating_Graph_Low_Explainable']
        if pd.notna(graph_data):
            try:
                graph_json = json.loads(graph_data)
                graph_scores = []
                for item in graph_json:
                    # Dynamically retrieve the specified metric score
                    score_str = item['scores'].get(metric_name, '')
                    
                    if score_str != 'I do not know' and '%' in score_str:
                        graph_scores.append(float(score_str.strip('%')) / 100)
                
                if graph_scores:
                    graph_avg = sum(graph_scores) / len(graph_scores)
                    data_extracted.append({
                        'Scoring_ID': participant_id,
                        'Evaluator_Type': 'Human_Annotator',
                        'Architecture': 'Graph',
                        'Score': round(graph_avg, 4)
                    })
            except Exception as e:
                print(f"Skipping Graph row for {participant_id} due to format error: {e}")

    # Convert the list of dictionaries to a pandas DataFrame
    out_df = pd.DataFrame(data_extracted)
    
    # Custom sort: Group by Architecture (Semantic first, then Graph), then alphabetically by ID
    # This matches the standard visual output expected in R studio dataframes
    out_df['Architecture'] = pd.Categorical(out_df['Architecture'], categories=['Semantic', 'Graph'], ordered=True)
    out_df = out_df.sort_values(by=['Architecture', 'Scoring_ID']).reset_index(drop=True)
    
    # Export to CSV for R ingestion
    out_df.to_csv(output_path, index=False)
    print(f"Extraction complete for '{metric_name}'! Data saved to {output_path}")
    
    return out_df

if __name__ == "__main__":
    # --- CONFIGURATION SECTION ---
    # Change this single variable to extract a different metric 
    # Options: 'faithfulness', 'answer_relevancy', 'contextual_recall', 'contextual_precision'
    TARGET_METRIC = 'faithfulness' 
    
    # Define input file and target output directory dynamically based on the metric
    input_file = r"E:\THSI\Code\statistical_analyses_Py\outputs\ExperimentData.xlsx"
    
    # Automatically names the file based on the target metric (e.g., "Human_Judgements_faithfulness.csv")
    output_path = rf"E:\THSI\Code\statistical_analyses_Py\outputs\Human_Judgements_{TARGET_METRIC}.csv"
    
    # Execute the extraction
    final_df = extract_metric_data(input_file, output_path, TARGET_METRIC)
    
    # Preview the first few rows in the console for manual verification
    print("\nPreview of output data:")
    print(final_df.head())