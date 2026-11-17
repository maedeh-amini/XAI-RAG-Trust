"""
=======================================================================================
Script: 01_extract_human_scores_from_excel.py
Purpose: Data Extraction Pipeline for Human Evaluator Scores (Research_Question_2)

This script processes the raw experimental data exported from the Streamlit application 
(ExperimentData.xlsx). The user study logs participant ratings for the four DeepEval 
metrics (Faithfulness, Answer Relevancy, Contextual Recall, Contextual Precision) 
as nested JSON arrays of strings containing percentage signs (e.g., "75%").

This script acts as the ETL (Extract, Transform, Load) step. It parses the JSON, 
cleans the strings into numeric floats, calculates robust descriptive statistics 
(Mean, Median, Standard Deviation, Min, Max) for each RAG pipeline, and outputs 
a clean JSON file ready to be analyzed against the LLM Judge scores.
=======================================================================================
"""

import pandas as pd
import json
import os

def calculate_average_metrics(
    input_filepath, 
    output_filepath=None, 
    column_prefix="single_question_rating_", 
    metrics=None
):
    """
    Extracts and calculates comprehensive metric statistics from JSON-formatted Excel columns.
    
    Args:
        input_filepath (str): Path to the raw ExperimentData.xlsx file.
        output_filepath (str): Target path for the cleaned JSON output.
        column_prefix (str): The column naming convention used in the Excel sheet for ratings.
        metrics (list): The specific RAG evaluation metrics to extract.
        
    Returns:
        dict: A nested dictionary containing descriptive statistics for each metric, grouped by architecture.
    """
    if metrics is None:
        metrics = ["faithfulness", "answer_relevancy", "contextual_recall", "contextual_precision"]

    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"[ERROR] Input file not found: {input_filepath}")
        
    print(f"[INFO] Reading data from: {input_filepath}")
    df = pd.read_excel(input_filepath, sheet_name=0)
    
    # Dynamically locate all evaluation columns tied to the 4 pipeline architectures
    target_columns = [col for col in df.columns if col.startswith(column_prefix)]
    
    if not target_columns:
        print(f"[WARNING] No columns found starting with '{column_prefix}'")
        return {}

    overall_scores = {}

    for col in target_columns:
        # Extract the specific architecture block name (e.g., 'Semantic_Low_Explainable')
        block_name = col.replace(column_prefix, "")
        scores_accumulator = {metric: [] for metric in metrics}
        
        # Iterate through every participant's row for this architecture
        for cell_value in df[col]:
            if pd.isna(cell_value):
                continue
                
            try:
                # The data is stored as a stringified JSON array
                ratings_list = json.loads(str(cell_value))
                for item in ratings_list:
                    scores = item.get("scores", {})
                    
                    for metric in metrics:
                        score_val = scores.get(metric)
                        
                        # Data Cleaning: Participant UI allowed "I do not know" or percentage strings
                        if isinstance(score_val, str) and "%" in score_val:
                            try:
                                # Strip the % sign and any extra whitespace, then convert to a float
                                numeric_score = float(score_val.replace("%", "").strip())
                                scores_accumulator[metric].append(numeric_score)
                            except ValueError:
                                pass # Skip if the value before '%' wasn't a valid number
            except json.JSONDecodeError:
                continue
                
        # Calculate the comprehensive descriptive statistics for each metric in this block
        overall_scores[block_name] = {}
        for metric in metrics:
            metric_values = scores_accumulator[metric]
            
            if metric_values:
                # Convert the list to a pandas Series to easily extract statistical functions
                s = pd.Series(metric_values)
                overall_scores[block_name][metric] = {
                    "Mean": round(s.mean(), 2),
                    "Median": round(s.median(), 2),
                    "SD": round(s.std(), 2) if len(s) > 1 else 0.0,
                    "Min": round(s.min(), 2),
                    "Max": round(s.max(), 2)
                }
            else:
                # Fallback for empty data
                overall_scores[block_name][metric] = {
                    "Mean": 0.0,
                    "Median": 0.0,
                    "SD": 0.0,
                    "Min": 0.0,
                    "Max": 0.0
                }

    # Save the transformed statistical structure to the target JSON file
    if output_filepath:
        # Ensure the directory for the output file exists before writing
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        with open(output_filepath, "w", encoding="utf-8") as outfile:
            json.dump(overall_scores, outfile, indent=4)
        print(f"[INFO] Results successfully saved to: {output_filepath}")

    return overall_scores


if __name__ == "__main__":
    # --- PATH SETUP ---
    # Dynamically locate the outputs folder based on the script's local execution path
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SCRIPT_DIR)
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    
    # Define file paths (Adjust relative paths as needed for your final repository structure)
    INPUT_FILE = os.path.join(OUTPUT_DIR, "../outputs/ExperimentData.xlsx")
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "human_evaluation_overall_scores_02.json")
    
    # Run the generalized extraction function
    results = calculate_average_metrics(
        input_filepath=INPUT_FILE,
        output_filepath=OUTPUT_FILE,
        column_prefix="single_question_rating_"
    )
    
    # Display the final descriptive statistics in the console
    print("\nOverall Human Evaluation Scores:")
    print(json.dumps(results, indent=4))