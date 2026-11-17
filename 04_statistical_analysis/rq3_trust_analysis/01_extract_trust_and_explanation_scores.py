"""
=======================================================================================
Script: 01_extract_trust_and_explanation_scores.py
Purpose: Data Extraction Pipeline for RQ3 (User Trust & Perceived Explainability)

This script processes the raw experimental data exported from the Streamlit application. 
It extracts the nested JSON Likert-scale responses and averages them into two primary 
composite metrics:
  1. Trust Score: The mean of Survey Items 1-4.
  2. Explanation Score: The mean of Survey Items 5-7.

It outputs a long-format CSV structure (Scoring_ID, Architecture, Explainability, 
trust_score, explanation_score) which serves as the foundational dataset for all 
subsequent outlier removal, reliability testing, and Mixed ART ANOVA modeling in RQ3.
=======================================================================================
"""

import pandas as pd
import json

def extract_trust_scores(input_file, output_file):
    """
    Parses nested JSON survey data and structures the output for repeated-measures analysis.
    
    Args:
        input_file (str): Path to the raw Streamlit output CSV.
        output_file (str): Destination path for the transformed CSV.
        
    Returns:
        pd.DataFrame: The long-format dataframe containing the composite scores.
    """
    # Load the raw experimental CSV file
    df = pd.read_csv(input_file)
    extracted_data = []
    
    # Map the raw column headers to their respective independent variable categorizations
    conditions = {
        'block_survey_Semantic_Low_Explainable': ('Semantic', 'Low'),
        'block_survey_Semantic_High_Explainable': ('Semantic', 'High'),
        'block_survey_Graph_Low_Explainable': ('Graph', 'Low'),
        'block_survey_Graph_High_Explainable': ('Graph', 'High')
    }
    
    # Iterate through each participant to preserve within-subjects grouping
    for idx, row in df.iterrows():
        participant_id = row['participant_id']
        
        # Check each of the 4 experimental blocks
        for col, (architecture, explainability) in conditions.items():
            cell_data = row[col]
            
            if pd.notna(cell_data):
                try:
                    # Parse the string into a JSON dictionary
                    data_json = json.loads(cell_data)
                    responses = data_json.get('survey_responses', {})
                    
                    # 1. Extract item1 through item4 (Composite Trust Score)
                    trust_items = [responses.get(f'item{i}') for i in range(1, 5)]
                    # Drop None values and convert string Likert integers to floats
                    valid_trust_items = [float(item) for item in trust_items if item is not None]
                    
                    # 2. Extract item5 through item7 (Subjective Explanation Score)
                    exp_items = [responses.get(f'item{i}') for i in range(5, 8)]
                    valid_exp_items = [float(item) for item in exp_items if item is not None]
                    
                    # Calculate mean scores if valid data exists for the block
                    if valid_trust_items:
                        trust_score = sum(valid_trust_items) / len(valid_trust_items)
                        
                        # Calculate explanation score if items 5-7 exist for this block
                        explanation_score = (sum(valid_exp_items) / len(valid_exp_items)) if valid_exp_items else None
                        
                        # Append as a new row (converting wide data to long format)
                        extracted_data.append({
                            'participant_id': participant_id,
                            'Architecture': architecture,
                            'Explainablity_Condition': explainability,
                            'trust_score': round(trust_score, 4),
                            'explanation_score': round(explanation_score, 4) if explanation_score is not None else None
                        })
                except Exception as e:
                    print(f"Skipping data for {participant_id} in {col} due to error: {e}")
    
    # Convert the extracted list of dictionaries to a pandas DataFrame
    out_df = pd.DataFrame(extracted_data)
    
    # Sort for clean organization: Semantic first, then Graph, then alphabetically by participant
    out_df['Architecture'] = pd.Categorical(out_df['Architecture'], categories=['Semantic', 'Graph'], ordered=True)
    out_df = out_df.sort_values(by=['Architecture', 'Explainablity_Condition', 'participant_id']).reset_index(drop=True)
    
    # Export to CSV for downstream R ingestion
    out_df.to_csv(output_file, index=False)
    print(f"Extraction complete! Data saved to {output_file}")
    
    return out_df

if __name__ == "__main__":
    # Define input file and target output directory
    input_file = r"E:\THSI\Code\statistical_analyses_Py\outputs\ExperimentData_csv.csv"
    output_file = r"E:\THSI\Code\statistical_analyses_Py\outputs\Trust_Explanation_Scores_Extracted.csv"

    # Execute the extraction
    final_df = extract_trust_scores(input_file, output_file)

    # Preview the first few rows in the console for manual verification
    print("\nPreview of extracted scores:")
    print(final_df.head())