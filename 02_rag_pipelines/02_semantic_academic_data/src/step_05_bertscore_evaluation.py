"""
BERTScore Evaluation Module for Academic Dataset Semantic RAG Pipeline.

This script executes an automated quantitative evaluation of the generated answers 
against the academic ground-truth references using BERTScore. It computes token-level 
semantic similarity (Precision, Recall, and F1) and exports the results into 
individual (CSV/JSONL) and summary datasets tailored for downstream statistical testing.
"""

import json
import pandas as pd
from tqdm import tqdm
from bert_score import score
from step_04_search import RAGSearch

def load_test_dataset(filepath="AcademicData.jsonl"):
    """
    Loads the evaluation dataset from a local JSONL file.
    
    Args:
        filepath (str): Path to the JSONL dataset containing questions and references.
                        Defaults to "AcademicData.jsonl".
                        
    Returns:
        list of dict: A list containing dictionaries with 'id', 'question', 
                      and 'reference' keys mapped for evaluation.
    """
    df = pd.read_json(filepath, lines=True)
    
    test_examples = []
    for _, row in df.iterrows():
        test_examples.append({
            # Map the exact keys from the new AcademicData.jsonl structure
            "id": row.get("Question_ID", f"q_{_}"),
            "question": row["Question"],
            "reference": row["Reference"]
        })
        
    print(f"[INFO] Loaded {len(test_examples)} test examples from {filepath}")
    return test_examples


def evaluate_rag_with_bertscore(rag_search: RAGSearch, test_examples, output_prefix="bertscore_semantic_AcademicData"):
    """
    Executes the RAG pipeline on test examples and evaluates generation quality using BERTScore.
    
    Generates three output files:
    1. A CSV file containing individual scores (for ANOVA/statistical analysis).
    2. A JSONL file containing the same individual scores.
    3. A JSON summary file with macro-averaged metrics.
    
    Args:
        rag_search (RAGSearch): The initialized RAG pipeline instance.
        test_examples (list of dict): The loaded testing data.
        output_prefix (str): Prefix for the generated output files.
        
    Returns:
        tuple: (avg_precision, avg_recall, avg_f1) as float values.
    """
    ids = []
    questions = []
    references = []
    predictions = []

    for example in tqdm(test_examples, desc="Evaluating RAG on test set"):
        q_id = example["id"]
        question = example["question"]
        reference = example["reference"]

        # 1. Execute RAG query (Requests DeepEval-compatible JSON output)
        rag_output = rag_search.query_rag(question, top_k=5)
        
        # 2. Parse the JSON response
        # Fallback mechanism in case the LLM fails to output valid JSON
        try:
            rag_output_json = json.loads(rag_output)
            predicted_answer = rag_output_json.get("answer", "")
        except json.JSONDecodeError:
            predicted_answer = rag_output

        ids.append(q_id)
        questions.append(question)
        references.append(reference)
        predictions.append(predicted_answer)

    # 3. Compute BERTScore
    print("\n[INFO] Computing BERTScore (Language: English)...")
    P, R, F1 = score(
        predictions,
        references,
        lang="en", # Updated to English to match the sample data
        batch_size=32, 
        rescale_with_baseline=False
    )
    
    # Convert PyTorch tensors to standard Python lists for serialization
    p_list = P.tolist()
    r_list = R.tolist()
    f1_list = F1.tolist()

    # 4. Create a DataFrame for individual scores
    results_df = pd.DataFrame({
        "Question_ID": ids,
        "Question": questions,
        "Reference": references,
        "Prediction": predictions,
        "BERTScore_Precision": p_list,
        "BERTScore_Recall": r_list,
        "BERTScore_F1": f1_list # This is the primary "BERTScore" for statistical comparison
    })
    
    # Save individual scores to BOTH CSV and JSONL to support diverse analysis workflows
    csv_filename = f"{output_prefix}.csv"
    jsonl_filename = f"{output_prefix}.jsonl"
    
    results_df.to_csv(csv_filename, index=False)
    results_df.to_json(jsonl_filename, orient="records", lines=True)
    print(f"[INFO] Saved individual scores to {csv_filename} and {jsonl_filename}")

    # Calculate overall macro-averages
    avg_precision = P.mean().item()
    avg_recall = R.mean().item()
    avg_f1 = F1.mean().item()

    # Save the final overall scores to a separate summary file
    summary_data = {
        "Overall_Precision": avg_precision,
        "Overall_Recall": avg_recall,
        "Overall_F1": avg_f1
    }
    summary_filename = f"{output_prefix}_summary.json"
    with open(summary_filename, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"[INFO] Saved overall summary scores to {summary_filename}")

    print(f"\n[RESULT] OVERALL BERTScore - Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
    return avg_precision, avg_recall, avg_f1


if __name__ == "__main__":
    # Initialize RAG pipeline
    rag_search = RAGSearch()

    # Load test data 
    test_data = load_test_dataset("AcademicData.jsonl")

    # Evaluate and save results
    evaluate_rag_with_bertscore(
        rag_search, 
        test_data, 
        output_prefix="bertscore_semantic_AcademicData"
    )

