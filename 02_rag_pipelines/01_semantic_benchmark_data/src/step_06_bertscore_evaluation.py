"""
BERTScore Evaluation Module for Semantic RAG Benchmark.

This script executes an automated quantitative evaluation of the generated answers 
against ground-truth references using BERTScore. It leverages a pre-trained language 
model (distilroberta-base) to compute token-level semantic similarity (Precision, 
Recall, and F1). The output is formatted into individual and summary datasets 
designed specifically for downstream statistical testing (e.g., ANOVA).
"""

import json
import pandas as pd
from tqdm import tqdm
from bert_score import score
from step_05_search import RAGSearch

def load_test_dataset(filepath="data/downsampled_data/ragbench_hotpotqa_50_sampled.jsonl"):
    """
    Loads the downsampled ground-truth dataset from a local JSONL file.
    
    Args:
        filepath (str): Path to the JSONL dataset containing questions and references.
        
    Returns:
        list of dict: A list containing dictionaries with 'id', 'question', 
                      and 'reference' keys for evaluation.
    """
    df = pd.read_json(filepath, lines=True)
    
    test_examples = []
    for _, row in df.iterrows():
        test_examples.append({
            "id": row["id"],
            "question": row["question"],
            "reference": row["response"]
        })
        
    print(f"[INFO] Loaded {len(test_examples)} test examples from {filepath}")
    return test_examples


def evaluate_rag_with_bertscore(rag_search: RAGSearch, test_examples, output_prefix="bertscore_semantic_BenchmarkData"):
    """
    Executes the RAG pipeline on test examples and evaluates generation quality using BERTScore.
    
    Generates three output files:
    1. A CSV file containing individual scores (ideal for R/Python ANOVA scripts).
    2. A JSONL file containing the same individual scores for downstream pipeline ingestion.
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
    print("\n[INFO] Computing BERTScore...")
    # 'distilroberta-base' is chosen for a strong balance between evaluation speed 
    # and metric correlation with human judgment on English text.
    P, R, F1 = score(
        predictions,
        references,
        lang="en",
        model_type="distilroberta-base",
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
        "BERTScore_F1": f1_list  # This is the primary "BERTScore" for statistical comparison
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

    # Load test data from the JSONL file generated earlier
    test_data = load_test_dataset("ragbench_hotpotqa_50_sampled.jsonl")

    # Evaluate and save results
    evaluate_rag_with_bertscore(
        rag_search, 
        test_data, 
        output_prefix="bertscore_semantic_BenchmarkData"
    )












