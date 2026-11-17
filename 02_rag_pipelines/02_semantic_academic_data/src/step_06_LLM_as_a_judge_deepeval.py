"""
DeepEval LLM-as-a-Judge Evaluation Module for Academic Dataset.

This script implements an automated LLM-as-a-judge evaluation pipeline using 
the DeepEval framework for the custom academic dataset. It queries the RAG pipeline 
and then uses a large language model (Gemma 31B) to grade the generated responses.

Features include single-batch processing for HPC stability and a generation 
checkpointing system to allow safe resumption if the script is interrupted.
"""
import os
import json
import random
import time
from typing import List
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd

# 1. Load configuration
load_dotenv(override=True)

# --- DeepEval Global Safety Settings ---
os.environ["DEEPEVAL_DISABLE_CONFIDENT"] = "YES"
os.environ["CONFIDENT_OPEN_BROWSER"] = "0"
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
os.environ["DEEPEVAL_ASYNC_MODE"] = "False"
os.environ["DEEPEVAL_MAX_CONCURRENCY"] = "1"
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "1200"

import deepeval
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric
)
from deepeval.models.base_model import DeepEvalBaseLLM

# Importing search components
from step_04_search import RAGSearch, LLMPipeline
from langchain_core.messages import HumanMessage

# -----------------------------
# Folder for results & Checkpoints
# -----------------------------
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

GENERATION_CHECKPOINT = os.path.join(RESULTS_DIR, "deepeval_generation_checkpoint_academic.json")

PROMPT_TEMPLATE = """You are a reliable AI assistant.
Answer the ‘Question’ based ONLY on the provided ‘Context’.
Context: {context}
Question: {question}
If you do not find any relevant information in the provided context, respond with: "I DONT KNOW"."""


# --------------------------------------------------
# 1. Load Academic JSONL Dataset
# --------------------------------------------------
def load_test_dataset(filepath="Academic_Data_DeepEval.jsonl"):
    """
    Loads the academic evaluation dataset from a local JSONL file.
    
    Args:
        filepath (str): Path to the JSONL dataset. Defaults to "Academic_Data_DeepEval.jsonl".
        
    Returns:
        list of dict: A list of dictionaries mapped to standard keys ('id', 'question', 
                      'reference', 'documents') for DeepEval ingestion.
    """
    print(f"[INFO] Loading academic dataset from local file: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    df = pd.read_json(filepath, lines=True)
    data = []
    for idx, row in df.iterrows():
        # Defensive extraction handling various key naming conventions
        q_id = row.get("Question_ID", row.get("id", f"q_{idx}"))
        question = row.get("Question", row.get("question", ""))
        reference = row.get("Reference", row.get("reference", row.get("response", "")))
        documents = row.get("documents", [])
        data.append({"id": str(q_id), "question": question, "reference": reference, "documents": documents})
        
    print(f"[INFO] Loaded {len(data)} examples from {filepath}")
    return data

# --------------------------------------------------
# 2. Build DeepEval test cases
# --------------------------------------------------
def build_test_cases(rag_search: RAGSearch, dataset):
    """
    Constructs DeepEval LLMTestCase objects from the academic dataset.
    
    Utilizes a JSON-based checkpointing mechanism to save progress after every row.
    
    Args:
        rag_search (RAGSearch): The initialized RAG retrieval and generation pipeline.
        dataset (list): List of dictionaries containing the parsed dataset.
        
    Returns:
        list: A list of fully populated test case dictionaries ready for evaluation.
    """
    test_cases_data = []
    
    if os.path.exists(GENERATION_CHECKPOINT):
        try:
            with open(GENERATION_CHECKPOINT, "r", encoding="utf-8") as f:
                test_cases_data = json.load(f)
            print(f"[INFO] Resuming test case generation. Loaded {len(test_cases_data)} pre-built cases.")
        except Exception as e:
            print(f"[WARNING] Failed to load generation checkpoint: {e}. Rebuilding...")

    start_idx = len(test_cases_data)

    for i, ex in enumerate(tqdm(dataset[start_idx:], desc="Building Academic Semantic RAG test cases", initial=start_idx, total=len(dataset)), start=start_idx):
        try:
            rag_output_raw = rag_search.query_rag(ex["question"], top_k=5)
            rag_output = json.loads(rag_output_raw) if isinstance(rag_output_raw, str) else rag_output_raw
            
            retrieved_docs = rag_output.get("retrieved_docs", [])
            retrieved_texts = [doc.get("page_content", "") for doc in retrieved_docs if doc.get("page_content")]
            answer = rag_output.get("answer", "")

            test_cases_data.append({
                "id": ex["id"],
                "input": ex["question"],
                "expected_output": ex["reference"],
                "actual_output": answer,
                "retrieval_context": retrieved_texts,
                "ground_truth_context": ex.get("documents", [])
            })

            with open(GENERATION_CHECKPOINT, "w", encoding="utf-8") as f:
                json.dump(test_cases_data, f, indent=2, ensure_ascii=False)
            time.sleep(1)
        except Exception as e:
            print(f"\n[SKIP] Row failed: {e}")
            time.sleep(2)
            continue
    return test_cases_data

# --------------------------------------------------
# 3. LLM Judge Model Wrapper
# --------------------------------------------------
class LLMJudge(DeepEvalBaseLLM):
    """
    Custom wrapper class to integrate the deployed LLM pipeline with DeepEval. 
    Inherits from DeepEvalBaseLLM to bypass standard OpenAI dependency requirements.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline
    def load_model(self): return self.pipeline
    def get_model_name(self): return "RedHatAI/gemma-4-31B-it-FP8-Dynamic"
    def generate(self, prompt: str) -> str:
        response = self.pipeline.generate([HumanMessage(content=prompt)])
        return response.generations[0][0].text
    async def a_generate(self, prompt: str) -> str: return self.generate(prompt)

# --------------------------------------------------
# 4. Evaluation Runner Function
# --------------------------------------------------
def run_evaluation(raw_cases, prefix="ContextualPrecision_academic"):
    """
    Executes the DeepEval Contextual Precision evaluation over the generated test cases.
    
    Processes cases sequentially to handle API rate limits and exports results 
    to both JSON and CSV formats for macro-level analysis.
    
    Args:
        raw_cases (list): The list of populated test case dictionaries.
        prefix (str): File prefix for the output logs and summary results.
    """
    judge_pipeline = LLMPipeline(
        model_name="RedHatAI/gemma-4-31B-it-FP8-Dynamic",
        api_key=os.getenv("UOS_API_KEY"),
        api_base=os.getenv("UOS_API_BASE")
    )
    judge = LLMJudge(judge_pipeline)
    threshold = 0.7
    metric = ContextualPrecisionMetric(model=judge, threshold=threshold, include_reason=True)
    
    all_test_results = []
    
    for i, tc in enumerate(raw_cases):
        print(f"\n>>> Processing Eval Item {i+1} of {len(raw_cases)}...")
        test_case = LLMTestCase(
            input=tc["input"],
            expected_output=tc["expected_output"],
            actual_output=tc["actual_output"],
            retrieval_context=tc["retrieval_context"],
            context=tc.get("ground_truth_context", [])
        )
        try:
            res = deepeval.evaluate(test_cases=[test_case], metrics=[metric])
            all_test_results.append(res.test_results[0])
            time.sleep(2)
        except Exception as e:
            print(f"[ERROR] Failed at item {i}: {e}")
            all_test_results.append(None)

    # Prepare outputs
    ordered_results = []
    metric_scores_sum = 0.0
    metric_counts = 0
    total_passed_cases = 0

    for idx, result in enumerate(all_test_results):
        score = 0.0
        success = False
        if result:
            for m in result.metrics_data:
                if m.name in ["Contextual Precision", "ContextualPrecision"]:
                    score = m.score
                    break
            
            # Explicit calculation of success based on threshold
            success = score >= threshold
            if success: total_passed_cases += 1
            metric_scores_sum += score
            metric_counts += 1
        
        ordered_results.append({
            "Question_ID": raw_cases[idx]["id"],
            "Question": raw_cases[idx]["input"],
            "Reference": raw_cases[idx]["expected_output"],
            "Prediction": raw_cases[idx]["actual_output"],
            "ContextualPrecision_Score": score,
            "Success": success,
            "Retrieval_Context": raw_cases[idx]["retrieval_context"]
        })

    # Save CSV and JSON
    final_results_json = os.path.join(RESULTS_DIR, f"{prefix}_results.json")
    final_results_csv = os.path.join(RESULTS_DIR, f"{prefix}_results.csv")
    summary_file = os.path.join(RESULTS_DIR, f"{prefix}_summary.json")

    with open(final_results_json, "w", encoding="utf-8") as f:
        json.dump(ordered_results, f, indent=2, ensure_ascii=False)
    pd.DataFrame(ordered_results).to_csv(final_results_csv, index=False)

    # Save Summary
    total_test_count = len(all_test_results)
    avg_score = (metric_scores_sum / metric_counts) if metric_counts > 0 else 0.0
    
    summary_data = {
        "summary_banner": f"Overall Academic ContextualPrecision Evaluation - ContextualPrecision",
        "hyperparameters": {
            "model": "RedHatAI/gemma-4-31B-it-FP8-Dynamic",
            "temperature": 0,
            "retrieval_k": 5,
            "pipeline": "Semantic RAG",
            "prompt_template": PROMPT_TEMPLATE
        },
        "overall_metrics": {
            "overall_pass_rate": f"{(total_passed_cases / total_test_count) * 100:.2f}%" if total_test_count > 0 else "0.00%",
            "total_test_cases": total_test_count,
            "passed_test_cases": total_passed_cases,
            "failed_test_cases": total_test_count - total_passed_cases,
            "Average_ContextualPrecision": avg_score
        },
        "detailed_results_json": final_results_json,
        "detailed_results_csv": final_results_csv
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Run complete. Summary saved to: {summary_file}")

# --------------------------------------------------
# 5. Main Execution
# --------------------------------------------------
if __name__ == "__main__":
    rag_search = RAGSearch()
    test_data = load_test_dataset("Academic_Data_DeepEval.jsonl") 
    raw_test_cases = build_test_cases(rag_search, test_data)
    
    if len(raw_test_cases) > 3:
        run_evaluation(raw_test_cases[:3], prefix="ContextualPrecision_academic_PREVIEW")
        if input("\nProceed with full evaluation? (y/n): ").lower() != 'y':
            exit()
            
    run_evaluation(raw_cases=raw_test_cases, prefix="ContextualPrecision_academic_results")

