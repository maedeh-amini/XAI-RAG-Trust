"""
This script serves as an empirical evaluation framework, developed and executed by the 
researcher, to test semantic RAG pipelines. It facilitates the researcher's deployment 
of various open-source large language models on a high-performance computing (HPC) 
cluster via vLLM. The primary objective is to benchmark the generative performance 
of these custom models against the results of the primary model detailed in the thesis 
manuscript, enabling a comparative analysis of retrieval and synthesis efficacy across 
different architectures.
"""

#!/bin/bash
#SBATCH --job-name="qwen_rag_test"
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=Qwen3.5_answer.txt
# The line below allows to run this file with 'sbatch search.py'
#SBATCH --wrap="python $0"


import os
import sys
import time
import subprocess
import json
from typing import List
from dotenv import load_dotenv

# Standard LangChain integrations
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Project internal imports
try:
    from vectorstore import FaissVectorStore
except ImportError:
    from src.vectorstore import FaissVectorStore

load_dotenv()

# -----------------------------
# 1. LLM Pipeline Wrapper
# -----------------------------
class LLMPipeline:
    """
    Wrapper for a self-hosted generation model using LangChain's ChatOpenAI.
    
    This class handles API headers and HTTP routing to a custom vLLM endpoint 
    spun up dynamically on an HPC compute node.
    """
    def __init__(self, model_name: str, api_base: str):
        """
        Initializes the LLM connection to the local vLLM server.
        
        Args:
            model_name (str): Identifier of the deployed model (e.g., Qwen3.5-0.8B).
            api_base (str): Base URL of the local API server (e.g., localhost:8000).
        """
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key="EMPTY", 
            openai_api_base=api_base.rstrip('/') + "/", 
            temperature=0.0,
            timeout=60
        )

    def generate(self, messages: List[HumanMessage]):
        """
        Executes generation based on a LangChain message list.
        
        Args:
            messages (List[HumanMessage]): The formatted prompt wrapped in LangChain message objects.
            
        Returns:
            Res: A mock object structure replicating legacy pipeline outputs to 
                 maintain backward compatibility with other scripts.
        """
        response = self.llm.invoke(messages)
        assistant_text = response.content

        # Mock classes to mimic complex return structures expected by legacy code
        class Gen:
            def __init__(self, text): self.text = text
        class Res:
            def __init__(self, text): self.generations = [[Gen(text)]]
        
        return Res(assistant_text)

# -----------------------------
# 2. RAG Search class
# -----------------------------
class RAGSearch:
    """
    Orchestrates the Retrieval-Augmented Generation pipeline for standalone HPC testing.
    
    Loads a pre-built FAISS index, retrieves documents, and prompts the locally 
    deployed vLLM model to generate an answer.
    """
    def __init__(self,
                 persist_dir: str = "faiss_vector_store",
                 embedding_model: str = "bge-m3",
                 llm_model: str = "Qwen3.5-0.8B",
                 api_base: str = "http://localhost:8000/v1"):
        """
        Initializes the vector store and the internal LLM wrapper.
        
        Args:
            persist_dir (str): Directory where the FAISS index is saved.
            embedding_model (str): Name of the embedding model used in the index.
            llm_model (str): Model name matching the vLLM server configuration.
            api_base (str): URL pointing to the active vLLM process.
        """

        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        self.vectorstore.load()
        print(f"[INFO] FAISS store loaded. Retrieval: {embedding_model}")

        self.llm = LLMPipeline(model_name=llm_model, api_base=api_base)

        # Path resolution logic for prompt template
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        prompt_path = os.path.join(root_dir, "prompts", "rag_prompt.txt")
        
        if not os.path.exists(prompt_path):
             prompt_path = os.path.join(current_dir, "prompts", "rag_prompt.txt")

        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
        print(f"[INFO] Loaded prompt template.")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """
        Executes a RAG query and returns the final synthesized answer.
        
        Args:
            query (str): The raw text of the user's question.
            top_k (int): Number of context blocks to retrieve.
            
        Returns:
            str: The raw text response generated by the LLM.
        """
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        
        if not context:
            return "No relevant documents found."

        prompt = self.prompt_template.format(query=query, context=context)
        response = self.llm.generate([HumanMessage(content=prompt)])
        return response.generations[0][0].text

# -----------------------------
# 3. Main Combined Execution
# -----------------------------
def main():
    """
    Main entry point for HPC execution.
    
    This function manages the full lifecycle of a standalone RAG test on an HPC node:
    1. Spawns a background subprocess running the vLLM API server on a GPU.
    2. Polls the server until it is fully initialized and responsive.
    3. Executes a test RAG query against the local server.
    4. Safely terminates the vLLM subprocess to free GPU resources.
    """
    # --- A. SERVER CONFIG ---
    MODEL_PATH = "/home/student/m/mamini/RAG_Project/hf_cache/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
    MODEL_NAME = "Qwen3.5-0.8B"
    PORT = "8000"
    API_BASE = f"http://localhost:{PORT}/v1"

    print(f"[INFO] Starting vLLM background process for: {MODEL_PATH}")
    
    vllm_proc = subprocess.Popen([
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--port", PORT,
        "--served-model-name", MODEL_NAME,
        "--gpu-memory-utilization", "0.8" 
    ])

    try:
        # --- B. WAIT FOR SERVER ---
        print("[INFO] Waiting for Qwen to initialize (this may take ~1 min)...")
        import requests
        connected = False
        for _ in range(30): # Increased wait time for HPC startup
            try:
                if requests.get(f"{API_BASE}/models").status_code == 200:
                    connected = True
                    break
            except:
                pass
            time.sleep(10)
        
        if not connected:
            print("[ERROR] vLLM failed to start. Check logs.")
            return

        # --- C. RUN RAG ---
        rag = RAGSearch(
            persist_dir="faiss_vector_store",
            llm_model=MODEL_NAME,
            api_base=API_BASE
        )

        query = "Which music group has the most members, DC Talk, or Manchester Orchestra?"
        print(f"\n[QUERY] {query}")
        
        answer = rag.search_and_summarize(query, top_k=5)
        print("\n--- FINAL ANSWER ---")
        print(answer)
        print("--------------------")

    finally:
        # --- D. CLEANUP ---
        print("\n[INFO] Terminating vLLM server...")
        vllm_proc.terminate()
        vllm_proc.wait()
        print("[INFO] Done.")

if __name__ == "__main__":
    if 'SBATCH_JOB_ID' not in os.environ:
         print("Please run this script using: sbatch src/search_hpc_models.py")
    else:
         main()