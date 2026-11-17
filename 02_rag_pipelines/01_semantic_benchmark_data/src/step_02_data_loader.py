"""
Data Loader Module for Semantic RAG Benchmark.

This script is responsible for ingesting the HotpotQA subset of the Galileo-AI 
RAGBench dataset from HuggingFace. It parses and formats the raw data into a 
standardized dictionary structure required for downstream vector store embedding. 
It includes an optional subsetting feature to load data fractions for rapid 
prototyping on the HPC cluster.

"""


import random
from tqdm import tqdm
from datasets import load_dataset

def load_source_data(subset_fraction: float = 1.0):
    """
    Loads data from HuggingFace, randomly samples a subset, and processes 
    it into a standardized list of dictionaries for vector store ingestion.
    
    Args:
        subset_fraction (float): The fraction of the dataset to load (0.0 to 1.0).
                                 Defaults to 1.0 (100% of the dataset).
                                 
    Returns:
        list of dict: A list where each element is a dictionary containing:
                      - 'hf_id' (str): The unique identifier from HuggingFace.
                      - 'text' (str): The concatenated document text.
    """
    # 1. Load the dataset
    dataset = load_dataset("galileo-ai/ragbench", "hotpotqa", split="test")
    
    # 2. Calculate subset size
    subset_size = max(1, int(len(dataset) * subset_fraction))
    indices = random.sample(range(len(dataset)), subset_size)
    
    processed_data = []
    
    # 3. Process rows
    for idx in tqdm(indices, desc="Fetching HF Data"):
        row = dataset[idx]
        
        # Capture the ID
        hf_id = str(row.get("id", idx)) 
        
        # Capture and clean the documents text
        docs = row.get("documents", [])
        content = " ".join(docs) if isinstance(docs, list) else str(docs)
        
        processed_data.append({
            "hf_id": hf_id,
            "text": content
        })
    
    return processed_data

# Example usage/ Manual Testing Block
if __name__ == "__main__":
    # Test loading a 5% fraction of the data for quick verification
    docs = load_source_data(subset_fraction=1.0)
    
    print(f"\n[INFO] Loaded {len(docs)} documents.")
    if docs:
        print("Example document ID:", docs[0]["hf_id"])
        print("Example text snippet:", docs[0]["text"][:100], "...") 





        



