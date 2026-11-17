"""
Data Loader Module for the Academic Dataset Semantic RAG Pipeline.

This script is responsible for ingesting the custom academic dataset (JSON format) 
for the semantic RAG pipeline. It parses the raw data, dynamically identifies text chunks, 
and formats them into a standardized dictionary structure required for downstream vector store embedding. 
It includes an optional subsetting feature to load data fractions for rapid prototyping on 
the HPC cluster.
"""

import json
import random
from tqdm import tqdm

def load_source_data(file_path: str = "full_chunks.json", subset_fraction: float = 1.0):
    """
    Loads data from a local JSON file, randomly samples a subset, and processes 
    it into a standardized list of dictionaries for vector store ingestion.
    
    Args:
        file_path (str): The local path to the JSON dataset file. Defaults to "full_chunks.json".
        subset_fraction (float): The fraction of the dataset to load (0.0 to 1.0).
                                 Defaults to 1.0 (100% of the dataset).
                                 
    Returns:
        list of dict: A list where each element is a dictionary containing:
                      - 'chunk_id' (str): The original key identifier (e.g., "chunk_1").
                      - 'text' (str): The extracted text content of the chunk.
                      - 'reference' (str): The ground-truth reference or source identifier.
    """
    # 1. Load the JSON dataset
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {file_path}. Please check the path.")
        return []
    
    # 2. Calculate subset size and generate random indices for unbiased downsampling
    subset_size = max(1, int(len(dataset) * subset_fraction))
    indices = random.sample(range(len(dataset)), subset_size)
    
    processed_data = []
    
    # 3. Process sampled rows
    for idx in tqdm(indices, desc="Loading JSON Data"):
        row = dataset[idx]
        
        # Dynamically find the key containing the chunk text.
        # This is necessary because the dataset keys vary per row (e.g., "chunk_1", "chunk_2").
        chunk_key = next((key for key in row.keys() if key.startswith("chunk_")), None)
        
        if not chunk_key:
            continue
            
        content = str(row[chunk_key])
        reference = str(row.get("reference", ""))
        
        # Structure the processed data into a unified format for the vector store
        processed_data.append({
            "chunk_id": chunk_key,
            "text": content,
            "reference": reference
        })
    
    return processed_data

# Example usage / Manual Testing Block
if __name__ == "__main__":
    # Pass a float for the fraction of data (e.g., 0.05 for 5%)
    docs = load_source_data(file_path="full_chunks.json", subset_fraction=1.0)
    
    print(f"\n[INFO] Loaded {len(docs)} documents.")
    if docs:
        print("Example document ID:", docs[0]["chunk_id"])
        print("Example reference:", docs[0]["reference"])
        # Replacing newlines with spaces for a cleaner console print
        print("Example text snippet:", docs[0]["text"][:100].replace("\n", " "), "...")

