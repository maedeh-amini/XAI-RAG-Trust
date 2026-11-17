"""
Embedding Generation Module for Semantic RAG Benchmark.

This script handles the transformation of parsed textual data into dense vector 
embeddings. It defines the `EmbeddingPipeline` class, which interfaces with a 
custom-deployed embedding model (e.g., BGE-M3) via an OpenAI-compatible API endpoint using LangChain.

"""


import os
import numpy as np
from typing import List
from dotenv import load_dotenv

# Standard LangChain OpenAI integration
from langchain_openai import OpenAIEmbeddings
from step_02_data_loader import load_source_data
# from data_loader import load_source_data    

# Load environment variables
load_dotenv(override=True)

EMB_KEY = os.environ.get("EMBEDDING_MODEL_API_KEY")
EMB_BASE = os.environ.get("EMBEDDING_MODEL_API_BASE")

class EmbeddingPipeline:
    """
    A pipeline for generating vector embeddings from textual documents.
    
    Utilizes LangChain's OpenAIEmbeddings to connect to an external, 
    OpenAI-compatible embedding server.
    """
    def __init__(self, model_name: str = "test/bge-m3"):
        """
        Initializes the embedding pipeline and configuring the LangChain embedder.
        
        Args:
            model_name (str): The specific identifier for the deployed model 
                              expected by the API endpoint. Defaults to "test/bge-m3".
        """
        self.model_name = model_name

        # Configure the LangChain embedder
        self.embedder = OpenAIEmbeddings(
            model=self.model_name,
            openai_api_key=EMB_KEY,
            openai_api_base=EMB_BASE,
            check_embedding_ctx_length=False,
            # Set encoding_format to None to stop 
            # LangChain/OpenAI from defaulting to 'base64'
            model_kwargs={"encoding_format": None},
            # Set the batch size to 32 to match the API's limit
            chunk_size=32 
        )
        print(f"[INFO] Initialized LangChain API pipeline: {model_name}")

    def create_block_chunks(self, documents: List[dict]) -> List[str]:
        """
        Extracts text from the document dictionaries for embedding.
        
        Chunking Strategy: No dynamic chunking or splitting is applied. 
        Each entire document string (block) is treated as a single chunk.
        
        Args:
            documents (List[dict]): The processed data from the data loader, 
                                    where each dict contains a 'text' key.
                                    
        Returns:
            List[str]: A list of raw text strings ready for embedding.
        """
        # Only process rows that actually contain text
        chunks = [doc["text"] for doc in documents if doc.get("text")]
        
        print(f"[INFO] Created {len(chunks)} block chunks (No chunking applied).")
        return chunks

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """
        Generates dense vector embeddings for the provided text chunks.
        
        Args:
            chunks (List[str]): A list of text strings to be embedded.
            
        Returns:
            np.ndarray: A 2D NumPy array containing the generated embeddings, 
                        where shape is (num_chunks, embedding_dimensions).
        """
        print(f"[INFO] Generating embeddings for {len(chunks)} chunks via LangChain API...")
        
        # .embed_documents() is the standard LangChain method for bulk text
        raw_embeddings = self.embedder.embed_documents(chunks)
        
        embeddings_array = np.array(raw_embeddings)
        print(f"[INFO] Embeddings shape: {embeddings_array.shape}")
        return embeddings_array


# Example usage
if __name__ == "__main__":
    # 1. Load data using step_02_data_loader script
    docs = load_source_data(subset_fraction=1.0)

    # 2. Initialize Pipeline
    emb_pipe = EmbeddingPipeline(model_name="test/bge-m3")

    # 3. Create Chunks (Block-level)
    chunks = emb_pipe.create_block_chunks(docs)

    # 4. Generate Embeddings
    # This now uses the LangChain embedder under the hood
    embeddings = emb_pipe.embed_chunks(chunks)

    if len(embeddings) > 0:
        print("[INFO] Example embedding snippet (first 5 dims):", embeddings[0][:5])








