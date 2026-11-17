"""
Semantic RAG Search and Generation Module.

This script implements the core RAG logic. 
It connects the FAISS vector database (retriever) with a deployed language 
model (generator) hosted on the server via an OpenAI-compatible API. 
It provides both standard string outputs and structured JSON outputs required 
for downstream evaluation using the DeepEval framework.
"""

import os
import json
from typing import List
from dotenv import load_dotenv

# Standard LangChain integrations
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Project internal imports
from step_02_data_loader import load_source_data
from step_04_vectorstore import FaissVectorStore

load_dotenv(override=True)

# -----------------------------
# 1. Custom LLM wrapper
# -----------------------------
class LLMPipeline:
    """
    Wrapper for a self-hosted generation model using LangChain's ChatOpenAI.
    
    This class handles API headers, authentication, and HTTP routing to custom 
    LLM endpoints on the server, abstracting away the raw request logic.
    """
    def __init__(self, model_name: str, api_key: str, api_base: str, temperature: float = 0.0):
        """
        Initializes the LLM connection.
        
        Args:
            model_name (str): Identifier of the deployed model (e.g., RedHatAI/gemma-4-31B).
            api_key (str): Authentication token for the cluster endpoint.
            api_base (str): Base URL of the API server.
            temperature (float): Generation temperature for response randomness.
        """
        # We use ChatOpenAI which is more robust against 403 errors than raw requests
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=api_base.rstrip('/') + "/", # Ensures standard pathing
            temperature=temperature
        )

    def generate(self, messages: List[HumanMessage]):
        """
        Executes generation based on a LangChain message list.
        
        Args:
            messages (List[HumanMessage]): The formatted prompt wrapped in LangChain message objects.
            
        Returns:
            Res: A mock object containing the 'generations' attribute to maintain 
                 backward compatibility with legacy evaluation scripts.
        """
        response = self.llm.invoke(messages)
        assistant_text = response.content

        # Keeping the existing class-based return structure to avoid breaking search_and_summarize
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
    Orchestrates the Retrieval-Augmented Generation pipeline.
    
    Loads the FAISS index, retrieves relevant document chunks based on semantic 
    similarity, injects them into a prompt template, and calls the LLM pipeline 
    to generate an answer.
    """
    def __init__(self,
                 persist_dir: str = "faiss_vector_store",
                 embedding_model: str = "test/bge-m3",
                 llm_model: str = "RedHatAI/gemma-4-31B-it-FP8-Dynamic",
                 api_key_env: str = "UOS_API_KEY",
                 api_base_env: str = "UOS_API_BASE"):
        """
        Initializes the vector store and the LLM pipeline.
        
        Args:
            persist_dir (str): Directory where the FAISS index is saved.
            embedding_model (str): Name of the embedding model used for the index.
            llm_model (str): Name of the generation model deployed on the HPC.
            api_key_env (str): Environment variable name containing the API key.
            api_base_env (str): Environment variable name containing the API URL.
        """

        # Load pre-built FAISS vector store
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        self.vectorstore.load()
        print(f"[INFO] FAISS vector store loaded from '{persist_dir}' using {embedding_model}")

        self.api_key = os.getenv(api_key_env)
        self.api_base = os.getenv(api_base_env)
        
        if not self.api_key or not self.api_base:
            raise ValueError(f"API key or base URL not found in environment.")

        # Initialize the refactored LLM pipeline
        self.llm = LLMPipeline(
            model_name=llm_model,
            api_key=self.api_key,
            api_base=self.api_base
        )
        print(f"[INFO] LLM initialized using model: {llm_model}")

        # Path logic: prompt is in the folder above 'src'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        prompt_path = os.path.join(root_dir, "prompts", "semantic_rag_prompt.txt")
        
        if not os.path.exists(prompt_path):
             raise FileNotFoundError(f"Could not find prompt template at: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
        print(f"[INFO] Loaded prompt template.")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """
        Executes a standard RAG workflow returning a simple string response.
        
        Args:
            query (str): The user's question.
            top_k (int): Number of context blocks to retrieve from FAISS.
            
        Returns:
            str: The LLM's generated answer.
        """
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        
        if not context:
            return "No relevant documents found."

        prompt = self.prompt_template.format(query=query, context=context)
        response = self.llm.generate([HumanMessage(content=prompt)])
        return response.generations[0][0].text

    def query_rag(self, question: str, top_k: int = 3) -> str:
        """
        Executes a RAG workflow returning a structured JSON response formatted 
        specifically for DeepEval testing.
        
        Args:
            question (str): The user's question.
            top_k (int): Number of context blocks to retrieve from FAISS.
            
        Returns:
            str: A JSON string containing both the generated 'answer' and 
                 the raw 'retrieved_docs' used to generate it.
        """
        results = self.vectorstore.query(question, top_k=top_k)
        retrieved_docs = []
        context_chunks = []

        for r in results:
            meta = r.get("metadata", {})
            text = meta.get("text", "")
            if text:
                retrieved_docs.append({"page_content": text})
                context_chunks.append(text)

        context = "\n\n".join(context_chunks)
        if not context:
            answer = "No relevant documents found."
        else:
            prompt = self.prompt_template.format(query=question, context=context)
            response = self.llm.generate([HumanMessage(content=prompt)])
            answer = response.generations[0][0].text

        return json.dumps({
            "answer": answer,
            "retrieved_docs": retrieved_docs
        })

# -----------------------------
# Execution
# -----------------------------
if __name__ == "__main__":
    # Point to the directory created by step_04_vectorstore.py
    rag_search = RAGSearch(persist_dir="faiss_vector_store")

    query = "Which music group has the most members, DC Talk, or Manchester Orchestra?"

    print("\n--- Running Standard RAG ---")
    summary = rag_search.search_and_summarize(query, top_k=5)
    print("Summary:", summary)

    print("\n--- Running DeepEval-compatible Query ---")
    ragas_output = rag_search.query_rag(query, top_k=1)
    print("Output:", ragas_output)


