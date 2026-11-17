"""
Knowledge Graph Retrieval and Structured Generation Module for Academic Dataset.

This script executes the Retrieval-Augmented Generation query phase against 
the Neo4j Knowledge Graph using the custom academic dataset. It retrieves relevant 
context blocks and leverages an OpenAI-compatible language model (utilizing structured 
Pydantic outputs) to generate both high-explainability (step-by-step reasoning) 
and low-explainability (direct) answers.

Key features include concurrent query execution, rigorous domain scoping (Cognitive Science), 
robust metadata parsing (chunk IDs and references), and an append-friendly JSON saving 
mechanism to prevent data loss during iterative testing runs.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings 

# =========================================================
# 1. CONFIGURATION & CLIENT INITIALIZATION
# =========================================================
load_dotenv(override=True)

# Model API Configurations (Generation Model)
UOS_API_KEY = os.environ["UOS_API_KEY"]
UOS_API_BASE = os.environ["UOS_API_BASE"]
UOS_MODEL = os.environ.get("UOS_MODEL","RedHatAI/gemma-4-31B-it-FP8-Dynamic")

# Initialize OpenAI client for structured generation (Pydantic parsing)
model_client = OpenAI(api_key=UOS_API_KEY, base_url=UOS_API_BASE)

# =========================================================
# 2. CUSTOM EMBEDDINGS PROVIDER
# =========================================================
class GatewayCompatibleEmbeddings(OpenAIEmbeddings):
    """
    Wrapper to ensure compatibility with server API gateways by 
    removing unsupported parameters that cause LangChain's default implementation to fail.
    """
    def _get_len_safe_embeddings(
        self, texts: List[str], *, engine: str = None, **kwargs
    ) -> List[List[float]]:
        # Strip unsupported parameters before hitting the local endpoint
        kwargs.pop("encoding_format", None)
        kwargs.pop("chunk_size", None)
        
        responses = self.client.create(
            input=texts,
            model=self.model,
            encoding_format=None, 
            **kwargs
        )
        return [d.embedding for d in responses.data]

embeddings = GatewayCompatibleEmbeddings(
    api_key=os.environ["EMBEDDING_MODEL_API_KEY"],
    base_url=os.environ["EMBEDDING_MODEL_API_BASE"],
    model="bge-m3"
)

# =========================================================
# 3. NEO4J RETRIEVAL SETUP
# =========================================================
# Custom Cypher Query: Retrieves the text chunk along with its extracted 
# entities and relationships, enriching the prompt context. It explicitly 
# extracts the custom academic metadata (reference and chunk_id) injected during the build phase.
retrieval_query = """
OPTIONAL MATCH (node)-[:HAS_ENTITY|MENTIONS|RELATED]->(e)
OPTIONAL MATCH (e)-[r]->(neighbor)
WITH node, score, 
     collect(DISTINCT coalesce(e.name, e.id, labels(e)[0])) as entities, 
     collect(DISTINCT type(r) + ' -> ' + coalesce(neighbor.name, neighbor.id, labels(neighbor)[0])) as relations
RETURN 
    node.text + "\\n\\nGRAPH ENRICHMENT:\\nEntities: " + apoc.text.join(entities, ", ") + 
    "\\nRelationships: " + apoc.text.join(relations, ", ") AS text,
    score, 
    {reference: node.reference, chunk_id: node.chunk_id} AS metadata
"""

vector_db = Neo4jVector.from_existing_index(
    embedding=embeddings,
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
    database=os.environ.get("NEO4J_DATABASE", "graphdb02"), 
    index_name="chunk_vector_index", 
    search_type="vector",  
    retrieval_query=retrieval_query,
)

retriever = vector_db.as_retriever(search_kwargs={'k': 5})

# =========================================================
# 4. STRUCTURED OUTPUT SCHEMA & SYSTEM INSTRUCTION
# =========================================================
class ProcessedQuestionItem(BaseModel):
    """
    Pydantic schema enforcing strict JSON output from the generation model, 
    ensuring answers are perfectly formatted for transparency evaluation.
    """
    id: int = Field(description="The original ID of the question.")
    question: str = Field(description="The original question text.")
    low_explainable_response: str = Field(
        description="The final golden answer: a definitive, short, fact-backed response without intermediate reasoning steps."
    )
    high_explainable_response: str = Field(
        description="The detailed reasoning process and step-by-step answer using the context. This must be broken down into clean, discrete atomic statements ('Key-Facts') showing a clear, logical deduction trail based *only* on the context."
    )
    context: str = Field(
        description="A tailored snippet of approx. 256 tokens. A clean slice extracted from the source chunk containing ALL specific sections and sentences required to ground and verify every atomic key-fact."
    )
    reference: str = Field(
        description="The exact PDF file name or source document name from which the context is extracted."
    )
    chunk_number_within_reference: str = Field(
        description="The exact source chunk identification or index within that reference (e.g., 'chunk_4217')."
    )

SYSTEM_INSTRUCTION = """
You are an expert agent specialized in evaluation frameworks for LLM transparency, factual recall, and explainable AI in RAG scenarios.

Your task is to answer the user's question based strictly on the provided context materials and format the response according to the requested schema.

DOMAIN SCOPE: Please note that all of the questions you will be asked are exclusively about the **COGNITIVE SCIENCE** field of study and not any other field.

TRANSPARENCY, CONTEXT INTEGRITY AND EXPLAINABILITY PROTOCOL:
1. low_explainable_response: Provide a minimal, direct, fact-backed final answer. Do not include any intermediate steps, bullet points of reasoning, or commentary. 
2. high_explainable_response: Please think carefully about the knowledge required to answer each question. Then reason the high-quality answer step by step using the provided context. Break down the explanation into explicit, sequential atomic statements ("Key-Facts"). Each step must show how individual facts link together logically to deduce the final answer. Avoid vague language.
3. context: Extract the exact text segment (approx. 256 tokens) from the provided materials needed to fully entail and verify every single fact mentioned in the high-explainable response.
4. reference: Identify the source PDF/document name provided in the context blocks.
5. chunk_number_within_reference: Identify the specific reference chunk index or number provided in the context blocks.
"""

# =========================================================
# 5. DATASET DEFINITION
# =========================================================
# Placeholder for the academic dataset iteration
QA_DATASET = [
   {
        "id": 47,
        "question": "Which bodies are responsible for deciding (a) whether an earlier degree program is relevant and comparable, and (b) an applicant’s language proficiency status when there is doubt?"
    }
]

# =========================================================
# 6. CORE WORKER FUNCTION
# =========================================================
def process_question(qa_item):
    """
    Executes the full retrieval and generation pipeline for a single question.
    
    Args:
        qa_item (Dict[str, Any]): A dictionary containing the 'id' and 'question'.
        
    Returns:
        Dict[str, Any]: A structured dictionary containing the generated answers, 
                        the retrieved context, and extraction metadata.
    """
    question = qa_item["question"]
    item_id = qa_item["id"]
    
    # Phase 1: Retrieve context from Neo4j Vector Store
    try:
        docs = retriever.invoke(question)
        
        context_blocks = []
        for idx, doc in enumerate(docs):
            # Parse reference and chunk_id from the correctly populated Neo4j metadata
            reference = doc.metadata.get("reference", "Unknown Reference")
            chunk_id = doc.metadata.get("chunk_id", f"Chunk {idx+1}")
            context_blocks.append(
                f"--- START OF Chunk {idx+1} [Reference: {reference} | ID: {chunk_id}] ---\n"
                f"{doc.page_content}\n"
                f"--- END OF Chunk {idx+1} ---"
            )
            
        context_string = "\n\n".join(context_blocks)
        
        if not context_string.strip():
            context_string = "No relevant context found in Neo4j graph database."

    except Exception as e:
        return {
            "id": item_id,
            "question": question,
            "low_explainable_answer": "Neo4j retrieval error.",
            "high_explainable_answer": f"Neo4j retrieval error: {str(e)}",
            "context": "",
            "original_chunk_number": "",
            "references": "[]",
            "status": f"Failed (Neo4j: {str(e)})",
        }

    # Phase 2: Generate Dual-Explainability Answers + Grounding Fields
    user_prompt = f"""
QUESTION TO ANSWER:
"{question}"

AVAILABLE CONTEXT POOL (Top Chunks from Neo4j Graph Database):
{context_string}

Analyze the context pool, extract the matching reference data, and generate the structured explainability fields.
Ensure that the 'context' field encapsulates all text segments needed to fully entail every atomic step listed in the high-explainable answer.
"""

    try:
        completion = model_client.beta.chat.completions.parse(
            model=UOS_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_prompt}
            ],
            response_format=ProcessedQuestionItem,
            temperature=0
        )
        parsed_output = completion.choices[0].message.parsed
        
        return {
            "id": item_id, 
            "question": question,
            "low_explainable_answer": parsed_output.low_explainable_response,
            "high_explainable_answer": parsed_output.high_explainable_response,
            "context": parsed_output.context, 
            "original_chunk_number": parsed_output.chunk_number_within_reference,
            "references": parsed_output.reference, 
            "status": "Success",
        }
    except Exception as e:
        return {
            "id": item_id, 
            "question": question, 
            "low_explainable_answer": "Generation failed.",
            "high_explainable_answer": f"Model error: {str(e)}", 
            "context": "",
            "original_chunk_number": "",
            "references": "",
            "status": f"Failed (Model: {str(e)})",
        }

# =========================================================
# 7. EXECUTION RUNNER
# =========================================================
def main():
    """
    Main entry point for batch processing questions.
    
    Executes retrieval and generation concurrently using a ThreadPoolExecutor 
    and appends the results to a local JSON file, preventing data loss across runs.
    """
    print(f"🚀 Starting parallel Graph RAG + Generation pipeline for {len(QA_DATASET)} questions...")
    results = []

    # Using 2 workers to balance processing speed without overwhelming the server API endpoints
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_question, item): item for item in QA_DATASET}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing pipeline"):
            results.append(future.result())

    # --- APPEND LOGIC START ---
    # This logic allows for safe script restarts by merging new runs with existing data
    output_file = "./graph_explainable_results.json"
    existing_data = []

    # 1. Read existing data if the file is already there
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
        except Exception as e:
            print(f"⚠️ Could not read existing JSON ({e}). Starting a new file.")
            existing_data = []

    # 2. Merge old data with new results (using a dictionary to prevent duplicate IDs)
    merged_results_dict = {item["id"]: item for item in existing_data}
    
    for new_result in results:
        merged_results_dict[new_result["id"]] = new_result  # Overwrites if ID already exists
        
    # Convert back to a list and sort by ID for consistent output formatting
    final_results = list(merged_results_dict.values())
    final_results.sort(key=lambda x: x["id"])

    # 3. Write the combined data back to the file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    # --- APPEND LOGIC END ---

    print("\n✅ Execution completed successfully!")
    print(f"- Saved {len(results)} new/updated questions.")
    print(f"- Total questions now in database: {len(final_results)}")
    print(f"- Saved JSON dataset to '{output_file}'")


if __name__ == "__main__":
    main()

