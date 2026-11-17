"""
Unified Knowledge Graph Builder for Academic Dataset.

This script consolidates the data ingestion, entity extraction, embedding, and 
Knowledge Graph construction phases for the custom academic dataset. It reads 
document chunks from a JSON file, utilizes a locally deployed LLM (e.g., Gemma 31B) 
to extract entities and relationships, and builds a graph in Neo4j.

Crucially, it executes a custom Cypher query after each chunk is processed to 
forcefully map the original JSON `chunk_id` and `reference` directly to the 
Neo4j Chunk nodes, ensuring full traceability during downstream evaluation.
"""

# =========================================================
# Graph Builder
# =========================================================

import os
import json
import asyncio
from tqdm import tqdm
from neo4j import GraphDatabase
from dotenv import load_dotenv

from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings 
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline, OnError
from neo4j_graphrag.indexes import create_vector_index

# =========================================================
# 1. Setup & Environment
# =========================================================
# load_dotenv()
load_dotenv(override=True)

UOS_API_KEY = os.environ["UOS_API_KEY"]
UOS_API_BASE = os.environ["UOS_API_BASE"]
EMB_KEY = os.environ["EMBEDDING_MODEL_API_KEY"]
EMB_BASE = os.environ["EMBEDDING_MODEL_API_BASE"]
DB_NAME = os.environ.get("NEO4J_DATABASE", "neo4j")

class GatewayCompatibleEmbeddings(OpenAIEmbeddings):
    """
    Custom override for Neo4j's OpenAIEmbeddings wrapper.
    
    Ensures compatibility with local server API gateways by 
    explicitly passing `encoding_format=None` and preventing default base64 encoding.
    """
    def embed_query(self, text):
        return self.client.embeddings.create(
            input=text, 
            model=self.model,
            encoding_format=None
        ).data[0].embedding

    def embed_nodes(self, texts):
        response = self.client.embeddings.create(
            input=texts, 
            model=self.model,
            encoding_format=None
        )
        return [d.embedding for d in response.data]

neo4j_driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
)

# =========================================================
# 2. Knowledge Graph Orchestrator
# =========================================================
class KnowledgeGraphOrchestrator:
    """
    Manages the end-to-end pipeline for parsing academic JSON data and 
    constructing the Neo4j Knowledge Graph.
    """
    def __init__(self, driver):
        """
        Initializes the Orchestrator with database drivers and server model connections.
        
        Args:
            driver: The active Neo4j GraphDatabase driver instance.
        """
        self.driver = driver
        self.processed_data = [] # List of dicts: {'chunk_id': ..., 'text': ..., 'reference': ...}

        # Initialize the generator LLM for entity and relationship extraction
        self.llm = OpenAILLM(
            model_name="RedHatAI/gemma-4-31B-it-FP8-Dynamic", 
            api_key=UOS_API_KEY,
            base_url=UOS_API_BASE,
            model_params={"temperature": 0}
        )

        # Initialize the custom embedder for node vectorization
        self.embedder = GatewayCompatibleEmbeddings(model="bge-m3") #model="test/bge-m3"
        self.embedder.client.api_key = EMB_KEY
        self.embedder.client.base_url = EMB_BASE

    def load_source_data(self, json_filepath):
        """
        Reads and parses the raw JSON dataset containing academic document chunks.
        
        Dynamically extracts keys starting with "chunk_" and maps them alongside 
        their document references for injection into the graph.
        
        Args:
            json_filepath (str): Path to the JSON dataset file.
        """
        # Load the JSON data
        with open(json_filepath, 'r', encoding='utf-8') as file:
            dataset = json.load(file)

        for item in tqdm(dataset, desc="Reading JSON Data"):
            # Dynamically find the key that starts with "chunk_" 
            chunk_key = next((k for k in item.keys() if k.startswith("chunk_")), None)
            reference = item.get("reference", "Unknown Reference")

            if chunk_key:
                text_content = item[chunk_key]
                self.processed_data.append({
                    "chunk_id": chunk_key,
                    "text": text_content,
                    "reference": reference
                })
                
        print(f"[INFO] Loaded {len(self.processed_data)} chunks from JSON.")

    async def execute_pipeline(self):
        """
        Executes the asynchronous graph building process.
        
        Steps:
        1. Ensures a vector index exists for entity embeddings.
        2. Iterates through the processed dataset.
        3. Uses `SimpleKGPipeline` to extract graph structures via the LLM.
        4. Executes a Cypher query to forcefully map the `chunk_id` and `reference` 
           metadata directly onto the generated Chunk nodes.
        """
        # 1. Verification of Index
        try:
            create_vector_index(
                self.driver,
                name="entity_vector_index",
                label="__Entity__",
                embedding_property="embedding",
                dimensions=1024, # Dimensionality for BGE-M3
                similarity_fn="cosine"
            )
        except Exception:
            # Index already exists or overlap
            pass

        # 2. Initialize Pipeline
        kg_pipeline = SimpleKGPipeline(
            llm=self.llm,
            driver=self.driver,
            neo4j_database=DB_NAME,
            embedder=self.embedder,
            from_pdf=False,
            schema="FREE", 
            on_error=OnError.IGNORE
        )

        print(f"[INFO] Building Graph & Injecting Attributes into {DB_NAME}...")
        
        # Process one-by-one to maintain attribute mapping
        for item in tqdm(self.processed_data, desc="Processing Rows"):
            try:
                # A. Run Extraction & Embedding (Embedding is handled by the pipeline/embedder)
                await kg_pipeline.run_async(text=item['text'])
                
                
                # B. Immediate Attribute Injection
                # SimpleKGPipeline creates Chunk nodes natively but lacks support for custom dict keys.
                # We find the node by text matching and set the chunk_id and reference properties manually.
                injection_query = """
                MATCH (c:Chunk) 
                WHERE c.text = $text AND c.chunk_id IS NULL
                SET c.chunk_id = $chunk_id,
                    c.reference = $reference
                """
                
                with self.driver.session(database=DB_NAME) as session:
                    session.run(
                        injection_query, 
                        text=item['text'], 
                        chunk_id=item['chunk_id'],
                        reference=item['reference']
                    )
                
                # Gateway cooldown to prevent server API rate limiting
                await asyncio.sleep(1.5) 
                
            except Exception as e:
                print(f"[ERROR] Chunk {item['chunk_id']} failed: {e}")

# =========================================================
# 3. Execution
# =========================================================
if __name__ == "__main__":
    # IMPORTANT: Clear database before re-running to avoid mismatches
    # Command: MATCH (n) DETACH DELETE n
    
    orchestrator = KnowledgeGraphOrchestrator(neo4j_driver)
    
    # Actual path of JSON file
    orchestrator.load_source_data("8.json") 
    
    asyncio.run(orchestrator.execute_pipeline())
    print("[SUCCESS] Builder Pipeline Finished with Custom JSON Attributes.")
