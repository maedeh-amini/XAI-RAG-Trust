"""
Unified Knowledge Graph Builder for Graph RAG Benchmark.

This script consolidates the data loading, entity extraction, embedding, and 
graph construction phases into a single execution pipeline. It fetches the 
HuggingFace benchmark dataset, uses a deployed LLM to extract entities and 
relationships, embeds the entities, and constructs a Knowledge Graph in Neo4j.

Critically, it includes a custom Cypher injection step to manually map original 
HuggingFace dataset IDs (`hf_id`) to the generated graph nodes, ensuring traceability 
for downstream DeepEval benchmarking.
"""

# =========================================================
# Graph Builder
# =========================================================

import os
import random
import asyncio
from tqdm import tqdm
from datasets import load_dataset
from neo4j import GraphDatabase
from dotenv import load_dotenv

from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings 
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline, OnError
from neo4j_graphrag.indexes import create_vector_index

# =========================================================
# 1. Setup & Environment
# =========================================================
load_dotenv()

UOS_API_KEY = os.environ["UOS_API_KEY"]
UOS_API_BASE = os.environ["UOS_API_BASE"]
EMB_KEY = os.environ["EMBEDDING_MODEL_API_KEY"]
EMB_BASE = os.environ["EMBEDDING_MODEL_API_BASE"]
DB_NAME = os.environ.get("NEO4J_DATABASE", "graphdb")

class GatewayCompatibleEmbeddings(OpenAIEmbeddings):
    """
    Custom override for Neo4j's OpenAIEmbeddings wrapper.
    
    Why: Standard integrations often default to base64 encoding or pass unsupported 
    parameters to local server endpoints. This explicitly forces 
    `encoding_format=None` to ensure compatibility with the researcher's deployed models.
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
    Manages the end-to-end pipeline for dataset ingestion and Knowledge Graph construction.
    """
    def __init__(self, driver, subset_fraction=1.0):
        """
        Initializes the Orchestrator with database drivers and server LLM connections.
        
        Args:
            driver: The active Neo4j GraphDatabase driver instance.
            subset_fraction (float): The fraction of the HF dataset to load (0.0 to 1.0).
        """
        self.driver = driver
        self.subset_fraction = subset_fraction
        self.processed_data = [] # List of dicts: {'id': ..., 'text': ...}

        # Initialize the generator LLM for entity and relationship extraction
        self.llm = OpenAILLM(
            model_name="RedHatAI/gemma-4-31B-it-FP8-Dynamic", 
            api_key=UOS_API_KEY,
            base_url=UOS_API_BASE,
            model_params={"temperature": 0}
        )

        # Initialize the custom embedder for node vectorization
        self.embedder = GatewayCompatibleEmbeddings(model="bge-m3")
        self.embedder.client.api_key = EMB_KEY
        self.embedder.client.base_url = EMB_BASE

    def load_source_data(self):
        """
        Loads and downsamples the HuggingFace dataset.
        
        Extracts the required text and preserves the original benchmark IDs 
        to allow for 1:1 evaluation mapping later.
        """
        dataset = load_dataset("galileo-ai/ragbench", "hotpotqa", split="test")
        subset_size = max(1, int(len(dataset) * self.subset_fraction))
        indices = random.sample(range(len(dataset)), subset_size)
        
        for idx in tqdm(indices, desc="Fetching HF Data"):
            row = dataset[idx]
            
            # CAPTURE THE ID HERE
            hf_id = str(row["id"]) 
            
            # Capture and clean the documents text
            docs = row["documents"]
            content = " ".join(docs) if isinstance(docs, list) else str(docs)
            
            self.processed_data.append({
                "hf_id": hf_id,
                "text": content
            })
        print(f"[INFO] Loaded {len(self.processed_data)} items with original IDs.")

    async def execute_pipeline(self):
        """
        Executes the graph building process asynchronously.
        
        Steps:
        1. Ensures a vector index exists on entity embeddings for similarity search.
        2. Iterates through the processed dataset.
        3. Uses `SimpleKGPipeline` to extract graph structures via LLM.
        4. Executes a custom Cypher query to forcefully map the `hf_id` to the Chunk node.
        """
        # 1. Verification of Index
        try:
            create_vector_index(
                self.driver,
                name="entity_vector_index",
                label="__Entity__",
                embedding_property="embedding",
                dimensions=1024, # BGE-M3 standard dimensionality
                similarity_fn="cosine"
            )
        except Exception:
            # Index already exists or configuration overlap
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

        print(f"[INFO] Building Graph & Injecting IDs into {DB_NAME}...")
        
        # We process one-by-one to maintain the ID mapping
        for item in tqdm(self.processed_data, desc="Processing Rows"):
            try:
                # A. Run Extraction
                await kg_pipeline.run_async(text=item['text'])
                
                # B. Immediate ID Injection
                # SimpleKGPipeline creates Chunk nodes, but doesn't natively accept custom metadata IDs.
                # We find the node by its exact text and set the hf_id property manually.
                injection_query = """
                MATCH (c:Chunk) 
                WHERE c.text = $text AND c.hf_id IS NULL
                SET c.hf_id = $hf_id
                """
                
                with self.driver.session(database=DB_NAME) as session:
                    session.run(injection_query, text=item['text'], hf_id=item['hf_id'])
                
                # Gateway cooldown to prevent server API rate limiting
                await asyncio.sleep(1.5) 
                
            except Exception as e:
                print(f"[ERROR] Row {item['hf_id']} failed: {e}")

# =========================================================
# 3. Execution
# =========================================================
if __name__ == "__main__":
    # IMPORTANT: Clear database before re-running to avoid ID mismatches
    # Command: MATCH (n) DETACH DELETE n
    
    orchestrator = KnowledgeGraphOrchestrator(neo4j_driver, subset_fraction=1.0
    )
    orchestrator.load_source_data()
    asyncio.run(orchestrator.execute_pipeline())
    print("[SUCCESS] Builder Pipeline Finished with ID Persistence.")







