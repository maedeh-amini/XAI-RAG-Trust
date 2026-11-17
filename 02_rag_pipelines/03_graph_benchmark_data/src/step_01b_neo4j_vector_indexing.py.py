"""
Neo4j Vector Index Setup Utility.

This script initializes the vector search capabilities within the Neo4j Knowledge Graph. 
It must be executed immediately after the database build phase (`step_01_graph_builder.py`).

It creates two distinct vector indexes:
1. `entity_vector_index`: Applied to `__Entity__` nodes, used for structural graph traversal.
2. `chunk_vector_index`: Applied to `Chunk` nodes, used for standard RAG semantic retrieval.

Double Sanity Check:
To manually verify the index state outside of this script, run the Cypher 
command `SHOW INDEXES` in the Neo4j Desktop console. Ensure that the 'type' column 
reads as 'Vector' (not a standard 'Lookup' index).

Reference Data:
For an example of a successful verification output, please see the sample JSON file 
located under the Miscellaneous folder: `neo4j_query_table_data_index 2026-5-9.json`.
"""

import os
import time
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables (override=True ensures local .env takes precedence)
load_dotenv(override=True)

# =========================================================
# Configuration & Driver Initialization
# =========================================================
uri = os.environ["NEO4J_URI"]
user = os.environ["NEO4J_USERNAME"]
password = os.environ["NEO4J_PASSWORD"]
# Target the specific database, defaulting to 'neo4j' if unspecified
db_name = os.environ.get("NEO4J_DATABASE", "neo4j") 

driver = GraphDatabase.driver(uri, auth=(user, password))

def setup_indexes():
    """
    Executes Cypher queries to construct vector indexes in the database and 
    polls the database to verify they have successfully come online.
    
    The dimensionality is explicitly set to 1024 to match the BGE-M3 embedding 
    model used throughout the graph pipeline.
    """
    # 1. Index for extracted Graph Entities (used for structural reasoning)
    entity_index_query = """
    CREATE VECTOR INDEX entity_vector_index IF NOT EXISTS
    FOR (n:__Entity__)
    ON (n.embedding)
    OPTIONS {indexConfig: {
      `vector.dimensions`: 1024,
      `vector.similarity_function`: 'cosine'
    }}
    """
    
    # 2. Index for Raw Text Chunks (used for standard RAG retrieval)
    chunk_index_query = """
    CREATE VECTOR INDEX chunk_vector_index IF NOT EXISTS
    FOR (n:Chunk)
    ON (n.embedding)
    OPTIONS {indexConfig: {
      `vector.dimensions`: 1024,
      `vector.similarity_function`: 'cosine'
    }}
    """
    
    print(f"[INFO] Connecting to {uri} | Database: {db_name}")
    
    with driver.session(database=db_name) as session:
        try:
            # Create Entity Index
            print(f"Creating 'entity_vector_index' on {db_name}...")
            session.run(entity_index_query)
            
            # Create Chunk Index
            print(f"Creating 'chunk_vector_index' on {db_name}...")
            session.run(chunk_index_query)
            
            # Verification Loop: Vector indexes build asynchronously in Neo4j.
            # This loop polls the database state until both indexes register as 'ONLINE'.
            print("Verifying index states...")
            for i in range(5):
                time.sleep(2)

                # Sanity check: Ensure index type is 'Vector', not 'Lookup'
                result = session.run("SHOW VECTOR INDEXES YIELD name, labelsOrTypes, state")
                indexes = list(result)
                
                online_count = sum(1 for r in indexes if r['state'] == 'ONLINE')
                
                for rec in indexes:
                    print(f"- Index: {rec['name']} | Label: {rec['labelsOrTypes']} | Status: {rec['state']}")
                
                if online_count >= 2:
                    print(f"\n✅ SUCCESS: Both Graph and Chunk indexes are ONLINE in '{db_name}'.")
                    return
                
                print(f"Waiting for indexes to populate (Attempt {i+1}/5)...")
                
        except Exception as e:
            print(f"❌ DATABASE ERROR: {e}")
        finally:
            driver.close()

if __name__ == "__main__":
    setup_indexes()







