"""
Diagnostic Utility: Neo4j Knowledge Graph Inspector.

This standalone script is used to verify the contents of the Neo4j graph database 
after the graph building phase. Because the pipeline only ingests a random subset 
(e.g., 5%) of the HuggingFace RagBench dataset, it is critical to know exactly 
which documents are stored in the graph. 

By extracting the `example_id`s of the ingested documents, the researcher can 
ensure that downstream RAG queries are valid (i.e., querying for answers that 
actually exist within the constructed graph).
"""

from neo4j import GraphDatabase
from typing import List
import os
from dotenv import load_dotenv


def list_graph_documents(uri: str, username: str, password: str, database: str) -> List[dict]:
    """
    Retrieves and lists all document/chunk nodes currently stored in the Neo4j graph.
    
    Args:
        uri (str): The connection URI for the Neo4j instance.
        username (str): Database authentication username.
        password (str): Database authentication password.
        database (str): The specific Neo4j database name to query.
        
    Returns:
        List[dict]: A list of dictionaries containing 'example_id', 'title', 
                    and 'content' for each node found in the graph.
    """
    driver = GraphDatabase.driver(uri, auth=(username, password))

    # The RagBench dataset 'id' column is mapped to 'id' or 'example_id' in Neo4j.
    # SimpleKGPipeline uses 'text' for the body content, while other loaders might use 'page_content'.
    # The `coalesce` function acts as a fallback to handle these schema variations dynamically.
    query = """
    MATCH (n)
    WHERE n.text IS NOT NULL OR n.page_content IS NOT NULL
    RETURN coalesce(n.id, n.example_id) AS example_id,
           coalesce(n.title, 'N/A') AS title,
           coalesce(n.text, n.page_content) AS content
    ORDER BY example_id
    """

    documents = []

    with driver.session(database=database) as session:
        results = session.run(query)

        for record in results:
            documents.append({
                "example_id": record["example_id"],
                "title": record["title"],
                "content": record["content"]
            })

    driver.close()

    print(f"\n[INFO] Total documents in graph: {len(documents)}\n")

    # Output formatting: Print the first 150 characters of the content for quick visual inspection
    for i, doc in enumerate(documents):
        print(f"--- Document {i} ---")
        print("Dataset ID (example_id):", doc["example_id"])
        print("Title:", doc["title"])
        print("Content Preview:", doc["content"][:150].replace("\n", " "), "...")
        print()

    return documents


def list_graph_documents_by_example_ids(
    uri: str,
    username: str,
    password: str,
    database: str,
    example_ids: List[str]
) -> List[dict]:
    """
    Retrieves specific document nodes based on a provided list of original dataset IDs.
    
    Useful for targeted verification when tracing a specific query failure 
    during DeepEval testing.
    
    Args:
        uri (str): The connection URI for the Neo4j instance.
        username (str): Database authentication username.
        password (str): Database authentication password.
        database (str): The specific Neo4j database name to query.
        example_ids (List[str]): A list of target IDs to search for in the graph.
        
    Returns:
        List[dict]: A list of dictionaries containing metadata for the matched nodes.
    """

    driver = GraphDatabase.driver(uri, auth=(username, password))

    # Safely filters the graph against the provided list of IDs
    query = """
    MATCH (n)
    WHERE n.id IN $ids OR n.example_id IN $ids
    RETURN coalesce(n.id, n.example_id) AS example_id,
           coalesce(n.title, 'N/A') AS title,
           coalesce(n.text, n.page_content) AS content
    ORDER BY example_id
    """

    documents = []

    with driver.session(database=database) as session:
        results = session.run(query, ids=example_ids)

        for record in results:
            documents.append({
                "example_id": record["example_id"],
                "title": record["title"],
                "content": record["content"]
            })

    driver.close()

    print(f"\n[INFO] Total documents in graph (filtered): {len(documents)}\n")

    for i, doc in enumerate(documents):
        print(f"--- Document {i} ---")
        print("example_id:", doc["example_id"])
        print("title:", doc["title"])
        print("content:", doc["content"][:200], "...")
        print()

    return documents


# ---------------------------------------------------------
# Run this file directly to inspect the graph database
# ---------------------------------------------------------
if __name__ == "__main__":

    load_dotenv()

    NEO4J_URI = os.environ["NEO4J_URI"]
    NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
    NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
    # This should point to the exact database used during the build phase
    NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "test01")

    print(f"\n[INFO] Inspecting Neo4j graph documents in: {NEO4J_DATABASE}...\n")

    docs = list_graph_documents(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE
    )



