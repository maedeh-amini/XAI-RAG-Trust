"""
Knowledge Graph Search and Retrieval Module for Benchmark Dataset.

This script executes the Retrieval-Augmented Generation query phase against 
the Neo4j Knowledge Graph using the benchmark dataset. It employs a hybrid retrieval 
approach: fetching semantically relevant nodes via vector search, and then enriching 
that context by traversing the graph to extract adjacent entities and relationships.

The enriched context is formatted and passed through a LangChain Expression Language 
(LCEL) chain to the locally deployed generation model (e.g., Gemma 31B) to synthesize 
a final answer.
"""

import os
from typing import List
from dotenv import load_dotenv

from langchain_neo4j import Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load configuration
load_dotenv(override=True)

# =========================================================
# 2. Custom Embedding Provider
# =========================================================
class GatewayCompatibleEmbeddings(OpenAIEmbeddings):
    """
    A custom wrapper to ensure compatibility with local server API gateways.
    
    Standard LangChain OpenAI integrations often pass parameters like `encoding_format` 
    or `chunk_size` which are rejected by non-standard OpenAI gateways. This class 
    intercepts the request and strips unsupported parameters before execution.
    """
    def _get_len_safe_embeddings(
        self, texts: List[str], *, engine: str = None, **kwargs
    ) -> List[List[float]]:
        """
        Generates embeddings while safely ignoring incompatible parameters.
        
        Args:
            texts (List[str]): The list of text strings to embed.
            engine (str, optional): Legacy parameter, largely unused.
            **kwargs: Additional keyword arguments passed from LangChain.
            
        Returns:
            List[List[float]]: A list of dense vector arrays.
        """
        # Remove parameters often rejected by non-standard OpenAI gateways
        kwargs.pop("encoding_format", None)
        kwargs.pop("chunk_size", None)
        
        responses = self.client.create(
            input=texts,
            model=self.model,
            encoding_format=None, 
            **kwargs
        )
        return [d.embedding for d in responses.data]

# =========================================================
# 3. Component Initialization
# =========================================================

# Initialize embedding provider
embeddings = GatewayCompatibleEmbeddings(
    api_key=os.environ["EMBEDDING_MODEL_API_KEY"],
    base_url=os.environ["EMBEDDING_MODEL_API_BASE"],
    model=os.environ["EMBEDDING_MODEL"]
)

# Initialize the LLM generator
llm = ChatOpenAI(
    api_key=os.environ["UOS_API_KEY"],
    base_url=os.environ["UOS_API_BASE"],
    model=os.environ["UOS_Model"],
    temperature=0
)

# =========================================================
# 4. Graph Retrieval Strategy
# =========================================================

# Clean Cypher query:
# Extracts ONLY the raw text and lightweight relationship strings.
# EXCLUSION LOGIC: Explicitly filters out structural graph edges ('NEXT_CHUNK', 
# 'PARENT', etc.) to prevent the LLM's context window from overflowing with 
# irrelevant system metadata, focusing only on semantic relationships.
retrieval_query = """
OPTIONAL MATCH (node)-[rel]->(neighbor)
WHERE NOT type(rel) IN ['NEXT_CHUNK', 'PREVIOUS_CHUNK', 'PARENT', 'PART_OF_DOCUMENT', 'SOURCE']
WITH node, score, 
     collect(DISTINCT type(rel) + ' -> ' + coalesce(neighbor.name, neighbor.id, neighbor.title, '')) AS raw_relations
WITH node, score,
     [r IN raw_relations WHERE NOT r ENDS WITH ' -> '] AS relations
RETURN 
    coalesce(node.text, '') + 
    CASE WHEN size(relations) > 0 THEN '\n\nGRAPH RELATIONSHIPS:\n' + apoc.text.join(relations, ' | ') ELSE '' END AS text,
    score, 
    {source: coalesce(node.source, node.hf_id, 'Unknown')} AS metadata
"""

# Establish connection to Neo4j Vector Index
vector_db = Neo4jVector.from_existing_index(
    embedding=embeddings,
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
    database=os.environ.get("NEO4J_DATABASE", "graphdb2"), 
    index_name="chunk_vector_index", 
    search_type="vector",  
    retrieval_query=retrieval_query
)

# =========================================================
# 5. RAG Chain Construction (LCEL)
# =========================================================

template = """You are a reliable AI assistant.
Answer the ‘Question’ based ONLY on the provided
‘Context’, which includes both raw text and structural graph relationships. 

Context: {context}

Question: {question}

If you do not find any relevant information in the
provided context, respond with: "I DONT KNOW"."""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    """
    Aggregates retrieved LangChain Document objects into a clean, unified context string.
    
    Args:
        docs (List[Document]): The documents returned by the Neo4j retriever.
        
    Returns:
        str: A concatenated string of all document contents, separated by dividers.
    """
    if not docs:
        return "No relevant information found in the database."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# Set top-k retrieval count to balance context richness with prompt limits
retriever = vector_db.as_retriever(search_kwargs={'k': 5})

# LangChain Expression Language (LCEL) Pipeline
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# =========================================================
# 6. Execution & Diagnostic Testing
# =========================================================
if __name__ == "__main__":
    question = "To what team was the 2014 NBA Rookie of the Year traded in October 2016? "
    print(f"\n[QUERYING SYSTEM]: {question}")
    
    try:
        # Step 1: Print raw retrieved documents to verify database contents and Cypher extraction
        print("\n--- [DEBUG] RETRIEVED CONTEXT ---")
        retrieved_docs = retriever.invoke(question)
        if not retrieved_docs:
            print("[WARNING] Vector search returned 0 documents.")
        else:
            for idx, doc in enumerate(retrieved_docs):
                print(f"\n[Doc {idx + 1} Preview]:\n{doc.page_content[:300]}...")
        print("---------------------------------\n")

        # Step 2: Invoke full generative chain
        response = rag_chain.invoke(question)
        print(f"[SYSTEM RESPONSE]:\n{response}\n")
    except Exception as e:
        print(f"\n[CRITICAL ERROR]: {e}")

