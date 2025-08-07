#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install unstructured
# !pip install -U transformers
# !pip install chromadb
# !pip install neo4j
# !pip install langchain_community
# !pip install llama_index
# !pip install torch transformers datasets peft matplotlib
# !pip install sentence-transformers


# In[2]:


import os
import json
import numpy as np


# --- For Data Loading and Processing ---
# Use these to load and split documents like PDFs, text files, etc.
# LlamaIndex and LangChain also have their own document loaders.
from pypdf import PdfReader
from unstructured.partition.auto import partition
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- For Embedding and Vector Search ---
# SentenceTransformer is used for generating embeddings.
# ChromaDB is a popular choice for a local vector store.
from sentence_transformers import SentenceTransformer
import chromadb

# --- For Graph Database Interaction ---
# The Neo4j driver to connect to and query a Neo4j graph database.
# You must have a Neo4j server running.
from neo4j import GraphDatabase

# --- For Core RAG Pipeline Orchestration ---
# Libraries for building the overall RAG pipeline and connecting components.
# You would typically choose either LlamaIndex or LangChain, not both.
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
# from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.graphs import Neo4jGraph

# from llama_index.readers.file import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, StorageContext, ServiceContext
# from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.graph_stores.neo4j import Neo4jGraphStore

# --- For Final Generation with Pleias-RAG ---
# This is the specialized model you've chosen for generating the final answer
# with a focus on factual citations.
# from pleias_rag.main import RAGWithCitations

# --- For Utility and Type Hinting ---
from typing import List, Dict, Any

# from langchain.llms import HuggingFaceHub
# from openai import OpenAI


# In[3]:


NEO4J_URI='neo4j+s://fdb1cdfe.databases.neo4j.io'
NEO4J_USERNAME='neo4j'
NEO4J_PASSWORD='4ygC6vXH3auM-yPJ8XW1oUjHQDSJCL0IXCSAK0xKUF4'
NEO4J_DATABASE='neo4j'
AURA_INSTANCEID='fdb1cdfe'
AURA_INSTANCENAME='Free instance'


# In[5]:


# import PyPDF2
import pypdf
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text


# In[6]:


# pdf_1_text = extract_text_from_pdf('./BAJHLIP23020V012223.pdf')
# pdf_2_text = extract_text_from_pdf('./EDLHLGA23009V012223.pdf')
# pdf_3_text = extract_text_from_pdf('./ICIHLIP22012V012223.pdf')
pdf_4_text = extract_text_from_pdf('./CHOTGDP23004V012223.pdf')
# pdf_5_text = extract_text_from_pdf('./HDFHLIP23024V072223.pdf')


# In[8]:


documents = pdf_4_text # + "\n\n" + pdf_2_text + "\n\n" + pdf_3_text + "\n\n" + pdf_4_text + "\n\n" + pdf_5_text  


# In[9]:


get_ipython().system('export HF_TOKEN="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"')
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 256,
    chunk_overlap = 32,
    is_separator_regex=False
)
Doc = text_splitter.create_documents([documents])


# In[10]:


Doc[0]


# In[11]:


import os
import json


os.environ["HF_TOKEN"] = "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"
from sentence_transformers import SentenceTransformer


sen_emb_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B',device="cpu")

chunks = [doc.page_content for doc in Doc]
embeddings = sen_emb_model.encode(chunks, convert_to_tensor=False)
print("\nEncoding successfully generated.")
print("The number of embeddings is:", len(embeddings))
print("The shape of the first embedding vector is:", embeddings[0].shape)
print("The first 5 values of the first embedding are:", embeddings[0][:5])


# In a real RAG system, you would store this embedding in a vector database.


# ## ChromaDB is a popular choice for a local vector store.

# In[12]:


# --- Step 3: Storing in Vector Database (ChromaDB) ---
print("\n3. Setting up ChromaDB and adding data...")
client = chromadb.PersistentClient(path="doc_db")# chromadb.EphemeralClient()

# Create a collection. If it already exists, we'll get the existing one.
collection_name = "document_chunks_collection"
collection = client.get_or_create_collection(name=collection_name)

# Generate unique IDs for each chunk.
ids = [f"chunk_{i}" for i in range(len(chunks))]

# Add the chunks (documents), their embeddings, and IDs to the collection.
# Note: ChromaDB expects embeddings as a list of lists/arrays.
collection.add(
    ids=ids,
    documents=chunks,
    embeddings=embeddings.tolist() # Convert numpy array to list
)
print(f"   - Added {collection.count()} items to the '{collection_name}' collection.")


# In[13]:


import json

def get_retrieved_clauses(query_text):
    # Step 1: Generate embedding
    query_embedding = sen_emb_model.encode([query_text], convert_to_tensor=True)

    # Step 2: Query the ChromaDB collection
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3
    )

    # Step 3: Retrieve documents
    retrieved_docs = results['documents'][0]  # List of top chunks

    # Step 4: Format into JSON
    retrieved_clauses_json = []
    for i, clause in enumerate(retrieved_docs):
        retrieved_clauses_json.append({
            "clause_id": f"C{i+1}",
            "text": clause.strip()
        })

    # Step 5: Return pretty-printed JSON string
    retrieved_clauses_json_str = json.dumps(retrieved_clauses_json, indent=2)
    return retrieved_clauses_json_str

print(get_retrieved_clauses("What is the definition of Burglary in the insurance policy?"))


# # Graph database

# In[14]:


# --- Step 4: Storing in Graph Database (Neo4j) ---
print("\n4. Setting up Neo4j and adding data...")

def add_chunks_to_neo4j(uri, user, password, chunks_data):
    """
    Connects to Neo4j and creates nodes for each document chunk,
    then creates relationships between sequential chunks.
    """
    # The driver is the entry point for any Neo4j application
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        # Create a uniqueness constraint on the chunk ID to prevent duplicates
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")

        # Iterate through chunks and their embeddings to create nodes
        for i, (text, embedding) in enumerate(chunks_data):
            # The embedding is stored as a list of floats.
            params = {
                'id': f"chunk_{i}",
                'text': text,
                'embedding': embedding.tolist(), # Convert numpy array to list
                'chunk_index': i
            }
            # MERGE finds or creates a node with the given properties.
            # ON CREATE SET adds properties only when the node is first created.
            session.run("""
                MERGE (c:Chunk {id: $id})
                ON CREATE SET c.text = $text, c.embedding = $embedding, c.chunkIndex = $chunk_index
            """, params)

        print(f"   - Merged {len(chunks_data)} chunk nodes into Neo4j.")

        # Create relationships between consecutive chunks
        for i in range(len(chunks_data) - 1):
            params = {'current_id': f"chunk_{i}", 'next_id': f"chunk_{i+1}"}
            session.run("""
                MATCH (current:Chunk {id: $current_id})
                MATCH (next:Chunk {id: $next_id})
                MERGE (current)-[:NEXT]->(next)
            """, params)
        print("   - Created :NEXT relationships between chunks.")

    # Always close the driver when your application is done
    driver.close()

# Combine chunks and embeddings for easier processing
chunks_with_embeddings = list(zip(chunks, embeddings))


# In[15]:


try:
    # Make sure you have replaced the placeholder credentials above
    if NEO4J_URI == "neo4j+s://your_aura_db_uri.databases.neo4j.io":
        print("   - SKIPPING NEO4J: Please update NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD.")
    else:
        add_chunks_to_neo4j(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, chunks_with_embeddings)
        print("   - Successfully added data to Neo4j.")

except Exception as e:
    print(f"\nAn error occurred with the Neo4j connection: {e}")
    print("Please ensure your Neo4j server is running and the credentials are correct.")

print("\n--- Pipeline execution finished ---")


# In[16]:


MODEL_NAME = 'Qwen/Qwen3-Embedding-0.6B'
EMBEDDING_DIMENSION = 1024 # 'all-MiniLM-L6-v2' has 384 dimensions
VECTOR_INDEX_NAME = "chunkEmbeddings"


def create_neo4j_vector_index(driver):
    """
    Creates a vector index in Neo4j if it doesn't already exist.
    This is a one-time setup step.
    """
    print("2. Checking for and creating Neo4j vector index...")
    with driver.session(database="neo4j") as session: # Specify the database for Aura
        # Check if the index already exists
        result = session.run("SHOW INDEXES YIELD name WHERE name = $index_name", index_name=VECTOR_INDEX_NAME)
        if result.single():
            print(f"   - Vector index '{VECTOR_INDEX_NAME}' already exists.")
            return

        # Create the vector index
        # This specifies the node label (Chunk), the property (embedding),
        # the dimensions of the vectors, and the similarity metric (cosine).
        query = f"""
        CREATE VECTOR INDEX `{VECTOR_INDEX_NAME}` IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS {{ indexConfig: {{
            `vector.dimensions`: {EMBEDDING_DIMENSION},
            `vector.similarity_function`: 'cosine'
        }}
        }}
        """
        try:
            session.run(query)
            print(f"   - Successfully created vector index '{VECTOR_INDEX_NAME}'.")
        except Exception as e:
            print(f"   - Error creating vector index: {e}")
            print("   - Please ensure your Neo4j version supports vector indexes (5.11+).")


# --- Retrieval Function ---
def retrieve_from_neo4j(driver, model, query_text, k=3):
    """
    Performs a vector similarity search directly in Neo4j to find the top k chunks.

    Args:
        driver: The Neo4j database driver.
        model: The initialized Sentence Transformer model.
        query_text (str): The user's query.
        k (int): The number of top chunks to retrieve.

    Returns:
        list: A list of dictionaries, each containing the chunk text and similarity score.
    """
    print(f"\n3. Retrieving top {k} chunks from Neo4j for query: '{query_text}'")

    # Step 1: Generate the embedding for the query
    query_embedding = sen_emb_model.encode(query_text).tolist()

    # Step 2: Query Neo4j using the vector index
    cypher_query = f"""
    CALL db.index.vector.queryNodes('{VECTOR_INDEX_NAME}', $k, $embedding)
    YIELD node, score
    RETURN node.text AS text, score
    """

    with driver.session(database="neo4j") as session: # Specify the database for Aura
        try:
            result = session.run(cypher_query, k=k, embedding=query_embedding)
            return [record.data() for record in result]
        except Exception as e:
            print(f"   - Error querying Neo4j vector index: {e}")
            print(f"   - Make sure the index '{VECTOR_INDEX_NAME}' was created successfully.")
            return []




# In[17]:


# --- Main Execution ---
print("1. Initializing models and database connections...")
try:
    # Initialize the Sentence Transformer model
    # model = SentenceTransformer(MODEL_NAME)

    # Initialize Neo4j driver
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("   - Connections established successfully.")
except Exception as e:
    print(f"   - A connection error occurred: {e}")
    exit()

# ONE-TIME SETUP: Create the index. You only need to run this once.
create_neo4j_vector_index(driver)



# # top_chunks = retrieve_from_neo4j(driver, sen_emb_model, user_query, k=3)
# 

# In[19]:


# --- Perform Retrieval ---
user_query = "What is the definition of Burglary in the insurance policy?"
top_chunks = retrieve_from_neo4j(driver, sen_emb_model, user_query, k=3)

# Display the results
print("\n4. Displaying retrieval results from Neo4j:")
if not top_chunks:
    print("   - No results found.")
else:
    for i, chunk in enumerate(top_chunks, 1):
        print("-" * 40)
        print(f"Result {i} (Similarity Score: {chunk['score']:.4f}):")
        print(f"   >>> {chunk['text'].strip()} <<<")

# Close the Neo4j driver connection
driver.close()
print("\n--- Retrieval finished ---")


# In[40]:


(top_chunks)


# In[ ]:





# # Loading the model deepseek-r1-7B

# In[13]:


# # Use a pipeline as a high-level helper
# from transformers import pipeline
# import torch

# pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# pipe(messages)


# In[20]:


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",torch_dtype=torch.float16,
    device_map="auto"  # This ensures all model parts go to GPU
)
model


# In[42]:


# retrieved_clauses_json = [
#     {
#         "clause_id": f"C{i+1}",
#         "text": chunk["text"].strip()
#     } for i, chunk in enumerate(top_chunks)
# ]


# user_query = input()
# prompt = f"""
# You are an expert insurance-policy decision assistant with deep domain knowledge in maternity and routine-care covers. You process natural-language queries by strictly analyzing **only** the user’s input and the provided policy clauses—no invented rules.

# Your task:
# 1. Parse the user’s query and extract structured information.
# 2. Evaluate eligibility **using only** the retrieved clauses—do not assume or invent any additional rules.
# 3. Return a final decision and justification by referencing the exact clause IDs.

# ——

# USER QUERY:
# "{user_query}"

# ——

# RETRIEVED CLAUSES (from internal document search):
# {retrieved_clauses_json(retrieve_from_neo4j(driver, sen_emb_model, user_query, k=3))}

# ——

# Step 1: Extract these fields from the query:
# - age (number)
# - gender
# - procedure
# - city
# - policy_duration_months
# - selected_cover_option (i, ii, or iii)

# Step 2: Using **only** the clauses above:
# - Determine if the requested procedure/cover is eligible under the selected option.
# - Identify any exclusions.
# - Compute payout amount (if applicable).

# Step 3: Output exactly and only this JSON (no extra text), then immediately stop:
# {{
#   "decision": "<approved|rejected>",
#   "amount": <number|null>,
#   "justification": [
#     {{
#       "clause_id": "C1",
#       "reason": "…"
#     }}
#   ]
# }}
# <<END_OF_JSON>>
# """


# In[47]:


user_query = input()
top_chunks = retrieve_from_neo4j(driver, sen_emb_model, user_query, k=3)

retrieved_clauses_json = [
    {
        "clause_id": f"C{i+1}",
        "text": chunk["text"].strip()
    } for i, chunk in enumerate(top_chunks)
]

retrieved_clauses_json_str = json.dumps(retrieved_clauses_json, indent=2)

prompt = f"""
You are an expert insurance-policy decision assistant with deep domain knowledge in maternity and routine-care covers. You process natural-language queries by strictly analyzing **only** the user’s input and the provided policy clauses—no invented rules.

Your task:
1. Parse the user’s query and extract structured information.
2. Evaluate eligibility **using only** the retrieved clauses—do not assume or invent any additional rules.
3. Return a final decision and justification by referencing the exact clause IDs.

——

USER QUERY:
"{user_query}"

——

RETRIEVED CLAUSES (from internal document search):
{retrieved_clauses_json_str}

——

Step 1: Extract these fields from the query:
- age (number)
- gender
- procedure
- city
- policy_duration_months
- selected_cover_option (i, ii, or iii)

Step 2: Using **only** the clauses above:
- Determine if the requested procedure/cover is eligible under the selected option.
- Identify any exclusions.
- Compute payout amount (if applicable).

Step 3: Output exactly and only this JSON (no extra text), then immediately stop:
{{
  "decision": "<approved|rejected>",
  "amount": <number|null>,
  "justification": [
    {{
      "clause_id": "C1",
      "reason": "…"
    }}
  ]
}}
<<END_OF_JSON>>
"""


# In[48]:


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=False,           # deterministic
    num_beams=3,               # greedy
)
raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
json_only = raw.split("<<END_OF_JSON>>", 1)[1].strip()

print(json_only)



# "What is covered under accidental death in this policy?"
# 
# "Who qualifies as a family member under the family travel policy?"
# 
# "What is the definition of pre-existing disease in this insurance document?"
# 
# "Explain the scope of medical evacuation coverage."
# 
# "What does this policy define as a medical emergency?"
# 
# "Is loss of baggage covered under the travel insurance policy?"
# 
# "What are the exclusions under trip cancellation benefits?"
# 
# "Does the policy cover hospitalization due to COVID-19?"
# 
# "What is meant by deductible in this policy?"
# 
# "Explain the conditions under which repatriation of remains is covered."

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




