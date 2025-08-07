#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import PyPDF2
from sentence_transformers import SentenceTransformer
import pypdf
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text


# In[2]:


# pdf_1_text = extract_text_from_pdf('./BAJHLIP23020V012223.pdf')
pdf_2_text = extract_text_from_pdf('./EDLHLGA23009V012223.pdf')
# pdf_3_text = extract_text_from_pdf('./ICIHLIP22012V012223.pdf')
# pdf_4_text = extract_text_from_pdf('./CHOTGDP23004V012223.pdf')
# pdf_5_text = extract_text_from_pdf('./HDFHLIP23024V072223.pdf')


# In[3]:


documents = "\n\n" + pdf_2_text # + "\n\n" + pdf_2_text + "\n\n" + pdf_3_text + "\n\n" + pdf_4_text + "\n\n" + pdf_5_text  


# In[4]:


# from langchain.text_splitter import RecursiveCharacterTextSplitter
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 300,
#     chunk_overlap = 32,
#     is_separator_regex=False
# )
# Doc = text_splitter.create_documents([documents])


# In[5]:


# Doc[0]


# In[6]:


from Pleias_Rag.RagSystem import RagSystem
emb_model = "Qwen/Qwen3-Embedding-0.6B" # 'Qwen/Qwen3-Embedding-0.6B'

# Initialize the RAG system with an optional model path
rag_system = RagSystem(
    search_type="vector",
    db_path="data/rag_system_db",
    embeddings_model=emb_model,
    chunk_size=128,
    # model_path= "PleIAs/Pleias-RAG-350M"   # "PleIAs/Pleias-RAG-1B"# "meta-llama/Llama-2-7b-chat-hf"  # Optional - can also load model later
)


# # You may need to trust remote code for this model
# rag_system.load_model(
#     model_path="microsoft/phi-2",
#     trust_remote_code=True
# )

# model_path="microsoft/Phi-3-mini-4k-instruct"  # <-- The only change needed



# Add documents to the system
rag_system.add_and_chunk_documents([
   documents
])

# Load the generative model with specific generation parameters to prevent babbling
print("Loading generative model with constrained parameters...")
rag_system.load_model(
    model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=256,  # <-- Set a strict limit on output length
    temperature=0.1,    # <-- Reduce creativity and repetition
    do_sample=False     # <-- Use greedy decoding
)
print("Model loaded.")


# In[11]:


# The number of documents you want to retrieve.
# Start with a small number and increase if needed.
k_responses = 1

query = "Routine Medical Care would include expenses recommended by a doctor and incurred on –Pharmacy, Diagnostics, Doctor Consultations and Therapy"

# Pass the 'limit' argument to your query method
result = rag_system.query(query)


# In[12]:


# Access the results
print(f"Query: {result['query']}")
print(f"Response: {result['response']}")


# In[9]:


# # End-to-end RAG query
# query = "Routine Medical Care would include expenses recommended by a doctor and incurred on –Pharmacy, Diagnostics, Doctor Consultations and Therapy"
# result = rag_system.query(query)

# # Access the results
# print(f"Query: {result['query']}")
# print(f"Response: {result['response']}")


# In[10]:


# # --- 2. Separate Retrieval and Generation ---
# query = "Routine Medical Care would include expenses recommended by a doctor and incurred on –Pharmacy, Diagnostics, Doctor Consultations and Therapy"
# k_responses = 2 # The number of top documents you want to retrieve

# # Perform only the retrieval step to get the top 'k' documents
# # The .vector_search() method is what you need.
# retrieved_results = rag_system.vector_search(query, limit=k_responses)

# # Access the results
# print(f"Query: {query}")
# print(f"Retrieved {k_responses} documents:")
# for i, doc in enumerate(retrieved_results['documents'][0]):
#     print(f"\nDocument {i+1}:")
#     print(f"  Content: {doc}")
#     # You can also access the metadata if you stored it.
#     # print(f"  Metadata: {retrieved_results['metadatas'][0][i]}")

# # --- 3. Optional: Use the retrieved results to generate an answer ---
# # If you want to continue the RAG process, you would pass these retrieved documents
# # to a generation function.
# # This part is just for demonstration and is not part of your original query.
# # Here we would load the generative model and format the prompt manually.

# # print("\n--- Generating a final answer from retrieved documents ---")
# # # Load your generative model
# # rag_system.load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# # # Format the prompt for the generative model
# # sources_text = "\n".join([f"Source {i+1}: {d}" for i, d in enumerate(retrieved_results['documents'][0])])
# # full_prompt = f"Using the following context, answer the question.\n\nContext:\n{sources_text}\n\nQuestion: {query}\n\nAnswer:"

# # # Generate the final answer
# # final_response = rag_system._generate(full_prompt)
# # print(f"Final RAG Answer: {final_response}")


# In[ ]:




