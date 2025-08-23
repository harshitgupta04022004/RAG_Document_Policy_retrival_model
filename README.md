# Insurance RAG System 🏥📋

A comprehensive **Retrieval-Augmented Generation (RAG)** system for intelligent insurance policy analysis and query processing. This system combines advanced vector search capabilities with graph database storage to provide accurate, context-aware responses to insurance-related queries.

## 🌟 Features

- **Hybrid Retrieval**: Combines ChromaDB (vector database) and Neo4j (graph database) for comprehensive document retrieval
- **Advanced NLP**: Uses state-of-the-art embedding models (Qwen3-Embedding) and language models (Phi-3/DeepSeek)
- **PDF Processing**: Automated extraction and chunking of insurance policy documents
- **Intelligent Querying**: Context-aware responses with structured JSON output
- **Scalable Architecture**: Designed for production-ready insurance policy analysis
- **Multi-Database Storage**: Leverages both vector similarity and graph relationships

## 🏗️ System Architecture

```
PDF Documents → Text Extraction → Text Chunking → Embedding Generation
                                                        ↓
Query Input → Embedding → Hybrid Retrieval ← ChromaDB + Neo4j Storage
                ↓                                       ↑
Response Generation ← Language Model ← Context Retrieval
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install torch transformers sentence-transformers
pip install chromadb neo4j pypdf langchain
pip install numpy
```

### Environment Setup

```python
# Set your credentials
NEO4J_URI = 'neo4j+s://your-instance.databases.neo4j.io'
NEO4J_USERNAME = 'neo4j'
NEO4J_PASSWORD = 'your-password'
HF_TOKEN = 'your-huggingface-token'  # Optional for private models
```

### Basic Usage

```python
from insurance_rag_system import InsuranceRAGSystem

# Initialize the system
rag_system = InsuranceRAGSystem(
    neo4j_uri=NEO4J_URI,
    neo4j_username=NEO4J_USERNAME,
    neo4j_password=NEO4J_PASSWORD,
    hf_token=HF_TOKEN
)

# Process insurance policy documents
pdf_files = ['policy_document.pdf', 'terms_conditions.pdf']
rag_system.process_documents(pdf_files)

# Query the system
query = "What is covered under accidental death in this policy?"
response = rag_system.query(query, retrieval_method="hybrid")
print(response)

# Clean up
rag_system.close_connections()
```

## 📊 Core Components

### 1. Document Processing Pipeline
- **PDF Text Extraction**: Uses PyPDF for reliable document parsing
- **Intelligent Chunking**: RecursiveCharacterTextSplitter for optimal context preservation
- **Embedding Generation**: Qwen3-Embedding-0.6B for high-quality vector representations

### 2. Dual Database Storage
- **ChromaDB**: Vector similarity search for semantic retrieval
- **Neo4j**: Graph relationships for contextual document structure

### 3. Hybrid Retrieval System
```python
# Available retrieval methods
response = rag_system.query(query, retrieval_method="chromadb")  # Vector only
response = rag_system.query(query, retrieval_method="neo4j")     # Graph only
response = rag_system.query(query, retrieval_method="hybrid")    # Combined
```

### 4. Language Model Integration
- **Primary Model**: Microsoft Phi-3-mini-128k-instruct
- **Alternative**: DeepSeek-R1-Distill-Qwen-1.5B
- **Structured Output**: JSON-formatted responses with clause references

## 🔧 Configuration Options

### Model Configuration
```python
rag_system = InsuranceRAGSystem(
    embedding_model="Qwen/Qwen3-Embedding-0.6B",
    llm_model="microsoft/Phi-3-mini-128k-instruct",
    chunk_size=256,
    chunk_overlap=32
)
```

### Database Configuration
- **ChromaDB**: Persistent storage with configurable collection names
- **Neo4j**: Vector indexing with cosine similarity for optimal retrieval
- **Embedding Dimension**: 1024-dimensional vectors for rich semantic representation

## 📋 Example Queries

```python
queries = [
    "What is the definition of Burglary in the insurance policy?",
    "What is covered under accidental death in this policy?",
    "Who qualifies as a family member under the family travel policy?",
    "What are the exclusions under trip cancellation benefits?",
    "Does the policy cover hospitalization due to COVID-19?",
    "What is meant by deductible in this policy?",
    "Explain the conditions under which repatriation of remains is covered."
]

for query in queries:
    response = rag_system.query(query)
    print(f"Query: {query}")
    print(f"Response: {response}\n")
```

## 📈 Performance Features

- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Batch Processing**: Efficient handling of large document collections
- **Memory Optimization**: Smart model loading and resource management
- **Error Handling**: Comprehensive exception handling and logging

## 🎯 Use Cases

### Insurance Industry Applications
- **Policy Analysis**: Automated policy document comprehension
- **Claims Processing**: Intelligent claim eligibility assessment
- **Customer Support**: Instant policy-related query resolution
- **Compliance Checking**: Automated regulatory compliance verification

### Business Benefits
- **Accuracy**: Eliminates manual policy interpretation errors
- **Efficiency**: 24/7 automated policy consultation
- **Scalability**: Handles thousands of concurrent queries
- **Cost Reduction**: Reduces manual customer support overhead

## 🔍 Technical Specifications

### Supported File Formats
- PDF documents (insurance policies, terms & conditions)
- Extensible to other document formats

### Database Requirements
- **Neo4j**: Version 4.0+ with vector indexing support
- **ChromaDB**: Latest version for optimal vector operations

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU-only operation
- **Recommended**: 16GB+ RAM, NVIDIA GPU with 6GB+ VRAM
- **Storage**: Varies based on document corpus size

## 🛠️ Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/harshitgupta04022004/insurance-rag-system.git
cd insurance-rag-system
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Databases
```bash
# Set up Neo4j (local or cloud)
# Configure ChromaDB storage path
# Set environment variables
```

### Step 4: Run Example
```bash
python example_usage.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone the repo
git clone https://github.com/harshitgupta04022004/insurance-rag-system.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```



## 🙋‍♂️ Author

**Harshit Gupta**
- GitHub: [@harshitgupta04022004](https://github.com/harshitgupta04022004)
- LinkedIn: [harshit-gupta-38023228b](https://www.linkedin.com/in/harshit-gupta-38023228b)
- Email: Harshitgupta040204@gmail.com

## 🎓 Academic Context

Developed as part of advanced AI/ML research at **Indian Institute of Information Technology, Nagpur**
- Course: B.Tech Computer Science and Engineering
- Focus: Natural Language Processing and Information Retrieval Systems

## 🏆 Recognition

- **IIT-BHU Hackathon**: 3rd Place for advanced NLP system implementation
- **Shell.ai Hackathon**: Top performance in predictive modeling competition

## 📚 References & Citations

- Qwen Embedding Models: [Qwen/Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- Microsoft Phi-3: [microsoft/Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)
- ChromaDB: [https://www.trychroma.com/](https://www.trychroma.com/)
- Neo4j: [https://neo4j.com/](https://neo4j.com/)

## 🔄 Version History

- **v1.0.0**: Initial release with hybrid RAG implementation
- **v1.1.0**: Enhanced query processing and response formatting
- **v1.2.0**: Added GPU acceleration and performance optimizations

## 🐛 Known Issues

- Large document processing may require significant memory
- Neo4j vector indexing requires sufficient storage space
- GPU memory optimization ongoing for very large models

## 🔮 Future Enhancements

- [ ] Support for additional document formats (DOCX, TXT)
- [ ] Real-time model fine-tuning capabilities
- [ ] Advanced graph relationship extraction
- [ ] Web interface for non-technical users
- [ ] Multi-language support for international policies
- [ ] Integration with insurance company APIs

---

⭐ **Star this repository if you find it helpful!**

For questions, issues, or suggestions, please open an issue or reach out directly.
