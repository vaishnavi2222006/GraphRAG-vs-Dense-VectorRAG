# RAG Benchmark Lab

**GraphRAG vs. Dense Vector RAG: Empirical Evaluation and Hybrid Optimization for Domain-Specific Complex Reasoning**

A comprehensive, multi-user experimental web application designed to evaluate, compare, and benchmark **Dense Vector RAG (Qdrant)**, **GraphRAG (Neo4j)**, and **Hybrid RAG (Reciprocal Rank Fusion)** over custom document corpora.

---

## 🚀 Key Features

1. **User Authentication & Multi-Tenancy:**
   - Registration and login with salted PBKDF2 password hashing.
   - User workspaces with private, isolated **Knowledge Bases**.
   - Secure per-user resource boundaries for documents, chunks, vectors, graph entities, and chat history.

2. **Dual-Store Ingestion Engine:**
   - Support for PDF, TXT, and DOCX documents with automated cleaning.
   - Recursive token chunking with stable chunk IDs and page/offset provenance.
   - Dense vector embedding generation indexed into **Qdrant**.
   - Entity & relationship extraction indexed into **Neo4j** with schema constraints.

3. **Multi-Strategy RAG & Intelligent Routing:**
   - **Dense Vector RAG:** Qdrant similarity search with cosine distance and Top-K filtering.
   - **GraphRAG:** Neo4j Cypher entity matching, bounded multi-hop neighborhood traversal, and source chunk mapping.
   - **Hybrid RAG:** Dual retrieval fused using standard Reciprocal Rank Fusion ($k=60$) with deduplication.
   - **Query Router:** Rule-based & heuristic classifier evaluating entity density and query characteristics with transparent routing rationale.

4. **Conversational Chat with Citations:**
   - Persistent multi-turn chat sessions stored in SQLite.
   - Clickable source citations with document name, page number, and chunk index.
   - Per-turn latency telemetry (retrieval ms, generation ms, total ms) and model identifiers.

5. **Side-by-Side 3-Way Comparison:**
   - Simultaneous execution of Vector, Graph, and Hybrid RAG on the identical prompt.
   - Side-by-side answer and evidence panels with a comparative metrics table.

6. **Empirical Evaluation & Research Analytics:**
   - Built-in and custom JSON/JSONL dataset execution.
   - Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall, Recall@K, Hit Rate@K, MRR.
   - Interactive Plotly visualizations for latency tradeoffs and metric comparisons.
   - One-click CSV export for downstream research reporting.

7. **System Diagnostics:**
   - Live health checks for Qdrant, Neo4j, Ollama, and SQLite with detailed connection diagnostics.

---

## 🛠️ Technology Stack

- **Frontend & UI:** Streamlit with custom CSS research design system.
- **Backend:** Python 3.12 (modular service architecture).
- **Vector Database:** Qdrant (Docker container on port 6333).
- **Graph Database:** Neo4j Community 5.23 (Docker container on Bolt port 7687).
- **Primary LLM Engine:** Groq API (`llama-3.3-70b-versatile` / high-speed inference, with optional Google Gemini and local Ollama support).
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`, 384 dimensions).
- **Metadata DB:** SQLite 3 (isolated relational tables for users, KBs, conversations, documents, results).

---

## ⚡ Quick Start

### 1. Environment Setup

```bash
# Clone the repository
cd "GraphRAG vs. Dense Vector RAG"

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies in editable mode
pip install -e .
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Ensure `.env` contains your desired endpoints and API keys:
```ini
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

QDRANT_URL=http://localhost:6333
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 3. Start Database Containers

```bash
sudo docker compose up -d
```

Verify containers are running:
```bash
docker ps
```

### 4. Run the Web Application

```bash
streamlit run src/rag_benchmark_lab/app.py
```

The application will be accessible at: `http://localhost:8501`

---

## 🧪 Running Tests

To run the complete automated test suite:

```bash
source .venv/bin/activate
pytest -v
```

---

## 📁 Project Structure

```
├── pages/
│   ├── 0_Login.py              # User registration and login
│   ├── 0_Knowledge_Bases.py    # Multi-tenant KB management
│   ├── 1_Overview.py           # Research project overview
│   ├── 2_Corpus_Management.py  # Document upload and dual indexing
│   ├── 3_Chat.py               # Persistent conversational chatbot
│   ├── 3_Query_Playground.py   # Interactive single query sandbox
│   ├── 4_Comparison.py         # Side-by-side 3-way RAG comparison
│   ├── 5_Graph_Explorer.py     # Neo4j interactive knowledge graph
│   ├── 6_Evaluation_Lab.py     # Benchmark dataset runner
│   ├── 7_Analytics.py          # Empirical metrics, charts & CSV export
│   └── 8_Settings.py           # Live service diagnostics & config
├── src/rag_benchmark_lab/
│   ├── config/                 # Pydantic application settings
│   ├── embeddings/             # Sentence transformer service
│   ├── evaluation/             # Metrics, runner, dataset parser
│   ├── generation/             # Ollama service & prompts
│   ├── graph_store/            # Neo4j store, extractor, retriever
│   ├── ingestion/              # Document loader & chunker
│   ├── retrieval/              # Vector, hybrid, RRF, router
│   ├── schemas/                # Dataclasses (auth, documents, evaluation)
│   ├── services/               # User registry, document service, query service
│   └── utils/                  # Auth helpers, UI injection, timing
├── tests/
│   ├── unit/                   # Unit tests (auth, chunker, rrf, metrics)
│   └── integration/            # Multi-service integration tests
├── ARCHITECTURE.md             # System architecture specification
├── IMPLEMENTATION_STATUS.md    # Acceptance criteria verification matrix
├── PROJECT_AUDIT.md            # Audit of bugs, fixes, and improvements
└── TESTING_REPORT.md           # Pytest execution report
```
