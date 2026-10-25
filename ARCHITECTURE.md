# ARCHITECTURE SPECIFICATION
## GraphRAG vs. Dense Vector RAG: Empirical Evaluation and Hybrid Optimization

---

## 1. High-Level System Architecture

```
                                 +--------------------------------+
                                 |        Web Client / UI         |
                                 | (Streamlit Lab + Multi-Tenant) |
                                 +---------------+----------------+
                                                 |
                                                 v
                                 +--------------------------------+
                                 |    Authentication & Session    |
                                 |  (UserRegistry & KB Isolation) |
                                 +---------------+----------------+
                                                 |
                         +-----------------------+-----------------------+
                         |                                               |
                         v                                               v
            +------------------------+                      +------------------------+
            |  Document Management   |                      |    Query & Chat Lab    |
            |   & Ingestion Pipeline |                      | (Vector/Graph/Hybrid)  |
            +------------+-----------+                      +------------+-----------+
                         |                                               |
         +---------------+---------------+                               |
         |                               |                               v
         v                               v                  +------------------------+
  [ Text Chunker ]             [ Entity Extractor ]         |      Query Router      |
         |                               |                  +------------+-----------+
         v                               v                               |
 [ Embedding Engine ]          [ Relation Discovery ]       +------------+-----------+
  (SentenceTransformers)       (Typed Domain Predicates)    |            |           |
         |                               |                  v            v           v
         v                               v               [Vector]     [Graph]     [Hybrid]
+-----------------+             +-----------------+        |            |            |
|   Qdrant VDB    |             |   Neo4j Graph   |        +-----+------+            |
|  (Dense Vector) |             |  (Cypher Nodes) |              |                   |
+-----------------+             +-----------------+              +--------+----------+
                                                                          |
                                                                          v
                                                             [ Reciprocal Rank Fusion ]
                                                                      (RRF)
                                                                          |
                                                                          v
                                                             [ Context Construction ]
                                                                          |
                                                                          v
                                                             [ Ollama LLM Generation ]
                                                                          |
                                                                          v
                                                             [ Evaluation & Telemetry ]
                                                              (Faithfulness, Recall, MRR)
```

---

## 2. Core Modules and Responsibilities

### 2.1 Multi-Tenant User Registry (`src/rag_benchmark_lab/services/user_registry.py`)
- **Authentication:** Enforces secure user registration, email validation, and salted password hashing (PBKDF2-SHA256 with optional bcrypt/argon2).
- **Knowledge Base Isolation:** Maps `User -> KnowledgeBase -> Documents -> Chunks -> Vector/Graph Records`.
- **Conversation Persistence:** Retains full conversation sessions, user queries, assistant responses, strategy choices, citations, and latencies.

### 2.2 Dual Ingestion Engine (`src/rag_benchmark_lab/services/document_service.py`)
- **Document Loading & Cleaning:** Normalizes PDF, TXT, and DOCX files.
- **Recursive Token Chunking:** Configurable chunk size and overlap with stable identifiers.
- **Dense Vector Generation:** Produces 384-dimensional embeddings indexed directly into Qdrant collections.
- **Knowledge Graph Extraction:** Extracts named entities (`Technology`, `Concept`, `Component`, `Metric`) and semantic relationships (`USES`, `EVALUATES`, `STORES`, `INDEXES`, `RETRIEVES`), persisting them with document and chunk provenance into Neo4j.

### 2.3 Retrieval & Query Routing Engine
- **Dense Vector RAG (`vector_retriever.py`):** Queries Qdrant using cosine similarity, applies Top-K filtering, and retrieves ranked passages.
- **GraphRAG (`graph_retriever.py`):** Performs Cypher entity matching, bounded multi-hop neighborhood traversal, and maps graph nodes back to source text chunks.
- **Hybrid RAG (`hybrid_retriever.py`):** Executes dual vector and graph searches concurrently, applies Reciprocal Rank Fusion ($k=60$), deduplicates evidence, and preserves provenance.
- **Query Router (`query_router.py`):** Explicit rule-based and semantic classifier evaluating entity density and query keywords with transparent rationale logging.

### 2.4 Empirical Benchmark & Evaluation Engine (`evaluation/benchmark_runner.py`)
- Evaluates retrieval quality: Context Precision, Context Recall, Recall@K, Hit Rate@K, MRR.
- Evaluates generation quality: Faithfulness, Answer Relevancy, Exact Match, Token F1.
- Tracks performance telemetry: Retrieval latency, generation latency, total latency, token usage, and cost estimates.
- Persists all experiment runs and per-query observations in SQLite with CSV export capability.

---

## 3. Data Schema & Isolation

### Relational Schema (SQLite):
- `users`: User credentials, password hashes, timestamps.
- `knowledge_bases`: Isolated knowledge base containers tied to `user_id`.
- `kb_documents`: Association table enforcing document ownership.
- `conversations`: Persistent chat threads scoped to `user_id` and `kb_id`.
- `messages`: Individual turn messages with latency, strategy, citations, and metadata.
- `documents`: Document records and dual-indexing status.
- `chunks`: Extracted text chunks with offsets and page numbers.
- `evaluation_runs` & `evaluation_results`: Benchmark experiment records.

### Graph Schema (Neo4j):
```cypher
(:Document {id, name, path})
    -[:HAS_CHUNK]->
(:Chunk {id, index, page_number})
    -[:MENTIONS]->
(:Entity {id, name, type})
    -[:RELATES_TO {type, confidence}]->
(:Entity)
```
