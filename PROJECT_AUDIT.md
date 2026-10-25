# RAG Benchmark Lab — Project Audit & Architecture Review

**Project Title:** GraphRAG vs. Dense Vector RAG: Empirical Evaluation and Hybrid Optimization for Domain-Specific Complex Reasoning  
**Date:** 2026-10-18  
**Author:** AI Research & Engineering Pair  
**Status:** Audit Complete — Phase 1 & 2 Completed

---

## 1. System Inventory & Infrastructure Status

| Component | Target Architecture | Current State | Root Cause & Resolution |
| :--- | :--- | :--- | :--- |
| **Vector DB (Qdrant)** | Qdrant v1.11.4 container (`http://localhost:6333`) | **Active & Healthy** | Docker Desktop was started; Qdrant container is online with collection `rag_chunks` containing 3 indexed points. |
| **Graph DB (Neo4j)** | Neo4j 5.23 (`bolt://localhost:7687`) with APOC | **Active & Healthy** | Container `rag-benchmark-lab-neo4j` is online. Bolt connection verified. Real graph has 3 Document nodes, 10 Entity nodes, 1 relationship. |
| **Local LLM (Ollama)** | Ollama Llama 3.1 8B (`http://localhost:11434`) | **Unavailable / Mock Dev Mode** | Ollama binary not installed; host root partition has only **102 MB** free space, preventing 4.7 GB model download. Transparent mock dev mode with explicit labeling is required. |
| **Embedding Model** | `all-MiniLM-L6-v2` (384-dim) | **Deterministic Normalized Mock Mode** | `sentence-transformers` / PyTorch not installed due to 102 MB disk constraint; uses deterministic 384-dim L2-normalized hashing compatible with Qdrant cosine distance. |
| **Metadata DB** | SQLite 3 (`data/runtime/rag_benchmark_lab.sqlite3`) | **Active** | Contains 3 documents, 3 chunks, 10 entities, 1 relationship, 9 evaluation runs, 30 results. |

---

## 2. Identified Defects & User Concerns

### A. Dashboard Reported "qdrant (in-memory)" & "graph store (networkx)"
- **Root Cause:** When Docker Desktop was stopped during initial launch, `build_context()` in `src/rag_benchmark_lab/app.py` caught connection exceptions and silently fell back to in-memory Qdrant (`:memory:`) and `NetworkXStore`. `st.cache_resource` cached these in-memory instances.
- **Resolution:** Bind directly to `QdrantStore` and `Neo4jStore`. Implement strict health verification without silent downgrades.

### B. Graph Explorer Displayed Generic Entities ("ALL", "The", "When Ollama") & Sparse Relationships
- **Root Cause:** `_heuristic_entities` in `entity_extractor.py` captured any capitalized word using a greedy regex without stopword filtering, and defaulted every entity category to `"Person"`. `_heuristic_relationships` only connected the first two entities with at most one hardcoded match.
- **Resolution:** Overhaul `EntityRelationshipExtractor` with comprehensive stopword filtering, grammatical entity validation, domain entity typing (`Technology`, `Concept`, `Component`, `Metric`), and sentence-level semantic predicate extraction (`USES`, `EVALUATES`, `STORES`, `INDEXES`, `RETRIEVES`, `COMPARES`).

### C. Analytics Contained "Mock response" Answers Without Separation
- **Root Cause:** Historical evaluation runs in SQLite were performed when Ollama was offline, returning raw mock completion text that was stored without prominent labeling.
- **Resolution:** Tag all evaluation runs with `mock_mode: true/false`. Display prominent dev-mode indicators in the UI. Generate structured, grounded fallback responses based on retrieved evidence rather than raw mock strings.

### D. UI Resembled Generic Administration Dashboard
- **Root Cause:** Standard Streamlit widgets with plain text, no research laboratory styling, missing guided sample questions, and no interactive exploration cards.
- **Resolution:** Redesign UI with clean research theme (deep navy `#0f172a`, research blue `#1d4ed8`, crisp background `#f8fafc`, card elevation, metric badges, breadcrumbs, and interactive tabs).

---

## 3. Repair & Implementation Plan

1. **Phase 3 — Database Adapters:**
   - Enhance `QdrantStore` with point count inspection and health validation.
   - Enhance `Neo4jStore` with real counts (`count_entities`, `count_relationships`, `count_documents`), filtered node/edge queries, and schema constraints.
   - Update `HealthService` to accurately report real Qdrant, Neo4j, and Ollama statuses.

2. **Phase 4 — Document Ingestion & Dual Indexing:**
   - Upgrade entity extraction and relationship discovery.
   - Ensure parallel dual-indexing to Qdrant (vector) and Neo4j (graph) with source provenance (`document_id`, `chunk_id`, `page_number`).
   - Clean up existing stopword entities from Neo4j and SQLite and enrich with high-quality domain relations.

3. **Phase 5 — Vector, Graph, and Hybrid RAG:**
   - Vector RAG: Real Qdrant vector similarity search.
   - GraphRAG: Real Neo4j multi-hop entity traversal and source chunk linkage.
   - Hybrid RAG: Reciprocal Rank Fusion (RRF) with configurable $k=60$.
   - Query Router: Rule-based routing with confidence score and explainable rationale.

4. **Phase 6 — Generation & Grounded Citations:**
   - Formatted citations with real document names, chunk indexes, and page numbers.
   - Graph traversal path visualization in answer panels.
   - Transparent mock-mode handling when Ollama is unavailable.

5. **Phase 7 — Evaluation & Analytics:**
   - Supported metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall, Recall@K, MRR, Exact Match, Token F1, Latency.
   - Distinguish real vs. mock evaluation runs.

6. **Phase 8 — Research Laboratory UI Redesign:**
   - Redesign all 8 pages (`1_Overview`, `2_Corpus_Management`, `3_Query_Playground`, `4_Comparison`, `5_Graph_Explorer`, `6_Evaluation_Lab`, `7_Analytics`, `8_Settings`).
