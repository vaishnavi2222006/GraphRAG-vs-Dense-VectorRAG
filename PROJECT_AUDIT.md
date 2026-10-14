# Project Audit Report

**Project Title:** GraphRAG vs. Dense Vector RAG: Empirical Evaluation and Hybrid Optimization for Domain-Specific Complex Reasoning  
**Date:** 2026-10-14  
**Audit Scope:** Full codebase static analysis, environment verification, database and vector store connectivity, ingestion pipeline, retrieval pipelines, hybrid fusion, query routing, evaluation framework, Streamlit dashboard, and end-to-end integration testing.

---

## 1. Project Inventory & Architecture

### A. Application Layer
- **Entry point:** `app.py` invoking `rag_benchmark_lab.app.main()`
- **Dashboard Framework:** Streamlit 1.65.0 with multi-page structure:
  - `pages/1_Overview.py`: System counts, service health, recent errors, architecture overview
  - `pages/2_Corpus_Management.py`: Document upload (PDF, TXT, DOCX), re-indexing, deletion, and corpus clearing
  - `pages/3_Query_Playground.py`: Interactive querying with AUTO/VECTOR/GRAPH/HYBRID strategies, latency tracking, citations, retrieved chunks, and graph evidence
  - `pages/4_Comparison.py`: Side-by-side strategy comparison across Vector, Graph, and Hybrid RAG
  - `pages/5_Graph_Explorer.py`: Knowledge graph inspection and interactive Plotly network graph visualization
  - `pages/6_Evaluation_Lab.py`: Benchmark dataset runner and evaluation history
  - `pages/7_Analytics.py`: Strategy latency distributions and metric summaries
  - `pages/8_Settings.py`: Configuration parameters and live service health diagnostics
- **State & Context Management:** `AppContext` dataclass cached via `st.cache_resource` in `rag_benchmark_lab.app.build_context()`

### B. Document Ingestion Pipeline
- **Parsers:**
  - PDF: PyMuPDF (`pymupdf`) with page-level text extraction
  - TXT: UTF-8 decoder with error replacement
  - DOCX: `python-docx`
- **Text Cleaning:** `rag_benchmark_lab.ingestion.text_cleaner.clean_text`
- **Chunking:** Sliding window chunker in `rag_benchmark_lab.ingestion.chunker.chunk_document` with configurable chunk size (default 900) and overlap (default 150)
- **Metadata & Hashing:** SHA-256 content hashes for documents and chunks, deterministic IDs, page numbers, timestamps, and upload size enforcement (default 25 MB)
- **Status Tracking:** Durable SQLite database storing document status (`pending`, `partial`, `indexed`, `failed`, `deleted`), chunk counts, entity counts, relationship counts, and error messages
- **Deduplication:** Automatic hash lookup in SQLite to skip redundant ingestion

### C. Embeddings & Vector Retrieval
- **Embedding Provider:** `EmbeddingService` in `rag_benchmark_lab.embeddings.embedding_service`:
  - Primary: `SentenceTransformer` (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions)
  - Fallback / Mock: Deterministic L2-normalized 384-dimensional pseudo-embeddings for environments without PyTorch/SentenceTransformers
- **Vector Database:** Qdrant (v1.11.4 in Docker container, client `qdrant-client` 1.19.1)
  - Collection: `rag_chunks` (Cosine distance)
  - Automatic collection creation and dimension adaptation
  - Deterministic UUID mapping for Point IDs
  - Payload storage with complete chunk metadata

### D. Knowledge Graph & GraphRAG
- **Graph Database:** Neo4j 5.23 (Community Edition with APOC plugin)
- **Connection:** Official `neo4j` Python driver (bolt://localhost:7687)
- **Schema & Constraints:**
  - Unique constraint on `Entity.entity_id`
  - Unique constraint on `Document.document_id`
  - Unique constraint on `RELATED_TO.relationship_id`
- **Entity & Relationship Extraction:** `EntityRelationshipExtractor`:
  - Structured extraction via LLM with JSON schema
  - Fallback regex and grammatical relationship extraction preserving source chunk and document provenance
- **Traversal & Provenance:**
  - Multi-hop traversal (1 to 5 hops, bounded)
  - Linking graph paths back to `RetrievedChunk` instances with source text and confidence scores

### E. Hybrid RAG & Query Routing
- **Hybrid Fusion:** Reciprocal Rank Fusion (RRF) in `rag_benchmark_lab.retrieval.rrf`:
  $$\text{RRF}(d) = \sum \frac{1}{k + \text{rank}(d)}$$
  (with configurable constant $k$, default 60)
- **Provenance Preservation:** Graph evidence mapped to source chunks and fused with vector chunks
- **Query Router:** Explainable rule-based / heuristic router in `rag_benchmark_lab.retrieval.query_router`:
  - Strategies: `VECTOR`, `GRAPH`, `HYBRID`, `ALL`
  - Confidence scoring and routing rationales
  - Graceful fallback for empty and ambiguous queries

### F. Evaluation Framework
- **Dataset Support:** JSON and JSONL datasets with questions, ground-truth expected answers, reference contexts, and relevant document IDs
- **Metrics Implemented:**
  - Exact Match (normalized)
  - Token-level F1
  - Retrieval Recall@K (flexible matching against document IDs and names)
  - Hit Rate@K
  - Mean Reciprocal Rank (MRR)
  - Context Precision
  - Context Recall
  - Retrieval and Generation Latency (Mean, Median, P95)
- **Storage:** Durable SQLite persistence in `evaluation_runs` and `evaluation_results` tables

---

## 2. Defects Identified & Repaired

| ID | Severity | Component | Root Cause | Repair Applied |
|---|---|---|---|---|
| **BUG-01** | **P0** | Environment | Virtualenv `.venv` contained hardcoded paths from original author's machine (`/home/vaibhav/...`), breaking binary execution (`python`, `streamlit`, `pytest`). | Replaced all invalid paths in `.venv/bin/*`, `.venv/pyvenv.cfg`, and editable `.pth` files with the local workspace directory. |
| **BUG-02** | **P0** | Configuration | Missing `.env` file; only `.env.example` was present. | Initialized `.env` from `.env.example` with valid defaults. |
| **BUG-03** | **P1** | Vector Store | Qdrant requires Point IDs to be unsigned integers or valid UUIDs. Passing SHA-256 64-hex strings caused HTTP 400 Bad Request during chunk upsert. | Added deterministic UUID conversion (`_to_point_id`) using `uuid.UUID` / `uuid.uuid5` while preserving original chunk IDs in payloads. |
| **BUG-04** | **P1** | Vector Store | `qdrant-client` 1.19.1 removed `client.search` in favor of `client.query_points`. | Updated `QdrantStore.search` to use `query_points` with backward compatibility check for `search`. |
| **BUG-05** | **P1** | Graph Store | Neo4j Cypher does not allow query parameters in variable-length relationship patterns (`*1..$max_hops`), causing `CypherSyntaxError`. | Formatted bounded integer literal `*1..{safe_hops}` directly into the Cypher query while parameterizing all entity IDs and limits. |
| **BUG-06** | **P1** | Hybrid RAG | `HybridRetriever` computed RRF scores and sorted chunks, but then called `_dedupe_chunks()` which re-sorted them by `chunk.score`, completely overwriting RRF ranking. | Removed redundant re-sorting in `HybridRetriever`; chunks are already deduplicated in `combined` and sorted strictly by RRF score. |
| **BUG-07** | **P2** | Data Flow | `HybridRetriever` assigned `graph_evidence=graph_result` (`GraphRetrievalResult`) instead of `graph_result.evidence` (`GraphEvidence`). | Corrected assignment to `graph_result.evidence`. |
| **BUG-08** | **P2** | Ingestion | Heuristic fallback entity/relationship extraction omitted `document_id`, `chunk_id`, and `page_number`, breaking source provenance. | Passed provenance parameters to `_heuristic_entities` and `_heuristic_relationships`. |
| **BUG-09** | **P2** | UI (Streamlit) | Slotted dataclasses (`Citation`, `RetrievedChunk`) do not have `__dict__`. Calling `citation.__dict__` in `3_Query_Playground.py` caused an `AttributeError` crash on query execution. | Replaced `.__dict__` with `dataclasses.asdict()` and added empty citation safeguards. |
| **BUG-10** | **P2** | UI (Streamlit) | Direct float formatting (`:.2f`) on `result.total_latency_ms` and `result.generation_latency_ms` crashed when mock mode returned `None`. | Populated `latency_ms` in mock LLM responses and added safe `None` checking in UI pages. |
| **BUG-11** | **P2** | Data Integrity | Neo4j document deletion only deleted `(d:Document)` nodes, leaving orphaned `Entity` and `RELATED_TO` graph records. | Updated `delete_document` to delete connected relationships and entities belonging to the document. |
| **BUG-12** | **P2** | Safety | "Clear corpus" in `pages/2_Corpus_Management.py` had no confirmation step and only wiped SQLite without cleaning Qdrant and Neo4j. | Added confirmation checkbox and updated `document_service.clear_corpus()` to clean Qdrant and Neo4j. |
| **BUG-13** | **P3** | Ingestion | PyMuPDF `import fitz` raised deprecation warnings on every load. | Preferred `import pymupdf as fitz` to eliminate deprecation warnings. |
| **BUG-14** | **P2** | Evaluation | Retrieval metrics (`recall@k`, `hit_rate`, `MRR`) checked only raw `document_id`, failing when datasets referenced document filenames or slugs. | Implemented normalized document matching across IDs, filenames, and slugs. |
| **BUG-15** | **P3** | Evaluation | `BenchmarkRunner` aggregates only tracked latency and omitted metric averages. | Added aggregated mean metric values to benchmark summaries. |
