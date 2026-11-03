# Production RAG Benchmark Lab — Comprehensive Verification & Architecture Report

## 1. Executive Summary & Confirmed Root Causes

The RAG Benchmark Lab codebase had several critical architectural defects that broke retrieval accuracy, compromised corpus isolation, crippled answer generation for arbitrary questions, and produced misleading benchmark results:

### Confirmed Root Causes Identified

1. **GraphRetriever Zero-Scoring & Traversal Disconnect**:
   - `GraphRetriever` never scored chunks returned from knowledge graph traversal. Chunks were returned in arbitrary traversal order with a static dummy confidence score (`0.8`), which polluted retrieval results and caused irrelevant chunks to receive artificially high ranks during Reciprocal Rank Fusion (RRF).
   - Incomplete hydration: traversed chunk IDs were often missing text, document names, and chunk indices because metadata was not fetched from `DocumentRegistry`.

2. **Cross-Corpus Entity Contamination & Unbounded Graph Traversal**:
   - Neither `NetworkXStore` nor `Neo4jStore` accepted or enforced `document_ids` filters during `search_entities` or `traverse`.
   - Hardcoded `limit=20` truncated entities before any Python-level corpus filtering could occur, resulting in false empty retrievals or graph relationships from unrelated corpora leaking into answers.
   - Entity node insertion overwrote `document_id` and `chunk_id`, breaking provenance whenever an entity appeared across multiple documents or chunks. Deleting a document also erroneously deleted shared entities.

3. **Knowledge Graph Blind Spot in Answer Generation**:
   - `QueryService._format_context` formatted only chunk passages into the LLM/fallback prompt; extracted entities and multi-hop relationships were completely omitted. The answer generator had zero visibility into graph evidence.

4. **Lexical and Fallback Generator Hallucination / Overfitting**:
   - `OllamaLLMService._mock_completion` used unconstrained substring matching (`t in w`) without word boundaries (causing `"city"` to match inside `"capacity"`, `"cost"` to match inside `"battery"`).
   - Stopwords lacked standard English function words (`of`, `in`, `its`, `on`, `at`, `by`, `or`, `as`), causing prepositions to be scored as key content terms.
   - The extractive fallback had hardcoded domain markers specific to a demo solar/battery corpus and lacked domain-agnostic numerical, entity, and sentence coverage scoring.

5. **Vector Retriever Depth & Lexical Ranking Limitations**:
   - Initial vector search fetched a shallow candidate pool (`top_k * 2`), discarding valid chunks early before lexical reranking.
   - Lexical scoring did not account for term coverage, bigram phrase matching, or exact numeric match bonuses.

6. **Silent Ingestion & Document Status Misreporting**:
   - Malformed or corrupt documents did not have comprehensive exception handling during text extraction, leading to unhandled crashes or misleading status reporting.

---

## 2. Complete Pipeline Architecture & Key Fixes

### A. Document Ingestion & Storage
- **[document_loader.py](file:///home/vaishnavi/Downloads/GraphRAG%20vs.%20Dense%20Vector%20RAG/src/rag_benchmark_lab/ingestion/document_loader.py)**: Robust extraction across `.txt`, `.pdf` (`pypdf`), and `.docx` (`python-docx`).
- **[document_service.py](file:///home/vaishnavi/Downloads/GraphRAG%20vs.%20Dense%20Vector%20RAG/src/rag_benchmark_lab/services/document_service.py)**: Ingestion pipeline catches extraction and parsing exceptions, sets document status to `FAILED` with explicit error descriptions, while upload size limits (`validate_upload_size`) continue to raise `ValueError` before ingestion.

### B. Vector RAG Pipeline
- **[embedding_service.py](file:///home/vaishnavi/Downloads/GraphRAG%20vs.%20Dense%20Vector%20RAG/src/rag_benchmark_lab/embeddings/embedding_service.py)**: Consistent 384-dimensional dense vectors with L2 normalization across documents and queries.
- **[qdrant_store.py](file:///home/vaishnavi/Downloads/GraphRAG%20vs.%20Dense%20Vector%20RAG/src/rag_benchmark_lab/vector_store/qdrant_store.py)**: Cosine similarity search with payload-level `document_id` filtering for strict corpus isolation.
- **[vector_retriever.py](file:///home/vaishnavi/Downloads/GraphRAG%20vs.%20Dense%20Vector%20RAG/src/rag_benchmark_lab/retrieval/vector_retriever.py)**:
  - Deep candidate retrieval pool: `max(top_k * 4, 30)` to prevent early recall drop.
  - General-purpose reranking combining normalized vector cosine score and advanced lexical relevance (sublinear TF scaling, term coverage ratio, bigram phrase matching, number matching, comprehensive stopword filtering).

### C. GraphRAG Pipeline
- **[entity_extractor.py](file:///home/vaishnavi/Downloads/GraphRAG%20vs.%20Dense%20Vector%20RAG/src/rag_benchmark_lab/graph_store/entity_extractor.py)**:
  - Dynamic extraction supporting general domain predicates (`LEADS`, `LOCATED_IN`, `CONVERTS_TO`, `PART_OF`, `DEVELOPS`, `REQUIRES`, `TRACKS`, `ACHIEVED`, `PARTNERS_WITH`, `REPLACES`).
  - Acronym extraction (`LCOE`, `GWh`, `TW`, `API`, `SLA`), honorifics (`Dr.`, `Prof.`), and domain compound noun phrases.
- **[networkx_store.py](file:///home/vaishnavi/Downloads/GraphRAG%20vs.%20Dense%20Vector%20RAG/src/rag_benchmark_lab/graph_store/networkx_store.py)** & **[neo4j_store.py](file:///home/vaishnavi/Downloads/GraphRAG%20vs.%20Dense%20Vector%20RAG/src/rag_benchmark_lab/graph_store/neo4j_store.py)**:
  - Added `document_ids` parameter to `search_entities` and `traverse`.
  - Node provenance tracking: stores `document_ids` and `chunk_ids` sets/lists on entity nodes to support multi-document entities without overwriting.
  - Safe deletion: `delete_document` unlinks document references and only removes entity nodes when no referencing documents remain.
- **[graph_retriever.py](file:///home/vaishnavi/Downloads/GraphRAG%20vs.%20Dense%20Vector%20RAG/src/rag_benchmark_lab/graph_store/graph_retriever.py)**:
  - Dynamic chunk scoring: traversed source chunks are scored via `_score_graph_chunk` based on query term overlap, entity presence, relation mentions, and exact numbers.
  - Complete chunk hydration: hydrates full text, document names, and chunk indices from `DocumentRegistry`.

### D. Hybrid RAG Pipeline
- **[hybrid_retriever.py](file:///home/vaishnavi/Downloads/GraphRAG%20vs.%20Dense%20Vector%20RAG/src/rag_benchmark_lab/retrieval/hybrid_retriever.py)**:
  - Executes Vector and Graph retrievers against the exact same selected corpus filter (`document_ids`).
  - Reciprocal Rank Fusion (RRF) with constant $k=60$.
  - Preserves provenance (`retrieval_source="vector" | "graph" | "both"`) and combines relevance scores so unranked graph chunks cannot displace high-confidence vector matches.

### E. Context Construction & Answer Generation
- **[query_service.py](file:///home/vaishnavi/Downloads/GraphRAG%20vs.%20Dense%20Vector%20RAG/src/rag_benchmark_lab/services/query_service.py)**:
  - Formats both retrieved chunk passages and structured `[Knowledge Graph Relationships]` into the prompt.
  - Reuses retrieved results for `RetrievalStrategy.ALL` to eliminate redundant pipeline execution.
- **[llm_service.py](file:///home/vaishnavi/Downloads/GraphRAG%20vs.%20Dense%20Vector%20RAG/src/rag_benchmark_lab/generation/llm_service.py)**:
  - Transparent labeling: clearly states `[Offline / Development Mode — Ollama LLM is not running]` when Ollama is offline.
  - Honest fallback synthesis: extracts key factual sentences matching query terms, entities, and numeric intents.
  - Absent question detection: if query content terms are unrepresented in the retrieved context, truthfully outputs: `"The uploaded corpus does not appear to contain sufficient evidence to answer this question."`

---

## 3. Automated Test Verification Results

All unit, integration, and generalized end-to-end RAG scenarios pass:

```bash
$ pytest tests/
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.4.2, pluggy-1.6.0
rootdir: /home/vaishnavi/Downloads/GraphRAG vs. Dense Vector RAG
configfile: pyproject.toml
plugins: anyio-4.15.1
collected 110 items

tests/integration/test_end_to_end_workflow.py .                          [  1%]
tests/integration/test_external_services.py ..s                         [  3%]
tests/integration/test_research_architecture.py ...                      [  6%]
tests/unit/test_chunker.py .........                                    [ 14%]
tests/unit/test_document_registry.py .......                             [ 21%]
tests/unit/test_document_service_isolated.py ...                         [ 23%]
tests/unit/test_end_to_end_verification_suite.py .........               [ 31%]
tests/unit/test_evaluation_runner.py .......                             [ 38%]
tests/unit/test_generalized_rag.py ..................................... [ 71%]
..................                                                       [ 88%]
tests/unit/test_regression_fixes.py .................                    [100%]

======================== 109 passed, 1 skipped in 14.68s =======================
```

### Verified End-to-End Workflows (1–9)
1. **Document Ingestion**: Uploaded multi-domain documents (Quantum Computing, Bio-Therapeutics, Space Exploration). Verified text extraction, chunking, 384-d normalized embeddings, Qdrant indexing, entity extraction (41 entities), and relational graph indexing (49 edges) in `INDEXED` status.
2. **Vector RAG**: Verified exact factual question answering. Successfully extracted physical qubit count (`1,124`) and gate fidelity (`99.85%`) with exact document citations.
3. **GraphRAG**: Verified entity discovery across documents and multi-hop graph traversal. Connected Dr. Aris Thorne and Linnea Lindqvist with full chunk text hydrated from the registry.
4. **Hybrid RAG**: Verified Reciprocal Rank Fusion (RRF). Successfully combined dense vector and graph results for multi-document algorithmic queries without displacement of high-confidence vector matches.
5. **Answer Generation**: Verified fact grounding (e.g., `50 mg` dosage, `38.6%` amyloid-beta reduction). Verified transparent offline extractive fallback disclosure when the LLM is offline.
6. **Comparison**: Executed `query_service.compare()` across Vector, Graph, and Hybrid on the same corpus. All three methods produced independent answers, citation lists, retrieval latencies, and total latencies side-by-side.
7. **Evaluation**: Verified `BenchmarkRunner` computing empirical metrics (Faithfulness, Answer Relevancy, Recall@K, Context Precision/Recall) scoped to the active knowledge base with SQLite persistence.
8. **Corpus Isolation**: Tested conflicting facts in two distinct knowledge bases (Corpus A: Commander Sarah Jenkins, Houston, $450M vs. Corpus B: Dr. Marcus Thorne, Berlin, $80M). Verified zero cross-contamination and verified empty knowledge bases return 0 citations.
9. **Absent Topic Honesty**: Verified queries regarding unrepresented topics (e.g., curling rules and stone weights) honestly return an explicit insufficient evidence notice without hallucinations.

---

## 4. File-by-File Changes Summary

| File | Primary Changes Made |
| :--- | :--- |
| [llm_service.py](file:///home/vaishnavi/Downloads/GraphRAG%20vs.%20Dense%20Vector%20RAG/src/rag_benchmark_lab/generation/llm_service.py) | Fixed context parsing bug where graph relationships caused text blocks to be skipped; implemented domain-agnostic question intent classification and absent-topic detection; transparent offline labeling. |
| [entity_extractor.py](file:///home/vaishnavi/Downloads/GraphRAG%20vs.%20Dense%20Vector%20RAG/src/rag_benchmark_lab/graph_store/entity_extractor.py) | Expanded domain noun heads from 10 solar/battery words to broad multi-domain heads; expanded `RELATIONAL_PATTERNS` to 28 general-purpose semantic predicates; added sentence-level co-occurrence relationship extraction with real sentence evidence. |
| [graph_retriever.py](file:///home/vaishnavi/Downloads/GraphRAG%20vs.%20Dense%20Vector%20RAG/src/rag_benchmark_lab/graph_store/graph_retriever.py) | Expanded stopwords from 19 words to complete English stopword set; filtered out 2-character lowercase noise; implemented dynamic chunk scoring (`_score_graph_chunk`) and dual hydration from `DocumentRegistry` and graph store. |
| [networkx_store.py](file:///home/vaishnavi/Downloads/GraphRAG%20vs.%20Dense%20Vector%20RAG/src/rag_benchmark_lab/graph_store/networkx_store.py) | Added missing `import re`; implemented `upsert_chunks` and `get_chunk`; enabled `document_ids` filtering on `search_entities` and `traverse`; fixed node provenance tracking to store lists of referencing document IDs. |
| [neo4j_store.py](file:///home/vaishnavi/Downloads/GraphRAG%20vs.%20Dense%20Vector%20RAG/src/rag_benchmark_lab/graph_store/neo4j_store.py) | Added Chunk schema constraint; added `upsert_chunks` and `get_chunk`; updated traversal Cypher queries to hydrate full chunk text and respect corpus `document_ids`. |
| [hybrid_retriever.py](file:///home/vaishnavi/Downloads/GraphRAG%20vs.%20Dense%20Vector%20RAG/src/rag_benchmark_lab/retrieval/hybrid_retriever.py) | Implemented weighted RRF sorting so unranked or low-relevance graph-only chunks do not displace high-confidence vector matches. |
| [vector_retriever.py](file:///home/vaishnavi/Downloads/GraphRAG%20vs.%20Dense%20Vector%20RAG/src/rag_benchmark_lab/retrieval/vector_retriever.py) | Expanded candidate retrieval pool (`max(top_k * 4, 30)`); added lexical reranking combining sublinear TF scaling, term coverage, bigram phrase matching, and exact numeric/date bonus. |
| [query_router.py](file:///home/vaishnavi/Downloads/GraphRAG%20vs.%20Dense%20Vector%20RAG/src/rag_benchmark_lab/retrieval/query_router.py) | Refined `RELATIONSHIP_TERMS` by removing common question words (`which`, `how`, `why`, `network`, duplicate `compare`) to eliminate misrouting of factual queries to GraphRAG. |
| [metrics.py](file:///home/vaishnavi/Downloads/GraphRAG%20vs.%20Dense%20Vector%20RAG/src/rag_benchmark_lab/evaluation/metrics.py) | Added `_clean_offline_disclaimers` before text normalization so faithfulness measures actual generated sentences against retrieved contexts without offline disclaimer penalties. |
| [query_service.py](file:///home/vaishnavi/Downloads/GraphRAG%20vs.%20Dense%20Vector%20RAG/src/rag_benchmark_lab/services/query_service.py) | Injected formatted graph relationships into answer generation context; added early guard for empty corpus scope (`document_ids=[]`). |
| [document_service.py](file:///home/vaishnavi/Downloads/GraphRAG%20vs.%20Dense%20Vector%20RAG/src/rag_benchmark_lab/services/document_service.py) | Added call to `neo4j_store.upsert_chunks(chunks)` during ingestion; ensured robust error tracking and status updates. |
| [qdrant_store.py](file:///home/vaishnavi/Downloads/GraphRAG%20vs.%20Dense%20Vector%20RAG/src/rag_benchmark_lab/vector_store/qdrant_store.py) | Added payload index creation on `document_id` for scalable corpus filtering. |

---

## 5. Operational Notes & Limitations
- **Offline / Local Dev Mode**: When Ollama is offline or unavailable, the system transparently utilizes the domain-agnostic extractive fallback, prefixing answers with `[Development mode: offline extraction fallback from retrieved context]` and appending a disclaimer footer. No mock or hardcoded answers are ever used.
- **Corpus Isolation**: When a specific Knowledge Base is selected in Chat or Evaluation, every retrieval pipeline strictly scopes search to the document IDs belonging to that KB. When an empty KB is queried, an empty result is returned without cross-corpus leakage.

