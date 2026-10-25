# TESTING REPORT — RAG BENCHMARK LAB

**Date:** October 2026  
**Test Framework:** Pytest 8.4.2 + AnyIO  
**Environment:** Linux / Python 3.12.3 virtualenv  
**Overall Status:** PASSED (30 tests passing, 6 skipped external container tests)

---

## 1. Test Suite Summary

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.4.2, pluggy-1.6.0
rootdir: /home/vaishnavi/Downloads/GraphRAG vs. Dense Vector RAG
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.15.1
collected 36 items

tests/evaluation/test_benchmark_runner.py .                              [  2%]
tests/integration/test_end_to_end_workflow.py s                          [  5%]
tests/integration/test_external_services.py sss                          [ 13%]
tests/integration/test_research_architecture.py s...s                    [ 27%]
tests/unit/test_chunker.py ..                                            [ 33%]
tests/unit/test_dataset.py ...                                           [ 41%]
tests/unit/test_document_registry.py .                                   [ 44%]
tests/unit/test_document_service_isolated.py ...                         [ 52%]
tests/unit/test_entity_extractor.py ...                                  [ 61%]
tests/unit/test_metrics.py ....                                          [ 72%]
tests/unit/test_query_service.py .                                       [ 75%]
tests/unit/test_rrf_router.py ...                                        [ 83%]
tests/unit/test_settings.py ..                                           [ 88%]
tests/unit/test_user_registry.py ....                                    [100%]

======================== 30 passed, 6 skipped in 1.48s =========================
```

---

## 2. Detailed Test Coverage

### 2.1 Authentication & Multi-Tenancy (`tests/unit/test_user_registry.py`)
- `test_user_registration_and_validation`: PASS — Validates registration, PBKDF2 password hashing, email format checks, and duplicate email rejections.
- `test_user_authentication`: PASS — Validates successful login and failed login error handling.
- `test_knowledge_base_isolation`: PASS — Validates multi-user KB boundary enforcement, ensuring User B cannot see, access, or delete User A's knowledge bases.
- `test_conversation_persistence`: PASS — Validates multi-turn chat persistence, storing queries, responses, strategy badges, latency breakdowns, and citations in SQLite.

### 2.2 Ingestion & Entity Extraction (`tests/unit/test_chunker.py`, `test_entity_extractor.py`)
- Chunking splits text into configurable token windows with overlap and preserved offsets.
- Entity extraction eliminates stopwords, normalizes technical terminology, and extracts semantic relationships (`USES`, `EVALUATES`, `STORES`, `INDEXES`, `RETRIEVES`).

### 2.3 Retrieval & Hybrid Fusion (`tests/unit/test_rrf_router.py`, `test_query_service.py`)
- Reciprocal Rank Fusion (RRF) merges ranked vector and graph candidates with formula $RRF(d) = \sum \frac{1}{k + rank(d)}$.
- Query router correctly selects Vector RAG, GraphRAG, or Hybrid RAG based on relational reasoning vs semantic question classification.

### 2.4 Empirical Benchmark & Metrics (`tests/unit/test_metrics.py`, `test_benchmark_runner.py`)
- Precision, Recall, Hit Rate@K, MRR, Faithfulness, and Answer Relevancy are computed accurately with proper edge case handling for empty contexts.
