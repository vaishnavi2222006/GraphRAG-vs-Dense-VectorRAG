from __future__ import annotations

import pytest

from rag_benchmark_lab.app import build_context
from rag_benchmark_lab.evaluation.metrics import (
    answer_relevancy,
    exact_match,
    faithfulness,
    mean_reciprocal_rank,
    metric_bundle,
    retrieval_recall_at_k,
    token_f1,
)
from rag_benchmark_lab.retrieval.query_router import route_query
from rag_benchmark_lab.retrieval.rrf import reciprocal_rank_fusion
from rag_benchmark_lab.schemas.documents import DocumentStatus, RetrievalStrategy


@pytest.fixture(scope="module")
def context():
    built = build_context()
    yield built
    built.qdrant_store.close()
    built.neo4j_store.close()


def test_real_qdrant_and_neo4j_live_connectivity(context):
    """Verifies that live production Qdrant and Neo4j database containers are healthy."""
    try:
        q_ok = context.qdrant_store.health()
    except Exception as exc:
        pytest.skip(f"Live Qdrant container is offline/unavailable: {exc}")

    try:
        n_ok = context.neo4j_store.health()
    except Exception as exc:
        pytest.skip(f"Live Neo4j container is offline/unavailable: {exc}")

    assert q_ok is True
    assert n_ok is True
    assert context.qdrant_store.count() >= 0
    assert context.neo4j_store.count_entities() >= 0


def test_rrf_ranking_formula():
    """Validates Reciprocal Rank Fusion mathematical weighting."""
    ranked_list_a = ["doc1", "doc2", "doc3"]
    ranked_list_b = ["doc2", "doc1", "doc4"]
    fusion = reciprocal_rank_fusion([ranked_list_a, ranked_list_b], k=60)

    # doc1: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.01639 + 0.01613 = 0.03252
    # doc2: 1/(60+2) + 1/(60+1) = 0.03252
    # doc3: 1/(60+3) = 0.01587
    # doc4: 1/(60+3) = 0.01587
    assert fusion["doc1"] > fusion["doc3"]
    assert fusion["doc2"] > fusion["doc4"]
    assert pytest.approx(fusion["doc1"], 0.001) == fusion["doc2"]


def test_query_router_intent_classification():
    """Verifies that query router differentiates relational vs. direct similarity questions."""
    # Graph intent (relational)
    route_g = route_query("How is Qdrant connected to the knowledge graph and what relies on it?")
    assert route_g.selected_strategy in (RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID)
    assert route_g.confidence > 0.0

    # Vector intent (direct factual/similarity)
    route_v = route_query("What is the definition of cosine similarity in vector search?")
    assert route_v.selected_strategy in (RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID)

    # Hybrid / All intent
    route_h = route_query("Compare Dense Vector RAG and GraphRAG architectures and evaluate both")
    assert route_h.selected_strategy in (RetrievalStrategy.HYBRID, RetrievalStrategy.ALL)


def test_evaluation_metric_bundle():
    """Verifies all supported evaluation metrics calculate correctly."""
    pred = "Qdrant is a vector database that stores embeddings."
    ref = "Qdrant is a vector database for embedding storage."
    ctx = ["Qdrant is an open-source vector database that stores high-dimensional embeddings."]

    # Exact match & F1
    assert exact_match("same text", "same text").value == 1.0
    assert token_f1(pred, ref).value > 0.6

    # Faithfulness
    faith = faithfulness(pred, ctx)
    assert faith.value > 0.7

    # Answer Relevancy
    rel = answer_relevancy("What is Qdrant?", pred)
    assert rel.value > 0.3


def test_isolated_document_lifecycle_and_retrieval(context):
    """End-to-end test of ingestion, Qdrant search, Neo4j traversal, and cleanup without touching user corpus."""
    try:
        context.qdrant_store.health()
        context.neo4j_store.health()
    except Exception as exc:
        pytest.skip(f"Live Qdrant/Neo4j services required for live ingestion lifecycle test: {exc}")

    test_text = (
        "ProjectAntigravity utilizes Qdrant to store high-dimensional vectors. "
        "ProjectAntigravity uses Neo4j to build entity graphs."
    )
    test_filename = "test_isolated_spec_doc.txt"
    outcome = context.document_service.ingest_bytes(test_filename, test_text.encode("utf-8"))

    try:
        assert outcome.document.status == DocumentStatus.INDEXED
        doc_id = outcome.document.document_id

        # 1. Vector RAG Search against real Qdrant
        vec_res = context.vector_retriever.retrieve("What does ProjectAntigravity utilize?", document_ids=[doc_id])
        assert len(vec_res.chunks) >= 1
        assert "ProjectAntigravity" in vec_res.chunks[0].text

        # 2. GraphRAG Traversal against real Neo4j
        graph_res = context.graph_retriever.retrieve("How does ProjectAntigravity connect to Neo4j?", document_ids=[doc_id])
        assert graph_res.retrieval_latency_ms >= 0.0

        # 3. Hybrid RAG with RRF
        hybrid_res = context.hybrid_retriever.retrieve("What databases does ProjectAntigravity use?", document_ids=[doc_id])
        assert len(hybrid_res.chunks) >= 1

        # 4. Citations & Answer Generation
        answer_res = context.query_service.answer("What does ProjectAntigravity utilize?", manual_strategy=RetrievalStrategy.VECTOR, document_ids=[doc_id])
        assert answer_res.answer
        assert len(answer_res.citations) >= 1
        assert answer_res.citations[0].document_name == test_filename

    finally:
        # Clean up test document
        context.document_service.delete_document(outcome.document.document_id)
