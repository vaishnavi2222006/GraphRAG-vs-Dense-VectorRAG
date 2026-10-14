from __future__ import annotations

import pytest

from rag_benchmark_lab.app import build_context
from rag_benchmark_lab.evaluation.benchmark_runner import BenchmarkRunner
from rag_benchmark_lab.evaluation.dataset import sample_dataset
from rag_benchmark_lab.schemas.documents import DocumentStatus, RetrievalStrategy


@pytest.fixture(scope="module")
def context():
    built = build_context()
    yield built
    built.qdrant_store.close()
    built.neo4j_store.close()


def test_end_to_end_ingestion_and_retrieval_flow(context) -> None:
    # 1. Ingest test document
    doc_text = "Vector embeddings capture semantic similarity. Graph structures capture entities and relational dependencies between concepts. GraphRAG connects entities."
    outcome = context.document_service.ingest_bytes("test_e2e_doc.txt", doc_text.encode("utf-8"))
    try:
        assert outcome.document.status == DocumentStatus.INDEXED
        assert outcome.document.chunk_count >= 1
        assert outcome.entity_count >= 1

        # 2. Vector RAG query
        vector_res = context.query_service.answer("What captures semantic similarity?", manual_strategy=RetrievalStrategy.VECTOR)
        assert vector_res.answer
        assert vector_res.retrieved_sources
        assert vector_res.citations
        assert vector_res.total_latency_ms is not None

        # 3. GraphRAG query
        graph_res = context.query_service.answer("Which concepts connect entities?", manual_strategy=RetrievalStrategy.GRAPH)
        assert graph_res.answer
        assert graph_res.selected_strategy == RetrievalStrategy.GRAPH

        # 4. Hybrid RAG query
        hybrid_res = context.query_service.answer("Explain how Vector embeddings and Graph structures relate", manual_strategy=RetrievalStrategy.HYBRID)
        assert hybrid_res.answer
        assert hybrid_res.selected_strategy == RetrievalStrategy.HYBRID
        assert hybrid_res.retrieved_sources

        # 5. Query Router
        decision_vector = context.query_service.answer("What is semantic similarity?")
        assert decision_vector.selected_strategy in {RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID}

        # 6. Evaluation benchmark
        runner = BenchmarkRunner(context.query_service, context.result_store)
        bench_result = runner.run(sample_dataset()[:1], strategies=[RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID])
        assert bench_result["run_id"]
        assert "VECTOR" in bench_result["aggregates"]

    finally:
        context.document_service.delete_document(outcome.document.document_id)
