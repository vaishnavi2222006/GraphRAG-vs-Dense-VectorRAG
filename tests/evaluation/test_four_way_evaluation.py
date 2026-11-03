"""Unit and integration tests for four-way evaluation and Direct LLM isolation."""
from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from rag_benchmark_lab.evaluation.benchmark_runner import BenchmarkRunner
from rag_benchmark_lab.evaluation.dataset import sample_dataset
from rag_benchmark_lab.evaluation.metrics import metric_bundle
from rag_benchmark_lab.evaluation.result_store import ResultStore
from rag_benchmark_lab.generation.llm_service import LLMResponse
from rag_benchmark_lab.schemas.documents import (
    AnswerResult,
    Citation,
    GraphEvidence,
    RelationshipRecord,
    RetrievedChunk,
    RetrievalStrategy,
)
from rag_benchmark_lab.schemas.evaluation import EvaluationExample
from rag_benchmark_lab.services.document_registry import DocumentRegistry
from rag_benchmark_lab.services.query_service import QueryService


class MockLLMService:
    def __init__(self, model_name: str = "mock-eval-llm") -> None:
        self.model_name = model_name
        self.mock_mode = True
        self.recorded_prompts: list[str] = []

    def is_configured(self) -> bool:
        return True

    def health(self) -> bool:
        return True

    def complete(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0) -> LLMResponse:
        self.recorded_prompts.append(prompt)
        # Check if direct prompt or RAG prompt
        if "general pre-trained knowledge" in prompt or "Direct LLM" in prompt or "Retrieved Context:" not in prompt:
            text = "Direct LLM response: Machine learning enables systems to learn from data patterns."
        else:
            text = "RAG response: The ingestion pipeline validates, chunks, embeds, and indexes documents."
        return LLMResponse(
            text=text,
            model_name=self.model_name,
            latency_ms=120.0,
            prompt_tokens=45,
            completion_tokens=20,
            mock_mode=True,
        )


def _build_test_query_service(mock_llm: MockLLMService | None = None) -> QueryService:
    llm = mock_llm or MockLLMService()
    vec_chunk = RetrievedChunk(
        chunk_id="chk_vec_1",
        document_id="doc_alpha",
        document_name="architecture.pdf",
        chunk_index=0,
        text="The ingestion pipeline validates, chunks, embeds, and indexes documents while preserving provenance.",
        score=0.92,
    )
    graph_rel = RelationshipRecord(
        relationship_id="rel_1",
        source_entity_id="e1",
        source_entity_name="IngestionPipeline",
        target_entity_id="e2",
        target_entity_name="VectorIndex",
        relationship_type="INDEXES_INTO",
        confidence=0.95,
        source_text="The ingestion pipeline indexes into the vector store.",
    )
    graph_evidence = GraphEvidence(entities=[], relationships=[graph_rel], source_chunks=[vec_chunk])

    vec_retriever = MagicMock()
    vec_retriever.retrieve.return_value = MagicMock(chunks=[vec_chunk], retrieval_latency_ms=45.0)

    graph_retriever = MagicMock()
    graph_retriever.retrieve.return_value = MagicMock(
        source_chunks=[vec_chunk],
        evidence=graph_evidence,
        retrieval_latency_ms=65.0,
        warnings=[],
    )

    hybrid_retriever = MagicMock()
    hybrid_retriever.retrieve.return_value = MagicMock(
        chunks=[vec_chunk],
        graph_evidence=graph_evidence,
        retrieval_latency_ms=80.0,
        warnings=[],
    )

    return QueryService(
        vector_retriever=vec_retriever,
        graph_retriever=graph_retriever,
        hybrid_retriever=hybrid_retriever,
        llm_service=llm,
        top_k=5,
        answer_temperature=0.0,
    )


def test_four_approaches_execution() -> None:
    mock_llm = MockLLMService()
    qs = _build_test_query_service(mock_llm)

    strategies = [
        RetrievalStrategy.VECTOR,
        RetrievalStrategy.GRAPH,
        RetrievalStrategy.HYBRID,
        RetrievalStrategy.DIRECT,
    ]

    results = {}
    for strat in strategies:
        res = qs.answer("What does the ingestion pipeline do?", manual_strategy=strat)
        results[strat.value] = res

    # Verify all 4 approaches executed
    assert len(results) == 4
    for s in ("VECTOR", "GRAPH", "HYBRID", "DIRECT"):
        assert s in results
        assert results[s].answer
        assert results[s].selected_strategy.value == s

    # Vector RAG checks
    vec_res = results["VECTOR"]
    assert len(vec_res.retrieved_sources) == 1
    assert vec_res.retrieval_latency_ms == 45.0
    assert vec_res.generation_latency_ms == 120.0
    assert vec_res.total_latency_ms > 0

    # GraphRAG checks
    graph_res = results["GRAPH"]
    assert graph_res.graph_evidence is not None
    assert len(graph_res.graph_evidence.relationships) == 1
    assert graph_res.retrieval_latency_ms == 65.0

    # Hybrid RAG checks
    hybrid_res = results["HYBRID"]
    assert len(hybrid_res.retrieved_sources) == 1
    assert hybrid_res.graph_evidence is not None
    assert hybrid_res.retrieval_latency_ms == 80.0

    # Direct LLM checks: strictly isolated
    direct_res = results["DIRECT"]
    assert len(direct_res.retrieved_sources) == 0
    assert direct_res.graph_evidence is None
    assert len(direct_res.citations) == 0
    assert direct_res.retrieval_latency_ms is None
    assert direct_res.generation_latency_ms == 120.0
    assert direct_res.total_latency_ms > 0


def test_direct_llm_isolation_from_corpus() -> None:
    """Ensure Direct LLM never accesses or passes corpus context to the LLM."""
    mock_llm = MockLLMService()
    qs = _build_test_query_service(mock_llm)

    ans = qs.answer("What is machine learning?", manual_strategy=RetrievalStrategy.DIRECT)

    assert ans.selected_strategy == RetrievalStrategy.DIRECT
    assert ans.retrieved_sources == []
    assert ans.citations == []
    assert ans.graph_evidence is None
    assert ans.retrieval_latency_ms is None

    # Verify the actual prompt sent to LLM contains no retrieved context block
    assert len(mock_llm.recorded_prompts) == 1
    prompt_sent = mock_llm.recorded_prompts[0]
    assert "Retrieved Context:" not in prompt_sent
    assert "architecture.pdf" not in prompt_sent
    assert "IngestionPipeline" not in prompt_sent
    assert "Do not reference, invent, or cite any external document context" in prompt_sent


def test_correct_na_handling_for_direct_llm() -> None:
    """Verify that retrieval metrics for Direct LLM are strictly None (N/A) and not zero."""
    qs = _build_test_query_service()
    ans = qs.answer("What is machine learning?", manual_strategy=RetrievalStrategy.DIRECT)

    ref_answer = "Machine learning enables systems to learn from data patterns."
    ref_docs = ["doc_alpha"]
    ref_contexts = ["Some context passage"]

    metrics = metric_bundle(ans, ref_answer, ref_docs, ref_contexts, question="What is machine learning?")

    # Quality metrics that apply to Direct LLM when reference answer is present
    assert metrics["answer_correctness"].available is True
    assert metrics["answer_correctness"].value is not None
    assert metrics["answer_correctness"].value > 0.5
    assert metrics["answer_relevancy"].available is True
    assert metrics["answer_relevancy"].value is not None
    assert metrics["completeness"].available is True
    assert metrics["completeness"].value is not None

    # Retrieval-specific metrics MUST be N/A (None), NOT zero!
    retrieval_metric_names = [
        "faithfulness",
        "citation_correctness",
        "context_precision",
        "context_recall",
        "retrieval_recall_at_k",
        "hit_rate_at_k",
        "mrr",
    ]
    for m_name in retrieval_metric_names:
        m_result = metrics[m_name]
        assert m_result.value is None, f"{m_name} should have value None (N/A) for Direct LLM, got {m_result.value}"
        assert m_result.available is False
        assert m_result.details.get("reason") == "not_applicable_for_direct_llm"


def test_latency_recording_accuracy() -> None:
    """Verify that actual non-zero latency measurements are recorded and not fabricated."""
    qs = _build_test_query_service()
    ans_vec = qs.answer("What is machine learning?", manual_strategy=RetrievalStrategy.VECTOR)
    ans_dir = qs.answer("What is machine learning?", manual_strategy=RetrievalStrategy.DIRECT)

    # Vector RAG has both retrieval and generation latency
    assert ans_vec.retrieval_latency_ms == 45.0
    assert ans_vec.generation_latency_ms == 120.0
    assert ans_vec.total_latency_ms is not None
    assert ans_vec.total_latency_ms > 0

    # Direct LLM has None for retrieval latency, real value for generation and total
    assert ans_dir.retrieval_latency_ms is None
    assert ans_dir.generation_latency_ms == 120.0
    assert ans_dir.total_latency_ms is not None
    assert ans_dir.total_latency_ms > 0


def test_failed_generation_attempt_latency_recorded() -> None:
    """Requirement 10: If generation fails, record failure status and actual elapsed time separately from zero."""
    import time

    class SlowFailingLLM:
        def __init__(self) -> None:
            self.model_name = "slow-failing"
            self.mock_mode = False

        def is_configured(self) -> bool:
            return True

        def complete(self, *args, **kwargs) -> LLMResponse:
            time.sleep(0.05)  # 50 ms delay before failing
            raise RuntimeError("API connection timeout")

    qs = QueryService(
        vector_retriever=MagicMock(retrieve=MagicMock(return_value=MagicMock(chunks=[], retrieval_latency_ms=15.0))),
        graph_retriever=MagicMock(),
        hybrid_retriever=MagicMock(),
        llm_service=SlowFailingLLM(),
        top_k=5,
        answer_temperature=0.0,
    )

    ans = qs.answer("Test question", manual_strategy=RetrievalStrategy.VECTOR)

    assert len(ans.errors) > 0
    assert ans.metadata.get("generation_status") == "failed"
    # Successful generation latency is None for failed generation (displays N/A, not 0.0)
    assert ans.generation_latency_ms is None
    # Failed attempt latency is measured and recorded separately in metadata
    assert ans.metadata.get("failed_generation_latency_ms") is not None
    assert float(ans.metadata.get("failed_generation_latency_ms")) >= 40.0
    # Total latency includes retrieval and generation attempt
    assert ans.total_latency_ms is not None
    assert ans.total_latency_ms >= float(ans.metadata["failed_generation_latency_ms"])
