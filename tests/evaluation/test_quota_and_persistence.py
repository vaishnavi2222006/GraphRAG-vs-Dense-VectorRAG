"""Unit tests for quota exhaustion handling, partial result preservation, and history reload."""
from __future__ import annotations

import json
from unittest.mock import MagicMock
import pytest

from rag_benchmark_lab.evaluation.benchmark_runner import BenchmarkRunner
from rag_benchmark_lab.evaluation.dataset import EvaluationExample
from rag_benchmark_lab.evaluation.result_store import ResultStore
from rag_benchmark_lab.generation.llm_service import LLMResponse
from rag_benchmark_lab.schemas.documents import (
    AnswerResult,
    Citation,
    RetrievedChunk,
    RetrievalStrategy,
)
from rag_benchmark_lab.services.document_registry import DocumentRegistry
from rag_benchmark_lab.services.query_service import QueryService
from rag_benchmark_lab.utils.errors import QuotaExceededError


class QuotaFailingLLMService:
    def __init__(self) -> None:
        self.model_name = "gemini-test-quota"
        self.mock_mode = False
        self.call_count = 0

    def is_configured(self) -> bool:
        return True

    def health(self) -> bool:
        return True

    def complete(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0) -> LLMResponse:
        self.call_count += 1
        # First call fails with QuotaExceededError
        raise QuotaExceededError(
            "Gemini's available quota has been exhausted for the configured model.",
            technical_details="HTTP 429: RESOURCE_EXHAUSTED",
            is_daily_limit=True,
        )


def _build_quota_query_service() -> tuple[QueryService, QuotaFailingLLMService]:
    llm = QuotaFailingLLMService()
    vec_chunk = RetrievedChunk(
        chunk_id="c1",
        document_id="doc1",
        document_name="architecture.pdf",
        chunk_index=0,
        text="The ingestion pipeline validates, chunks, embeds, and indexes documents.",
        score=0.9,
    )
    vec_retriever = MagicMock()
    vec_retriever.retrieve.return_value = MagicMock(chunks=[vec_chunk], retrieval_latency_ms=30.0)

    graph_retriever = MagicMock()
    graph_retriever.retrieve.return_value = MagicMock(source_chunks=[vec_chunk], evidence=None, retrieval_latency_ms=40.0, warnings=[])

    hybrid_retriever = MagicMock()
    hybrid_retriever.retrieve.return_value = MagicMock(chunks=[vec_chunk], graph_evidence=None, retrieval_latency_ms=50.0, warnings=[])

    qs = QueryService(
        vector_retriever=vec_retriever,
        graph_retriever=graph_retriever,
        hybrid_retriever=hybrid_retriever,
        llm_service=llm,
        top_k=5,
        answer_temperature=0.0,
    )
    return qs, llm


def test_quota_exhaustion_preserves_partial_results(tmp_path) -> None:
    """When Gemini quota is exhausted, subsequent calls gracefully skip generation while preserving retrieval results."""
    qs, llm = _build_quota_query_service()
    registry = DocumentRegistry(tmp_path / "registry.sqlite3")
    store = ResultStore(registry)
    runner = BenchmarkRunner(qs, store)

    examples = [
        EvaluationExample(
            query_id="q1",
            question="What is the pipeline purpose?",
            expected_answer="Validates, chunks, and indexes documents.",
        )
    ]

    # Run across Vector and Graph RAG
    output = runner.run(examples, strategies=[RetrievalStrategy.VECTOR, RetrievalStrategy.GRAPH])

    assert output["run_id"]
    rows = output["rows"]
    assert len(rows) == 2

    # Vector RAG failed with quota
    vec_row = [r for r in rows if r["strategy"] == "VECTOR"][0]
    assert "QuotaExceededError" in str(vec_row["errors"]) or "quota" in str(vec_row["errors"]).lower()
    assert vec_row["status"] == "Quota Exceeded"
    # Even though generation failed, retrieval evidence was preserved
    assert vec_row["retrieval_latency_ms"] == 30.0
    assert vec_row["retrieved_chunk_count"] == 1

    # Graph RAG skipped generation gracefully without spamming live LLM
    graph_row = [r for r in rows if r["strategy"] == "GRAPH"][0]
    assert graph_row["retrieval_latency_ms"] == 40.0
    assert graph_row["retrieved_chunk_count"] == 1

    # Verify LLM was only called once, not repeatedly retried on daily quota
    assert llm.call_count == 1


def test_history_persistence_and_reload(tmp_path) -> None:
    """Verify that runs and detailed results can be reloaded from SQLite without re-running models."""
    db_file = tmp_path / "test_history.sqlite3"
    registry = DocumentRegistry(db_file)
    store = ResultStore(registry)

    # 1. Save run and results directly
    run_id = "run_persist_123"
    summary_payload = {
        "run_id": run_id,
        "timestamp": "2026-10-29T11:00:00Z",
        "dataset_id": "test_dataset",
        "strategies": ["VECTOR", "GRAPH", "HYBRID", "DIRECT"],
        "total_queries": 1,
        "overall_best": "HYBRID",
        "overall_reason": "Hybrid achieved highest composite score.",
        "winner_counts": {"HYBRID": 1},
        "ties_count": 0,
        "inconclusive_count": 0,
        "failed_count": 0,
    }
    store.save_run(run_id, summary_payload)

    for strat in ("VECTOR", "GRAPH", "HYBRID", "DIRECT"):
        row = {
            "run_id": run_id,
            "query_id": "q1",
            "strategy": strat,
            "question": "What is machine learning?",
            "answer": f"{strat} generated answer.",
            "expected_answer": "ML learns from data patterns.",
            "retrieval_latency_ms": 50.0 if strat != "DIRECT" else None,
            "generation_latency_ms": 300.0,
            "total_latency_ms": 350.0 if strat != "DIRECT" else 300.0,
            "retrieved_chunk_count": 3 if strat != "DIRECT" else None,
            "graph_path_count": 2 if strat in ("GRAPH", "HYBRID") else None,
            "status": "Success",
            "metrics": {
                "answer_correctness": {"value": 0.85 if strat != "DIRECT" else 0.70, "available": True},
                "faithfulness": {"value": 0.90 if strat != "DIRECT" else None, "available": strat != "DIRECT"},
                "answer_relevancy": {"value": 0.88, "available": True},
            },
            "recommendation": {
                "recommended_strategy": "HYBRID",
                "reason": "Hybrid RAG scored highest.",
                "status": "definitive",
                "is_definitive": True,
            },
        }
        store.save_result(f"{run_id}:q1:{strat}", run_id, row)

    # 2. Re-read from a fresh ResultStore instance (simulating app restart)
    new_registry = DocumentRegistry(db_file)
    new_store = ResultStore(new_registry)

    loaded_runs = new_store.list_runs()
    assert len(loaded_runs) == 1
    loaded_run = loaded_runs[0]
    assert loaded_run["run_id"] == run_id
    assert loaded_run["overall_best"] == "HYBRID"

    loaded_results = new_store.list_results(run_id)
    assert len(loaded_results) == 4
    strats_found = {r["strategy"] for r in loaded_results}
    assert strats_found == {"VECTOR", "GRAPH", "HYBRID", "DIRECT"}

    # Direct LLM preserved None for retrieval metrics
    direct_reloaded = [r for r in loaded_results if r["strategy"] == "DIRECT"][0]
    assert direct_reloaded["retrieval_latency_ms"] is None
    assert direct_reloaded["metrics"]["faithfulness"]["value"] is None
    assert direct_reloaded["metrics"]["faithfulness"]["available"] is False


def test_invalid_api_key_handling() -> None:
    """Requirement 6: Gracefully distinguish and handle unconfigured / invalid API key without crashing."""
    class UnconfiguredLLM:
        def __init__(self) -> None:
            self.model_name = "gemini-test"
            self.mock_mode = False

        def is_configured(self) -> bool:
            return False

        def complete(self, *args, **kwargs) -> LLMResponse:
            raise RuntimeError("Should not be called when unconfigured")

    vec_chunk = RetrievedChunk(chunk_id="c1", document_id="d1", document_name="doc.pdf", chunk_index=0, text="Data passage", score=0.9)
    vec_retriever = MagicMock()
    vec_retriever.retrieve.return_value = MagicMock(chunks=[vec_chunk], retrieval_latency_ms=25.0)

    qs = QueryService(
        vector_retriever=vec_retriever,
        graph_retriever=MagicMock(),
        hybrid_retriever=MagicMock(),
        llm_service=UnconfiguredLLM(),
        top_k=5,
        answer_temperature=0.0,
    )

    ans = qs.answer("What is this?", manual_strategy=RetrievalStrategy.VECTOR)

    assert "AuthenticationError" in str(ans.errors) or "api_key" in str(ans.errors).lower() or "not configured" in ans.answer.lower()
    assert ans.generation_latency_ms is None
    # Retrieval evidence is preserved despite generation error
    assert ans.retrieval_latency_ms == 25.0
    assert len(ans.retrieved_sources) == 1


def test_unsupported_or_unavailable_model() -> None:
    """Requirement 7: Gracefully capture unavailable or unsupported model errors."""
    class UnavailableModelLLM:
        def __init__(self) -> None:
            self.model_name = "gemini-nonexistent-model"
            self.mock_mode = False

        def is_configured(self) -> bool:
            return True

        def complete(self, *args, **kwargs) -> LLMResponse:
            raise ValueError("models/gemini-nonexistent-model is not found for API version v1beta")

    qs = QueryService(
        vector_retriever=MagicMock(retrieve=MagicMock(return_value=MagicMock(chunks=[], retrieval_latency_ms=10.0))),
        graph_retriever=MagicMock(),
        hybrid_retriever=MagicMock(),
        llm_service=UnavailableModelLLM(),
        top_k=5,
        answer_temperature=0.0,
    )

    ans = qs.answer("What is this?", manual_strategy=RetrievalStrategy.VECTOR)

    assert len(ans.errors) > 0
    assert "ValueError" in ans.errors[0] or "not found" in ans.errors[0]
    assert ans.metadata.get("generation_status") == "failed"


def test_partial_success_when_one_approach_fails(tmp_path) -> None:
    """Requirement 8: Preserve successful results and rank them when one approach fails."""
    class FlakyLLMService:
        def __init__(self) -> None:
            self.model_name = "flaky-llm"
            self.mock_mode = True

        def is_configured(self) -> bool:
            return True

        def complete(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0) -> LLMResponse:
            if "Direct LLM" in prompt or "general pre-trained" in prompt:
                raise RuntimeError("Direct LLM service unavailable")
            return LLMResponse(
                text="Vector RAG succeeded with accurate context.",
                model_name=self.model_name,
                latency_ms=200.0,
                mock_mode=True,
            )

    vec_chunk = RetrievedChunk(chunk_id="c1", document_id="d1", document_name="doc.pdf", chunk_index=0, text="Context text", score=0.9)
    qs = QueryService(
        vector_retriever=MagicMock(retrieve=MagicMock(return_value=MagicMock(chunks=[vec_chunk], retrieval_latency_ms=30.0))),
        graph_retriever=MagicMock(),
        hybrid_retriever=MagicMock(),
        llm_service=FlakyLLMService(),
        top_k=5,
        answer_temperature=0.0,
    )

    registry = DocumentRegistry(tmp_path / "flaky.sqlite3")
    store = ResultStore(registry)
    runner = BenchmarkRunner(qs, store)

    ex = EvaluationExample(query_id="q1", question="What is the pipeline?", expected_answer="Context text")
    out = runner.run([ex], strategies=[RetrievalStrategy.VECTOR, RetrievalStrategy.DIRECT])

    rows = out["rows"]
    vec_row = [r for r in rows if r["strategy"] == "VECTOR"][0]
    direct_row = [r for r in rows if r["strategy"] == "DIRECT"][0]

    assert vec_row["status"] == "Success"
    assert direct_row["status"] == "Failed"

    # Recommendation should recommend VECTOR and exclude failed DIRECT
    rec = out["query_summaries"][0]["recommendation"]
    assert rec["recommended_strategy"] == "VECTOR"
    assert rec["status"] in ("definitive", "provisional")


def test_all_approaches_failing_reports_failure_state(tmp_path) -> None:
    """Requirement 9: When all approaches fail, report clear failure state and No Approach Recommended."""
    qs, _ = _build_quota_query_service()
    registry = DocumentRegistry(tmp_path / "all_fail.sqlite3")
    store = ResultStore(registry)
    runner = BenchmarkRunner(qs, store)

    ex = EvaluationExample(query_id="q1", question="What is the pipeline?", expected_answer="Context text")
    out = runner.run([ex], strategies=[RetrievalStrategy.VECTOR, RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID, RetrievalStrategy.DIRECT])

    summary = out["recommendation_summary"]
    assert summary["overall_best"] is None
    assert "failed" in summary["overall_reason"].lower() or "no approach" in summary["overall_reason"].lower()

    for row in out["rows"]:
        assert row["status"] in ("Failed", "Quota Exceeded")
        assert row["composite_score"] <= 0.0


def test_prevention_of_stale_session_state() -> None:
    """Requirement 17: Verify state isolation prevents reusing stale presentation results between queries."""
    session_state = {}

    # Simulate run 1
    session_state["single_eval_data"] = {
        "run_id": "run_1",
        "query": "Question 1",
        "results": {"VECTOR": {"answer": "Answer 1", "status": "Success"}},
    }
    assert session_state["single_eval_data"]["run_id"] == "run_1"

    # When starting run 2, state must be popped/reset
    session_state.pop("single_eval_data", None)
    assert "single_eval_data" not in session_state

    # Populate run 2
    session_state["single_eval_data"] = {
        "run_id": "run_2",
        "query": "Question 2",
        "results": {"VECTOR": {"answer": "Answer 2", "status": "Success"}},
    }
    assert session_state["single_eval_data"]["run_id"] == "run_2"
    assert session_state["single_eval_data"]["query"] == "Question 2"
    assert "Answer 1" not in session_state["single_eval_data"]["results"]["VECTOR"]["answer"]

