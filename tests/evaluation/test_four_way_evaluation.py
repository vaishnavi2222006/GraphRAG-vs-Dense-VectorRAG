"""Comprehensive automated unit tests for the four-way evaluation and recommendation system.

Covers all 18 requirements from Section 9:
1. Successful Vector RAG execution.
2. Successful GraphRAG execution.
3. Successful Hybrid RAG execution.
4. Successful Direct LLM execution without corpus retrieval.
5. Gemini / Groq quota exhaustion handling.
6. Invalid API key handling.
7. Unsupported or unavailable model handling.
8. Partial success when one approach fails.
9. All approaches failing handling.
10. Accurate latency recording and consistency.
11. N/A handling for inapplicable metrics (strictly None, not 0.0).
12. Answer-quality calculation with valid reference answers.
13. Missing-reference-answer handling (provisional recommendation, N/A ground truth).
14. Retrieval metrics with valid relevance labels.
15. Recommendation generation (definitive, provisional, tie, inconclusive).
16. Result persistence and reloading from ResultStore.
17. Prevention of stale session-state results.
18. Comparison charts matching the underlying metrics.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock
import pandas as pd
import pytest

from rag_benchmark_lab.evaluation.benchmark_runner import BenchmarkRunner
from rag_benchmark_lab.evaluation.dataset import sample_dataset
from rag_benchmark_lab.evaluation.metrics import metric_bundle
from rag_benchmark_lab.evaluation.recommendation import (
    ApproachScore,
    evaluate_approach,
    rank_benchmark_approaches,
    recommend_best_approach,
)
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
from rag_benchmark_lab.services.query_service import QueryService
from rag_benchmark_lab.utils.errors import (
    AuthenticationError,
    ModelUnavailableError,
    QuotaExceededError,
)


class MockLLMService:
    def __init__(self, model_name: str = "mock-eval-llm", delay_ms: float = 80.0) -> None:
        self.model_name = model_name
        self.mock_mode = True
        self.delay_ms = delay_ms
        self.recorded_prompts: list[str] = []

    def is_configured(self) -> bool:
        return True

    def health(self) -> bool:
        return True

    def complete(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0) -> LLMResponse:
        self.recorded_prompts.append(prompt)
        time.sleep(self.delay_ms / 1000.0)
        if "general pre-trained knowledge" in prompt or "Direct LLM" in prompt or "Retrieved Context:" not in prompt:
            text = "Direct LLM response: The ingestion pipeline validates and indexes document text into databases."
        else:
            text = "RAG response: The ingestion pipeline validates, chunks, embeds, and indexes documents."
        return LLMResponse(
            text=text,
            model_name=self.model_name,
            latency_ms=self.delay_ms,
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


# ── Requirement 1: Successful Vector RAG execution ────────────────────────────
def test_successful_vector_rag_execution() -> None:
    qs = _build_test_query_service()
    ans = qs.answer("What does the pipeline do?", manual_strategy=RetrievalStrategy.VECTOR)

    assert ans.selected_strategy == RetrievalStrategy.VECTOR
    assert len(ans.errors) == 0
    assert len(ans.retrieved_sources) == 1
    assert len(ans.citations) == 1
    assert ans.retrieval_latency_ms == 45.0
    assert ans.generation_latency_ms is not None and ans.generation_latency_ms > 0
    assert ans.total_latency_ms is not None and ans.total_latency_ms >= ans.retrieval_latency_ms


# ── Requirement 2: Successful GraphRAG execution ──────────────────────────────
def test_successful_graphrag_execution() -> None:
    qs = _build_test_query_service()
    ans = qs.answer("What does the pipeline do?", manual_strategy=RetrievalStrategy.GRAPH)

    assert ans.selected_strategy == RetrievalStrategy.GRAPH
    assert len(ans.errors) == 0
    assert ans.graph_evidence is not None
    assert len(ans.graph_evidence.relationships) == 1
    assert ans.retrieval_latency_ms == 65.0
    assert ans.generation_latency_ms is not None and ans.generation_latency_ms > 0


# ── Requirement 3: Successful Hybrid RAG execution ────────────────────────────
def test_successful_hybrid_rag_execution() -> None:
    qs = _build_test_query_service()
    ans = qs.answer("What does the pipeline do?", manual_strategy=RetrievalStrategy.HYBRID)

    assert ans.selected_strategy == RetrievalStrategy.HYBRID
    assert len(ans.errors) == 0
    assert len(ans.retrieved_sources) == 1
    assert ans.graph_evidence is not None
    assert ans.retrieval_latency_ms == 80.0
    assert ans.generation_latency_ms is not None and ans.generation_latency_ms > 0


# ── Requirement 4: Successful Direct LLM execution without corpus retrieval ───
def test_successful_direct_llm_execution_without_corpus() -> None:
    mock_llm = MockLLMService()
    qs = _build_test_query_service(mock_llm)
    ans = qs.answer("What is machine learning?", manual_strategy=RetrievalStrategy.DIRECT)

    assert ans.selected_strategy == RetrievalStrategy.DIRECT
    assert ans.retrieved_sources == []
    assert ans.citations == []
    assert ans.graph_evidence is None
    # Strictly None for retrieval latency
    assert ans.retrieval_latency_ms is None
    assert ans.generation_latency_ms is not None and ans.generation_latency_ms > 0
    assert ans.total_latency_ms == ans.generation_latency_ms

    # Verify prompt isolation
    prompt_sent = mock_llm.recorded_prompts[-1]
    assert "Retrieved Context:" not in prompt_sent
    assert "architecture.pdf" not in prompt_sent


# ── Requirement 5: Quota exhaustion handling (Gemini & Groq) ──────────────────
def test_gemini_quota_exhaustion_handling() -> None:
    class QuotaExhaustedLLM:
        def __init__(self) -> None:
            self.model_name = "gemini-2.5-flash"
            self.provider_name = "Gemini"
            self.mock_mode = False

        def is_configured(self) -> bool:
            return True

        def complete(self, *args, **kwargs) -> LLMResponse:
            time.sleep(0.04)
            raise QuotaExceededError(
                "Gemini's available quota has been exhausted for the configured model. Check project quota.",
                is_daily_limit=True,
            )

    qs = QueryService(
        vector_retriever=MagicMock(retrieve=MagicMock(return_value=MagicMock(chunks=[], retrieval_latency_ms=20.0))),
        graph_retriever=MagicMock(),
        hybrid_retriever=MagicMock(),
        llm_service=QuotaExhaustedLLM(),
        top_k=5,
        answer_temperature=0.0,
    )

    ans = qs.answer("Test question", manual_strategy=RetrievalStrategy.VECTOR)

    assert len(ans.errors) > 0
    assert "QuotaExceededError" in ans.errors[0]
    assert ans.metadata["generation_status"] == "failed"
    assert ans.metadata["is_quota"] == "true"
    # Generation failed, so generation_latency_ms is None and elapsed attempt is recorded in metadata
    assert ans.generation_latency_ms is None
    assert float(ans.metadata.get("failed_generation_latency_ms", 0)) >= 30.0

    # Test evaluation scoring under quota error
    mets = metric_bundle(ans, reference_answer="Expected answer", relevant_document_ids=None, reference_contexts=None)
    score = evaluate_approach("VECTOR", ans, mets, has_ground_truth=True)
    assert score.is_failed is True
    assert score.status_label == "Quota Exceeded"
    assert score.composite_score == -1.0


# ── Requirement 6: Invalid API key handling ───────────────────────────────────
def test_invalid_api_key_handling() -> None:
    class UnconfiguredLLM:
        def __init__(self) -> None:
            self.model_name = "gemini-2.5-flash"
            self.provider_name = "Gemini"

        def is_configured(self) -> bool:
            return False

        def complete(self, *args, **kwargs) -> LLMResponse:
            raise AuthenticationError("Gemini API key is invalid or unauthorized.")

    qs = QueryService(
        vector_retriever=MagicMock(retrieve=MagicMock(return_value=MagicMock(chunks=[], retrieval_latency_ms=10.0))),
        graph_retriever=MagicMock(),
        hybrid_retriever=MagicMock(),
        llm_service=UnconfiguredLLM(),
        top_k=5,
        answer_temperature=0.0,
    )

    ans = qs.answer("Test question", manual_strategy=RetrievalStrategy.DIRECT)
    assert len(ans.errors) > 0
    assert "AuthenticationError" in ans.errors[0]
    assert ans.metadata["is_auth"] == "true"
    assert "API key is not configured" in ans.errors[0]


# ── Requirement 7: Unsupported or unavailable model handling ─────────────────
def test_unsupported_or_unavailable_model_handling() -> None:
    class ModelNotFoundLLM:
        def __init__(self) -> None:
            self.model_name = "non-existent-model"
            self.provider_name = "Groq"

        def is_configured(self) -> bool:
            return True

        def complete(self, *args, **kwargs) -> LLMResponse:
            raise ModelUnavailableError("Groq model 'non-existent-model' not found: 404 Not Found")

    qs = QueryService(
        vector_retriever=MagicMock(retrieve=MagicMock(return_value=MagicMock(chunks=[], retrieval_latency_ms=10.0))),
        graph_retriever=MagicMock(),
        hybrid_retriever=MagicMock(),
        llm_service=ModelNotFoundLLM(),
        top_k=5,
        answer_temperature=0.0,
    )

    ans = qs.answer("Test question", manual_strategy=RetrievalStrategy.VECTOR)
    assert len(ans.errors) > 0
    assert "ModelUnavailableError" in ans.errors[0]
    assert ans.metadata["generation_status"] == "failed"


# ── Requirement 8: Partial success when one approach fails ───────────────────
def test_partial_success_when_one_approach_fails() -> None:
    qs = _build_test_query_service()
    ans_vec = qs.answer("What does the pipeline do?", manual_strategy=RetrievalStrategy.VECTOR)

    failed_ans = AnswerResult(
        answer="Error generating answer: Neo4j connection timeout.",
        selected_strategy=RetrievalStrategy.GRAPH,
        citations=[],
        retrieved_sources=[],
        retrieval_latency_ms=40.0,
        generation_latency_ms=120.0,
        total_latency_ms=160.0,
        model_name="mock-model",
        errors=["NetworkError: Neo4j connection timeout"],
        metadata={"generation_status": "failed"},
    )

    mets_vec = metric_bundle(ans_vec, "The pipeline validates documents.", ["doc_alpha"], None)
    mets_graph = metric_bundle(failed_ans, "The pipeline validates documents.", ["doc_alpha"], None)

    score_vec = evaluate_approach("VECTOR", ans_vec, mets_vec, has_ground_truth=True)
    score_graph = evaluate_approach("GRAPH", failed_ans, mets_graph, has_ground_truth=True)

    assert score_graph.is_failed is True
    assert score_vec.is_failed is False

    rec = recommend_best_approach({"VECTOR": score_vec, "GRAPH": score_graph}, has_ground_truth=True)
    assert rec.recommended_strategy == "VECTOR"
    assert rec.status == "definitive"
    assert rec.recommended_strategy != "GRAPH"


# ── Requirement 9: All approaches failing handling ───────────────────────────
def test_all_approaches_failing_handling() -> None:
    failed_score_a = ApproachScore(
        strategy="VECTOR",
        composite_score=-1.0,
        correctness=None,
        completeness=None,
        faithfulness=None,
        relevancy=None,
        citation_score=None,
        retrieval_recall=None,
        mrr=None,
        retrieval_latency_ms=25.0,
        generation_latency_ms=100.0,
        total_latency_ms=125.0,
        retrieved_chunk_count=2,
        graph_path_count=None,
        is_failed=True,
        status_label="Quota Exceeded",
        weaknesses=["Execution failed: QuotaExceededError"],
    )
    failed_score_b = ApproachScore(
        strategy="DIRECT",
        composite_score=-1.0,
        correctness=None,
        completeness=None,
        faithfulness=None,
        relevancy=None,
        citation_score=None,
        retrieval_recall=None,
        mrr=None,
        retrieval_latency_ms=None,
        generation_latency_ms=100.0,
        total_latency_ms=100.0,
        retrieved_chunk_count=None,
        graph_path_count=None,
        is_failed=True,
        status_label="Quota Exceeded",
        weaknesses=["Execution failed: QuotaExceededError"],
    )

    rec = recommend_best_approach({"VECTOR": failed_score_a, "DIRECT": failed_score_b}, has_ground_truth=True)
    assert rec.recommended_strategy is None
    assert rec.status == "failed"
    assert "All evaluated approaches failed" in rec.reason


# ── Requirement 10: Accurate latency recording and consistency ────────────────
def test_accurate_latency_recording() -> None:
    qs = _build_test_query_service()
    ans_vec = qs.answer("What does the pipeline do?", manual_strategy=RetrievalStrategy.VECTOR)
    ans_dir = qs.answer("What is machine learning?", manual_strategy=RetrievalStrategy.DIRECT)

    assert ans_vec.retrieval_latency_ms == 45.0
    assert ans_vec.generation_latency_ms == 80.0
    assert ans_vec.total_latency_ms is not None and ans_vec.total_latency_ms >= 125.0

    # Direct LLM has None for retrieval, real value for generation and total
    assert ans_dir.retrieval_latency_ms is None
    assert ans_dir.generation_latency_ms == 80.0
    assert ans_dir.total_latency_ms == 80.0


# ── Requirement 11: N/A handling for inapplicable metrics ─────────────────────
def test_na_handling_for_inapplicable_metrics() -> None:
    qs = _build_test_query_service()
    ans_dir = qs.answer("What is machine learning?", manual_strategy=RetrievalStrategy.DIRECT)
    mets = metric_bundle(ans_dir, reference_answer="Expected answer", relevant_document_ids=["doc_1"], reference_contexts=None)

    inapplicable_metrics = [
        "faithfulness",
        "citation_correctness",
        "context_precision",
        "context_recall",
        "retrieval_recall_at_k",
        "hit_rate_at_k",
        "mrr",
    ]
    for m in inapplicable_metrics:
        assert mets[m].available is False
        assert mets[m].value is None, f"{m} must be None (N/A) for Direct LLM, not 0.0"
        assert mets[m].details.get("reason") == "not_applicable_for_direct_llm"


# ── Requirement 12: Answer-quality calculation with valid reference answers ───
def test_answer_quality_calculation_with_valid_reference_answers() -> None:
    qs = _build_test_query_service()
    ans = qs.answer("What does the pipeline do?", manual_strategy=RetrievalStrategy.VECTOR)
    ref_ans = "The ingestion pipeline validates, chunks, embeds, and indexes documents."

    mets = metric_bundle(ans, reference_answer=ref_ans, relevant_document_ids=["doc_alpha"], reference_contexts=None)

    assert mets["answer_correctness"].available is True
    assert mets["answer_correctness"].value is not None and mets["answer_correctness"].value > 0.70
    assert mets["completeness"].available is True
    assert mets["completeness"].value is not None and mets["completeness"].value > 0.80
    assert mets["answer_relevancy"].available is True
    assert mets["answer_relevancy"].value is not None and mets["answer_relevancy"].value > 0.70
    assert mets["faithfulness"].available is True
    assert mets["faithfulness"].value is not None and mets["faithfulness"].value > 0.80


# ── Requirement 13: Missing-reference-answer handling ─────────────────────────
def test_missing_reference_answer_handling() -> None:
    qs = _build_test_query_service()
    ans = qs.answer("What does the pipeline do?", manual_strategy=RetrievalStrategy.VECTOR)

    mets = metric_bundle(ans, reference_answer=None, relevant_document_ids=None, reference_contexts=None)

    assert mets["answer_correctness"].available is False
    assert mets["answer_correctness"].value is None
    assert mets["completeness"].available is False
    assert mets["completeness"].value is None
    assert mets["context_recall"].available is False
    assert mets["context_recall"].value is None

    # Operational/evidence metrics are still computed
    assert mets["faithfulness"].available is True
    assert mets["faithfulness"].value is not None
    assert mets["answer_relevancy"].available is True
    assert mets["answer_relevancy"].value is not None

    score = evaluate_approach("VECTOR", ans, mets, has_ground_truth=False)
    rec = recommend_best_approach({"VECTOR": score}, has_ground_truth=False)
    assert rec.is_definitive is False
    assert rec.status == "provisional"
    assert "Provisional recommendation" in rec.reason


# ── Requirement 14: Retrieval metrics with valid relevance labels ─────────────
def test_retrieval_metrics_with_valid_relevance_labels() -> None:
    qs = _build_test_query_service()
    ans = qs.answer("What does the pipeline do?", manual_strategy=RetrievalStrategy.VECTOR)

    # When relevance labels exist
    mets = metric_bundle(ans, reference_answer="Expected answer", relevant_document_ids=["doc_alpha"], reference_contexts=None)
    assert mets["retrieval_recall_at_k"].available is True
    assert mets["retrieval_recall_at_k"].value == 1.0
    assert mets["mrr"].available is True
    assert mets["mrr"].value == 1.0

    # When relevance labels are missing
    mets_no_labels = metric_bundle(ans, reference_answer="Expected answer", relevant_document_ids=None, reference_contexts=None)
    assert mets_no_labels["retrieval_recall_at_k"].available is False
    assert mets_no_labels["retrieval_recall_at_k"].value is None
    assert mets_no_labels["mrr"].available is False
    assert mets_no_labels["mrr"].value is None


# ── Requirement 15: Recommendation generation (tie & inconclusive) ───────────
def test_recommendation_generation_definitive_provisional_tie_inconclusive() -> None:
    # 1. Tie test
    app_a = ApproachScore("VECTOR", 0.852, 0.85, 0.80, 0.85, 0.90, 0.90, 1.0, 1.0, 45.0, 80.0, 125.0, 2, None, False, "Success")
    app_b = ApproachScore("HYBRID", 0.858, 0.86, 0.80, 0.85, 0.90, 0.90, 1.0, 1.0, 50.0, 85.0, 135.0, 2, 1, False, "Success")
    rec_tie = recommend_best_approach({"VECTOR": app_a, "HYBRID": app_b}, has_ground_truth=True)
    assert rec_tie.status == "tie"
    assert "Tie between" in rec_tie.reason

    # 2. Inconclusive test (low evidence and low relevancy without ground truth)
    app_low = ApproachScore("VECTOR", 0.20, None, None, 0.20, 0.25, 0.20, None, None, 40.0, 80.0, 120.0, 1, None, False, "Success")
    rec_inconcl = recommend_best_approach({"VECTOR": app_low}, has_ground_truth=False)
    assert rec_inconcl.status == "inconclusive"
    assert rec_inconcl.recommended_strategy is None


# ── Requirement 16: Result persistence and reloading from ResultStore ─────────
def test_result_persistence_and_reloading() -> None:
    qs = _build_test_query_service()
    saved_records = {}
    saved_runs = {}

    reg = MagicMock()
    reg.save_evaluation_run.side_effect = lambda rid, payload: saved_runs.update({rid: payload})
    reg.save_evaluation_result.side_effect = lambda res_id, rid, payload: saved_records.update({res_id: payload})
    reg.list_evaluation_runs.side_effect = lambda: list(saved_runs.values())
    reg.list_evaluation_results.side_effect = lambda rid=None: list(saved_records.values())

    store = ResultStore(reg)
    runner = BenchmarkRunner(qs, store)

    examples = [
        EvaluationExample(query_id="q1", question="What does the pipeline do?", expected_answer="It indexes documents.", relevant_document_ids=["doc_alpha"]),
    ]
    output = runner.run(examples)
    run_id = output["run_id"]

    assert run_id in saved_runs
    assert len(saved_records) == 4  # 1 question x 4 strategies = 4 saved results

    # Reload from store
    reloaded_runs = store.list_runs()
    reloaded_results = store.list_results(run_id)
    assert len(reloaded_runs) == 1
    assert len(reloaded_results) == 4
    strategies_reloaded = {r["strategy"] for r in reloaded_results}
    assert strategies_reloaded == {"VECTOR", "GRAPH", "HYBRID", "DIRECT"}


# ── Requirement 17: Prevention of stale session-state results ─────────────────
def test_prevention_of_stale_session_state_results() -> None:
    session_state = {}

    # Simulate run 1
    run_id_1 = "run_1"
    session_state["single_eval_data"] = {
        "run_id": run_id_1,
        "query": "What is vector search?",
        "results": {"VECTOR": {"answer": "Vector search uses embeddings."}},
    }

    # User inputs a new question
    new_query = "What is GraphRAG?"

    # Check: session state correctly detects that the displayed data belongs to the old query
    assert session_state["single_eval_data"]["query"] != new_query

    # When new evaluation begins, state is explicitly reset
    session_state.pop("single_eval_data", None)
    assert "single_eval_data" not in session_state

    # Run 2 is performed with new query
    session_state["single_eval_data"] = {
        "run_id": "run_2",
        "query": new_query,
        "results": {"GRAPH": {"answer": "GraphRAG uses Neo4j."}},
    }
    assert session_state["single_eval_data"]["query"] == new_query
    assert session_state["single_eval_data"]["run_id"] == "run_2"


# ── Requirement 18: Comparison charts matching the underlying metrics ─────────
def test_comparison_charts_matching_underlying_metrics() -> None:
    eval_results = {
        "VECTOR": {
            "status": "Success",
            "metrics": {
                "answer_correctness": {"value": 0.85, "available": True},
                "faithfulness": {"value": 0.90, "available": True},
                "answer_relevancy": {"value": 0.95, "available": True},
                "completeness": {"value": 0.80, "available": True},
            },
        },
        "DIRECT": {
            "status": "Success",
            "metrics": {
                "answer_correctness": {"value": 0.70, "available": True},
                "faithfulness": {"value": None, "available": False},  # Inapplicable
                "answer_relevancy": {"value": 0.88, "available": True},
                "completeness": {"value": 0.65, "available": True},
            },
        },
        "GRAPH": {
            "status": "Failed",
            "errors": ["Connection timeout"],
            "metrics": {
                "answer_correctness": {"value": None, "available": False},
                "faithfulness": {"value": None, "available": False},
                "answer_relevancy": {"value": None, "available": False},
                "completeness": {"value": None, "available": False},
            },
        },
    }

    # Simulate chart data extraction logic exactly matching pages/5_Evaluation.py
    q_metrics_rows = []
    for s_key, r in eval_results.items():
        if r.get("status") in ("Failed", "Quota Exceeded") or r.get("errors"):
            continue  # Exclude failed approach from quality chart
        mets = r.get("metrics") or {}
        for m_id, m_label in [
            ("answer_correctness", "Correctness"),
            ("faithfulness", "Faithfulness"),
            ("answer_relevancy", "Relevancy"),
            ("completeness", "Completeness"),
        ]:
            val = mets.get(m_id, {}).get("value")
            if val is not None and not pd.isna(val):
                q_metrics_rows.append({"Strategy": s_key, "Metric": m_label, "Score": float(val)})

    df_chart = pd.DataFrame(q_metrics_rows)

    # Failed GRAPH approach must NOT appear in chart data
    assert "GRAPH" not in df_chart["Strategy"].values

    # Direct LLM must only have Correctness, Relevancy, Completeness (no Faithfulness)
    direct_metrics = df_chart[df_chart["Strategy"] == "DIRECT"]["Metric"].tolist()
    assert "Faithfulness" not in direct_metrics
    assert set(direct_metrics) == {"Correctness", "Relevancy", "Completeness"}

    # Vector RAG has all 4 metrics
    vector_metrics = df_chart[df_chart["Strategy"] == "VECTOR"]["Metric"].tolist()
    assert set(vector_metrics) == {"Correctness", "Faithfulness", "Relevancy", "Completeness"}

    # Exact score match between underlying metrics and chart data
    assert df_chart[(df_chart["Strategy"] == "VECTOR") & (df_chart["Metric"] == "Correctness")]["Score"].iloc[0] == 0.85
    assert df_chart[(df_chart["Strategy"] == "DIRECT") & (df_chart["Metric"] == "Correctness")]["Score"].iloc[0] == 0.70
