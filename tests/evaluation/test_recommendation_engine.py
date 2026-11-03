"""Unit tests for the algorithmic recommendation engine and benchmark ranking."""
from __future__ import annotations

import pytest

from rag_benchmark_lab.evaluation.recommendation import (
    ApproachScore,
    evaluate_approach,
    rank_benchmark_approaches,
    recommend_best_approach,
)
from rag_benchmark_lab.schemas.documents import AnswerResult, RetrievalStrategy
from rag_benchmark_lab.schemas.evaluation import MetricResult


def test_recommendation_with_ground_truth_prioritizes_correctness() -> None:
    """When reference answer is present, correctness and evidence grounding outrank raw latency."""
    # Fast Direct LLM with lower correctness
    fast_direct = ApproachScore(
        strategy="DIRECT",
        composite_score=0.45,
        correctness=0.40,
        completeness=0.50,
        faithfulness=None,
        relevancy=0.85,
        citation_score=None,
        retrieval_recall=None,
        mrr=None,
        retrieval_latency_ms=None,
        generation_latency_ms=150.0,
        total_latency_ms=150.0,
        retrieved_chunk_count=None,
        graph_path_count=None,
        is_failed=False,
        status_label="Success",
        strengths=["Low latency execution (150 ms)"],
        weaknesses=["No corpus evidence or citations available to independently verify claims"],
    )

    # Slower Hybrid RAG with near-perfect correctness and strong grounding
    accurate_hybrid = ApproachScore(
        strategy="HYBRID",
        composite_score=0.91,
        correctness=0.95,
        completeness=0.90,
        faithfulness=0.95,
        relevancy=0.92,
        citation_score=1.0,
        retrieval_recall=1.0,
        mrr=1.0,
        retrieval_latency_ms=80.0,
        generation_latency_ms=850.0,
        total_latency_ms=930.0,
        retrieved_chunk_count=5,
        graph_path_count=4,
        is_failed=False,
        status_label="Success",
        strengths=["High ground-truth correctness (0.95)", "Strong corpus evidence grounding (0.95)"],
        weaknesses=[],
    )

    evals = {"DIRECT": fast_direct, "HYBRID": accurate_hybrid}
    rec = recommend_best_approach(evals, has_ground_truth=True, question="What are the details?")

    assert rec.recommended_strategy == "HYBRID"
    assert rec.is_definitive is True
    assert rec.status == "definitive"
    assert rec.confidence == "High"
    assert "highest composite accuracy" in rec.reason


def test_provisional_recommendation_when_ground_truth_missing() -> None:
    """When reference answer is absent, recommendation is marked provisional and explains limitations."""
    vec_rag = ApproachScore(
        strategy="VECTOR",
        composite_score=0.76,
        correctness=None,
        completeness=None,
        faithfulness=0.85,
        relevancy=0.80,
        citation_score=0.80,
        retrieval_recall=None,
        mrr=None,
        retrieval_latency_ms=50.0,
        generation_latency_ms=500.0,
        total_latency_ms=550.0,
        retrieved_chunk_count=3,
        graph_path_count=None,
        is_failed=False,
        status_label="Success",
        strengths=["Strong corpus evidence grounding (0.85)"],
        weaknesses=[],
    )
    direct_llm = ApproachScore(
        strategy="DIRECT",
        composite_score=0.48,
        correctness=None,
        completeness=None,
        faithfulness=None,
        relevancy=0.80,
        citation_score=None,
        retrieval_recall=None,
        mrr=None,
        retrieval_latency_ms=None,
        generation_latency_ms=200.0,
        total_latency_ms=200.0,
        retrieved_chunk_count=None,
        graph_path_count=None,
        is_failed=False,
        status_label="Success",
        strengths=["Low latency execution (200 ms)"],
        weaknesses=["No corpus evidence or citations available to independently verify claims"],
    )

    evals = {"VECTOR": vec_rag, "DIRECT": direct_llm}
    rec = recommend_best_approach(evals, has_ground_truth=False, question="Explain this feature")

    assert rec.recommended_strategy == "VECTOR"
    assert rec.is_definitive is False
    assert rec.status == "provisional"
    assert "Provisional recommendation" in rec.reason


def test_tie_detection() -> None:
    """When two approaches have nearly identical quality and latency, report a tie."""
    app_vec = ApproachScore(
        strategy="VECTOR",
        composite_score=0.82,
        correctness=0.85,
        completeness=0.80,
        faithfulness=0.85,
        relevancy=0.85,
        citation_score=0.9,
        retrieval_recall=1.0,
        mrr=1.0,
        retrieval_latency_ms=40.0,
        generation_latency_ms=450.0,
        total_latency_ms=490.0,
        retrieved_chunk_count=3,
        graph_path_count=None,
        is_failed=False,
        status_label="Success",
        strengths=[],
        weaknesses=[],
    )
    app_graph = ApproachScore(
        strategy="GRAPH",
        composite_score=0.825,
        correctness=0.85,
        completeness=0.80,
        faithfulness=0.85,
        relevancy=0.85,
        citation_score=0.9,
        retrieval_recall=1.0,
        mrr=1.0,
        retrieval_latency_ms=45.0,
        generation_latency_ms=460.0,
        total_latency_ms=505.0,
        retrieved_chunk_count=3,
        graph_path_count=2,
        is_failed=False,
        status_label="Success",
        strengths=[],
        weaknesses=[],
    )

    evals = {"VECTOR": app_vec, "GRAPH": app_graph}
    rec = recommend_best_approach(evals, has_ground_truth=True, question="Question")

    assert rec.status == "tie"
    assert "VECTOR" in (rec.recommended_strategy or "")
    assert "GRAPH" in (rec.recommended_strategy or "")
    assert "Tie between" in rec.reason



def test_inconclusive_detection_when_evidence_insufficient() -> None:
    """When corpus evidence is missing and all approaches have low relevance, report inconclusive."""
    app_vec = ApproachScore(
        strategy="VECTOR",
        composite_score=0.20,
        correctness=None,
        completeness=None,
        faithfulness=0.10,
        relevancy=0.25,
        citation_score=0.0,
        retrieval_recall=None,
        mrr=None,
        retrieval_latency_ms=30.0,
        generation_latency_ms=300.0,
        total_latency_ms=330.0,
        retrieved_chunk_count=0,
        graph_path_count=None,
        is_failed=False,
        status_label="Success",
        strengths=[],
        weaknesses=["No matching document chunks retrieved from corpus"],
    )

    evals = {"VECTOR": app_vec}
    rec = recommend_best_approach(evals, has_ground_truth=False, question="Question about unknown topic")

    assert rec.status == "inconclusive"
    assert rec.recommended_strategy is None
    assert "insufficient supporting evidence" in rec.reason


def test_aggregate_benchmark_ranking() -> None:
    """Test ranking calculation across an entire benchmark dataset."""
    run_rows = [
        {"strategy": "VECTOR", "total_latency_ms": 300.0, "metrics": {"answer_correctness": {"value": 0.8}, "faithfulness": {"value": 0.85}}},
        {"strategy": "HYBRID", "total_latency_ms": 500.0, "metrics": {"answer_correctness": {"value": 0.95}, "faithfulness": {"value": 0.95}}},
        {"strategy": "DIRECT", "total_latency_ms": 150.0, "metrics": {"answer_correctness": {"value": 0.6}, "faithfulness": {"value": None, "available": False}}},
    ]
    query_summaries = [
        {"query_id": "q1", "recommendation": {"status": "definitive", "recommended_strategy": "HYBRID"}},
        {"query_id": "q2", "recommendation": {"status": "definitive", "recommended_strategy": "HYBRID"}},
        {"query_id": "q3", "recommendation": {"status": "tie", "recommended_strategy": "VECTOR / HYBRID"}},
    ]

    ranking = rank_benchmark_approaches(run_rows, query_summaries)

    assert ranking["overall_best"] == "HYBRID"
    assert ranking["winner_counts"].get("HYBRID") == 2
    assert ranking["ties_count"] == 1
    assert len(ranking["strategy_rankings"]) == 3
    # Check median and mean latency are recorded
    for s_rank in ranking["strategy_rankings"]:
        assert s_rank["average_latency_ms"] is not None
        assert s_rank["median_latency_ms"] is not None


def test_aggregate_benchmark_all_failed_reports_none() -> None:
    """When all approaches fail across all benchmark queries, overall_best must be None."""
    run_rows = [
        {"strategy": "VECTOR", "status": "Quota Exceeded", "total_latency_ms": 50.0, "composite_score": -1.0, "metrics": {}},
        {"strategy": "GRAPH", "status": "Quota Exceeded", "total_latency_ms": 60.0, "composite_score": -1.0, "metrics": {}},
        {"strategy": "HYBRID", "status": "Quota Exceeded", "total_latency_ms": 70.0, "composite_score": -1.0, "metrics": {}},
        {"strategy": "DIRECT", "status": "Quota Exceeded", "total_latency_ms": 10.0, "composite_score": -1.0, "metrics": {}},
    ]
    query_summaries = [
        {"query_id": "q1", "recommendation": {"status": "failed", "recommended_strategy": None}},
        {"query_id": "q2", "recommendation": {"status": "failed", "recommended_strategy": None}},
    ]

    ranking = rank_benchmark_approaches(run_rows, query_summaries)

    assert ranking["overall_best"] is None
    assert "failed" in ranking["overall_reason"].lower() or "no approach" in ranking["overall_reason"].lower()
    assert ranking["failed_count"] == 2
    assert ranking["winner_counts"] == {}
