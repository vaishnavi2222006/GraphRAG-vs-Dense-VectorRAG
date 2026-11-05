"""Unit tests for RAG Intelligence & Query Analytics.

Tests:
1. QueryClassifier categorization accuracy across all 9 categories.
2. SQLite caching of classification results.
3. IntelligenceService historical aggregation and score calculation.
4. Strict separation of Actual Evaluation scores from Estimated/Provisional scores.
5. 4-Pillar Normalized Weighted Performance Scoring with missing metric re-weighting.
6. Category-wise recommendations identifying sub-winners across answer, retrieval, latency, cost.
7. Confidence levels and "Insufficient data" handling for sparse/untested categories.
8. Score mode filtering ('all', 'actual_only', 'provisional_only').
"""
from __future__ import annotations

from dataclasses import dataclass, field
import sqlite3
from typing import Any
import pytest

from rag_benchmark_lab.analytics.intelligence_service import (
    ApproachRecord,
    CategoryIntelligence,
    IntelligenceService,
    IntelligenceSummary,
    QueryIntelligenceRecord,
    ScoringWeights,
)
from rag_benchmark_lab.analytics.query_classifier import (
    CATEGORY_COLOR_MAP,
    CATEGORY_DESCRIPTIONS,
    QUERY_CATEGORIES,
    QueryClassification,
    QueryClassifier,
)


@dataclass
class DummyCompResult:
    strategy: str
    status: str = "success"
    retrieval_latency_ms: float = 200.0
    generation_latency_ms: float = 800.0
    total_latency_ms: float = 1000.0
    prompt_tokens: int = 1200
    completion_tokens: int = 250
    model_name: str = "qwen/qwen3.8-27b"
    citations: list = field(default_factory=lambda: [{"doc": "d1"}])
    graph_evidence: dict = field(default_factory=dict)
    answer: str = "A comprehensive response..."


@dataclass
class DummyCompRun:
    run_id: str
    query: str
    created_at: str = "2026-11-05T00:00:00Z"
    results: list[DummyCompResult] = field(default_factory=list)


class DummyDocRegistry:
    def __init__(self, eval_results: list[dict] | None = None) -> None:
        self._results = eval_results or []

    def list_evaluation_results(self, run_id: str | None = None) -> list[dict]:
        return self._results


class DummyUserRegistry:
    def __init__(self, comp_runs: list | None = None, convs: list | None = None) -> None:
        self._comp_runs = comp_runs or []
        self._convs = convs or []

    def list_comparison_runs(self, kb_id: str | None = None) -> list:
        return self._comp_runs

    def get_comparison_run(self, run_id: str):
        for r in self._comp_runs:
            if getattr(r, "run_id", None) == run_id:
                return r
        return None

    def list_conversations(self, user_id: str | None = None) -> list:
        return self._convs

    def list_messages(self, conversation_id: str) -> list:
        return []


def test_query_categories_completeness():
    assert len(QUERY_CATEGORIES) == 9
    for cat in QUERY_CATEGORIES:
        assert cat in CATEGORY_DESCRIPTIONS
        assert cat in CATEGORY_COLOR_MAP


def test_query_classifier_all_nine_categories():
    qc = QueryClassifier()

    cases = [
        ("Where is MediCore Hospital located and what is its address?", "Factual / Simple Lookup"),
        ("What is the bed capacity for MediCore Community Hospital?", "Factual / Simple Lookup"),
        ("What is the hospital name, address, and bed capacity?", "Factual / Simple Lookup"),
        ("What diagnosis was recorded for patient P-101?", "Factual / Simple Lookup"),
        ("General description of the clinic facility", "Semantic Search"),
        ("Trace patient P-101 journey from initial admission through follow-up", "Multi-hop Reasoning"),
        ("Who manages the system, which team maintains it, and what are their duties?", "Multi-hop Reasoning"),
        ("What is the relationship between Cardiology and Radiology departments?", "Entity Relationship"),
        ("Which doctors work in Pediatrics and what are their specialties?", "Entity Relationship"),
        ("How do Vector RAG and GraphRAG differ?", "Comparative"),
        ("Compare the advantages of hybrid retrieval versus dense vector search", "Comparative"),
        ("How many total beds are available in the facility?", "Aggregation"),
        ("Count of registered physicians in the hospital", "Aggregation"),
        ("What steps are described for patient admission and triage protocol?", "Procedural / Step-by-Step"),
        ("Explain the standard procedure for emergency patient discharge", "Procedural / Step-by-Step"),
        ("Why did the emergency triage procedure change?", "Analytical / Complex Reasoning"),
        ("Explain why the clinic established an escalation guideline", "Analytical / Complex Reasoning"),
        ("Based on that earlier record, what follow-up is scheduled?", "Contextual"),
    ]

    for query, expected_cat in cases:
        res = qc.classify(query)
        assert res.category == expected_cat, f"Failed for query: '{query}' -> got '{res.category}', expected '{expected_cat}'"
        assert res.confidence > 0.6
        assert len(res.query_hash) == 16


def test_query_classifier_sqlite_caching(tmp_path):
    db_file = tmp_path / "test_cache.sqlite3"

    def get_conn():
        return sqlite3.connect(db_file)

    qc1 = QueryClassifier(sqlite_conn_factory=get_conn)
    q = "What is the address of the clinic?"
    res1 = qc1.classify(q)
    assert res1.category == "Factual / Simple Lookup"

    # Verify persisted in SQLite
    with get_conn() as conn:
        row = conn.execute("SELECT category, confidence FROM query_classifications WHERE query_hash = ?", (res1.query_hash,)).fetchone()
        assert row is not None
        assert row[0] == "Factual / Simple Lookup"

    # New classifier instance reading from same SQLite
    qc2 = QueryClassifier(sqlite_conn_factory=get_conn)
    res2 = qc2.classify(q)
    assert res2.category == "Factual / Simple Lookup"
    assert res2.query_hash == res1.query_hash


def test_intelligence_service_empty_registries():
    svc = IntelligenceService(
        doc_registry=DummyDocRegistry([]),
        user_registry=DummyUserRegistry([], []),
    )
    summary = svc.get_intelligence_summary()
    assert summary.total_queries == 0
    assert summary.classified_categories_count == 0
    assert summary.best_overall_architecture is None
    for cat in QUERY_CATEGORIES:
        assert summary.categories[cat].query_count == 0
        assert summary.categories[cat].best_approach == "Insufficient data"
        assert summary.categories[cat].confidence_level == "Insufficient data"
        assert "No historical queries recorded" in summary.categories[cat].data_sufficiency_reason


def test_actual_vs_provisional_separation_and_non_overwriting():
    # 1. Query with genuine actual evaluation
    actual_eval = [
        {
            "result_id": "r1",
            "run_id": "eval-run-1",
            "query_id": "q1",
            "strategy": "VECTOR",
            "question": "Where is MediCore Community Hospital located?",
            "answer": "18 Lakeview Road, Riverton",
            "composite_score": 0.88,
            "total_latency_ms": 1100.0,
            "metrics": {
                "answer_correctness": 0.85,
                "faithfulness": 0.90,
                "answer_relevancy": 0.90,
                "context_precision": 0.95,
            },
        }
    ]

    # 2. Comparison run for the same query with different telemetry
    comp_run = DummyCompRun(
        run_id="comp-run-1",
        query="Where is MediCore Community Hospital located?",
        results=[
            DummyCompResult(
                strategy="VECTOR",
                total_latency_ms=1300.0,
                citations=[{"doc": "doc1"}, {"doc": "doc2"}],
                answer="18 Lakeview Road",
            )
        ],
    )

    svc = IntelligenceService(
        doc_registry=DummyDocRegistry(actual_eval),
        user_registry=DummyUserRegistry([comp_run], []),
    )
    summary = svc.get_intelligence_summary()

    q_rec = summary.all_queries[0]
    app = q_rec.approaches["VECTOR"]

    # Verify actual score was preserved and NOT overwritten by provisional calculation
    assert app.actual_quality_score == 0.88
    assert app.is_provisional is False
    assert app.score_source == "Actual Evaluation"
    assert app.actual_metrics["answer_correctness"] == 0.85

    # 3. Test a query that exists ONLY in comparison runs
    comp_only_run = DummyCompRun(
        run_id="comp-only-1",
        query="Trace patient P-101 journey through follow-up",
        results=[
            DummyCompResult(
                strategy="GRAPH",
                total_latency_ms=2100.0,
                citations=[{"doc": "d1"}, {"doc": "d2"}, {"doc": "d3"}, {"doc": "d4"}],
                graph_evidence={"entities": [{"e": 1}], "relationships": [{"r": 1}]},
                answer="Patient P-101 transitioned through Emergency, Cardiology, and Rehabilitation.",
            )
        ],
    )

    svc2 = IntelligenceService(
        doc_registry=DummyDocRegistry([]),
        user_registry=DummyUserRegistry([comp_only_run], []),
    )
    summary2 = svc2.get_intelligence_summary()

    q_rec2 = summary2.all_queries[0]
    app2 = q_rec2.approaches["GRAPH"]

    # Verify marked as provisional
    assert app2.actual_quality_score is None
    assert app2.is_provisional is True
    assert app2.score_source == "Estimated / Provisional"
    assert app2.provisional_quality_score is not None
    assert app2.actual_metrics == {}  # Actual metrics remain empty


def test_4_pillar_weighted_scoring_normalization():
    weights = ScoringWeights(answer_quality=0.35, retrieval_quality=0.35, latency=0.15, cost=0.15)
    normalized = weights.normalized()
    assert round(normalized.answer_quality + normalized.retrieval_quality + normalized.latency + normalized.cost, 4) == 1.0

    eval_data = [
        {
            "result_id": "r1",
            "run_id": "run-1",
            "query_id": "q1",
            "strategy": "VECTOR",
            "question": "What is the hospital address?",
            "answer": "18 Lakeview Road",
            "composite_score": 0.80,
            "total_latency_ms": 1000.0,
            "retrieval_latency_ms": 300.0,
            "generation_latency_ms": 700.0,
            "metrics": {
                "answer_correctness": 0.85,
                "faithfulness": 0.85,
                "answer_relevancy": 0.85,
                "context_precision": 0.90,
                "context_recall": 0.90,
                "retrieval_recall_at_k": 0.90,
            },
        },
        {
            "result_id": "r2",
            "run_id": "run-1",
            "query_id": "q1",
            "strategy": "HYBRID",
            "question": "What is the hospital address?",
            "answer": "18 Lakeview Road, Riverton",
            "composite_score": 0.95,
            "total_latency_ms": 2500.0,
            "retrieval_latency_ms": 500.0,
            "generation_latency_ms": 2000.0,
            "metrics": {
                "answer_correctness": 0.95,
                "faithfulness": 0.95,
                "answer_relevancy": 0.95,
                "context_precision": 0.95,
                "context_recall": 0.95,
                "retrieval_recall_at_k": 0.95,
            },
        },
    ]

    svc = IntelligenceService(
        doc_registry=DummyDocRegistry(eval_data),
        user_registry=DummyUserRegistry([], []),
    )
    summary = svc.get_intelligence_summary(weights=weights)

    q = summary.all_queries[0]
    vec_app = q.approaches["VECTOR"]
    hyb_app = q.approaches["HYBRID"]

    # Verify individual pillar scores
    assert vec_app.answer_quality_score == 0.85
    assert vec_app.retrieval_quality_score == 0.90
    assert vec_app.latency_score == 1.0  # Fastest latency receives top relative score
    assert hyb_app.answer_quality_score == 0.95
    assert hyb_app.retrieval_quality_score == 0.95
    assert hyb_app.latency_score == 0.0  # Slowest latency receives lowest relative score

    # Overall weighted score
    assert 0.0 <= vec_app.overall_weighted_score <= 1.0
    assert 0.0 <= hyb_app.overall_weighted_score <= 1.0


def test_missing_metric_dynamic_reweighting():
    # Approach without ground-truth retrieval metrics or cost
    eval_no_gt = [
        {
            "result_id": "r1",
            "run_id": "run-1",
            "query_id": "q1",
            "strategy": "VECTOR",
            "question": "What is the policy for cancellations?",
            "answer": "Notice 4 hours ahead.",
            "composite_score": 0.80,
            "total_latency_ms": 1000.0,
            "metrics": {
                "faithfulness": 0.80,
                "answer_relevancy": 0.80,
            },
        }
    ]

    svc = IntelligenceService(
        doc_registry=DummyDocRegistry(eval_no_gt),
        user_registry=DummyUserRegistry([], []),
    )
    summary = svc.get_intelligence_summary()
    app = summary.all_queries[0].approaches["VECTOR"]

    # Retrieval quality is None, Cost is None
    assert app.retrieval_quality_score is None
    assert app.cost_score is None
    # Overall score must still be calculated fairly based on available Answer Quality (35%) and Latency (15%)
    assert app.overall_weighted_score is not None
    assert app.overall_weighted_score > 0.70  # Missing metrics did NOT penalize it with zeros


def test_category_recommendations_and_sub_winners():
    eval_data = [
        {
            "result_id": "r1",
            "run_id": "run-1",
            "query_id": "q1",
            "strategy": "VECTOR",
            "question": "What is the hospital address?",
            "answer": "120 beds",
            "composite_score": 0.70,
            "total_latency_ms": 800.0,
            "metrics": {"answer_correctness": 0.70, "faithfulness": 0.70, "context_precision": 0.60},
        },
        {
            "result_id": "r2",
            "run_id": "run-1",
            "query_id": "q1",
            "strategy": "GRAPH",
            "question": "What is the hospital address?",
            "answer": "MediCore has 120 beds.",
            "composite_score": 0.95,
            "total_latency_ms": 2200.0,
            "metrics": {"answer_correctness": 0.95, "faithfulness": 0.95, "context_precision": 0.98},
        },
    ]

    svc = IntelligenceService(
        doc_registry=DummyDocRegistry(eval_data),
        user_registry=DummyUserRegistry([], []),
    )
    summary = svc.get_intelligence_summary()

    cat_intel = summary.categories["Factual / Simple Lookup"]
    assert cat_intel.best_overall_architecture == "GRAPH"
    assert cat_intel.best_answer_quality_arch == "GRAPH"
    assert cat_intel.best_retrieval_quality_arch == "GRAPH"
    assert cat_intel.fastest_arch == "VECTOR"  # VECTOR was 800ms vs GRAPH 2200ms


def test_score_mode_filtering():
    actual_eval = [
        {
            "result_id": "r1",
            "run_id": "eval-1",
            "query_id": "q1",
            "strategy": "VECTOR",
            "question": "What is the emergency phone number?",
            "answer": "Ext 111",
            "composite_score": 0.90,
            "total_latency_ms": 900.0,
            "metrics": {"faithfulness": 0.90},
        }
    ]
    comp_only = DummyCompRun(
        run_id="comp-1",
        query="What are the cardiology doctors names?",
        results=[DummyCompResult(strategy="GRAPH", total_latency_ms=1500.0, answer="Dr. Asha Mehta")],
    )

    svc = IntelligenceService(
        doc_registry=DummyDocRegistry(actual_eval),
        user_registry=DummyUserRegistry([comp_only], []),
    )

    # 1. All mode: should contain both
    s_all = svc.get_intelligence_summary(score_mode="all")
    assert s_all.total_queries == 2

    # 2. Actual only mode: should contain only actual_eval
    s_actual = svc.get_intelligence_summary(score_mode="actual_only")
    assert s_actual.total_queries == 1
    assert s_actual.all_queries[0].query_text == "What is the emergency phone number?"

    # 3. Provisional only mode: should contain only comp_only
    s_prov = svc.get_intelligence_summary(score_mode="provisional_only")
    assert s_prov.total_queries == 1
    assert s_prov.all_queries[0].query_text == "What are the cardiology doctors names?"
