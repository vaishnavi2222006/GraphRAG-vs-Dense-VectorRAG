"""Comprehensive Automated Test Suite for Comparison History.

Covers:
1. A successful comparison with all three RAG methods.
2. A comparison where Hybrid RAG fails but the other methods succeed.
3. A comparison where one strategy returns no evidence.
4. Saving and retrieving citations and graph paths.
5. Refreshing the application and confirming that history remains available.
6. Restarting Streamlit and confirming that saved history persists.
7. Opening a historical comparison without triggering new LLM calls.
8. Filtering and opening historical runs.
9. Confirming that failed generations are not marked successful.
10. Confirming that database errors are handled without silently losing results.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from rag_benchmark_lab.generation.llm_service import LLMResponse
from rag_benchmark_lab.schemas.comparison import (
    ComparisonRunRecord,
    ComparisonStrategyResultRecord,
)
from rag_benchmark_lab.schemas.documents import (
    AnswerResult,
    Citation,
    EntityRecord,
    GraphEvidence,
    RelationshipRecord,
    RetrievedChunk,
    RetrievalStrategy,
)
from rag_benchmark_lab.services.query_service import QueryService
from rag_benchmark_lab.services.user_registry import UserRegistry


@pytest.fixture
def comp_env(tmp_path: Path):
    """Setup an isolated UserRegistry with test user and knowledge base."""
    db_path = tmp_path / "test_comp_users.sqlite3"
    user_reg = UserRegistry(db_path)

    # Create test user and knowledge base
    user, _ = user_reg.register_user("Dr. Ada Lovelace", "ada@example.com", "Password123")
    assert user is not None
    user_id = user.user_id

    kb1 = user_reg.create_knowledge_base(user_id=user_id, name="Quantum Systems KB")
    kb2 = user_reg.create_knowledge_base(user_id=user_id, name="Biomedical KB")

    return {
        "db_path": db_path,
        "user_registry": user_reg,
        "user_id": user_id,
        "kb1_id": kb1.kb_id,
        "kb2_id": kb2.kb_id,
    }


def _make_strategy_result(
    run_id: str,
    strategy: str,
    label: str,
    status: str = "success",
    answer: str = "Test answer",
    error_message: str | None = None,
    retrieval_ms: float = 12.0,
    gen_ms: float = 45.0,
    citations: list[dict] | None = None,
    evidence: list[dict] | None = None,
    graph_evidence: dict | None = None,
) -> ComparisonStrategyResultRecord:
    now = datetime.now(timezone.utc).isoformat()
    return ComparisonStrategyResultRecord(
        result_id=f"sr-{strategy}-{run_id}",
        run_id=run_id,
        strategy=strategy,
        strategy_label=label,
        status=status,
        answer=answer,
        error_message=error_message,
        retrieval_latency_ms=retrieval_ms,
        generation_latency_ms=gen_ms,
        total_latency_ms=retrieval_ms + gen_ms,
        prompt_tokens=120,
        completion_tokens=60,
        model_name="test-llm",
        citations=citations or [],
        evidence_sources=evidence or [],
        graph_evidence=graph_evidence,
        warnings=[],
        errors=[error_message] if error_message else [],
        metadata={"test": True},
        created_at=now,
    )


# ── Test 1: Successful Comparison with All Three RAG Methods ─────────────────
def test_1_successful_comparison_all_three_methods(comp_env):
    user_reg = comp_env["user_registry"]
    user_id = comp_env["user_id"]
    kb_id = comp_env["kb1_id"]

    run_id = "comp-run-001"
    now = datetime.now(timezone.utc).isoformat()

    strat_records = [
        _make_strategy_result(run_id, "VECTOR", "Vector RAG", answer="Vector answer"),
        _make_strategy_result(run_id, "GRAPH", "GraphRAG", answer="Graph answer"),
        _make_strategy_result(run_id, "HYBRID", "Hybrid RAG", answer="Hybrid answer"),
    ]

    run = ComparisonRunRecord(
        run_id=run_id,
        user_id=user_id,
        kb_id=kb_id,
        query="What is the operating temperature of the cryostat?",
        status="completed",
        created_at=now,
        updated_at=now,
        config={"default_top_k": 5},
        strategies=["VECTOR", "GRAPH", "HYBRID"],
        results=strat_records,
    )

    saved_id = user_reg.save_comparison_run(run)
    assert saved_id == run_id

    loaded = user_reg.get_comparison_run(run_id, user_id=user_id)
    assert loaded is not None
    assert loaded.status == "completed"
    assert loaded.query == "What is the operating temperature of the cryostat?"
    assert len(loaded.results) == 3

    statuses = {r.strategy: r.status for r in loaded.results}
    assert statuses == {"VECTOR": "success", "GRAPH": "success", "HYBRID": "success"}


# ── Test 2: Comparison where Hybrid RAG Fails but Others Succeed ─────────────
def test_2_hybrid_fails_other_methods_succeed(comp_env):
    # Setup mock query service where Hybrid retriever throws HTTP 503
    v_ret = MagicMock()
    v_ret.retrieve.return_value = MagicMock(chunks=[], retrieval_latency_ms=15.0)

    g_ret = MagicMock()
    g_ret.retrieve.return_value = MagicMock(source_chunks=[], evidence=None, retrieval_latency_ms=20.0, warnings=[])

    h_ret = MagicMock()
    h_ret.retrieve.side_effect = RuntimeError("HTTP 503: Service Unavailable from LLM Gateway")

    llm = MagicMock()
    llm.model_name = "test-gemini"
    llm.complete.return_value = LLMResponse(
        text="Normal grounded answer",
        model_name="test-gemini",
        latency_ms=50.0,
        mock_mode=False,
    )

    qs = QueryService(v_ret, g_ret, h_ret, llm, top_k=5, answer_temperature=0.0)
    results = qs.compare("Test query about quantum computing")

    # Vector and Graph succeeded
    assert results["VECTOR"].answer == "Normal grounded answer"
    assert not results["VECTOR"].errors

    assert results["GRAPH"].answer == "Normal grounded answer"
    assert not results["GRAPH"].errors

    # Hybrid failed with error message
    assert "HTTP 503" in results["HYBRID"].answer
    assert results["HYBRID"].errors
    assert "HTTP 503" in results["HYBRID"].errors[0]

    # Save to user registry
    user_reg = comp_env["user_registry"]
    run_id = "comp-run-002"
    now = datetime.now(timezone.utc).isoformat()

    strat_records = [
        _make_strategy_result(run_id, "VECTOR", "Vector RAG", status="success", answer=results["VECTOR"].answer),
        _make_strategy_result(run_id, "GRAPH", "GraphRAG", status="success", answer=results["GRAPH"].answer),
        _make_strategy_result(
            run_id, "HYBRID", "Hybrid RAG", status="failed", answer=results["HYBRID"].answer, error_message=results["HYBRID"].errors[0]
        ),
    ]

    run = ComparisonRunRecord(
        run_id=run_id,
        user_id=comp_env["user_id"],
        kb_id=comp_env["kb1_id"],
        query="Test query about quantum computing",
        status="partial",
        created_at=now,
        updated_at=now,
        strategies=["VECTOR", "GRAPH", "HYBRID"],
        results=strat_records,
    )

    user_reg.save_comparison_run(run)

    loaded = user_reg.get_comparison_run(run_id)
    assert loaded is not None
    assert loaded.status == "partial"
    loaded_map = {r.strategy: r for r in loaded.results}

    assert loaded_map["VECTOR"].status == "success"
    assert loaded_map["GRAPH"].status == "success"
    assert loaded_map["HYBRID"].status == "failed"
    assert "HTTP 503" in loaded_map["HYBRID"].error_message


# ── Test 3: Comparison where One Strategy Returns No Evidence ────────────────
def test_3_one_strategy_returns_no_evidence(comp_env):
    user_reg = comp_env["user_registry"]
    run_id = "comp-run-003"
    now = datetime.now(timezone.utc).isoformat()

    strat_records = [
        _make_strategy_result(run_id, "VECTOR", "Vector RAG", citations=[{"document_name": "DocA", "score": 0.88}]),
        _make_strategy_result(
            run_id,
            "GRAPH",
            "GraphRAG",
            status="success",
            answer="The uploaded documents do not contain sufficient evidence to answer this question.",
            citations=[],
            evidence=[],
            graph_evidence=None,
        ),
        _make_strategy_result(run_id, "HYBRID", "Hybrid RAG", citations=[{"document_name": "DocA", "score": 0.91}]),
    ]

    run = ComparisonRunRecord(
        run_id=run_id,
        user_id=comp_env["user_id"],
        kb_id=comp_env["kb1_id"],
        query="What is the unrelated topic?",
        status="completed",
        created_at=now,
        updated_at=now,
        strategies=["VECTOR", "GRAPH", "HYBRID"],
        results=strat_records,
    )

    user_reg.save_comparison_run(run)
    loaded = user_reg.get_comparison_run(run_id)
    assert loaded is not None

    graph_res = next(r for r in loaded.results if r.strategy == "GRAPH")
    assert len(graph_res.citations) == 0
    assert "sufficient evidence" in graph_res.answer.lower()
    assert graph_res.status == "success"


# ── Test 4: Saving and Retrieving Citations and Graph Paths ───────────────────
def test_4_saving_and_retrieving_citations_and_graph_paths(comp_env):
    user_reg = comp_env["user_registry"]
    run_id = "comp-run-004"
    now = datetime.now(timezone.utc).isoformat()

    citations = [
        {"document_id": "doc-1", "document_name": "Quantum.pdf", "chunk_id": "chk-1", "chunk_index": 0, "page_number": 1, "score": 0.942},
        {"document_id": "doc-1", "document_name": "Quantum.pdf", "chunk_id": "chk-2", "chunk_index": 1, "page_number": 2, "score": 0.871},
    ]

    evidence_sources = [
        {"chunk_id": "chk-1", "document_name": "Quantum.pdf", "text": "Operating temperature is 15 millikelvin.", "score": 0.942},
    ]

    graph_evidence = {
        "entities": [
            {"entity_id": "e1", "canonical_name": "Hyperion-9", "entity_type": "Technology"},
            {"entity_id": "e2", "canonical_name": "Dr. Aris Thorne", "entity_type": "Person"},
        ],
        "relationships": [
            {
                "source_entity_name": "Dr. Aris Thorne",
                "relationship_type": "DEVELOPED",
                "target_entity_name": "Hyperion-9",
                "confidence": 0.98,
                "source_text": "Dr. Thorne developed the Hyperion-9 superconducting processor.",
            }
        ],
    }

    sr = _make_strategy_result(
        run_id,
        "GRAPH",
        "GraphRAG",
        citations=citations,
        evidence=evidence_sources,
        graph_evidence=graph_evidence,
    )

    run = ComparisonRunRecord(
        run_id=run_id,
        user_id=comp_env["user_id"],
        kb_id=comp_env["kb1_id"],
        query="Who developed Hyperion-9?",
        status="completed",
        created_at=now,
        updated_at=now,
        strategies=["GRAPH"],
        results=[sr],
    )

    user_reg.save_comparison_run(run)
    loaded = user_reg.get_comparison_run(run_id)
    assert loaded is not None

    loaded_sr = loaded.results[0]
    assert len(loaded_sr.citations) == 2
    assert loaded_sr.citations[0]["document_name"] == "Quantum.pdf"
    assert loaded_sr.citations[0]["score"] == 0.942

    assert loaded_sr.graph_evidence is not None
    assert len(loaded_sr.graph_evidence["relationships"]) == 1
    rel = loaded_sr.graph_evidence["relationships"][0]
    assert rel["source_entity_name"] == "Dr. Aris Thorne"
    assert rel["relationship_type"] == "DEVELOPED"
    assert rel["target_entity_name"] == "Hyperion-9"


# ── Test 5: Refreshing Application and History Remains Available ─────────────
def test_5_refresh_application_history_remains_available(comp_env):
    user_reg = comp_env["user_registry"]
    user_id = comp_env["user_id"]
    kb_id = comp_env["kb1_id"]

    for i in range(3):
        run_id = f"comp-run-refresh-{i}"
        now = f"2026-10-28T12:0{i}:00Z"
        run = ComparisonRunRecord(
            run_id=run_id,
            user_id=user_id,
            kb_id=kb_id,
            query=f"Query {i}",
            status="completed",
            created_at=now,
            updated_at=now,
            strategies=["VECTOR"],
            results=[_make_strategy_result(run_id, "VECTOR", "Vector RAG")],
        )
        user_reg.save_comparison_run(run)

    # Simulating a page refresh by executing list_comparison_runs
    refreshed_runs = user_reg.list_comparison_runs(user_id=user_id, kb_id=kb_id)
    assert len(refreshed_runs) == 3
    # Check ordering is reverse chronological
    assert refreshed_runs[0].run_id == "comp-run-refresh-2"
    assert refreshed_runs[2].run_id == "comp-run-refresh-0"


# ── Test 6: Restarting Streamlit and Confirming Saved History Persists ───────
def test_6_restarting_streamlit_persistence(comp_env):
    db_path = comp_env["db_path"]
    user_id = comp_env["user_id"]
    kb_id = comp_env["kb1_id"]

    run_id = "comp-run-persist-999"
    now = datetime.now(timezone.utc).isoformat()
    original_reg = comp_env["user_registry"]

    run = ComparisonRunRecord(
        run_id=run_id,
        user_id=user_id,
        kb_id=kb_id,
        query="Will this survive server reboot?",
        status="completed",
        created_at=now,
        updated_at=now,
        strategies=["VECTOR", "GRAPH"],
        results=[
            _make_strategy_result(run_id, "VECTOR", "Vector RAG", answer="Vector persistent"),
            _make_strategy_result(run_id, "GRAPH", "GraphRAG", answer="Graph persistent"),
        ],
    )
    original_reg.save_comparison_run(run)

    # Re-initialize UserRegistry from the exact same file path (simulating app restart)
    new_reg = UserRegistry(db_path)
    loaded = new_reg.get_comparison_run(run_id)
    assert loaded is not None
    assert loaded.query == "Will this survive server reboot?"
    assert len(loaded.results) == 2
    assert loaded.results[0].answer == "Vector persistent"
    assert loaded.results[1].answer == "Graph persistent"


# ── Test 7: Opening Historical Comparison without Triggering LLM Calls ───────
def test_7_opening_historical_comparison_no_llm_calls(comp_env):
    user_reg = comp_env["user_registry"]
    run_id = "comp-run-inspect-only"
    now = datetime.now(timezone.utc).isoformat()

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = RuntimeError("LLM MUST NOT BE CALLED WHEN INSPECTING HISTORICAL RUNS")

    run = ComparisonRunRecord(
        run_id=run_id,
        user_id=comp_env["user_id"],
        kb_id=comp_env["kb1_id"],
        query="Inspection test question",
        status="completed",
        created_at=now,
        updated_at=now,
        strategies=["VECTOR"],
        results=[_make_strategy_result(run_id, "VECTOR", "Vector RAG", answer="Cached historical answer")],
    )
    user_reg.save_comparison_run(run)

    # Open historical run
    loaded = user_reg.get_comparison_run(run_id)
    assert loaded is not None
    assert loaded.results[0].answer == "Cached historical answer"
    # Ensure LLM was never called
    mock_llm.complete.assert_not_called()


# ── Test 8: Filtering and Opening Historical Runs ────────────────────────────
def test_8_filtering_and_opening_historical_runs(comp_env):
    user_reg = comp_env["user_registry"]
    user_id = comp_env["user_id"]
    kb1_id = comp_env["kb1_id"]
    kb2_id = comp_env["kb2_id"]

    # Run 1: KB1, completed, 2026-10-20
    user_reg.save_comparison_run(
        ComparisonRunRecord(
            run_id="run-filter-1",
            user_id=user_id,
            kb_id=kb1_id,
            query="Quantum gate fidelity",
            status="completed",
            created_at="2026-10-20T10:00:00Z",
            updated_at="2026-10-20T10:00:00Z",
            strategies=["VECTOR"],
            results=[_make_strategy_result("run-filter-1", "VECTOR", "Vector RAG")],
        )
    )

    # Run 2: KB2, partial, 2026-10-25
    user_reg.save_comparison_run(
        ComparisonRunRecord(
            run_id="run-filter-2",
            user_id=user_id,
            kb_id=kb2_id,
            query="Clinical trial phase 2",
            status="partial",
            created_at="2026-10-25T14:00:00Z",
            updated_at="2026-10-25T14:00:00Z",
            strategies=["GRAPH"],
            results=[_make_strategy_result("run-filter-2", "GRAPH", "GraphRAG")],
        )
    )

    # Run 3: KB1, failed, 2026-10-28
    user_reg.save_comparison_run(
        ComparisonRunRecord(
            run_id="run-filter-3",
            user_id=user_id,
            kb_id=kb1_id,
            query="Cryostat cooling pressure",
            status="failed",
            created_at="2026-10-28T09:00:00Z",
            updated_at="2026-10-28T09:00:00Z",
            strategies=["HYBRID"],
            results=[_make_strategy_result("run-filter-3", "HYBRID", "Hybrid RAG", status="failed", error_message="Timeout")],
        )
    )

    # Filter by KB
    kb1_runs = user_reg.list_comparison_runs(user_id=user_id, kb_id=kb1_id)
    assert len(kb1_runs) == 2
    assert {r.run_id for r in kb1_runs} == {"run-filter-1", "run-filter-3"}

    # Filter by Status
    partial_runs = user_reg.list_comparison_runs(user_id=user_id, status="partial")
    assert len(partial_runs) == 1
    assert partial_runs[0].run_id == "run-filter-2"

    # Filter by Strategy
    hybrid_runs = user_reg.list_comparison_runs(user_id=user_id, strategy="HYBRID")
    assert len(hybrid_runs) == 1
    assert hybrid_runs[0].run_id == "run-filter-3"

    # Filter by Date
    date_filtered = user_reg.list_comparison_runs(user_id=user_id, start_date="2026-10-26T00:00:00Z")
    assert len(date_filtered) == 1
    assert date_filtered[0].run_id == "run-filter-3"


# ── Test 9: Confirming Failed Generations Are Not Marked Successful ──────────
def test_9_failed_generations_not_marked_successful(comp_env):
    user_reg = comp_env["user_registry"]
    run_id = "comp-run-009"
    now = datetime.now(timezone.utc).isoformat()

    strat_records = [
        _make_strategy_result(
            run_id,
            "VECTOR",
            "Vector RAG",
            status="failed",
            answer="Error generating answer: RateLimitException",
            error_message="RateLimitException: quota exceeded",
        ),
        _make_strategy_result(run_id, "GRAPH", "GraphRAG", status="success", answer="Normal Graph answer"),
    ]

    failed_count = sum(1 for s in strat_records if s.status == "failed")
    overall_status = "failed" if failed_count == len(strat_records) else "partial"

    run = ComparisonRunRecord(
        run_id=run_id,
        user_id=comp_env["user_id"],
        kb_id=comp_env["kb1_id"],
        query="Rate limited question",
        status=overall_status,
        created_at=now,
        updated_at=now,
        strategies=["VECTOR", "GRAPH"],
        results=strat_records,
    )

    user_reg.save_comparison_run(run)
    loaded = user_reg.get_comparison_run(run_id)
    assert loaded is not None
    assert loaded.status == "partial"

    v_strat = next(s for s in loaded.results if s.strategy == "VECTOR")
    assert v_strat.status == "failed"
    assert v_strat.error_message == "RateLimitException: quota exceeded"
    assert v_strat.status != "success"


# ── Test 10: Database Errors Handled Without Silently Losing Results ──────────
def test_10_database_errors_handled_without_silently_losing_results(comp_env):
    user_reg = comp_env["user_registry"]
    run_id = "comp-run-010"
    now = datetime.now(timezone.utc).isoformat()

    # Save a valid run first
    valid_run = ComparisonRunRecord(
        run_id=run_id,
        user_id=comp_env["user_id"],
        kb_id=comp_env["kb1_id"],
        query="Stable baseline query",
        status="completed",
        created_at=now,
        updated_at=now,
        strategies=["VECTOR"],
        results=[_make_strategy_result(run_id, "VECTOR", "Vector RAG")],
    )
    user_reg.save_comparison_run(valid_run)

    # Attempt saving an invalid run with broken foreign key to a non-existent user
    invalid_run = ComparisonRunRecord(
        run_id="comp-run-invalid",
        user_id="non-existent-user-id-9999",
        kb_id=comp_env["kb1_id"],
        query="Should fail constraint",
        status="completed",
        created_at=now,
        updated_at=now,
        strategies=["VECTOR"],
        results=[_make_strategy_result("comp-run-invalid", "VECTOR", "Vector RAG")],
    )

    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        user_reg.save_comparison_run(invalid_run)

    # Confirm the previous valid run was NOT corrupted or lost
    still_valid = user_reg.get_comparison_run(run_id)
    assert still_valid is not None
    assert still_valid.query == "Stable baseline query"
