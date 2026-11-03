"""Unit tests for Gemini API key loading, quota handling, and retrieval preservation.

Covers all 10 verification requirements without consuming real Gemini API quota.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_benchmark_lab.config.settings import AppSettings, _load_dotenv, get_settings
from rag_benchmark_lab.generation.gemini_service import GeminiLLMService
from rag_benchmark_lab.generation.llm_service import create_llm_service
from rag_benchmark_lab.schemas.comparison import ComparisonRunRecord, ComparisonStrategyResultRecord
from rag_benchmark_lab.schemas.documents import (
    GraphEvidence,
    RelationshipRecord,
    RetrievalStrategy,
    RetrievedChunk,
)
from rag_benchmark_lab.services.health_service import HealthService
from rag_benchmark_lab.services.query_service import QueryService
from rag_benchmark_lab.services.user_registry import UserRegistry
from rag_benchmark_lab.utils.errors import (
    AuthenticationError,
    ModelUnavailableError,
    QuotaExceededError,
    RateLimitError,
    ServiceUnavailableError,
    sanitize_secrets,
)
from rag_benchmark_lab.utils.ui import format_latency, format_metric


# ─── 1. Verify required imports work ──────────────────────────────────────────

def test_1_required_imports_work():
    """Verify that all core components can be imported cleanly."""
    assert GeminiLLMService is not None
    assert QueryService is not None
    assert AppSettings is not None
    assert get_settings is not None
    assert QuotaExceededError is not None
    assert AuthenticationError is not None
    assert RateLimitError is not None
    assert ServiceUnavailableError is not None
    assert sanitize_secrets is not None


# ─── 2. Verify .env is loaded correctly without exposing the key ──────────────

def test_2_env_loaded_correctly_without_exposing_key(tmp_path):
    """Verify .env loading updates configuration and secrets are masked."""
    secret_key = "AIzaSyTestSecretKey1234567890abcdef"
    env_file = tmp_path / ".env"
    env_file.write_text(f"GEMINI_API_KEY={secret_key}\nLLM_PROVIDER=gemini\n")

    with patch.dict(os.environ, {}, clear=True):
        _load_dotenv(override=True, env_file=env_file)
        settings = AppSettings.from_env(override=True, env_file=env_file)
        assert settings.gemini_api_key == secret_key

        # Verify to_dict() masks the key
        data = settings.to_dict()
        assert data["gemini_api_key"] == "***"

        # Verify __repr__() masks the key
        repr_str = repr(settings)
        assert secret_key not in repr_str
        assert "***" in repr_str

        # Verify sanitize_secrets redacts the key
        test_msg = f"Error communicating with Gemini using key {secret_key}"
        sanitized = sanitize_secrets(test_msg, extra_secrets=[secret_key])
        assert secret_key not in sanitized
        assert "[REDACTED" in sanitized


# ─── 3. Check client uses updated configuration after restart ─────────────────

def test_3_gemini_client_uses_updated_configuration_after_restart(tmp_path):
    """Check that changing the API key and reinitializing creates a client with the new key."""
    key_v1 = "AIzaSyKeyVersion11111111111111111111"
    key_v2 = "AIzaSyKeyVersion22222222222222222222"

    svc1 = GeminiLLMService(api_key=key_v1)
    assert svc1.api_key == key_v1

    # Simulate updated key loaded into settings
    svc2 = GeminiLLMService(api_key=key_v2)
    assert svc2.api_key == key_v2
    assert svc2.api_key != svc1.api_key

    # Client is initialized per instance with its specific key
    with patch("google.genai.Client") as mock_genai_cls:
        svc2._get_client()
        mock_genai_cls.assert_called_once_with(api_key=key_v2)


# ─── 4. Test successful generation using mocked responses ─────────────────────

def test_4_successful_generation_using_mocked_response():
    """Verify Gemini complete() returns structured LLMResponse when API succeeds."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "This is a grounded factual answer generated from evidence."
    mock_resp.usage_metadata.prompt_token_count = 50
    mock_resp.usage_metadata.candidates_token_count = 20
    mock_resp.usage_metadata.total_token_count = 70
    mock_client.models.generate_content.return_value = mock_resp

    svc = GeminiLLMService(api_key="valid_test_key", model_name="gemini-2.5-flash")
    svc._client = mock_client

    response = svc.complete("Test question prompt", system_prompt="Test system")
    assert response.text == "This is a grounded factual answer generated from evidence."
    assert response.model_name == "gemini-2.5-flash"
    assert response.prompt_tokens == 50
    assert response.completion_tokens == 20
    assert response.mock_mode is False


# ─── 5. Test missing API-key handling ─────────────────────────────────────────

def test_5_missing_api_key_handling():
    """Verify missing API key produces the required friendly message without API calls."""
    expected_msg = "Gemini API key is not configured. Please update GEMINI_API_KEY in your .env file and restart the application."

    for empty_val in ("", "   ", "change-me", "your-api-key-here"):
        svc = GeminiLLMService(api_key=empty_val)
        assert svc.is_configured() is False
        assert svc.health() is False

        with pytest.raises(AuthenticationError) as exc_info:
            svc.complete("Test prompt")
        assert expected_msg in str(exc_info.value)
        # Secret value must never be displayed
        assert "change-me" not in str(exc_info.value)

    # Test QueryService handles missing key without crashing or calling LLM
    unconfigured_svc = GeminiLLMService(api_key="")
    query_svc = QueryService(
        vector_retriever=MagicMock(),
        graph_retriever=MagicMock(),
        hybrid_retriever=MagicMock(),
        llm_service=unconfigured_svc,
        top_k=2,
        answer_temperature=0.0,
    )
    ans_text, resp, errs, meta = query_svc._generate("Test question?", RetrievalStrategy.VECTOR, [])
    assert expected_msg in ans_text
    assert any(expected_msg in e for e in errs)
    assert meta["is_auth"] == "true"


# ─── 6. Test HTTP 429 quota errors ────────────────────────────────────────────

def test_6_http_429_quota_errors():
    """Verify 429 quota errors are detected with clean messages and sanitized technical details."""
    from google.genai import errors

    svc = GeminiLLMService(api_key="test_key_for_quota", model_name="gemini-2.5-flash")
    mock_client = MagicMock()
    svc._client = mock_client

    # 1. Daily quota exhaustion
    mock_client.models.generate_content.side_effect = errors.APIError(
        429,
        {"message": "RESOURCE_EXHAUSTED: Quota exceeded for quota metric 'GenerateContentRequestsPerDay'"},
    )

    with pytest.raises(QuotaExceededError) as exc_429_daily:
        svc.complete("Test prompt")

    err_str = str(exc_429_daily.value)
    assert "Gemini's available quota has been exhausted for the configured model" in err_str
    assert exc_429_daily.value.is_daily_limit is True
    # Explanation does not claim a new key in the same project resets quota
    assert "same project" in exc_429_daily.value.explanation

    # Technical details are sanitized
    assert "RESOURCE_EXHAUSTED" in exc_429_daily.value.technical_details
    assert "test_key_for_quota" not in exc_429_daily.value.technical_details

    # 2. Per-minute rate limit (RPM)
    mock_client.models.generate_content.side_effect = errors.APIError(
        429,
        {"message": "RESOURCE_EXHAUSTED: Rate limit exceeded (requests per minute)"},
    )

    with pytest.raises(QuotaExceededError) as exc_429_rpm:
        svc.complete("Test prompt")

    assert "per-minute request rate limit reached" in str(exc_429_rpm.value)
    assert exc_429_rpm.value.is_daily_limit is False


# ─── 7. Test HTTP 503 errors ──────────────────────────────────────────────────

def test_7_http_503_errors():
    """Verify HTTP 503 temporary service failure is caught and cleanly reported."""
    from google.genai import errors

    svc = GeminiLLMService(api_key="test_key_503", model_name="gemini-2.5-flash")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = errors.APIError(
        503,
        {"message": "The service is temporarily unavailable. Please retry."},
    )
    svc._client = mock_client

    with pytest.raises(ServiceUnavailableError) as exc_503:
        svc.complete("Test prompt")

    assert "temporarily unavailable" in str(exc_503.value)
    assert "503" in str(exc_503.value)


# ─── 8. Test partial retrieval success when generation fails ──────────────────

def test_8_partial_retrieval_success_when_generation_fails():
    """Verify retrieved chunks, citations, and graph paths are preserved when generation fails."""
    sample_chunk = RetrievedChunk(
        chunk_id="chunk-001",
        document_id="doc-001",
        document_name="Hospital Policy.pdf",
        chunk_index=1,
        text="Visiting hours are 10:00 to 20:00 daily.",
        score=0.92,
        page_number=1,
    )

    mock_vec = MagicMock()
    mock_vec.retrieve.return_value = MagicMock(chunks=[sample_chunk], retrieval_latency_ms=14.5)

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = QuotaExceededError(
        "Gemini's available quota has been exhausted for the configured model. Check the project's quota or wait until it resets."
    )
    mock_llm.model_name = "gemini-2.5-flash"
    mock_llm.is_configured.return_value = True

    query_svc = QueryService(
        vector_retriever=mock_vec,
        graph_retriever=MagicMock(),
        hybrid_retriever=MagicMock(),
        llm_service=mock_llm,
        top_k=5,
        answer_temperature=0.0,
    )

    result = query_svc.answer("What are visiting hours?", manual_strategy=RetrievalStrategy.VECTOR)

    # 1. Retrieval evidence must be preserved
    assert len(result.retrieved_sources) == 1
    assert result.retrieved_sources[0].text == "Visiting hours are 10:00 to 20:00 daily."
    assert len(result.citations) == 1
    assert result.citations[0].document_name == "Hospital Policy.pdf"
    assert result.retrieval_latency_ms == 14.5

    # 2. Generation must be marked as failed / unavailable, NOT a fabricated answer
    assert result.errors
    assert any("QuotaExceededError" in e for e in result.errors)
    assert result.metadata["retrieval_status"] == "success"
    assert result.metadata["generation_status"] == "failed"
    assert result.generation_latency_ms is None


def test_8b_compare_skips_redundant_generation_after_quota_exhaustion():
    """Verify Compare skips generation on subsequent strategies after quota is exhausted."""
    sample_chunk = RetrievedChunk(
        chunk_id="c1", document_id="d1", document_name="doc.txt", chunk_index=0, text="content", score=0.9
    )

    mock_vec = MagicMock()
    mock_vec.retrieve.return_value = MagicMock(chunks=[sample_chunk], retrieval_latency_ms=10.0)

    mock_graph = MagicMock()
    rel = RelationshipRecord(
        relationship_id="r1", source_entity_id="e1", source_entity_name="Hospital",
        target_entity_id="e2", target_entity_name="Riverton", relationship_type="LOCATED_IN"
    )
    graph_ev = GraphEvidence(entities=[], relationships=[rel], source_chunks=[sample_chunk])
    mock_graph.retrieve.return_value = MagicMock(source_chunks=[sample_chunk], evidence=graph_ev, retrieval_latency_ms=15.0, warnings=[])

    mock_hybrid = MagicMock()
    mock_hybrid.retrieve.return_value = MagicMock(chunks=[sample_chunk], graph_evidence=graph_ev, retrieval_latency_ms=20.0, warnings=[])

    # LLM fails on the first call with QuotaExceededError
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = QuotaExceededError("Gemini's available quota has been exhausted.")
    mock_llm.model_name = "gemini-2.5-flash"
    mock_llm.is_configured.return_value = True

    query_svc = QueryService(
        vector_retriever=mock_vec,
        graph_retriever=mock_graph,
        hybrid_retriever=mock_hybrid,
        llm_service=mock_llm,
        top_k=2,
        answer_temperature=0.0,
    )

    results = query_svc.compare("Where is the hospital located?")

    # Vector RAG failed generation on call 1
    assert "VECTOR" in results
    assert len(results["VECTOR"].retrieved_sources) == 1
    assert results["VECTOR"].metadata["generation_status"] == "failed"

    # GraphRAG and Hybrid RAG executed retrieval successfully
    assert "GRAPH" in results
    assert len(results["GRAPH"].retrieved_sources) == 1
    assert results["GRAPH"].graph_evidence is not None
    assert results["GRAPH"].metadata["retrieval_status"] == "success"

    assert "HYBRID" in results
    assert len(results["HYBRID"].retrieved_sources) == 1

    # But mock_llm.complete was called ONLY ONCE, preventing duplicate 429 requests!
    assert mock_llm.complete.call_count == 1


# ─── 9. Test history continues to save and reload correctly ───────────────────

def test_9_history_continues_to_save_and_reload_correctly(tmp_path):
    """Verify saved comparison records with partial failures persist and reload accurately."""
    db_path = tmp_path / "test_history.sqlite3"
    registry = UserRegistry(db_path)
    user, _ = registry.register_user("Test User", "test@example.com", "Password123")
    assert user is not None
    user_id = user.user_id
    kb = registry.create_knowledge_base(user_id=user_id, name="Medical Test Corpus")
    kb_id = kb.kb_id

    run_rec = ComparisonRunRecord(
        run_id="comp-test-12345",
        user_id=user_id,
        kb_id=kb_id,
        query="What are the operating hours?",
        status="failed",
        strategies=["VECTOR", "GRAPH", "HYBRID"],
        kb_name="Medical Test Corpus",
        results=[
            ComparisonStrategyResultRecord(
                result_id="res-001",
                run_id="comp-test-12345",
                strategy="VECTOR",
                strategy_label="Vector RAG",
                status="failed",
                answer="Gemini quota exceeded",
                error_message="QuotaExceededError: 429",
                retrieval_latency_ms=12.0,
                generation_latency_ms=None,
                total_latency_ms=None,
                citations=[{"document_name": "Hours.pdf", "chunk_index": 0, "score": 0.85}],
                evidence_sources=[{"text": "Laboratory hours: 08:00 - 17:00"}],
                errors=["QuotaExceededError: 429"],
                metadata={"retrieval_status": "success", "generation_status": "failed"},
            )
        ],
    )

    registry.save_comparison_run(run_rec)
    loaded = registry.get_comparison_run("comp-test-12345", user_id=user_id)

    assert loaded is not None
    assert loaded.run_id == "comp-test-12345"
    assert loaded.query == "What are the operating hours?"
    assert len(loaded.results) == 1
    strat_res = loaded.results[0]
    assert strat_res.strategy == "VECTOR"
    assert strat_res.retrieval_latency_ms == 12.0
    assert len(strat_res.citations) == 1
    assert strat_res.citations[0]["document_name"] == "Hours.pdf"
    assert len(strat_res.evidence_sources) == 1
    assert "Laboratory hours" in strat_res.evidence_sources[0]["text"]


# ─── 10. Confirm no unnecessary Gemini requests are made on page renders ──────

def test_10_no_unnecessary_gemini_requests_during_page_rendering():
    """Verify that health_service.check() performs safe local validation without API calls."""
    mock_client = MagicMock()
    svc = GeminiLLMService(api_key="valid_looking_key", model_name="gemini-2.5-flash")
    svc._client = mock_client

    mock_qdrant = MagicMock()
    mock_neo4j = MagicMock()
    mock_registry = MagicMock()
    mock_registry.counts.return_value = {"documents": 2, "chunks": 5, "evaluation_runs": 0}

    health_svc = HealthService(mock_qdrant, mock_neo4j, svc, mock_registry)

    # Calling check() during page rendering
    health_results = health_svc.check(check_remote_llm=False)
    assert len(health_results) == 4

    # Crucial: NO calls made to Gemini remote endpoints!
    mock_client.models.get.assert_not_called()
    mock_client.models.generate_content.assert_not_called()


# ─── 11. Test unavailable Gemini model (HTTP 404 NOT_FOUND) ───────────────────

def test_11_unavailable_gemini_model_404():
    """Verify 404 Model Not Found raises ModelUnavailableError and QueryService handles it gracefully."""
    from google.genai import errors

    svc = GeminiLLMService(api_key="valid_key", model_name="non-existent-gemini-model")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = errors.APIError(
        404,
        {"message": "models/non-existent-gemini-model is not found for API version v1alpha"},
    )
    svc._client = mock_client

    with pytest.raises(ModelUnavailableError) as exc_info:
        svc.complete("Test question")

    assert "not found" in str(exc_info.value).lower()
    assert "non-existent-gemini-model" in str(exc_info.value)

    # Test QueryService gracefully captures ModelUnavailableError
    sample_chunk = RetrievedChunk(
        chunk_id="c1", document_id="d1", document_name="doc.txt", chunk_index=0, text="content", score=0.88
    )
    mock_vec = MagicMock()
    mock_vec.retrieve.return_value = MagicMock(chunks=[sample_chunk], retrieval_latency_ms=18.0)

    query_svc = QueryService(
        vector_retriever=mock_vec,
        graph_retriever=MagicMock(),
        hybrid_retriever=MagicMock(),
        llm_service=svc,
        top_k=2,
        answer_temperature=0.0,
    )

    ans = query_svc.answer("Explain the topic", manual_strategy=RetrievalStrategy.VECTOR)
    # Retrieval evidence preserved
    assert len(ans.retrieved_sources) == 1
    assert len(ans.citations) == 1
    assert ans.retrieval_latency_ms == 18.0
    # Generation marked as failed with no fabricated answer
    assert ans.metadata["generation_status"] == "failed"
    assert ans.generation_latency_ms is None
    assert "ModelUnavailableError" in ans.errors[0]
    assert "Error generating answer" in ans.answer


# ─── 12. Test latency and metric formatting edge cases ────────────────────────

def test_12_latency_and_metric_formatting_edge_cases():
    """Verify latency formatting handles None, invalid types, zero, and normal values safely."""
    # 1. Missing generation latency must show "Unavailable", NOT "0 ms"
    assert format_latency(None) == "Unavailable"
    assert format_latency(None, fallback="N/A") == "N/A"
    assert format_latency(0.0, treat_zero_as_unavailable=True) == "Unavailable"
    assert format_latency(0, treat_zero_as_unavailable=True) == "Unavailable"

    # 2. When valid 0.0 latency is allowed (e.g. retrieval elapsed time)
    assert format_latency(0.0, treat_zero_as_unavailable=False) == "0 ms"

    # 3. Missing retrieval latency
    assert format_latency(None, fallback="Unavailable") == "Unavailable"

    # 4. Normal latency metrics
    assert format_latency(45.2) == "45 ms"
    assert format_latency(180.7) == "181 ms"
    assert format_latency(225.9, precision=1) == "225.9 ms"
    assert format_latency(15.4, unit="") == "15"

    # 5. Invalid / corrupt types (strings, nan, inf)
    assert format_latency("not_a_number") == "Unavailable"
    assert format_latency(float("nan")) == "Unavailable"
    assert format_latency(float("inf")) == "Unavailable"

    # 6. Quality metrics formatting (scores, recall, faithfulness)
    assert format_metric(None) == "N/A"
    assert format_metric(0.92345, precision=3) == "0.923"
    assert format_metric(float("nan")) == "N/A"
    assert format_metric("invalid") == "N/A"


# ─── 13. Retrieval success followed by generation failure preserves all evidence

def test_13_retrieval_success_followed_by_generation_failure_preserves_all_evidence():
    """Verify all evidence (chunks, citations, graph triples) and retrieval latency are preserved when LLM fails."""
    sample_chunk = RetrievedChunk(
        chunk_id="chunk-alpha",
        document_id="doc-alpha",
        document_name="Architecture_Guide.pdf",
        chunk_index=3,
        text="Dense Vector RAG captures semantic meaning through embedding proximity.",
        score=0.945,
        page_number=2,
    )
    rel = RelationshipRecord(
        relationship_id="rel-1",
        source_entity_id="e1",
        source_entity_name="Vector RAG",
        target_entity_id="e2",
        target_entity_name="Semantic Search",
        relationship_type="ENABLES",
        confidence=0.99,
    )
    graph_ev = GraphEvidence(
        entities=[],
        relationships=[rel],
        source_chunks=[sample_chunk],
    )

    mock_hybrid = MagicMock()
    mock_hybrid.retrieve.return_value = MagicMock(
        chunks=[sample_chunk],
        graph_evidence=graph_ev,
        retrieval_latency_ms=32.4,
        warnings=[],
    )

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = QuotaExceededError("RESOURCE_EXHAUSTED: daily quota limit reached")
    mock_llm.model_name = "gemini-2.5-flash"
    mock_llm.is_configured.return_value = True

    query_svc = QueryService(
        vector_retriever=MagicMock(),
        graph_retriever=MagicMock(),
        hybrid_retriever=mock_hybrid,
        llm_service=mock_llm,
        top_k=3,
        answer_temperature=0.0,
    )

    result = query_svc.answer("Explain Vector RAG", manual_strategy=RetrievalStrategy.HYBRID)

    # Retrieval evidence MUST be preserved
    assert len(result.retrieved_sources) == 1
    assert result.retrieved_sources[0].chunk_id == "chunk-alpha"
    assert len(result.citations) == 1
    assert result.citations[0].document_name == "Architecture_Guide.pdf"
    assert result.citations[0].score == 0.945
    assert result.graph_evidence is not None
    assert len(result.graph_evidence.relationships) == 1
    assert result.graph_evidence.relationships[0].relationship_type == "ENABLES"

    # Latencies
    assert result.retrieval_latency_ms == 32.4
    assert result.generation_latency_ms is None
    assert result.total_latency_ms is not None and result.total_latency_ms > 0

    # Status & Error tracking
    assert result.metadata["retrieval_status"] == "success"
    assert result.metadata["generation_status"] == "failed"
    assert "QuotaExceededError" in result.errors[0]
    assert "Error generating answer" in result.answer


# ─── 14. Safe display of metrics in Chat and Compare simulations ───────────────

def test_14_safe_metric_display_in_chat_and_compare_simulations():
    """Verify that Chat message rendering and Compare summary generation never crash with TypeError on None latencies."""
    # 1. Simulate Chat message dictionary where generation failed
    chat_msg_failed_gen = {
        "role": "assistant",
        "content": "Error generating answer: Quota exceeded",
        "retrieval_strategy": "hybrid",
        "retrieval_latency_ms": 28.6,
        "generation_latency_ms": None,  # Key is present with None value
        "total_latency_ms": 35.1,
        "citations": [
            {"document_name": "Report.pdf", "chunk_index": 2, "score": None, "page_number": 1}
        ],
        "errors": ["QuotaExceededError: 429 quota exhausted"],
    }

    # Verify formatting Chat message fields never raises TypeError: unsupported format string passed to NoneType.__format__
    ret_str = format_latency(chat_msg_failed_gen.get("retrieval_latency_ms"), fallback="Unavailable")
    gen_str = format_latency(
        chat_msg_failed_gen.get("generation_latency_ms"),
        fallback="Unavailable",
        treat_zero_as_unavailable=True,
    )
    tot_str = format_latency(
        chat_msg_failed_gen.get("total_latency_ms"),
        fallback="Unavailable",
        treat_zero_as_unavailable=True,
    )

    assert ret_str == "29 ms"
    assert gen_str == "Unavailable"  # Crucial: NOT "0 ms", NOT crashing!
    assert tot_str == "35 ms"

    # Verify score with None safely formats without crashing
    c = chat_msg_failed_gen["citations"][0]
    score_val = c.get("score")
    score_str = f" · Similarity Score: {format_metric(score_val, fallback='', precision=3)}" if score_val is not None else ""
    assert score_str == ""

    # 2. Simulate Chat message where BOTH latencies are None
    both_none_msg = {
        "retrieval_latency_ms": None,
        "generation_latency_ms": None,
        "total_latency_ms": None,
    }
    assert format_latency(both_none_msg.get("retrieval_latency_ms"), fallback="Unavailable") == "Unavailable"
    assert format_latency(both_none_msg.get("generation_latency_ms"), fallback="Unavailable", treat_zero_as_unavailable=True) == "Unavailable"
    assert format_latency(both_none_msg.get("total_latency_ms"), fallback="Unavailable", treat_zero_as_unavailable=True) == "Unavailable"

    # 3. Simulate Compare summary row generation with failed generation
    compare_record = ComparisonStrategyResultRecord(
        result_id="res-001",
        run_id="comp-001",
        strategy="vector",
        strategy_label="Vector RAG",
        status="failed",
        answer="Error generating answer",
        retrieval_latency_ms=16.2,
        generation_latency_ms=None,
        total_latency_ms=22.4,
        citations=[{"document_name": "Doc.pdf", "score": 0.88}],
        evidence_sources=[{"text": "evidence"}],
        errors=["ModelUnavailableError: model not found"],
    )

    compare_summary_row = {
        "Strategy": compare_record.strategy_label,
        "Retrieval Latency (ms)": format_latency(compare_record.retrieval_latency_ms, unit="", fallback="Unavailable"),
        "Generation Latency (ms)": (
            format_latency(compare_record.generation_latency_ms, unit="", fallback="Unavailable", treat_zero_as_unavailable=True)
            if (compare_record.status == "success" and not compare_record.errors)
            else "Unavailable"
        ),
        "Total Latency (ms)": (
            format_latency(compare_record.total_latency_ms, unit="", fallback="Unavailable", treat_zero_as_unavailable=True)
            if (compare_record.status == "success" and not compare_record.errors)
            else format_latency(compare_record.retrieval_latency_ms, unit="", fallback="Unavailable")
        ),
    }

    assert compare_summary_row["Retrieval Latency (ms)"] == "16"
    assert compare_summary_row["Generation Latency (ms)"] == "Unavailable"
    assert compare_summary_row["Total Latency (ms)"] == "16"

