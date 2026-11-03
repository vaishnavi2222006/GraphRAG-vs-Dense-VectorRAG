from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from rag_benchmark_lab.config.settings import AppSettings
from rag_benchmark_lab.embeddings.embedding_service import EmbeddingService
from rag_benchmark_lab.evaluation.benchmark_runner import BenchmarkRunner
from rag_benchmark_lab.evaluation.result_store import ResultStore
from rag_benchmark_lab.generation.gemini_service import GeminiLLMService
from rag_benchmark_lab.generation.llm_service import (
    LLMResponse,
    OllamaLLMService,
    create_llm_service,
)
from rag_benchmark_lab.generation.prompts import SYSTEM_PROMPT, build_answer_prompt
from rag_benchmark_lab.retrieval.hybrid_retriever import HybridRetriever
from rag_benchmark_lab.retrieval.vector_retriever import VectorRetriever, _lexical_relevance
from rag_benchmark_lab.schemas.documents import RetrievedChunk, RetrievalStrategy
from rag_benchmark_lab.schemas.evaluation import EvaluationExample
from rag_benchmark_lab.services.document_registry import DocumentRegistry
from rag_benchmark_lab.services.query_service import QueryService
from rag_benchmark_lab.utils.errors import (
    AuthenticationError,
    ModelUnavailableError,
    NetworkError,
    QuotaExceededError,
    ServiceUnavailableError,
)
from rag_benchmark_lab.vector_store.qdrant_store import QdrantStore


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="chunk_overview_0",
            document_id="doc_hosp_1",
            document_name="01_Hospital_Overview.docx",
            chunk_index=0,
            text="MediCore Community Hospital is a fictional 120-bed general hospital. Its main campus is at 18 Lakeview Road, Riverton.",
            score=0.92,
            metadata={"dense_score": 0.90, "lexical_score": 0.94},
        ),
        RetrievedChunk(
            chunk_id="chunk_contacts_0",
            document_id="doc_hosp_2",
            document_name="02_Departments_and_Contacts.docx",
            chunk_index=0,
            text="Radiology is extension 220, Pharmacy is extension 240, Billing is extension 260, and Rehabilitation is extension 250.",
            score=0.85,
            metadata={"dense_score": 0.82, "lexical_score": 0.88},
        ),
    ]


@pytest.fixture
def mock_gemini_client():
    client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "MediCore Community Hospital is a 120-bed hospital located at 18 Lakeview Road, Riverton. [01_Hospital_Overview.docx : 0]"
    mock_resp.usage_metadata.prompt_token_count = 120
    mock_resp.usage_metadata.candidates_token_count = 35
    mock_resp.usage_metadata.total_token_count = 155
    client.models.generate_content.return_value = mock_resp
    client.models.get.return_value = MagicMock()
    return client


# ─── 1. Gemini receives retrieved context and question ────────────────────────

def test_gemini_receives_context_and_question(mock_gemini_client, sample_chunks):
    service = GeminiLLMService(api_key="test_key_abc", model_name="gemini-2.5-flash")
    service._client = mock_gemini_client

    q = "What is the name of the hospital, where is it located, and how many beds does it have?"
    prompt = build_answer_prompt(q, sample_chunks[0].text, RetrievalStrategy.VECTOR)
    response = service.complete(prompt, system_prompt=SYSTEM_PROMPT, temperature=0.0)

    # Verify Gemini was invoked with prompt containing context and question
    mock_gemini_client.models.generate_content.assert_called_once()
    call_kwargs = mock_gemini_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"
    assert q in call_kwargs["contents"]
    assert "18 Lakeview Road, Riverton" in call_kwargs["contents"]
    assert "120-bed" in call_kwargs["contents"]
    assert response.prompt_tokens == 120
    assert response.completion_tokens == 35


# ─── 2. Generator returns direct answer grounded in context ───────────────────

def test_generator_returns_grounded_answer(mock_gemini_client, sample_chunks, tmp_path):
    gemini_svc = GeminiLLMService(api_key="test_key_abc", model_name="gemini-2.5-flash")
    gemini_svc._client = mock_gemini_client

    mock_vec = MagicMock()
    mock_vec.retrieve.return_value = MagicMock(chunks=sample_chunks, retrieval_latency_ms=10.0)
    mock_graph = MagicMock()
    mock_graph.retrieve.return_value = MagicMock(source_chunks=[], evidence=None, retrieval_latency_ms=5.0, warnings=[])
    mock_hybrid = MagicMock()
    mock_hybrid.retrieve.return_value = MagicMock(chunks=sample_chunks, graph_evidence=None, retrieval_latency_ms=12.0, warnings=[])

    query_svc = QueryService(
        vector_retriever=mock_vec,
        graph_retriever=mock_graph,
        hybrid_retriever=mock_hybrid,
        llm_service=gemini_svc,
        top_k=2,
        answer_temperature=0.0,
    )

    result = query_svc.answer("What is the hospital name and address?", manual_strategy=RetrievalStrategy.VECTOR)
    assert not result.errors
    assert "MediCore Community Hospital" in result.answer
    assert "18 Lakeview Road, Riverton" in result.answer
    assert result.model_name == "gemini-2.5-flash"
    assert len(result.citations) == 2


# ─── 3. Unsupported questions do not produce invented facts ──────────────────

def test_unsupported_questions_return_insufficient_evidence(mock_gemini_client):
    insufficient_text = "The provided documents do not contain sufficient evidence to establish an exact price for cardiac surgery."
    mock_gemini_client.models.generate_content.return_value.text = insufficient_text

    gemini_svc = GeminiLLMService(api_key="test_key_abc", model_name="gemini-2.5-flash")
    gemini_svc._client = mock_gemini_client

    mock_vec = MagicMock()
    mock_vec.retrieve.return_value = MagicMock(chunks=[], retrieval_latency_ms=2.0)
    mock_graph = MagicMock()
    mock_hybrid = MagicMock()

    query_svc = QueryService(
        vector_retriever=mock_vec,
        graph_retriever=mock_graph,
        hybrid_retriever=mock_hybrid,
        llm_service=gemini_svc,
        top_k=5,
        answer_temperature=0.0,
    )

    result = query_svc.answer("What is the exact price of cardiac surgery?", manual_strategy=RetrievalStrategy.VECTOR)
    assert "insufficient" in result.answer.lower() or "not contain sufficient evidence" in result.answer.lower()


# ─── 4. Citations reference actual retrieved chunks ───────────────────────────

def test_citations_reference_actual_retrieved_chunks(mock_gemini_client, sample_chunks):
    gemini_svc = GeminiLLMService(api_key="test_key_abc", model_name="gemini-2.5-flash")
    gemini_svc._client = mock_gemini_client

    mock_vec = MagicMock()
    mock_vec.retrieve.return_value = MagicMock(chunks=sample_chunks, retrieval_latency_ms=5.0)
    query_svc = QueryService(mock_vec, MagicMock(), MagicMock(), gemini_svc, top_k=2, answer_temperature=0.0)

    result = query_svc.answer("Any question", manual_strategy=RetrievalStrategy.VECTOR)
    assert len(result.citations) == 2
    assert result.citations[0].chunk_id == "chunk_overview_0"
    assert result.citations[0].document_name == "01_Hospital_Overview.docx"
    assert result.citations[1].chunk_id == "chunk_contacts_0"
    assert result.citations[1].document_name == "02_Departments_and_Contacts.docx"


# ─── 5. Vector, Graph, and Hybrid remain independently functional ─────────────

def test_retrieval_strategies_independently_functional(mock_gemini_client, sample_chunks):
    gemini_svc = GeminiLLMService(api_key="test_key_abc", model_name="gemini-2.5-flash")
    gemini_svc._client = mock_gemini_client

    mock_vec = MagicMock()
    mock_vec.retrieve.return_value = MagicMock(chunks=[sample_chunks[0]], retrieval_latency_ms=4.0)
    mock_graph = MagicMock()
    mock_graph.retrieve.return_value = MagicMock(source_chunks=[sample_chunks[1]], evidence=None, retrieval_latency_ms=6.0, warnings=[])
    mock_hybrid = MagicMock()
    mock_hybrid.retrieve.return_value = MagicMock(chunks=sample_chunks, graph_evidence=None, retrieval_latency_ms=9.0, warnings=[])

    query_svc = QueryService(mock_vec, mock_graph, mock_hybrid, gemini_svc, top_k=2, answer_temperature=0.0)

    v_res = query_svc.answer("test query", manual_strategy=RetrievalStrategy.VECTOR)
    assert v_res.selected_strategy == RetrievalStrategy.VECTOR
    assert len(v_res.citations) == 1
    assert v_res.citations[0].chunk_id == "chunk_overview_0"

    g_res = query_svc.answer("test query", manual_strategy=RetrievalStrategy.GRAPH)
    assert g_res.selected_strategy == RetrievalStrategy.GRAPH
    assert len(g_res.citations) == 1
    assert g_res.citations[0].chunk_id == "chunk_contacts_0"

    h_res = query_svc.answer("test query", manual_strategy=RetrievalStrategy.HYBRID)
    assert h_res.selected_strategy == RetrievalStrategy.HYBRID
    assert len(h_res.citations) == 2


# ─── 6. Compare uses same question and corpus across all three methods ────────

def test_compare_uses_same_question_and_corpus(mock_gemini_client, sample_chunks):
    gemini_svc = GeminiLLMService(api_key="test_key_abc", model_name="gemini-2.5-flash")
    gemini_svc._client = mock_gemini_client

    mock_vec = MagicMock()
    mock_vec.retrieve.return_value = MagicMock(chunks=[sample_chunks[0]], retrieval_latency_ms=5.0)
    mock_graph = MagicMock()
    mock_graph.retrieve.return_value = MagicMock(source_chunks=[sample_chunks[1]], evidence=None, retrieval_latency_ms=5.0, warnings=[])
    mock_hybrid = MagicMock()
    mock_hybrid.retrieve.return_value = MagicMock(chunks=sample_chunks, graph_evidence=None, retrieval_latency_ms=8.0, warnings=[])

    query_svc = QueryService(mock_vec, mock_graph, mock_hybrid, gemini_svc, top_k=2, answer_temperature=0.0)

    test_q = "What are the hospital operating hours?"
    target_docs = ["doc_hosp_1", "doc_hosp_2"]
    comparison = query_svc.compare(test_q, document_ids=target_docs)

    assert set(comparison.keys()) == {"VECTOR", "GRAPH", "HYBRID"}
    for strat, res in comparison.items():
        assert res.retrieval_latency_ms is not None
        assert res.generation_latency_ms is not None
        assert res.total_latency_ms is not None
    # Check that document_ids were forwarded
    mock_vec.retrieve.assert_called_with(test_q, document_ids=target_docs)
    mock_graph.retrieve.assert_called_with(test_q, document_ids=target_docs)


# ─── 7. Evaluation is isolated to the selected corpus ─────────────────────────

def test_evaluation_isolated_to_selected_corpus(tmp_path, mock_gemini_client, sample_chunks):
    gemini_svc = GeminiLLMService(api_key="test_key_abc", model_name="gemini-2.5-flash")
    gemini_svc._client = mock_gemini_client

    mock_vec = MagicMock()
    mock_vec.retrieve.return_value = MagicMock(chunks=sample_chunks, retrieval_latency_ms=5.0)
    mock_graph = MagicMock()
    mock_graph.retrieve.return_value = MagicMock(source_chunks=sample_chunks, evidence=None, retrieval_latency_ms=5.0, warnings=[])
    mock_hybrid = MagicMock()
    mock_hybrid.retrieve.return_value = MagicMock(chunks=sample_chunks, graph_evidence=None, retrieval_latency_ms=8.0, warnings=[])

    query_svc = QueryService(mock_vec, mock_graph, mock_hybrid, gemini_svc, top_k=2, answer_temperature=0.0)
    registry = DocumentRegistry(tmp_path / "test.db")
    result_store = ResultStore(registry)
    runner = BenchmarkRunner(query_svc, result_store)

    ex = EvaluationExample(
        query_id="q1",
        question="What is the hospital name?",
        expected_answer="MediCore Community Hospital",
        relevant_document_ids=["doc_hosp_1"],
        reference_contexts=["MediCore Community Hospital is a 120-bed hospital."],
    )

    out = runner.run([ex], strategies=[RetrievalStrategy.VECTOR], document_ids=["doc_hosp_1"])
    assert out["summary"]["document_ids"] == ["doc_hosp_1"]
    mock_vec.retrieve.assert_called_with(ex.question, document_ids=["doc_hosp_1"])


def test_empty_corpus_evaluation_returns_empty_scope(mock_gemini_client):
    gemini_svc = GeminiLLMService(api_key="test_key_abc", model_name="gemini-2.5-flash")
    gemini_svc._client = mock_gemini_client
    query_svc = QueryService(MagicMock(), MagicMock(), MagicMock(), gemini_svc, top_k=2, answer_temperature=0.0)

    result = query_svc.answer("Any question", document_ids=[])
    assert "no documents" in result.answer
    assert "empty_corpus_scope" in result.warnings


# ─── 8. Missing API keys produce clear errors ─────────────────────────────────

def test_missing_api_key_produces_clear_error():
    service = GeminiLLMService(api_key="")
    with pytest.raises(AuthenticationError) as exc_info:
        service.complete("test prompt")
    assert "GEMINI_API_KEY" in str(exc_info.value)
    assert not service.health()


# ─── 9. Gemini API failures are handled correctly ─────────────────────────────

def test_gemini_api_failures_handled_correctly():
    service = GeminiLLMService(api_key="valid_looking_key", model_name="gemini-2.5-flash")
    mock_client = MagicMock()
    service._client = mock_client

    from google.genai import errors

    # 429 Quota Exceeded
    mock_client.models.generate_content.side_effect = errors.APIError(429, {"message": "RESOURCE_EXHAUSTED: quota exceeded"})
    with pytest.raises(QuotaExceededError) as exc_429:
        service.complete("test prompt")
    assert "quota" in str(exc_429.value).lower()

    # 404 Model Not Found
    mock_client.models.generate_content.side_effect = errors.APIError(404, {"message": "NOT_FOUND: model unavailable"})
    with pytest.raises(ModelUnavailableError) as exc_404:
        service.complete("test prompt")
    assert "not found" in str(exc_404.value).lower()

    # Network / Timeout
    mock_client.models.generate_content.side_effect = TimeoutError("Connection timed out")
    with pytest.raises(NetworkError) as exc_net:
        service.complete("test prompt")
    assert "network" in str(exc_net.value).lower() or "connection" in str(exc_net.value).lower()


# ─── 10. Existing Ollama functionality remains available ───────────────────────

def test_existing_ollama_functionality_remains_available():
    ollama_svc = create_llm_service(
        provider="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.1:8b",
        mock_mode=True,
    )
    assert isinstance(ollama_svc, OllamaLLMService)
    resp = ollama_svc.complete("Question: Who is lead cardiologist?\nRetrieved Context: Dr. Sana is lead.")
    assert resp.text
    assert resp.mock_mode is True


# ─── 11. Irrelevant README content does not automatically dominate ────────────

def test_irrelevant_readme_does_not_dominate_factual_document():
    q = "What is the name of the hospital, where is it located, and how many beds does it have?"

    factual_chunk = (
        "MEDICORE COMMUNITY HOSPITAL\n"
        "FICTIONAL TEST CORPUS\n"
        "Hospital Overview\n"
        "MediCore Community Hospital is a fictional 120-bed general hospital created for RAG system testing. "
        "Its main campus is at 18 Lakeview Road, Riverton."
    )

    readme_question_chunk = (
        "Suggested test questions:\n"
        "What is the name of the hospital, where is it located, and how many beds does it have?\n"
        "Which extension handles appointments, and when must a patient cancel?\n"
        "Who is the lead cardiologist and what is the Cardiology extension?\n"
        "Which departments may coordinate during an emergency?\n"
        "What are the laboratory operating hours?\n"
        "What is the visitor limit and general visiting window?"
    )

    fact_score = _lexical_relevance(q, factual_chunk)
    readme_score = _lexical_relevance(q, readme_question_chunk)

    # Factual chunk must outscore the interrogative question list chunk
    assert fact_score > readme_score, (
        f"Factual chunk score ({fact_score}) should exceed README question list score ({readme_score})"
    )


# ─── 12. App does not report successful generation when generation failed ──────

def test_application_does_not_report_success_when_generation_fails(sample_chunks):
    failing_llm = MagicMock()
    failing_llm.complete.side_effect = AuthenticationError("Gemini API key is invalid")
    failing_llm.model_name = "gemini-2.5-flash"

    mock_vec = MagicMock()
    mock_vec.retrieve.return_value = MagicMock(chunks=sample_chunks, retrieval_latency_ms=5.0)

    query_svc = QueryService(
        vector_retriever=mock_vec,
        graph_retriever=MagicMock(),
        hybrid_retriever=MagicMock(),
        llm_service=failing_llm,
        top_k=2,
        answer_temperature=0.0,
    )

    result = query_svc.answer("What is the hospital name?", manual_strategy=RetrievalStrategy.VECTOR)
    # Must report error in result.errors and in text, NOT a fake success
    assert result.errors
    assert any("AuthenticationError" in e for e in result.errors)
    assert "Error generating answer" in result.answer
