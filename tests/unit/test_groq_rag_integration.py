from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from rag_benchmark_lab.generation.groq_service import GroqLLMService
from rag_benchmark_lab.schemas.documents import (
    GraphEvidence,
    RetrievedChunk,
    RetrievalStrategy,
    RelationshipRecord,
    EntityRecord,
)
from rag_benchmark_lab.services.query_service import QueryService


@pytest.fixture
def sample_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="chunk_hosp_0",
            document_id="doc_hosp_1",
            document_name="01_Hospital_Overview.docx",
            chunk_index=0,
            text="MediCore Community Hospital is a fictional 120-bed general hospital located at 18 Lakeview Road, Riverton.",
            score=0.92,
            metadata={"dense_score": 0.90, "lexical_score": 0.94},
        ),
        RetrievedChunk(
            chunk_id="chunk_hosp_1",
            document_id="doc_hosp_2",
            document_name="02_Departments_and_Contacts.docx",
            chunk_index=0,
            text="Radiology is extension 220, Pharmacy is extension 240, Billing is extension 260, and Rehabilitation is extension 250.",
            score=0.85,
            metadata={"dense_score": 0.82, "lexical_score": 0.88},
        ),
    ]


@pytest.fixture
def sample_graph_evidence(sample_chunks) -> GraphEvidence:
    entity1 = EntityRecord(
        entity_id="e1",
        canonical_name="MediCore",
        original_mention="MediCore",
        entity_type="Organization",
        confidence=0.9,
    )
    entity2 = EntityRecord(
        entity_id="e2",
        canonical_name="Riverton",
        original_mention="Riverton",
        entity_type="Location",
        confidence=0.9,
    )
    rel = RelationshipRecord(
        relationship_id="r1",
        source_entity_id="e1",
        source_entity_name="MediCore",
        target_entity_id="e2",
        target_entity_name="Riverton",
        relationship_type="LOCATED_IN",
        confidence=0.95,
        source_text="MediCore is located in Riverton.",
    )
    return GraphEvidence(
        entities=[entity1, entity2],
        relationships=[rel],
        source_chunks=sample_chunks,
    )


@pytest.fixture
def mock_groq_llm():
    svc = GroqLLMService(api_key="gsk_valid_key_12345", model_name="llama-3.3-70b-versatile")
    mock_client = MagicMock()
    choice = MagicMock()
    choice.message.content = "MediCore Community Hospital is located at 18 Lakeview Road, Riverton."
    mock_resp = MagicMock()
    mock_resp.choices = [choice]
    mock_resp.model = "llama-3.3-70b-versatile"
    mock_resp.usage.prompt_tokens = 40
    mock_resp.usage.completion_tokens = 20
    mock_resp.usage.total_tokens = 60
    mock_client.chat.completions.create.return_value = mock_resp
    svc._client = mock_client
    return svc


def test_groq_receives_context_and_generates_vector_rag_answer(mock_groq_llm, sample_chunks):
    mock_vec = MagicMock()
    mock_vec.retrieve.return_value = MagicMock(chunks=sample_chunks, retrieval_latency_ms=4.0)
    mock_graph = MagicMock()
    mock_hybrid = MagicMock()

    query_svc = QueryService(
        vector_retriever=mock_vec,
        graph_retriever=mock_graph,
        hybrid_retriever=mock_hybrid,
        llm_service=mock_groq_llm,
        top_k=2,
        answer_temperature=0.0,
    )

    result = query_svc.answer("Where is MediCore located?", manual_strategy=RetrievalStrategy.VECTOR)

    assert result.selected_strategy == RetrievalStrategy.VECTOR
    assert result.model_name == "llama-3.3-70b-versatile"
    assert "18 Lakeview Road, Riverton" in result.answer
    assert len(result.citations) == 2
    assert result.citations[0].chunk_id == "chunk_hosp_0"

    # Verify Groq was invoked
    mock_groq_llm._client.chat.completions.create.assert_called_once()
    call_kwargs = mock_groq_llm._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "llama-3.3-70b-versatile"
    messages = call_kwargs["messages"]
    assert any("MediCore" in m["content"] for m in messages)


def test_groq_generates_graph_rag_answer(mock_groq_llm, sample_chunks, sample_graph_evidence):
    mock_vec = MagicMock()
    mock_graph = MagicMock()
    mock_graph.retrieve.return_value = MagicMock(
        source_chunks=[sample_chunks[0]],
        evidence=sample_graph_evidence,
        retrieval_latency_ms=6.0,
        warnings=[],
    )
    mock_hybrid = MagicMock()

    query_svc = QueryService(
        vector_retriever=mock_vec,
        graph_retriever=mock_graph,
        hybrid_retriever=mock_hybrid,
        llm_service=mock_groq_llm,
        top_k=2,
        answer_temperature=0.0,
    )

    result = query_svc.answer("What is the location of MediCore?", manual_strategy=RetrievalStrategy.GRAPH)

    assert result.selected_strategy == RetrievalStrategy.GRAPH
    assert result.model_name == "llama-3.3-70b-versatile"
    assert result.graph_evidence is not None
    assert len(result.graph_evidence.relationships) == 1
    assert result.graph_evidence.relationships[0].relationship_type == "LOCATED_IN"


def test_groq_generates_hybrid_rag_answer(mock_groq_llm, sample_chunks, sample_graph_evidence):
    mock_vec = MagicMock()
    mock_graph = MagicMock()
    mock_hybrid = MagicMock()
    mock_hybrid.retrieve.return_value = MagicMock(
        chunks=sample_chunks,
        graph_evidence=sample_graph_evidence,
        retrieval_latency_ms=9.0,
        warnings=[],
    )

    query_svc = QueryService(
        vector_retriever=mock_vec,
        graph_retriever=mock_graph,
        hybrid_retriever=mock_hybrid,
        llm_service=mock_groq_llm,
        top_k=2,
        answer_temperature=0.0,
    )

    result = query_svc.answer("Give comprehensive info about MediCore", manual_strategy=RetrievalStrategy.HYBRID)

    assert result.selected_strategy == RetrievalStrategy.HYBRID
    assert result.model_name == "llama-3.3-70b-versatile"
    assert len(result.citations) >= 1


def test_side_by_side_comparison_with_groq(mock_groq_llm, sample_chunks, sample_graph_evidence):
    mock_vec = MagicMock()
    mock_vec.retrieve.return_value = MagicMock(chunks=sample_chunks, retrieval_latency_ms=4.0)
    mock_graph = MagicMock()
    mock_graph.retrieve.return_value = MagicMock(
        source_chunks=[sample_chunks[0]],
        evidence=sample_graph_evidence,
        retrieval_latency_ms=6.0,
        warnings=[],
    )
    mock_hybrid = MagicMock()
    mock_hybrid.retrieve.return_value = MagicMock(
        chunks=sample_chunks,
        graph_evidence=sample_graph_evidence,
        retrieval_latency_ms=9.0,
        warnings=[],
    )

    query_svc = QueryService(
        vector_retriever=mock_vec,
        graph_retriever=mock_graph,
        hybrid_retriever=mock_hybrid,
        llm_service=mock_groq_llm,
        top_k=2,
        answer_temperature=0.0,
    )

    comparison = query_svc.compare("Where is MediCore located?")

    assert RetrievalStrategy.VECTOR.value in comparison
    assert RetrievalStrategy.GRAPH.value in comparison
    assert RetrievalStrategy.HYBRID.value in comparison

    for strat_key, res in comparison.items():
        assert res.model_name == "llama-3.3-70b-versatile"
        assert len(res.citations) > 0


def test_groq_rate_limit_preserves_evidence_in_query_service(sample_chunks):
    failing_groq = GroqLLMService(api_key="gsk_test_key", model_name="llama-3.3-70b-versatile")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("Rate limit reached")
    failing_groq._client = mock_client

    mock_vec = MagicMock()
    mock_vec.retrieve.return_value = MagicMock(chunks=sample_chunks, retrieval_latency_ms=4.0)
    mock_graph = MagicMock()
    mock_hybrid = MagicMock()

    query_svc = QueryService(
        vector_retriever=mock_vec,
        graph_retriever=mock_graph,
        hybrid_retriever=mock_hybrid,
        llm_service=failing_groq,
        top_k=2,
        answer_temperature=0.0,
    )

    result = query_svc.answer("Where is MediCore located?", manual_strategy=RetrievalStrategy.VECTOR)

    # Retrieval evidence MUST be preserved even when LLM generation fails
    assert len(result.retrieved_sources) == 2
    assert len(result.citations) == 2
    assert len(result.errors) > 0
    assert "Error generating answer" in result.answer


def test_benchmark_evaluation_with_groq(tmp_path, mock_groq_llm, sample_chunks):
    from rag_benchmark_lab.evaluation.benchmark_runner import BenchmarkRunner
    from rag_benchmark_lab.evaluation.result_store import ResultStore
    from rag_benchmark_lab.schemas.evaluation import EvaluationExample
    from rag_benchmark_lab.services.document_registry import DocumentRegistry

    mock_vec = MagicMock()
    mock_vec.retrieve.return_value = MagicMock(chunks=sample_chunks, retrieval_latency_ms=4.0)
    mock_graph = MagicMock()
    mock_hybrid = MagicMock()

    query_svc = QueryService(
        vector_retriever=mock_vec,
        graph_retriever=mock_graph,
        hybrid_retriever=mock_hybrid,
        llm_service=mock_groq_llm,
        top_k=2,
        answer_temperature=0.0,
    )
    registry = DocumentRegistry(tmp_path / "test.db")
    result_store = ResultStore(registry)
    runner = BenchmarkRunner(query_svc, result_store)

    ex = EvaluationExample(
        query_id="q1",
        question="Where is MediCore located?",
        expected_answer="18 Lakeview Road, Riverton",
        relevant_document_ids=["doc_hosp_1"],
        reference_contexts=["MediCore is located at 18 Lakeview Road, Riverton."],
    )

    run_output = runner.run([ex], strategies=[RetrievalStrategy.VECTOR], document_ids=["doc_hosp_1"])
    assert "summary" in run_output
    assert "aggregates" in run_output
    assert run_output["summary"]["document_ids"] == ["doc_hosp_1"]
    assert "VECTOR" in run_output["aggregates"]
    assert "answer_correctness" in run_output["aggregates"]["VECTOR"]["metrics"]

