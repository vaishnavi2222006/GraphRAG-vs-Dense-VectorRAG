from __future__ import annotations

from dataclasses import dataclass

from rag_benchmark_lab.schemas.documents import GraphEvidence, RetrievedChunk, RetrievalStrategy
from rag_benchmark_lab.services.query_service import QueryService


@dataclass
class FakeEmbeddingRetriever:
    chunks: list[RetrievedChunk]

    def retrieve(self, question: str, document_ids=None):
        return type("Result", (), {"chunks": self.chunks, "retrieval_latency_ms": 1.0})()


@dataclass
class FakeGraphRetriever:
    chunks: list[RetrievedChunk]

    def retrieve(self, question: str, document_ids=None):
        evidence = GraphEvidence(entities=[], relationships=[], source_chunks=self.chunks)
        return type("Result", (), {"evidence": evidence, "source_chunks": self.chunks, "retrieval_latency_ms": 2.0, "warnings": []})()


@dataclass
class FakeHybridRetriever:
    chunks: list[RetrievedChunk]

    def retrieve(self, question: str, document_ids=None):
        evidence = GraphEvidence(entities=[], relationships=[], source_chunks=self.chunks)
        return type("Result", (), {"chunks": self.chunks, "graph_evidence": evidence, "retrieval_latency_ms": 3.0, "warnings": []})()


@dataclass
class FakeLLM:
    def health(self):
        return True

    def complete(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0):
        return type("Response", (), {"text": "answer", "model_name": "mock", "prompt_tokens": 1, "completion_tokens": 1, "latency_ms": 1.0, "mock_mode": True, "warnings": []})()


def test_query_service_generates_answer_and_citations() -> None:
    chunks = [RetrievedChunk(chunk_id="c1", document_id="d1", document_name="doc", chunk_index=0, text="evidence", score=0.9)]
    service = QueryService(FakeEmbeddingRetriever(chunks), FakeGraphRetriever(chunks), FakeHybridRetriever(chunks), FakeLLM(), top_k=5, answer_temperature=0.0)
    result = service.answer("What is evidence?", manual_strategy=RetrievalStrategy.VECTOR)
    assert result.answer == "answer"
    assert result.citations
