from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from ..generation.llm_service import OllamaLLMService
from ..generation.prompts import SYSTEM_PROMPT, build_answer_prompt
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.query_router import route_query
from ..retrieval.vector_retriever import VectorRetriever
from ..schemas.documents import AnswerResult, Citation, RetrievedChunk, RetrievalStrategy


def _format_context(chunks: list[RetrievedChunk], limit_chars: int = 7000) -> str:
    parts: list[str] = []
    total = 0
    for chunk in chunks:
        snippet = f"[{chunk.document_id}:{chunk.chunk_id}] {chunk.text}"
        if total + len(snippet) > limit_chars:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n\n".join(parts)


class QueryService:
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        graph_retriever,
        hybrid_retriever: HybridRetriever,
        llm_service: OllamaLLMService,
        top_k: int,
        answer_temperature: float,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.hybrid_retriever = hybrid_retriever
        self.llm_service = llm_service
        self.top_k = top_k
        self.answer_temperature = answer_temperature

    def _generate(self, question: str, strategy: RetrievalStrategy, chunks: list[RetrievedChunk]) -> tuple[str, object]:
        context = _format_context(chunks)
        prompt = build_answer_prompt(question, context, strategy)
        response = self.llm_service.complete(prompt, system_prompt=SYSTEM_PROMPT, temperature=self.answer_temperature)
        return response.text, response

    def answer(self, question: str, manual_strategy: RetrievalStrategy | None = None, document_ids: list[str] | None = None, top_k: int | None = None) -> AnswerResult:
        start = perf_counter()
        routing = route_query(question, manual_strategy=manual_strategy)
        warnings: list[str] = []
        errors: list[str] = []
        retrieved: list[RetrievedChunk] = []
        graph_evidence = None
        retrieval_latency_ms = 0.0
        if routing.selected_strategy == RetrievalStrategy.VECTOR:
            vector_result = self.vector_retriever.retrieve(question, document_ids=document_ids)
            retrieved = vector_result.chunks
            retrieval_latency_ms = vector_result.retrieval_latency_ms
        elif routing.selected_strategy == RetrievalStrategy.GRAPH:
            graph_result = self.graph_retriever.retrieve(question, document_ids=document_ids)
            retrieved = graph_result.source_chunks
            graph_evidence = graph_result.evidence
            retrieval_latency_ms = graph_result.retrieval_latency_ms
            warnings.extend(graph_result.warnings)
        elif routing.selected_strategy == RetrievalStrategy.HYBRID:
            hybrid_result = self.hybrid_retriever.retrieve(question, document_ids=document_ids)
            retrieved = hybrid_result.chunks
            graph_evidence = hybrid_result.graph_evidence
            retrieval_latency_ms = hybrid_result.retrieval_latency_ms
            warnings.extend(hybrid_result.warnings)
        elif routing.selected_strategy == RetrievalStrategy.ALL:
            vector_result = self.vector_retriever.retrieve(question, document_ids=document_ids)
            graph_result = self.graph_retriever.retrieve(question, document_ids=document_ids)
            hybrid_result = self.hybrid_retriever.retrieve(question, document_ids=document_ids)
            retrieved = hybrid_result.chunks
            graph_evidence = hybrid_result.graph_evidence
            retrieval_latency_ms = vector_result.retrieval_latency_ms + graph_result.retrieval_latency_ms + hybrid_result.retrieval_latency_ms
            warnings.extend(vector_result.chunks and [] or ["vector_retrieval_returned_no_chunks"])
            warnings.extend(graph_result.warnings)
            warnings.extend(hybrid_result.warnings)
        effective_top_k = top_k or self.top_k
        answer_text, response = self._generate(question, routing.selected_strategy, retrieved[: effective_top_k])
        citations = [
            Citation(
                document_id=chunk.document_id,
                document_name=chunk.document_name,
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                score=chunk.score,
            )
            for chunk in retrieved[: effective_top_k]
        ]
        total_latency_ms = (perf_counter() - start) * 1000.0
        return AnswerResult(
            answer=answer_text,
            selected_strategy=routing.selected_strategy,
            citations=citations,
            retrieved_sources=retrieved,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=response.latency_ms,
            total_latency_ms=total_latency_ms,
            model_name=response.model_name,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            estimated_cost=0.0 if response.mock_mode else None,
            warnings=warnings + (response.warnings or []),
            errors=errors,
            routing_rationale=routing.rationale,
            routing_confidence=routing.confidence,
            raw_context=_format_context(retrieved),
            graph_evidence=graph_evidence,
            metadata={"overridden": routing.overridden},
        )

    def compare(self, question: str, document_ids: list[str] | None = None) -> dict[str, AnswerResult]:
        results = {}
        for strategy in (RetrievalStrategy.VECTOR, RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID):
            results[strategy.value] = self.answer(question, manual_strategy=strategy, document_ids=document_ids)
        return results

