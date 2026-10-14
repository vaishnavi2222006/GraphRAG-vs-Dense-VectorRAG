from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from ..schemas.documents import Citation, GraphEvidence, RetrievedChunk, RetrievalStrategy
from .rrf import reciprocal_rank_fusion


@dataclass(slots=True)
class HybridRetrievalResult:
    chunks: list[RetrievedChunk]
    vector_chunks: list[RetrievedChunk]
    graph_evidence: GraphEvidence
    fusion_scores: dict[str, float]
    retrieval_latency_ms: float
    warnings: list[str]


def _dedupe_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    deduped: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        if chunk.chunk_id not in deduped or chunk.score > deduped[chunk.chunk_id].score:
            deduped[chunk.chunk_id] = chunk
    return sorted(deduped.values(), key=lambda item: item.score, reverse=True)


class HybridRetriever:
    def __init__(self, vector_retriever, graph_retriever, fusion_k: int = 60) -> None:
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.fusion_k = fusion_k

    def retrieve(self, question: str, document_ids: list[str] | None = None) -> HybridRetrievalResult:
        start = perf_counter()
        warnings: list[str] = []
        vector_result = self.vector_retriever.retrieve(question, document_ids=document_ids)
        graph_result = self.graph_retriever.retrieve(question, document_ids=document_ids)

        vector_ranked_ids = [chunk.chunk_id for chunk in vector_result.chunks]
        graph_ranked_ids = [chunk.chunk_id for chunk in graph_result.source_chunks]
        fusion_scores = reciprocal_rank_fusion([vector_ranked_ids, graph_ranked_ids], k=self.fusion_k)

        combined = {chunk.chunk_id: chunk for chunk in vector_result.chunks}
        for chunk in graph_result.source_chunks:
            combined.setdefault(chunk.chunk_id, chunk)

        ranked_chunks = sorted(
            combined.values(),
            key=lambda chunk: (fusion_scores.get(chunk.chunk_id, 0.0), chunk.score),
            reverse=True,
        )
        ranked_chunks = _dedupe_chunks(ranked_chunks)
        if not graph_result.source_chunks:
            warnings.append("graph_retrieval_returned_no_source_chunks")

        return HybridRetrievalResult(
            chunks=ranked_chunks,
            vector_chunks=vector_result.chunks,
            graph_evidence=graph_result,
            fusion_scores=fusion_scores,
            retrieval_latency_ms=(perf_counter() - start) * 1000.0,
            warnings=warnings,
        )

