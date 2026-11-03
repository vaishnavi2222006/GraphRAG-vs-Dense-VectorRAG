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

        if document_ids is not None and len(document_ids) == 0:
            evidence = GraphEvidence(entities=[], relationships=[], source_chunks=[])
            return HybridRetrievalResult(
                chunks=[],
                vector_chunks=[],
                graph_evidence=evidence,
                fusion_scores={},
                retrieval_latency_ms=(perf_counter() - start) * 1000.0,
                warnings=["empty_corpus_scope"],
            )

        vector_result = self.vector_retriever.retrieve(question, document_ids=document_ids)
        graph_result = self.graph_retriever.retrieve(question, document_ids=document_ids)

        vector_ranked_ids = [chunk.chunk_id for chunk in vector_result.chunks]
        graph_ranked_ids = [chunk.chunk_id for chunk in graph_result.source_chunks]
        fusion_scores = reciprocal_rank_fusion([vector_ranked_ids, graph_ranked_ids], k=self.fusion_k)

        combined: dict[str, RetrievedChunk] = {}
        for chunk in vector_result.chunks:
            chunk.metadata["retrieval_source"] = "vector"
            combined[chunk.chunk_id] = chunk

        for chunk in graph_result.source_chunks:
            if chunk.chunk_id in combined:
                existing = combined[chunk.chunk_id]
                existing.score = max(existing.score, chunk.score)
                existing.metadata["retrieval_source"] = "hybrid_both"
            else:
                chunk.metadata["retrieval_source"] = "graph"
                combined[chunk.chunk_id] = chunk

        def _sort_key(chunk: RetrievedChunk):
            fs = fusion_scores.get(chunk.chunk_id, 0.0)
            # Prevent irrelevant graph-only chunks from displacing relevant vector matches
            if chunk.metadata.get("retrieval_source") == "graph" and chunk.score < 0.15:
                fs = fs * 0.5
            return (fs, chunk.score)

        ranked_chunks = sorted(combined.values(), key=_sort_key, reverse=True)
        for chunk in ranked_chunks:
            chunk.metadata["rrf_score"] = fusion_scores.get(chunk.chunk_id, 0.0)

        if not vector_result.chunks:
            warnings.append("vector_retrieval_returned_no_chunks")
        if not graph_result.source_chunks:
            warnings.append("graph_retrieval_returned_no_source_chunks")
        warnings.extend(graph_result.warnings)

        return HybridRetrievalResult(
            chunks=ranked_chunks,
            vector_chunks=vector_result.chunks,
            graph_evidence=graph_result.evidence,
            fusion_scores=fusion_scores,
            retrieval_latency_ms=(perf_counter() - start) * 1000.0,
            warnings=warnings,
        )

