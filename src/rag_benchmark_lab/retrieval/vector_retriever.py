from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from ..embeddings.embedding_service import EmbeddingService
from ..schemas.documents import RetrievedChunk
from ..vector_store.qdrant_store import QdrantStore


@dataclass(slots=True)
class VectorRetrievalResult:
    chunks: list[RetrievedChunk]
    retrieval_latency_ms: float


class VectorRetriever:
    def __init__(self, embedding_service: EmbeddingService, qdrant_store: QdrantStore, top_k: int) -> None:
        self.embedding_service = embedding_service
        self.qdrant_store = qdrant_store
        self.top_k = top_k

    def retrieve(self, question: str, document_ids: list[str] | None = None) -> VectorRetrievalResult:
        start = perf_counter()
        query_vector = self.embedding_service.embed_query(question)
        chunks = self.qdrant_store.search(query_vector, self.top_k, document_ids=document_ids)
        return VectorRetrievalResult(chunks=chunks, retrieval_latency_ms=(perf_counter() - start) * 1000.0)

