from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

try:
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.http import models as qmodels  # type: ignore
except Exception:  # pragma: no cover
    QdrantClient = None
    qmodels = None

from ..schemas.documents import ChunkRecord, RetrievedChunk
from ..utils.errors import ServiceUnavailableError


def _to_point_id(chunk_id: str) -> str:
    import uuid
    try:
        return str(uuid.UUID(chunk_id))
    except (ValueError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, str(chunk_id)))


class QdrantStore:
    def __init__(self, url: str, api_key: str | None, collection_name: str) -> None:
        self.url = url
        self.api_key = api_key or None
        self.collection_name = collection_name
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if QdrantClient is None:
            raise ServiceUnavailableError("qdrant-client is not installed")
        if self.url in (":memory:", "memory", ":local:"):
            # Fully in-memory mode — no network connection needed
            self._client = QdrantClient(":memory:")
        else:
            self._client = QdrantClient(url=self.url, api_key=self.api_key, check_compatibility=False)
        return self._client

    def close(self) -> None:
        client = self._client
        if client is not None:
            client.close()
            self._client = None

    def health(self) -> bool:
        client = self._get_client()
        client.get_collections()
        return True

    def count(self) -> int:
        try:
            client = self._get_client()
            if not client.collection_exists(self.collection_name):
                return 0
            info = client.get_collection(self.collection_name)
            return int(info.points_count or 0)
        except Exception:
            return 0

    def ensure_collection(self, vector_size: int) -> None:
        client = self._get_client()
        if qmodels is None:
            raise ServiceUnavailableError("qdrant-client is not installed")
        existing = client.collection_exists(self.collection_name)
        if existing:
            info = client.get_collection(self.collection_name)
            current_size = info.config.params.vectors.size
            if current_size != vector_size:
                client.delete_collection(self.collection_name)
                existing = False
        if not existing:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
            )

    def upsert_chunks(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        client = self._get_client()
        points = []
        for chunk, vector in zip(chunks, embeddings, strict=True):
            payload = asdict(chunk)
            points.append(
                qmodels.PointStruct(
                    id=_to_point_id(chunk.chunk_id),
                    vector=vector,
                    payload=payload,
                )
            )
        client.upsert(collection_name=self.collection_name, points=points)

    def search(self, vector: list[float], top_k: int, document_ids: list[str] | None = None) -> list[RetrievedChunk]:
        client = self._get_client()
        # If collection does not exist yet (no documents ingested), return empty results
        try:
            if not client.collection_exists(self.collection_name):
                return []
        except Exception:
            return []
        filter_ = None
        if document_ids and qmodels is not None:
            filter_ = qmodels.Filter(
                must=[qmodels.FieldCondition(key="document_id", match=qmodels.MatchAny(any=document_ids))]
            )
        try:
            if hasattr(client, "query_points"):
                response = client.query_points(
                    collection_name=self.collection_name,
                    query=vector,
                    limit=top_k,
                    query_filter=filter_,
                )
                results = response.points
            else:
                results = client.search(
                    collection_name=self.collection_name,
                    query_vector=vector,
                    limit=top_k,
                    query_filter=filter_,
                )
        except Exception:
            return []
        retrieved: list[RetrievedChunk] = []
        for result in results:
            payload = result.payload or {}
            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(payload.get("chunk_id") or result.id),
                    document_id=str(payload.get("document_id", "")),
                    document_name=str(payload.get("document_name", "")),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    text=str(payload.get("text", "")),
                    score=float(result.score or 0.0),
                    page_number=payload.get("page_number"),
                    source_path=payload.get("source_path"),
                    metadata=payload.get("metadata", {}) or {},
                )
            )
        return retrieved

    def delete_document(self, document_id: str) -> None:
        client = self._get_client()
        if qmodels is None:
            raise ServiceUnavailableError("qdrant-client is not installed")
        client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(must=[qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=document_id))])
            ),
        )

