from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"


class RetrievalStrategy(str, Enum):
    VECTOR = "VECTOR"
    GRAPH = "GRAPH"
    HYBRID = "HYBRID"
    ALL = "ALL"
    DIRECT = "DIRECT"



@dataclass(slots=True)
class DocumentRecord:
    document_id: str
    document_name: str
    content_hash: str
    source_path: str | None = None
    mime_type: str | None = None
    file_size: int | None = None
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    entity_count: int = 0
    relationship_count: int = 0
    error_message: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(slots=True)
class ChunkRecord:
    document_id: str
    document_name: str
    chunk_id: str
    chunk_index: int
    text: str
    content_hash: str
    page_number: int | None = None
    source_path: str | None = None
    ingestion_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_name: str
    chunk_index: int
    text: str
    score: float
    page_number: int | None = None
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EntityRecord:
    entity_id: str
    canonical_name: str
    original_mention: str
    entity_type: str
    confidence: float | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    page_number: int | None = None
    extraction_method: str = "llm"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RelationshipRecord:
    relationship_id: str
    source_entity_id: str
    source_entity_name: str
    target_entity_id: str
    target_entity_name: str
    relationship_type: str
    confidence: float | None = None
    source_document_id: str | None = None
    source_chunk_id: str | None = None
    page_number: int | None = None
    extraction_method: str = "llm"
    source_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphEvidence:
    entities: list[EntityRecord]
    relationships: list[RelationshipRecord]
    source_chunks: list[RetrievedChunk] = field(default_factory=list)


@dataclass(slots=True)
class Citation:
    document_id: str
    document_name: str
    chunk_id: str
    chunk_index: int
    page_number: int | None = None
    score: float | None = None


@dataclass(slots=True)
class AnswerResult:
    answer: str
    selected_strategy: RetrievalStrategy
    citations: list[Citation] = field(default_factory=list)
    retrieved_sources: list[RetrievedChunk] = field(default_factory=list)
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    total_latency_ms: float | None = None
    model_name: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated_cost: float | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    routing_rationale: str | None = None
    routing_confidence: float | None = None
    raw_context: str | None = None
    graph_evidence: GraphEvidence | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
