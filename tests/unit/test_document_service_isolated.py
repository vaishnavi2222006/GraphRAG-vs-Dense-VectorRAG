from __future__ import annotations

import hashlib
from pathlib import Path
import pytest

from rag_benchmark_lab.schemas.documents import ChunkRecord, DocumentRecord, DocumentStatus, EntityRecord, RelationshipRecord
from rag_benchmark_lab.services.document_registry import DocumentRegistry
from rag_benchmark_lab.services.document_service import DocumentService
from rag_benchmark_lab.utils.errors import IngestionError


class FakeEmbeddingService:
    dimension = 384

    def embed_texts(self, texts: list[str]):
        return type("EmbRes", (), {"dimension": 384, "vectors": [[0.1] * 384 for _ in texts]})()


class FakeQdrantStore:
    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.points: dict[str, list] = {}

    def ensure_collection(self, dimension: int) -> None:
        if self.should_fail:
            raise RuntimeError("Qdrant connection timeout")

    def upsert_chunks(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        if self.should_fail:
            raise RuntimeError("Qdrant upsert failed")
        self.points["chunks"] = chunks

    def delete_document(self, document_id: str) -> None:
        if self.should_fail:
            raise RuntimeError("Qdrant delete failed")
        self.points.pop("chunks", None)


class FakeNeo4jStore:
    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.entities = []
        self.relationships = []

    def upsert_document(self, document_id: str, document_name: str) -> None:
        if self.should_fail:
            raise RuntimeError("Neo4j connection error")

    def upsert_entities(self, entities: list[EntityRecord]) -> None:
        if self.should_fail:
            raise RuntimeError("Neo4j entity upsert error")
        self.entities.extend(entities)

    def upsert_relationships(self, relationships: list[RelationshipRecord]) -> None:
        if self.should_fail:
            raise RuntimeError("Neo4j relation upsert error")
        self.relationships.extend(relationships)

    def delete_document(self, document_id: str) -> None:
        if self.should_fail:
            raise RuntimeError("Neo4j delete error")
        self.entities = [e for e in self.entities if e.document_id != document_id]
        self.relationships = [r for r in self.relationships if r.source_document_id != document_id]


class FakeExtractor:
    def extract(self, text: str, document_id=None, chunk_id=None, page_number=None):
        entity = EntityRecord(
            entity_id="e1",
            canonical_name="Qdrant",
            original_mention="Qdrant",
            entity_type="Technology",
            confidence=0.9,
            document_id=document_id,
            chunk_id=chunk_id,
            page_number=page_number,
            extraction_method="heuristic",
        )
        relation = RelationshipRecord(
            relationship_id="r1",
            source_entity_id="e1",
            source_entity_name="Qdrant",
            target_entity_id="e2",
            target_entity_name="Neo4j",
            relationship_type="CONNECTS_TO",
            confidence=0.85,
            source_document_id=document_id,
            source_chunk_id=chunk_id,
            page_number=page_number,
            extraction_method="heuristic",
            source_text=text,
        )
        return type("ExtRes", (), {"entities": [entity], "relationships": [relation]})()


def test_document_ingestion_and_deletion(tmp_path: Path) -> None:
    registry = DocumentRegistry(tmp_path / "test_reg.sqlite3")
    emb = FakeEmbeddingService()
    qdrant = FakeQdrantStore()
    neo4j = FakeNeo4jStore()
    extractor = FakeExtractor()

    service = DocumentService(
        registry=registry,
        embedding_service=emb,
        qdrant_store=qdrant,
        neo4j_store=neo4j,
        extractor=extractor,
        chunk_size=500,
        chunk_overlap=50,
        upload_size_limit_mb=10,
    )

    doc_text = "Qdrant and Neo4j provide dual-indexing capabilities."
    outcome = service.ingest_bytes("test_doc.txt", doc_text.encode("utf-8"))

    assert outcome.document.status == DocumentStatus.INDEXED
    assert outcome.document.chunk_count == 1
    assert outcome.entity_count == 1
    assert outcome.relationship_count == 1
    assert len(outcome.errors) == 0

    # Verify duplicate detection
    dup_outcome = service.ingest_bytes("test_doc.txt", doc_text.encode("utf-8"))
    assert "duplicate_document_skipped" in dup_outcome.warnings

    # Test deletion cascade
    del_warnings = service.delete_document(outcome.document.document_id)
    assert len(del_warnings) == 0
    assert len(registry.list_documents()) == 0
    assert len(registry.list_chunks()) == 0


def test_failure_handling_when_vector_store_fails(tmp_path: Path) -> None:
    registry = DocumentRegistry(tmp_path / "test_reg_fail.sqlite3")
    service = DocumentService(
        registry=registry,
        embedding_service=FakeEmbeddingService(),
        qdrant_store=FakeQdrantStore(should_fail=True),
        neo4j_store=FakeNeo4jStore(),
        extractor=FakeExtractor(),
        chunk_size=500,
        chunk_overlap=50,
        upload_size_limit_mb=10,
    )

    doc_text = "Failure handling verification text."
    outcome = service.ingest_bytes("fail_doc.txt", doc_text.encode("utf-8"))

    # When one pipeline fails, document must be marked PARTIAL, not INDEXED
    assert outcome.document.status == DocumentStatus.PARTIAL
    assert any("vector_index_failure" in err for err in outcome.errors)


def test_upload_size_limit_enforcement(tmp_path: Path) -> None:
    registry = DocumentRegistry(tmp_path / "test_reg_limit.sqlite3")
    service = DocumentService(
        registry=registry,
        embedding_service=FakeEmbeddingService(),
        qdrant_store=FakeQdrantStore(),
        neo4j_store=FakeNeo4jStore(),
        extractor=FakeExtractor(),
        chunk_size=500,
        chunk_overlap=50,
        upload_size_limit_mb=1,  # 1 MB limit
    )

    oversized_bytes = b"x" * (2 * 1024 * 1024)  # 2 MB
    with pytest.raises(ValueError):
        service.ingest_bytes("big.txt", oversized_bytes)
