from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from ..embeddings.embedding_service import EmbeddingService
from ..graph_store.entity_extractor import EntityRelationshipExtractor
from ..graph_store.neo4j_store import Neo4jStore
from ..ingestion.chunker import chunk_document
from ..ingestion.document_loader import load_bytes, validate_upload_size
from ..schemas.documents import ChunkRecord, DocumentRecord, DocumentStatus
from ..utils.errors import IngestionError
from ..vector_store.qdrant_store import QdrantStore
from .document_registry import DocumentRegistry


@dataclass(slots=True)
class IngestionOutcome:
    document: DocumentRecord
    chunks: list[ChunkRecord]
    entity_count: int
    relationship_count: int
    warnings: list[str]
    errors: list[str]


class DocumentService:
    def __init__(
        self,
        registry: DocumentRegistry,
        embedding_service: EmbeddingService,
        qdrant_store: QdrantStore,
        neo4j_store: Neo4jStore,
        extractor: EntityRelationshipExtractor,
        chunk_size: int,
        chunk_overlap: int,
        upload_size_limit_mb: int,
    ) -> None:
        self.registry = registry
        self.embedding_service = embedding_service
        self.qdrant_store = qdrant_store
        self.neo4j_store = neo4j_store
        self.extractor = extractor
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.upload_size_limit_mb = upload_size_limit_mb

    def ingest_bytes(self, document_name: str, raw_bytes: bytes, source_path: str | None = None) -> IngestionOutcome:
        validate_upload_size(raw_bytes, self.upload_size_limit_mb)
        loaded = load_bytes(document_name, raw_bytes, source_path=source_path)
        content_hash = loaded.content_hash
        existing = self.registry.get_document_by_hash(content_hash)
        if existing and existing.get("status") == DocumentStatus.INDEXED.value:
            document = DocumentRecord(
                document_id=existing["document_id"],
                document_name=existing["document_name"],
                content_hash=existing["content_hash"],
                source_path=existing.get("source_path"),
                mime_type=existing.get("mime_type"),
                file_size=existing.get("file_size"),
                status=DocumentStatus.INDEXED,
                chunk_count=existing.get("chunk_count", 0),
                entity_count=existing.get("entity_count", 0),
                relationship_count=existing.get("relationship_count", 0),
                error_message=existing.get("error_message"),
                created_at=existing.get("created_at", datetime.now(timezone.utc).isoformat()),
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            return IngestionOutcome(document=document, chunks=[], entity_count=document.entity_count, relationship_count=document.relationship_count, warnings=["duplicate_document_skipped"], errors=[])

        document_id = hashlib.sha256(content_hash.encode("utf-8")).hexdigest()
        document = DocumentRecord(
            document_id=document_id,
            document_name=document_name,
            content_hash=content_hash,
            source_path=source_path,
            mime_type=loaded.mime_type,
            file_size=len(raw_bytes),
            status=DocumentStatus.PENDING,
        )
        self.registry.upsert_document(document)

        chunks: list[ChunkRecord] = []
        entities = []
        relationships = []
        warnings: list[str] = []
        errors: list[str] = []
        try:
            for page in loaded.pages:
                page_chunks = chunk_document(
                    document_id=document_id,
                    document_name=document_name,
                    text=page.text,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    source_path=source_path,
                    page_number=page.page_number,
                    extra_metadata={"mime_type": loaded.mime_type, "page_number": page.page_number},
                )
                chunks.extend(page_chunks)
            self.registry.store_chunks(chunks)

            embedding_result = self.embedding_service.embed_texts([chunk.text for chunk in chunks])
            self.qdrant_store.ensure_collection(embedding_result.dimension)
            self.qdrant_store.upsert_chunks(chunks, embedding_result.vectors)
        except Exception as exc:
            errors.append(f"vector_index_failure:{exc.__class__.__name__}")

        try:
            self.neo4j_store.upsert_document(document_id, document_name)
            for chunk in chunks:
                extraction = self.extractor.extract(chunk.text, document_id=document_id, chunk_id=chunk.chunk_id, page_number=chunk.page_number)
                entities.extend(extraction.entities)
                relationships.extend(extraction.relationships)
            self.registry.store_entities(entities)
            self.registry.store_relationships(relationships)
            self.neo4j_store.upsert_entities(entities)
            self.neo4j_store.upsert_relationships(relationships)
        except Exception as exc:
            errors.append(f"graph_index_failure:{exc.__class__.__name__}")

        document.chunk_count = len(chunks)
        document.entity_count = len(entities)
        document.relationship_count = len(relationships)
        document.updated_at = datetime.now(timezone.utc).isoformat()

        if errors:
            document.status = DocumentStatus.PARTIAL if chunks else DocumentStatus.FAILED
            document.error_message = "; ".join(errors)
        else:
            document.status = DocumentStatus.INDEXED
        self.registry.upsert_document(document)
        self.registry.update_counts(document.document_id, document.chunk_count, document.entity_count, document.relationship_count)
        self.registry.mark_status(document.document_id, document.status, document.error_message)

        return IngestionOutcome(document=document, chunks=chunks, entity_count=len(entities), relationship_count=len(relationships), warnings=warnings, errors=errors)

    def ingest_path(self, path: str | Path) -> IngestionOutcome:
        resolved = Path(path)
        raw_bytes = resolved.read_bytes()
        return self.ingest_bytes(resolved.name, raw_bytes, source_path=str(resolved))

    def delete_document(self, document_id: str) -> list[str]:
        warnings: list[str] = []
        errors: list[str] = []
        try:
            self.qdrant_store.delete_document(document_id)
        except Exception as exc:
            errors.append(f"vector_delete_failure:{exc.__class__.__name__}")
        try:
            self.neo4j_store.delete_document(document_id)
        except Exception as exc:
            errors.append(f"graph_delete_failure:{exc.__class__.__name__}")
        self.registry.delete_document(document_id)
        if errors:
            warnings.extend(errors)
        return warnings

    def clear_corpus(self) -> None:
        try:
            client = self.qdrant_store._get_client()
            if client.collection_exists(self.qdrant_store.collection_name):
                client.delete_collection(self.qdrant_store.collection_name)
        except Exception:
            pass
        try:
            driver = self.neo4j_store._get_driver()
            with driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
        except Exception:
            pass
        self.registry.clear_all()

