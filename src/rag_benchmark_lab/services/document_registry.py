from __future__ import annotations

from dataclasses import asdict
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from ..schemas.documents import ChunkRecord, DocumentRecord, DocumentStatus, EntityRecord, RelationshipRecord


class DocumentRegistry:
    def __init__(self, sqlite_path: Path | str) -> None:
        self.sqlite_path = Path(sqlite_path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    document_name TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    source_path TEXT,
                    mime_type TEXT,
                    file_size INTEGER,
                    status TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    entity_count INTEGER NOT NULL DEFAULT 0,
                    relationship_count INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    document_name TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    page_number INTEGER,
                    source_path TEXT,
                    ingestion_timestamp TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS entities (
                    entity_id TEXT PRIMARY KEY,
                    document_id TEXT,
                    chunk_id TEXT,
                    canonical_name TEXT NOT NULL,
                    original_mention TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    confidence REAL,
                    page_number INTEGER,
                    extraction_method TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS relationships (
                    relationship_id TEXT PRIMARY KEY,
                    source_entity_id TEXT NOT NULL,
                    target_entity_id TEXT NOT NULL,
                    source_entity_name TEXT NOT NULL,
                    target_entity_name TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    confidence REAL,
                    source_document_id TEXT,
                    source_chunk_id TEXT,
                    page_number INTEGER,
                    extraction_method TEXT NOT NULL,
                    source_text TEXT,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    result_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )

    def upsert_document(self, document: DocumentRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, document_name, content_hash, source_path, mime_type, file_size, status,
                    chunk_count, entity_count, relationship_count, error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    document_name=excluded.document_name,
                    content_hash=excluded.content_hash,
                    source_path=excluded.source_path,
                    mime_type=excluded.mime_type,
                    file_size=excluded.file_size,
                    status=excluded.status,
                    chunk_count=excluded.chunk_count,
                    entity_count=excluded.entity_count,
                    relationship_count=excluded.relationship_count,
                    error_message=excluded.error_message,
                    updated_at=excluded.updated_at
                """,
                (
                    document.document_id,
                    document.document_name,
                    document.content_hash,
                    document.source_path,
                    document.mime_type,
                    document.file_size,
                    document.status.value,
                    document.chunk_count,
                    document.entity_count,
                    document.relationship_count,
                    document.error_message,
                    document.created_at,
                    document.updated_at,
                ),
            )

    def mark_status(self, document_id: str, status: DocumentStatus, error_message: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE documents SET status = ?, error_message = ?, updated_at = datetime('now') WHERE document_id = ?",
                (status.value, error_message, document_id),
            )

    def update_counts(self, document_id: str, chunk_count: int, entity_count: int, relationship_count: int) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE documents SET chunk_count = ?, entity_count = ?, relationship_count = ?, updated_at = datetime('now') WHERE document_id = ?",
                (chunk_count, entity_count, relationship_count, document_id),
            )

    def store_chunks(self, chunks: list[ChunkRecord]) -> None:
        if not chunks:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, document_id, document_name, chunk_index, text, content_hash, page_number,
                    source_path, ingestion_timestamp, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    text=excluded.text,
                    content_hash=excluded.content_hash,
                    page_number=excluded.page_number,
                    source_path=excluded.source_path,
                    ingestion_timestamp=excluded.ingestion_timestamp,
                    metadata_json=excluded.metadata_json
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.document_name,
                        chunk.chunk_index,
                        chunk.text,
                        chunk.content_hash,
                        chunk.page_number,
                        chunk.source_path,
                        chunk.ingestion_timestamp,
                        json.dumps(chunk.metadata, ensure_ascii=False),
                    )
                    for chunk in chunks
                ],
            )

    def store_entities(self, entities: list[EntityRecord]) -> None:
        if not entities:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO entities (
                    entity_id, document_id, chunk_id, canonical_name, original_mention, entity_type,
                    confidence, page_number, extraction_method, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_id) DO UPDATE SET
                    document_id=excluded.document_id,
                    chunk_id=excluded.chunk_id,
                    canonical_name=excluded.canonical_name,
                    original_mention=excluded.original_mention,
                    entity_type=excluded.entity_type,
                    confidence=excluded.confidence,
                    page_number=excluded.page_number,
                    extraction_method=excluded.extraction_method,
                    metadata_json=excluded.metadata_json
                """,
                [
                    (
                        entity.entity_id,
                        entity.document_id,
                        entity.chunk_id,
                        entity.canonical_name,
                        entity.original_mention,
                        entity.entity_type,
                        entity.confidence,
                        entity.page_number,
                        entity.extraction_method,
                        json.dumps(entity.metadata, ensure_ascii=False),
                    )
                    for entity in entities
                ],
            )

    def store_relationships(self, relationships: list[RelationshipRecord]) -> None:
        if not relationships:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO relationships (
                    relationship_id, source_entity_id, target_entity_id, source_entity_name, target_entity_name,
                    relationship_type, confidence, source_document_id, source_chunk_id, page_number,
                    extraction_method, source_text, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(relationship_id) DO UPDATE SET
                    source_entity_id=excluded.source_entity_id,
                    target_entity_id=excluded.target_entity_id,
                    source_entity_name=excluded.source_entity_name,
                    target_entity_name=excluded.target_entity_name,
                    relationship_type=excluded.relationship_type,
                    confidence=excluded.confidence,
                    source_document_id=excluded.source_document_id,
                    source_chunk_id=excluded.source_chunk_id,
                    page_number=excluded.page_number,
                    extraction_method=excluded.extraction_method,
                    source_text=excluded.source_text,
                    metadata_json=excluded.metadata_json
                """,
                [
                    (
                        relationship.relationship_id,
                        relationship.source_entity_id,
                        relationship.target_entity_id,
                        relationship.source_entity_name,
                        relationship.target_entity_name,
                        relationship.relationship_type,
                        relationship.confidence,
                        relationship.source_document_id,
                        relationship.source_chunk_id,
                        relationship.page_number,
                        relationship.extraction_method,
                        relationship.source_text,
                        json.dumps(relationship.metadata, ensure_ascii=False),
                    )
                    for relationship in relationships
                ],
            )

    def get_document_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM documents WHERE content_hash = ?", (content_hash,)).fetchone()
            return dict(row) if row else None

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,)).fetchone()
            return dict(row) if row else None

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM documents ORDER BY updated_at DESC").fetchall()
            return [dict(row) for row in rows]

    def list_chunks(self, document_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if document_id is None:
                rows = connection.execute("SELECT * FROM chunks ORDER BY document_id, chunk_index").fetchall()
            else:
                rows = connection.execute("SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index", (document_id,)).fetchall()
            return [dict(row) for row in rows]

    def list_entities(self, document_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if document_id is None:
                rows = connection.execute("SELECT * FROM entities ORDER BY canonical_name").fetchall()
            else:
                rows = connection.execute("SELECT * FROM entities WHERE document_id = ? ORDER BY canonical_name", (document_id,)).fetchall()
            return [dict(row) for row in rows]

    def list_relationships(self, document_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if document_id is None:
                rows = connection.execute("SELECT * FROM relationships ORDER BY relationship_type").fetchall()
            else:
                rows = connection.execute("SELECT * FROM relationships WHERE source_document_id = ? ORDER BY relationship_type", (document_id,)).fetchall()
            return [dict(row) for row in rows]

    def delete_document(self, document_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM relationships WHERE source_document_id = ?", (document_id,))
            connection.execute("DELETE FROM entities WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))

    def clear_all(self) -> None:
        with self._connect() as connection:
            for table in ("relationships", "entities", "chunks", "documents", "evaluation_results", "evaluation_runs"):
                connection.execute(f"DELETE FROM {table}")

    def counts(self) -> dict[str, int]:
        with self._connect() as connection:
            return {
                "documents": connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
                "chunks": connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
                "entities": connection.execute("SELECT COUNT(*) FROM entities").fetchone()[0],
                "relationships": connection.execute("SELECT COUNT(*) FROM relationships").fetchone()[0],
                "evaluation_runs": connection.execute("SELECT COUNT(*) FROM evaluation_runs").fetchone()[0],
            }

    def save_evaluation_run(self, run_id: str, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO evaluation_runs (run_id, payload_json) VALUES (?, ?) ON CONFLICT(run_id) DO UPDATE SET payload_json = excluded.payload_json",
                (run_id, json.dumps(payload, ensure_ascii=False)),
            )

    def save_evaluation_result(self, result_id: str, run_id: str, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO evaluation_results (result_id, run_id, payload_json) VALUES (?, ?, ?) ON CONFLICT(result_id) DO UPDATE SET payload_json = excluded.payload_json",
                (result_id, run_id, json.dumps(payload, ensure_ascii=False)),
            )

    def list_evaluation_runs(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM evaluation_runs ORDER BY rowid DESC").fetchall()
            return [{"run_id": row["run_id"], **json.loads(row["payload_json"])} for row in rows]

    def list_evaluation_results(self, run_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if run_id is None:
                rows = connection.execute("SELECT * FROM evaluation_results ORDER BY rowid DESC").fetchall()
            else:
                rows = connection.execute("SELECT * FROM evaluation_results WHERE run_id = ? ORDER BY rowid DESC", (run_id,)).fetchall()
            return [{"result_id": row["result_id"], "run_id": row["run_id"], **json.loads(row["payload_json"])} for row in rows]

