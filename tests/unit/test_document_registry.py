from __future__ import annotations

from pathlib import Path

from rag_benchmark_lab.schemas.documents import ChunkRecord, DocumentRecord, DocumentStatus
from rag_benchmark_lab.services.document_registry import DocumentRegistry


def test_registry_persists_documents_and_chunks(tmp_path: Path) -> None:
    registry = DocumentRegistry(tmp_path / "registry.sqlite3")
    document = DocumentRecord(document_id="doc-1", document_name="demo.txt", content_hash="hash-1", status=DocumentStatus.PENDING)
    registry.upsert_document(document)
    registry.store_chunks([ChunkRecord(document_id="doc-1", document_name="demo.txt", chunk_id="chunk-1", chunk_index=0, text="hello", content_hash="chunk-hash")])
    registry.update_counts("doc-1", 1, 0, 0)
    documents = registry.list_documents()
    assert documents[0]["document_id"] == "doc-1"
    assert registry.list_chunks("doc-1")[0]["chunk_id"] == "chunk-1"
