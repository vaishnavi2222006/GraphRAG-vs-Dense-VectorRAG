from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Iterable

from ..schemas.documents import ChunkRecord


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> Iterable[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(text_length, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= text_length:
            break
        start = max(end - chunk_overlap, start + 1)


def chunk_document(
    *,
    document_id: str,
    document_name: str,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    source_path: str | None = None,
    page_number: int | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> list[ChunkRecord]:
    metadata = dict(extra_metadata or {})
    chunks: list[ChunkRecord] = []
    for index, chunk_text in enumerate(_chunk_text(text, chunk_size, chunk_overlap)):
        content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        chunk_id = hashlib.sha256(f"{document_id}:{index}:{content_hash}".encode("utf-8")).hexdigest()
        chunks.append(
            ChunkRecord(
                document_id=document_id,
                document_name=document_name,
                chunk_id=chunk_id,
                chunk_index=index,
                text=chunk_text,
                content_hash=content_hash,
                page_number=page_number,
                source_path=source_path,
                metadata={**metadata, "chunk_length": len(chunk_text)},
            )
        )
    return chunks


def chunk_count(text: str, chunk_size: int, chunk_overlap: int) -> int:
    return sum(1 for _ in _chunk_text(text, chunk_size, chunk_overlap))

