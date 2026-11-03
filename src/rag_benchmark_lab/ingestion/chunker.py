from __future__ import annotations

import hashlib
import re
from dataclasses import asdict
from typing import Iterable

from ..schemas.documents import ChunkRecord


def _find_breakpoint(text: str, start: int, max_end: int, min_end: int) -> int:
    """Find a natural semantic breakpoint between min_end and max_end."""
    slice_to_search = text[min_end:max_end]
    # Priority 1: Paragraph boundary
    pos = slice_to_search.rfind("\n\n")
    if pos != -1:
        return min_end + pos + 2
    # Priority 2: Single newline
    pos = slice_to_search.rfind("\n")
    if pos != -1:
        return min_end + pos + 1
    # Priority 3: Sentence terminators (. ! ?) followed by space or end
    for match in re.finditer(r"[.!?](?:\s|$)", slice_to_search):
        pass  # find the last match
    try:
        last_match = list(re.finditer(r"[.!?](?:\s|$)", slice_to_search))[-1]
        return min_end + last_match.end()
    except IndexError:
        pass
    # Priority 4: Semicolon or colon
    pos = max(slice_to_search.rfind("; "), slice_to_search.rfind(": "))
    if pos != -1:
        return min_end + pos + 2
    # Priority 5: Space/word boundary
    pos = slice_to_search.rfind(" ")
    if pos != -1:
        return min_end + pos + 1
    # Fallback to max_end if no natural boundary is found
    return max_end


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> Iterable[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    start = 0
    text_length = len(text)
    while start < text_length:
        if start + chunk_size >= text_length:
            chunk = text[start:].strip()
            if chunk:
                yield chunk
            break

        max_end = start + chunk_size
        min_end = max(start + 1, max_end - chunk_overlap)
        break_point = _find_breakpoint(text, start, max_end, min_end)

        chunk = text[start:break_point].strip()
        if chunk:
            yield chunk

        # Advance start respecting chunk_overlap from the actual break_point
        start = max(break_point - chunk_overlap, start + 1)


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

