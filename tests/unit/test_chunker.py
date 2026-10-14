from __future__ import annotations

from rag_benchmark_lab.ingestion.chunker import chunk_count, chunk_document


def test_chunk_document_preserves_metadata() -> None:
    chunks = chunk_document(
        document_id="doc-1",
        document_name="demo.txt",
        text="abcdefghij" * 20,
        chunk_size=50,
        chunk_overlap=10,
        source_path="/tmp/demo.txt",
        page_number=2,
        extra_metadata={"mime_type": "text/plain"},
    )
    assert chunks
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].page_number == 2
    assert chunks[0].metadata["mime_type"] == "text/plain"


def test_chunk_count_matches_chunk_document() -> None:
    text = "abcdefghij" * 20
    assert chunk_count(text, 50, 10) == len(chunk_document(document_id="doc", document_name="a.txt", text=text, chunk_size=50, chunk_overlap=10))
