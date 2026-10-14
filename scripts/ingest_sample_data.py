from __future__ import annotations

from pathlib import Path

from rag_benchmark_lab.app import build_context


def main() -> None:
    context = build_context()
    sample_dir = Path("data/sample_documents")
    for path in sorted(sample_dir.glob("*.txt")):
        outcome = context.document_service.ingest_path(path)
        print(path.name, outcome.document.status.value, outcome.entity_count, outcome.relationship_count)


if __name__ == "__main__":
    main()
