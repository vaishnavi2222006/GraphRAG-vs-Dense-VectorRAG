from __future__ import annotations

from rag_benchmark_lab.app import build_context


def main() -> None:
    context = build_context()
    try:
        context.qdrant_store.ensure_collection(context.embedding_service.dimension)
    except Exception as exc:
        print(f"Qdrant initialization failed: {exc}")
    try:
        context.neo4j_store.ensure_schema()
    except Exception as exc:
        print(f"Neo4j initialization failed: {exc}")


if __name__ == "__main__":
    main()
