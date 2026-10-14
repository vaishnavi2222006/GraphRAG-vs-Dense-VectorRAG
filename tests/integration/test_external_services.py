from __future__ import annotations

import pytest

from rag_benchmark_lab.app import build_context


@pytest.fixture(scope="module")
def context():
    built = build_context()
    yield built
    built.qdrant_store.close()
    built.neo4j_store.close()


def test_qdrant_health_or_skip(context) -> None:
    try:
        assert context.qdrant_store.health() is True
    except Exception as exc:
        pytest.skip(f"Qdrant unavailable: {exc}")


def test_neo4j_health_or_skip(context) -> None:
    try:
        assert context.neo4j_store.health() is True
    except Exception as exc:
        pytest.skip(f"Neo4j unavailable: {exc}")


def test_ollama_health_or_skip(context) -> None:
    try:
        assert context.llm_service.health() is True
    except Exception as exc:
        pytest.skip(f"Ollama unavailable: {exc}")
