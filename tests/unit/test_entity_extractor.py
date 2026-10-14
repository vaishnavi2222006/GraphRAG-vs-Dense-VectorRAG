from __future__ import annotations

from dataclasses import dataclass

from rag_benchmark_lab.graph_store.entity_extractor import EntityRelationshipExtractor, normalize_entity_name


@dataclass
class FakeLLM:
    text: str

    def complete(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0):
        return type("Response", (), {"text": self.text})()


def test_normalize_entity_name_collapses_whitespace() -> None:
    assert normalize_entity_name("  Graph   RAG  ") == "Graph RAG"


def test_entity_extractor_parses_structured_output() -> None:
    llm = FakeLLM(
        text='{"entities": [{"canonical_name": "Neo4j", "original_mention": "Neo4j", "entity_type": "Technology", "confidence": 0.9}], "relationships": []}'
    )
    extractor = EntityRelationshipExtractor(llm)  # type: ignore[arg-type]
    result = extractor.extract("Neo4j stores graph data.")
    assert result.entities[0].canonical_name == "Neo4j"
    assert result.used_fallback is False


def test_entity_extractor_falls_back_on_malformed_output() -> None:
    llm = FakeLLM(text="not json")
    extractor = EntityRelationshipExtractor(llm)  # type: ignore[arg-type]
    result = extractor.extract("Graph RAG uses Neo4j and Qdrant.")
    assert result.used_fallback is True
