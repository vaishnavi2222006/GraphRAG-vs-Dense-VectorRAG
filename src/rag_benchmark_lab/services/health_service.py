from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ServiceHealth:
    name: str
    healthy: bool
    details: str | None = None
    count: int | None = None


class HealthService:
    def __init__(self, qdrant_store: Any, neo4j_store: Any, llm_service: Any, registry: Any) -> None:
        self.qdrant_store = qdrant_store
        self.neo4j_store = neo4j_store
        self.llm_service = llm_service
        self.registry = registry

    def check(self) -> list[ServiceHealth]:
        results: list[ServiceHealth] = []

        # 1. Qdrant
        is_in_memory_qdrant = getattr(self.qdrant_store, "url", "").startswith(":")
        q_label = "Qdrant (In-Memory Fallback)" if is_in_memory_qdrant else "Qdrant Vector Database"
        try:
            self.qdrant_store.health()
            q_cnt = getattr(self.qdrant_store, "count", lambda: 0)()
            details = f"{q_cnt} vector points indexed in collection '{getattr(self.qdrant_store, 'collection_name', 'rag_chunks')}'"
            if is_in_memory_qdrant:
                details += " [WARNING: Using in-memory adapter]"
            results.append(ServiceHealth(name=q_label, healthy=True, details=details, count=q_cnt))
        except Exception as exc:
            results.append(ServiceHealth(name="Qdrant Vector Database", healthy=False, details=f"Qdrant: Unavailable ({exc.__class__.__name__}: {exc})"))

        # 2. Neo4j
        is_networkx = hasattr(self.neo4j_store, "_graph")
        g_label = "Graph Store (NetworkX Fallback)" if is_networkx else "Neo4j Knowledge Graph"
        try:
            self.neo4j_store.health()
            if hasattr(self.neo4j_store, "count_entities"):
                n_entities = self.neo4j_store.count_entities()
                n_rels = self.neo4j_store.count_relationships()
                details = f"{n_entities} entities, {n_rels} relationships in graph"
            else:
                details = "Neo4j graph connection established"
            if is_networkx:
                details += " [WARNING: Using NetworkX fallback adapter]"
            results.append(ServiceHealth(name=g_label, healthy=True, details=details, count=n_entities if not is_networkx else 0))
        except Exception as exc:
            results.append(ServiceHealth(name="Neo4j Knowledge Graph", healthy=False, details=f"Neo4j: Unavailable ({exc.__class__.__name__}: {exc})"))

        # 3. Ollama / LLM Service
        try:
            if getattr(self.llm_service, "mock_mode", False):
                results.append(
                    ServiceHealth(
                        name="Ollama LLM (Mock / Dev Mode)",
                        healthy=False,
                        details="LLM unavailable: running in mock/dev mode. Check the configured Ollama service and model.",
                    )
                )
            else:
                self.llm_service.health()
                model_name = getattr(self.llm_service, "model_name", "configured model")
                results.append(ServiceHealth(name="Ollama LLM Service", healthy=True, details=f"Connected. Model: {model_name}"))
        except Exception as exc:
            results.append(
                ServiceHealth(
                    name="Ollama LLM Service",
                    healthy=False,
                    details=f"LLM unavailable: check the configured Ollama service and model ({exc.__class__.__name__})",
                )
            )

        # 4. SQLite Document Registry
        try:
            counts = self.registry.counts()
            results.append(
                ServiceHealth(
                    name="SQLite Document Registry",
                    healthy=True,
                    details=f"{counts['documents']} documents, {counts['chunks']} chunks recorded",
                )
            )
        except Exception as exc:
            results.append(ServiceHealth(name="SQLite Document Registry", healthy=False, details=f"Database error: {exc}"))

        return results

    def get_real_counts(self) -> dict[str, int]:
        """Fetch actual database counts directly from live backends."""
        vector_count = 0
        if hasattr(self.qdrant_store, "count"):
            vector_count = self.qdrant_store.count()

        graph_entities = 0
        graph_relationships = 0
        if hasattr(self.neo4j_store, "count_entities"):
            graph_entities = self.neo4j_store.count_entities()
            graph_relationships = self.neo4j_store.count_relationships()

        reg_counts = self.registry.counts()
        return {
            "documents": reg_counts.get("documents", 0),
            "chunks": reg_counts.get("chunks", 0),
            "vector_points": vector_count,
            "entities": graph_entities,
            "relationships": graph_relationships,
            "evaluation_runs": reg_counts.get("evaluation_runs", 0),
        }
