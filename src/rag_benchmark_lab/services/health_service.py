from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ServiceHealth:
    name: str
    healthy: bool
    details: str | None = None


class HealthService:
    def __init__(self, qdrant_store, neo4j_store, llm_service, registry) -> None:
        self.qdrant_store = qdrant_store
        self.neo4j_store = neo4j_store
        self.llm_service = llm_service
        self.registry = registry

    def check(self) -> list[ServiceHealth]:
        results: list[ServiceHealth] = []
        for name, checker in (
            ("qdrant", self.qdrant_store.health),
            ("neo4j", self.neo4j_store.health),
            ("ollama", self.llm_service.health),
        ):
            try:
                checker()
                results.append(ServiceHealth(name=name, healthy=True))
            except Exception as exc:
                results.append(ServiceHealth(name=name, healthy=False, details=f"{exc.__class__.__name__}: {exc}"))
        return results

