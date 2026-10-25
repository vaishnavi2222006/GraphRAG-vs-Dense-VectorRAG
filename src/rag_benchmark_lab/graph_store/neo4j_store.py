from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:  # pragma: no cover
    GraphDatabase = None

from ..schemas.documents import EntityRecord, GraphEvidence, RelationshipRecord, RetrievedChunk
from ..utils.errors import ServiceUnavailableError


class Neo4jStore:
    def __init__(self, uri: str, username: str, password: str) -> None:
        self.uri = uri
        self.username = username
        self.password = password
        self._driver = None

    def _get_driver(self):
        if self._driver is not None:
            return self._driver
        if GraphDatabase is None:
            raise ServiceUnavailableError("neo4j driver is not installed")
        self._driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        return self._driver

    def close(self) -> None:
        driver = self._driver
        if driver is not None:
            driver.close()
            self._driver = None

    def health(self) -> bool:
        driver = self._get_driver()
        with driver.session() as session:
            session.run("RETURN 1 AS ok").single()
        return True

    def ensure_schema(self) -> None:
        driver = self._get_driver()
        statements = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
            "CREATE CONSTRAINT relationship_id_unique IF NOT EXISTS FOR ()-[r:RELATED_TO]-() REQUIRE r.relationship_id IS UNIQUE",
        ]
        with driver.session() as session:
            for statement in statements:
                session.run(statement)

    def upsert_entities(self, entities: list[EntityRecord]) -> None:
        if not entities:
            return
        driver = self._get_driver()
        query = (
            "UNWIND $entities AS entity "
            "MERGE (e:Entity {entity_id: entity.entity_id}) "
            "SET e.canonical_name = entity.canonical_name, "
            "    e.original_mention = entity.original_mention, "
            "    e.entity_type = entity.entity_type, "
            "    e.confidence = entity.confidence, "
            "    e.document_id = entity.document_id, "
            "    e.chunk_id = entity.chunk_id, "
            "    e.page_number = entity.page_number, "
            "    e.extraction_method = entity.extraction_method, "
            "    e.updated_at = datetime()"
        )
        with driver.session() as session:
            session.run(query, entities=[asdict(entity) for entity in entities])

    def upsert_relationships(self, relationships: list[RelationshipRecord]) -> None:
        if not relationships:
            return
        driver = self._get_driver()
        query = (
            "UNWIND $relationships AS relationship "
            "MATCH (source:Entity {entity_id: relationship.source_entity_id}) "
            "MATCH (target:Entity {entity_id: relationship.target_entity_id}) "
            "MERGE (source)-[r:RELATED_TO {relationship_id: relationship.relationship_id}]->(target) "
            "SET r.relationship_type = relationship.relationship_type, "
            "    r.confidence = relationship.confidence, "
            "    r.source_document_id = relationship.source_document_id, "
            "    r.source_chunk_id = relationship.source_chunk_id, "
            "    r.page_number = relationship.page_number, "
            "    r.extraction_method = relationship.extraction_method, "
            "    r.source_text = relationship.source_text, "
            "    r.updated_at = datetime()"
        )
        with driver.session() as session:
            session.run(query, relationships=[asdict(relationship) for relationship in relationships])

    def upsert_document(self, document_id: str, document_name: str) -> None:
        driver = self._get_driver()
        with driver.session() as session:
            session.run(
                "MERGE (d:Document {document_id: $document_id}) SET d.document_name = $document_name, d.updated_at = datetime()",
                document_id=document_id,
                document_name=document_name,
            )

    def delete_document(self, document_id: str) -> None:
        driver = self._get_driver()
        with driver.session() as session:
            session.run(
                "MATCH ()-[r:RELATED_TO]->() WHERE r.source_document_id = $document_id DELETE r",
                document_id=document_id,
            )
            session.run(
                "MATCH (e:Entity) WHERE e.document_id = $document_id DETACH DELETE e",
                document_id=document_id,
            )
            session.run(
                "MATCH (d:Document {document_id: $document_id}) DETACH DELETE d",
                document_id=document_id,
            )

    def search_entities(self, terms: list[str], limit: int = 10) -> list[EntityRecord]:
        driver = self._get_driver()
        query = (
            "MATCH (e:Entity) "
            "WHERE any(term IN $terms WHERE toLower(e.canonical_name) CONTAINS term OR toLower(e.original_mention) CONTAINS term) "
            "RETURN e ORDER BY coalesce(e.confidence, 0) DESC, e.canonical_name ASC LIMIT $limit"
        )
        with driver.session() as session:
            results = session.run(query, terms=[term.lower() for term in terms if term], limit=limit)
            records: list[EntityRecord] = []
            for record in results:
                entity = record["e"]
                records.append(
                    EntityRecord(
                        entity_id=entity.get("entity_id"),
                        canonical_name=entity.get("canonical_name"),
                        original_mention=entity.get("original_mention", entity.get("canonical_name")),
                        entity_type=entity.get("entity_type", "Concept"),
                        confidence=entity.get("confidence"),
                        document_id=entity.get("document_id"),
                        chunk_id=entity.get("chunk_id"),
                        page_number=entity.get("page_number"),
                        extraction_method=entity.get("extraction_method", "llm"),
                    )
                )
            return records

    def traverse(self, entity_ids: list[str], max_hops: int = 2, limit: int = 20) -> GraphEvidence:
        driver = self._get_driver()
        safe_hops = max(1, min(int(max_hops), 5))
        query = (
            f"MATCH path = (start:Entity)-[rels:RELATED_TO*1..{safe_hops}]-(other:Entity) "
            "WHERE start.entity_id IN $entity_ids "
            "RETURN path LIMIT $limit"
        )
        seed_query = (
            "MATCH (e:Entity) WHERE e.entity_id IN $entity_ids RETURN e"
        )
        entities: dict[str, EntityRecord] = {}
        relationships: dict[str, RelationshipRecord] = {}
        source_chunks: dict[str, RetrievedChunk] = {}
        with driver.session() as session:
            seed_results = session.run(seed_query, entity_ids=entity_ids)
            for record in seed_results:
                node = record["e"]
                entity_id = node.get("entity_id")
                if entity_id:
                    entities[entity_id] = EntityRecord(
                        entity_id=entity_id,
                        canonical_name=node.get("canonical_name", ""),
                        original_mention=node.get("original_mention", node.get("canonical_name", "")),
                        entity_type=node.get("entity_type", "Concept"),
                        confidence=node.get("confidence"),
                        document_id=node.get("document_id"),
                        chunk_id=node.get("chunk_id"),
                        page_number=node.get("page_number"),
                        extraction_method=node.get("extraction_method", "llm"),
                    )
            results = session.run(query, entity_ids=entity_ids, limit=limit)
            for record in results:
                path = record["path"]
                for node in path.nodes:
                    if "entity_id" not in node:
                        continue
                    entity_id = node["entity_id"]
                    entities.setdefault(
                        entity_id,
                        EntityRecord(
                            entity_id=entity_id,
                            canonical_name=node.get("canonical_name", ""),
                            original_mention=node.get("original_mention", node.get("canonical_name", "")),
                            entity_type=node.get("entity_type", "Concept"),
                            confidence=node.get("confidence"),
                            document_id=node.get("document_id"),
                            chunk_id=node.get("chunk_id"),
                            page_number=node.get("page_number"),
                            extraction_method=node.get("extraction_method", "llm"),
                        ),
                    )
                for relationship in path.relationships:
                    relationship_id = relationship.get("relationship_id")
                    if relationship_id in relationships:
                        continue
                    relationships[relationship_id] = RelationshipRecord(
                        relationship_id=relationship_id,
                        source_entity_id=relationship.start_node["entity_id"],
                        source_entity_name=relationship.start_node.get("canonical_name", ""),
                        target_entity_id=relationship.end_node["entity_id"],
                        target_entity_name=relationship.end_node.get("canonical_name", ""),
                        relationship_type=relationship.get("relationship_type", "RELATED_TO"),
                        confidence=relationship.get("confidence"),
                        source_document_id=relationship.get("source_document_id"),
                        source_chunk_id=relationship.get("source_chunk_id"),
                        page_number=relationship.get("page_number"),
                        extraction_method=relationship.get("extraction_method", "llm"),
                        source_text=relationship.get("source_text"),
                    )
                    chunk_id = relationship.get("source_chunk_id")
                    if chunk_id and chunk_id not in source_chunks:
                        source_chunks[chunk_id] = RetrievedChunk(
                            chunk_id=chunk_id,
                            document_id=str(relationship.get("source_document_id") or ""),
                            document_name=str(relationship.get("source_document_id") or ""),
                            chunk_index=0,
                            text=str(relationship.get("source_text") or ""),
                            score=float(relationship.get("confidence") or 0.0),
                            page_number=relationship.get("page_number"),
                            metadata={"relationship_type": relationship.get("relationship_type", "RELATED_TO")},
                        )
        return GraphEvidence(entities=list(entities.values()), relationships=list(relationships.values()), source_chunks=list(source_chunks.values()))

    def count_entities(self) -> int:
        try:
            driver = self._get_driver()
            with driver.session() as session:
                rec = session.run("MATCH (e:Entity) RETURN count(e) AS cnt").single()
                return int(rec["cnt"]) if rec else 0
        except Exception:
            return 0

    def count_relationships(self) -> int:
        try:
            driver = self._get_driver()
            with driver.session() as session:
                rec = session.run("MATCH ()-[r:RELATED_TO]->() RETURN count(r) AS cnt").single()
                return int(rec["cnt"]) if rec else 0
        except Exception:
            return 0

    def count_documents(self) -> int:
        try:
            driver = self._get_driver()
            with driver.session() as session:
                rec = session.run("MATCH (d:Document) RETURN count(d) AS cnt").single()
                return int(rec["cnt"]) if rec else 0
        except Exception:
            return 0

    def list_entities(self, entity_type: str | None = None, search: str | None = None, limit: int = 150) -> list[dict]:
        try:
            driver = self._get_driver()
            clauses = ["MATCH (e:Entity)"]
            where = []
            params: dict[str, object] = {"limit": limit}
            if entity_type and entity_type != "All":
                where.append("e.entity_type = $entity_type")
                params["entity_type"] = entity_type
            if search and search.strip():
                where.append("toLower(e.canonical_name) CONTAINS toLower($search)")
                params["search"] = search.strip()
            if where:
                clauses.append("WHERE " + " AND ".join(where))
            clauses.append("RETURN e.entity_id AS entity_id, e.canonical_name AS canonical_name, e.entity_type AS entity_type, e.confidence AS confidence, e.document_id AS document_id, e.chunk_id AS chunk_id, e.page_number AS page_number, e.extraction_method AS extraction_method ORDER BY e.canonical_name ASC LIMIT $limit")
            query = " ".join(clauses)
            with driver.session() as session:
                results = session.run(query, **params)
                return [dict(record) for record in results]
        except Exception:
            return []

    def list_relationships(self, relationship_type: str | None = None, entity_id: str | None = None, limit: int = 250) -> list[dict]:
        try:
            driver = self._get_driver()
            clauses = ["MATCH (source:Entity)-[r:RELATED_TO]->(target:Entity)"]
            where = []
            params: dict[str, object] = {"limit": limit}
            if relationship_type and relationship_type != "All":
                where.append("r.relationship_type = $rel_type")
                params["rel_type"] = relationship_type
            if entity_id:
                where.append("(source.entity_id = $entity_id OR target.entity_id = $entity_id)")
                params["entity_id"] = entity_id
            if where:
                clauses.append("WHERE " + " AND ".join(where))
            clauses.append("RETURN r.relationship_id AS relationship_id, source.entity_id AS source_entity_id, source.canonical_name AS source_entity_name, target.entity_id AS target_entity_id, target.canonical_name AS target_entity_name, r.relationship_type AS relationship_type, r.confidence AS confidence, r.source_document_id AS source_document_id, r.source_chunk_id AS source_chunk_id, r.page_number AS page_number, r.extraction_method AS extraction_method, r.source_text AS source_text ORDER BY r.relationship_type ASC LIMIT $limit")
            query = " ".join(clauses)
            with driver.session() as session:
                results = session.run(query, **params)
                return [dict(record) for record in results]
        except Exception:
            return []

    def clear_graph(self) -> None:
        driver = self._get_driver()
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")


