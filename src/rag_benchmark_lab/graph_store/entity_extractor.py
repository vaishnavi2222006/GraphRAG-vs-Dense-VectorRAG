from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Iterable

from ..generation.llm_service import OllamaLLMService
from ..generation.prompts import build_structured_extraction_prompt
from ..schemas.documents import EntityRecord, RelationshipRecord
from ..utils.errors import StructuredOutputError


ENTITY_PATTERN = re.compile(r"\b([A-Z][A-Za-z0-9_]+(?:\s+[A-Z][A-Za-z0-9_]+)*)\b")
RELATIONSHIP_TERMS = ["uses", "depends on", "connects to", "stores", "creates", "triggers", "relates to", "feeds"]


def _stable_entity_id(name: str, entity_type: str) -> str:
    normalized = f"{name.strip().lower()}|{entity_type.strip().lower()}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_entity_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip(" ,;:.-")
    return name


def _heuristic_entities(text: str, entity_categories: list[str]) -> list[EntityRecord]:
    records: list[EntityRecord] = []
    seen: set[tuple[str, str]] = set()
    for match in ENTITY_PATTERN.findall(text):
        canonical = normalize_entity_name(match)
        entity_type = entity_categories[0] if entity_categories else "Concept"
        key = (canonical.lower(), entity_type.lower())
        if key in seen:
            continue
        seen.add(key)
        records.append(
            EntityRecord(
                entity_id=_stable_entity_id(canonical, entity_type),
                canonical_name=canonical,
                original_mention=match,
                entity_type=entity_type,
                confidence=0.35,
            )
        )
    return records


def _heuristic_relationships(text: str, entities: list[EntityRecord]) -> list[RelationshipRecord]:
    relationships: list[RelationshipRecord] = []
    if len(entities) < 2:
        return relationships
    for term in RELATIONSHIP_TERMS:
        if term not in text.lower():
            continue
        source = entities[0]
        target = entities[1]
        relationships.append(
            RelationshipRecord(
                relationship_id=hashlib.sha256(f"{source.entity_id}:{term}:{target.entity_id}".encode("utf-8")).hexdigest(),
                source_entity_id=source.entity_id,
                source_entity_name=source.canonical_name,
                target_entity_id=target.entity_id,
                target_entity_name=target.canonical_name,
                relationship_type=term.upper().replace(" ", "_"),
                confidence=0.25,
                source_text=term,
            )
        )
        break
    return relationships


@dataclass(slots=True)
class ExtractionResult:
    entities: list[EntityRecord]
    relationships: list[RelationshipRecord]
    raw_output: str | None = None
    used_fallback: bool = False


class EntityRelationshipExtractor:
    def __init__(self, llm_service: OllamaLLMService, entity_categories: list[str] | None = None) -> None:
        self.llm_service = llm_service
        self.entity_categories = entity_categories or ["Person", "Organization", "Concept", "Technology", "Process", "Event", "Location", "Product"]

    def _parse_json(self, output: str) -> dict:
        try:
            start = output.find("{")
            end = output.rfind("}") + 1
            return json.loads(output[start:end])
        except Exception as exc:
            raise StructuredOutputError("LLM output was not valid JSON") from exc

    def _build_from_json(self, payload: dict, document_id: str | None, chunk_id: str | None, page_number: int | None) -> ExtractionResult:
        entities: list[EntityRecord] = []
        relationships: list[RelationshipRecord] = []
        for item in payload.get("entities", []):
            name = normalize_entity_name(str(item.get("canonical_name") or item.get("original_mention") or ""))
            if not name:
                continue
            entity_type = str(item.get("entity_type") or "Concept")
            entity = EntityRecord(
                entity_id=_stable_entity_id(name, entity_type),
                canonical_name=name,
                original_mention=str(item.get("original_mention") or name),
                entity_type=entity_type,
                confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
                document_id=document_id,
                chunk_id=chunk_id,
                page_number=page_number,
                extraction_method="llm",
            )
            entities.append(entity)

        lookup = {entity.canonical_name.lower(): entity for entity in entities}
        for item in payload.get("relationships", []):
            source_name = normalize_entity_name(str(item.get("source_entity") or item.get("source") or ""))
            target_name = normalize_entity_name(str(item.get("target_entity") or item.get("target") or ""))
            source_entity = lookup.get(source_name.lower())
            target_entity = lookup.get(target_name.lower())
            if source_entity is None or target_entity is None:
                continue
            relationship_type = str(item.get("relationship_type") or item.get("type") or "RELATED_TO")
            evidence_text = str(item.get("evidence") or item.get("source_text") or "")
            relationships.append(
                RelationshipRecord(
                    relationship_id=hashlib.sha256(f"{source_entity.entity_id}:{relationship_type}:{target_entity.entity_id}:{evidence_text}".encode("utf-8")).hexdigest(),
                    source_entity_id=source_entity.entity_id,
                    source_entity_name=source_entity.canonical_name,
                    target_entity_id=target_entity.entity_id,
                    target_entity_name=target_entity.canonical_name,
                    relationship_type=relationship_type,
                    confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
                    source_document_id=document_id,
                    source_chunk_id=chunk_id,
                    page_number=page_number,
                    extraction_method="llm",
                    source_text=evidence_text or None,
                )
            )
        return ExtractionResult(entities=entities, relationships=relationships, raw_output=None, used_fallback=False)

    def extract(self, text: str, document_id: str | None = None, chunk_id: str | None = None, page_number: int | None = None) -> ExtractionResult:
        prompt = build_structured_extraction_prompt(text, self.entity_categories)
        try:
            response = self.llm_service.complete(prompt)
            payload = self._parse_json(response.text)
            result = self._build_from_json(payload, document_id, chunk_id, page_number)
            result.raw_output = response.text
            return result
        except Exception:
            entities = _heuristic_entities(text, self.entity_categories)
            relationships = _heuristic_relationships(text, entities)
            return ExtractionResult(entities=entities, relationships=relationships, used_fallback=True)

