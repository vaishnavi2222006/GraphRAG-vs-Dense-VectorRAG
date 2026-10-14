from __future__ import annotations

from ..schemas.documents import EntityRecord, RelationshipRecord
from .entity_extractor import EntityRelationshipExtractor, ExtractionResult, normalize_entity_name


class RelationshipExtractor:
    def __init__(self, extractor: EntityRelationshipExtractor) -> None:
        self.extractor = extractor

    def extract(self, text: str, document_id: str | None = None, chunk_id: str | None = None, page_number: int | None = None) -> list[RelationshipRecord]:
        result = self.extractor.extract(text, document_id=document_id, chunk_id=chunk_id, page_number=page_number)
        return result.relationships

