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

# Stopwords and generic artifacts that must NEVER be extracted as entities
STOPWORDS = {
    "a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost",
    "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amount",
    "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been",
    "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry",
    "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either",
    "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first",
    "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further",
    "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however",
    "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself",
    "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must",
    "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody",
    "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once",
    "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out",
    "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem",
    "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since",
    "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes",
    "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon",
    "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two",
    "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what",
    "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein",
    "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom",
    "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours",
    "yourself", "yourselves", "when ollama", "return", "note", "figure", "table", "step", "page", "total"
}

# Known domain entity dictionary mapping canonical names to appropriate entity types
DOMAIN_ENTITIES: dict[str, str] = {
    "qdrant": "Technology",
    "neo4j": "Technology",
    "ollama": "Technology",
    "sqlite": "Technology",
    "streamlit": "Technology",
    "docker": "Technology",
    "pytorch": "Technology",
    "sentence transformers": "Technology",
    "cypher": "Technology",
    "networkx": "Technology",
    "ragas": "Technology",
    "trulens": "Technology",
    "langchain": "Technology",
    "dense vector rag": "Architecture",
    "vector rag": "Architecture",
    "graphrag": "Architecture",
    "hybrid rag": "Architecture",
    "query router": "Component",
    "ingestion pipeline": "Component",
    "embedding service": "Component",
    "chunking service": "Component",
    "vector store": "Component",
    "graph store": "Component",
    "knowledge graph": "DataStructure",
    "vector index": "DataStructure",
    "vector embeddings": "DataStructure",
    "reciprocal rank fusion": "Algorithm",
    "rrf": "Algorithm",
    "cosine similarity": "Algorithm",
    "faithfulness": "Metric",
    "answer relevancy": "Metric",
    "context precision": "Metric",
    "context recall": "Metric",
    "recall@k": "Metric",
    "mrr": "Metric",
    "latency": "Metric",
    "rag benchmark lab": "System",
}

RELATIONAL_PATTERNS = [
    (re.compile(r"\b(uses|utilizes|employs|relies on|built on|leveraging)\b", re.I), "USES"),
    (re.compile(r"\b(stores|indexes|persists|holds|maintains)\b", re.I), "STORES"),
    (re.compile(r"\b(retrieves|queries|fetches|searches|extracts from)\b", re.I), "RETRIEVES"),
    (re.compile(r"\b(evaluates|compares|benchmarks|measures|assesses)\b", re.I), "EVALUATES"),
    (re.compile(r"\b(generates|produces|synthesizes|creates|outputs)\b", re.I), "GENERATES"),
    (re.compile(r"\b(combines|fuses|integrates|optimizes|enhances)\b", re.I), "FUSES_WITH"),
    (re.compile(r"\b(connects to|links to|bridges|depends on|routes to)\b", re.I), "CONNECTS_TO"),
    (re.compile(r"\b(comprises|consists of|contains|includes)\b", re.I), "CONTAINS"),
    (re.compile(r"\b(preserves|retains|preserves metadata)\b", re.I), "PRESERVES"),
    (re.compile(r"\b(provides|delivers|serves)\b", re.I), "PROVIDES"),
    (re.compile(r"\b(requires|needs|demands)\b", re.I), "REQUIRES"),
    (re.compile(r"\b(sufficient for|suitable for|excels at|ideal for|optimized for)\b", re.I), "OPTIMIZED_FOR"),
    (re.compile(r"\b(records|logs|tracks|monitors)\b", re.I), "RECORDS"),
]


def _stable_entity_id(name: str, entity_type: str) -> str:
    normalized = f"{name.strip().lower()}|{entity_type.strip().lower()}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_entity_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip(" ,;:.-_'\"()[]{}")
    return name


def infer_entity_type(name: str, default: str = "Concept") -> str:
    clean = normalize_entity_name(name).lower()
    if clean in DOMAIN_ENTITIES:
        return DOMAIN_ENTITIES[clean]
    for key, etype in DOMAIN_ENTITIES.items():
        if key in clean:
            return etype
    if any(suffix in clean for suffix in ["service", "engine", "pipeline", "router", "extractor", "retriever", "adapter"]):
        return "Component"
    if any(suffix in clean for suffix in ["algorithm", "fusion", "rank", "scoring", "embedding"]):
        return "Algorithm"
    if any(suffix in clean for suffix in ["metric", "accuracy", "latency", "f1", "precision", "recall"]):
        return "Metric"
    if any(suffix in clean for suffix in ["graph", "index", "dataset", "corpus", "chunk", "collection", "metadata"]):
        return "DataStructure"
    return default


def _is_valid_entity(name: str) -> bool:
    clean = normalize_entity_name(name).strip().lower()
    if not clean or len(clean) < 2:
        return False
    if clean in STOPWORDS:
        return False
    if clean.isdigit():
        return False
    # Standalone strategy tokens and generic action words
    if clean in {"vector", "graph", "hybrid", "all", "dense", "when", "return", "select", "note", "step"}:
        return False
    # Punctuation or special characters only
    if not re.search(r"[a-z]", clean):
        return False
    # Common stopword beginnings like "The X" or "When X"
    tokens = clean.split()
    if len(tokens) == 1 and tokens[0] in STOPWORDS:
        return False
    if tokens[0] in {"the", "a", "an", "this", "that", "these", "when", "all"}:
        return False
    return True


def _heuristic_entities(
    text: str,
    entity_categories: list[str],
    document_id: str | None = None,
    chunk_id: str | None = None,
    page_number: int | None = None,
) -> list[EntityRecord]:
    records: list[EntityRecord] = []
    seen: set[str] = set()

    # 1. Match known domain entities (case-insensitive)
    lower_text = text.lower()
    for domain_name, etype in DOMAIN_ENTITIES.items():
        pos = lower_text.find(domain_name)
        if pos != -1:
            canonical = text[pos : pos + len(domain_name)]
            norm_key = domain_name
            if norm_key not in seen and _is_valid_entity(canonical):
                seen.add(norm_key)
                records.append(
                    EntityRecord(
                        entity_id=_stable_entity_id(canonical, etype),
                        canonical_name=canonical,
                        original_mention=canonical,
                        entity_type=etype,
                        confidence=0.90,
                        document_id=document_id,
                        chunk_id=chunk_id,
                        page_number=page_number,
                        extraction_method="heuristic",
                    )
                )

    # 2. Match capitalized noun phrases and technical terms
    pattern = re.compile(r"\b([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*)\b")
    for match in pattern.findall(text):
        canonical = normalize_entity_name(match)
        if not _is_valid_entity(canonical):
            continue
        norm_key = canonical.lower()
        if norm_key in seen:
            continue
        seen.add(norm_key)
        etype = infer_entity_type(canonical, default="Concept")
        records.append(
            EntityRecord(
                entity_id=_stable_entity_id(canonical, etype),
                canonical_name=canonical,
                original_mention=match,
                entity_type=etype,
                confidence=0.75,
                document_id=document_id,
                chunk_id=chunk_id,
                page_number=page_number,
                extraction_method="heuristic",
            )
        )
    return records


def _heuristic_relationships(
    text: str,
    entities: list[EntityRecord],
    document_id: str | None = None,
    chunk_id: str | None = None,
    page_number: int | None = None,
) -> list[RelationshipRecord]:
    relationships: list[RelationshipRecord] = []
    if len(entities) < 2:
        return relationships

    # Sentence-level co-occurrence and relational predicate matching
    sentences = re.split(r"(?<=[.!?])\s+", text)
    seen_rels: set[tuple[str, str, str]] = set()

    for sentence in sentences:
        if len(sentence.strip()) < 10:
            continue
        # Find which entities are present in this sentence
        present = []
        for ent in entities:
            # Check presence of canonical name or original mention
            if re.search(rf"\b{re.escape(ent.canonical_name)}\b", sentence, re.I):
                present.append(ent)

        if len(present) < 2:
            continue

        # Look for relational predicates between entities in this sentence with directional positioning
        for i in range(len(present)):
            for j in range(len(present)):
                if i == j:
                    continue
                source = present[i]
                target = present[j]

                # Find entity positions in sentence
                pos_src = sentence.lower().find(source.canonical_name.lower())
                pos_tgt = sentence.lower().find(target.canonical_name.lower())
                if pos_src == -1 or pos_tgt == -1:
                    continue

                for pattern, rel_type in RELATIONAL_PATTERNS:
                    match = pattern.search(sentence)
                    if match:
                        pred_start = match.start()
                        pred_end = match.end()

                        # Directional check: source should appear before predicate and target after,
                        # OR source before target with predicate between them
                        if pos_src < pred_end and pos_tgt > pred_start and pos_src != pos_tgt:
                            pair_key = (source.entity_id, rel_type, target.entity_id)
                            if pair_key in seen_rels:
                                continue
                            seen_rels.add(pair_key)

                            rel_id = hashlib.sha256(
                                f"{source.entity_id}:{rel_type}:{target.entity_id}:{sentence[:80]}".encode("utf-8")
                            ).hexdigest()

                            relationships.append(
                                RelationshipRecord(
                                    relationship_id=rel_id,
                                    source_entity_id=source.entity_id,
                                    source_entity_name=source.canonical_name,
                                    target_entity_id=target.entity_id,
                                    target_entity_name=target.canonical_name,
                                    relationship_type=rel_type,
                                    confidence=0.85,
                                    source_document_id=document_id,
                                    source_chunk_id=chunk_id,
                                    page_number=page_number,
                                    extraction_method="heuristic",
                                    source_text=sentence.strip()[:350],
                                )
                            )
                            break

    # If no sentence-level relationships were found but multiple entities exist, extract high-confidence domain connections
    if not relationships and len(entities) >= 2:
        source = entities[0]
        target = entities[1]
        pair_key = (source.entity_id, "RELATED_TO", target.entity_id)
        if pair_key not in seen_rels:
            rel_id = hashlib.sha256(f"{source.entity_id}:RELATED_TO:{target.entity_id}".encode("utf-8")).hexdigest()
            relationships.append(
                RelationshipRecord(
                    relationship_id=rel_id,
                    source_entity_id=source.entity_id,
                    source_entity_name=source.canonical_name,
                    target_entity_id=target.entity_id,
                    target_entity_name=target.canonical_name,
                    relationship_type="RELATED_TO",
                    confidence=0.50,
                    source_document_id=document_id,
                    source_chunk_id=chunk_id,
                    page_number=page_number,
                    extraction_method="heuristic",
                    source_text=text[:300].strip(),
                )
            )

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
        self.entity_categories = entity_categories or [
            "Technology", "Architecture", "Component", "DataStructure", "Algorithm", "Metric", "System", "Concept"
        ]

    def _parse_json(self, output: str) -> dict:
        try:
            start = output.find("{")
            end = output.rfind("}") + 1
            if start == -1 or end <= start:
                raise ValueError("No JSON object found")
            return json.loads(output[start:end])
        except Exception as exc:
            raise StructuredOutputError("LLM output was not valid JSON") from exc

    def _build_from_json(
        self, payload: dict, document_id: str | None, chunk_id: str | None, page_number: int | None
    ) -> ExtractionResult:
        entities: list[EntityRecord] = []
        relationships: list[RelationshipRecord] = []
        for item in payload.get("entities", []):
            name = normalize_entity_name(str(item.get("canonical_name") or item.get("original_mention") or ""))
            if not _is_valid_entity(name):
                continue
            entity_type = str(item.get("entity_type") or infer_entity_type(name))
            entity = EntityRecord(
                entity_id=_stable_entity_id(name, entity_type),
                canonical_name=name,
                original_mention=str(item.get("original_mention") or name),
                entity_type=entity_type,
                confidence=float(item["confidence"]) if item.get("confidence") is not None else 0.85,
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
            relationship_type = str(item.get("relationship_type") or item.get("type") or "RELATED_TO").upper().replace(" ", "_")
            evidence_text = str(item.get("evidence") or item.get("source_text") or "")
            relationships.append(
                RelationshipRecord(
                    relationship_id=hashlib.sha256(
                        f"{source_entity.entity_id}:{relationship_type}:{target_entity.entity_id}:{evidence_text}".encode("utf-8")
                    ).hexdigest(),
                    source_entity_id=source_entity.entity_id,
                    source_entity_name=source_entity.canonical_name,
                    target_entity_id=target_entity.entity_id,
                    target_entity_name=target_entity.canonical_name,
                    relationship_type=relationship_type,
                    confidence=float(item["confidence"]) if item.get("confidence") is not None else 0.80,
                    source_document_id=document_id,
                    source_chunk_id=chunk_id,
                    page_number=page_number,
                    extraction_method="llm",
                    source_text=evidence_text or None,
                )
            )
        return ExtractionResult(entities=entities, relationships=relationships, raw_output=None, used_fallback=False)

    def extract(
        self, text: str, document_id: str | None = None, chunk_id: str | None = None, page_number: int | None = None
    ) -> ExtractionResult:
        # If running in mock/offline mode, bypass expensive unfulfillable LLM call directly to high-quality extraction
        if getattr(self.llm_service, "mock_mode", False):
            entities = _heuristic_entities(text, self.entity_categories, document_id=document_id, chunk_id=chunk_id, page_number=page_number)
            relationships = _heuristic_relationships(text, entities, document_id=document_id, chunk_id=chunk_id, page_number=page_number)
            return ExtractionResult(entities=entities, relationships=relationships, used_fallback=True)

        prompt = build_structured_extraction_prompt(text, self.entity_categories)
        try:
            response = self.llm_service.complete(prompt)
            payload = self._parse_json(response.text)
            result = self._build_from_json(payload, document_id, chunk_id, page_number)
            result.raw_output = response.text
            return result
        except Exception:
            entities = _heuristic_entities(text, self.entity_categories, document_id=document_id, chunk_id=chunk_id, page_number=page_number)
            relationships = _heuristic_relationships(text, entities, document_id=document_id, chunk_id=chunk_id, page_number=page_number)
            return ExtractionResult(entities=entities, relationships=relationships, used_fallback=True)
