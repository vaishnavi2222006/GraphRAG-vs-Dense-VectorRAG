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
    (re.compile(r"\b(leads|heads|headed by|directed by|founded by|appointed as|serves as|chair of|director of|cto at|ceo at|lead|leader of)\b", re.I), "LEADS"),
    (re.compile(r"\b(headquartered in|based in|located in|headquarters in|situated in|operating from)\b", re.I), "LOCATED_IN"),
    (re.compile(r"\b(uses|utilizes|employs|relies on|built on|leveraging|operates on|powered by)\b", re.I), "USES"),
    (re.compile(r"\b(stores|indexes|persists|holds|maintains|archives)\b", re.I), "STORES"),
    (re.compile(r"\b(retrieves|queries|fetches|searches|extracts from|accesses)\b", re.I), "RETRIEVES"),
    (re.compile(r"\b(evaluates|compares|benchmarks|measures|assesses|analyzes|tests)\b", re.I), "EVALUATES"),
    (re.compile(r"\b(generates|produces|synthesizes|creates|outputs|emits|yields)\b", re.I), "GENERATES"),
    (re.compile(r"\b(converts directly into|converts to|converts into|converts|transforms into|translates to)\b", re.I), "CONVERTS_TO"),
    (re.compile(r"\b(combines|fuses|integrates|optimizes|enhances|merges with)\b", re.I), "FUSES_WITH"),
    (re.compile(r"\b(connects to|links to|bridges|depends on|routes to|interacts with|interfaces with)\b", re.I), "CONNECTS_TO"),
    (re.compile(r"\b(comprises|consists of|contains|includes|composed of|encompasses)\b", re.I), "CONTAINS"),
    (re.compile(r"\b(division of|part of|subsidiary of|branch of|member of|belongs to|affiliated with|associated with)\b", re.I), "PART_OF"),
    (re.compile(r"\b(authored by|developed by|created by|published by|designed by|invented by|researched by|develops|designs|manufactures|discovered by|built by)\b", re.I), "DEVELOPS"),
    (re.compile(r"\b(preserves|retains|preserves metadata|maintains state)\b", re.I), "PRESERVES"),
    (re.compile(r"\b(provides|delivers|serves|offers|supplies)\b", re.I), "PROVIDES"),
    (re.compile(r"\b(requires|needs|demands|mandates|targets|aims to limit|aims to|stipulates)\b", re.I), "REQUIRES"),
    (re.compile(r"\b(sufficient for|suitable for|excels at|ideal for|optimized for|specialized in)\b", re.I), "OPTIMIZED_FOR"),
    (re.compile(r"\b(records|logs|tracks|monitors|regulates|oversees|manages|governs|administers)\b", re.I), "TRACKS"),
    (re.compile(r"\b(achieved|achieves|reached|exceeded|fell to|decreased to|increased to|priced at|costs|valued at|rated at|weighs|measures)\b", re.I), "ACHIEVED"),
    (re.compile(r"\b(collaborates with|partnered with|partners with|in partnership with|contracted with|alliance with)\b", re.I), "PARTNERS_WITH"),
    (re.compile(r"\b(replaces|substitutes for|supersedes|displaces)\b", re.I), "REPLACES"),
    (re.compile(r"\b(owns|acquired|purchased|bought|controls|holding)\b", re.I), "OWNS"),
    (re.compile(r"\b(causes|triggers|results in|leads to|induces|promotes|stimulates)\b", re.I), "CAUSES"),
    (re.compile(r"\b(prevents|inhibits|blocks|reduces|decreases|lowers|alleviates|stops)\b", re.I), "INHIBITS"),
    (re.compile(r"\b(treats|cures|heals|diagnoses|prescribed for)\b", re.I), "TREATS"),
    (re.compile(r"\b(binds to|expresses|encodes|catalyzes|metabolizes)\b", re.I), "BINDS_TO"),
    (re.compile(r"\b(invests in|finances|funds|backs)\b", re.I), "INVESTS_IN"),
    (re.compile(r"\b(competes with|rivals|opposes|differs from)\b", re.I), "COMPETES_WITH"),
]


def _stable_entity_id(name: str, entity_type: str) -> str:
    normalized = f"{name.strip().lower()}|{entity_type.strip().lower()}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_entity_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip(" ,;:.-_'\"()[]{}")
    return name


def infer_entity_type(name: str, default: str = "Concept") -> str:
    clean = normalize_entity_name(name).lower()
    # Check DOMAIN_ENTITIES first (demo project specifics get exact/substring match)
    if clean in DOMAIN_ENTITIES:
        return DOMAIN_ENTITIES[clean]
    for key, etype in DOMAIN_ENTITIES.items():
        if key in clean:
            return etype
    # --- General-purpose heuristics for arbitrary documents ---
    # People: title prefix or known name suffix patterns
    person_prefixes = {"dr", "dr.", "prof", "prof.", "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "sir", "rev"}
    if clean.split()[0] in person_prefixes:
        return "Person"
    # Organizations
    org_tokens = ["inc", "inc.", "corp", "corp.", "ltd", "ltd.", "llc", "plc", "gmbh",
                  "foundation", "institute", "university", "college", "school",
                  "association", "organization", "agency", "department", "ministry",
                  "bureau", "committee", "company", "enterprise", "group"]
    if any(clean.endswith(t) or (f" {t}" in clean) for t in org_tokens):
        return "Organization"
    # Locations
    location_tokens = ["city", "town", "village", "country", "state", "province",
                       "region", "district", "county", "territory", "island",
                       "ocean", "river", "mountain", "street", "avenue", "road"]
    if any(t in clean for t in location_tokens):
        return "Location"
    # Technical components
    if any(s in clean for s in ["service", "engine", "pipeline", "router", "extractor",
                                  "retriever", "adapter", "module", "framework", "platform",
                                  "system", "application", "tool", "library", "api"]):
        return "Component"
    # Algorithms and methods
    if any(s in clean for s in ["algorithm", "fusion", "rank", "scoring", "embedding",
                                  "method", "approach", "technique", "model", "network"]):
        return "Algorithm"
    # Metrics
    if any(s in clean for s in ["metric", "accuracy", "latency", "f1", "precision", "recall",
                                  "score", "rate", "ratio", "percentage", "performance"]):
        return "Metric"
    # Data structures and storage
    if any(s in clean for s in ["graph", "index", "dataset", "corpus", "chunk", "collection",
                                  "database", "store", "registry", "schema", "table", "record",
                                  "metadata"]):
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

    # 2. Match names with honorifics (Dr. Elena Vasquez, Prof. James Okonkwo)
    honorific_pattern = re.compile(r"\b((?:Dr\.|Prof\.|Mr\.|Mrs\.|Ms\.)\s+[A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*)\b")
    for match in honorific_pattern.findall(text):
        canonical = normalize_entity_name(match)
        if _is_valid_entity(canonical):
            norm_key = canonical.lower()
            if norm_key not in seen:
                seen.add(norm_key)
                records.append(
                    EntityRecord(
                        entity_id=_stable_entity_id(canonical, "Person"),
                        canonical_name=canonical,
                        original_mention=match,
                        entity_type="Person",
                        confidence=0.92,
                        document_id=document_id,
                        chunk_id=chunk_id,
                        page_number=page_number,
                        extraction_method="heuristic",
                    )
                )

    # 3. Match capitalized noun phrases and technical terms
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
                confidence=0.80,
                document_id=document_id,
                chunk_id=chunk_id,
                page_number=page_number,
                extraction_method="heuristic",
            )
        )

    # 4. Match domain acronyms and abbreviations (LCOE, GWh, IEA, TW, PV, etc.)
    acronym_pattern = re.compile(r"\b([A-Z]{2,6})\b")
    for match in acronym_pattern.findall(text):
        if match.lower() in STOPWORDS or match in {"THE", "AND", "FOR", "WITH", "THAT", "THIS"}:
            continue
        norm_key = match.lower()
        if norm_key not in seen:
            seen.add(norm_key)
            records.append(
                EntityRecord(
                    entity_id=_stable_entity_id(match, "Concept"),
                    canonical_name=match,
                    original_mention=match,
                    entity_type="Concept",
                    confidence=0.82,
                    document_id=document_id,
                    chunk_id=chunk_id,
                    page_number=page_number,
                    extraction_method="heuristic",
                )
            )

    # 5. Match technical compounds with domain noun heads across arbitrary domains
    domain_heads = (
        r"cells|panels|batteries|electrolytes|efficiency|storage|capacity|energy|materials?|alloys?|polymers?|elements?|"
        r"systems?|engines?|platforms?|pipelines?|frameworks?|architectures?|servers?|clients?|clusters?|nodes?|databases?|indexes?|"
        r"algorithms?|models?|networks?|protocols?|services?|apis?|interfaces?|modules?|components?|libraries?|tools?|"
        r"processes?|reactions?|compounds?|molecules?|proteins?|genes?|enzymes?|acids?|drugs?|vaccines?|treatments?|therapies?|"
        r"symptoms?|diseases?|disorders?|syndromes?|infections?|viruses?|bacteria?|pathogens?|sensors?|devices?|"
        r"agreements?|contracts?|treaties?|statutes?|regulations?|policies?|divisions?|consortiums?|corporations?|companies?|"
        r"portfolios?|assets?|liabilities?|revenues?|expenses?|currencies?|equities?|shares?|metrics?|standards?"
    )
    compound_pattern = re.compile(rf"\b([a-z0-9_\-]+(?:\s+[a-z0-9_\-]+){{1,3}}\s+(?:{domain_heads}))\b", re.I)
    for match in compound_pattern.findall(text):
        canonical = normalize_entity_name(match)
        if _is_valid_entity(canonical):
            norm_key = canonical.lower()
            if norm_key not in seen:
                seen.add(norm_key)
                etype = infer_entity_type(canonical, default="Technology")
                records.append(
                    EntityRecord(
                        entity_id=_stable_entity_id(canonical, etype),
                        canonical_name=canonical,
                        original_mention=match,
                        entity_type=etype,
                        confidence=0.78,
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

        # Also connect co-occurring entities within the sentence that had no explicit predicate matched
        if len(present) >= 2:
            for i in range(len(present) - 1):
                source = present[i]
                target = present[i + 1]
                pair_key = (source.entity_id, "RELATED_TO", target.entity_id)
                rev_key = (target.entity_id, "RELATED_TO", source.entity_id)
                if pair_key not in seen_rels and rev_key not in seen_rels:
                    seen_rels.add(pair_key)
                    rel_id = hashlib.sha256(
                        f"{source.entity_id}:RELATED_TO:{target.entity_id}:{sentence[:80]}".encode("utf-8")
                    ).hexdigest()
                    relationships.append(
                        RelationshipRecord(
                            relationship_id=rel_id,
                            source_entity_id=source.entity_id,
                            source_entity_name=source.canonical_name,
                            target_entity_id=target.entity_id,
                            target_entity_name=target.canonical_name,
                            relationship_type="RELATED_TO",
                            confidence=0.70,
                            source_document_id=document_id,
                            source_chunk_id=chunk_id,
                            page_number=page_number,
                            extraction_method="heuristic",
                            source_text=sentence.strip()[:350],
                        )
                    )

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
            # Technical / software domain
            "Technology", "Architecture", "Component", "DataStructure", "Algorithm", "Metric", "System",
            # General-purpose categories for arbitrary documents
            "Person", "Organization", "Location", "Event", "Product", "Concept",
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

