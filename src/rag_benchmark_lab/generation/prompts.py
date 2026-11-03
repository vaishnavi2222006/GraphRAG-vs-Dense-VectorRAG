from __future__ import annotations

from ..schemas.documents import RetrievalStrategy


SYSTEM_PROMPT = (
    "You are a production-grade, evidence-grounded question-answering assistant.\n"
    "Your primary responsibility is factual precision and strict grounding in the provided context.\n"
    "Follow these core instructions at all times:\n"
    "1. Answer using ONLY the supplied retrieved document context. Do NOT use any external or unmentioned knowledge.\n"
    "2. Address the user's actual question directly and concisely.\n"
    "3. Extract specific facts accurately, including names, addresses, dates, quantities, contact extensions, and relationships.\n"
    "4. Avoid mixing unrelated passages into the answer.\n"
    "5. Never invent or hallucinate missing information.\n"
    "6. If the supplied evidence is insufficient to answer the question or establish an exact fact (e.g. an exact price or unknown detail), state clearly that the provided documents do not contain sufficient evidence.\n"
    "7. Distinguish explicit facts from reasonable inferences.\n"
    "8. Preserve relevant source-document identifiers for citations."
)


def build_answer_prompt(question: str, context: str, strategy: RetrievalStrategy) -> str:
    ctx_display = context.strip() if context.strip() else "[No context was retrieved]"
    return (
        f"Question: {question}\n\n"
        "Retrieved Context:\n"
        "---\n"
        f"{ctx_display}\n"
        "---\n\n"
        f"Retrieval strategy used: {strategy.value}\n\n"
        "Instructions:\n"
        "- Answer the user's question directly using ONLY the facts provided in the Retrieved Context above.\n"
        "- Extract specific facts accurately (exact hospital/company names, physical addresses, numbers of beds, telephone extensions, operating hours, and entity relationships).\n"
        "- If the retrieved context does not provide sufficient evidence to establish the answer (for example, if an exact price or specific detail is not stated), explicitly state that the supplied documents do not contain sufficient evidence to answer.\n"
        "- Never speculate, extrapolate, or invent details not present in the retrieved context.\n"
        "- Conclude your answer with bullet-point citations specifying the source documents and chunk indexes that support your answer."
    )


DIRECT_SYSTEM_PROMPT = (
    "You are an intelligent, precise question-answering assistant.\n"
    "Answer the user's question directly, clearly, and concisely based on your general knowledge.\n"
    "Do NOT fabricate facts or cite documents. If you do not know the answer, state that you do not know."
)


def build_direct_answer_prompt(question: str) -> str:
    return (
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- Answer the user's question directly and concisely using only your general pre-trained knowledge.\n"
        "- Do not reference, invent, or cite any external document context or chunks.\n"
        "- If you do not know the answer, state that you do not have sufficient information."
    )



def build_structured_extraction_prompt(text: str, entity_categories: list[str]) -> str:
    categories = ", ".join(entity_categories)
    return (
        "Extract only grounded entities and relationships from the source text. "
        f"Allowed entity categories: {categories}. "
        "Return valid JSON with keys entities and relationships. "
        "Each entity needs entity_type, canonical_name, original_mention, and confidence. "
        "Each relationship needs source_entity, relationship_type, target_entity, confidence, and evidence text.\n\n"
        f"Source text:\n{text}"
    )
