from __future__ import annotations

from ..schemas.documents import RetrievalStrategy


SYSTEM_PROMPT = (
    "You are a production-grade, evidence-grounded question-answering assistant.\n"
    "Your primary responsibility is factual precision and strict grounding in the provided context.\n"
    "Follow these core instructions at all times:\n"
    "1. Answer using ONLY the supplied retrieved document context. Do NOT use any external or unmentioned knowledge.\n"
    "2. Address the user's actual question directly and concisely.\n"
    "3. Extract specific facts accurately, including names, addresses, dates, quantities, contact extensions, and relationships.\n"
    "4. Format answers in a clear, enhanced structured format: start with a direct summary statement, followed by organized bullet points with bold field labels (e.g. **Name:**, **Address:**, **Bed Capacity:**, **Operating Hours:**, **Phone Extension:**). Use clean tables when comparing multiple departments or items.\n"
    "5. Avoid mixing unrelated passages into the answer.\n"
    "6. Never invent or hallucinate missing information.\n"
    "7. If the supplied evidence is insufficient to answer the question or establish an exact fact (e.g. an exact price or unknown detail), state clearly that the provided documents do not contain sufficient evidence.\n"
    "8. Distinguish explicit facts from reasonable inferences.\n"
    "9. Conclude with a clean '**Citations:**' section specifying supporting source documents and chunk indexes."
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
        "- Provide an enhanced, well-structured output: begin with a direct summary, followed by clean bullet points with bold field labels for all extracted facts (e.g., **Name:**, **Address:**, **Bed Capacity:**, **Operating Hours:**, **Extension:**).\n"
        "- If comparing multiple entities or presenting multi-attribute data, format them into structured bullet points, numbered lists, or a markdown table.\n"
        "- If the retrieved context does not provide sufficient evidence to establish the answer, explicitly state that the supplied documents do not contain sufficient evidence to answer.\n"
        "- Never speculate, extrapolate, or invent details not present in the retrieved context.\n"
        "- Conclude your answer with a '**Citations:**' section listing the source documents and chunk indexes that support your answer."
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
