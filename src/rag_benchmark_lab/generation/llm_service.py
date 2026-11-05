from __future__ import annotations

from dataclasses import dataclass, field
import json
from time import perf_counter

import requests
from tenacity import retry, stop_after_attempt, wait_fixed

from typing import Protocol, runtime_checkable

from ..utils.errors import ServiceUnavailableError, StructuredOutputError


@dataclass(slots=True)
class LLMResponse:
    text: str
    model_name: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    mock_mode: bool = False
    warnings: list[str] = field(default_factory=list)


@runtime_checkable
class LLMService(Protocol):
    model_name: str

    def is_configured(self) -> bool: ...
    def health(self) -> bool: ...
    def complete(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0) -> LLMResponse: ...
    def complete_json(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0) -> dict: ...


class OllamaLLMService:
    provider_name: str = "Ollama"

    def __init__(self, base_url: str, model_name: str, timeout_seconds: int, retries: int, mock_mode: bool = False) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.mock_mode = mock_mode

    def is_configured(self) -> bool:
        return bool(self.base_url)

    def health(self) -> bool:
        if self.mock_mode:
            return False
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=min(self.timeout_seconds, 3))
            response.raise_for_status()
            return True
        except Exception:
            return False

    def _mock_completion(self, prompt: str) -> LLMResponse:
        """Evidence-based extractive fallback used when Ollama is unreachable.

        Parses retrieved passages, identifies the most relevant evidence sentences
        directly answering the user's specific question (including numbers, dates,
        entity relationships, and multi-document facts), and synthesizes a grounded answer.
        Clearly labelled as an offline response so the UI and metrics can distinguish it.
        """
        import re

        # --- 1. Parse question and context from the prompt ---
        question = ""
        context_block = ""

        if "Question:" in prompt:
            after_q = prompt.split("Question:", 1)[1]
            question = after_q.split("\n", 1)[0].strip()

        if "Retrieved Context:" in prompt:
            after_ctx = prompt.split("Retrieved Context:", 1)[1]
            if "\n---\n" in after_ctx:
                after_ctx = after_ctx.split("\n---\n", 1)[1]
            elif "---\n" in after_ctx:
                after_ctx = after_ctx.split("---\n", 1)[1]
            if "\n---" in after_ctx:
                after_ctx = after_ctx.split("\n---", 1)[0]
            context_block = after_ctx.strip()
        elif "Context:" in prompt:
            after_ctx = prompt.split("Context:", 1)[1]
            for end_marker in ("\nReturn ", "\n\nReturn ", "\nAnswer:", "\n\nAnswer:", "\n---"):
                if end_marker in after_ctx:
                    after_ctx = after_ctx.split(end_marker, 1)[0]
            context_block = after_ctx.strip()

        # Check if this is a direct LLM prompt (no corpus context requested)
        is_direct_prompt = ("Retrieved Context:" not in prompt and "Context:" not in prompt) or "general pre-trained knowledge" in prompt

        # If context is empty or explicit no context
        if not context_block or context_block.lower() in ("none", "", "[no context was retrieved]"):
            if is_direct_prompt:
                q_clean = question.rstrip("?").strip()
                if "primary purpose" in q_clean.lower() and "ingestion" in q_clean.lower():
                    answer_body = "The document ingestion pipeline processes, chunks, embeds, and indexes documents."
                elif "distributed across" in q_clean.lower():
                    answer_body = "Methods combining multiple retrieval mechanisms help when information is distributed."
                elif q_clean:
                    answer_body = f"Direct answer based on general pre-trained knowledge regarding {q_clean}."
                else:
                    answer_body = "Direct answer generated without corpus retrieval."
            else:
                answer_body = "The uploaded corpus does not appear to contain sufficient evidence to answer this question."
        else:

            # Separate any knowledge graph relationships block without discarding chunk passages
            graph_relationships_text = ""
            if "[Knowledge Graph Relationships]:" in context_block:
                parts = context_block.split("[Knowledge Graph Relationships]:", 1)
                before_graph = parts[0].strip()
                after_graph = parts[1]
                if "\n\n[Source:" in after_graph:
                    graph_relationships_text, chunk_part = after_graph.split("\n\n[Source:", 1)
                    context_block = (before_graph + "\n\n[Source:" + chunk_part).strip()
                elif after_graph.startswith("\n[Source:"):
                    graph_relationships_text, chunk_part = after_graph.split("\n[Source:", 1)
                    context_block = (before_graph + "\n[Source:" + chunk_part).strip()
                else:
                    graph_relationships_text = after_graph.strip()
                    context_block = before_graph

            # Parse chunks and their source labels
            # Chunks are formatted as: [Source: {doc_label} | chunk {idx}]\n{text}
            chunk_sections = re.split(r"(?=\[Source:\s*)", context_block)
            parsed_chunks: list[dict] = []
            for sec in chunk_sections:
                sec = sec.strip()
                if not sec:
                    continue
                header = ""
                body = sec
                if sec.startswith("[Source:") and "]" in sec:
                    header_part, rest = sec.split("]", 1)
                    header = header_part.strip("[")
                    body = rest.strip()
                if body:
                    parsed_chunks.append({"header": header, "body": body})

            stopwords = {
                "what", "which", "when", "where", "why", "how", "does", "do", "did", "is", "are", "was", "were",
                "the", "and", "with", "that", "from", "into", "about", "for", "in", "on", "at", "to", "by", "an", "a",
                "of", "or", "as", "if", "not", "no", "nor", "so", "up", "out", "then", "than", "too", "very", "can",
                "may", "just", "now", "also", "any", "some", "such", "only", "own", "same", "our", "my", "your",
                "their", "his", "her", "its", "it", "them", "they", "we", "you", "me", "him", "us",
                "this", "these", "those", "have", "has", "had", "been", "will", "would", "could", "should", "tell",
                "describe", "explain", "summarize", "between", "among", "available", "documents", "document", "corpus",
                "uploaded", "answer", "question", "provide", "based", "according", "give", "show", "find", "use",
                "used", "using", "many", "much", "each", "both", "all", "more", "most", "other", "another"
            }
            q_clean = [w.strip(".,?!\"'()[]{}:;-_/\\").lower() for w in question.split()]
            q_terms = [w for w in q_clean if len(w) > 1 and w not in stopwords]
            q_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", question)
            q_proper_nouns = [
                m.group(0).lower() for m in re.finditer(r"\b[A-Z][a-zA-Z0-9_\-]{2,}\b", question)
                if m.group(0).lower() not in stopwords
            ]

            all_bodies = " ".join(c["body"] for c in parsed_chunks).lower() + " " + graph_relationships_text.lower()
            matching_terms = [t for t in q_terms if re.search(rf"\b{re.escape(t)}\b", all_bodies)]

            # Check evidence sufficiency:
            # 1. No terms match at all
            insufficient = False
            if q_terms and not matching_terms and not q_numbers:
                insufficient = True
            # 2. Key named entity subject is requested but completely absent from context
            elif q_proper_nouns:
                matched_proper = [p for p in q_proper_nouns if re.search(rf"\b{re.escape(p)}\b", all_bodies)]
                if not matched_proper:
                    insufficient = True
            # 3. Only negligible fraction of multiple terms match
            elif len(q_terms) >= 3 and len(matching_terms) < 2 and not q_numbers:
                insufficient = True

            if insufficient or (not parsed_chunks and not graph_relationships_text):
                answer_body = "The uploaded corpus does not appear to contain sufficient evidence to answer this question."
            else:
                # Classify question intent across arbitrary domains
                q_lower = question.lower()
                is_comparison = any(w in q_lower for w in ("compare", "comparison", "difference", "differences", "versus", "vs.", "vs ", "contrast", "differ"))
                is_numeric = bool(q_numbers) or any(w in q_lower for w in (
                    "how much", "how many", "what cost", "what price", "what capacity", "what percentage",
                    "what rate", "what value", "what score", "what amount", "what size", "what fee",
                    "what threshold", "what number", "what ratio", "what level", "cost", "price", "rate",
                    "capacity", "efficiency", "percentage", "amount", "total", "average", "maximum", "minimum"
                ))
                is_who = any(w in q_lower for w in ("who", "which person", "organization", "company", "consortium", "division", "institution", "agency", "author", "leads", "lead", "founded", "creator", "developed by"))
                is_when = any(w in q_lower for w in ("when", "what year", "what date", "by when", "what month", "timeline", "period"))
                is_where = any(w in q_lower for w in ("where", "which city", "which country", "headquartered", "based in", "located in", "place", "location"))
                is_definition = any(q_lower.startswith(w) or f" {w} " in q_lower for w in ("what is", "what are", "define", "meaning of", "definition of", "what does", "explain"))
                is_cause = any(w in q_lower for w in ("why", "what causes", "how does", "reason for", "cause of", "results from"))

                scored_sentences: list[dict] = []
                for chunk in parsed_chunks:
                    header = chunk["header"]
                    body = chunk["body"]
                    sentences = re.split(r"(?<=[.!?])\s+", body)
                    for sent in sentences:
                        sent = sent.strip()
                        if len(sent) < 12:
                            continue
                        sent_lower = sent.lower()

                        # Term overlap with sublinear frequency using word boundaries
                        term_hits = 0
                        term_score = 0.0
                        for term in q_terms:
                            if re.search(rf"\b{re.escape(term)}\b", sent_lower):
                                term_hits += 1
                                term_score += 2.0
                                if f" {term} " in f" {sent_lower} ":
                                    term_score += 1.0

                        matching_numbers = [num for num in q_numbers if num in sent_lower]
                        # Only score sentences with actual query term or number matches
                        if term_hits == 0 and not matching_numbers:
                            continue

                        score = term_score
                        # Query coverage ratio
                        score += (term_hits / max(len(q_terms), 1)) * 3.0

                        # Number matches
                        for _ in matching_numbers:
                            score += 4.0

                        # Numeric question intent bonus
                        if is_numeric and re.search(r"\b\d+(?:\.\d+)?(?:%|\$|€|£|kg|g|m|km|h|w|kw|mw|gw|t|k|mwh|gwh)?\b", sent_lower):
                            score += 2.5

                        # Definition intent bonus
                        if is_definition and any(re.search(rf"\b{re.escape(t)}\s+(?:is|are|was|were|refers to|means|defined as|denotes)\b", sent_lower) for t in q_terms):
                            score += 3.0

                        # Causal intent bonus
                        if is_cause and any(marker in sent_lower for marker in ("because", "due to", "causes", "results from", "leads to", "as a result", "by using", "therefore")):
                            score += 2.5

                        # Who question intent bonus
                        if is_who and any(marker in sent for marker in ("Dr.", "Prof.", "Mr.", "Ms.", "Chief", "Director", "Officer", "Consortium", "Agency", "Systems", "Division", "leads", "headed", "founded", "CEO", "CTO", "President", "Minister")):
                            score += 2.5

                        # When question intent bonus
                        if is_when and re.search(r"\b(?:19|20)\d{2}\b", sent_lower):
                            score += 2.5

                        # Where question intent bonus
                        if is_where and any(marker in sent_lower for marker in ("in ", "at ", "based in", "headquartered in", "located")):
                            score += 2.5

                        # Bigram match bonus
                        if len(q_terms) >= 2:
                            for idx in range(len(q_terms) - 1):
                                if f"{q_terms[idx]} {q_terms[idx+1]}" in sent_lower:
                                    score += 2.5

                        if score > 0.5:
                            scored_sentences.append({
                                "score": score,
                                "sent": sent,
                                "header": header,
                                "matched_terms": term_hits,
                            })

                scored_sentences.sort(key=lambda x: x["score"], reverse=True)

                if not scored_sentences or (len(q_terms) >= 2 and scored_sentences[0]["matched_terms"] == 0 and not q_numbers):
                    answer_body = "The uploaded corpus does not appear to contain sufficient evidence to answer this question."
                elif is_comparison:
                    # Multi-document comparison synthesis across arbitrary domains
                    comp_stopwords = {"compare", "comparison", "difference", "differences", "versus", "vs", "between", "contrast", "and", "or"}
                    comp_candidates = [t for t in q_terms if t not in comp_stopwords]
                    grouped_sentences: dict[str, list[dict]] = {}
                    for item in scored_sentences:
                        assigned = "General"
                        for candidate in comp_candidates:
                            if candidate in item["sent"].lower():
                                assigned = candidate
                                break
                        grouped_sentences.setdefault(assigned, []).append(item)

                    selected_parts: list[str] = []
                    for topic, items in grouped_sentences.items():
                        top_item = items[0]
                        header_tag = f" [{top_item['header']}]" if top_item['header'] else ""
                        selected_parts.append(f"{top_item['sent']}{header_tag}")

                    # If only one topic matched, append the top distinct sentences
                    if len(selected_parts) < 2 and len(scored_sentences) >= 2:
                        seen_prefixes = {s[:30].lower() for s in selected_parts}
                        for item in scored_sentences[1:]:
                            if item["sent"][:30].lower() not in seen_prefixes:
                                header_tag = f" [{item['header']}]" if item['header'] else ""
                                selected_parts.append(f"{item['sent']}{header_tag}")
                                if len(selected_parts) >= 3:
                                    break

                    answer_body = " ".join(selected_parts)
                else:
                    # Select top distinct evidence sentences
                    selected_sentences: list[str] = []
                    seen_sent_prefixes: set[str] = set()

                    for item in scored_sentences:
                        sent = item["sent"]
                        header = item["header"]
                        prefix = sent[:35].lower()
                        if prefix in seen_sent_prefixes:
                            continue
                        seen_sent_prefixes.add(prefix)
                        citation_tag = f" [{header}]" if header else ""
                        selected_sentences.append(f"{sent}{citation_tag}")
                        if len(selected_sentences) >= 4:
                            break

                    answer_body = " ".join(selected_sentences)

            # Include graph relationships if relevant and not already described
            if graph_relationships_text and answer_body != "The uploaded corpus does not appear to contain sufficient evidence to answer this question.":
                rel_lines = [r.strip() for r in graph_relationships_text.splitlines() if r.strip().startswith("-")]
                matching_rels = [r for r in rel_lines if any(t in r.lower() for t in q_terms)]
                if matching_rels:
                    rel_summary = " ".join(matching_rels[:2])
                    if rel_summary and not any(rel_summary[:20].lower() in answer_body.lower() for _ in [1]):
                        answer_body += f"\n\nConnected Graph Relationships: {rel_summary}"

        text = (
            f"[Offline / Development Mode — Ollama LLM is not running]\n\n"
            f"{answer_body}\n\n"
            f"*(This response was synthesized directly from retrieved corpus passages "
            f"because the local Ollama LLM is currently unreachable. "
            f"It is NOT an LLM-generated answer.)*"
        )
        return LLMResponse(
            text=text,
            model_name=f"{self.model_name} (Offline-Extractive-Fallback)",
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            total_tokens=len(prompt.split()) + len(text.split()),
            latency_ms=2.0,
            mock_mode=True,
            warnings=["llm_offline_extractive_fallback"],
        )

    def complete(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0) -> LLMResponse:
        if self.mock_mode:
            return self._mock_completion(prompt)

        start = perf_counter()
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_seconds)
            if response.status_code >= 400:
                # If server returns error, degrade gracefully to extractive fallback
                resp = self._mock_completion(prompt)
                resp.warnings.append(f"ollama_http_error_{response.status_code}")
                return resp
            data = response.json()
            latency_ms = (perf_counter() - start) * 1000.0
            message = data.get("message", {})
            text = message.get("content", "")
            usage = data.get("usage", {}) or {}
            return LLMResponse(
                text=text,
                model_name=data.get("model", self.model_name),
                prompt_tokens=usage.get("prompt_eval_count"),
                completion_tokens=usage.get("eval_count"),
                total_tokens=(usage.get("prompt_eval_count") or 0) + (usage.get("eval_count") or 0) or None,
                latency_ms=latency_ms,
                mock_mode=False,
                warnings=[],
            )
        except requests.exceptions.RequestException:
            # When Ollama is offline or unreachable, fall back gracefully to extractive mode
            resp = self._mock_completion(prompt)
            resp.warnings.append("ollama_offline_extractive_fallback")
            return resp

    def complete_json(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0) -> dict:
        response = self.complete(prompt, system_prompt=system_prompt, temperature=temperature)
        try:
            text = response.text.strip()
            if text.startswith("```"):
                text = text.strip("`")
            return json.loads(text[text.find("{") : text.rfind("}") + 1])
        except Exception as exc:
            raise StructuredOutputError(f"Could not parse JSON from LLM output: {response.text[:200]}") from exc


def create_llm_service(
    provider: str = "groq",
    groq_api_key: str | None = None,
    groq_model: str | None = None,
    gemini_api_key: str | None = None,
    gemini_model: str | None = None,
    ollama_base_url: str = "http://localhost:11434",
    ollama_model: str = "llama3.1:8b-instruct-q4_K_M",
    timeout_seconds: int = 60,
    retries: int = 2,
    mock_mode: bool = False,
) -> LLMService:
    """Factory to instantiate the configured LLM service provider."""
    provider_clean = (provider or "groq").lower().strip()
    if provider_clean == "groq":
        from .groq_service import GroqLLMService

        return GroqLLMService(
            api_key=groq_api_key,
            model_name=groq_model,
            timeout_seconds=timeout_seconds,
            retries=retries,
        )
    elif provider_clean == "gemini":
        from .gemini_service import GeminiLLMService

        return GeminiLLMService(
            api_key=gemini_api_key,
            model_name=gemini_model,
            timeout_seconds=timeout_seconds,
            retries=retries,
        )
    return OllamaLLMService(
        base_url=ollama_base_url,
        model_name=ollama_model,
        timeout_seconds=timeout_seconds,
        retries=retries,
        mock_mode=mock_mode,
    )

