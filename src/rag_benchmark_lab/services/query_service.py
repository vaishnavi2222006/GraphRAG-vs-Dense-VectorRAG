from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from ..generation.llm_service import LLMResponse, LLMService, OllamaLLMService
from ..generation.prompts import SYSTEM_PROMPT, build_answer_prompt
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.query_router import route_query
from ..retrieval.vector_retriever import VectorRetriever
from ..schemas.documents import AnswerResult, Citation, GraphEvidence, RetrievedChunk, RetrievalStrategy
from ..utils.errors import AuthenticationError, QuotaExceededError, sanitize_secrets


def _format_context(chunks: list[RetrievedChunk], limit_chars: int = 8000, graph_evidence: GraphEvidence | None = None) -> str:
    """Format retrieved chunks and graph evidence into a readable context block for the LLM.

    Each chunk is prefixed with its source document name and chunk index so
    the model can produce accurate citations. Chunks are ordered by score
    (highest relevance first). If graph relationships are present, they are
    formatted as structured knowledge connections.
    """
    parts: list[str] = []
    total = 0

    if graph_evidence and graph_evidence.relationships:
        rel_lines = []
        for rel in graph_evidence.relationships[:10]:
            line = f"- {rel.source_entity_name} -> [{rel.relationship_type}] -> {rel.target_entity_name}"
            if rel.source_text and len(rel.source_text.strip()) > 10:
                line += f' (Evidence: "{rel.source_text.strip()}")'
            rel_lines.append(line)
        if rel_lines:
            graph_block = "[Knowledge Graph Relationships]:\n" + "\n".join(rel_lines)
            parts.append(graph_block)
            total += len(graph_block)

    for chunk in chunks:
        doc_label = chunk.document_name or chunk.document_id
        snippet = f"[Source: {doc_label} | chunk {chunk.chunk_index}]\n{chunk.text}"
        if total + len(snippet) > limit_chars:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n\n".join(parts)


class QueryService:
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        graph_retriever,
        hybrid_retriever: HybridRetriever,
        llm_service: LLMService | OllamaLLMService,
        top_k: int,
        answer_temperature: float,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.hybrid_retriever = hybrid_retriever
        self.llm_service = llm_service
        self.top_k = top_k
        self.answer_temperature = answer_temperature

    def _generate(
        self,
        question: str,
        strategy: RetrievalStrategy,
        chunks: list[RetrievedChunk],
        graph_evidence: GraphEvidence | None = None,
        skip_generation_reason: str | None = None,
    ) -> tuple[str, LLMResponse, list[str], dict[str, str]]:
        """Generate answer with safe validation, secrets sanitization, and quota protection."""
        # 1. If skipping generation because quota is already known to be exhausted
        if skip_generation_reason:
            clean_err = skip_generation_reason
            fallback_response = LLMResponse(
                text=f"Error generating answer: {clean_err}",
                model_name=getattr(self.llm_service, "model_name", "unknown"),
                latency_ms=None,
                mock_mode=False,
                warnings=["generation_skipped_due_to_exhausted_quota"],
            )
            err_msg = f"QuotaExceededError: {clean_err}"
            meta = {
                "clean_error": clean_err,
                "technical_details": sanitize_secrets(clean_err),
                "is_quota": "true",
                "generation_attempted": "false",
            }
            return fallback_response.text, fallback_response, [err_msg], meta

        # 2. Safe API-key validation without making network calls
        if hasattr(self.llm_service, "is_configured") and not self.llm_service.is_configured():
            msg = "Gemini API key is not configured. Please update GEMINI_API_KEY in your .env file and restart the application."
            fallback_response = LLMResponse(
                text=f"Error generating answer: {msg}",
                model_name=getattr(self.llm_service, "model_name", "unknown"),
                latency_ms=None,
                mock_mode=False,
                warnings=["gemini_api_key_not_configured"],
            )
            err_msg = f"AuthenticationError: {msg}"
            meta = {
                "clean_error": msg,
                "technical_details": sanitize_secrets(msg),
                "is_auth": "true",
                "generation_attempted": "false",
            }
            return fallback_response.text, fallback_response, [err_msg], meta

        # 3. Call LLM service
        if strategy == RetrievalStrategy.DIRECT:
            from ..generation.prompts import DIRECT_SYSTEM_PROMPT, build_direct_answer_prompt
            prompt = build_direct_answer_prompt(question)
            sys_prompt = DIRECT_SYSTEM_PROMPT
        else:
            context = _format_context(chunks, graph_evidence=graph_evidence)
            prompt = build_answer_prompt(question, context, strategy)
            sys_prompt = SYSTEM_PROMPT

        gen_start = perf_counter()
        try:
            response = self.llm_service.complete(prompt, system_prompt=sys_prompt, temperature=self.answer_temperature)
            gen_elapsed_ms = (perf_counter() - gen_start) * 1000.0
            actual_latency = response.latency_ms if (response.latency_ms is not None and response.latency_ms > 0) else gen_elapsed_ms
            response.latency_ms = actual_latency
            return response.text, response, [], {"generation_attempted": "true", "generation_latency_ms": str(actual_latency)}
        except Exception as exc:
            gen_elapsed_ms = (perf_counter() - gen_start) * 1000.0
            clean_msg = getattr(exc, "clean_message", str(exc))
            tech_details = getattr(exc, "technical_details", sanitize_secrets(str(exc)))
            err_msg = f"{exc.__class__.__name__}: {clean_msg}"
            fallback_response = LLMResponse(
                text=f"Error generating answer: {clean_msg}",
                model_name=getattr(self.llm_service, "model_name", "unknown"),
                latency_ms=gen_elapsed_ms,
                mock_mode=False,
                warnings=[f"generation_error_{exc.__class__.__name__}"],
            )
            meta = {
                "clean_error": clean_msg,
                "technical_details": tech_details,
                "is_quota": "true" if isinstance(exc, QuotaExceededError) else "false",
                "generation_attempted": "true",
                "failed_generation_latency_ms": str(gen_elapsed_ms),
            }
            return fallback_response.text, fallback_response, [err_msg], meta

    def answer(
        self,
        question: str,
        manual_strategy: RetrievalStrategy | None = None,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
        skip_generation_reason: str | None = None,
    ) -> AnswerResult:
        start = perf_counter()
        routing = route_query(question, manual_strategy=manual_strategy)

        if routing.selected_strategy != RetrievalStrategy.DIRECT and document_ids is not None and len(document_ids) == 0:
            total_latency_ms = (perf_counter() - start) * 1000.0
            return AnswerResult(
                answer="The selected knowledge base contains no documents to retrieve answers from.",
                selected_strategy=routing.selected_strategy,
                citations=[],
                retrieved_sources=[],
                retrieval_latency_ms=0.0,
                generation_latency_ms=0.0,
                total_latency_ms=total_latency_ms,
                model_name=getattr(self.llm_service, "model_name", "system"),
                warnings=["empty_corpus_scope"],
                errors=[],
                routing_rationale=routing.rationale,
                routing_confidence=routing.confidence,
                raw_context="",
                graph_evidence=None,
                metadata={
                    "overridden": routing.overridden,
                    "retrieval_status": "empty",
                    "generation_status": "skipped",
                },
            )

        warnings: list[str] = []
        errors: list[str] = []
        retrieved: list[RetrievedChunk] = []
        graph_evidence: GraphEvidence | None = None
        retrieval_latency_ms: float | None = 0.0

        # ── Step 1: Execute Retrieval independently ──────────────────────────
        if routing.selected_strategy == RetrievalStrategy.DIRECT:
            # DIRECT LLM baseline: strictly isolated from corpus
            retrieved = []
            graph_evidence = None
            retrieval_latency_ms = None
        elif routing.selected_strategy == RetrievalStrategy.VECTOR:
            vector_result = self.vector_retriever.retrieve(question, document_ids=document_ids)
            retrieved = vector_result.chunks
            retrieval_latency_ms = vector_result.retrieval_latency_ms
        elif routing.selected_strategy == RetrievalStrategy.GRAPH:
            graph_result = self.graph_retriever.retrieve(question, document_ids=document_ids)
            retrieved = graph_result.source_chunks
            graph_evidence = graph_result.evidence
            retrieval_latency_ms = graph_result.retrieval_latency_ms
            warnings.extend(graph_result.warnings)
        elif routing.selected_strategy == RetrievalStrategy.HYBRID:
            hybrid_result = self.hybrid_retriever.retrieve(question, document_ids=document_ids)
            retrieved = hybrid_result.chunks
            graph_evidence = hybrid_result.graph_evidence
            retrieval_latency_ms = hybrid_result.retrieval_latency_ms
            warnings.extend(hybrid_result.warnings)
        elif routing.selected_strategy == RetrievalStrategy.ALL:
            hybrid_result = self.hybrid_retriever.retrieve(question, document_ids=document_ids)
            retrieved = hybrid_result.chunks
            graph_evidence = hybrid_result.graph_evidence
            retrieval_latency_ms = hybrid_result.retrieval_latency_ms
            warnings.extend(hybrid_result.warnings)

        effective_top_k = top_k or self.top_k

        # ── Step 2: Build Citations from retrieved evidence ──────────────────
        if routing.selected_strategy == RetrievalStrategy.DIRECT:
            citations = []
        else:
            citations = [
                Citation(
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    chunk_id=chunk.chunk_id,
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    score=chunk.score,
                )
                for chunk in retrieved[:effective_top_k]
            ]

        # ── Step 3: Attempt Answer Generation ────────────────────────────────
        answer_text, response, gen_errors, gen_meta = self._generate(
            question,
            routing.selected_strategy,
            retrieved[:effective_top_k],
            graph_evidence=graph_evidence,
            skip_generation_reason=skip_generation_reason,
        )
        errors.extend(gen_errors)

        if routing.selected_strategy == RetrievalStrategy.DIRECT:
            retrieval_status = "not_applicable"
            graph_path_count = None
            raw_context_val = ""
        else:
            has_retrieval_evidence = bool(retrieved or (graph_evidence and (graph_evidence.entities or graph_evidence.relationships)))
            retrieval_status = "success" if has_retrieval_evidence else "no_evidence"
            graph_path_count = len(graph_evidence.relationships) if graph_evidence and graph_evidence.relationships else 0
            raw_context_val = _format_context(retrieved, graph_evidence=graph_evidence)

        generation_status = "failed" if gen_errors else "success"
        gen_latency = response.latency_ms if generation_status == "success" else None
        total_latency = (perf_counter() - start) * 1000.0

        metadata = {
            "overridden": routing.overridden,
            "retrieval_status": retrieval_status,
            "generation_status": generation_status,
            "evidence_count": len(retrieved) if routing.selected_strategy != RetrievalStrategy.DIRECT else None,
            "citation_count": len(citations) if routing.selected_strategy != RetrievalStrategy.DIRECT else None,
            "graph_path_count": graph_path_count,
            **gen_meta,
        }

        return AnswerResult(
            answer=answer_text,
            selected_strategy=routing.selected_strategy,
            citations=citations,
            retrieved_sources=retrieved,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=gen_latency,
            total_latency_ms=total_latency,
            model_name=response.model_name,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            estimated_cost=0.0 if response.mock_mode else None,
            warnings=warnings + (response.warnings or []),
            errors=errors,
            routing_rationale=routing.rationale,
            routing_confidence=routing.confidence,
            raw_context=raw_context_val,
            graph_evidence=graph_evidence,
            metadata=metadata,
        )

    def compare(
        self,
        question: str,
        document_ids: list[str] | None = None,
        strategies: list[RetrievalStrategy] | None = None,
    ) -> dict[str, AnswerResult]:
        """Execute comparison across selected strategies.

        Defaults to Vector, Graph, Hybrid for backwards compatibility.
        Can also include DIRECT for four-way evaluation.
        """
        results: dict[str, AnswerResult] = {}
        daily_quota_exhausted = False
        quota_reason = ""
        active_strategies = strategies or [RetrievalStrategy.VECTOR, RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID]

        for strategy in active_strategies:
            try:
                skip_reason = quota_reason if daily_quota_exhausted else None
                res = self.answer(
                    question,
                    manual_strategy=strategy,
                    document_ids=document_ids,
                    skip_generation_reason=skip_reason,
                )
                results[strategy.value] = res


                # Detect if this strategy triggered quota exhaustion
                for err in res.errors:
                    if "QuotaExceededError" in err or "quota" in err.lower() or "resource_exhausted" in err.lower():
                        daily_quota_exhausted = True
                        quota_reason = (
                            "Gemini's available quota has been exhausted for the configured model. "
                            "Check the project's quota or wait until it resets."
                        )
                        break
            except Exception as exc:
                err_msg = f"{exc.__class__.__name__}: {str(exc)}"
                results[strategy.value] = AnswerResult(
                    answer=f"Error executing {strategy.value}: {err_msg}",
                    selected_strategy=strategy,
                    citations=[],
                    retrieved_sources=[],
                    retrieval_latency_ms=0.0,
                    generation_latency_ms=None,
                    total_latency_ms=None,
                    model_name=getattr(self.llm_service, "model_name", "unknown"),
                    warnings=[f"execution_error_{exc.__class__.__name__}"],
                    errors=[err_msg],
                    routing_rationale=f"Failed execution for {strategy.value}",
                    routing_confidence=0.0,
                    raw_context="",
                    graph_evidence=None,
                    metadata={"error": str(exc), "retrieval_status": "failed", "generation_status": "failed"},
                )
        return results


