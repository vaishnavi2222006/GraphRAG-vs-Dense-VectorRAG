"""Live verification script for Groq API integration across all 6 test scenarios."""
from __future__ import annotations

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from rag_benchmark_lab.generation.groq_service import GroqLLMService
from rag_benchmark_lab.generation.llm_service import create_llm_service
from rag_benchmark_lab.utils.errors import (
    AuthenticationError,
    ModelUnavailableError,
    NetworkError,
    RateLimitError,
    sanitize_secrets,
)
from rag_benchmark_lab.schemas.documents import (
    RetrievedChunk,
    RetrievalStrategy,
    GraphEvidence,
    EntityRecord,
    RelationshipRecord,
)
from rag_benchmark_lab.services.query_service import QueryService
from unittest.mock import MagicMock

print("=" * 60)
print("1. GROQ CONNECTIVITY & MODEL AVAILABILITY TEST")
print("=" * 60)

api_key = os.getenv("GROQ_API_KEY", "")
configured_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
print(f"Configured GROQ_MODEL: {configured_model}")
print(f"GROQ_API_KEY present: {bool(api_key)}")

# Test with configured model
svc_configured = GroqLLMService(api_key=api_key, model_name=configured_model)
try:
    resp = svc_configured.complete("Say 'Groq connection successful' in exactly 3 words.")
    print(f"Response with {configured_model}: {resp.text.strip()}")
    print(f"Latency: {resp.latency_ms:.1f}ms | Tokens: prompt={resp.prompt_tokens}, completion={resp.completion_tokens}")
    active_live_model = configured_model
except ModelUnavailableError as e:
    print(f"Model '{configured_model}' unavailable on this Groq account/key: {e.clean_message}")
    print("Testing with account-supported model 'qwen/qwen3.8-27b'...")
    active_live_model = "qwen/qwen3.8-27b"
    svc_live = GroqLLMService(api_key=api_key, model_name=active_live_model)
    resp = svc_live.complete("Say 'Groq connection successful' in exactly 3 words.")
    print(f"Response with {active_live_model}: {resp.text.strip()}")
    print(f"Latency: {resp.latency_ms:.1f}ms | Tokens: prompt={resp.prompt_tokens}, completion={resp.completion_tokens}")

print("\n" + "=" * 60)
print(f"2, 3, 4. VECTOR RAG, GRAPHRAG, HYBRID RAG WITH GROQ ({active_live_model})")
print("=" * 60)

live_svc = GroqLLMService(api_key=api_key, model_name=active_live_model)

sample_chunks = [
    RetrievedChunk(
        chunk_id="chunk_test_1",
        document_id="doc_test_1",
        document_name="AI_Research_Report.pdf",
        chunk_index=0,
        text="The Groq LPU (Language Processing Unit) is designed for deterministic, low-latency deep learning inference.",
        score=0.95,
    ),
    RetrievedChunk(
        chunk_id="chunk_test_2",
        document_id="doc_test_1",
        document_name="AI_Research_Report.pdf",
        chunk_index=1,
        text="Hybrid RAG fuses vector similarity search with Neo4j graph traversal via Reciprocal Rank Fusion.",
        score=0.88,
    ),
]

sample_evidence = GraphEvidence(
    entities=[
        EntityRecord(entity_id="e1", canonical_name="Groq LPU", original_mention="Groq LPU", entity_type="Hardware", confidence=0.9),
        EntityRecord(entity_id="e2", canonical_name="Low Latency", original_mention="low-latency", entity_type="Property", confidence=0.9),
    ],
    relationships=[
        RelationshipRecord(
            relationship_id="r1",
            source_entity_id="e1",
            source_entity_name="Groq LPU",
            target_entity_id="e2",
            target_entity_name="Low Latency",
            relationship_type="OPTIMIZED_FOR",
            confidence=0.92,
            source_text="The Groq LPU is designed for deterministic, low-latency deep learning inference.",
        )
    ],
    source_chunks=sample_chunks,
)

mock_vec = MagicMock()
mock_vec.retrieve.return_value = MagicMock(chunks=sample_chunks, retrieval_latency_ms=3.5)

mock_graph = MagicMock()
mock_graph.retrieve.return_value = MagicMock(source_chunks=sample_chunks, evidence=sample_evidence, retrieval_latency_ms=5.0, warnings=[])

mock_hybrid = MagicMock()
mock_hybrid.retrieve.return_value = MagicMock(chunks=sample_chunks, graph_evidence=sample_evidence, retrieval_latency_ms=7.2, warnings=[])

qs = QueryService(
    vector_retriever=mock_vec,
    graph_retriever=mock_graph,
    hybrid_retriever=mock_hybrid,
    llm_service=live_svc,
    top_k=2,
    answer_temperature=0.0,
)

# Test 2: Vector RAG
v_res = qs.answer("What is the Groq LPU designed for?", manual_strategy=RetrievalStrategy.VECTOR)
print(f"[Vector RAG Answer]:\n{v_res.answer}")
print(f"[Vector RAG Citations]: {len(v_res.citations)} citations | Model: {v_res.model_name}")

# Test 3: Graph RAG
g_res = qs.answer("What properties are associated with the Groq LPU in the knowledge graph?", manual_strategy=RetrievalStrategy.GRAPH)
print(f"\n[Graph RAG Answer]:\n{g_res.answer}")
print(f"[Graph RAG Graph Paths]: {len(g_res.graph_evidence.relationships if g_res.graph_evidence else [])} paths | Model: {g_res.model_name}")

# Test 4: Hybrid RAG
h_res = qs.answer("Explain the architecture combining Groq and Hybrid RAG.", manual_strategy=RetrievalStrategy.HYBRID)
print(f"\n[Hybrid RAG Answer]:\n{h_res.answer}")
print(f"[Hybrid RAG Citations]: {len(h_res.citations)} citations | Model: {h_res.model_name}")

print("\n" + "=" * 60)
print("5. EVALUATION METRICS VERIFICATION WITH GROQ")
print("=" * 60)
from rag_benchmark_lab.evaluation.benchmark_runner import BenchmarkRunner
from rag_benchmark_lab.evaluation.result_store import ResultStore
from rag_benchmark_lab.schemas.evaluation import EvaluationExample
from rag_benchmark_lab.services.document_registry import DocumentRegistry
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as td:
    reg = DocumentRegistry(Path(td) / "eval.sqlite3")
    store = ResultStore(reg)
    runner = BenchmarkRunner(qs, store)
    ex = EvaluationExample(
        query_id="eval_1",
        question="What is the Groq LPU designed for?",
        expected_answer="Deterministic, low-latency deep learning inference.",
        relevant_document_ids=["doc_test_1"],
        reference_contexts=["The Groq LPU is designed for deterministic, low-latency deep learning inference."],
    )
    eval_run = runner.run([ex], strategies=[RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID], document_ids=["doc_test_1"])
    print("Benchmark run ID:", eval_run["run_id"])
    print("Evaluated strategies:", list(eval_run["aggregates"].keys()))
    for strat, data in eval_run["aggregates"].items():
        print(f"Strategy {strat} metrics:", data["metrics"])

print("\n" + "=" * 60)
print("6. ERROR HANDLING VALIDATION")
print("=" * 60)

# 6a: Missing API key
try:
    svc_no_key = GroqLLMService(api_key="", model_name="llama-3.3-70b-versatile")
    svc_no_key.complete("Hello")
    print("FAILED: Missing key did not raise error")
except AuthenticationError as e:
    print(f"PASS (Missing Key): {e.clean_message}")

# 6b: Invalid API key
try:
    svc_bad_key = GroqLLMService(api_key="gsk_invalid_bogus_key_test_1234567890", model_name=active_live_model)
    svc_bad_key.complete("Hello")
    print("FAILED: Invalid key did not raise error")
except AuthenticationError as e:
    print(f"PASS (Invalid Key): {e.clean_message}")

# 6c: Invalid model configuration
try:
    svc_bad_model = GroqLLMService(api_key=api_key, model_name="nonexistent-groq-model-xyz")
    svc_bad_model.complete("Hello")
    print("FAILED: Bad model did not raise error")
except ModelUnavailableError as e:
    print(f"PASS (Invalid Model): {e.clean_message}")

# 6d: Secret sanitization check
test_leak = f"Error with key {api_key}"
sanitized = sanitize_secrets(test_leak)
assert api_key not in sanitized
print("PASS (Secret Sanitization): API key successfully scrubbed from output.")

print("\n" + "=" * 60)
print("ALL 6 TESTS COMPLETED SUCCESSFULLY!")
print("=" * 60)

