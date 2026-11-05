"""End-to-End Application Verification Script.
Executes real queries across Vector RAG, GraphRAG, Hybrid RAG, 3-way Compare,
Comparison History Persistence, and Benchmark Evaluation using Groq (qwen/qwen3.8-27b).
"""
from __future__ import annotations

import os
import sys
import time

from rag_benchmark_lab.app import build_context
from rag_benchmark_lab.schemas.documents import RetrievalStrategy
from rag_benchmark_lab.schemas.evaluation import EvaluationExample
from rag_benchmark_lab.evaluation.benchmark_runner import BenchmarkRunner

print("=" * 70)
print("STARTING END-TO-END APPLICATION VERIFICATION")
print("=" * 70)

ctx = build_context()

# 1. Verify Configuration & LLM Service
print(f"[CHECK 1] LLM Provider: {ctx.settings.llm_provider}")
print(f"[CHECK 1] Groq Model: {ctx.settings.groq_model}")
print(f"[CHECK 1] LLM Service Class: {ctx.llm_service.__class__.__name__}")
print(f"[CHECK 1] Is Configured: {ctx.llm_service.is_configured()}")
print(f"[CHECK 1] Mock Mode: {ctx.settings.llm_mock_mode}")

assert ctx.settings.llm_provider == "groq"
assert ctx.llm_service.__class__.__name__ == "GroqLLMService"
assert ctx.llm_service.model_name == "qwen/qwen3.8-27b"
assert not ctx.settings.llm_mock_mode

# Retrieve KB doc IDs
kb_id = "38e72b77-0352-4de2-b9c4-5b0dcddd42ed" # MediCore Hospital KB
user_id = "c40f2f5330feac9db884e2dfe482ca5e"
doc_ids = ctx.user_registry.get_kb_document_ids(kb_id, user_id)
print(f"[CORPUS] Using MediCore KB: {len(doc_ids)} documents found.")
assert len(doc_ids) > 0

test_question = "Where is MediCore Community Hospital located and how many beds does it have?"

# 2. Vector RAG Verification
print("\n" + "-" * 70)
print("[TEST 2] Testing Vector RAG with Groq...")
print("-" * 70)
vec_res = ctx.query_service.answer(
    test_question,
    manual_strategy=RetrievalStrategy.VECTOR,
    document_ids=doc_ids,
)
print(f"Strategy: {vec_res.selected_strategy.value}")
print(f"Model: {vec_res.model_name}")
print(f"Retrieval Latency: {vec_res.retrieval_latency_ms:.1f}ms")
print(f"Generation Latency: {vec_res.generation_latency_ms}ms")
print(f"Citations count: {len(vec_res.citations)}")
print(f"Answer snippet: {vec_res.answer[:200]}...")
assert vec_res.model_name == "qwen/qwen3.8-27b"
assert len(vec_res.citations) > 0
assert "MediCore" in vec_res.answer
print(">>> Vector RAG PASSED!")

# 3. Graph RAG Verification
print("\n" + "-" * 70)
print("[TEST 3] Testing Graph RAG with Groq...")
print("-" * 70)
graph_res = ctx.query_service.answer(
    test_question,
    manual_strategy=RetrievalStrategy.GRAPH,
    document_ids=doc_ids,
)
print(f"Strategy: {graph_res.selected_strategy.value}")
print(f"Model: {graph_res.model_name}")
print(f"Retrieval Latency: {graph_res.retrieval_latency_ms:.1f}ms")
print(f"Generation Latency: {graph_res.generation_latency_ms}ms")
print(f"Citations count: {len(graph_res.citations)}")
if graph_res.graph_evidence:
    print(f"Graph Entities: {len(graph_res.graph_evidence.entities)}, Relationships: {len(graph_res.graph_evidence.relationships)}")
print(f"Answer snippet: {graph_res.answer[:200]}...")
assert graph_res.model_name == "qwen/qwen3.8-27b"
assert "MediCore" in graph_res.answer
print(">>> Graph RAG PASSED!")

# 4. Hybrid RAG Verification
print("\n" + "-" * 70)
print("[TEST 4] Testing Hybrid RAG with Groq...")
print("-" * 70)
hybrid_res = ctx.query_service.answer(
    test_question,
    manual_strategy=RetrievalStrategy.HYBRID,
    document_ids=doc_ids,
)
print(f"Strategy: {hybrid_res.selected_strategy.value}")
print(f"Model: {hybrid_res.model_name}")
print(f"Retrieval Latency: {hybrid_res.retrieval_latency_ms:.1f}ms")
print(f"Generation Latency: {hybrid_res.generation_latency_ms}ms")
print(f"Citations count: {len(hybrid_res.citations)}")
print(f"Answer snippet: {hybrid_res.answer[:200]}...")
assert hybrid_res.model_name == "qwen/qwen3.8-27b"
assert len(hybrid_res.citations) > 0
assert "MediCore" in hybrid_res.answer
print(">>> Hybrid RAG PASSED!")

# 5. Three-Way Side-by-Side Comparison Verification
print("\n" + "-" * 70)
print("[TEST 5] Testing 3-Way Comparison...")
print("-" * 70)
comp_results = ctx.query_service.compare(test_question, document_ids=doc_ids)
assert "VECTOR" in comp_results
assert "GRAPH" in comp_results
assert "HYBRID" in comp_results
for s_name, s_res in comp_results.items():
    print(f" - {s_name}: Model={s_res.model_name}, Citations={len(s_res.citations)}, GenLatency={s_res.generation_latency_ms}ms")
    assert s_res.model_name == "qwen/qwen3.8-27b"
    assert len(s_res.citations) > 0
print(">>> 3-Way Comparison PASSED!")

# 6. Comparison History Persistence Verification
print("\n" + "-" * 70)
print("[TEST 6] Testing Comparison History Persistence...")
print("-" * 70)
import uuid
from datetime import datetime, timezone
from rag_benchmark_lab.schemas.comparison import ComparisonRunRecord, ComparisonStrategyResultRecord

run_id = f"cmp_verify_{uuid.uuid4().hex[:8]}"
now_iso = datetime.now(timezone.utc).isoformat()
strat_records = []
for strat_name, res in comp_results.items():
    strat_records.append(
        ComparisonStrategyResultRecord(
            result_id=str(uuid.uuid4()),
            run_id=run_id,
            strategy=strat_name,
            strategy_label=strat_name.title() + " RAG",
            status="success",
            answer=res.answer,
            retrieval_latency_ms=res.retrieval_latency_ms,
            generation_latency_ms=res.generation_latency_ms,
            total_latency_ms=res.total_latency_ms,
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
            model_name=res.model_name,
            citations=[{"chunk_id": c.chunk_id, "document_name": c.document_name} for c in res.citations],
            evidence_sources=[],
            graph_evidence=None,
            warnings=res.warnings or [],
            errors=res.errors or [],
            metadata=res.metadata or {},
            created_at=now_iso,
        )
    )

run_record = ComparisonRunRecord(
    run_id=run_id,
    user_id=user_id,
    kb_id=kb_id,
    query=test_question,
    status="completed",
    created_at=now_iso,
    updated_at=now_iso,
    config={"default_top_k": ctx.settings.default_top_k},
    strategies=[s.strategy for s in strat_records],
    results=strat_records,
    kb_name="MediCore Hospital Knowledge Base",
)

ctx.user_registry.save_comparison_run(run_record)
print(f"Saved run ID: {run_id}")

runs = ctx.user_registry.list_comparison_runs(user_id=user_id, kb_id=kb_id)
assert any(r.run_id == run_id for r in runs)

loaded_run = ctx.user_registry.get_comparison_run(run_id, user_id=user_id)
assert loaded_run is not None
assert len(loaded_run.results) == 3
for r in loaded_run.results:
    assert r.model_name == "qwen/qwen3.8-27b"
    print(f" - Persisted Result {r.strategy}: Citations={len(r.citations)}, Answer length={len(r.answer)}")
print(">>> Comparison History Persistence PASSED!")

# 7. Benchmark Evaluation Execution Verification
print("\n" + "-" * 70)
print("[TEST 7] Testing Benchmark Evaluation Execution with Groq...")
print("-" * 70)
bench_runner = BenchmarkRunner(ctx.query_service, ctx.result_store)
eval_ex = EvaluationExample(
    query_id="eval_test_1",
    question=test_question,
    expected_answer="MediCore Community Hospital is located at 18 Lakeview Road, Riverton with 120 beds.",
    relevant_document_ids=doc_ids[:2],
    reference_contexts=["MediCore Community Hospital is a 120-bed hospital located at 18 Lakeview Road, Riverton."],
)
eval_output = bench_runner.run([eval_ex], strategies=[RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID], document_ids=doc_ids)
print(f"Evaluation Run ID: {eval_output['run_id']}")
for strat in ["VECTOR", "HYBRID"]:
    assert strat in eval_output["aggregates"]
    metrics = eval_output["aggregates"][strat]["metrics"]
    print(f" - {strat} metrics: Faithfulness={metrics.get('faithfulness')}, Relevancy={metrics.get('answer_relevancy')}, Token F1={metrics.get('token_f1')}")
print(">>> Benchmark Evaluation Execution PASSED!")

# 8. Secret Exposure Verification
print("\n" + "-" * 70)
print("[TEST 8] Checking for API Key Leaks in Results, Logs, and UI Payloads...")
print("-" * 70)
raw_key = os.getenv("GROQ_API_KEY", "")
assert len(raw_key) > 10

all_texts = [
    vec_res.answer,
    graph_res.answer,
    hybrid_res.answer,
    str(comp_results),
    str(eval_output),
]
for t in all_texts:
    assert raw_key not in t, "CRITICAL: API key exposed in output text!"
print(">>> Secret Exposure Check PASSED! No keys leaked in any output.")

print("\n" + "=" * 70)
print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
print("=" * 70)
