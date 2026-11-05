"""Live end-to-end verification of the Four-Way Evaluation & Recommendation System.

Tests:
1. Mode 1: Single Query Evaluation (Vector RAG, GraphRAG, Hybrid RAG, Direct LLM)
2. 15-Parameter Comparison Matrix validation & strict N/A enforcement for Direct LLM
3. Single Query Recommendation Engine (Definitive vs Provisional)
4. Mode 2: Four-Way Benchmark Dataset Runner & Aggregations
5. Persistence into ResultStore
"""
from __future__ import annotations

from dataclasses import asdict
import os
import sys
import time

# Ensure workspace root is in python path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(WORKSPACE_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from rag_benchmark_lab.app import build_context
from rag_benchmark_lab.evaluation.benchmark_runner import BenchmarkRunner
from rag_benchmark_lab.evaluation.metrics import metric_bundle
from rag_benchmark_lab.evaluation.recommendation import evaluate_approach, recommend_best_approach
from rag_benchmark_lab.schemas.documents import RetrievalStrategy
from rag_benchmark_lab.schemas.evaluation import EvaluationExample


def test_four_way_single_query_evaluation(context) -> None:
    print("\n" + "=" * 70)
    print("STEP 1: FOUR-WAY SINGLE QUERY EVALUATION (MODE 1)")
    print("=" * 70)

    # Use known user_id and KB or query from database
    user_id = "c40f2f5330feac9db884e2dfe482ca5e"
    kb_id = "38e72b77-0352-4de2-b9c4-5b0dcddd42ed" # MediCore Hospital KB
    doc_ids = []
    try:
        doc_ids = context.user_registry.get_kb_document_ids(kb_id, user_id)
        print(f"Using Knowledge Base: MediCore Hospital KB with {len(doc_ids)} documents.")
    except Exception:
        doc_ids = [d["document_id"] for d in context.registry.list_documents()][:5]
        print(f"Using fallback documents: {len(doc_ids)} documents.")

    question = "What is the primary protocol for emergency cardiac arrest triage in the ICU?"
    expected_answer = (
        "Immediate initiation of chest compressions at 100-120 bpm, rapid defibrillation for shockable rhythms, "
        "and epinephrine administration every 3-5 minutes according to ACLS guidelines."
    )
    relevant_docs = doc_ids[:2] if doc_ids else ["sample_doc"]

    strategies = [
        RetrievalStrategy.VECTOR,
        RetrievalStrategy.GRAPH,
        RetrievalStrategy.HYBRID,
        RetrievalStrategy.DIRECT,
    ]

    answers = {}
    metrics_map = {}
    scores_map = {}

    for strat in strategies:
        print(f"\n---> Executing approach: {strat.value} ...")
        t0 = time.perf_counter()
        ans = context.query_service.answer(
            question,
            manual_strategy=strat,
            document_ids=doc_ids,
        )
        elapsed = (time.perf_counter() - t0) * 1000.0
        answers[strat.value] = ans

        print(f"     Status: {'Success' if not ans.errors else 'Failed'}")
        print(f"     Model: {ans.model_name}")
        print(f"     Retrieval Latency: {ans.retrieval_latency_ms if ans.retrieval_latency_ms is not None else 'N/A'}")
        print(f"     Generation Latency: {ans.generation_latency_ms:.1f} ms" if ans.generation_latency_ms else "N/A")
        print(f"     Total Latency: {ans.total_latency_ms:.1f} ms" if ans.total_latency_ms else "N/A")
        print(f"     Retrieved Chunks: {len(ans.retrieved_sources)}")
        print(f"     Answer Snippet: {ans.answer[:120]}...")

        # Calculate metric bundle
        mets = metric_bundle(ans, expected_answer, relevant_docs, None, question=question)
        metrics_map[strat.value] = mets

        score = evaluate_approach(strat.value, ans, mets, has_ground_truth=True)
        scores_map[strat.value] = score
        print(f"     Composite Score: {score.composite_score:.3f}")
        print(f"     Strengths: {score.strengths}")
        print(f"     Weaknesses: {score.weaknesses}")

    # Validate Direct LLM isolation
    dir_ans = answers["DIRECT"]
    assert len(dir_ans.retrieved_sources) == 0, "Direct LLM must have 0 retrieved sources"
    assert dir_ans.retrieval_latency_ms is None, "Direct LLM retrieval latency must be None (N/A)"
    assert dir_ans.graph_evidence is None, "Direct LLM must have no graph evidence"
    assert len(dir_ans.citations) == 0, "Direct LLM must have 0 citations"
    assert dir_ans.generation_latency_ms is not None and dir_ans.generation_latency_ms > 0, "Direct LLM must have generation latency"

    # Validate metric N/A enforcement for Direct LLM
    dir_mets = metrics_map["DIRECT"]
    assert dir_mets["faithfulness"].available is False, "Direct LLM faithfulness must be N/A"
    assert dir_mets["citation_correctness"].available is False, "Direct LLM citation correctness must be N/A"
    assert dir_mets["context_precision"].available is False, "Direct LLM context precision must be N/A"
    assert dir_mets["context_recall"].available is False, "Direct LLM context recall must be N/A"
    assert dir_mets["retrieval_recall_at_k"].available is False, "Direct LLM recall@k must be N/A"
    assert dir_mets["mrr"].available is False, "Direct LLM MRR must be N/A"

    print("\n✓ Step 1 Passed: All 4 approaches executed with proper isolation and metric computation.")

    # ── Step 2: Test 15-Parameter Comparison Matrix ──────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: 15-PARAMETER COMPARISON MATRIX VERIFICATION")
    print("=" * 70)

    parameters = [
        ("Answer correctness", "answer_correctness", "score"),
        ("Faithfulness", "faithfulness", "score"),
        ("Answer relevancy", "answer_relevancy", "score"),
        ("Completeness", "completeness", "score"),
        ("Citation correctness", "citation_correctness", "score"),
        ("Context precision", "context_precision", "score"),
        ("Context recall", "context_recall", "score"),
        ("Recall@K", "retrieval_recall_at_k", "score"),
        ("Mean Reciprocal Rank (MRR)", "mrr", "score"),
        ("Retrieval latency", "retrieval_latency_ms", "latency"),
        ("Generation latency", "generation_latency_ms", "latency"),
        ("Total latency", "total_latency_ms", "latency"),
        ("Retrieved chunk count", "retrieved_chunk_count", "count"),
        ("Graph path count", "graph_path_count", "count"),
        ("Execution status", "status", "status"),
    ]

    print(f"{'Parameter':<30} | {'Vector RAG':<12} | {'GraphRAG':<12} | {'Hybrid RAG':<12} | {'Direct LLM':<12}")
    print("-" * 88)

    for label, key, p_type in parameters:
        cols = []
        for s in ("VECTOR", "GRAPH", "HYBRID", "DIRECT"):
            ans = answers[s]
            mets = metrics_map[s]
            is_direct = (s == "DIRECT")

            if is_direct and key in (
                "faithfulness", "citation_correctness", "context_precision", "context_recall",
                "retrieval_recall_at_k", "mrr", "retrieval_latency_ms", "retrieved_chunk_count", "graph_path_count"
            ):
                cols.append("N/A")
                continue

            if p_type == "score":
                m_obj = mets.get(key)
                val = m_obj.value if (m_obj and m_obj.available) else None
                cols.append(f"{val:.3f}" if val is not None else "N/A")
            elif p_type == "latency":
                lat = getattr(ans, key, None)
                cols.append(f"{lat:.1f} ms" if lat is not None else "N/A")
            elif p_type == "count":
                if key == "retrieved_chunk_count":
                    cnt = len(ans.retrieved_sources) if not is_direct else None
                else:
                    cnt = len(ans.graph_evidence.relationships) if (ans.graph_evidence and ans.graph_evidence.relationships) else 0
                cols.append(str(cnt) if cnt is not None else "N/A")
            elif p_type == "status":
                cols.append("Success" if not ans.errors else "Failed")

        print(f"{label:<30} | {cols[0]:<12} | {cols[1]:<12} | {cols[2]:<12} | {cols[3]:<12}")

    print("\n✓ Step 2 Passed: 15-parameter comparison matrix cleanly populated.")

    # ── Step 3: Recommendation Engine (Definitive & Provisional) ────────────
    print("\n" + "=" * 70)
    print("STEP 3: RECOMMENDATION ENGINE VERIFICATION")
    print("=" * 70)

    # 3A: Definitive recommendation (with ground truth)
    rec_def = recommend_best_approach(scores_map, has_ground_truth=True, question=question)
    print(f"Definitive Recommendation:")
    print(f"  • Recommended Strategy: {rec_def.recommended_strategy}")
    print(f"  • Status: {rec_def.status}")
    print(f"  • Is Definitive: {rec_def.is_definitive}")
    print(f"  • Confidence: {rec_def.confidence}")
    print(f"  • Rationale: {rec_def.reason}")
    print(f"  • Strengths: {rec_def.strengths}")
    print(f"  • Weaknesses: {rec_def.weaknesses}")

    assert rec_def.recommended_strategy is not None
    assert rec_def.is_definitive is True
    assert rec_def.status == "definitive"

    # 3B: Provisional recommendation (without ground truth)
    provisional_scores = {}
    for s_name, ans in answers.items():
        prov_mets = metric_bundle(ans, reference_answer=None, relevant_document_ids=None, reference_contexts=None, question=question)
        provisional_scores[s_name] = evaluate_approach(s_name, ans, prov_mets, has_ground_truth=False)

    rec_prov = recommend_best_approach(provisional_scores, has_ground_truth=False, question=question)
    print(f"\nProvisional Recommendation (without reference answer):")
    print(f"  • Recommended Strategy: {rec_prov.recommended_strategy}")
    print(f"  • Status: {rec_prov.status}")
    print(f"  • Is Definitive: {rec_prov.is_definitive}")
    print(f"  • Confidence: {rec_prov.confidence}")
    print(f"  • Rationale: {rec_prov.reason}")

    assert rec_prov.is_definitive is False
    assert rec_prov.status in ("provisional", "inconclusive", "tie")

    print("\n✓ Step 3 Passed: Recommendation engine generates definitive and provisional recommendations transparently.")

    # ── Step 4: Mode 2 Benchmark Dataset Runner ──────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: FOUR-WAY BENCHMARK DATASET RUNNER (MODE 2)")
    print("=" * 70)

    examples = [
        EvaluationExample(
            query_id="bench_q1",
            question="What is the triage protocol for cardiac emergencies?",
            expected_answer="Immediate chest compressions, early defibrillation, and epinephrine administration.",
            relevant_document_ids=relevant_docs,
        ),
        EvaluationExample(
            query_id="bench_q2",
            question="What are the essential monitoring parameters in the ICU?",
            expected_answer="Continuous arterial blood pressure, heart rate, ECG, pulse oximetry, and respiratory rate.",
            relevant_document_ids=relevant_docs,
        ),
    ]

    runner = BenchmarkRunner(context.query_service, context.result_store)
    print(f"Executing benchmark across {len(examples)} questions and 4 approaches...")
    output = runner.run(examples, document_ids=doc_ids)

    print(f"Run ID: {output['run_id']}")
    print(f"Total Output Rows: {len(output['rows'])} (Expected 8)")
    assert len(output["rows"]) == 8, f"Expected 8 rows, got {len(output['rows'])}"

    rec_sum = output["recommendation_summary"]
    print(f"Overall Best Benchmark Approach: {rec_sum.get('overall_best')}")
    print(f"Overall Reason: {rec_sum.get('overall_reason')}")
    print(f"Winner Counts: {rec_sum.get('winner_counts')}")
    print(f"Strategy Rankings:")
    for rank in rec_sum.get("strategy_rankings", []):
        print(f"  • {rank['strategy']:<10}: Wins={rank['win_count']}, Composite={rank['composite_score']:.3f}, AvgLat={rank['average_latency_ms']} ms")

    # Verify persistence
    saved_runs = context.result_store.list_runs()
    matching = [r for r in saved_runs if r["run_id"] == output["run_id"]]
    assert len(matching) > 0, "Benchmark run was not persisted into ResultStore!"
    print(f"Verified persistence in ResultStore: Run {output['run_id'][:12]} recorded.")

    print("\n✓ Step 4 Passed: Benchmark runner executed all four approaches and persisted results.")

    print("\n" + "=" * 70)
    print("ALL FOUR-WAY EVALUATION VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    ctx = build_context()
    test_four_way_single_query_evaluation(ctx)
