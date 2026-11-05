from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any

from ..schemas.documents import AnswerResult, RetrievalStrategy
from ..schemas.evaluation import MetricResult, RecommendationResult


@dataclass(slots=True)
class ApproachScore:
    strategy: str
    composite_score: float
    correctness: float | None
    completeness: float | None
    faithfulness: float | None
    relevancy: float | None
    citation_score: float | None
    retrieval_recall: float | None
    mrr: float | None
    retrieval_latency_ms: float | None
    generation_latency_ms: float | None
    total_latency_ms: float | None
    retrieved_chunk_count: int | None
    graph_path_count: int | None
    is_failed: bool
    status_label: str
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)


def _get_metric_val(metrics: dict[str, Any], key: str) -> float | None:
    if key not in metrics:
        return None
    val = metrics[key]
    if isinstance(val, MetricResult):
        return val.value if val.available else None
    if isinstance(val, dict):
        return val.get("value") if val.get("available", True) else None
    if isinstance(val, (int, float)):
        return float(val)
    return None


def evaluate_approach(
    strategy: str,
    answer: AnswerResult,
    metrics: dict[str, Any],
    has_ground_truth: bool,
) -> ApproachScore:
    """Score an individual approach and extract its strengths and weaknesses."""
    is_direct = (strategy == RetrievalStrategy.DIRECT.value or strategy == "DIRECT")
    errors = getattr(answer, "errors", []) or []
    metadata = getattr(answer, "metadata", {}) or {}
    is_failed = bool(errors) or (metadata.get("generation_status") == "failed")
    error_msg = "; ".join(errors) if errors else ""

    correctness = _get_metric_val(metrics, "answer_correctness")
    completeness_val = _get_metric_val(metrics, "completeness")
    faithfulness_val = _get_metric_val(metrics, "faithfulness")
    relevancy_val = _get_metric_val(metrics, "answer_relevancy")
    citation_val = _get_metric_val(metrics, "citation_correctness")
    recall_val = _get_metric_val(metrics, "retrieval_recall_at_k")
    mrr_val = _get_metric_val(metrics, "mrr")

    retrieval_lat = getattr(answer, "retrieval_latency_ms", None) if not is_direct else None
    gen_lat = getattr(answer, "generation_latency_ms", None)
    tot_lat = getattr(answer, "total_latency_ms", None)

    retrieved_sources = getattr(answer, "retrieved_sources", []) or []
    chunk_count = len(retrieved_sources) if not is_direct else None
    graph_path_count = None
    if strategy in (RetrievalStrategy.GRAPH.value, RetrievalStrategy.HYBRID.value, "GRAPH", "HYBRID"):
        graph_evidence = getattr(answer, "graph_evidence", None)
        if graph_evidence and getattr(graph_evidence, "relationships", None):
            graph_path_count = len(graph_evidence.relationships)
        else:
            graph_path_count = 0

    strengths: list[str] = []
    weaknesses: list[str] = []


    if is_failed:
        status_label = "Failed"
        if "quota" in error_msg.lower() or "resource_exhausted" in error_msg.lower():
            status_label = "Quota Exceeded"
        composite_score = -1.0
        weaknesses.append(f"Execution failed: {error_msg or 'Generation error'}")
        return ApproachScore(
            strategy=strategy,
            composite_score=composite_score,
            correctness=None,
            completeness=None,
            faithfulness=None,
            relevancy=None,
            citation_score=None,
            retrieval_recall=None,
            mrr=None,
            retrieval_latency_ms=retrieval_lat,
            generation_latency_ms=gen_lat,
            total_latency_ms=tot_lat,
            retrieved_chunk_count=chunk_count,
            graph_path_count=graph_path_count,
            is_failed=True,
            status_label=status_label,
            strengths=[],
            weaknesses=weaknesses,
        )

    status_label = "Success"

    # Latency penalty (minor tie-breaker, max 0.05 reduction for very high latency)
    latency_penalty = min(0.05, (tot_lat or 0.0) / 60000.0) if tot_lat else 0.0

    if has_ground_truth and correctness is not None:
        if is_direct:
            composite_score = (
                0.55 * correctness
                + 0.25 * (completeness_val if completeness_val is not None else 0.0)
                + 0.20 * (relevancy_val if relevancy_val is not None else 0.0)
                - latency_penalty
            )
        else:
            composite_score = (
                0.40 * correctness
                + 0.20 * (completeness_val if completeness_val is not None else 0.0)
                + 0.20 * (faithfulness_val if faithfulness_val is not None else 0.0)
                + 0.10 * (relevancy_val if relevancy_val is not None else 0.0)
                + 0.10 * (citation_val if citation_val is not None else 0.8)
                - latency_penalty
            )
    else:
        # No reference answer: score is provisional based on evidence support and relevancy
        if is_direct:
            composite_score = (
                0.60 * (relevancy_val if relevancy_val is not None else 0.5)
                - latency_penalty
            )
        else:
            composite_score = (
                0.45 * (faithfulness_val if faithfulness_val is not None else 0.0)
                + 0.35 * (relevancy_val if relevancy_val is not None else 0.0)
                + 0.20 * (citation_val if citation_val is not None else 0.5)
                - latency_penalty
            )

    composite_score = round(max(0.0, min(1.0, composite_score)), 4)

    # Compile Strengths
    if correctness is not None and correctness >= 0.75:
        strengths.append(f"High ground-truth correctness ({correctness:.2f})")
    if completeness_val is not None and completeness_val >= 0.80:
        strengths.append(f"Comprehensive reference coverage ({completeness_val:.2f})")
    if faithfulness_val is not None and faithfulness_val >= 0.80:
        strengths.append(f"Strong corpus evidence grounding ({faithfulness_val:.2f})")
    if relevancy_val is not None and relevancy_val >= 0.80:
        strengths.append(f"Highly responsive to user query ({relevancy_val:.2f})")
    if citation_val is not None and citation_val >= 0.75:
        strengths.append(f"Verified source citations ({citation_val:.2f})")
    if recall_val is not None and recall_val >= 0.80:
        strengths.append(f"High reference doc retrieval recall ({recall_val:.2f})")
    if graph_path_count and graph_path_count > 0:
        strengths.append(f"Extracted {graph_path_count} multi-hop graph relationship(s)")
    if chunk_count and chunk_count > 0 and not is_direct:
        strengths.append(f"Retrieved {chunk_count} corpus passage(s)")
    if tot_lat is not None and tot_lat < 1200:
        strengths.append(f"Low latency execution ({tot_lat:.0f} ms)")

    # Compile Weaknesses
    if is_direct:
        weaknesses.append("No corpus evidence or citations available to independently verify claims")
    if correctness is not None and correctness < 0.45:
        weaknesses.append(f"Low factual correctness against reference answer ({correctness:.2f})")
    if faithfulness_val is not None and faithfulness_val < 0.50:
        weaknesses.append(f"Weak evidence grounding ({faithfulness_val:.2f}); potential hallucination risk")
    if citation_val is not None and citation_val < 0.40 and not is_direct:
        weaknesses.append(f"Low citation support ({citation_val:.2f})")
    if tot_lat is not None and tot_lat > 3500:
        weaknesses.append(f"Higher total latency ({tot_lat:.0f} ms)")
    if chunk_count == 0 and not is_direct:
        weaknesses.append("No matching document chunks retrieved from corpus")

    return ApproachScore(
        strategy=strategy,
        composite_score=composite_score,
        correctness=correctness,
        completeness=completeness_val,
        faithfulness=faithfulness_val,
        relevancy=relevancy_val,
        citation_score=citation_val,
        retrieval_recall=recall_val,
        mrr=mrr_val,
        retrieval_latency_ms=retrieval_lat,
        generation_latency_ms=gen_lat,
        total_latency_ms=tot_lat,
        retrieved_chunk_count=chunk_count,
        graph_path_count=graph_path_count,
        is_failed=False,
        status_label=status_label,
        strengths=strengths,
        weaknesses=weaknesses,
    )


def recommend_best_approach(
    evaluations: dict[str, ApproachScore],
    has_ground_truth: bool,
    question: str = "",
) -> RecommendationResult:
    """Recommend the optimal approach based on measured results and transparent ranking policy."""
    if not evaluations:
        return RecommendationResult(
            recommended_strategy=None,
            reason="No evaluation approaches were executed.",
            confidence="Low",
            is_definitive=False,
            status="inconclusive",
        )

    successful = [app for app in evaluations.values() if not app.is_failed]
    if not successful:
        return RecommendationResult(
            recommended_strategy=None,
            reason="All evaluated approaches failed or encountered API/generation errors.",
            confidence="Low",
            is_definitive=False,
            status="failed",
            strengths=[],
            weaknesses=["All approaches failed execution"],
        )

    # Inconclusive detection: no ground truth and all methods have very low evidence support & relevance
    if not has_ground_truth:
        rag_apps = [a for a in successful if a.strategy != "DIRECT"]
        all_rag_low = all((a.faithfulness or 0.0) < 0.35 for a in rag_apps) if rag_apps else False
        top_relevancy = max((a.relevancy or 0.0) for a in successful) if successful else 0.0
        if all_rag_low and top_relevancy < 0.40:
            return RecommendationResult(
                recommended_strategy=None,
                reason="Corpus retrieval yielded insufficient supporting evidence and relevance across all approaches to rank definitively.",
                strengths=[],
                weaknesses=["Insufficient corpus evidence", "Low answer relevancy"],
                relevant_metrics={},
                confidence="Low",
                is_definitive=False,
                status="inconclusive",
                score_breakdown={a.strategy: a.composite_score for a in successful},
            )

    if len(successful) == 1:
        only = successful[0]
        reason = f"{only.strategy} was the only approach that completed successfully."
        if not has_ground_truth:
            reason += " (Provisional recommendation: no reference answer supplied for objective verification)."
        return RecommendationResult(
            recommended_strategy=only.strategy,
            reason=reason,
            strengths=only.strengths,
            weaknesses=only.weaknesses,
            relevant_metrics={"composite_score": only.composite_score, "status": only.status_label},
            confidence="High",
            is_definitive=has_ground_truth,
            status="definitive" if has_ground_truth else "provisional",
            score_breakdown={only.strategy: only.composite_score},
        )


    # Sort successful approaches by composite score descending
    sorted_apps = sorted(successful, key=lambda a: a.composite_score, reverse=True)
    top = sorted_apps[0]
    runner_up = sorted_apps[1]
    score_diff = top.composite_score - runner_up.composite_score

    score_map = {app.strategy: app.composite_score for app in sorted_apps}
    relevant_metrics = {
        "composite_score": top.composite_score,
        "correctness": top.correctness,
        "faithfulness": top.faithfulness,
        "answer_relevancy": top.relevancy,
        "total_latency_ms": top.total_latency_ms,
        "runner_up_strategy": runner_up.strategy,
        "runner_up_score": runner_up.composite_score,
    }

    # 1. Tie detection: score difference < 0.02 and latencies within 200 ms
    lat_diff = abs((top.total_latency_ms or 0.0) - (runner_up.total_latency_ms or 0.0))
    if score_diff < 0.02 and lat_diff < 200:
        return RecommendationResult(
            recommended_strategy=f"{top.strategy} / {runner_up.strategy}",
            reason=(
                f"Tie between {top.strategy} ({top.composite_score:.2f}) and {runner_up.strategy} "
                f"({runner_up.composite_score:.2f}). Both approaches exhibited nearly identical quality "
                f"and latency characteristics."
            ),
            strengths=top.strengths + [f"Matches performance of {runner_up.strategy}"],
            weaknesses=[],
            relevant_metrics=relevant_metrics,
            confidence="Low",
            is_definitive=has_ground_truth,
            status="tie",
            score_breakdown=score_map,
        )

    # 2. Inconclusive detection: no ground truth and all methods have very low evidence support
    if not has_ground_truth:
        rag_apps = [a for a in successful if a.strategy != "DIRECT"]
        all_rag_low = all((a.faithfulness or 0.0) < 0.35 for a in rag_apps) if rag_apps else False
        if all_rag_low and (top.relevancy or 0.0) < 0.40:
            return RecommendationResult(
                recommended_strategy=None,
                reason="Corpus retrieval yielded insufficient supporting evidence and relevance across all approaches to rank definitively.",
                strengths=[],
                weaknesses=["Insufficient corpus evidence", "Low answer relevancy"],
                relevant_metrics=relevant_metrics,
                confidence="Low",
                is_definitive=False,
                status="inconclusive",
                score_breakdown=score_map,
            )

    # 3. Clear winner
    is_definitive = has_ground_truth
    status = "definitive" if is_definitive else "provisional"
    confidence = "High" if score_diff >= 0.10 else ("Medium" if score_diff >= 0.04 else "Low")

    if is_definitive:
        reason = (
            f"{top.strategy} achieved the highest composite accuracy ({top.composite_score:.2f}) "
            f"with verified ground-truth correctness ({top.correctness or 0.0:.2f}) "
            f"and reliable supporting evidence."
        )
    else:
        reason = (
            f"{top.strategy} delivered the strongest evidence support ({top.faithfulness or 0.0:.2f}) "
            f"and query relevancy ({top.relevancy or 0.0:.2f}) among evaluated approaches. "
            f"(Provisional recommendation: no reference answer supplied for objective verification)."
        )

    return RecommendationResult(
        recommended_strategy=top.strategy,
        reason=reason,
        strengths=top.strengths,
        weaknesses=top.weaknesses,
        relevant_metrics=relevant_metrics,
        confidence=confidence,
        is_definitive=is_definitive,
        status=status,
        score_breakdown=score_map,
    )


def rank_benchmark_approaches(run_rows: list[dict], query_summaries: list[dict]) -> dict[str, Any]:
    """Aggregate benchmark results and compute overall ranking, win counts, and quality comparisons."""
    from collections import defaultdict

    strategy_stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    latencies: dict[str, list[float]] = defaultdict(list)
    winner_counts: dict[str, int] = defaultdict(int)

    ties_count = 0
    inconclusive_count = 0
    failed_queries = 0

    for q_sum in query_summaries:
        rec = q_sum.get("recommendation", {})
        status = rec.get("status")
        winner = rec.get("recommended_strategy")
        if status == "tie":
            ties_count += 1
        elif status == "inconclusive":
            inconclusive_count += 1
        elif status == "failed":
            failed_queries += 1
        elif winner and " / " not in winner:
            winner_counts[winner] += 1

    for row in run_rows:
        strat = row.get("strategy")
        if not strat:
            continue
        tot_lat = row.get("total_latency_ms")
        if tot_lat is not None:
            latencies[strat].append(tot_lat)

        mets = row.get("metrics") or {}
        for m_name, m_data in mets.items():
            val = None
            if isinstance(m_data, dict) and m_data.get("available", True):
                val = m_data.get("value")
            elif isinstance(m_data, (int, float)):
                val = float(m_data)
            if val is not None:
                strategy_stats[strat][m_name].append(val)

    strategy_rankings = []
    for strat, lats in latencies.items():
        m_dict = {
            m_name: round(sum(vals) / len(vals), 4)
            for m_name, vals in strategy_stats[strat].items()
            if vals
        }
        avg_lat = round(sum(lats) / len(lats), 1) if lats else None
        med_lat = round(median(lats), 1) if lats else None
        wins = winner_counts.get(strat, 0)

        # Overall composite score for benchmark ranking
        corr = m_dict.get("answer_correctness")
        faith = m_dict.get("faithfulness")
        rel = m_dict.get("answer_relevancy")

        components = [c for c in (corr, faith, rel) if c is not None]
        composite = round(sum(components) / len(components), 4) if components else 0.0

        strategy_rankings.append(
            {
                "strategy": strat,
                "win_count": wins,
                "composite_score": composite,
                "average_latency_ms": avg_lat,
                "median_latency_ms": med_lat,
                "metrics": m_dict,
            }
        )

    # Sort strategies by win count descending, then by composite score descending
    strategy_rankings.sort(key=lambda s: (s["win_count"], s["composite_score"]), reverse=True)

    has_any_success = any(s["win_count"] > 0 for s in strategy_rankings) or any(s["composite_score"] > 0 for s in strategy_rankings)

    overall_best = strategy_rankings[0]["strategy"] if (strategy_rankings and has_any_success) else None
    overall_reason = ""
    if overall_best:
        best_data = strategy_rankings[0]
        overall_reason = (
            f"{overall_best} demonstrated the strongest overall benchmark performance with "
            f"{best_data['win_count']} per-query win(s) and composite score of {best_data['composite_score']:.3f} "
            f"across evaluated queries."
        )
    else:
        overall_best = None
        overall_reason = "All evaluated approaches failed or encountered API/generation errors across the benchmark dataset. No approach could be recommended."

    return {
        "overall_best": overall_best,
        "overall_reason": overall_reason,
        "winner_counts": dict(winner_counts),
        "ties_count": ties_count,
        "inconclusive_count": inconclusive_count,
        "failed_count": failed_queries,
        "strategy_rankings": strategy_rankings,
    }
