"""RAG Benchmark Lab — Evaluation & Recommendation System."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import math
from pathlib import Path
import uuid

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rag_benchmark_lab.app import get_context
from rag_benchmark_lab.evaluation.benchmark_runner import BenchmarkRunner
from rag_benchmark_lab.evaluation.dataset import load_dataset, sample_dataset
from rag_benchmark_lab.evaluation.metrics import latency_summary, metric_bundle
from rag_benchmark_lab.evaluation.recommendation import (
    ApproachScore,
    evaluate_approach,
    rank_benchmark_approaches,
    recommend_best_approach,
)
from rag_benchmark_lab.schemas.documents import RetrievalStrategy
from rag_benchmark_lab.schemas.evaluation import EvaluationExample
from rag_benchmark_lab.utils.auth import get_current_user_name, require_auth
from rag_benchmark_lab.utils.ui import (
    format_chat_markdown,
    format_latency,
    format_metric,
    get_current_theme,
    get_plotly_layout,
    inject_custom_css,
    render_dev_mode_banner,
    render_html,
    render_kb_selector,
    render_topnav,
)

st.set_page_config(
    page_title="Evaluate RAG Performance — RAG Benchmark Lab",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_topnav(active_page="Evaluation")

user_id = require_auth()
context = get_context()
user_name = get_current_user_name() or "Researcher"
current_theme = get_current_theme()

# Strategy metadata mapping
STRAT_META = {
    RetrievalStrategy.VECTOR.value: {
        "label": "Vector RAG",
        "icon": "⚡",
        "color": "#60A5FA" if current_theme == "dark" else "#2563EB",
        "desc": "Qdrant dense embeddings",
    },
    RetrievalStrategy.GRAPH.value: {
        "label": "GraphRAG",
        "icon": "🕸️",
        "color": "#A78BFA" if current_theme == "dark" else "#7C3AED",
        "desc": "Neo4j entities & relations",
    },
    RetrievalStrategy.HYBRID.value: {
        "label": "Hybrid RAG",
        "icon": "🔀",
        "color": "#34D399" if current_theme == "dark" else "#059669",
        "desc": "Vector + Graph with RRF fusion",
    },
}

ALL_STRAT_KEYS = [
    RetrievalStrategy.VECTOR.value,
    RetrievalStrategy.GRAPH.value,
    RetrievalStrategy.HYBRID.value,
]

# Theme-adaptive Plotly layout
DARK_LAYOUT = get_plotly_layout(current_theme)


def _fmt_score(val: float | None) -> str:
    if val is None or pd.isna(val):
        return "N/A"
    return f"{float(val):.3f}"


def _fmt_lat(val: float | None) -> str:
    if val is None or pd.isna(val):
        return "N/A"
    return f"{float(val):.1f} ms"


def _fmt_count(val: int | float | None) -> str:
    if val is None or pd.isna(val):
        return "N/A"
    return str(int(val))


def build_comparison_matrix_df(eval_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Construct comparison matrix table across Vector RAG, GraphRAG, and Hybrid RAG."""
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

    col_headers = ["Parameter"] + [STRAT_META[strat]["label"] for strat in ALL_STRAT_KEYS]
    data_rows = []

    for label, key, p_type in parameters:
        row_dict = {"Parameter": label}
        for strat in ALL_STRAT_KEYS:
            hdr = STRAT_META[strat]["label"]
            res = eval_results.get(strat)
            if not res:
                row_dict[hdr] = "N/A"
                continue

            if p_type == "score":
                mets = res.get("metrics") or {}
                m_val = mets.get(key)
                if isinstance(m_val, dict):
                    row_dict[hdr] = _fmt_score(m_val.get("value") if m_val.get("available", True) else None)
                else:
                    row_dict[hdr] = _fmt_score(m_val)
            elif p_type == "latency":
                row_dict[hdr] = _fmt_lat(res.get(key))
            elif p_type == "count":
                row_dict[hdr] = _fmt_count(res.get(key))
            elif p_type == "status":
                row_dict[hdr] = str(res.get("status") or "Success")

        data_rows.append(row_dict)

    return pd.DataFrame(data_rows, columns=col_headers)


def render_recommendation_card(rec: dict[str, Any], query_text: str = "") -> None:
    """Render the recommended approach card with strengths, weaknesses, and confidence."""
    strat = rec.get("recommended_strategy")
    status = rec.get("status", "provisional")
    is_def = rec.get("is_definitive", False)
    confidence = rec.get("confidence", "Medium")
    reason = rec.get("reason", "")
    strengths = rec.get("strengths", [])
    weaknesses = rec.get("weaknesses", [])
    metrics = rec.get("relevant_metrics", {})

    status_pill_bg = "#7F1D1D" if status == "failed" else ("#064E3B" if is_def else "#78350F")
    status_pill_text = "#FCA5A5" if status == "failed" else ("#6EE7B7" if is_def else "#FCD34D")
    status_title = "All Approaches Failed" if status == "failed" else ("Definitive Ground Truth" if is_def else ("Tie" if status == "tie" else ("Inconclusive" if status == "inconclusive" else "Provisional Recommendation")))

    strat_color = "#EF4444" if status == "failed" else "#34D399"
    strat_label = strat or "No Approach Recommended"
    if status == "failed":
        strat_label = "No Approach Recommended"
    elif strat and strat in STRAT_META:
        strat_color = STRAT_META[strat]["color"]
        strat_label = f"{STRAT_META[strat]['icon']} {STRAT_META[strat]['label']}"
    elif strat and " / " in strat:
        strat_color = "#F5B942"
        strat_label = f"⚖️ {strat}"

    st.markdown(
        f"""
        <div class="rag-card" style="border-left: 5px solid {strat_color}; padding: 20px 24px; margin: 16px 0 24px 0;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">
            <div>
              <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); font-weight: 700;">
                Recommended Approach for This Query
              </span>
              <h2 style="font-size: 24px; font-weight: 800; color: var(--text); margin: 2px 0 0 0;">
                {strat_label}
              </h2>
            </div>
            <div style="display: flex; gap: 8px; align-items: center;">
              <span style="background: {status_pill_bg}; color: {status_pill_text}; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 12px; border: 1px solid {status_pill_text}40;">
                {status_title}
              </span>
              <span style="background: var(--card-hi); color: var(--accent); font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 12px; border: 1px solid var(--accent-border);">
                Confidence: {confidence}
              </span>
            </div>
          </div>
          <p style="font-size: 14px; color: var(--text); margin: 0 0 14px 0; line-height: 1.55;">
            {reason}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_str, c_wk = st.columns(2)
    with c_str:
        st.markdown(
            """
            <div class="rag-card" style="padding: 14px 18px; min-height: 120px;">
              <strong style="color: var(--success); font-size: 13px;">✓ Main Strengths</strong>
              <div style="margin-top: 6px; font-size: 13px; color: var(--text);">
            """,
            unsafe_allow_html=True,
        )
        if strengths:
            for s in strengths:
                st.markdown(f"• {s}")
        else:
            st.caption("No notable strengths recorded.")
        st.markdown("</div></div>", unsafe_allow_html=True)

    with c_wk:
        st.markdown(
            """
            <div class="rag-card" style="padding: 14px 18px; min-height: 120px;">
              <strong style="color: var(--error); font-size: 13px;">⚠ Main Weaknesses / Limitations</strong>
              <div style="margin-top: 6px; font-size: 13px; color: var(--text);">
            """,
            unsafe_allow_html=True,
        )
        if weaknesses:
            for w in weaknesses:
                st.markdown(f"• {w}")
        else:
            st.caption("No significant weaknesses detected.")
        st.markdown("</div></div>", unsafe_allow_html=True)


# ── Page Header ──────────────────────────────────────────────────────────────
top_c1, top_c2 = st.columns([3, 1], vertical_alignment="center")
with top_c1:
    render_html(
        """
        <div style="margin-top: 20px; margin-bottom: 16px;">
          <div style="display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;color:var(--accent);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">
            <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--accent);"></span>
            Benchmarking Suite
          </div>
          <h1 style="font-size: 28px; font-weight: 800; color: var(--text); margin: 0 0 4px 0; letter-spacing: -0.03em;">
            Evaluate RAG Performance
          </h1>
          <p style="font-size: 14px; color: var(--text-muted); margin: 0;">
            Three-way empirical evaluation and algorithmic recommendation: Vector RAG, GraphRAG, and Hybrid RAG.
          </p>
        </div>
        """
    )
with top_c2:
    active_kb_id = render_kb_selector(context, user_id, key_prefix="eval")

_kb_doc_ids: list[str] = []
if active_kb_id:
    try:
        _kb_doc_ids = context.user_registry.get_kb_document_ids(active_kb_id, user_id)
    except Exception:
        _kb_doc_ids = []

# ── Scale & Methodology Helper ───────────────────────────────────────────────
with st.expander("Understanding Evaluation Metrics & Scale (0.0 to 1.0)", expanded=False):
    st.markdown(
        """
        <div style="font-size: 13px; color: var(--text-muted); margin-bottom: 12px;">
          All quality scores are normalized on a consistent scale from <strong>0.000 to 1.000</strong> (higher is better).
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; font-size: 13px;">
          <div style="background: var(--card-hi); padding: 12px 14px; border-radius: 8px; border: 1px solid var(--border);">
            <strong style="color: var(--success);">Answer Correctness</strong><br/>
            Deterministic exact match, token overlap, and key fact alignment against ground-truth reference answer.
          </div>
          <div style="background: var(--card-hi); padding: 12px 14px; border-radius: 8px; border: 1px solid var(--border);">
            <strong style="color: var(--accent);">Faithfulness</strong><br/>
            Measures whether claims in the generated answer are grounded in retrieved source chunks and graph relationships.
          </div>
          <div style="background: var(--card-hi); padding: 12px 14px; border-radius: 8px; border: 1px solid var(--border);">
            <strong style="color: var(--vector);">Answer Relevancy & Completeness</strong><br/>
            Assesses whether the response directly addresses the query and covers all necessary facts from the reference answer.
          </div>
          <div style="background: var(--card-hi); padding: 12px 14px; border-radius: 8px; border: 1px solid var(--border);">
            <strong style="color: var(--warning);">Citation Correctness & MRR</strong><br/>
            Verifies that citations point to valid retrieved chunks that support the accompanying statements.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SELECT EVALUATION MODE
# ═════════════════════════════════════════════════════════════════════════════
eval_mode = st.radio(
    "Select Evaluation Mode",
    options=["Mode 1: Single Query Evaluation", "Mode 2: Benchmark Dataset Evaluation"],
    horizontal=True,
    index=0,
    key="eval_mode_selector",
)

st.markdown("<hr style='border-color: var(--border); margin: 16px 0;'/>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# MODE 1: SINGLE QUERY EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
if eval_mode == "Mode 1: Single Query Evaluation":
    st.markdown("### Single Query Evaluation & Recommendation")
    st.caption("Submit any question to evaluate Vector RAG, GraphRAG, and Hybrid RAG approaches and receive an algorithmic recommendation.")

    built_in_examples = sample_dataset()
    built_in_map = {ex.question.strip().lower(): ex for ex in built_in_examples}

    sq_col1, sq_col2 = st.columns([3, 2])
    with sq_col1:
        single_query = st.text_area(
            "Enter Question",
            value="What is the primary purpose of the document ingestion pipeline?",
            height=100,
            key="single_eval_query_input",
            help="Type any free-form question. If it matches a benchmark question, reference answers are auto-loaded.",
        )
    with sq_col2:
        norm_q = single_query.strip().lower()
        matched_ex = built_in_map.get(norm_q)
        default_ref = matched_ex.expected_answer if matched_ex and matched_ex.expected_answer else ""

        # Synchronize default reference answer when query input changes
        if st.session_state.get("_prev_sq_query") != single_query.strip():
            st.session_state["_prev_sq_query"] = single_query.strip()
            st.session_state["single_eval_ref_input"] = default_ref

        if matched_ex:
            st.success(f"✓ Matched benchmark test case `{matched_ex.query_id}`. Ground-truth answer auto-detected.")

        single_ref_answer = st.text_area(
            "Expected Reference Answer (Optional)",
            value=default_ref,
            height=85,
            key="single_eval_ref_input",
            help="Providing an expected answer enables deterministic ground-truth correctness, completeness, and context recall scoring.",
        )

        default_rel_docs = ", ".join(matched_ex.relevant_document_ids) if matched_ex and matched_ex.relevant_document_ids else ""
        if st.session_state.get("_prev_sq_query_docs") != single_query.strip():
            st.session_state["_prev_sq_query_docs"] = single_query.strip()
            st.session_state["single_eval_reldocs_input"] = default_rel_docs

        single_rel_docs_input = st.text_input(
            "Known Relevant Document/Chunk Identifiers (Optional for Recall@K & MRR)",
            value=default_rel_docs,
            key="single_eval_reldocs_input",
            help="Comma-separated document names or chunk IDs (e.g., sample-doc-1, sample_doc_1.txt). If omitted, Recall@K and MRR display N/A with an explanation.",
        )

        if not single_ref_answer.strip():
            st.info("ℹ️ No reference answer supplied. Operational and evidence metrics will be computed; ground-truth metrics will be N/A.")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    c_m1, c_m2 = st.columns([3, 1])
    with c_m1:
        sq_strategies = st.multiselect(
            "Approaches to Evaluate",
            options=ALL_STRAT_KEYS,
            default=ALL_STRAT_KEYS,
            format_func=lambda s: f"{STRAT_META[s]['icon']} {STRAT_META[s]['label']} ({STRAT_META[s]['desc']})",
            key="sq_selected_strategies",
        )
    with c_m2:
        st.markdown("**Configured Provider & Model:**")
        svc_name = getattr(context.llm_service, "model_name", context.settings.groq_model)
        prov_title = getattr(context.llm_service, "provider_name", context.settings.llm_provider.upper())
        is_mock = getattr(context.llm_service, "mock_mode", False)
        st.caption(f"• Provider: **{prov_title}**")
        st.caption(f"• Model: **{svc_name}**")
        st.caption(f"• Mode: **{'Dev Mock Mode' if is_mock else 'Live LLM Provider'}**")
        if st.button("🔄 Reload Config", key="btn_reload_eval_config", help="Reload .env settings and re-initialize LLM provider"):
            st.cache_resource.clear()
            st.session_state.pop("single_eval_data", None)
            st.rerun()

    # Empty KB notice
    if len(_kb_doc_ids) == 0:
        st.warning(
            "⚠️ The active knowledge base has no documents. Vector RAG, GraphRAG, and Hybrid RAG will have empty corpus scope."
        )

    btn_c1, btn_c2 = st.columns([3, 1])
    with btn_c1:
        sq_btn = st.button("Evaluate Query Across Approaches", type="primary", key="btn_run_single_eval")
    with btn_c2:
        if st.session_state.get("single_eval_data"):
            if st.button("Clear Results", key="btn_clear_single_eval", help="Clear current evaluation presentation state"):
                st.session_state.pop("single_eval_data", None)
                st.rerun()

    if sq_btn and single_query.strip():
        # Reset presentation state for the new evaluation run
        st.session_state.pop("single_eval_data", None)
        st.session_state["sq_running"] = True
        run_id = f"single_{uuid.uuid4().hex[:12]}"
        has_gt = bool(single_ref_answer and single_ref_answer.strip())

        user_entered_docs = [d.strip() for d in single_rel_docs_input.split(",") if d.strip()]
        relevant_docs = user_entered_docs if user_entered_docs else (matched_ex.relevant_document_ids if matched_ex else [])
        ref_ctxs = matched_ex.reference_contexts if matched_ex else []

        eval_results: dict[str, dict[str, Any]] = {}
        approach_scores: dict[str, ApproachScore] = {}

        daily_quota_exhausted = False
        quota_reason = ""

        with st.spinner("Executing selected approaches and measuring quality & latency..."):
            for s_key in sq_strategies:
                strat_enum = RetrievalStrategy[s_key]
                skip_reason = quota_reason if daily_quota_exhausted else None

                ans = context.query_service.answer(
                    single_query.strip(),
                    manual_strategy=strat_enum,
                    document_ids=_kb_doc_ids,
                    skip_generation_reason=skip_reason,
                )

                for err in ans.errors:
                    if "QuotaExceededError" in err or "quota" in err.lower() or "resource_exhausted" in err.lower() or "RateLimitError" in err:
                        daily_quota_exhausted = True
                        prov_n = getattr(context.llm_service, "provider_name", "LLM")
                        quota_reason = f"{prov_n} quota or rate limit reached for the configured model."
                        break

                mets = metric_bundle(
                    ans,
                    single_ref_answer.strip() if has_gt else None,
                    relevant_docs,
                    ref_ctxs,
                    question=single_query.strip(),
                )

                app_score = evaluate_approach(s_key, ans, mets, has_gt)
                approach_scores[s_key] = app_score

                paths_cnt = None
                if s_key in (RetrievalStrategy.GRAPH.value, RetrievalStrategy.HYBRID.value):
                    if ans.graph_evidence and ans.graph_evidence.relationships:
                        paths_cnt = len(ans.graph_evidence.relationships)
                    else:
                        paths_cnt = 0

                citations_data = [
                    {
                        "document_id": c.document_id,
                        "document_name": c.document_name,
                        "chunk_id": c.chunk_id,
                        "chunk_index": c.chunk_index,
                        "score": c.score,
                    }
                    for c in ans.citations
                ]

                row = {
                    "run_id": run_id,
                    "query_id": matched_ex.query_id if matched_ex else "single_query",
                    "strategy": s_key,
                    "question": single_query.strip(),
                    "answer": ans.answer,
                    "expected_answer": single_ref_answer.strip() if has_gt else None,
                    "relevant_document_ids": relevant_docs,
                    "selected_strategy": ans.selected_strategy.value,
                    "model_name": ans.model_name or getattr(context.llm_service, "model_name", "unknown"),
                    "retrieval_latency_ms": ans.retrieval_latency_ms,
                    "generation_latency_ms": ans.generation_latency_ms,
                    "total_latency_ms": ans.total_latency_ms,
                    "retrieved_chunk_count": len(ans.retrieved_sources),
                    "graph_path_count": paths_cnt,
                    "citations": citations_data,
                    "retrieved_sources": [
                        {
                            "chunk_id": c.chunk_id,
                            "document_name": c.document_name,
                            "score": c.score,
                            "text": c.text[:300],
                        }
                        for c in ans.retrieved_sources
                    ],
                    "graph_evidence": {
                        "relationships": [
                            f"{r.source_entity_name} -> [{r.relationship_type}] -> {r.target_entity_name}"
                            for r in (ans.graph_evidence.relationships if ans.graph_evidence else [])
                        ]
                    },
                    "metrics": {name: asdict(metric) for name, metric in mets.items()},
                    "warnings": ans.warnings,
                    "errors": ans.errors,
                    "status": app_score.status_label,
                    "composite_score": app_score.composite_score,
                    "strengths": app_score.strengths,
                    "weaknesses": app_score.weaknesses,
                    "is_ground_truth_based": has_gt,
                }
                eval_results[s_key] = row

            rec = recommend_best_approach(approach_scores, has_gt, question=single_query.strip())
            rec_dict = asdict(rec)

            # Persist into result store
            summary_payload = {
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dataset_id": "single_query_evaluation",
                "dataset_hash": "single_eval",
                "strategies": sq_strategies,
                "total_queries": 1,
                "mock_mode": getattr(context.llm_service, "mock_mode", False),
                "model_name": getattr(context.llm_service, "model_name", "unknown"),
                "document_ids": _kb_doc_ids,
                "overall_best": rec.recommended_strategy,
                "overall_reason": rec.reason,
                "winner_counts": {rec.recommended_strategy: 1} if rec.recommended_strategy else {},
                "ties_count": 1 if rec.status == "tie" else 0,
                "inconclusive_count": 1 if rec.status == "inconclusive" else 0,
                "failed_count": 1 if rec.status == "failed" else 0,
            }
            context.result_store.save_run(run_id, summary_payload)

            for s_key, row in eval_results.items():
                row["recommendation"] = rec_dict
                context.result_store.save_result(f"{run_id}:single:{s_key}", run_id, row)

            st.session_state["single_eval_data"] = {
                "run_id": run_id,
                "query": single_query.strip(),
                "ref_answer": single_ref_answer.strip() if has_gt else None,
                "results": eval_results,
                "recommendation": rec_dict,
            }
            st.session_state["sq_running"] = False

    # Display Single Query Results if available
    single_data = st.session_state.get("single_eval_data")
    if single_data:
        st.markdown("<hr style='border-color: var(--border); margin: 24px 0;'/>", unsafe_allow_html=True)
        if single_data.get("query") and single_data.get("query").strip() != single_query.strip():
            st.info(f"ℹ️ Displaying evaluated results for: *\"{single_data.get('query')}\"*. (Click **Evaluate Query Across Approaches** to evaluate your updated question).")

        res_map = single_data["results"]
        all_failed = all(r.get("status") in ("Failed", "Quota Exceeded") or r.get("errors") for r in res_map.values())
        if all_failed:
            st.error(
                "❌ **All Evaluation Approaches Failed to Generate Answers**\n\n"
                "• **Root Cause:** The configured LLM provider reported errors (e.g. Quota Exceeded or Authentication Error) across all approaches.\n"
                "• **Quality Metrics & Recommendations:** Comparative quality scores, charts, and recommendations cannot be evaluated when generation fails.\n"
                "• **Resolution:** Check your LLM API quota or update your model/provider configuration, then rerun the evaluation."
            )

        # 1. Recommendation Section
        render_recommendation_card(single_data["recommendation"], query_text=single_data["query"])

        # 2. Side-by-side Answer Cards
        st.markdown("#### Generated Answers Across Approaches")
        res_map = single_data["results"]
        active_keys = [k for k in ALL_STRAT_KEYS if k in res_map]
        cols = st.columns(len(active_keys))

        for idx, k in enumerate(active_keys):
            row_data = res_map[k]
            meta = STRAT_META[k]
            with cols[idx]:
                is_failed = row_data.get("status") in ("Failed", "Quota Exceeded") or bool(row_data.get("errors"))
                status_color = "#F87171" if is_failed else "#34D399"
                lat_str = format_latency(row_data.get("total_latency_ms"), fallback="N/A", precision=1, treat_zero_as_unavailable=True)

                if is_failed:
                    err_text = "; ".join(row_data.get("errors", [])) or row_data.get("answer", "Generation failed.")
                    if err_text.startswith("Error generating answer: "):
                        err_text = err_text.replace("Error generating answer: ", "")
                    ans_html = f"""
                    <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.35); border-radius: 6px; padding: 10px; font-size: 12.5px; color: #FCA5A5; line-height: 1.45; max-height: 200px; overflow-y: auto;">
                      <strong>⚠️ Generation Failed:</strong><br/>
                      {err_text}
                    </div>
                    """
                else:
                    ans_html = f"""
                    <div style="background: var(--card-hi); border-radius: 4px; padding: 10px; font-size: 13px; color: var(--text); max-height: 200px; overflow-y: auto; line-height: 1.45;">
                      {format_chat_markdown(row_data.get('answer', ''))}
                    </div>
                    """

                st.markdown(
                    f"""
                    <div style="background: var(--card); border: 1px solid var(--border); border-top: 3px solid {meta['color']}; border-radius: 8px; padding: 12px 14px; margin-bottom: 12px; min-height: 260px;">
                      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 700; color: var(--text); font-size: 14px;">
                          {meta['icon']} {meta['label']}
                        </span>
                        <span style="font-size: 11px; color: {status_color}; font-weight: 700;">
                          {row_data.get('status')}
                        </span>
                      </div>
                      <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 8px;">
                        Latency: <strong>{lat_str}</strong> • Model: {row_data.get('model_name', 'LLM')}
                      </div>
                      {ans_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # 3. Three-Way Comparison Matrix Table
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("#### Three-Way Comparison Matrix")
        st.caption("Empirical measurements across answer quality, retrieval quality, and performance.")
        matrix_df = build_comparison_matrix_df(res_map)
        st.dataframe(matrix_df, use_container_width=True, hide_index=True)

        # Explanatory note if any metric is N/A
        na_explanations: list[str] = []

        for strat in active_keys:
            res = res_map.get(strat, {})
            mets = res.get("metrics") or {}
            for m_key, m_label in [
                ("answer_correctness", "Answer correctness"),
                ("completeness", "Completeness"),
                ("context_precision", "Context precision"),
                ("context_recall", "Context recall"),
                ("retrieval_recall_at_k", "Recall@K"),
                ("mrr", "Mean Reciprocal Rank (MRR)"),
            ]:
                m_obj = mets.get(m_key)
                if isinstance(m_obj, dict) and not m_obj.get("available", True):
                    raw_reason = m_obj.get("details", {}).get("reason", "missing_required_input")
                    if raw_reason == "not_applicable_for_direct_llm":
                        continue
                    elif raw_reason == "missing_reference_answer":
                        exp = f"**{m_label}**: Requires a valid reference answer to extract and verify expected facts."
                    elif raw_reason in ("missing_relevant_document_ids", "missing_ground_truth_relevance_labels"):
                        exp = f"**{m_label}**: Requires known ground-truth relevant document or chunk identifiers."
                    elif raw_reason == "missing_ground_truth_and_reference_answer":
                        exp = f"**{m_label}**: Requires a reference answer or ground-truth relevance labels to evaluate chunk relevance."
                    elif raw_reason == "generation_failed":
                        exp = f"**{m_label}**: Could not be calculated because answer generation encountered an error."
                    else:
                        exp = f"**{m_label}**: {raw_reason.replace('_', ' ').capitalize()}."
                    if exp not in na_explanations:
                        na_explanations.append(exp)

        if na_explanations:
            with st.expander("ℹ️ Metric Explanations for N/A Entries", expanded=False):
                st.caption("The following metrics could not be calculated or are inherently Not Applicable (N/A):")
                for exp in na_explanations:
                    st.markdown(f"• {exp}")

        # 4. Comparative Visualizations
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("#### Comparative Quality & Latency Breakdown")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            q_metrics_rows = []
            for k in active_keys:
                r = res_map[k]
                mets = r.get("metrics") or {}
                for m_id, m_label in [
                    ("answer_correctness", "Correctness"),
                    ("faithfulness", "Faithfulness"),
                    ("answer_relevancy", "Relevancy"),
                    ("completeness", "Completeness"),
                ]:
                    val = mets.get(m_id, {}).get("value") if isinstance(mets.get(m_id), dict) else mets.get(m_id)
                    if val is not None and not pd.isna(val):
                        q_metrics_rows.append({"Strategy": STRAT_META[k]["label"], "Metric": m_label, "Score": float(val)})

            if q_metrics_rows:
                df_q = pd.DataFrame(q_metrics_rows)
                fig_q = px.bar(
                    df_q,
                    x="Metric",
                    y="Score",
                    color="Strategy",
                    barmode="group",
                    title="Quality Scores by Approach (0.0 to 1.0)",
                    color_discrete_map={STRAT_META[k]["label"]: STRAT_META[k]["color"] for k in ALL_STRAT_KEYS},
                    height=360,
                )
                fig_q.update_layout(**DARK_LAYOUT)
                fig_q.update_layout(yaxis=dict(range=[0, 1.05]))
                st.plotly_chart(fig_q, use_container_width=True)
            elif all_failed:
                st.info("Quality comparison chart unavailable because all approaches failed answer generation.")
            else:
                st.info("No comparative quality metrics available for this query.")

        with chart_col2:
            lat_rows = []
            for k in active_keys:
                r = res_map[k]
                ret_l = r.get("retrieval_latency_ms")
                gen_l = r.get("generation_latency_ms")
                lbl = STRAT_META[k]["label"]
                if ret_l is not None:
                    lat_rows.append({"Strategy": lbl, "Stage": "Retrieval Latency", "ms": ret_l})
                if gen_l is not None:
                    lat_rows.append({"Strategy": lbl, "Stage": "Generation Latency", "ms": gen_l})

            if lat_rows:
                df_l = pd.DataFrame(lat_rows)
                fig_l = px.bar(
                    df_l,
                    x="Strategy",
                    y="ms",
                    color="Stage",
                    title="Retrieval vs. Generation Latency Breakdown (ms)",
                    color_discrete_map={"Retrieval Latency": "#39C6B5", "Generation Latency": "#8B7CFF"},
                    height=360,
                )
                fig_l.update_layout(**DARK_LAYOUT)
                st.plotly_chart(fig_l, use_container_width=True)
            else:
                st.info("No latency measurements recorded.")

        # 5. Expandable Evidence and Citation Inspector
        with st.expander("Expand Evidence, Citations & Knowledge Graph Inspector", expanded=False):
            insp_tabs = st.tabs([STRAT_META[k]["label"] for k in active_keys])
            for idx, k in enumerate(active_keys):
                r = res_map[k]
                with insp_tabs[idx]:
                    st.markdown("**Retrieved Chunks:**")
                    chunks = r.get("retrieved_sources", [])
                    if chunks:
                        for c in chunks:
                            st.caption(f"• **{c.get('document_name', 'doc')}** (Score: {c.get('score', 0):.3f}): {c.get('text', '')}")
                    else:
                        st.caption("No chunks retrieved.")

                    if k in (RetrievalStrategy.GRAPH.value, RetrievalStrategy.HYBRID.value):
                        st.markdown("**Knowledge Graph Paths:**")
                        rels = r.get("graph_evidence", {}).get("relationships", [])
                        if rels:
                            for rel in rels:
                                st.caption(f"• `{rel}`")
                        else:
                            st.caption("No graph relationships retrieved.")

                    st.markdown("**Citations:**")
                    cits = r.get("citations", [])
                    if cits:
                        for cit in cits:
                            st.caption(f"• Document: `{cit.get('document_name')}` | Chunk: `{cit.get('chunk_index')}`")
                    else:
                        st.caption("No formal citations generated.")


# ═════════════════════════════════════════════════════════════════════════════
# MODE 2: BENCHMARK DATASET EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("### Benchmark Dataset Evaluation")
    st.caption("Execute structured benchmark test cases across selected retrieval approaches and analyze aggregate rankings.")

    col_ds1, col_ds2 = st.columns([1, 2])
    with col_ds1:
        dataset_source = st.radio(
            "Dataset Source",
            options=["Built-in Benchmark Dataset", "Upload Custom JSON/JSONL"],
            index=0,
            key="bench_ds_source",
        )

    examples = []
    with col_ds2:
        if dataset_source == "Built-in Benchmark Dataset":
            examples = sample_dataset()
            st.success(f"Loaded built-in benchmark dataset with {len(examples)} validated test questions.")
        else:
            uploaded = st.file_uploader(
                "Upload JSON or JSONL benchmark dataset",
                type=["json", "jsonl"],
                help="Dataset must contain query_id, question, and expected_answer fields.",
                key="bench_uploader",
            )
            if uploaded:
                temp_dir = context.settings.app_data_dir
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_path = temp_dir / uploaded.name
                temp_path.write_bytes(uploaded.getvalue())
                try:
                    examples = load_dataset(temp_path)
                    st.success(f"Uploaded and parsed {len(examples)} valid test cases from '{uploaded.name}'.")
                except Exception as exc:
                    st.error(f"Failed to validate uploaded dataset: {exc}")
            else:
                st.info("Upload a JSON or JSONL file to benchmark your own custom questions.")

    if examples:
        with st.expander(f"Preview Benchmark Questions ({len(examples)} test cases)", expanded=False):
            df_prev = pd.DataFrame(
                [
                    {
                        "Query ID": ex.query_id,
                        "Question": ex.question,
                        "Expected Answer": ex.expected_answer or "N/A",
                        "Query Type": ex.metadata.get("query_type", "factual"),
                        "Difficulty": ex.metadata.get("difficulty", "medium"),
                        "Hops": ex.metadata.get("expected_hops", 1),
                    }
                    for ex in examples
                ]
            )
            st.dataframe(df_prev, use_container_width=True)

    st.markdown("<hr style='border-color: var(--border); margin: 16px 0;'/>", unsafe_allow_html=True)
    st.markdown("#### Configure Benchmark Execution")

    cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
    with cfg_col1:
        selected_bench_strategies = st.multiselect(
            "Approaches to Benchmark",
            options=ALL_STRAT_KEYS,
            default=ALL_STRAT_KEYS,
            format_func=lambda s: f"{STRAT_META[s]['icon']} {STRAT_META[s]['label']}",
            key="bench_strats_sel",
        )
    with cfg_col2:
        max_q = max(1, len(examples))
        query_limit = st.slider(
            "Questions to Evaluate",
            min_value=1,
            max_value=max_q,
            value=min(5, max_q),
            key="bench_query_limit",
        )
    with cfg_col3:
        st.markdown("**Execution Environment:**")
        st.caption(f"• LLM Model: **{getattr(context.llm_service, 'model_name', context.settings.groq_model)}**")
        st.caption(f"• Execution Mode: **{'Dev Offline Mock' if context.settings.llm_mock_mode else 'Live Provider'}**")

    bench_disabled = not examples or not selected_bench_strategies
    if st.session_state.get("bench_running", False):
        bench_disabled = True

    start_bench = st.button("Run Benchmark Evaluation", type="primary", disabled=bench_disabled, key="btn_run_bench")

    if start_bench:
        # Reset presentation state for new benchmark run
        st.session_state.pop("latest_bench_output", None)
        st.session_state["bench_running"] = True
        active_examples = examples[:query_limit]
        active_strat_enums = [RetrievalStrategy[s] for s in selected_bench_strategies]

        prog_bar = st.progress(0.0)
        status_msg = st.empty()
        runner = BenchmarkRunner(context.query_service, context.result_store)

        with st.spinner("Executing benchmark across selected approaches..."):
            try:
                run_output = runner.run(
                    active_examples,
                    strategies=active_strat_enums,
                    document_ids=_kb_doc_ids,
                )
                prog_bar.progress(1.0)
                status_msg.success(f"Benchmark completed. Persisted Run ID: `{run_output['run_id'][:12]}`")
                st.session_state["latest_bench_output"] = run_output
            except Exception as exc:
                status_msg.error(f"Benchmark execution error: {exc}")
            finally:
                st.session_state["bench_running"] = False

    # ═════════════════════════════════════════════════════════════════════════
    # BENCHMARK RESULTS & ANALYTICS
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown("<hr style='border-color: var(--border); margin: 24px 0;'/>", unsafe_allow_html=True)
    st.markdown("### Benchmark Results & Aggregated Rankings")

    all_results = context.result_store.list_results()
    all_runs = context.result_store.list_runs()

    if not all_results:
        st.info("No benchmark results recorded yet. Click 'Run Benchmark Evaluation' above to generate empirical data.")
    else:
        rows = []
        for res in all_results:
            if not isinstance(res, dict):
                continue
            if res.get("strategy") in (RetrievalStrategy.DIRECT.value, "DIRECT"):
                continue
            mets = res.get("metrics") or {}
            rec = res.get("recommendation") or {}
            rows.append(
                {
                    "run_id": res.get("run_id", "unknown"),
                    "query_id": res.get("query_id", "unknown"),
                    "strategy": res.get("strategy", "unknown"),
                    "question": res.get("question", ""),
                    "answer": res.get("answer", ""),
                    "expected_answer": res.get("expected_answer", ""),
                    "retrieval_latency_ms": res.get("retrieval_latency_ms"),
                    "generation_latency_ms": res.get("generation_latency_ms"),
                    "total_latency_ms": res.get("total_latency_ms"),
                    "answer_correctness": (
                        mets.get("answer_correctness", {}).get("value")
                        if isinstance(mets.get("answer_correctness"), dict)
                        else mets.get("answer_correctness")
                    ),
                    "faithfulness": (
                        mets.get("faithfulness", {}).get("value")
                        if isinstance(mets.get("faithfulness"), dict)
                        else mets.get("faithfulness")
                    ),
                    "answer_relevancy": (
                        mets.get("answer_relevancy", {}).get("value")
                        if isinstance(mets.get("answer_relevancy"), dict)
                        else mets.get("answer_relevancy")
                    ),
                    "completeness": (
                        mets.get("completeness", {}).get("value")
                        if isinstance(mets.get("completeness"), dict)
                        else mets.get("completeness")
                    ),
                    "citation_correctness": (
                        mets.get("citation_correctness", {}).get("value")
                        if isinstance(mets.get("citation_correctness"), dict)
                        else mets.get("citation_correctness")
                    ),
                    "context_precision": (
                        mets.get("context_precision", {}).get("value")
                        if isinstance(mets.get("context_precision"), dict)
                        else mets.get("context_precision")
                    ),
                    "context_recall": (
                        mets.get("context_recall", {}).get("value")
                        if isinstance(mets.get("context_recall"), dict)
                        else mets.get("context_recall")
                    ),
                    "retrieval_recall_at_k": (
                        mets.get("retrieval_recall_at_k", {}).get("value")
                        if isinstance(mets.get("retrieval_recall_at_k"), dict)
                        else mets.get("retrieval_recall_at_k")
                    ),
                    "mrr": (
                        mets.get("mrr", {}).get("value")
                        if isinstance(mets.get("mrr"), dict)
                        else mets.get("mrr")
                    ),
                    "retrieved_chunk_count": res.get("retrieved_chunk_count"),
                    "graph_path_count": res.get("graph_path_count"),
                    "status": res.get("status", "Success"),
                    "composite_score": res.get("composite_score"),
                    "recommended_strategy": rec.get("recommended_strategy"),
                    "recommendation_reason": rec.get("reason", ""),
                    "recommendation_status": rec.get("status", "provisional"),
                    "mock_mode": res.get("mock_mode", False),
                }
            )

        df_all = pd.DataFrame(rows)

        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            run_options = ["All Runs"] + sorted(list(df_all["run_id"].unique()), reverse=True)
            selected_run = st.selectbox("Filter by Run ID", options=run_options, index=0, key="bench_filter_run")
        with fcol2:
            strat_options = ["All Approaches"] + sorted(list(df_all["strategy"].unique()))
            selected_strat = st.selectbox("Filter by Approach", options=strat_options, index=0, key="bench_filter_strat")
        with fcol3:
            mode_filter = st.selectbox("Execution Mode", options=["All", "Real LLM Only", "Dev Mode Only"], index=0, key="bench_filter_mode")

        df = df_all.copy()
        if selected_run != "All Runs":
            df = df[df["run_id"] == selected_run]
        if selected_strat != "All Approaches":
            df = df[df["strategy"] == selected_strat]
        if mode_filter == "Real LLM Only":
            df = df[df["mock_mode"] == False]  # noqa: E712
        elif mode_filter == "Dev Mode Only":
            df = df[df["mock_mode"] == True]  # noqa: E712

        if df.empty:
            st.warning("No records match the current filter selection.")
        else:
            # 1. KPI Cards
            kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5, kpi_c6 = st.columns(6)
            with kpi_c1:
                st.metric("Total Answers", len(df))
            with kpi_c2:
                m_c = df["answer_correctness"].dropna().mean() if not df["answer_correctness"].dropna().empty else None
                st.metric("Avg Correctness", format_metric(m_c, fallback="N/A", precision=3) if m_c is not None else "N/A")
            with kpi_c3:
                m_f = df["faithfulness"].dropna().mean() if not df["faithfulness"].dropna().empty else None
                st.metric("Avg Faithfulness", format_metric(m_f, fallback="N/A", precision=3) if m_f is not None else "N/A")
            with kpi_c4:
                m_r = df["answer_relevancy"].dropna().mean() if not df["answer_relevancy"].dropna().empty else None
                st.metric("Avg Relevancy", format_metric(m_r, fallback="N/A", precision=3) if m_r is not None else "N/A")
            with kpi_c5:
                m_lat = df["total_latency_ms"].dropna().mean() if not df["total_latency_ms"].dropna().empty else None
                st.metric("Mean Latency", format_latency(m_lat, fallback="N/A", precision=1) if m_lat is not None else "N/A")
            with kpi_c6:
                winners = df["recommended_strategy"].dropna().value_counts()
                top_winner = winners.index[0] if not winners.empty else "N/A"
                st.metric("Top Approach", top_winner)

            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

            # 2. Overall Benchmark Recommendation Card
            current_run_summary = None
            if selected_run != "All Runs":
                for r_item in all_runs:
                    if r_item.get("run_id") == selected_run:
                        current_run_summary = r_item
                        break

            if current_run_summary and current_run_summary.get("overall_best"):
                b_best = current_run_summary.get("overall_best")
                b_reason = current_run_summary.get("overall_reason") or f"{b_best} achieved the highest aggregate score."
                w_counts = current_run_summary.get("winner_counts", {})
                ties = current_run_summary.get("ties_count", 0)
                inconcl = current_run_summary.get("inconclusive_count", 0)
                failed = current_run_summary.get("failed_count", 0)

                b_color = STRAT_META.get(b_best, {}).get("color", "#34D399")
                b_icon = STRAT_META.get(b_best, {}).get("icon", "🏆")
                b_label = STRAT_META.get(b_best, {}).get("label", b_best)

                wins_pills = " • ".join([f"<strong>{k}</strong>: {v} win(s)" for k, v in w_counts.items()])
                if ties:
                    wins_pills += f" • <strong>Ties</strong>: {ties}"
                if inconcl:
                    wins_pills += f" • <strong>Inconclusive</strong>: {inconcl}"
                if failed:
                    wins_pills += f" • <strong>Failed</strong>: {failed}"

                st.markdown(
                    f"""
                    <div style="background: var(--card); border: 1px solid var(--border); border-left: 5px solid {b_color}; border-radius: 8px; padding: 18px 22px; margin-bottom: 20px;">
                      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); font-weight: 700;">
                          Overall Best-Performing Benchmark Approach
                        </span>
                        <span style="background: var(--success-bg); color: var(--success); border: 1px solid var(--success-border); font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 12px;">
                          Benchmark Winner
                        </span>
                      </div>
                      <h3 style="font-size: 20px; font-weight: 800; color: var(--text); margin: 0 0 6px 0;">
                        {b_icon} {b_label}
                      </h3>
                      <p style="font-size: 14px; color: var(--text); margin: 0 0 10px 0;">
                        {b_reason}
                      </p>
                      <div style="font-size: 12px; color: var(--text-muted);">
                        {wins_pills}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif current_run_summary and not current_run_summary.get("overall_best"):
                failed_cnt = current_run_summary.get("failed_count", 0)
                reason_txt = current_run_summary.get("overall_reason") or "All evaluated approaches failed execution."
                st.markdown(
                    f"""
                    <div style="background: var(--card); border: 1px solid var(--border); border-left: 5px solid var(--error); border-radius: 8px; padding: 18px 22px; margin-bottom: 20px;">
                      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); font-weight: 700;">
                          Overall Benchmark Recommendation
                        </span>
                        <span style="background: var(--error-bg); color: var(--error); border: 1px solid var(--error-border); font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 12px;">
                          All Approaches Failed
                        </span>
                      </div>
                      <h3 style="font-size: 20px; font-weight: 800; color: var(--text); margin: 0 0 6px 0;">
                        No Approach Recommended
                      </h3>
                      <p style="font-size: 14px; color: var(--text); margin: 0 0 10px 0;">
                        {reason_txt}
                      </p>
                      <div style="font-size: 12px; color: var(--error);">
                        Failed Queries: {failed_cnt} • Check provider configuration and quota.
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 3. Benchmark Tabs
            b_tab_recs, b_tab_matrix, b_tab_charts, b_tab_inspect, b_tab_hist = st.tabs(
                [
                    "Per-Query Winners",
                    "Strategy Comparison Aggregates",
                    "Comparative Visualizations",
                    "Per-Question Inspector",
                    "Run History",
                ]
            )

            with b_tab_recs:
                st.markdown("#### Per-Query Recommendations & Winner Summary")
                # Deduplicate by query_id for the recommendations table
                df_q_recs = df.drop_duplicates(subset=["run_id", "query_id"]).copy()
                disp_cols = [
                    "query_id", "question", "recommended_strategy",
                    "recommendation_status", "recommendation_reason",
                ]
                valid_disp = [c for c in disp_cols if c in df_q_recs.columns]
                rec_table = df_q_recs[valid_disp].rename(
                    columns={
                        "query_id": "Query ID",
                        "question": "Question",
                        "recommended_strategy": "Recommended Approach",
                        "recommendation_status": "Status",
                        "recommendation_reason": "Recommendation Rationale",
                    }
                )
                st.dataframe(rec_table, use_container_width=True, hide_index=True)

            with b_tab_matrix:
                st.markdown("#### Three-Way Strategy Comparison Aggregates")
                st.caption("Mean values computed across all evaluated queries for each strategy.")

                metric_cols = [
                    "answer_correctness", "faithfulness", "answer_relevancy", "completeness",
                    "citation_correctness", "context_precision", "context_recall",
                    "retrieval_recall_at_k", "mrr",
                ]
                lat_cols = ["retrieval_latency_ms", "generation_latency_ms", "total_latency_ms"]

                agg_rows = []
                for strat_k in sorted(list(df["strategy"].unique())):
                    if strat_k in (RetrievalStrategy.DIRECT.value, "DIRECT"):
                        continue
                    sub = df[df["strategy"] == strat_k]
                    row_data = {"Approach": STRAT_META.get(strat_k, {}).get("label", strat_k)}
                    for m in metric_cols:
                        v = sub[m].dropna()
                        row_data[m] = round(v.mean(), 3) if not v.empty else None
                    for l in lat_cols:
                        v = sub[l].dropna()
                        row_data[l] = round(v.mean(), 1) if not v.empty else None
                    row_data["Evaluated Queries"] = len(sub)
                    agg_rows.append(row_data)

                agg_df = pd.DataFrame(agg_rows).rename(
                    columns={
                        "answer_correctness": "Correctness",
                        "faithfulness": "Faithfulness",
                        "answer_relevancy": "Relevancy",
                        "completeness": "Completeness",
                        "citation_correctness": "Citations",
                        "context_precision": "Ctx Precision",
                        "context_recall": "Ctx Recall",
                        "retrieval_recall_at_k": "Recall@K",
                        "mrr": "MRR",
                        "retrieval_latency_ms": "Retrieval Latency (ms)",
                        "generation_latency_ms": "Gen Latency (ms)",
                        "total_latency_ms": "Total Latency (ms)",
                    }
                )
                st.dataframe(agg_df.fillna("N/A"), use_container_width=True, hide_index=True)

            with b_tab_charts:
                st.markdown("#### Aggregate Strategy Visualizations")
                ch_c1, ch_c2 = st.columns(2)

                strat_group = df.groupby("strategy")[
                    ["answer_correctness", "faithfulness", "answer_relevancy", "completeness"]
                ].mean().reset_index()

                with ch_c1:
                    if not strat_group.empty:
                        df_melted = pd.melt(
                            strat_group,
                            id_vars=["strategy"],
                            var_name="Metric",
                            value_name="Score",
                        )
                        df_melted["Score"] = df_melted["Score"].fillna(0.0)
                        df_melted["Strategy Label"] = df_melted["strategy"].apply(lambda s: STRAT_META.get(s, {}).get("label", s))

                        fig_bar = px.bar(
                            df_melted,
                            x="Metric",
                            y="Score",
                            color="Strategy Label",
                            barmode="group",
                            title="Quality Metrics Across Approaches (0.0 to 1.0)",
                            color_discrete_map={v["label"]: v["color"] for v in STRAT_META.values()},
                            height=380,
                        )
                        fig_bar.update_layout(**DARK_LAYOUT)
                        fig_bar.update_layout(yaxis=dict(range=[0, 1.05]))
                        st.plotly_chart(fig_bar, use_container_width=True)

                with ch_c2:
                    fig_box = px.box(
                        df,
                        x="strategy",
                        y="total_latency_ms",
                        color="strategy",
                        points="all",
                        title="Total Latency Distribution (ms)",
                        color_discrete_map={k: v["color"] for k, v in STRAT_META.items()},
                        height=380,
                    )
                    fig_box.update_layout(**DARK_LAYOUT)
                    st.plotly_chart(fig_box, use_container_width=True)

                col_lat1, col_lat2 = st.columns(2)
                with col_lat1:
                    df_lat = df.groupby("strategy")[["retrieval_latency_ms", "generation_latency_ms"]].mean().reset_index()
                    df_lat["Strategy Label"] = df_lat["strategy"].apply(lambda s: STRAT_META.get(s, {}).get("label", s))
                    fig_stacked = go.Figure(
                        data=[
                            go.Bar(name="Retrieval Latency", x=df_lat["Strategy Label"], y=df_lat["retrieval_latency_ms"].fillna(0), marker_color="#39C6B5"),
                            go.Bar(name="Generation Latency", x=df_lat["Strategy Label"], y=df_lat["generation_latency_ms"].fillna(0), marker_color="#8B7CFF"),
                        ]
                    )
                    fig_stacked.update_layout(
                        barmode="stack",
                        title="Retrieval vs. Generation Latency Breakdown (Mean ms)",
                        height=360,
                        **DARK_LAYOUT,
                    )
                    st.plotly_chart(fig_stacked, use_container_width=True)

                with col_lat2:
                    w_series = df.drop_duplicates(subset=["run_id", "query_id"])["recommended_strategy"].value_counts()
                    if not w_series.empty:
                        fig_pie = px.pie(
                            values=w_series.values,
                            names=[STRAT_META.get(k, {}).get("label", k) for k in w_series.index],
                            title="Win Distribution Across Queries",
                            color=[STRAT_META.get(k, {}).get("label", k) for k in w_series.index],
                            color_discrete_map={v["label"]: v["color"] for v in STRAT_META.values()},
                            hole=0.45,
                            height=360,
                        )
                        fig_pie.update_layout(**DARK_LAYOUT)
                        st.plotly_chart(fig_pie, use_container_width=True)

            with b_tab_inspect:
                st.markdown("#### Per-Question Inspection")
                unique_queries = df[["query_id", "question"]].drop_duplicates()
                for _, q_row in unique_queries.iterrows():
                    qid = q_row["query_id"]
                    q_text = q_row["question"]
                    q_sub = df[df["query_id"] == qid]

                    winner = q_sub["recommended_strategy"].dropna().iloc[0] if not q_sub["recommended_strategy"].dropna().empty else "Tie / Inconclusive"
                    winner_lbl = STRAT_META.get(winner, {}).get("label", winner)
                    with st.expander(f"Query `{qid}`: {q_text[:75]}... [Winner: {winner_lbl}]", expanded=False):
                        st.markdown(f"**Question:** {q_text}")
                        ref_ans = q_sub["expected_answer"].dropna().iloc[0] if not q_sub["expected_answer"].dropna().empty else None
                        if ref_ans:
                            st.markdown(f"**Expected Reference Answer:** {ref_ans}")

                        st.markdown("<hr style='border-color: var(--border); margin: 8px 0;'/>", unsafe_allow_html=True)

                        # Side by side answers
                        cols = st.columns(len(q_sub))
                        for i, (_, a_row) in enumerate(q_sub.iterrows()):
                            s_name = a_row["strategy"]
                            with cols[i]:
                                st.markdown(f"**{STRAT_META.get(s_name, {}).get('icon', '')} {STRAT_META.get(s_name, {}).get('label', s_name)}**")
                                st.caption(f"Latency: {format_latency(a_row.get('total_latency_ms'))} • Status: {a_row.get('status')}")
                                st.markdown(f"> {a_row.get('answer', '')}")
                                st.caption(
                                    f"Correctness: {_fmt_score(a_row.get('answer_correctness'))} | "
                                    f"Faithfulness: {_fmt_score(a_row.get('faithfulness'))} | "
                                    f"Relevancy: {_fmt_score(a_row.get('answer_relevancy'))}"
                                )

                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Export Benchmark Results to CSV",
                    data=csv_data,
                    file_name="rag_benchmark_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_eval_csv",
                )

            with b_tab_hist:
                st.markdown("#### Historical Benchmark Runs")
                if all_runs:
                    df_runs = []
                    for run in all_runs:
                        df_runs.append(
                            {
                                "Run ID": run["run_id"][:12],
                                "Timestamp": run.get("timestamp", "")[:19].replace("T", " "),
                                "Dataset": run.get("dataset_id", "default"),
                                "Queries": run.get("total_queries", 0),
                                "Approaches": ", ".join(run.get("strategies", [])),
                                "Overall Winner": run.get("overall_best", "N/A"),
                                "Mode": "Dev Mock" if run.get("mock_mode") else "Live Provider",
                            }
                        )
                    st.dataframe(pd.DataFrame(df_runs), use_container_width=True, hide_index=True)
                else:
                    st.info("No recorded historical benchmark runs.")

st.markdown("<hr style='border-color: var(--border); margin: 24px 0;'/>", unsafe_allow_html=True)

# ── Knowledge Graph Explorer (Expandable) ────────────────────────────────────
with st.expander("Advanced: Knowledge Graph Explorer & Entity Inspection", expanded=False):
    st.markdown(
        """
        <div style="margin-bottom: 12px;">
          <h4 style="color: #F5F7FA; margin: 0 0 4px 0; font-size: 16px;">Knowledge Graph Neighborhood Inspector</h4>
          <p style="font-size: 13px; color: #A0AABC; margin: 0;">
            Inspect entity nodes, relational edges, and multi-hop radial neighborhoods extracted from your documents.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    neo4j_available = False
    try:
        context.neo4j_store.health()
        neo4j_available = True
        total_nodes = context.neo4j_store.count_entities()
        total_edges = context.neo4j_store.count_relationships()
        total_docs = context.neo4j_store.count_documents()
        source_mode = "neo4j"
    except Exception:
        neo4j_available = False
        reg_counts = context.registry.counts()
        total_nodes = reg_counts.get("entities", 0)
        total_edges = reg_counts.get("relationships", 0)
        total_docs = reg_counts.get("documents", 0)
        source_mode = "registry"

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric("Total Entities", total_nodes)
    with mcol2:
        st.metric("Total Relationships", total_edges)
    with mcol3:
        st.metric("Indexed Documents", total_docs)
    with mcol4:
        st.metric("Graph Backend", "Neo4j Active" if neo4j_available else "NetworkX / Registry Fallback")

    if total_nodes == 0:
        st.info("The Knowledge Graph is currently empty. Upload documents in Documents to extract entities and build relationships.")
    else:
        if source_mode == "neo4j":
            all_entities = context.neo4j_store.list_entities(limit=200)
            all_rels = context.neo4j_store.list_relationships(limit=300)
        else:
            all_entities = context.registry.list_entities()
            all_rels = context.registry.list_relationships()

        if all_entities:
            entity_map = {e["entity_id"]: e for e in all_entities}
            entity_ids = list(entity_map.keys())

            col_sel1, col_sel2 = st.columns([2, 1], vertical_alignment="center")
            with col_sel1:
                selected_entity_id = st.selectbox(
                    "Select Focal Entity (Center Node)",
                    options=entity_ids,
                    format_func=lambda eid: f"{entity_map[eid].get('canonical_name', eid)} ({entity_map[eid].get('entity_type', 'Concept')})",
                    key="eval_graph_focal_entity",
                )
            with col_sel2:
                st.caption(f"Entity ID: `{selected_entity_id[:16]}`")

            neighborhood_rels = [
                r for r in all_rels
                if (r.get("source_entity_id") == selected_entity_id or r.get("target_entity_id") == selected_entity_id)
            ]

            node_ids = {selected_entity_id}
            for r in neighborhood_rels:
                node_ids.add(r["source_entity_id"])
                node_ids.add(r["target_entity_id"])

            fig_g = go.Figure()
            node_positions: dict[str, tuple[float, float]] = {selected_entity_id: (0.0, 0.0)}
            periph = [nid for nid in node_ids if nid != selected_entity_id]
            radius = 2.0
            for i, nid in enumerate(periph):
                angle = 2 * math.pi * i / max(1, len(periph))
                node_positions[nid] = (radius * math.cos(angle), radius * math.sin(angle))

            for r in neighborhood_rels:
                sid, tid = r.get("source_entity_id"), r.get("target_entity_id")
                if sid in node_positions and tid in node_positions:
                    x0, y0 = node_positions[sid]
                    x1, y1 = node_positions[tid]
                    fig_g.add_trace(
                        go.Scatter(
                            x=[x0, x1, None],
                            y=[y0, y1, None],
                            mode="lines",
                            line=dict(width=1.5, color="#475569"),
                            hoverinfo="none",
                            showlegend=False,
                        )
                    )

            TYPE_COLORS = {
                "Technology": "#60A5FA",
                "Architecture": "#A78BFA",
                "Component": "#34D399",
                "Concept": "#8B7CFF",
                "Algorithm": "#F5B942",
            }
            for nid in node_ids:
                if nid not in node_positions:
                    continue
                nx, ny = node_positions[nid]
                ent = entity_map.get(nid, {})
                cname = ent.get("canonical_name", nid[:8])
                etype = ent.get("entity_type", "Concept")
                clr = TYPE_COLORS.get(etype, "#8B7CFF")
                sz = 26 if nid == selected_entity_id else 16

                fig_g.add_trace(
                    go.Scatter(
                        x=[nx],
                        y=[ny],
                        mode="markers+text",
                        text=[cname],
                        textposition="top center",
                        textfont=dict(size=11, color="#F5F7FA"),
                        marker=dict(size=sz, color=clr, line=dict(width=1.5, color="#10141D")),
                        name=etype,
                        hoverinfo="text",
                        hovertext=f"<b>{cname}</b><br/>Type: {etype}<br/>ID: {nid[:12]}...",
                        showlegend=False,
                    )
                )

            fig_g.update_layout(
                plot_bgcolor="#121723" if current_theme == "dark" else "#FFFFFF",
                paper_bgcolor="#121723" if current_theme == "dark" else "#FFFFFF",
                height=380,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
            st.plotly_chart(fig_g, use_container_width=True)
