"""RAG Benchmark Lab — Compare Retrieval Methods Page."""
from __future__ import annotations

from datetime import datetime, timezone
import uuid

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rag_benchmark_lab.app import get_context
from rag_benchmark_lab.schemas.comparison import (
    ComparisonRunRecord,
    ComparisonStrategyResultRecord,
)
from rag_benchmark_lab.schemas.documents import RetrievalStrategy
from rag_benchmark_lab.utils.auth import get_current_user_name, require_auth
from rag_benchmark_lab.utils.ui import (
    apply_plotly_theme,
    format_latency,
    format_metric,
    get_current_theme,
    inject_custom_css,
    render_html,
    render_strategy_badge,
    render_topnav,
)

st.set_page_config(
    page_title="Compare RAG — RAG Benchmark Lab",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_topnav(active_page="Compare")

user_id = require_auth()
context = get_context()
user_name = get_current_user_name() or "User"
current_theme = get_current_theme()

# ── Page Header ──────────────────────────────────────────────────────────────
header_col1, header_col2 = st.columns([4, 1.4], vertical_alignment="center")
with header_col1:
    render_html(
        """<div style="margin-top:20px;margin-bottom:16px;">
<div style="display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;color:var(--accent);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">
  <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--accent);"></span>
  Empirical Architecture Comparison
</div>
<h1 style="font-size:28px;font-weight:800;letter-spacing:-0.03em;color:var(--text);margin:0 0 4px 0;">
  Compare Retrieval Methods
</h1>
<p style="font-size:14px;color:var(--text-muted);margin:0;">
  Benchmark the exact same question side-by-side across Vector RAG, GraphRAG, and Hybrid RAG.
</p>
</div>"""
    )
with header_col2:
    if st.button("View Run History", icon=":material/history:", key="comp_goto_history_btn", use_container_width=True):
        st.switch_page("pages/6_History.py")

# ── Knowledge Base Validation ────────────────────────────────────────────────
kbs = context.user_registry.list_knowledge_bases(user_id)
if not kbs:
    st.warning("No knowledge bases found. Create a knowledge base and upload documents first.")
    if st.button("Go to Documents", key="comp_goto_docs"):
        st.switch_page("pages/2_Documents.py")
    st.stop()

kb_options = {kb.kb_id: kb.name for kb in kbs}
active_kb_id = st.session_state.get("active_kb_id")
if active_kb_id not in kb_options:
    active_kb_id = list(kb_options.keys())[0]
    st.session_state["active_kb_id"] = active_kb_id
    st.session_state["active_kb_name"] = kb_options[active_kb_id]

kb_col, spacer = st.columns([2.2, 2.8], vertical_alignment="center")
with kb_col:
    selected_kb = st.selectbox(
        "Active Knowledge Base",
        options=list(kb_options.keys()),
        format_func=lambda x: f"Knowledge Base: {kb_options[x]}",
        index=list(kb_options.keys()).index(active_kb_id),
        key="comp_kb_sel",
        label_visibility="collapsed",
    )
    if selected_kb != active_kb_id:
        st.session_state["active_kb_id"] = selected_kb
        st.session_state["active_kb_name"] = kb_options[selected_kb]
        st.session_state.pop("comp_active_run_id", None)
        st.rerun()
    active_kb_id = selected_kb
    active_kb_name = kb_options[active_kb_id]

kb_doc_ids = context.user_registry.get_kb_document_ids(active_kb_id, user_id)
if not kb_doc_ids:
    render_html(
        f"""<div class="rag-card" style="text-align:center;padding:36px 20px;">
          <h4 style="font-size:17px;font-weight:700;color:var(--text);margin:0 0 6px 0;">No Documents in '{active_kb_name}'</h4>
          <p style="font-size:13.5px;color:var(--text-muted);margin:0 auto 16px;">Upload documents before running side-by-side RAG comparisons.</p>
        </div>"""
    )
    if st.button("Upload Documents", type="primary", key="comp_upload_btn"):
        st.switch_page("pages/2_Documents.py")
    st.stop()

st.caption(f"Active Knowledge Base: **{active_kb_name}** ({len(kb_doc_ids)} documents available)")
st.markdown("<hr style='border-color:var(--border);margin:14px 0;'/>", unsafe_allow_html=True)

# ── Main Tabs: Compare vs Recent History ─────────────────────────────────────
tab_compare, tab_recent = st.tabs(["Compare Methods", "Recent Comparisons"])

with tab_compare:
    # ── Query Input Area ─────────────────────────────────────────────────────
    render_html(
        """<div style="font-size:15px;font-weight:700;color:var(--text);margin-bottom:8px;">
          Enter Question for Multi-Method Benchmark
        </div>"""
    )

    SAMPLE_QUERIES = [
        "What are the key technical findings and architectural principles?",
        "How do the components and entities connect across documents?",
        "Summarize the performance trade-offs between speed and accuracy.",
    ]
    sq_cols = st.columns(len(SAMPLE_QUERIES))
    for idx, sq in enumerate(SAMPLE_QUERIES):
        with sq_cols[idx]:
            if st.button(f"💡 {sq[:40]}…", key=f"btn_sq_{idx}", use_container_width=True):
                st.session_state["comp_query_input"] = sq

    question_text = st.text_input(
        "Question",
        value=st.session_state.get("comp_query_input", ""),
        placeholder="Type a research question to evaluate across Vector, Graph, and Hybrid RAG...",
        key="comp_input_field",
        label_visibility="collapsed",
    )

    btn_col, _ = st.columns([2, 4])
    with btn_col:
        run_submitted = st.button("Run 3-Way Comparison", type="primary", icon=":material/play_arrow:", use_container_width=True, key="btn_run_compare")

    if run_submitted and not question_text.strip():
        st.warning("Please enter a question to compare.")

    if run_submitted and question_text.strip():
        now_iso = datetime.now(timezone.utc).isoformat()
        run_id = f"cmp-{uuid.uuid4().hex[:10]}"

        query_strats = [
            (RetrievalStrategy.VECTOR, "Vector RAG"),
            (RetrievalStrategy.GRAPH, "GraphRAG"),
            (RetrievalStrategy.HYBRID, "Hybrid RAG"),
        ]

        strat_records: list[ComparisonStrategyResultRecord] = []

        progress_bar = st.progress(0.0)
        status_box = st.empty()

        for idx, (strat, strat_label) in enumerate(query_strats):
            status_box.info(f"Executing {strat_label} retrieval and generation...")
            res = context.query_service.answer(
                question=question_text.strip(),
                manual_strategy=strat,
                document_ids=kb_doc_ids,
                top_k=context.settings.default_top_k,
            )
            progress_bar.progress((idx + 1) / len(query_strats))

            res_id = f"res-{uuid.uuid4().hex[:8]}"
            strat_key = strat.value

            is_failed = bool(res.errors)
            answer_save = res.answer
            if is_failed and not answer_save:
                answer_save = "Generation unavailable due to error."

            citations_save = [
                {
                    "document_id": c.document_id,
                    "document_name": c.document_name,
                    "chunk_id": c.chunk_id,
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number,
                    "score": float(c.score) if c.score is not None else None,
                }
                for c in res.citations
            ]

            evidence_save = [
                {
                    "chunk_id": ch.chunk_id,
                    "document_id": ch.document_id,
                    "document_name": ch.document_name,
                    "chunk_index": ch.chunk_index,
                    "page_number": ch.page_number,
                    "score": float(ch.score) if ch.score is not None else None,
                    "text": ch.text,
                }
                for ch in res.retrieved_sources
            ]

            graph_ev_save = None
            if res.graph_evidence:
                graph_ev_save = {
                    "entities": [
                        {"entity_id": e.entity_id, "canonical_name": e.canonical_name, "entity_type": e.entity_type}
                        for e in res.graph_evidence.entities[:12]
                    ],
                    "relationships": [
                        {
                            "source_entity_name": r.source_entity_name,
                            "relationship_type": r.relationship_type,
                            "target_entity_name": r.target_entity_name,
                            "confidence": r.confidence,
                            "source_text": getattr(r, "source_text", None),
                        }
                        for r in res.graph_evidence.relationships[:10]
                    ],
                }

            strat_records.append(
                ComparisonStrategyResultRecord(
                    result_id=res_id,
                    run_id=run_id,
                    strategy=strat_key,
                    strategy_label=strat_label,
                    status="failed" if is_failed else "success",
                    answer=answer_save,
                    error_message="; ".join(res.errors) if is_failed else None,
                    retrieval_latency_ms=res.retrieval_latency_ms,
                    generation_latency_ms=res.generation_latency_ms,
                    total_latency_ms=res.total_latency_ms,
                    prompt_tokens=res.prompt_tokens,
                    completion_tokens=res.completion_tokens,
                    model_name=res.model_name,
                    citations=citations_save,
                    evidence_sources=evidence_save,
                    graph_evidence=graph_ev_save,
                    warnings=res.warnings or [],
                    errors=res.errors or [],
                    metadata=res.metadata or {},
                    created_at=now_iso,
                )
            )

        status_box.empty()
        progress_bar.empty()

        failed_count = sum(1 for s in strat_records if s.status == "failed")
        if failed_count == len(strat_records):
            overall_status = "failed"
        elif failed_count > 0:
            overall_status = "partial"
        else:
            overall_status = "completed"

        run_record = ComparisonRunRecord(
            run_id=run_id,
            user_id=user_id,
            kb_id=active_kb_id,
            query=question_text.strip(),
            status=overall_status,
            created_at=now_iso,
            updated_at=now_iso,
            config={"default_top_k": context.settings.default_top_k},
            strategies=[s.strategy for s in strat_records],
            results=strat_records,
            kb_name=active_kb_name,
        )

        context.user_registry.save_comparison_run(run_record)
        st.session_state["comp_active_run_id"] = run_id
        st.rerun()

    # ── Render Active Comparison Results ─────────────────────────────────────
    active_run_id = st.session_state.get("comp_active_run_id")
    if active_run_id:
        active_run = context.user_registry.get_comparison_run(active_run_id, user_id=user_id)
        if active_run:
            st.markdown("<hr style='border-color:var(--border);margin:24px 0;'/>", unsafe_allow_html=True)
            render_html(
                f"""<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;flex-wrap:wrap;gap:8px;">
                  <div>
                    <span style="font-size:11px;font-weight:700;color:var(--accent);text-transform:uppercase;letter-spacing:0.05em;">Comparison Evaluation</span>
                    <h3 style="margin:2px 0 0 0;font-size:20px;font-weight:800;color:var(--text);">{active_run.query}</h3>
                  </div>
                  <span style="font-size:12px;color:var(--text-muted);">Run ID: <code style="background:var(--card-hi);padding:2px 6px;border-radius:4px;">{active_run.run_id}</code></span>
                </div>"""
            )

            # Error banners if common quota / auth issues
            def _has_quota_err(errors_list: list[str] | None, err_msg: str | None = None) -> bool:
                combined = " ".join(errors_list or []) + " " + (err_msg or "")
                combined_lower = combined.lower()
                return "quota" in combined_lower or "resource_exhausted" in combined_lower or "429" in combined

            def _has_auth_err(errors_list: list[str] | None, err_msg: str | None = None) -> bool:
                combined = " ".join(errors_list or []) + " " + (err_msg or "")
                combined_lower = combined.lower()
                return "not configured" in combined_lower or "api_key" in combined_lower or "authentication" in combined_lower

            all_quota_exhausted = (
                len(active_run.results) == 3
                and all(_has_quota_err(sr.errors, sr.error_message) for sr in active_run.results)
            )
            all_auth_missing = (
                len(active_run.results) == 3
                and all(_has_auth_err(sr.errors, sr.error_message) for sr in active_run.results)
            )

            if all_quota_exhausted:
                st.warning(
                    "⚠️ **Gemini Quota Exceeded**: Available quota has been reached for the configured model. "
                    "Evidence retrieval succeeded for all strategies below."
                )
            elif all_auth_missing:
                st.warning(
                    "⚠️ **Gemini API Key Not Configured**: Please configure `GEMINI_API_KEY` in your `.env` file. "
                    "Evidence retrieval succeeded for all strategies below."
                )

            # ── 3-Way Side-by-Side Cards ─────────────────────────────────────
            PANELS = [
                (RetrievalStrategy.VECTOR.value, "Vector RAG", "var(--vector)", "Semantic embeddings via Qdrant"),
                (RetrievalStrategy.GRAPH.value, "GraphRAG", "var(--graph)", "Entity-relationship paths via Neo4j"),
                (RetrievalStrategy.HYBRID.value, "Hybrid RAG", "var(--hybrid)", "Reciprocal Rank Fusion (Vector + Graph)"),
            ]

            panel_cols = st.columns(3)
            strat_map = {sr.strategy: sr for sr in active_run.results}
            summary_rows = []

            chart_latencies = []
            chart_chunks = []

            for col, (strat_key, strat_label, accent_color, strat_sub) in zip(panel_cols, PANELS):
                sr = strat_map.get(strat_key)
                with col:
                    with st.container(border=True):
                        badge_html = render_strategy_badge(strat_key)
                        render_html(
                            f"""
                            <div style="border-top:3px solid {accent_color};padding-top:8px;margin-bottom:12px;">
                              <div style="display:flex;justify-content:space-between;align-items:center;">
                                <span style="font-size:16px;font-weight:800;color:var(--text);">{strat_label}</span>
                                {badge_html}
                              </div>
                              <div style="font-size:11px;color:var(--text-muted);margin-top:2px;">{strat_sub}</div>
                            </div>
                            """
                        )

                        if sr is None:
                            st.error("Strategy execution failed.")
                            continue

                        ev_count = len(sr.evidence_sources)
                        cit_count = len(sr.citations)
                        has_graph = sr.graph_evidence and sr.graph_evidence.get("relationships")
                        graph_paths_count = len(sr.graph_evidence.get("relationships", [])) if has_graph else 0

                        # Status line
                        if ev_count > 0 or graph_paths_count > 0:
                            ret_status_label = f"Evidence Found ({ev_count} passages)"
                            ret_badge_cls = "badge-success"
                        else:
                            ret_status_label = "No Evidence Found"
                            ret_badge_cls = "badge-warning"

                        is_quota = _has_quota_err(sr.errors, sr.error_message)
                        is_auth = _has_auth_err(sr.errors, sr.error_message)

                        if sr.status == "success" and not sr.errors:
                            gen_status_label = "Generated"
                            gen_badge_cls = "badge-success"
                        elif is_quota:
                            gen_status_label = "Quota Exceeded"
                            gen_badge_cls = "badge-error"
                        elif is_auth:
                            gen_status_label = "Key Missing"
                            gen_badge_cls = "badge-error"
                        else:
                            gen_status_label = "Failed"
                            gen_badge_cls = "badge-error"

                        render_html(
                            f"""
                            <div style="display:flex;flex-direction:column;gap:6px;margin-bottom:12px;">
                              <div style="display:flex;justify-content:space-between;align-items:center;background:var(--card-hi);border:1px solid var(--border);border-radius:6px;padding:5px 10px;font-size:12px;">
                                <span style="color:var(--text-muted);">Retrieval:</span>
                                <span class="badge {ret_badge_cls}" style="font-size:10.5px;">{ret_status_label}</span>
                              </div>
                              <div style="display:flex;justify-content:space-between;align-items:center;background:var(--card-hi);border:1px solid var(--border);border-radius:6px;padding:5px 10px;font-size:12px;">
                                <span style="color:var(--text-muted);">Generation:</span>
                                <span class="badge {gen_badge_cls}" style="font-size:10.5px;">{gen_status_label}</span>
                              </div>
                            </div>
                            """
                        )

                        # Answer container
                        answer_display = sr.answer
                        if is_quota:
                            answer_display = "Gemini quota exhausted for configured model. Retrieved evidence is intact below."
                        elif is_auth:
                            answer_display = "Gemini API key is not configured. Retrieved evidence is intact below."

                        render_html(
                            f"""
                            <div style="background:var(--card-hi);border:1px solid var(--border);border-radius:8px;padding:14px 16px;font-size:13.5px;line-height:1.6;color:var(--text);min-height:110px;max-height:220px;overflow-y:auto;margin-bottom:12px;">
                              {answer_display}
                            </div>
                            """
                        )

                        # Telemetry breakdown
                        m1, m2 = st.columns(2)
                        with m1:
                            ret_ms = format_latency(sr.retrieval_latency_ms, fallback="Unavailable")
                            st.caption(f"Retrieval: **{ret_ms}**")
                            gen_ms = (
                                format_latency(sr.generation_latency_ms, fallback="Unavailable", treat_zero_as_unavailable=True)
                                if (sr.status == "success" and not sr.errors)
                                else "Unavailable"
                            )
                            st.caption(f"Generation: **{gen_ms}**")
                            tot_ms = (
                                format_latency(sr.total_latency_ms, fallback="Unavailable", treat_zero_as_unavailable=True)
                                if (sr.status == "success" and not sr.errors)
                                else format_latency(sr.retrieval_latency_ms, fallback="Unavailable")
                            )
                            st.caption(f"Total: **{tot_ms}**")
                        with m2:
                            st.caption(f"Evidence: **{ev_count}** passages")
                            st.caption(f"Citations: **{cit_count}** chunks")
                            if strat_key == RetrievalStrategy.VECTOR.value:
                                st.caption("Graph: *N/A (Vector)*")
                            else:
                                st.caption(f"Graph Paths: **{graph_paths_count}**")

                        # Collect data for charts
                        ret_num = sr.retrieval_latency_ms if sr.retrieval_latency_ms else 0
                        gen_num = sr.generation_latency_ms if (sr.generation_latency_ms and sr.status == "success") else 0
                        chart_latencies.append({
                            "Strategy": strat_label,
                            "Retrieval (ms)": ret_num,
                            "Generation (ms)": gen_num,
                            "Total (ms)": ret_num + gen_num,
                        })
                        chart_chunks.append({
                            "Strategy": strat_label,
                            "Passages": ev_count,
                            "Citations": cit_count,
                            "Graph Paths": graph_paths_count,
                        })

                    # Evidence expander
                    if sr.citations or (sr.graph_evidence and sr.graph_evidence.get("relationships")):
                        exp_title = f"Evidence & Citations ({len(sr.citations)} citations" + (f", {graph_paths_count} graph paths" if graph_paths_count else "") + ")"
                        with st.expander(exp_title):
                            if sr.citations:
                                st.markdown("**Document Citations:**")
                                for c in sr.citations[:6]:
                                    score_val = c.get("score")
                                    score_str = f" · Score: {format_metric(score_val, fallback='', precision=3)}" if score_val is not None else ""
                                    st.caption(f"• **{c.get('document_name','Doc')}** [Chunk #{c.get('chunk_index',0)}]{score_str}")
                            if sr.evidence_sources:
                                st.markdown("**Retrieved Passages:**")
                                for p in sr.evidence_sources[:3]:
                                    st.text(p.get("text", "")[:180] + ("…" if len(p.get("text","")) > 180 else ""))
                            if sr.graph_evidence and sr.graph_evidence.get("relationships"):
                                st.markdown("**Graph Paths:**")
                                for r in sr.graph_evidence.get("relationships", [])[:5]:
                                    st.caption(f"• {r.get('source_entity_name')} → {r.get('relationship_type')} → {r.get('target_entity_name')}")

                    summary_rows.append(
                        {
                            "Strategy": strat_label,
                            "Mechanism": "Dense Vector Embeddings" if "Vector" in strat_label else ("Entity-Relation Traversal" if "Graph" in strat_label else "Reciprocal Rank Fusion"),
                            "Evidence Count": ev_count,
                            "Citations": cit_count,
                            "Graph Paths": str(graph_paths_count) if strat_key != RetrievalStrategy.VECTOR.value else "—",
                            "Retrieval (ms)": format_latency(sr.retrieval_latency_ms, unit="", fallback="N/A"),
                            "Generation (ms)": (
                                format_latency(sr.generation_latency_ms, unit="", fallback="N/A", treat_zero_as_unavailable=True)
                                if (sr.status == "success" and not sr.errors)
                                else "N/A"
                            ),
                            "Total Latency (ms)": (
                                format_latency(sr.total_latency_ms, unit="", fallback="N/A", treat_zero_as_unavailable=True)
                                if (sr.status == "success" and not sr.errors)
                                else format_latency(sr.retrieval_latency_ms, unit="", fallback="N/A")
                            ),
                            "Status": gen_status_label,
                        }
                    )

            # ── Interactive Comparison Charts ────────────────────────────────
            st.markdown("<hr style='border-color:var(--border);margin:24px 0;'/>", unsafe_allow_html=True)
            render_html(
                """<div style="margin-bottom:12px;">
                  <h3 style="font-size:18px;font-weight:800;color:var(--text);margin:0 0 4px 0;">Approach Performance Telemetry</h3>
                  <p style="font-size:13px;color:var(--text-muted);margin:0;">Visual comparison of latency breakdown and evidence retrieval volume.</p>
                </div>"""
            )

            ch1, ch2 = st.columns(2)
            with ch1:
                if chart_latencies:
                    df_lat = pd.DataFrame(chart_latencies)
                    fig_lat = px.bar(
                        df_lat,
                        x="Strategy",
                        y=["Retrieval (ms)", "Generation (ms)"],
                        barmode="group",
                        title="Latency by Strategy (ms)",
                        color_discrete_map={
                            "Retrieval (ms)": "#2DD4BF" if current_theme == "dark" else "#0D9488",
                            "Generation (ms)": "#818CF8" if current_theme == "dark" else "#4F46E5",
                        },
                        height=330,
                    )
                    apply_plotly_theme(fig_lat, current_theme)
                    st.plotly_chart(fig_lat, use_container_width=True)

            with ch2:
                if chart_chunks:
                    df_ch = pd.DataFrame(chart_chunks)
                    fig_chunks = px.bar(
                        df_ch,
                        x="Strategy",
                        y=["Passages", "Citations", "Graph Paths"],
                        barmode="group",
                        title="Evidence & Citations Volume",
                        color_discrete_map={
                            "Passages": "#60A5FA" if current_theme == "dark" else "#2563EB",
                            "Citations": "#A78BFA" if current_theme == "dark" else "#7C3AED",
                            "Graph Paths": "#34D399" if current_theme == "dark" else "#059669",
                        },
                        height=330,
                    )
                    apply_plotly_theme(fig_chunks, current_theme)
                    st.plotly_chart(fig_chunks, use_container_width=True)

            # ── Comparison Matrix Table ──────────────────────────────────────
            st.markdown("<hr style='border-color:var(--border);margin:20px 0;'/>", unsafe_allow_html=True)
            render_html(
                """<div style="margin-bottom:12px;">
                  <h3 style="font-size:18px;font-weight:800;color:var(--text);margin:0 0 4px 0;">Strategy Comparison Matrix</h3>
                  <p style="font-size:13px;color:var(--text-muted);margin:0;">Side-by-side architectural and performance matrix for this query.</p>
                </div>"""
            )

            if summary_rows:
                df_summary = pd.DataFrame(summary_rows)
                st.dataframe(df_summary, use_container_width=True, hide_index=True)

with tab_recent:
    render_html(
        f"""<div style="margin-bottom:16px;">
          <h3 style="font-size:18px;font-weight:800;color:var(--text);margin:0 0 4px 0;">Recent Comparisons</h3>
          <p style="font-size:13px;color:var(--text-muted);margin:0;">Stored benchmark runs in <strong>{active_kb_name}</strong>.</p>
        </div>"""
    )
    recent_runs = context.user_registry.list_comparison_runs(user_id=user_id, kb_id=active_kb_id, limit=25)

    if not recent_runs:
        st.info("No saved comparisons for this knowledge base yet. Run a comparison to view historical logs here.")
    else:
        for r_item in recent_runs:
            with st.container(border=True):
                c_q, c_stat, c_act = st.columns([5, 2, 2], vertical_alignment="center")
                with c_q:
                    st.markdown(f"**{r_item.query}**")
                    st.caption(f"📅 {r_item.created_at[:19].replace('T', ' ')} UTC &nbsp;·&nbsp; ID: `{r_item.run_id}`")
                with c_stat:
                    s_color = "var(--success)" if r_item.status == "completed" else ("var(--warning)" if r_item.status == "partial" else "var(--error)")
                    passed_strats = sum(1 for s in r_item.results if s.status == "success")
                    render_html(
                        f"""<span style="color:{s_color};font-weight:700;font-size:12px;">{r_item.status.upper()}</span>
                        <div style="font-size:11.5px;color:var(--text-muted);">{passed_strats}/{len(r_item.results)} strategies passed</div>"""
                    )
                with c_act:
                    c_load, c_del2 = st.columns(2)
                    with c_load:
                        if st.button("Inspect", key=f"rec_load_{r_item.run_id}", use_container_width=True):
                            st.session_state["comp_active_run_id"] = r_item.run_id
                            st.rerun()
                    with c_del2:
                        if st.button("Delete", key=f"rec_del_{r_item.run_id}", use_container_width=True):
                            context.user_registry.delete_comparison_run(r_item.run_id, user_id=user_id)
                            if st.session_state.get("comp_active_run_id") == r_item.run_id:
                                st.session_state.pop("comp_active_run_id", None)
                            st.rerun()
