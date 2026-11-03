"""RAG Benchmark Lab — Comparison History & Analytics Page."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rag_benchmark_lab.app import get_context
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
    page_title="Comparison History & Analytics — RAG Benchmark Lab",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_topnav(active_page="History")

user_id = require_auth()
context = get_context()
user_name = get_current_user_name() or "User"
current_theme = get_current_theme()

# ── Page Header ──────────────────────────────────────────────────────────────
render_html(
    """<div style="margin-top:20px;margin-bottom:18px;">
<div style="display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;color:var(--accent);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">
  <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--accent);"></span>
  Audit &amp; Telemetry Analytics
</div>
<h1 style="font-size:28px;font-weight:800;letter-spacing:-0.03em;color:var(--text);margin:0 0 4px 0;">
  Comparison Run History &amp; Analytics
</h1>
<p style="font-size:14px;color:var(--text-muted);margin:0;">
  Persistent audit trail of side-by-side RAG evaluations. Inspect historical latency trends, strategy win-rates, and multi-hop citations without re-running LLMs.
</p>
</div>"""
)

# ── Knowledge Base Resolution ────────────────────────────────────────────────
kbs = context.user_registry.list_knowledge_bases(user_id)
kb_options = {kb.kb_id: kb.name for kb in kbs}

# ── Filters Bar ──────────────────────────────────────────────────────────────
with st.container(border=True):
    f_col1, f_col2, f_col3, f_col4 = st.columns([2.5, 2, 2, 2.5], vertical_alignment="center")

    with f_col1:
        kb_filter_choices = ["All Knowledge Bases"] + [f"{name} ({kbid[:6]}…)" for kbid, name in kb_options.items()]
        kb_filter_sel = st.selectbox(
            "Filter by Knowledge Base",
            options=kb_filter_choices,
            index=0,
            key="hist_filter_kb",
        )
        selected_filter_kb_id = None
        if kb_filter_sel != "All Knowledge Bases":
            for kbid, name in kb_options.items():
                if kb_filter_sel.startswith(f"{name} ("):
                    selected_filter_kb_id = kbid
                    break

    with f_col2:
        status_filter = st.selectbox(
            "Filter by Status",
            options=["All Statuses", "Completed", "Partial", "Failed"],
            index=0,
            key="hist_filter_status",
        )
        selected_status = None if status_filter == "All Statuses" else status_filter.lower()

    with f_col3:
        strategy_filter = st.selectbox(
            "Filter by Strategy",
            options=["All Methods", "VECTOR", "GRAPH", "HYBRID"],
            index=0,
            key="hist_filter_strat",
        )
        selected_strategy = None if strategy_filter == "All Methods" else strategy_filter

    with f_col4:
        query_search = st.text_input(
            "Search Query Text",
            placeholder="Search questions...",
            key="hist_filter_search",
        )

# ── Fetch History from SQLite ────────────────────────────────────────────────
runs = context.user_registry.list_comparison_runs(
    user_id=user_id,
    kb_id=selected_filter_kb_id,
    status=selected_status,
    strategy=selected_strategy,
    limit=200,
)

if query_search.strip():
    runs = [r for r in runs if query_search.strip().lower() in r.query.lower()]

# ── Metric Summary Cards ─────────────────────────────────────────────────────
total_runs_cnt = len(runs)
completed_cnt = sum(1 for r in runs if r.status == "completed")
all_lats = []
for r in runs:
    for s in r.results:
        if s.total_latency_ms and s.total_latency_ms > 0:
            all_lats.append(s.total_latency_ms)

avg_lat_str = f"{sum(all_lats)/len(all_lats):.0f} ms" if all_lats else "N/A"
completion_rate = f"{(completed_cnt / total_runs_cnt * 100):.0f}%" if total_runs_cnt else "N/A"

m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
    render_html(
        f"""<div class="stat-card">
          <div class="stat-value">{total_runs_cnt}</div>
          <div class="stat-label">Comparison Runs</div>
        </div>"""
    )
with m_col2:
    render_html(
        f"""<div class="stat-card">
          <div class="stat-value" style="color:var(--success);">{completed_cnt}</div>
          <div class="stat-label">Successful Runs</div>
        </div>"""
    )
with m_col3:
    render_html(
        f"""<div class="stat-card">
          <div class="stat-value" style="color:var(--accent);">{completion_rate}</div>
          <div class="stat-label">Completion Rate</div>
        </div>"""
    )
with m_col4:
    render_html(
        f"""<div class="stat-card">
          <div class="stat-value" style="color:var(--vector);">{avg_lat_str}</div>
          <div class="stat-label">Avg Total Latency</div>
        </div>"""
    )

render_html("<div style='height:24px;'></div>")

# ── Analytics Visualizations ─────────────────────────────────────────────────
if runs:
    with st.expander("Historical Analytics Visualizations", expanded=True):
        ch_tab1, ch_tab2 = st.columns(2)

        # 1. Historical Latency Trends (Line chart)
        time_series_rows = []
        strat_lat_acc: dict[str, list[float]] = {"Vector RAG": [], "GraphRAG": [], "Hybrid RAG": []}

        for r in reversed(runs[:30]):
            ts = r.created_at[:16].replace("T", " ")
            for s in r.results:
                if s.total_latency_ms and s.total_latency_ms > 0:
                    lbl = s.strategy_label or s.strategy
                    time_series_rows.append({
                        "Timestamp": ts,
                        "Strategy": lbl,
                        "Latency (ms)": s.total_latency_ms,
                    })
                    if lbl in strat_lat_acc:
                        strat_lat_acc[lbl].append(s.total_latency_ms)

        with ch_tab1:
            if time_series_rows:
                df_ts = pd.DataFrame(time_series_rows)
                fig_ts = px.line(
                    df_ts,
                    x="Timestamp",
                    y="Latency (ms)",
                    color="Strategy",
                    markers=True,
                    title="Historical Latency Trend Across Runs (ms)",
                    color_discrete_map={
                        "Vector RAG": "#60A5FA" if current_theme == "dark" else "#2563EB",
                        "GraphRAG": "#A78BFA" if current_theme == "dark" else "#7C3AED",
                        "Hybrid RAG": "#34D399" if current_theme == "dark" else "#059669",
                    },
                    height=320,
                )
                apply_plotly_theme(fig_ts, current_theme)
                st.plotly_chart(fig_ts, use_container_width=True)

        with ch_tab2:
            # 2. Average Latency by Strategy (Bar chart)
            bar_rows = []
            for strat_name, lats in strat_lat_acc.items():
                if lats:
                    bar_rows.append({
                        "Strategy": strat_name,
                        "Mean Latency (ms)": sum(lats) / len(lats),
                    })

            if bar_rows:
                df_bar = pd.DataFrame(bar_rows)
                fig_bar = px.bar(
                    df_bar,
                    x="Strategy",
                    y="Mean Latency (ms)",
                    color="Strategy",
                    title="Mean Total Latency by Strategy (ms)",
                    color_discrete_map={
                        "Vector RAG": "#60A5FA" if current_theme == "dark" else "#2563EB",
                        "GraphRAG": "#A78BFA" if current_theme == "dark" else "#7C3AED",
                        "Hybrid RAG": "#34D399" if current_theme == "dark" else "#059669",
                    },
                    height=320,
                )
                apply_plotly_theme(fig_bar, current_theme)
                st.plotly_chart(fig_bar, use_container_width=True)

# ── Inspected Run State ──────────────────────────────────────────────────────
inspected_run_id = st.session_state.get("hist_inspect_run_id")

if inspected_run_id:
    inspected_run = context.user_registry.get_comparison_run(inspected_run_id, user_id=user_id)
    if not inspected_run:
        st.session_state.pop("hist_inspect_run_id", None)
        st.rerun()

    st.markdown("<hr style='border-color:var(--border);margin:20px 0;'/>", unsafe_allow_html=True)

    c_back, c_title, c_del = st.columns([1.5, 6, 1.5], vertical_alignment="center")
    with c_back:
        if st.button("← Back to List", icon=":material/arrow_back:", key="btn_close_inspect", use_container_width=True):
            st.session_state.pop("hist_inspect_run_id", None)
            st.rerun()
    with c_title:
        created_display = inspected_run.created_at[:19].replace("T", " ")
        render_html(
            f"""<div style="font-size:14px;color:var(--text-muted);">
            Inspecting Run: <strong style="color:var(--accent);">{inspected_run.run_id}</strong> · {created_display} UTC · Knowledge Base: <strong>{inspected_run.kb_name or inspected_run.kb_id}</strong>
            </div>"""
        )
    with c_del:
        if st.button("Delete Run", icon=":material/delete:", key=f"btn_del_inspect_{inspected_run.run_id}", type="secondary", use_container_width=True):
            context.user_registry.delete_comparison_run(inspected_run.run_id, user_id=user_id)
            st.session_state.pop("hist_inspect_run_id", None)
            st.success("Comparison run deleted.")
            st.rerun()

    # Query callout
    render_html(
        f"""
        <div class="rag-card" style="border-left:4px solid var(--accent);padding:16px 20px;margin:14px 0 20px;">
          <div style="font-size:11px;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">Evaluated Question</div>
          <div style="font-size:16px;color:var(--text);font-weight:700;">{inspected_run.query}</div>
        </div>
        """
    )

    PANELS = [
        (RetrievalStrategy.VECTOR.value, "Vector RAG", "var(--vector)"),
        (RetrievalStrategy.GRAPH.value, "GraphRAG", "var(--graph)"),
        (RetrievalStrategy.HYBRID.value, "Hybrid RAG", "var(--hybrid)"),
    ]

    panel_cols = st.columns(3)
    strat_results_map = {sr.strategy: sr for sr in inspected_run.results}
    summary_rows = []

    for col, (strat_key, strat_label, accent_color) in zip(panel_cols, PANELS):
        sr = strat_results_map.get(strat_key)
        with col:
            with st.container(border=True):
                badge_html = render_strategy_badge(strat_key)
                render_html(
                    f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;border-top:3px solid {accent_color};padding-top:6px;">
                      <span style="font-size:15px;font-weight:700;color:var(--text);">{strat_label}</span>
                      {badge_html}
                    </div>
                    """
                )

                if sr is None:
                    st.error("Strategy was not executed in this run.")
                    continue

                if sr.status == "failed" or sr.error_message:
                    st.error(f"Error: {sr.error_message or 'Execution failed'}")

                render_html(
                    f"""
                    <div style="background:var(--card-hi);border:1px solid var(--border);border-radius:8px;padding:12px 14px;font-size:13.5px;line-height:1.6;color:var(--text);min-height:110px;max-height:220px;overflow-y:auto;">
                      {sr.answer}
                    </div>
                    """
                )

                st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                with m1:
                    ret_ms = format_latency(sr.retrieval_latency_ms, fallback="Unavailable")
                    st.caption(f"Retrieval: **{ret_ms}**")
                    gen_ms = (
                        format_latency(sr.generation_latency_ms, fallback="Unavailable", treat_zero_as_unavailable=True)
                        if (sr.status == "success" and not sr.error_message)
                        else "Unavailable"
                    )
                    st.caption(f"Generation: **{gen_ms}**")
                with m2:
                    tot_ms = (
                        format_latency(sr.total_latency_ms, fallback="Unavailable", treat_zero_as_unavailable=True)
                        if (sr.status == "success" and not sr.error_message)
                        else format_latency(sr.retrieval_latency_ms, fallback="Unavailable")
                    )
                    st.caption(f"Total: **{tot_ms}**")
                    st.caption(f"Evidence: **{len(sr.evidence_sources)}** passages")

            if sr.citations or (sr.graph_evidence and sr.graph_evidence.get("relationships")):
                with st.expander(f"View Evidence ({len(sr.citations)} citations)"):
                    if sr.citations:
                        st.markdown("**Document Citations:**")
                        for c in sr.citations[:6]:
                            score_val = c.get("score")
                            score_str = f" · Score: {format_metric(score_val, fallback='', precision=3)}" if score_val is not None else ""
                            st.caption(f"• **{c.get('document_name','Doc')}** [Chunk #{c.get('chunk_index',0)}]{score_str}")
                    if sr.evidence_sources:
                        st.markdown("**Retrieved Passages:**")
                        for p in sr.evidence_sources[:3]:
                            st.text(p.get("text", "")[:200] + ("…" if len(p.get("text","")) > 200 else ""))
                    if sr.graph_evidence and sr.graph_evidence.get("relationships"):
                        st.markdown("**Graph Paths:**")
                        for r in sr.graph_evidence.get("relationships", [])[:5]:
                            st.caption(f"• {r.get('source_entity_name')} → {r.get('relationship_type')} → {r.get('target_entity_name')}")

            summary_rows.append(
                {
                    "Strategy": strat_label,
                    "Status": sr.status.capitalize(),
                    "Evidence Count": len(sr.evidence_sources),
                    "Citations": len(sr.citations),
                    "Graph Paths": len(sr.graph_evidence.get("relationships", [])) if sr.graph_evidence else 0,
                    "Retrieval (ms)": format_latency(sr.retrieval_latency_ms, unit="", fallback="Unavailable"),
                    "Generation (ms)": (
                        format_latency(sr.generation_latency_ms, unit="", fallback="Unavailable", treat_zero_as_unavailable=True)
                        if (sr.status == "success" and not sr.error_message)
                        else "Unavailable"
                    ),
                    "Total Latency (ms)": (
                        format_latency(sr.total_latency_ms, unit="", fallback="Unavailable", treat_zero_as_unavailable=True)
                        if (sr.status == "success" and not sr.error_message)
                        else format_latency(sr.retrieval_latency_ms, unit="", fallback="Unavailable")
                    ),
                }
            )

    st.markdown("### Comparison Telemetry Summary")
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.markdown("<hr style='border-color:var(--border);margin:24px 0;'/>", unsafe_allow_html=True)

# ── Comparison Runs List ─────────────────────────────────────────────────────
render_html(
    f"""<div style="font-size:18px;font-weight:800;color:var(--text);margin-bottom:14px;">
      Logged Benchmark Runs ({len(runs)} found)
    </div>"""
)

if not runs:
    st.info("No comparison runs found matching the selected filters. Execute a comparison in the Compare page to start building your audit history.")
    if st.button("Go to Compare", key="hist_goto_comp"):
        st.switch_page("pages/4_Compare.py")
else:
    for run in runs:
        with st.container(border=True):
            r_top1, r_top2, r_top3, r_top4 = st.columns([4, 2, 2, 2], vertical_alignment="center")

            if run.status == "completed":
                status_color = "var(--success)"
                status_badge = "COMPLETED"
            elif run.status == "partial":
                status_color = "var(--warning)"
                status_badge = "PARTIAL"
            else:
                status_color = "var(--error)"
                status_badge = "FAILED"

            success_count = sum(1 for s in run.results if s.status == "success")
            total_strats = len(run.results)

            tot_lats = [s.total_latency_ms for s in run.results if s.total_latency_ms is not None and isinstance(s.total_latency_ms, (int, float)) and s.total_latency_ms > 0]
            avg_tot_lat = f"{sum(tot_lats)/len(tot_lats):.0f} ms" if tot_lats else "Unavailable"

            with r_top1:
                render_html(
                    f"""<div style="font-size:15px;font-weight:700;color:var(--text);margin-bottom:4px;">
                    {run.query}
                    </div>
                    <div style="font-size:12px;color:var(--text-muted);">
                    KB: <strong style="color:var(--text);">{run.kb_name or run.kb_id}</strong> &nbsp;·&nbsp; ID: <code>{run.run_id}</code>
                    </div>"""
                )

            with r_top2:
                created_dt = run.created_at[:19].replace("T", " ")
                st.caption(f"📅 **{created_dt}** UTC")
                st.caption(f"Methods: **{', '.join(run.strategies) or 'VECTOR, GRAPH, HYBRID'}**")

            with r_top3:
                render_html(
                    f"""
                    <div>
                      <span style="background:var(--card-hi);border:1px solid {status_color};color:{status_color};padding:3px 8px;border-radius:6px;font-size:11px;font-weight:700;">
                        {status_badge}
                      </span>
                      <div style="font-size:12px;color:var(--text-muted);margin-top:5px;">
                        Strategies: <strong>{success_count}/{total_strats}</strong> passed<br/>
                        Avg Total Latency: <strong>{avg_tot_lat}</strong>
                      </div>
                    </div>
                    """
                )

            with r_top4:
                col_btn_ins, col_btn_del = st.columns(2)
                with col_btn_ins:
                    if st.button("Inspect", key=f"inspect_{run.run_id}", use_container_width=True, type="primary"):
                        st.session_state["hist_inspect_run_id"] = run.run_id
                        st.rerun()
                with col_btn_del:
                    if st.button("Delete", key=f"del_{run.run_id}", use_container_width=True):
                        context.user_registry.delete_comparison_run(run.run_id, user_id=user_id)
                        if st.session_state.get("hist_inspect_run_id") == run.run_id:
                            st.session_state.pop("hist_inspect_run_id", None)
                        st.rerun()
