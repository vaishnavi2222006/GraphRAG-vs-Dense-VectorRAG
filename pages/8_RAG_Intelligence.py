"""RAG Benchmark Lab — RAG Intelligence & Query Analytics Page.

Empirical categorization of historical queries across Chat and Evaluation,
analyzing performance across Vector RAG, GraphRAG, and Hybrid RAG using a
4-pillar normalized weighted scoring methodology:
- Answer Quality (35%)
- Retrieval Quality (35%)
- Latency (15%)
- Cost (15%)

Strictly separates Actual Evaluation scores from Estimated/Provisional scores,
normalizes metrics dynamically without penalizing missing data, and provides
category-wise multi-dimensional recommendations and telemetry dashboards.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import statistics
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rag_benchmark_lab.analytics.intelligence_service import (
    ApproachRecord,
    CategoryIntelligence,
    IntelligenceService,
    IntelligenceSummary,
    QueryIntelligenceRecord,
    ScoringWeights,
)
from rag_benchmark_lab.analytics.query_classifier import (
    CATEGORY_COLOR_MAP,
    CATEGORY_DESCRIPTIONS,
    QUERY_CATEGORIES,
    QueryClassifier,
)
from rag_benchmark_lab.app import get_context
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
    page_title="RAG Intelligence & Query Analytics — RAG Benchmark Lab",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_topnav(active_page="Intelligence")

user_id = require_auth()
context = get_context()
user_name = get_current_user_name() or "User"
current_theme = get_current_theme()

# ── Color Constants ────────────────────────────────────────────────────────────
STRATEGY_COLORS = {
    "VECTOR": "#3B82F6",            # Blue
    "GRAPH": "#10B981",             # Emerald
    "HYBRID": "#8B5CF6",            # Purple
    "Tie": "#F59E0B",               # Amber
    "Insufficient data": "#64748B",  # Slate
}


def _render_approach_badge(strat: str, score: float | None = None) -> str:
    s = (strat or "").strip()
    score_txt = f" ({score:.3f})" if score is not None else ""
    if s == "VECTOR":
        return f'<span style="background:rgba(59,130,246,0.18);color:#60A5FA;border:1px solid rgba(59,130,246,0.35);padding:3px 8px;border-radius:4px;font-weight:600;font-size:11px;">Vector RAG{score_txt}</span>'
    elif s == "GRAPH":
        return f'<span style="background:rgba(16,185,129,0.18);color:#34D399;border:1px solid rgba(16,185,129,0.35);padding:3px 8px;border-radius:4px;font-weight:600;font-size:11px;">GraphRAG{score_txt}</span>'
    elif s == "HYBRID":
        return f'<span style="background:rgba(139,92,246,0.18);color:#A78BFA;border:1px solid rgba(139,92,246,0.35);padding:3px 8px;border-radius:4px;font-weight:600;font-size:11px;">Hybrid RAG{score_txt}</span>'
    elif s == "Tie":
        return f'<span style="background:rgba(245,158,11,0.18);color:#FBBF24;border:1px solid rgba(245,158,11,0.35);padding:3px 8px;border-radius:4px;font-weight:600;font-size:11px;">Tie / Parity{score_txt}</span>'
    else:
        return f'<span style="background:rgba(100,116,139,0.18);color:#94A3B8;border:1px solid rgba(100,116,139,0.35);padding:3px 8px;border-radius:4px;font-weight:600;font-size:11px;">Insufficient data</span>'


def _render_source_badge(source_label: str) -> str:
    if "Actual" in source_label:
        return '<span style="background:rgba(16,185,129,0.15);color:#34D399;border:1px solid rgba(16,185,129,0.3);padding:2px 6px;border-radius:4px;font-size:10px;font-weight:600;">Verified Evaluation</span>'
    elif "Estimated" in source_label or "Provisional" in source_label:
        return '<span style="background:rgba(245,158,11,0.15);color:#FBBF24;border:1px solid rgba(245,158,11,0.3);padding:2px 6px;border-radius:4px;font-size:10px;font-weight:600;">Estimated / Provisional</span>'
    else:
        return '<span style="background:rgba(100,116,139,0.15);color:#94A3B8;border:1px solid rgba(100,116,139,0.3);padding:2px 6px;border-radius:4px;font-size:10px;font-weight:600;">Telemetry Only</span>'


def _render_confidence_badge(level: str) -> str:
    if level == "High":
        return '<span style="background:rgba(16,185,129,0.18);color:#34D399;border:1px solid rgba(16,185,129,0.35);padding:2px 6px;border-radius:4px;font-size:10px;font-weight:700;">High Confidence</span>'
    elif level == "Medium":
        return '<span style="background:rgba(59,130,246,0.18);color:#60A5FA;border:1px solid rgba(59,130,246,0.35);padding:2px 6px;border-radius:4px;font-size:10px;font-weight:700;">Medium Confidence</span>'
    elif level == "Low":
        return '<span style="background:rgba(245,158,11,0.18);color:#FBBF24;border:1px solid rgba(245,158,11,0.35);padding:2px 6px;border-radius:4px;font-size:10px;font-weight:700;">Low Confidence</span>'
    else:
        return '<span style="background:rgba(100,116,139,0.18);color:#94A3B8;border:1px solid rgba(100,116,139,0.35);padding:2px 6px;border-radius:4px;font-size:10px;font-weight:700;">Insufficient Data</span>'


# ── Page Header ──────────────────────────────────────────────────────────────
render_html(
    """<div style="margin-top:20px;margin-bottom:18px;">
<div style="display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;color:var(--accent);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">
  <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--accent);"></span>
  Empirical Strategy Mapping &amp; Historical Telemetry
</div>
<h2 style="margin:0 0 6px 0;font-weight:700;font-size:26px;color:var(--text);letter-spacing:-0.02em;">RAG Intelligence &amp; Query Analytics</h2>
<p style="margin:0;font-size:13px;color:var(--text-muted);max-width:900px;line-height:1.5;">
Classify historical queries into 9 semantic/structural categories and evaluate architectural performance across Vector RAG, GraphRAG, and Hybrid RAG using a normalized multi-pillar scoring model. Actual evaluation metrics are strictly separated from provisional estimates.
</p>
</div>"""
)

# ── Knowledge Base Scope & Score Source Filter Controls ───────────────────────
c_filter_kb, c_score_mode, c_refresh = st.columns([2.5, 2.5, 1], vertical_alignment="bottom")

with c_filter_kb:
    user_kbs = context.user_registry.list_knowledge_bases(user_id) if hasattr(context, "user_registry") else []
    kb_options = {"all": "All Knowledge Bases"}
    for kb in user_kbs:
        kb_options[kb.kb_id] = kb.name

    selected_kb_key = st.selectbox(
        "Knowledge Base Scope",
        options=list(kb_options.keys()),
        format_func=lambda k: kb_options[k],
        index=0,
        help="Filter historical evaluation and comparison queries by knowledge base scope.",
    )
    filter_kb_id = None if selected_kb_key == "all" else selected_kb_key

with c_score_mode:
    score_mode_labels = {
        "all": "All Data (Actual + Provisional)",
        "actual_only": "Actual Evaluations Only (Verified Metrics)",
        "provisional_only": "Provisional Estimates Only",
    }
    selected_score_mode = st.selectbox(
        "Score Evaluation Mode",
        options=list(score_mode_labels.keys()),
        format_func=lambda k: score_mode_labels[k],
        index=0,
        help="Strictly separate actual measured evaluations from provisional comparison estimates.",
    )

with c_refresh:
    if st.button("Refresh Telemetry", icon=":material/refresh:", use_container_width=True):
        st.session_state.pop("cached_intelligence_summary", None)
        st.rerun()

# ── Configurable Scoring Weights Expander ─────────────────────────────────────
with st.expander("⚙️ Configurable Scoring Weights & Methodology", expanded=False):
    st.markdown(
        """<div style="font-size:12px;color:var(--text-muted);margin-bottom:12px;line-height:1.5;">
The overall performance score combines 4 normalized evaluation pillars:
<b>Answer Quality</b> (35%), <b>Retrieval Quality</b> (35%), <b>Performance / Latency</b> (15%), and <b>Cost / Efficiency</b> (15%).<br>
Higher quality scores improve the overall score; lower response time and lower token cost improve the score.<br>
<b>Missing Metric Re-weighting:</b> If any pillar is unavailable for an approach (e.g. Ground Truth reference answer is absent), active weights are dynamically re-normalized so the approach is evaluated fairly on its measured dimensions without penalty.
</div>""",
        unsafe_allow_html=True,
    )

    preset_c1, preset_c2, preset_c3 = st.columns(3)
    if preset_c1.button("Default Preset (35 / 35 / 15 / 15)", use_container_width=True):
        st.session_state["w_ans"] = 35
        st.session_state["w_ret"] = 35
        st.session_state["w_lat"] = 15
        st.session_state["w_cost"] = 15
        st.rerun()
    if preset_c2.button("Quality Priority (45 / 45 / 5 / 5)", use_container_width=True):
        st.session_state["w_ans"] = 45
        st.session_state["w_ret"] = 45
        st.session_state["w_lat"] = 5
        st.session_state["w_cost"] = 5
        st.rerun()
    if preset_c3.button("Speed & Cost Priority (20 / 20 / 30 / 30)", use_container_width=True):
        st.session_state["w_ans"] = 20
        st.session_state["w_ret"] = 20
        st.session_state["w_lat"] = 30
        st.session_state["w_cost"] = 30
        st.rerun()

    sw_col1, sw_col2, sw_col3, sw_col4 = st.columns(4)
    w_ans_in = sw_col1.slider("Answer Quality Weight", min_value=0, max_value=100, value=st.session_state.get("w_ans", 35), step=5)
    w_ret_in = sw_col2.slider("Retrieval Quality Weight", min_value=0, max_value=100, value=st.session_state.get("w_ret", 35), step=5)
    w_lat_in = sw_col3.slider("Latency Weight", min_value=0, max_value=100, value=st.session_state.get("w_lat", 15), step=5)
    w_cost_in = sw_col4.slider("API Cost Weight", min_value=0, max_value=100, value=st.session_state.get("w_cost", 15), step=5)

    custom_weights = ScoringWeights(
        answer_quality=float(w_ans_in),
        retrieval_quality=float(w_ret_in),
        latency=float(w_lat_in),
        cost=float(w_cost_in),
    ).normalized()

    st.caption(
        f"Active Normalized Weights: **Answer Quality: {custom_weights.answer_quality * 100:.1f}%** | "
        f"**Retrieval Quality: {custom_weights.retrieval_quality * 100:.1f}%** | "
        f"**Latency: {custom_weights.latency * 100:.1f}%** | "
        f"**Cost: {custom_weights.cost * 100:.1f}%**"
    )

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

# ── Service Execution ─────────────────────────────────────────────────────────
intelligence_svc = IntelligenceService(
    doc_registry=context.registry,
    user_registry=context.user_registry,
    cost_service=getattr(context, "cost_service", None),
)

summary: IntelligenceSummary = intelligence_svc.get_intelligence_summary(
    kb_id=filter_kb_id,
    weights=custom_weights,
    score_mode=selected_score_mode,
)

# ── KPI Cards ────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    render_html(
        f"""<div style="background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;">
<div style="font-size:11px;color:var(--text-muted);font-weight:600;text-transform:uppercase;letter-spacing:0.04em;">Queries Analyzed</div>
<div style="font-size:28px;font-weight:700;color:var(--text);margin-top:4px;">{summary.total_queries}</div>
<div style="font-size:11px;color:var(--text-muted);margin-top:2px;">
  <span style="color:#34D399;font-weight:600;">{summary.actual_evaluated_queries}</span> verified &bull; 
  <span style="color:#FBBF24;font-weight:600;">{summary.provisional_queries}</span> provisional
</div>
</div>"""
    )

with col2:
    active_cats = summary.classified_categories_count
    render_html(
        f"""<div style="background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;">
<div style="font-size:11px;color:var(--text-muted);font-weight:600;text-transform:uppercase;letter-spacing:0.04em;">Active Categories</div>
<div style="font-size:28px;font-weight:700;color:var(--text);margin-top:4px;">{active_cats} <span style="font-size:16px;color:var(--text-muted);font-weight:500;">/ {len(QUERY_CATEGORIES)}</span></div>
<div style="font-size:11px;color:var(--text-muted);margin-top:2px;">Semantic clusters detected</div>
</div>"""
    )

with col3:
    lead_arch = summary.best_overall_architecture or "Insufficient data"
    arch_badge = _render_approach_badge(lead_arch)
    win_txt = f"{summary.architecture_wins.get(lead_arch, 0)} category win(s)" if lead_arch in summary.architecture_wins else "Awaiting benchmark"
    render_html(
        f"""<div style="background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;">
<div style="font-size:11px;color:var(--text-muted);font-weight:600;text-transform:uppercase;letter-spacing:0.04em;">Top Overall Architecture</div>
<div style="font-size:18px;font-weight:700;margin-top:8px;">{arch_badge}</div>
<div style="font-size:11px;color:var(--text-muted);margin-top:6px;">{win_txt}</div>
</div>"""
    )

with col4:
    mean_lat_str = f"{summary.mean_benchmark_latency_ms:,.0f} ms" if summary.mean_benchmark_latency_ms else "N/A"
    render_html(
        f"""<div style="background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;">
<div style="font-size:11px;color:var(--text-muted);font-weight:600;text-transform:uppercase;letter-spacing:0.04em;">Mean Benchmark Latency</div>
<div style="font-size:28px;font-weight:700;color:var(--text);margin-top:4px;">{mean_lat_str}</div>
<div style="font-size:11px;color:var(--text-muted);margin-top:2px;">Across active architectures</div>
</div>"""
    )

st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

# ── Visual Analytics Dashboard ────────────────────────────────────────────────
st.subheader("Comparison Dashboard", divider=False)

tab_overview, tab_matrix, tab_table, tab_tradeoffs = st.tabs([
    "Category Distribution & Optimal Architecture",
    "Overall Weighted Performance & Heatmap",
    "Detailed Metric Comparison Matrix",
    "Latency & Cost Trade-offs",
])

with tab_overview:
    chart_c1, chart_c2 = st.columns([1, 1], gap="medium")

    with chart_c1:
        st.markdown("<div style='font-weight:600;font-size:13px;margin-bottom:8px;color:var(--text);'>Query Category Distribution</div>", unsafe_allow_html=True)
        cat_counts = {k: v for k, v in summary.category_distribution.items() if v > 0}
        if cat_counts:
            df_dist = pd.DataFrame([
                {"Category": cat, "Count": count, "Color": CATEGORY_COLOR_MAP.get(cat, "#6366F1")}
                for cat, count in cat_counts.items()
            ])
            fig_pie = px.pie(
                df_dist,
                names="Category",
                values="Count",
                color="Category",
                color_discrete_map=CATEGORY_COLOR_MAP,
                hole=0.45,
            )
            fig_pie.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
            )
            fig_pie.update_layout(
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                height=300,
            )
            apply_plotly_theme(fig_pie, current_theme)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No queries classified in this scope yet.")

    with chart_c2:
        st.markdown("<div style='font-weight:600;font-size:13px;margin-bottom:8px;color:var(--text);'>Optimal Architecture by Category</div>", unsafe_allow_html=True)
        active_cats_data = [
            {
                "Category": cat_name,
                "Best Approach": c_intel.best_approach,
                "Queries": c_intel.query_count,
                "Color": STRATEGY_COLORS.get(c_intel.best_approach, "#64748B"),
                "Margin": c_intel.margin_summary,
                "Confidence": c_intel.confidence_level,
            }
            for cat_name, c_intel in summary.categories.items()
            if c_intel.query_count > 0
        ]
        if active_cats_data:
            df_arch = pd.DataFrame(active_cats_data)
            fig_bar = px.bar(
                df_arch,
                y="Category",
                x="Queries",
                color="Best Approach",
                color_discrete_map=STRATEGY_COLORS,
                orientation="h",
                text="Best Approach",
                hover_data=["Margin", "Confidence"],
            )
            fig_bar.update_traces(textposition="inside")
            fig_bar.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=300,
                xaxis_title="Historical Queries Analyzed",
                yaxis_title="",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            apply_plotly_theme(fig_bar, current_theme)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Execute comparative benchmarks to determine optimal architecture by category.")

with tab_matrix:
    st.markdown("<div style='font-weight:600;font-size:13px;margin-bottom:8px;color:var(--text);'>Overall Weighted Performance Scores across Categories</div>", unsafe_allow_html=True)

    rows_quality = []
    for cat_name, c_intel in summary.categories.items():
        if c_intel.query_count > 0:
            for strat in ("VECTOR", "GRAPH", "HYBRID"):
                st_data = c_intel.approach_stats.get(strat, {})
                ov_val = st_data.get("overall_weighted_score")
                if ov_val is not None:
                    rows_quality.append({
                        "Category": cat_name,
                        "Strategy": strat,
                        "Strategy Label": "Vector RAG" if strat == "VECTOR" else ("GraphRAG" if strat == "GRAPH" else "Hybrid RAG"),
                        "Overall Weighted Score": round(ov_val, 3),
                        "Answer Quality": round(st_data.get("answer_quality_score") or 0.0, 3),
                        "Retrieval Quality": round(st_data.get("retrieval_quality_score") or 0.0, 3),
                        "Total Latency (ms)": round(st_data.get("total_latency_ms") or 0.0, 0),
                    })

    if rows_quality:
        df_q = pd.DataFrame(rows_quality)
        fig_q = px.bar(
            df_q,
            x="Category",
            y="Overall Weighted Score",
            color="Strategy Label",
            barmode="group",
            color_discrete_map={
                "Vector RAG": "#3B82F6",
                "GraphRAG": "#10B981",
                "Hybrid RAG": "#8B5CF6",
            },
            hover_data=["Answer Quality", "Retrieval Quality", "Total Latency (ms)"],
        )
        fig_q.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=320,
            xaxis_title="",
            yaxis_title="Normalized Weighted Score (0.0 – 1.0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        apply_plotly_theme(fig_q, current_theme)
        st.plotly_chart(fig_q, use_container_width=True)

        # Performance Heatmap using Overall Weighted Score
        st.markdown("<div style='font-weight:600;font-size:13px;margin-top:16px;margin-bottom:8px;color:var(--text);'>Category vs. Architecture Performance Heatmap (Weighted Score)</div>", unsafe_allow_html=True)
        cats_list = [c for c, intel in summary.categories.items() if intel.query_count > 0]
        strats_list = ["Vector RAG", "GraphRAG", "Hybrid RAG"]
        strat_keys = ["VECTOR", "GRAPH", "HYBRID"]

        z_matrix = []
        for cat in cats_list:
            row_z = []
            for s_key in strat_keys:
                s_stat = summary.categories[cat].approach_stats.get(s_key, {})
                sc = s_stat.get("overall_weighted_score")
                row_z.append(round(sc, 3) if sc is not None else 0.0)
            z_matrix.append(row_z)

        fig_heat = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=strats_list,
            y=cats_list,
            colorscale="Viridis",
            text=[[f"{val:.3f}" if val > 0 else "N/A" for val in r] for r in z_matrix],
            texttemplate="%{text}",
            colorbar=dict(title="Score"),
        ))
        fig_heat.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
        )
        apply_plotly_theme(fig_heat, current_theme)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No comparative evaluation telemetry available in this scope.")

with tab_table:
    st.markdown("<div style='font-weight:600;font-size:13px;margin-bottom:8px;color:var(--text);'>Comprehensive Architecture Comparison Matrix (Across 4 Pillars)</div>", unsafe_allow_html=True)
    st.caption("Aggregated empirical metrics across all active historical queries. Missing metrics remain unavailable ('—') rather than fabricated.")

    # Compute dataset-level pillar averages for Vector, Graph, and Hybrid RAG
    all_apps = [app for q in summary.all_queries for app in q.approaches.values() if app.status == "success"]

    def _avg(strat: str, key: str, is_metric: bool = False) -> str:
        vals = []
        for app in all_apps:
            if app.strategy == strat:
                if is_metric:
                    m = app.actual_metrics if not app.is_provisional else {**app.actual_metrics, **app.provisional_metrics}
                    v = m.get(key)
                else:
                    v = getattr(app, key, None)
                if v is not None and isinstance(v, (int, float)):
                    vals.append(v)
        if not vals:
            return "—"
        mean_v = statistics.mean(vals)
        if "latency" in key:
            return f"{mean_v:,.0f} ms"
        elif "cost" in key:
            return f"${mean_v:.5f} USD"
        elif "token" in key:
            return f"{mean_v:,.0f}"
        else:
            return f"{mean_v:.4f}"

    def _determine_winner(v_val: str, g_val: str, h_val: str, higher_is_better: bool = True) -> str:
        def _parse(val_str):
            clean = val_str.replace(" ms", "").replace("$", "").replace(" USD", "").replace(",", "").strip()
            try:
                return float(clean)
            except Exception:
                return None
        pv, pg, ph = _parse(v_val), _parse(g_val), _parse(h_val)
        items = [(pv, "Vector RAG"), (pg, "GraphRAG"), (ph, "Hybrid RAG")]
        valid = [it for it in items if it[0] is not None]
        if not valid:
            return "—"
        if higher_is_better:
            return max(valid, key=lambda x: x[0])[1]
        else:
            return min(valid, key=lambda x: x[0])[1]

    matrix_rows = [
        # Overall
        {"Pillar": "Overall", "Parameter": "Overall Weighted Score", "Vector RAG": _avg("VECTOR", "overall_weighted_score"), "GraphRAG": _avg("GRAPH", "overall_weighted_score"), "Hybrid RAG": _avg("HYBRID", "overall_weighted_score"), "Winner": _determine_winner(_avg("VECTOR", "overall_weighted_score"), _avg("GRAPH", "overall_weighted_score"), _avg("HYBRID", "overall_weighted_score"), True)},
        # Answer Quality
        {"Pillar": "Answer Quality (35%)", "Parameter": "Answer Quality Sub-Score", "Vector RAG": _avg("VECTOR", "answer_quality_score"), "GraphRAG": _avg("GRAPH", "answer_quality_score"), "Hybrid RAG": _avg("HYBRID", "answer_quality_score"), "Winner": _determine_winner(_avg("VECTOR", "answer_quality_score"), _avg("GRAPH", "answer_quality_score"), _avg("HYBRID", "answer_quality_score"), True)},
        {"Pillar": "Answer Quality (35%)", "Parameter": "Answer Correctness", "Vector RAG": _avg("VECTOR", "answer_correctness", True), "GraphRAG": _avg("GRAPH", "answer_correctness", True), "Hybrid RAG": _avg("HYBRID", "answer_correctness", True), "Winner": _determine_winner(_avg("VECTOR", "answer_correctness", True), _avg("GRAPH", "answer_correctness", True), _avg("HYBRID", "answer_correctness", True), True)},
        {"Pillar": "Answer Quality (35%)", "Parameter": "Faithfulness (Grounding)", "Vector RAG": _avg("VECTOR", "faithfulness", True), "GraphRAG": _avg("GRAPH", "faithfulness", True), "Hybrid RAG": _avg("HYBRID", "faithfulness", True), "Winner": _determine_winner(_avg("VECTOR", "faithfulness", True), _avg("GRAPH", "faithfulness", True), _avg("HYBRID", "faithfulness", True), True)},
        {"Pillar": "Answer Quality (35%)", "Parameter": "Answer Relevancy", "Vector RAG": _avg("VECTOR", "answer_relevancy", True), "GraphRAG": _avg("GRAPH", "answer_relevancy", True), "Hybrid RAG": _avg("HYBRID", "answer_relevancy", True), "Winner": _determine_winner(_avg("VECTOR", "answer_relevancy", True), _avg("GRAPH", "answer_relevancy", True), _avg("HYBRID", "answer_relevancy", True), True)},
        # Retrieval Quality
        {"Pillar": "Retrieval Quality (35%)", "Parameter": "Retrieval Quality Sub-Score", "Vector RAG": _avg("VECTOR", "retrieval_quality_score"), "GraphRAG": _avg("GRAPH", "retrieval_quality_score"), "Hybrid RAG": _avg("HYBRID", "retrieval_quality_score"), "Winner": _determine_winner(_avg("VECTOR", "retrieval_quality_score"), _avg("GRAPH", "retrieval_quality_score"), _avg("HYBRID", "retrieval_quality_score"), True)},
        {"Pillar": "Retrieval Quality (35%)", "Parameter": "Context Precision", "Vector RAG": _avg("VECTOR", "context_precision", True), "GraphRAG": _avg("GRAPH", "context_precision", True), "Hybrid RAG": _avg("HYBRID", "context_precision", True), "Winner": _determine_winner(_avg("VECTOR", "context_precision", True), _avg("GRAPH", "context_precision", True), _avg("HYBRID", "context_precision", True), True)},
        {"Pillar": "Retrieval Quality (35%)", "Parameter": "Context Recall", "Vector RAG": _avg("VECTOR", "context_recall", True), "GraphRAG": _avg("GRAPH", "context_recall", True), "Hybrid RAG": _avg("HYBRID", "context_recall", True), "Winner": _determine_winner(_avg("VECTOR", "context_recall", True), _avg("GRAPH", "context_recall", True), _avg("HYBRID", "context_recall", True), True)},
        {"Pillar": "Retrieval Quality (35%)", "Parameter": "Retrieval Recall @ K", "Vector RAG": _avg("VECTOR", "retrieval_recall_at_k", True), "GraphRAG": _avg("GRAPH", "retrieval_recall_at_k", True), "Hybrid RAG": _avg("HYBRID", "retrieval_recall_at_k", True), "Winner": _determine_winner(_avg("VECTOR", "retrieval_recall_at_k", True), _avg("GRAPH", "retrieval_recall_at_k", True), _avg("HYBRID", "retrieval_recall_at_k", True), True)},
        # Performance
        {"Pillar": "Performance (15%)", "Parameter": "Retrieval Latency", "Vector RAG": _avg("VECTOR", "retrieval_latency_ms"), "GraphRAG": _avg("GRAPH", "retrieval_latency_ms"), "Hybrid RAG": _avg("HYBRID", "retrieval_latency_ms"), "Winner": _determine_winner(_avg("VECTOR", "retrieval_latency_ms"), _avg("GRAPH", "retrieval_latency_ms"), _avg("HYBRID", "retrieval_latency_ms"), False)},
        {"Pillar": "Performance (15%)", "Parameter": "Generation Latency", "Vector RAG": _avg("VECTOR", "generation_latency_ms"), "GraphRAG": _avg("GRAPH", "generation_latency_ms"), "Hybrid RAG": _avg("HYBRID", "generation_latency_ms"), "Winner": _determine_winner(_avg("VECTOR", "generation_latency_ms"), _avg("GRAPH", "generation_latency_ms"), _avg("HYBRID", "generation_latency_ms"), False)},
        {"Pillar": "Performance (15%)", "Parameter": "Total Response Time", "Vector RAG": _avg("VECTOR", "total_latency_ms"), "GraphRAG": _avg("GRAPH", "total_latency_ms"), "Hybrid RAG": _avg("HYBRID", "total_latency_ms"), "Winner": _determine_winner(_avg("VECTOR", "total_latency_ms"), _avg("GRAPH", "total_latency_ms"), _avg("HYBRID", "total_latency_ms"), False)},
        # Efficiency
        {"Pillar": "Efficiency (15%)", "Parameter": "Input Tokens Consumed", "Vector RAG": _avg("VECTOR", "prompt_tokens"), "GraphRAG": _avg("GRAPH", "prompt_tokens"), "Hybrid RAG": _avg("HYBRID", "prompt_tokens"), "Winner": _determine_winner(_avg("VECTOR", "prompt_tokens"), _avg("GRAPH", "prompt_tokens"), _avg("HYBRID", "prompt_tokens"), False)},
        {"Pillar": "Efficiency (15%)", "Parameter": "Output Tokens Consumed", "Vector RAG": _avg("VECTOR", "completion_tokens"), "GraphRAG": _avg("GRAPH", "completion_tokens"), "Hybrid RAG": _avg("HYBRID", "completion_tokens"), "Winner": _determine_winner(_avg("VECTOR", "completion_tokens"), _avg("GRAPH", "completion_tokens"), _avg("HYBRID", "completion_tokens"), False)},
        {"Pillar": "Efficiency (15%)", "Parameter": "Total Tokens Consumed", "Vector RAG": _avg("VECTOR", "total_tokens"), "GraphRAG": _avg("GRAPH", "total_tokens"), "Hybrid RAG": _avg("HYBRID", "total_tokens"), "Winner": _determine_winner(_avg("VECTOR", "total_tokens"), _avg("GRAPH", "total_tokens"), _avg("HYBRID", "total_tokens"), False)},
        {"Pillar": "Efficiency (15%)", "Parameter": "Estimated API Cost (USD)", "Vector RAG": _avg("VECTOR", "cost_usd"), "GraphRAG": _avg("GRAPH", "cost_usd"), "Hybrid RAG": _avg("HYBRID", "cost_usd"), "Winner": _determine_winner(_avg("VECTOR", "cost_usd"), _avg("GRAPH", "cost_usd"), _avg("HYBRID", "cost_usd"), False)},
    ]

    st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)

with tab_tradeoffs:
    st.markdown("<div style='font-weight:600;font-size:13px;margin-bottom:8px;color:var(--text);'>Latency and API Cost Trade-offs</div>", unsafe_allow_html=True)

    tradeoff_rows = []
    for cat_name, c_intel in summary.categories.items():
        if c_intel.query_count > 0:
            for strat in ("VECTOR", "GRAPH", "HYBRID"):
                st_data = c_intel.approach_stats.get(strat, {})
                tot_l = st_data.get("total_latency_ms")
                cost_u = st_data.get("cost_usd")
                qual = st_data.get("overall_weighted_score") or 0.5
                if tot_l is not None:
                    tradeoff_rows.append({
                        "Category": cat_name,
                        "Strategy": "Vector RAG" if strat == "VECTOR" else ("GraphRAG" if strat == "GRAPH" else "Hybrid RAG"),
                        "Total Latency (ms)": tot_l,
                        "Retrieval Latency (ms)": st_data.get("retrieval_latency_ms") or 0.0,
                        "Generation Latency (ms)": st_data.get("generation_latency_ms") or 0.0,
                        "Cost (USD)": cost_u or 0.0001,
                        "Tokens": st_data.get("tokens") or 1000,
                        "Overall Score": qual,
                    })

    if tradeoff_rows:
        df_trade = pd.DataFrame(tradeoff_rows)
        tc1, tc2 = st.columns(2)

        with tc1:
            st.markdown("<div style='font-size:12px;color:var(--text-muted);margin-bottom:4px;'>Latency Comparison across Architectures</div>", unsafe_allow_html=True)
            fig_lat = px.bar(
                df_trade,
                x="Category",
                y="Total Latency (ms)",
                color="Strategy",
                barmode="group",
                color_discrete_map={
                    "Vector RAG": "#3B82F6",
                    "GraphRAG": "#10B981",
                    "Hybrid RAG": "#8B5CF6",
                },
                hover_data=["Retrieval Latency (ms)", "Generation Latency (ms)"],
            )
            fig_lat.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
            apply_plotly_theme(fig_lat, current_theme)
            st.plotly_chart(fig_lat, use_container_width=True)

        with tc2:
            st.markdown("<div style='font-size:12px;color:var(--text-muted);margin-bottom:4px;'>Token Consumption &amp; API Cost</div>", unsafe_allow_html=True)
            fig_cost = px.bar(
                df_trade,
                x="Category",
                y="Cost (USD)",
                color="Strategy",
                barmode="group",
                color_discrete_map={
                    "Vector RAG": "#3B82F6",
                    "GraphRAG": "#10B981",
                    "Hybrid RAG": "#8B5CF6",
                },
                hover_data=["Tokens"],
            )
            fig_cost.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
            apply_plotly_theme(fig_cost, current_theme)
            st.plotly_chart(fig_cost, use_container_width=True)
    else:
        st.info("No latency telemetry recorded yet in this scope.")

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# ── Architecture Recommendation Engine ────────────────────────────────────────
st.subheader("Category-Wise Recommendations", divider=False)
st.caption(
    "Data-derived architectural recommendations across all 9 query categories identifying: "
    "Best Overall Architecture, Best Answer Quality, Best Retrieval Quality, Fastest Architecture, and Most Cost-Efficient Architecture. "
    "Confidence levels prevent over-interpreting sparse sample sizes."
)

cat_keys = list(summary.categories.keys())
for row_idx in range(0, len(cat_keys), 3):
    cols = st.columns(3)
    for col_idx in range(3):
        item_idx = row_idx + col_idx
        if item_idx < len(cat_keys):
            c_name = cat_keys[item_idx]
            c_intel = summary.categories[c_name]
            with cols[col_idx]:
                cat_color = c_intel.color
                winner_badge = _render_approach_badge(c_intel.best_approach)
                conf_badge = _render_confidence_badge(c_intel.confidence_level)

                has_data = c_intel.query_count > 0 and c_intel.best_approach != "Insufficient data"
                border_color = "var(--border)" if not has_data else f"{STRATEGY_COLORS.get(c_intel.best_approach, 'var(--accent)')}40"

                strengths_html = "".join([f"<li style='margin-bottom:3px;'>{s}</li>" for s in c_intel.strengths])

                warning_notice = ""
                if c_intel.confidence_level in ("Low", "Insufficient data"):
                    warning_notice = f"""<div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.25);border-radius:4px;padding:6px 8px;font-size:10px;color:#FCD34D;margin-top:8px;">
⚠️ <b>Advisory:</b> {c_intel.data_sufficiency_reason}
</div>"""

                st.markdown(
                    f"""<div style="background:var(--card);border:1px solid {border_color};border-radius:8px;padding:16px;height:100%;margin-bottom:14px;display:flex;flex-direction:column;justify-content:space-between;">
<div>
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;">
    <div style="font-size:14px;font-weight:700;color:var(--text);display:flex;align-items:center;gap:6px;">
      <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{cat_color};"></span>
      {c_name}
    </div>
    <div>{conf_badge}</div>
  </div>
  <p style="font-size:11px;color:var(--text-muted);line-height:1.4;margin:0 0 10px 0;">{c_intel.description}</p>
  
  <div style="margin-bottom:10px;">
    <div style="font-size:11px;color:var(--text-muted);text-transform:uppercase;font-weight:600;margin-bottom:3px;">Best Overall Architecture</div>
    <div>{winner_badge}</div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:11px;background:rgba(255,255,255,0.02);border:1px solid var(--border);border-radius:6px;padding:8px;margin-bottom:10px;">
    <div>🎯 <b>Answer Quality:</b><br><span style="color:var(--text-muted);">{c_intel.best_answer_quality_arch}</span></div>
    <div>🔍 <b>Retrieval Quality:</b><br><span style="color:var(--text-muted);">{c_intel.best_retrieval_quality_arch}</span></div>
    <div>⚡ <b>Fastest:</b><br><span style="color:var(--text-muted);">{c_intel.fastest_arch}</span></div>
    <div>💰 <b>Cost-Efficient:</b><br><span style="color:var(--text-muted);">{c_intel.most_cost_efficient_arch}</span></div>
  </div>

  <div style="background:rgba(255,255,255,0.03);border-radius:6px;padding:8px 10px;margin-bottom:10px;">
    <div style="font-size:11px;font-weight:600;color:var(--text);">{c_intel.margin_summary}</div>
    <div style="font-size:11px;color:var(--text-muted);margin-top:2px;line-height:1.4;">{c_intel.empirical_reason}</div>
  </div>

  <div style="font-size:11px;color:var(--text-muted);margin-bottom:8px;">
    <div style="font-weight:600;color:var(--text);margin-bottom:3px;">Trade-offs &amp; Rationale:</div>
    <div style="line-height:1.4;">{c_intel.tradeoffs}</div>
  </div>
</div>

<div>
  <ul style="font-size:11px;color:var(--text-muted);padding-left:16px;margin:0;">
    {strengths_html}
  </ul>
  {warning_notice}
</div>
</div>""",
                    unsafe_allow_html=True,
                )

st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

# ── Historical Query Explorer ────────────────────────────────────────────────
st.subheader("Historical Query Explorer", divider=False)
st.caption("Search, filter, and drill into individual queries. Actual evaluation results are explicitly distinguished from estimated provisional runs.")

# Filters
q_filter_c1, q_filter_c2, q_filter_c3, q_filter_c4 = st.columns([3, 2, 2, 2])

with q_filter_c1:
    search_query = st.text_input("Search query text", placeholder="Type keywords, e.g. 'hospital', 'P-101', 'protocol'...")

with q_filter_c2:
    cat_filter_options = ["All Categories"] + QUERY_CATEGORIES
    selected_cat_filter = st.selectbox("Filter by Category", options=cat_filter_options, index=0)

with q_filter_c3:
    approach_filter_options = ["All Approaches", "VECTOR", "GRAPH", "HYBRID", "Tie", "Insufficient data"]
    selected_approach_filter = st.selectbox("Filter by Best Approach", options=approach_filter_options, index=0)

with q_filter_c4:
    source_filter_options = ["All Sources", "Actual Evaluation Only", "Estimated / Provisional Only"]
    selected_source_filter = st.selectbox("Filter by Score Source", options=source_filter_options, index=0)

# Filter logic
filtered_queries = summary.all_queries

if search_query:
    filtered_queries = [
        q for q in filtered_queries if search_query.lower() in q.query_text.lower()
    ]

if selected_cat_filter != "All Categories":
    filtered_queries = [
        q for q in filtered_queries if q.category == selected_cat_filter
    ]

if selected_approach_filter != "All Approaches":
    filtered_queries = [
        q for q in filtered_queries if q.best_approach == selected_approach_filter
    ]

if selected_source_filter == "Actual Evaluation Only":
    filtered_queries = [q for q in filtered_queries if not q.is_provisional]
elif selected_source_filter == "Estimated / Provisional Only":
    filtered_queries = [q for q in filtered_queries if q.is_provisional]

st.markdown(f"<div style='font-size:12px;color:var(--text-muted);margin-bottom:8px;'>Showing <b>{len(filtered_queries)}</b> of {len(summary.all_queries)} queries</div>", unsafe_allow_html=True)

if filtered_queries:
    table_rows = []
    for q in filtered_queries:
        v_app = q.approaches.get("VECTOR")
        g_app = q.approaches.get("GRAPH")
        h_app = q.approaches.get("HYBRID")

        v_score = f"{v_app.overall_weighted_score:.3f}" if (v_app and v_app.overall_weighted_score is not None) else "—"
        g_score = f"{g_app.overall_weighted_score:.3f}" if (g_app and g_app.overall_weighted_score is not None) else "—"
        h_score = f"{h_app.overall_weighted_score:.3f}" if (h_app and h_app.overall_weighted_score is not None) else "—"

        best_rec = q.approaches.get(q.best_approach)
        best_lat = f"{best_rec.total_latency_ms:,.0f} ms" if (best_rec and best_rec.total_latency_ms) else "—"
        best_cost = f"${best_rec.cost_usd:.5f}" if (best_rec and best_rec.cost_usd) else "—"

        table_rows.append({
            "Query": q.query_text,
            "Category": q.category,
            "Score Source": q.score_source,
            "Vector RAG": v_score,
            "GraphRAG": g_score,
            "Hybrid RAG": h_score,
            "Best Approach": q.best_approach,
            "Margin": q.margin_summary or "—",
            "Latency": best_lat,
            "Cost": best_cost,
            "QueryID": q.query_id,
        })

    df_table = pd.DataFrame(table_rows)
    st.dataframe(
        df_table[["Query", "Category", "Score Source", "Vector RAG", "GraphRAG", "Hybrid RAG", "Best Approach", "Margin", "Latency", "Cost"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ── Drill-down Query Inspector ───────────────────────────────────────────
    st.markdown("<div style='font-weight:600;font-size:14px;color:var(--text);margin-bottom:8px;'>Individual Query Drill-down Inspector</div>", unsafe_allow_html=True)

    query_labels = {q.query_id: f"[{q.category}] [{q.score_source}] {q.query_text[:70]}..." for q in filtered_queries}
    selected_qid = st.selectbox(
        "Select query to inspect multi-pillar execution telemetry:",
        options=list(query_labels.keys()),
        format_func=lambda qid: query_labels[qid],
    )

    selected_q = next((q for q in filtered_queries if q.query_id == selected_qid), None)

    if selected_q:
        q_card_html = f"""<div style="background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px;margin-bottom:14px;">
<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;">
  <div style="display:flex;align-items:center;gap:8px;">
    <span style="background:{CATEGORY_COLOR_MAP.get(selected_q.category, '#6366F1')}20;color:{CATEGORY_COLOR_MAP.get(selected_q.category, '#6366F1')};border:1px solid {CATEGORY_COLOR_MAP.get(selected_q.category, '#6366F1')}40;padding:2px 6px;border-radius:4px;font-size:11px;font-weight:600;">{selected_q.category}</span>
    {_render_source_badge(selected_q.score_source)}
    <span style="font-size:11px;color:var(--text-muted);">Confidence: {selected_q.category_confidence:.2f}</span>
  </div>
  <div>{_render_approach_badge(selected_q.best_approach)}</div>
</div>
<div style="font-size:14px;font-weight:600;color:var(--text);margin-bottom:4px;">"{selected_q.query_text}"</div>
<div style="font-size:12px;color:var(--text-muted);">{selected_q.recommendation_reason}</div>
</div>"""
        render_html(q_card_html)

        # 3 Side-by-side Strategy Telemetry Columns
        c_vec, c_graph, c_hybrid = st.columns(3)

        strategies_meta = [
            ("VECTOR", "Vector RAG", c_vec, "#3B82F6"),
            ("GRAPH", "GraphRAG", c_graph, "#10B981"),
            ("HYBRID", "Hybrid RAG", c_hybrid, "#8B5CF6"),
        ]

        for s_key, s_label, col_container, col_accent in strategies_meta:
            app_rec = selected_q.approaches.get(s_key)
            with col_container:
                if app_rec:
                    ov_score_str = f"{app_rec.overall_weighted_score:.3f}" if app_rec.overall_weighted_score is not None else "—"
                    ans_score_str = f"{app_rec.answer_quality_score:.3f}" if app_rec.answer_quality_score is not None else "—"
                    ret_score_str = f"{app_rec.retrieval_quality_score:.3f}" if app_rec.retrieval_quality_score is not None else "—"
                    tot_lat_str = f"{app_rec.total_latency_ms:,.0f} ms" if app_rec.total_latency_ms is not None else "—"
                    ret_lat_str = f"{app_rec.retrieval_latency_ms:,.0f} ms" if app_rec.retrieval_latency_ms is not None else "—"
                    gen_lat_str = f"{app_rec.generation_latency_ms:,.0f} ms" if app_rec.generation_latency_ms is not None else "—"
                    tok_str = f"{app_rec.total_tokens:,}" if app_rec.total_tokens else "—"
                    cost_str = f"${app_rec.cost_usd:.5f}" if app_rec.cost_usd else "—"
                    src_badge = _render_source_badge(app_rec.score_source)

                    cit_count = len(app_rec.citations)
                    g_ent_count = len(app_rec.graph_entities)
                    g_rel_count = len(app_rec.graph_relationships)

                    st.markdown(
                        f"""<div style="background:var(--card);border:1px solid {col_accent}40;border-radius:8px;padding:12px;margin-bottom:12px;">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
  <span style="font-weight:700;font-size:13px;color:{col_accent};">{s_label}</span>
  <div>{src_badge}</div>
</div>

<div style="background:rgba(255,255,255,0.03);border-radius:4px;padding:6px;margin-bottom:8px;">
  <div style="font-size:11px;color:var(--text-muted);">Overall Weighted Score: <b style="color:var(--text);font-size:13px;">{ov_score_str}</b></div>
  <div style="font-size:10px;color:var(--text-muted);display:flex;gap:8px;margin-top:2px;">
    <span>Answer Qual: <b>{ans_score_str}</b></span>
    <span>Retrieval Qual: <b>{ret_score_str}</b></span>
  </div>
</div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:11px;margin-bottom:8px;">
  <div style="color:var(--text-muted);">Total Latency: <b style="color:var(--text);">{tot_lat_str}</b></div>
  <div style="color:var(--text-muted);">Retrieval: <b style="color:var(--text);">{ret_lat_str}</b></div>
  <div style="color:var(--text-muted);">Tokens: <b style="color:var(--text);">{tok_str}</b></div>
  <div style="color:var(--text-muted);">Cost: <b style="color:var(--text);">{cost_str}</b></div>
</div>

<div style="font-size:11px;color:var(--text-muted);margin-bottom:6px;">
  Evidence: <b>{cit_count}</b> citations | <b>{g_ent_count + g_rel_count}</b> graph elements
</div>
</div>""",
                        unsafe_allow_html=True,
                    )

                    with st.expander(f"View {s_label} Answer & Citations", expanded=False):
                        st.markdown(f"**Generated Answer:**\n\n{app_rec.answer or 'No answer recorded.'}")
                        if app_rec.citations:
                            st.markdown(f"**Citations ({len(app_rec.citations)}):**")
                            for c in app_rec.citations[:4]:
                                doc_n = c.get("document_name", "Source")
                                ch_idx = c.get("chunk_index", "0")
                                st.caption(f"• {doc_n} (Chunk {ch_idx})")
                        if app_rec.graph_entities or app_rec.graph_relationships:
                            st.markdown(f"**Graph Elements:**")
                            st.caption(f"Entities: {len(app_rec.graph_entities)} | Relationships: {len(app_rec.graph_relationships)}")
                else:
                    st.markdown(
                        f"""<div style="background:var(--card);border:1px dashed var(--border);border-radius:8px;padding:20px;text-align:center;color:var(--text-muted);font-size:12px;">
{s_label} was not executed for this query.
</div>""",
                        unsafe_allow_html=True,
                    )
else:
    st.info("No queries match the selected filters.")
