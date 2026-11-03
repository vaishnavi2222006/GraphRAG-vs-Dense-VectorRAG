from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rag_benchmark_lab.app import get_context
from rag_benchmark_lab.utils.auth import get_current_user_name, require_auth
from rag_benchmark_lab.utils.ui import (
    format_latency,
    format_metric,
    inject_custom_css,
    render_dev_mode_banner,
    render_lab_header,
)

# page config handled in main app
inject_custom_css()
user_id = require_auth()
context = get_context()
user_name = get_current_user_name() or "User"

# Sidebar: KB selector and logout
with st.sidebar:
    st.markdown(f"**👤 {user_name}**")
    kbs = context.user_registry.list_knowledge_bases(user_id)
    active_kb_id = st.session_state.get("active_kb_id")
    if kbs:
        kb_options = {kb.kb_id: kb.name for kb in kbs}
        if active_kb_id not in kb_options:
            active_kb_id = list(kb_options.keys())[0]
            st.session_state["active_kb_id"] = active_kb_id
        selected_kb_id = st.selectbox(
            "📂 Knowledge Base",
            options=list(kb_options.keys()),
            format_func=lambda x: kb_options[x],
            index=list(kb_options.keys()).index(active_kb_id),
            key="analytics_kb_selector",
        )
        if selected_kb_id != active_kb_id:
            st.session_state["active_kb_id"] = selected_kb_id
            st.rerun()
        active_kb_id = selected_kb_id
    else:
        active_kb_id = None
        st.info("No knowledge bases yet.")
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True, key="analytics_logout"):
        from rag_benchmark_lab.utils.auth import logout_user
        logout_user()
        st.rerun()

render_lab_header(
    title="Comparative Evaluation Analytics",
    subtitle="In-depth analysis of retrieval recall, answer quality, latency tradeoffs, and multi-hop reasoning performance.",
    badge_text="RESEARCH ANALYTICS",
)

render_dev_mode_banner(context.settings.llm_mock_mode)

results = context.result_store.list_results()
runs = context.result_store.list_runs()

if not results:
    st.info("📊 No verified evaluation results are available yet. Run an experiment in **Evaluation Lab** to populate this dashboard.")
    st.stop()

# Build DataFrame from stored JSON payloads
rows = []
for res in results:
    payload = res
    if not isinstance(payload, dict):
        continue
    mets = payload.get("metrics") or {}
    rows.append(
        {
            "run_id": payload.get("run_id", "unknown"),
            "query_id": payload.get("query_id", "unknown"),
            "strategy": payload.get("strategy", "unknown"),
            "question": payload.get("question", ""),
            "answer": payload.get("answer", ""),
            "retrieval_latency_ms": payload.get("retrieval_latency_ms") or 0.0,
            "generation_latency_ms": payload.get("generation_latency_ms") or 0.0,
            "total_latency_ms": payload.get("total_latency_ms") or 0.0,
            "faithfulness": mets.get("faithfulness", {}).get("value") if isinstance(mets.get("faithfulness"), dict) else mets.get("faithfulness"),
            "answer_relevancy": mets.get("answer_relevancy", {}).get("value") if isinstance(mets.get("answer_relevancy"), dict) else mets.get("answer_relevancy"),
            "context_precision": mets.get("context_precision", {}).get("value") if isinstance(mets.get("context_precision"), dict) else mets.get("context_precision"),
            "context_recall": mets.get("context_recall", {}).get("value") if isinstance(mets.get("context_recall"), dict) else mets.get("context_recall"),
            "retrieval_recall_at_k": mets.get("retrieval_recall_at_k", {}).get("value") if isinstance(mets.get("retrieval_recall_at_k"), dict) else mets.get("retrieval_recall_at_k"),
            "mrr": mets.get("mrr", {}).get("value") if isinstance(mets.get("mrr"), dict) else mets.get("mrr"),
            "mock_mode": payload.get("mock_mode", False),
        }
    )

df_all = pd.DataFrame(rows)

# Analytics Filters
st.markdown("### 🔍 Filter Analytics Data")
f_col1, f_col2, f_col3 = st.columns(3)

all_run_ids = ["All Runs"] + sorted(list(df_all["run_id"].unique()))
with f_col1:
    selected_run = st.selectbox("Filter by Benchmark Run", options=all_run_ids, index=0)

all_strats = ["All Strategies"] + sorted(list(df_all["strategy"].unique()))
with f_col2:
    selected_strat = st.selectbox("Filter by Strategy", options=all_strats, index=0)

with f_col3:
    mode_filter = st.selectbox("Execution Mode", options=["All", "Real LLM Only", "Mock / Dev Only"], index=0)

# Apply filters
df = df_all.copy()
if selected_run != "All Runs":
    df = df[df["run_id"] == selected_run]
if selected_strat != "All Strategies":
    df = df[df["strategy"] == selected_strat]
if mode_filter == "Real LLM Only":
    df = df[df["mock_mode"] == False]  # noqa: E712
elif mode_filter == "Mock / Dev Only":
    df = df[df["mock_mode"] == True]  # noqa: E712

if df.empty:
    if mode_filter == "Real LLM Only":
        st.info("No verified evaluation results are available yet. Run an experiment to populate this dashboard.")
    else:
        st.warning("No records match the selected filter combination.")
    st.stop()

# If showing mock/dev records, render prominent development banner
has_mock = bool(df["mock_mode"].any())
all_mock = bool(df["mock_mode"].all())

if all_mock:
    st.warning(
        "⚡ **Development / Mock Mode Records:** All filtered evaluation runs were executed in offline development mode (Ollama LLM offline). "
        "These are synthetic experimental runs for pipeline validation and should not be cited as genuine model benchmark scores."
    )
elif has_mock:
    st.info("ℹ️ Note: Filtered dataset includes a mix of genuine LLM runs and development mock runs.")

# Top KPI Summary Cards
st.markdown("### 📌 Executive Research KPI Summary")
kpi_cols = st.columns(5)

with kpi_cols[0]:
    st.metric("Queries Analyzed", len(df))
with kpi_cols[1]:
    mean_faith = df["faithfulness"].dropna().mean() if not df["faithfulness"].dropna().empty else None
    st.metric("Avg Faithfulness", format_metric(mean_faith, fallback="N/A", precision=3) if (mean_faith is not None and not pd.isna(mean_faith)) else "N/A")
with kpi_cols[2]:
    mean_rel = df["answer_relevancy"].dropna().mean() if not df["answer_relevancy"].dropna().empty else None
    st.metric("Avg Relevancy", format_metric(mean_rel, fallback="N/A", precision=3) if (mean_rel is not None and not pd.isna(mean_rel)) else "N/A")
with kpi_cols[3]:
    mean_rec = df["retrieval_recall_at_k"].dropna().mean() if not df["retrieval_recall_at_k"].dropna().empty else None
    st.metric("Avg Recall@K", format_metric(mean_rec, fallback="N/A", precision=3) if (mean_rec is not None and not pd.isna(mean_rec)) else "N/A")
with kpi_cols[4]:
    mean_lat = df["total_latency_ms"].dropna().mean() if not df["total_latency_ms"].dropna().empty else None
    st.metric("Mean Latency", format_latency(mean_lat, fallback="N/A", precision=1) if (mean_lat is not None and not pd.isna(mean_lat)) else "N/A")

st.markdown("---")

# Visualizations Section
st.markdown("### 📊 Empirical Comparative Visualizations")
tab_charts1, tab_charts2, tab_table = st.tabs(["Quality Metrics by Strategy", "Latency Distributions", "Per-Query Results"])

with tab_charts1:
    strat_group = df.groupby("strategy")[["faithfulness", "answer_relevancy", "context_precision", "context_recall", "retrieval_recall_at_k", "mrr"]].mean().reset_index()
    if not strat_group.empty:
        df_melted = pd.melt(
            strat_group,
            id_vars=["strategy"],
            var_name="Metric",
            value_name="Score",
        )
        df_melted["Score"] = df_melted["Score"].fillna(0.0)
        
        fig_bar = px.bar(
            df_melted,
            x="Metric",
            y="Score",
            color="strategy",
            barmode="group",
            title="Quality Metrics Across Retrieval Architectures (Vector vs. Graph vs. Hybrid)",
            color_discrete_map={"VECTOR": "#0284c7", "GRAPH": "#059669", "HYBRID": "#7c3aed"},
            height=450,
        )
        fig_bar.update_layout(
            plot_bgcolor="#f8fafc",
            paper_bgcolor="#f8fafc",
            yaxis=dict(range=[0, 1.05], title="Score (0.0 to 1.0)"),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Insufficient quality metric data for chart generation.")

with tab_charts2:
    col_lat1, col_lat2 = st.columns(2)
    with col_lat1:
        fig_box = px.box(
            df,
            x="strategy",
            y="total_latency_ms",
            color="strategy",
            points="all",
            title="Total Latency Distribution (ms)",
            color_discrete_map={"VECTOR": "#0284c7", "GRAPH": "#059669", "HYBRID": "#7c3aed"},
            height=420,
        )
        fig_box.update_layout(plot_bgcolor="#f8fafc", paper_bgcolor="#f8fafc", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_box, use_container_width=True)

    with col_lat2:
        df_lat = df.groupby("strategy")[["retrieval_latency_ms", "generation_latency_ms"]].mean().reset_index()
        fig_stacked = go.Figure(
            data=[
                go.Bar(name="Retrieval Latency", x=df_lat["strategy"], y=df_lat["retrieval_latency_ms"], marker_color="#3b82f6"),
                go.Bar(name="Generation Latency", x=df_lat["strategy"], y=df_lat["generation_latency_ms"], marker_color="#8b5cf6"),
            ]
        )
        fig_stacked.update_layout(
            barmode="stack",
            title="Retrieval vs. Generation Latency Breakdown (Mean ms)",
            plot_bgcolor="#f8fafc",
            paper_bgcolor="#f8fafc",
            height=420,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_stacked, use_container_width=True)

with tab_table:
    st.markdown("#### Detailed Evaluated Queries")
    df_display = df.copy()

    def _format_preview(a: str) -> str:
        if not a:
            return "No response"
        if "Mock response." in a:
            # Extract question or context if present in old string
            cleaned = a.replace("Mock response. The system is running in mock mode because Ollama is unavailable or disabled. This response should not be used as a real experiment result.", "").strip()
            return f"[MOCK/DEV] {cleaned[:90]}..." if cleaned else "[MOCK/DEV SYNTHESIS]"
        if "[DEVELOPMENT / MOCK MODE" in a:
            body = a.split("*(Notice:", 1)[0].replace("[DEVELOPMENT / MOCK MODE: Ollama Service Offline]", "").strip()
            return f"[DEV SYNTHESIS] {body[:90]}..."
        return a[:100] + "..." if len(a) > 100 else a

    df_display["Answer Preview"] = df_display["answer"].apply(_format_preview)
    df_display["Question Preview"] = df_display["question"].apply(lambda q: q[:60] + "..." if len(q) > 60 else q)
    df_display["Mode"] = df_display["mock_mode"].apply(lambda m: "MOCK / DEV" if m else "REAL LLM")
    
    st.dataframe(
        df_display[
            [
                "query_id",
                "strategy",
                "Question Preview",
                "Answer Preview",
                "total_latency_ms",
                "faithfulness",
                "answer_relevancy",
                "retrieval_recall_at_k",
                "mrr",
                "Mode",
            ]
        ],
        use_container_width=True,
    )

    csv_data = df_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Evaluation Results to CSV",
        data=csv_data,
        file_name="rag_benchmark_evaluation_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("---")

# Infrastructure Cost & Resource Overhead Disclosures
st.markdown("### 💰 Financial Cost & Overhead Telemetry")
st.markdown(
    """
    - **Local Hardware Infrastructure (Qdrant + Neo4j + Ollama):** $0.00 cloud inference fees.
    - **Cloud Provider Charges:** None. Self-hosted on local containerized instances.
    - **Storage Overhead:**
      - Vector embeddings: ~1.5 KB per chunk in Qdrant (384 float32 dimensions).
      - Knowledge Graph: Entity nodes & directed Cypher relationships in Neo4j.
    """
)
