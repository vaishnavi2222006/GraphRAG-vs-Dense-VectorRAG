"""RAG Benchmark Lab — Evaluation & Performance Benchmarking Page."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rag_benchmark_lab.app import get_context
from rag_benchmark_lab.evaluation.benchmark_runner import BenchmarkRunner
from rag_benchmark_lab.evaluation.dataset import load_dataset, sample_dataset
from rag_benchmark_lab.schemas.documents import RetrievalStrategy
from rag_benchmark_lab.utils.auth import get_current_user_name, require_auth
from rag_benchmark_lab.utils.ui import (
    inject_custom_css,
    render_dev_mode_banner,
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

# ── Page Header ──────────────────────────────────────────────────────────────
top_c1, top_c2 = st.columns([3, 1], vertical_alignment="center")
with top_c1:
    st.markdown(
        """
        <div style="margin-bottom: 8px;">
          <h1 style="font-size: 26px; font-weight: 800; color: #F5F7FA; margin: 0 0 4px 0; letter-spacing: -0.03em;">
            Evaluate RAG Performance
          </h1>
          <p style="font-size: 14px; color: #A0AABC; margin: 0;">
            Measure retrieval quality, evidence support, and response performance.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with top_c2:
    active_kb_id = render_kb_selector(context, user_id, key_prefix="eval")

render_dev_mode_banner(context.settings.llm_mock_mode)

# ── Metric Definitions Helper ────────────────────────────────────────────────
with st.expander("Understanding Evaluation Metrics", expanded=False):
    st.markdown(
        """
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; font-size: 13px;">
          <div style="background: #1B2230; padding: 12px 14px; border-radius: 8px; border: 1px solid #293244;">
            <strong style="color: #8B7CFF;">Faithfulness (0–1.0)</strong><br/>
            Measures whether all claims in the generated answer are grounded in the retrieved source passages without hallucination.
          </div>
          <div style="background: #1B2230; padding: 12px 14px; border-radius: 8px; border: 1px solid #293244;">
            <strong style="color: #39C6B5;">Answer Relevancy (0–1.0)</strong><br/>
            Assesses how directly and completely the generated answer addresses the original query.
          </div>
          <div style="background: #1B2230; padding: 12px 14px; border-radius: 8px; border: 1px solid #293244;">
            <strong style="color: #60A5FA;">Recall@K & MRR</strong><br/>
            Measures the proportion of ground-truth evidence chunks retrieved within the top K results and the Mean Reciprocal Rank.
          </div>
          <div style="background: #1B2230; padding: 12px 14px; border-radius: 8px; border: 1px solid #293244;">
            <strong style="color: #F5B942;">Context Precision & Recall</strong><br/>
            Evaluates signal-to-noise ratio in retrieved context and whether all necessary facts were discovered.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# ── Workflow Indicator ───────────────────────────────────────────────────────
st.markdown(
    """
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 24px; font-size: 12px;">
      <div style="background: #151A24; border: 1px solid #293244; border-radius: 6px; padding: 8px 12px;">
        <span style="color: #8B7CFF; font-weight: 700;">1.</span> Select Dataset
      </div>
      <div style="background: #151A24; border: 1px solid #293244; border-radius: 6px; padding: 8px 12px;">
        <span style="color: #8B7CFF; font-weight: 700;">2.</span> Choose Methods
      </div>
      <div style="background: #151A24; border: 1px solid #293244; border-radius: 6px; padding: 8px 12px;">
        <span style="color: #8B7CFF; font-weight: 700;">3.</span> Run Evaluation
      </div>
      <div style="background: #151A24; border: 1px solid #293244; border-radius: 6px; padding: 8px 12px;">
        <span style="color: #8B7CFF; font-weight: 700;">4.</span> Inspect Results
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════════════════════
# 1. BENCHMARK DATASET
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("### 1. Select Benchmark Dataset")
st.caption(
    "Benchmark questions and reference answers are used as ground-truth to objectively evaluate and score generated answers."
)

col_ds1, col_ds2 = st.columns([1, 2])
with col_ds1:
    dataset_source = st.radio(
        "Dataset Source",
        options=["Built-in Benchmark Dataset", "Upload Custom JSON/JSONL"],
        index=0,
        key="eval_ds_source",
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
            key="eval_uploader",
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

st.markdown("<hr style='border-color: #293244; margin: 20px 0;'/>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# 2. CHOOSE METHODS & CONFIGURE RUN
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("### 2. Choose Retrieval Methods")
st.caption("Select the retrieval strategies to benchmark and configure query limits.")

cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

with cfg_col1:
    selected_strategies = st.multiselect(
        "Retrieval Methods to Benchmark",
        options=["VECTOR", "GRAPH", "HYBRID"],
        default=["VECTOR", "GRAPH", "HYBRID"],
        format_func=lambda s: f"{s.capitalize()} RAG",
        key="eval_strats",
    )

with cfg_col2:
    max_q = max(1, len(examples))
    query_limit = st.slider(
        "Questions to Evaluate",
        min_value=1,
        max_value=max_q,
        value=min(5, max_q),
        key="eval_query_limit",
    )

with cfg_col3:
    st.markdown("**Execution Environment:**")
    st.caption(f"• LLM Mode: **{'Offline Dev Synthesis' if context.settings.llm_mock_mode else 'Ollama Real'}**")
    st.caption("• Vector Database: **Qdrant**")
    st.caption("• Graph Database: **Neo4j**")

# ═════════════════════════════════════════════════════════════════════════════
# 3. RUN EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("<hr style='border-color: #293244; margin: 20px 0;'/>", unsafe_allow_html=True)
st.markdown("### 3. Run Evaluation")

eval_disabled = not examples or not selected_strategies
if st.session_state.get("eval_running", False):
    eval_disabled = True

start_eval = st.button(
    "Run Evaluation",
    type="primary",
    disabled=eval_disabled,
    key="btn_run_eval",
)

if start_eval:
    st.session_state["eval_running"] = True
    active_examples = examples[:query_limit]
    active_strat_enums = [RetrievalStrategy[s] for s in selected_strategies]

    prog_bar = st.progress(0.0)
    status_msg = st.empty()

    runner = BenchmarkRunner(context.query_service, context.result_store)

    with st.spinner("Executing benchmark across selected strategies..."):
        try:
            run_output = runner.run(active_examples, strategies=active_strat_enums)
            prog_bar.progress(1.0)
            status_msg.success(f"Benchmark run completed. Persisted Run ID: `{run_output['run_id'][:12]}`")
            st.session_state["latest_run_output"] = run_output
        except Exception as exc:
            status_msg.error(f"Evaluation error: {exc}")
        finally:
            st.session_state["eval_running"] = False

# ═════════════════════════════════════════════════════════════════════════════
# 4. INSPECT RESULTS & ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("<hr style='border-color: #293244; margin: 20px 0;'/>", unsafe_allow_html=True)
st.markdown("### 4. Inspect Results")
st.caption("Analyze metric cards, comparative quality charts, per-question inspector, and run history.")

all_results = context.result_store.list_results()
all_runs = context.result_store.list_runs()

if not all_results:
    st.info("No evaluation results recorded yet. Click 'Run Evaluation' above to generate empirical data.")
else:
    rows = []
    for res in all_results:
        if not isinstance(res, dict):
            continue
        mets = res.get("metrics") or {}
        rows.append(
            {
                "run_id": res.get("run_id", "unknown"),
                "query_id": res.get("query_id", "unknown"),
                "strategy": res.get("strategy", "unknown"),
                "question": res.get("question", ""),
                "answer": res.get("answer", ""),
                "expected_answer": res.get("expected_answer", ""),
                "retrieval_latency_ms": res.get("retrieval_latency_ms") or 0.0,
                "generation_latency_ms": res.get("generation_latency_ms") or 0.0,
                "total_latency_ms": res.get("total_latency_ms") or 0.0,
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
                "mock_mode": res.get("mock_mode", False),
            }
        )

    df_all = pd.DataFrame(rows)

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        run_options = ["All Runs"] + sorted(list(df_all["run_id"].unique()), reverse=True)
        selected_run = st.selectbox("Filter by Run ID", options=run_options, index=0)
    with fcol2:
        strat_options = ["All Strategies"] + sorted(list(df_all["strategy"].unique()))
        selected_strat = st.selectbox("Filter by Strategy", options=strat_options, index=0)
    with fcol3:
        mode_filter = st.selectbox("Execution Mode", options=["All", "Real LLM Only", "Dev Mode Only"], index=0)

    df = df_all.copy()
    if selected_run != "All Runs":
        df = df[df["run_id"] == selected_run]
    if selected_strat != "All Strategies":
        df = df[df["strategy"] == selected_strat]
    if mode_filter == "Real LLM Only":
        df = df[df["mock_mode"] == False]  # noqa: E712
    elif mode_filter == "Dev Mode Only":
        df = df[df["mock_mode"] == True]  # noqa: E712

    if df.empty:
        st.warning("No records match the current filter selection.")
    else:
        # KPI Cards
        kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
        with kpi_c1:
            st.metric("Queries Analyzed", len(df))
        with kpi_c2:
            m_f = df["faithfulness"].dropna().mean() if not df["faithfulness"].dropna().empty else None
            st.metric("Avg Faithfulness", f"{m_f:.3f}" if m_f is not None else "N/A")
        with kpi_c3:
            m_r = df["answer_relevancy"].dropna().mean() if not df["answer_relevancy"].dropna().empty else None
            st.metric("Avg Relevancy", f"{m_r:.3f}" if m_r is not None else "N/A")
        with kpi_c4:
            m_rec = df["retrieval_recall_at_k"].dropna().mean() if not df["retrieval_recall_at_k"].dropna().empty else None
            st.metric("Avg Recall@K", f"{m_rec:.3f}" if m_rec is not None else "N/A")
        with kpi_c5:
            m_lat = df["total_latency_ms"].mean() if not df["total_latency_ms"].empty else None
            st.metric("Mean Latency", f"{m_lat:.1f} ms" if m_lat is not None else "N/A")

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        tab_aggs, tab_charts, tab_inspector, tab_history = st.tabs(
            ["Aggregate Summary", "Comparative Charts", "Per-Question Inspector", "Run History"]
        )

        with tab_aggs:
            st.markdown("#### Strategy Comparison Aggregates")
            group_cols = [
                "faithfulness", "answer_relevancy", "context_precision", "context_recall",
                "retrieval_recall_at_k", "mrr", "total_latency_ms", "retrieval_latency_ms",
            ]
            valid_cols = [c for c in group_cols if c in df.columns]
            agg_summary = df.groupby("strategy")[valid_cols].mean().reset_index()
            agg_display = agg_summary.rename(
                columns={
                    "strategy": "Strategy",
                    "faithfulness": "Faithfulness",
                    "answer_relevancy": "Answer Relevancy",
                    "context_precision": "Context Precision",
                    "context_recall": "Context Recall",
                    "retrieval_recall_at_k": "Recall@K",
                    "mrr": "MRR",
                    "total_latency_ms": "Total Latency (ms)",
                    "retrieval_latency_ms": "Retrieval Latency (ms)",
                }
            )
            st.dataframe(agg_display.round(3), use_container_width=True)

        with tab_charts:
            dark_layout = dict(
                plot_bgcolor="#151A24",
                paper_bgcolor="#151A24",
                font=dict(color="#F5F7FA", family="Inter, sans-serif"),
                xaxis=dict(gridcolor="#293244", zerolinecolor="#293244"),
                yaxis=dict(gridcolor="#293244", zerolinecolor="#293244"),
                margin=dict(l=20, r=20, t=40, b=20),
            )
            strat_colors = {"VECTOR": "#60A5FA", "GRAPH": "#A78BFA", "HYBRID": "#34D399"}

            strat_group = df.groupby("strategy")[
                ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "retrieval_recall_at_k", "mrr"]
            ].mean().reset_index()

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
                    title="Quality Metrics Across Retrieval Architectures",
                    color_discrete_map=strat_colors,
                    height=400,
                )
                fig_bar.update_layout(**dark_layout)
                fig_bar.update_layout(yaxis=dict(range=[0, 1.05], title="Score (0.0 to 1.0)"))
                st.plotly_chart(fig_bar, use_container_width=True)

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                fig_box = px.box(
                    df,
                    x="strategy",
                    y="total_latency_ms",
                    color="strategy",
                    points="all",
                    title="Total Latency Distribution (ms)",
                    color_discrete_map=strat_colors,
                    height=360,
                )
                fig_box.update_layout(**dark_layout)
                st.plotly_chart(fig_box, use_container_width=True)

            with col_c2:
                df_lat = df.groupby("strategy")[["retrieval_latency_ms", "generation_latency_ms"]].mean().reset_index()
                fig_stacked = go.Figure(
                    data=[
                        go.Bar(name="Retrieval Latency", x=df_lat["strategy"], y=df_lat["retrieval_latency_ms"], marker_color="#39C6B5"),
                        go.Bar(name="Generation Latency", x=df_lat["strategy"], y=df_lat["generation_latency_ms"], marker_color="#8B7CFF"),
                    ]
                )
                fig_stacked.update_layout(
                    barmode="stack",
                    title="Retrieval vs. Generation Latency Breakdown (Mean ms)",
                    height=360,
                    **dark_layout,
                )
                st.plotly_chart(fig_stacked, use_container_width=True)

        with tab_inspector:
            st.markdown("#### Per-Question Inspection")
            for idx, r in df.iterrows():
                with st.expander(f"Question: {r['question'][:75]}... [{r['strategy']}]", expanded=False):
                    c_i1, c_i2 = st.columns(2)
                    with c_i1:
                        st.markdown("**User Question:**")
                        st.markdown(f"> {r['question']}")
                        if r.get("expected_answer"):
                            st.markdown("**Expected Reference Answer:**")
                            st.markdown(f"> {r['expected_answer']}")
                    with c_i2:
                        st.markdown(f"**Generated Answer ({r['strategy']}):**")
                        st.markdown(f"> {r['answer']}")

                    st.markdown("<hr style='border-color: #293244; margin: 10px 0;'/>", unsafe_allow_html=True)
                    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
                    with s_col1:
                        st.caption(f"Latency: **{r['total_latency_ms']:.1f} ms**")
                    with s_col2:
                        st.caption(f"Faithfulness: **{r['faithfulness']}**")
                    with s_col3:
                        st.caption(f"Relevancy: **{r['answer_relevancy']}**")
                    with s_col4:
                        st.caption(f"Recall@K: **{r['retrieval_recall_at_k']}**")

            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Export Results to CSV",
                data=csv_data,
                file_name="rag_benchmark_results.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_eval_csv",
            )

        with tab_history:
            st.markdown("#### Historical Benchmark Runs")
            if all_runs:
                df_runs = []
                for run in all_runs:
                    df_runs.append(
                        {
                            "Run ID": run["run_id"][:12],
                            "Timestamp": run.get("timestamp", "")[:19].replace("T", " "),
                            "Dataset": run.get("dataset_id", "default"),
                            "Total Queries": run.get("total_queries", 0),
                            "Strategies": ", ".join(run.get("strategies", [])),
                            "Mode": "Dev Mode" if run.get("mock_mode") else "Real LLM",
                        }
                    )
                st.dataframe(pd.DataFrame(df_runs), use_container_width=True)
            else:
                st.info("No recorded historical runs.")

st.markdown("<hr style='border-color: #293244; margin: 24px 0;'/>", unsafe_allow_html=True)

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
    neo4j_err = None
    try:
        context.neo4j_store.health()
        neo4j_available = True
        total_nodes = context.neo4j_store.count_entities()
        total_edges = context.neo4j_store.count_relationships()
        total_docs = context.neo4j_store.count_documents()
        source_mode = "neo4j"
    except Exception as exc:
        neo4j_available = False
        neo4j_err = str(exc)
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
        st.metric("Graph Backend", "Neo4j Active" if neo4j_available else "Registry Fallback")

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
                plot_bgcolor="#151A24",
                paper_bgcolor="#151A24",
                height=380,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
            st.plotly_chart(fig_g, use_container_width=True)
