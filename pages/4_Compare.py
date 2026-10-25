"""RAG Benchmark Lab — Compare Retrieval Methods Page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from rag_benchmark_lab.app import get_context
from rag_benchmark_lab.schemas.documents import RetrievalStrategy
from rag_benchmark_lab.utils.auth import get_current_user_name, require_auth
from rag_benchmark_lab.utils.ui import (
    inject_custom_css,
    render_dev_mode_banner,
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

render_dev_mode_banner(context.settings.llm_mock_mode)

# ── Page Header ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-bottom: 20px;">
      <h1 style="font-size: 26px; font-weight: 800; letter-spacing: -0.03em; color: #F5F7FA; margin: 0 0 4px 0;">
        Compare Retrieval Methods
      </h1>
      <p style="font-size: 14px; color: #A0AABC; margin: 0;">
        Ask the same question across all three methods and compare the evidence they retrieve.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

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

kb_col, spacer = st.columns([2, 3], vertical_alignment="center")
with kb_col:
    selected_kb = st.selectbox(
        "Active Knowledge Base",
        options=list(kb_options.keys()),
        format_func=lambda x: kb_options[x],
        index=list(kb_options.keys()).index(active_kb_id),
        key="comp_kb_sel",
        label_visibility="collapsed",
    )
    if selected_kb != active_kb_id:
        st.session_state["active_kb_id"] = selected_kb
        st.session_state["active_kb_name"] = kb_options[selected_kb]
        st.rerun()
    active_kb_id = selected_kb
    active_kb_name = kb_options[active_kb_id]

kb_doc_ids = context.user_registry.get_kb_document_ids(active_kb_id, user_id)
if not kb_doc_ids:
    st.info(f"No documents uploaded to '{active_kb_name}'. Upload documents before running comparisons.")
    if st.button("Upload Documents", key="comp_upload_btn"):
        st.switch_page("pages/2_Documents.py")
    st.stop()

st.caption(f"Active Knowledge Base: **{active_kb_name}** ({len(kb_doc_ids)} documents available)")
st.markdown("<hr style='border-color: #293244; margin: 16px 0;'/>", unsafe_allow_html=True)

# ── Question Input & Sample Questions ─────────────────────────────────────────
st.markdown("### Enter Question")

SAMPLE_QUESTIONS = [
    "What are the main findings or conclusions in these documents?",
    "Explain the key concepts and how they relate to each other.",
    "What entities (technologies, components, frameworks) are mentioned?",
]

if "comp_question" not in st.session_state:
    st.session_state.comp_question = ""

sug_cols = st.columns(len(SAMPLE_QUESTIONS))
for idx, q in enumerate(SAMPLE_QUESTIONS):
    with sug_cols[idx]:
        if st.button(q, key=f"comp_sug_{idx}", use_container_width=True):
            st.session_state.comp_question = q

question_text = st.text_area(
    "Question:",
    value=st.session_state.comp_question,
    height=80,
    placeholder="Ask a question to compare how Vector RAG, GraphRAG, and Hybrid RAG answer it...",
    key="comp_question_input",
    label_visibility="collapsed",
)

run_btn = st.button(
    "Compare All Methods",
    type="primary",
    disabled=not question_text.strip(),
    key="btn_run_compare",
)

# ── Comparison Execution & 3 Results Panels ──────────────────────────────────
if run_btn and question_text.strip():
    st.session_state.comp_question = question_text.strip()

    with st.spinner("Executing Vector RAG, GraphRAG, and Hybrid RAG simultaneously..."):
        results = context.query_service.compare(
            question_text.strip(),
            document_ids=kb_doc_ids if kb_doc_ids else None,
        )

    st.markdown("<hr style='border-color: #293244; margin: 24px 0;'/>", unsafe_allow_html=True)
    st.markdown("### Side-by-Side Results")

    PANELS = [
        (RetrievalStrategy.VECTOR.value, "Vector RAG", "comp-panel-vector", "#60A5FA"),
        (RetrievalStrategy.GRAPH.value, "GraphRAG", "comp-panel-graph", "#A78BFA"),
        (RetrievalStrategy.HYBRID.value, "Hybrid RAG", "comp-panel-hybrid", "#34D399"),
    ]

    panel_cols = st.columns(3)
    summary_rows = []

    for col, (strat_key, strat_label, panel_cls, accent_color) in zip(panel_cols, PANELS):
        res = results.get(strat_key)
        with col:
            badge_html = render_strategy_badge(strat_key)
            st.markdown(
                f"""
                <div class="rag-card" style="border-top: 3px solid {accent_color}; margin-bottom: 16px;">
                  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 15px; font-weight: 700; color: #F5F7FA;">{strat_label}</span>
                    {badge_html}
                  </div>
                """,
                unsafe_allow_html=True,
            )

            if res is None:
                st.error("Execution failed for this strategy.")
                st.markdown("</div>", unsafe_allow_html=True)
                summary_rows.append(
                    {
                        "Strategy": strat_label,
                        "Status": "Failed",
                        "Evidence Count": 0,
                        "Citations": 0,
                        "Retrieval Latency (ms)": "—",
                        "Total Latency (ms)": "—",
                    }
                )
                continue

            if res.errors:
                st.error(f"Error: {'; '.join(res.errors)}")

            # Groundedness notice
            answer_display = res.answer or "No answer generated."
            if not res.citations and not (res.graph_evidence and res.graph_evidence.entities):
                answer_display = "The uploaded documents do not contain sufficient evidence to answer this question."

            st.markdown(
                f"""
                <div style="background: #1B2230; border: 1px solid #293244; border-radius: 8px; padding: 12px 14px; font-size: 13.5px; line-height: 1.6; color: #F5F7FA; min-height: 120px; max-height: 240px; overflow-y: auto;">
                  {answer_display}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Performance telemetry
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            with m1:
                ret_ms = f"{res.retrieval_latency_ms:.0f} ms" if res.retrieval_latency_ms else "—"
                st.caption(f"Retrieval: **{ret_ms}**")
                gen_ms = f"{res.generation_latency_ms:.0f} ms" if res.generation_latency_ms else "—"
                st.caption(f"Generation: **{gen_ms}**")
            with m2:
                tot_ms = f"{res.total_latency_ms:.0f} ms" if res.total_latency_ms else "—"
                st.caption(f"Total: **{tot_ms}**")
                st.caption(f"Evidence: **{len(res.retrieved_sources)}** passages")

            st.markdown("</div>", unsafe_allow_html=True)

            # Evidence expander
            if res.citations or (res.graph_evidence and res.graph_evidence.relationships):
                with st.expander(f"View Evidence ({len(res.citations)} citations)"):
                    if res.citations:
                        st.markdown("**Document Citations:**")
                        for c in res.citations[:6]:
                            score_str = f" · Score: {c.score:.3f}" if c.score is not None else ""
                            st.caption(f"• **{c.document_name}** [Chunk #{c.chunk_index}]{score_str}")
                    if res.graph_evidence and res.graph_evidence.relationships:
                        st.markdown("**Graph Paths:**")
                        for r in res.graph_evidence.relationships[:5]:
                            st.caption(f"• {r.source_entity_name} → {r.relationship_type} → {r.target_entity_name}")

            summary_rows.append(
                {
                    "Strategy": strat_label,
                    "Status": "Available" if not res.errors else "Partial",
                    "Evidence Count": len(res.retrieved_sources),
                    "Citations": len(res.citations),
                    "Graph Paths": len(res.graph_evidence.relationships) if res.graph_evidence else 0,
                    "Retrieval Latency (ms)": f"{res.retrieval_latency_ms:.0f}" if res.retrieval_latency_ms else "—",
                    "Generation Latency (ms)": f"{res.generation_latency_ms:.0f}" if res.generation_latency_ms else "—",
                    "Total Latency (ms)": f"{res.total_latency_ms:.0f}" if res.total_latency_ms else "—",
                }
            )

    # ── Summary Comparison Table ─────────────────────────────────────────────
    st.markdown("<hr style='border-color: #293244; margin: 24px 0;'/>", unsafe_allow_html=True)
    st.markdown("### Comparison Summary")

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

    st.caption(
        "Note: Vector RAG finds semantically similar text passages, GraphRAG finds relational multi-hop connections, "
        "and Hybrid RAG fuses both using Reciprocal Rank Fusion."
    )
