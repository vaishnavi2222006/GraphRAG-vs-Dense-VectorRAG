from __future__ import annotations

from dataclasses import asdict
import pandas as pd
import streamlit as st

from rag_benchmark_lab.app import get_context
from rag_benchmark_lab.schemas.documents import RetrievalStrategy
from rag_benchmark_lab.utils.auth import get_current_user_name, require_auth
from rag_benchmark_lab.utils.ui import (
    format_latency,
    format_metric,
    inject_custom_css,
    render_dev_mode_banner,
    render_lab_header,
    render_strategy_badge,
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
            key="comp_kb_selector",
        )
        if selected_kb_id != active_kb_id:
            st.session_state["active_kb_id"] = selected_kb_id
            st.rerun()
        active_kb_id = selected_kb_id
    else:
        active_kb_id = None
        st.info("No knowledge bases yet.")
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True, key="comp_logout"):
        from rag_benchmark_lab.utils.auth import logout_user
        logout_user()
        st.rerun()

render_lab_header(
    title="3-Way Strategy Comparison",
    subtitle="Simultaneous side-by-side evaluation of Dense Vector RAG, GraphRAG, and Hybrid RAG on the identical query.",
    badge_text="COMPARATIVE EVALUATION",
)

render_dev_mode_banner(context.settings.llm_mock_mode)

# Suggested comparison questions
st.markdown("##### 💡 Suggested Comparative Questions")
sample_comp_questions = [
    "How do Qdrant and Neo4j contribute to the RAG Benchmark Lab architecture?",
    "Explain how Hybrid RAG optimizes retrieval using Reciprocal Rank Fusion.",
    "What components and metrics are used to evaluate retrieval and answer quality?",
]

comp_cols = st.columns(len(sample_comp_questions))
if "comp_question" not in st.session_state:
    st.session_state.comp_question = sample_comp_questions[0]

for idx, q_txt in enumerate(sample_comp_questions):
    with comp_cols[idx]:
        if st.button(f"🔍 Comparative Query {idx+1}", help=q_txt, key=f"csq_{idx}"):
            st.session_state.comp_question = q_txt

# Question Input Box
question_text = st.text_area(
    "Enter a question to evaluate across all three retrieval architectures:",
    value=st.session_state.comp_question,
    height=90,
    key="comp_question_input",
)

comp_btn = st.button("⚖️ Run Simultaneous 3-Way Comparison", type="primary", use_container_width=True)

if comp_btn and question_text.strip():
    kb_doc_ids = context.user_registry.get_kb_document_ids(active_kb_id, user_id) if active_kb_id else None
    with st.spinner("Executing Vector RAG (Qdrant), GraphRAG (Neo4j), and Hybrid RAG (RRF)..."):
        results = context.query_service.compare(question_text.strip(), document_ids=kb_doc_ids)

    st.markdown("---")
    st.markdown("### 🔬 Side-by-Side Execution Panels")


    columns = st.columns(3)
    strategies = [
        (RetrievalStrategy.VECTOR.value, "Dense Vector RAG (Qdrant)", "⚡"),
        (RetrievalStrategy.GRAPH.value, "GraphRAG (Neo4j)", "🕸️"),
        (RetrievalStrategy.HYBRID.value, "Hybrid RAG (RRF Fusion)", "🔀"),
    ]

    summary_rows = []

    for idx, (strat_key, strat_title, icon) in enumerate(strategies):
        res = results.get(strat_key)
        with columns[idx]:
            st.markdown(
                f"""
                <div style="background: #ffffff; border: 1px solid #cbd5e1; border-top: 4px solid {'#0284c7' if idx==0 else '#059669' if idx==1 else '#7c3aed'}; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <h4 style="margin: 0; font-size: 15px; color: #0f172a;">{icon} {strat_title}</h4>
                        {render_strategy_badge(strat_key)}
                    </div>
                """,
                unsafe_allow_html=True,
            )

            if res is None:
                st.error("Strategy execution failed.")
                continue

            # Answer
            st.markdown("**Generated Answer:**")
            st.markdown(
                f"""
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; font-size: 13.5px; line-height: 1.55; color: #1e293b; min-height: 180px; max-height: 280px; overflow-y: auto;">
{res.answer}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Telemetry Metrics
            st.markdown("**Telemetry:**")
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.caption(f"⏱️ Retrieval: **{format_latency(res.retrieval_latency_ms, fallback='N/A', precision=1)}**")
                st.caption(f"⚡ Generation: **{format_latency(res.generation_latency_ms, fallback='N/A', precision=1, treat_zero_as_unavailable=True)}**")
            with mcol2:
                st.caption(f"⌛ Total: **{format_latency(res.total_latency_ms, fallback='N/A', precision=1, treat_zero_as_unavailable=True)}**")
                st.caption(f"📦 Passages: **{len(res.retrieved_sources)}**")

            # Supporting Citations
            if res.citations:
                with st.expander(f"Citations ({len(res.citations)})"):
                    for c in res.citations:
                        st.caption(f"• **{c.document_name}** [Chunk #{c.chunk_index}] (Score: {c.score:.3f if c.score else 0.0})")
            else:
                st.caption("No citations retrieved.")

            # Graph Evidence if Graph or Hybrid
            if res.graph_evidence and (res.graph_evidence.entities or res.graph_evidence.relationships):
                with st.expander(f"Graph Evidence ({len(res.graph_evidence.relationships)} rels)"):
                    for r in res.graph_evidence.relationships[:4]:
                        st.caption(f"• {r.source_entity_name} &rarr; *{r.relationship_type}* &rarr; {r.target_entity_name}")

            # Warnings / Errors
            if res.warnings:
                st.caption(f"⚠️ Warnings: {', '.join(res.warnings)}")
            if res.errors:
                st.error(f"❌ Errors: {', '.join(res.errors)}")

            st.markdown("</div>", unsafe_allow_html=True)

            summary_rows.append(
                {
                    "Strategy": strat_title,
                    "Retrieval Latency (ms)": round(res.retrieval_latency_ms or 0, 1),
                    "Generation Latency (ms)": round(res.generation_latency_ms or 0, 1),
                    "Total Latency (ms)": round(res.total_latency_ms or 0, 1),
                    "Retrieved Passages": len(res.retrieved_sources),
                    "Graph Relationships": len(res.graph_evidence.relationships) if res.graph_evidence else 0,
                    "Tokens (In/Out)": f"{res.prompt_tokens or 0} / {res.completion_tokens or 0}",
                    "Status": "PASS" if not res.errors else "FAIL",
                }
            )

    # Comparative Summary Table
    st.markdown("### 📊 Quantitative Comparison Summary")
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    st.markdown(
        """
        > **Research Note:** No single strategy is uniformly superior across all query types. 
        > Dense Vector RAG excels at localized semantic similarity; GraphRAG captures cross-document relational paths; 
        > and Hybrid RAG with RRF optimizes recall by bridging structural and semantic dimensions.
        """
    )
