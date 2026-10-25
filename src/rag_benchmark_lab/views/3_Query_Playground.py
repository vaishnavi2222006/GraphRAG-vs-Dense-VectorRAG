from __future__ import annotations

from dataclasses import asdict
import pandas as pd
import streamlit as st

from rag_benchmark_lab.app import get_context
from rag_benchmark_lab.schemas.documents import RetrievalStrategy
from rag_benchmark_lab.utils.auth import get_current_user_name, require_auth
from rag_benchmark_lab.utils.ui import (
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
            key="qp_kb_selector",
        )
        if selected_kb_id != active_kb_id:
            st.session_state["active_kb_id"] = selected_kb_id
            st.rerun()
        active_kb_id = selected_kb_id
    else:
        active_kb_id = None
        st.info("No knowledge bases yet.")
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        from rag_benchmark_lab.utils.auth import logout_user
        logout_user()
        st.rerun()

render_lab_header(
    title="Research Query Playground",
    subtitle="Interactive question answering with Vector RAG, GraphRAG, and Hybrid RAG over your indexed corpus.",
    badge_text="QUERY EXECUTION",
)

render_dev_mode_banner(context.settings.llm_mock_mode)

# Sample research questions to solve empty-state issue
st.markdown("##### 💡 Suggested Research Questions")
sample_questions = [
    "What technologies and databases does RAG Benchmark Lab use?",
    "How does Hybrid RAG combine Vector RAG and GraphRAG?",
    "When is GraphRAG more useful than Dense Vector RAG?",
    "What does the ingestion pipeline preserve across documents?",
]

sample_cols = st.columns(len(sample_questions))
if "current_question" not in st.session_state:
    st.session_state.current_question = sample_questions[0]

for idx, q_text in enumerate(sample_questions):
    with sample_cols[idx]:
        if st.button(f"🔍 Question {idx+1}", help=q_text, key=f"sq_{idx}"):
            st.session_state.current_question = q_text

# Main Query Formulation Card
with st.container():
    st.markdown("### ❓ Formulate Research Query")
    question_input = st.text_area(
        "Enter your research question:",
        value=st.session_state.current_question,
        height=100,
        placeholder="Type a domain query requiring factual, multi-hop, or relational reasoning...",
        key="qa_question_box",
    )

    conf_cols = st.columns([2, 1, 1, 2])
    with conf_cols[0]:
        strategy_option = st.selectbox(
            "Retrieval Strategy",
            options=["AUTOMATIC (Query Router)", "VECTOR (Dense Vector RAG)", "GRAPH (GraphRAG)", "HYBRID (Vector + Graph RRF)"],
            index=0,
            help="Choose an explicit retrieval pipeline or allow the explainable Query Router to classify the query.",
        )
    with conf_cols[1]:
        top_k_input = st.number_input(
            "Top-K Chunks",
            min_value=1,
            max_value=20,
            value=context.settings.default_top_k,
            help="Number of chunks retrieved for context assembly.",
        )
    with conf_cols[2]:
        graph_hops = st.number_input(
            "Graph Max Hops",
            min_value=1,
            max_value=4,
            value=context.settings.graph_max_hops,
            help="Maximum traversal radius in Neo4j knowledge graph.",
        )
    with conf_cols[3]:
        # Scope document filter to the active knowledge base only
        if active_kb_id:
            kb_doc_ids = context.user_registry.get_kb_document_ids(active_kb_id, user_id)
        else:
            kb_doc_ids = [d["document_id"] for d in context.registry.list_documents()]
        docs = context.registry.list_documents()
        doc_filter_options = {d["document_id"]: d["document_name"] for d in docs if d["document_id"] in set(kb_doc_ids)}
        selected_doc_filters = st.multiselect(
            "Filter by Document (Optional)",
            options=list(doc_filter_options.keys()),
            format_func=lambda x: doc_filter_options[x],
            help="Restrict retrieval to specific documents in your knowledge base.",
        )

    run_clicked = st.button("🚀 Execute Retrieval & Generate Answer", type="primary", use_container_width=True)

# Query Execution Pipeline
if run_clicked and question_input.strip():
    # Map selection to enum
    manual_strategy = None
    if "VECTOR" in strategy_option:
        manual_strategy = RetrievalStrategy.VECTOR
    elif "GRAPH" in strategy_option:
        manual_strategy = RetrievalStrategy.GRAPH
    elif "HYBRID" in strategy_option:
        manual_strategy = RetrievalStrategy.HYBRID

    # Scope to KB documents by default, apply any additional user filter
    base_doc_ids = kb_doc_ids if active_kb_id else None
    if selected_doc_filters:
        # Further filter within KB scope
        doc_ids = [d for d in selected_doc_filters if not base_doc_ids or d in base_doc_ids]
    elif base_doc_ids:
        doc_ids = base_doc_ids
    else:
        doc_ids = None

    with st.spinner("Executing retrieval pipeline & generating answer..."):
        result = context.query_service.answer(
            question=question_input.strip(),
            manual_strategy=manual_strategy,
            document_ids=doc_ids,
            top_k=int(top_k_input),
        )

    st.markdown("---")

    # 1. Answer Card (Primary Focus)
    st.markdown("### 📝 Grounded Answer")
    badge_html = render_strategy_badge(result.selected_strategy.value)

    st.markdown(
        f"""
        <div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 10px; padding: 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.03); margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; border-bottom: 1px solid #f1f5f9; padding-bottom: 12px;">
                <div>
                    <span style="font-size: 13px; font-weight: 600; color: #64748b; margin-right: 8px;">STRATEGY APPLIED:</span>
                    {badge_html}
                </div>
                <div style="font-size: 12px; color: #64748b;">
                    Model: <strong>{result.model_name}</strong>
                </div>
            </div>
            <div style="font-size: 15.5px; line-height: 1.65; color: #1e293b; white-space: pre-wrap;">
{result.answer}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Performance & Latency Telemetry
    st.markdown("### ⏱️ Latency & Performance Breakdown")
    lat_cols = st.columns(4)
    with lat_cols[0]:
        st.metric(
            "Retrieval Latency",
            f"{result.retrieval_latency_ms:.2f} ms" if result.retrieval_latency_ms is not None else "N/A",
            help="Time taken to search Qdrant / traverse Neo4j.",
        )
    with lat_cols[1]:
        st.metric(
            "Generation Latency",
            f"{result.generation_latency_ms:.2f} ms" if result.generation_latency_ms is not None else "N/A",
            help="Time taken by LLM to produce answer tokens.",
        )
    with lat_cols[2]:
        st.metric(
            "Total Pipeline Latency",
            f"{result.total_latency_ms:.2f} ms" if result.total_latency_ms is not None else "N/A",
        )
    with lat_cols[3]:
        st.metric(
            "Retrieved Passages",
            len(result.retrieved_sources),
            help="Supporting passages passed to generator.",
        )

    # 3. Routing Rationale (if router was used)
    with st.expander("🧭 Query Router Explainability & Decision Rationale", expanded=True):
        st.markdown(f"**Selected Strategy:** `{result.selected_strategy.value}`")
        st.markdown(f"**Routing Rationale:** {result.routing_rationale}")
        st.markdown(f"**Confidence Score:** `{result.routing_confidence:.2f}`")
        if result.metadata.get("overridden"):
            st.info("Strategy was manually overridden by user selection.")

    # 4. Source Evidence & Real Citations
    st.markdown("### 🔍 Supporting Evidence & Citations")
    if result.citations:
        df_citations = pd.DataFrame(
            [
                {
                    "Document Name": c.document_name,
                    "Document ID": c.document_id[:12] + "...",
                    "Chunk Index": f"#{c.chunk_index}",
                    "Page": c.page_number or 1,
                    "Relevance Score": f"{c.score:.4f}" if c.score else "N/A",
                }
                for c in result.citations
            ]
        )
        st.dataframe(df_citations, use_container_width=True)
    else:
        st.info("No citations recorded for this answer.")

    # 5. Retrieved Chunks & Graph Paths
    tab_chunks, tab_graph, tab_raw = st.tabs(["📄 Retrieved Text Chunks", "🕸️ Graph Paths & Triples", "📜 Full Raw Context"])

    with tab_chunks:
        if result.retrieved_sources:
            for i, chunk in enumerate(result.retrieved_sources):
                rrf_str = f" | RRF Score: {chunk.metadata.get('rrf_score', 0):.4f}" if "rrf_score" in chunk.metadata else ""
                with st.expander(f"Chunk #{i+1} — {chunk.document_name} [Chunk {chunk.chunk_id[:8]}]{rrf_str}"):
                    st.markdown(f"**Source Document:** `{chunk.document_name}` (Page {chunk.page_number or 1})")
                    st.markdown(f"**Relevance Score:** `{chunk.score:.4f}`")
                    st.text(chunk.text)
        else:
            st.info("No text chunks were retrieved.")

    with tab_graph:
        if result.graph_evidence and (result.graph_evidence.entities or result.graph_evidence.relationships):
            st.markdown(f"**Connected Graph Entities ({len(result.graph_evidence.entities)}):**")
            st.write([f"{e.canonical_name} ({e.entity_type})" for e in result.graph_evidence.entities])

            if result.graph_evidence.relationships:
                st.markdown(f"**Traversed Relationships ({len(result.graph_evidence.relationships)}):**")
                df_rel = pd.DataFrame(
                    [
                        {
                            "Source Entity": r.source_entity_name,
                            "Relationship": r.relationship_type,
                            "Target Entity": r.target_entity_name,
                            "Confidence": r.confidence,
                            "Evidence Passage": r.source_text or "N/A",
                        }
                        for r in result.graph_evidence.relationships
                    ]
                )
                st.dataframe(df_rel, use_container_width=True)
        else:
            st.info("No graph paths were traversed for this strategy.")

    with tab_raw:
        st.code(result.raw_context or "No context assembled.", language="markdown")
