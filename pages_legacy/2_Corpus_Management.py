from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from rag_benchmark_lab.app import get_context
from rag_benchmark_lab.schemas.documents import DocumentStatus
from rag_benchmark_lab.utils.auth import get_current_user_name, require_auth
from rag_benchmark_lab.utils.ui import inject_custom_css, render_dev_mode_banner, render_lab_header

st.set_page_config(page_title="Document Corpus — RAG Benchmark Lab", page_icon="📁", layout="wide")
inject_custom_css()

user_id = require_auth()
context = get_context()
user_name = get_current_user_name() or "User"

# Sidebar: user info, KB selector, logout
with st.sidebar:
    st.markdown(f"**👤 {user_name}**")
    kbs = context.user_registry.list_knowledge_bases(user_id)

    if kbs:
        kb_options = {kb.kb_id: kb.name for kb in kbs}
        active_kb_id = st.session_state.get("active_kb_id")
        if active_kb_id not in kb_options:
            active_kb_id = list(kb_options.keys())[0]
            st.session_state["active_kb_id"] = active_kb_id
            st.session_state["active_kb_name"] = kb_options[active_kb_id]

        selected_kb_id = st.selectbox(
            "📂 Knowledge Base",
            options=list(kb_options.keys()),
            format_func=lambda x: kb_options[x],
            index=list(kb_options.keys()).index(active_kb_id),
            key="sidebar_kb_selector",
        )
        if selected_kb_id != active_kb_id:
            st.session_state["active_kb_id"] = selected_kb_id
            st.session_state["active_kb_name"] = kb_options[selected_kb_id]
            st.rerun()
        active_kb_id = selected_kb_id
    else:
        active_kb_id = None
        st.info("Create a Knowledge Base first.")
        st.page_link("pages/0_Knowledge_Bases.py", label="→ Knowledge Bases", icon="🗂️")

    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        from rag_benchmark_lab.utils.auth import logout_user
        logout_user()
        st.rerun()

render_lab_header(
    title="Document Corpus Management",
    subtitle="Upload documents and build parallel vector + graph indexes. Documents are scoped to the selected knowledge base.",
    badge_text="INGESTION PIPELINE",
)

render_dev_mode_banner(context.settings.llm_mock_mode)

if not active_kb_id:
    st.warning("⚠️ No knowledge base selected. Please create a knowledge base first.")
    st.page_link("pages/0_Knowledge_Bases.py", label="→ Create Knowledge Base", icon="🗂️")
    st.stop()

kb = context.user_registry.get_knowledge_base(active_kb_id, user_id)
active_kb_name = kb.name if kb else "Unknown"
st.markdown(f"**Active Knowledge Base:** 🗂️ `{active_kb_name}`")

# ─── Upload Section ─────────────────────────────────────────────────────────────
st.markdown("### 📥 Document Upload & Parallel Indexing")
st.markdown("""
Uploaded files are processed through the dual-indexing pipeline:
- **Vector Pipeline:** Text Cleaning → Sliding Window Chunking → Embeddings → **Qdrant**
- **Graph Pipeline:** Chunks → Entity/Relation Extraction → Schema Validation → **Neo4j**
""")

uploaded_files = st.file_uploader(
    "Choose documents to ingest (PDF, TXT, DOCX)",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
    help="Upload research papers, technical reports, or domain documents.",
)

if uploaded_files:
    if st.button("🚀 Ingest & Index Uploaded Documents", type="primary"):
        progress_bar = st.progress(0)
        status_box = st.empty()

        for idx, uploaded in enumerate(uploaded_files):
            file_bytes = uploaded.getvalue()
            doc_name = uploaded.name
            status_box.info(f"Processing `{doc_name}` ({len(file_bytes)} bytes)...")

            try:
                outcome = context.document_service.ingest_bytes(doc_name, file_bytes)
                # Associate this document with the active knowledge base
                context.user_registry.add_document_to_kb(active_kb_id, outcome.document.document_id)

                if outcome.errors:
                    st.error(f"❌ **{doc_name}**: Errors during indexing: {'; '.join(outcome.errors)}")
                elif "duplicate_document_skipped" in outcome.warnings:
                    # Still associate with KB even if duplicate
                    st.warning(f"⚠️ **{doc_name}**: Document already exists in corpus. Associated with this KB.")
                else:
                    st.success(
                        f"✅ **{doc_name}** indexed! "
                        f"({len(outcome.chunks)} chunks → Qdrant, {outcome.entity_count} entities, {outcome.relationship_count} relationships → Neo4j)"
                    )
            except Exception as exc:
                st.error(f"❌ Failed to ingest {doc_name}: {exc}")

            progress_bar.progress((idx + 1) / len(uploaded_files))
        status_box.empty()
        st.rerun()

st.markdown("---")

# ─── Documents in This KB ─────────────────────────────────────────────────────
st.markdown(f"### 📚 Documents in '{active_kb_name}'")

# Get document IDs scoped to this KB and user
kb_doc_ids = context.user_registry.get_kb_document_ids(active_kb_id, user_id)
all_documents = context.registry.list_documents()
# Filter to only show documents belonging to this KB
documents = [d for d in all_documents if d["document_id"] in set(kb_doc_ids)]

if documents:
    df_rows = []
    for d in documents:
        df_rows.append({
            "Document ID": d["document_id"][:12] + "...",
            "Name": d["document_name"],
            "MIME Type": d.get("mime_type", "text/plain"),
            "File Size (KB)": round(float(d.get("file_size") or 0) / 1024, 1),
            "Status": d["status"].upper(),
            "Chunks (Qdrant)": d.get("chunk_count", 0),
            "Entities (Neo4j)": d.get("entity_count", 0),
            "Relations (Neo4j)": d.get("relationship_count", 0),
            "Errors": d.get("error_message") or "None",
        })
    st.dataframe(pd.DataFrame(df_rows), use_container_width=True)
else:
    st.info(f"No documents in '{active_kb_name}' yet. Upload documents above.")

# ─── Document Inspection & Actions ─────────────────────────────────────────────
st.markdown("### ⚙️ Document Inspection & Actions")
if documents:
    doc_options = {d["document_id"]: f"{d['document_name']} ({d['document_id'][:8]}...)" for d in documents}
    selected_id = st.selectbox(
        "Select Document to Inspect / Manage",
        options=list(doc_options.keys()),
        format_func=lambda x: doc_options[x],
    )

    selected_doc = next(d for d in documents if d["document_id"] == selected_id)

    col_meta, col_actions = st.columns([2, 1])

    with col_meta:
        st.markdown(f"**Document Name:** `{selected_doc['document_name']}`")
        st.markdown(f"**Full Document ID:** `{selected_doc['document_id']}`")
        st.markdown(f"**Content SHA-256 Hash:** `{selected_doc['content_hash']}`")
        status_val = selected_doc["status"].upper()
        status_color = {"INDEXED": "✅", "PARTIAL": "⚠️", "FAILED": "❌", "PENDING": "⏳"}.get(status_val, "•")
        st.markdown(f"**Status:** {status_color} `{status_val}`")
        if selected_doc.get("error_message"):
            st.error(f"Error: {selected_doc['error_message']}")

    with col_actions:
        st.markdown("**Management Actions:**")
        act_col1, act_col2 = st.columns(2)
        with act_col1:
            if st.button("🔄 Re-Index", help="Re-chunks and re-indexes into Qdrant & Neo4j"):
                src_path = selected_doc.get("source_path")
                if src_path and Path(src_path).exists():
                    with st.spinner("Re-indexing document..."):
                        outcome = context.document_service.ingest_path(src_path)
                        context.user_registry.add_document_to_kb(active_kb_id, outcome.document.document_id)
                        st.success(f"Re-indexed `{outcome.document.document_name}`")
                        st.rerun()
                else:
                    st.warning("Original source file not available for re-read.")
        with act_col2:
            if st.button("🗑️ Delete", help="Removes document from this KB"):
                with st.spinner("Deleting document..."):
                    context.document_service.delete_document(selected_id)
                    context.user_registry.remove_document_from_kb(active_kb_id, selected_id)
                    st.success("Document deleted from this knowledge base.")
                    st.rerun()

    # Chunks inspection tabs
    st.markdown("#### 🔍 Extracted Chunks & Evidence")
    chunks = context.registry.list_chunks(selected_id)
    if chunks:
        tab_chunks, tab_entities = st.tabs([f"Chunks ({len(chunks)})", "Graph Entities"])
        with tab_chunks:
            for idx, chunk in enumerate(chunks):
                with st.expander(f"Chunk #{chunk['chunk_index']} — ID: {chunk['chunk_id'][:12]}... (Page {chunk.get('page_number') or 1})"):
                    st.text(chunk["text"])
        with tab_entities:
            all_entities = context.registry.list_entities()
            doc_entities = [e for e in all_entities if e.get("document_id") == selected_id]
            if doc_entities:
                st.dataframe(pd.DataFrame(doc_entities), use_container_width=True)
            else:
                st.info("No entities recorded for this document.")
    else:
        st.info("No chunks found for this document.")

st.markdown("---")
# Danger Zone — only clears this KB's documents
with st.expander("⚠️ Danger Zone: Remove All Documents from This KB"):
    st.markdown(f"Removing all documents from **{active_kb_name}** will delete their vector records, graph entities, and metadata.")
    confirm_box = st.checkbox(f"I confirm I want to remove all documents from '{active_kb_name}'", key="clear_kb_confirm")
    if st.button("🚨 Remove All Documents from This KB", disabled=not confirm_box):
        with st.spinner("Removing all documents..."):
            doc_ids = context.user_registry.get_kb_document_ids(active_kb_id, user_id)
            for doc_id in doc_ids:
                try:
                    context.document_service.delete_document(doc_id)
                    context.user_registry.remove_document_from_kb(active_kb_id, doc_id)
                except Exception:
                    pass
            st.success(f"All documents removed from '{active_kb_name}'.")
            st.rerun()
