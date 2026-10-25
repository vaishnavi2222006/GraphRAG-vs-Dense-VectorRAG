"""RAG Benchmark Lab — Documents Management Page."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from rag_benchmark_lab.app import get_context
from rag_benchmark_lab.utils.auth import get_current_user_name, require_auth
from rag_benchmark_lab.utils.ui import (
    inject_custom_css,
    render_dev_mode_banner,
    render_topnav,
)

st.set_page_config(
    page_title="Your Documents — RAG Benchmark Lab",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_topnav(active_page="Documents")

user_id = require_auth()
context = get_context()
user_name = get_current_user_name() or "User"

render_dev_mode_banner(context.settings.llm_mock_mode)

# ── Page Header ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-bottom: 20px;">
      <h1 style="font-size: 26px; font-weight: 800; letter-spacing: -0.03em; color: #F5F7FA; margin: 0 0 4px 0;">
        Your Documents
      </h1>
      <p style="font-size: 14px; color: #A0AABC; margin: 0;">
        Upload and manage the documents used by your RAG system.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Knowledge Base Selector & Creation ───────────────────────────────────────
kbs = context.user_registry.list_knowledge_bases(user_id)

kb_col, create_col = st.columns([3, 2], vertical_alignment="center")

with kb_col:
    if kbs:
        kb_options = {kb.kb_id: kb.name for kb in kbs}
        active_kb_id = st.session_state.get("active_kb_id")
        if active_kb_id not in kb_options:
            active_kb_id = list(kb_options.keys())[0]
            st.session_state["active_kb_id"] = active_kb_id
            st.session_state["active_kb_name"] = kb_options[active_kb_id]

        selected_kb_id = st.selectbox(
            "Active Knowledge Base",
            options=list(kb_options.keys()),
            format_func=lambda x: kb_options[x],
            index=list(kb_options.keys()).index(active_kb_id),
            key="docs_kb_selector",
        )
        if selected_kb_id != active_kb_id:
            st.session_state["active_kb_id"] = selected_kb_id
            st.session_state["active_kb_name"] = kb_options[selected_kb_id]
            st.rerun()
        active_kb_id = selected_kb_id
        active_kb_name = kb_options[active_kb_id]
    else:
        active_kb_id = None
        active_kb_name = ""
        st.info("Create a knowledge base to start uploading and managing documents.")

with create_col:
    if st.button("New Knowledge Base", use_container_width=True, key="btn_open_create_kb"):
        st.session_state["show_create_kb"] = True

# ── Create Knowledge Base Form ───────────────────────────────────────────────
if st.session_state.get("show_create_kb"):
    with st.form("create_kb_form", clear_on_submit=True):
        st.markdown("<h4 style='color:#F5F7FA;margin:0 0 12px 0;font-size:16px;'>Create Knowledge Base</h4>", unsafe_allow_html=True)
        new_kb_name = st.text_input(
            "Knowledge Base Name *",
            placeholder="e.g. Legal Research Q3",
            key="new_kb_name",
        )
        new_kb_desc = st.text_input(
            "Description (optional)",
            placeholder="e.g. Ingested regulatory guidelines and filings",
            key="new_kb_desc",
        )
        col_s, col_c = st.columns(2)
        with col_s:
            submitted = st.form_submit_button("Create", type="primary", use_container_width=True)
        with col_c:
            cancelled = st.form_submit_button("Cancel", use_container_width=True)

    if submitted:
        if not new_kb_name.strip():
            st.error("Please enter a name for the knowledge base.")
        else:
            kb = context.user_registry.create_knowledge_base(user_id, new_kb_name.strip(), new_kb_desc.strip())
            st.session_state["active_kb_id"] = kb.kb_id
            st.session_state["active_kb_name"] = kb.name
            st.session_state.pop("show_create_kb", None)
            st.success(f"Knowledge base '{kb.name}' created.")
            st.rerun()
    elif cancelled:
        st.session_state.pop("show_create_kb", None)
        st.rerun()

if not active_kb_id:
    st.stop()

# ── Manage Knowledge Bases (Rename / Delete) ─────────────────────────────────
with st.expander("Manage Knowledge Bases", expanded=False):
    for kb in kbs:
        col_info, col_ren, col_del = st.columns([4, 1, 1], vertical_alignment="center")
        with col_info:
            doc_count = kb.document_count
            st.markdown(
                f"**{kb.name}** &nbsp;·&nbsp; "
                f"<span style='color:#A0AABC;font-size:12px;'>"
                f"{doc_count} documents · Created {kb.created_at[:10]}</span>",
                unsafe_allow_html=True,
            )
        with col_ren:
            if st.button("Rename", key=f"ren_{kb.kb_id}", use_container_width=True):
                st.session_state[f"renaming_{kb.kb_id}"] = True
        with col_del:
            if st.button("Delete", key=f"del_{kb.kb_id}", use_container_width=True):
                st.session_state[f"confirm_del_{kb.kb_id}"] = True

        if st.session_state.get(f"renaming_{kb.kb_id}"):
            with st.form(f"ren_form_{kb.kb_id}"):
                new_name = st.text_input("New name", value=kb.name)
                rc, cc = st.columns(2)
                with rc:
                    if st.form_submit_button("Save", type="primary"):
                        context.user_registry.rename_knowledge_base(kb.kb_id, user_id, new_name.strip())
                        st.session_state.pop(f"renaming_{kb.kb_id}", None)
                        st.rerun()
                with cc:
                    if st.form_submit_button("Cancel"):
                        st.session_state.pop(f"renaming_{kb.kb_id}", None)
                        st.rerun()

        if st.session_state.get(f"confirm_del_{kb.kb_id}"):
            st.warning(f"Delete '{kb.name}'? All its documents, vector indexes, and graph data will be removed.")
            dc1, dc2 = st.columns(2)
            with dc1:
                if st.button("Confirm Delete", key=f"conf_del_{kb.kb_id}", type="primary"):
                    doc_ids = context.user_registry.get_kb_document_ids(kb.kb_id, user_id)
                    for doc_id in doc_ids:
                        try:
                            context.document_service.delete_document(doc_id)
                        except Exception:
                            pass
                    context.user_registry.delete_knowledge_base(kb.kb_id, user_id)
                    if st.session_state.get("active_kb_id") == kb.kb_id:
                        for k in ("active_kb_id", "active_kb_name"):
                            st.session_state.pop(k, None)
                    st.session_state.pop(f"confirm_del_{kb.kb_id}", None)
                    st.success(f"Deleted {kb.name}")
                    st.rerun()
            with dc2:
                if st.button("Cancel", key=f"cancel_del_{kb.kb_id}"):
                    st.session_state.pop(f"confirm_del_{kb.kb_id}", None)
                    st.rerun()

st.markdown("<hr style='border-color: #293244; margin: 20px 0;'/>", unsafe_allow_html=True)

# ── Upload Section ───────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="margin-bottom: 12px;">
      <span style="font-size: 17px; font-weight: 700; color: #F5F7FA;">Upload Documents</span>
      <span style="font-size: 13px; color: #A0AABC; margin-left: 8px;">
        to <strong style="color: #F5F7FA;">{active_kb_name}</strong>
      </span>
    </div>
    <div style="background: #151A24; border: 1px dashed #293244; border-radius: 8px; padding: 10px 16px; margin-bottom: 12px; font-size: 13px; color: #A0AABC;">
      Supported formats: <strong style="color: #F5F7FA;">PDF, DOCX, TXT</strong> &nbsp;·&nbsp;
      Max file size: <strong style="color: #F5F7FA;">{context.settings.upload_size_limit_mb} MB</strong> per file
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Drag and drop documents or click to browse",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
    label_visibility="collapsed",
    key="doc_uploader",
)

if uploaded_files:
    st.markdown(f"**{len(uploaded_files)} file(s) ready for ingestion:**")
    for f in uploaded_files:
        size_kb = round(len(f.getvalue()) / 1024, 1)
        st.caption(f"• {f.name} ({size_kb} KB)")

    if st.button("Upload Documents", type="primary", key="btn_trigger_upload"):
        progress = st.progress(0.0)
        status = st.empty()
        for idx, uf in enumerate(uploaded_files):
            file_bytes = uf.getvalue()
            status.info(f"Ingesting {uf.name} ({round(len(file_bytes)/1024, 1)} KB)...")
            try:
                outcome = context.document_service.ingest_bytes(uf.name, file_bytes)
                context.user_registry.add_document_to_kb(active_kb_id, outcome.document.document_id)
                if outcome.errors:
                    st.error(f"{uf.name}: {'; '.join(outcome.errors)}")
                elif "duplicate_document_skipped" in outcome.warnings:
                    st.warning(f"{uf.name} already exists in registry.")
                else:
                    st.success(
                        f"Indexed {uf.name} — "
                        f"{len(outcome.chunks)} chunks, "
                        f"{outcome.entity_count} entities, "
                        f"{outcome.relationship_count} relationships."
                    )
            except Exception as exc:
                st.error(f"Failed to ingest {uf.name}: {exc}")
            progress.progress((idx + 1) / len(uploaded_files))
        status.empty()
        st.rerun()

st.markdown("<hr style='border-color: #293244; margin: 20px 0;'/>", unsafe_allow_html=True)

# ── Documents Table ──────────────────────────────────────────────────────────
kb_doc_ids = context.user_registry.get_kb_document_ids(active_kb_id, user_id)
all_docs = context.registry.list_documents()
documents = [d for d in all_docs if d["document_id"] in set(kb_doc_ids)]

st.markdown(
    f"""
    <div style="font-size: 17px; font-weight: 700; color: #F5F7FA; margin-bottom: 12px;">
      Documents in <span style="color: #8B7CFF;">{active_kb_name}</span>
      <span style="font-size: 13px; font-weight: 400; color: #A0AABC; margin-left: 6px;">
        ({len(documents)} documents)
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)

if not documents:
    st.markdown(
        """
        <div class="rag-card" style="text-align: center; padding: 36px 20px;">
          <div style="font-size: 16px; font-weight: 600; color: #F5F7FA;">No documents uploaded yet</div>
          <div style="font-size: 13px; color: #A0AABC; margin-top: 6px;">
            Use the upload area above to add documents to your knowledge corpus.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    STATUS_LABELS = {
        "indexed": "Ready",
        "partial": "Partial",
        "failed": "Failed",
        "pending": "Pending",
        "indexing": "Indexing",
    }
    MIME_LABELS = {
        "application/pdf": "PDF",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "DOCX",
        "text/plain": "TXT",
        "text/markdown": "TXT",
    }

    df_rows = []
    for d in documents:
        status_raw = d.get("status", "").lower()
        df_rows.append(
            {
                "Document Name": d["document_name"],
                "File Type": MIME_LABELS.get(d.get("mime_type", ""), d.get("mime_type", "TXT")),
                "File Size (KB)": round(float(d.get("file_size") or 0) / 1024, 1),
                "Processing Status": STATUS_LABELS.get(status_raw, status_raw.capitalize()),
                "Chunks": d.get("chunk_count", 0),
                "Entities": d.get("entity_count", 0),
                "Upload Date": (d.get("created_at") or "")[:10],
            }
        )

    st.dataframe(pd.DataFrame(df_rows), use_container_width=True, hide_index=True)

    # ── Document Details Inspector & Actions ─────────────────────────────────
    st.markdown("<hr style='border-color: #293244; margin: 20px 0;'/>", unsafe_allow_html=True)
    st.markdown("### Document Details & Actions")

    doc_options = {
        d["document_id"]: f"{d['document_name']} [{STATUS_LABELS.get(d.get('status',''), d.get('status',''))}]"
        for d in documents
    }
    selected_id = st.selectbox(
        "Select document to inspect",
        options=list(doc_options.keys()),
        format_func=lambda x: doc_options[x],
        key="docs_detail_sel",
    )

    if selected_id:
        sel = next(d for d in documents if d["document_id"] == selected_id)
        status_raw = sel.get("status", "").lower()

        meta_col, action_col = st.columns([3, 1])
        with meta_col:
            st.markdown(
                f"""
                <div class="rag-card">
                  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <div>
                      <div style="font-size: 11px; color: #A0AABC; text-transform: uppercase;">Document Name</div>
                      <div style="font-size: 14px; font-weight: 600; color: #F5F7FA; margin-top: 2px;">{sel['document_name']}</div>
                    </div>
                    <div>
                      <div style="font-size: 11px; color: #A0AABC; text-transform: uppercase;">Status</div>
                      <div style="font-size: 14px; font-weight: 600; color: #F5F7FA; margin-top: 2px;">{STATUS_LABELS.get(status_raw, status_raw.capitalize())}</div>
                    </div>
                    <div>
                      <div style="font-size: 11px; color: #A0AABC; text-transform: uppercase;">Vector Chunks</div>
                      <div style="font-size: 14px; font-weight: 600; color: #60A5FA; margin-top: 2px;">{sel.get('chunk_count', 0)}</div>
                    </div>
                    <div>
                      <div style="font-size: 11px; color: #A0AABC; text-transform: uppercase;">Graph Provenance</div>
                      <div style="font-size: 14px; font-weight: 600; color: #A78BFA; margin-top: 2px;">
                        {sel.get('entity_count', 0)} entities · {sel.get('relationship_count', 0)} relationships
                      </div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if sel.get("error_message"):
                st.error(f"Processing Error: {sel['error_message']}")

        with action_col:
            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
            if st.button("Re-index Document", key="reindex_btn", use_container_width=True):
                src_path = sel.get("source_path")
                if src_path and Path(src_path).exists():
                    with st.spinner("Re-indexing document..."):
                        try:
                            outcome = context.document_service.ingest_path(src_path)
                            context.user_registry.add_document_to_kb(active_kb_id, outcome.document.document_id)
                            st.success(f"Re-indexed '{outcome.document.document_name}'.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Re-index failed: {exc}")
                else:
                    st.warning("Original source file not found on disk.")

            if st.button("Delete Document", key="del_doc_btn", use_container_width=True):
                st.session_state["confirm_doc_delete"] = selected_id

        if st.session_state.get("confirm_doc_delete") == selected_id:
            st.warning(f"Delete '{sel['document_name']}'? This permanently removes vector and graph records.")
            cc1, cc2 = st.columns(2)
            with cc1:
                if st.button("Confirm Delete", type="primary", key="confirm_del_doc"):
                    with st.spinner("Deleting..."):
                        context.document_service.delete_document(selected_id)
                        context.user_registry.remove_document_from_kb(active_kb_id, selected_id)
                    st.session_state.pop("confirm_doc_delete", None)
                    st.success("Document deleted.")
                    st.rerun()
            with cc2:
                if st.button("Cancel", key="cancel_del_doc"):
                    st.session_state.pop("confirm_doc_delete", None)
                    st.rerun()

        # Tabs for chunks, entities, and graph preview
        chunks = context.registry.list_chunks(selected_id)
        all_entities = context.registry.list_entities()
        doc_entities = [e for e in all_entities if e.get("document_id") == selected_id]

        if chunks or doc_entities:
            tab_chunks, tab_entities, tab_graph = st.tabs(
                [f"Chunks ({len(chunks)})", f"Entities ({len(doc_entities)})", "Knowledge Graph Preview"]
            )
            with tab_chunks:
                if chunks:
                    for chunk in chunks[:15]:
                        with st.expander(f"Chunk #{chunk['chunk_index']} (Page {chunk.get('page_number') or 1})"):
                            st.text(chunk["text"][:1200])
                    if len(chunks) > 15:
                        st.caption(f"Showing first 15 of {len(chunks)} chunks.")
                else:
                    st.info("No chunks indexed for this document.")

            with tab_entities:
                if doc_entities:
                    df_ents = pd.DataFrame(
                        [
                            {
                                "Entity Name": e.get("canonical_name", ""),
                                "Entity Type": e.get("entity_type", "Concept"),
                                "Confidence": round(float(e.get("confidence", 0.85)), 3),
                            }
                            for e in doc_entities
                        ]
                    )
                    st.dataframe(df_ents, use_container_width=True, hide_index=True)
                else:
                    st.info("No entities extracted.")

            with tab_graph:
                st.caption("Extracted entity relationships for this document:")
                all_rels = context.registry.list_relationships()
                doc_ent_ids = {e["entity_id"] for e in doc_entities}
                doc_rels = [
                    r for r in all_rels
                    if r.get("source_entity_id") in doc_ent_ids or r.get("target_entity_id") in doc_ent_ids
                ]
                if doc_rels:
                    df_rels = pd.DataFrame(
                        [
                            {
                                "Source Entity": r.get("source_entity_name", ""),
                                "Relationship": r.get("relationship_type", ""),
                                "Target Entity": r.get("target_entity_name", ""),
                            }
                            for r in doc_rels[:40]
                        ]
                    )
                    st.dataframe(df_rels, use_container_width=True, hide_index=True)
                else:
                    st.info("No relational edges found for this document.")

# ── Bulk Management ──────────────────────────────────────────────────────────
if documents:
    st.markdown("<hr style='border-color: #293244; margin: 24px 0;'/>", unsafe_allow_html=True)
    with st.expander("Danger Zone: Clear Knowledge Base Documents", expanded=False):
        st.markdown(f"Remove all {len(documents)} documents from **{active_kb_name}**.")
        confirm_clear = st.checkbox(
            f"I confirm I want to permanently remove all documents from '{active_kb_name}'",
            key="clear_all_confirm",
        )
        if st.button("Remove All Documents", disabled=not confirm_clear, type="primary"):
            with st.spinner("Removing documents..."):
                for doc_id in kb_doc_ids:
                    try:
                        context.document_service.delete_document(doc_id)
                        context.user_registry.remove_document_from_kb(active_kb_id, doc_id)
                    except Exception:
                        pass
            st.success(f"All documents removed from '{active_kb_name}'.")
            st.rerun()
