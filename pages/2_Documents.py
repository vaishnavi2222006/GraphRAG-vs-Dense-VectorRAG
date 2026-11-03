"""RAG Benchmark Lab — Documents Management Page."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

import pandas as pd
import streamlit as st

from rag_benchmark_lab.app import get_context
from rag_benchmark_lab.utils.auth import get_current_user_name, require_auth
from rag_benchmark_lab.utils.ui import (
    inject_custom_css,
    render_dev_mode_banner,
    render_html,
    render_topnav,
)


def _safe_filename(original_name: str, content_hash: str) -> str:
    """Return a safe, unique filename for the uploads cache."""
    safe_stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(original_name).stem)
    suffix = Path(original_name).suffix or ""
    return f"{content_hash[:12]}_{safe_stem}{suffix}"


def _cache_upload(file_bytes: bytes, original_name: str, upload_dir: Path) -> str:
    """Write *file_bytes* into *upload_dir* with a safe unique filename."""
    content_hash = hashlib.sha256(file_bytes).hexdigest()
    dest = upload_dir / _safe_filename(original_name, content_hash)
    if not dest.exists():
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(file_bytes)
    return str(dest)


st.set_page_config(
    page_title="Documents — RAG Benchmark Lab",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_topnav(active_page="Documents")

user_id = require_auth()
context = get_context()
user_name = get_current_user_name() or "User"

# ── Page Header ──────────────────────────────────────────────────────────────
render_html(
    """<div style="margin-top:20px;margin-bottom:20px;">
<div style="display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;color:var(--accent);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">
  <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--accent);"></span>
  Corpus Management
</div>
<h1 style="font-size:28px;font-weight:800;letter-spacing:-0.03em;color:var(--text);margin:0 0 4px 0;">
  Document Ingestion &amp; Knowledge Bases
</h1>
<p style="font-size:14px;color:var(--text-muted);margin:0;">
  Upload domain literature, manage isolated knowledge bases, and monitor vector chunking and entity extraction.
</p>
</div>"""
)

# ── Knowledge Base Selector & Creation ───────────────────────────────────────
kbs = context.user_registry.list_knowledge_bases(user_id)

kb_col, create_col = st.columns([3.5, 1.5], vertical_alignment="center")

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
            format_func=lambda x: f"Knowledge Base: {kb_options[x]}",
            index=list(kb_options.keys()).index(active_kb_id),
            key="docs_kb_selector",
            label_visibility="collapsed",
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
        st.info("Create a knowledge base to start uploading and managing research documents.")

with create_col:
    if st.button("New Knowledge Base", icon=":material/add:", use_container_width=True, key="btn_open_create_kb"):
        st.session_state["show_create_kb"] = True

# ── Create Knowledge Base Form ───────────────────────────────────────────────
if st.session_state.get("show_create_kb"):
    with st.container(border=True):
        with st.form("create_kb_form", clear_on_submit=True):
            st.markdown("<h4 style='color:var(--text);margin:0 0 12px 0;font-size:16px;font-weight:700;'>Create Knowledge Base</h4>", unsafe_allow_html=True)
            new_kb_name = st.text_input(
                "Knowledge Base Name *",
                placeholder="e.g. Financial Filings 2024",
                key="new_kb_name",
            )
            new_kb_desc = st.text_input(
                "Description (optional)",
                placeholder="e.g. Annual reports and regulatory filings",
                key="new_kb_desc",
            )
            col_s, col_c = st.columns(2)
            with col_s:
                submitted = st.form_submit_button("Create Knowledge Base", type="primary", use_container_width=True)
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
with st.expander(f"Manage Knowledge Bases ({len(kbs)} available)", expanded=False):
    for kb in kbs:
        col_info, col_ren, col_del = st.columns([4, 1, 1], vertical_alignment="center")
        with col_info:
            doc_count = kb.document_count
            st.markdown(
                f"**{kb.name}** &nbsp;·&nbsp; "
                f"<span style='color:var(--text-muted);font-size:12px;'>"
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

st.markdown("<hr style='border-color:var(--border);margin:20px 0;'/>", unsafe_allow_html=True)

# ── Upload Section ───────────────────────────────────────────────────────────
render_html(
    f"""
    <div style="margin-bottom:12px;">
      <span style="font-size:18px;font-weight:800;color:var(--text);">Upload &amp; Ingest Documents</span>
      <span style="font-size:13.5px;color:var(--text-muted);margin-left:8px;">
        into <strong style="color:var(--accent);">{active_kb_name}</strong>
      </span>
    </div>
    <div style="background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:14px 18px;margin-bottom:14px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
      <div style="display:flex;align-items:center;gap:12px;">
        <span style="font-size:13px;font-weight:600;color:var(--text-muted);">Supported Formats:</span>
        <span class="badge" style="background:var(--accent-light);color:var(--accent);border:1px solid var(--accent-border);">PDF</span>
        <span class="badge" style="background:var(--vector-bg);color:var(--vector);border:1px solid var(--vector-border);">DOCX</span>
        <span class="badge" style="background:var(--hybrid-bg);color:var(--hybrid);border:1px solid var(--hybrid-border);">TXT</span>
      </div>
      <div style="font-size:12.5px;color:var(--text-muted);">
        Max size: <strong style="color:var(--text);">{context.settings.upload_size_limit_mb} MB</strong> per file
      </div>
    </div>
    """
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

    if st.button("Ingest & Index Documents", type="primary", icon=":material/upload:", key="btn_trigger_upload"):
        progress = st.progress(0.0)
        status = st.empty()
        upload_dir = context.settings.upload_dir
        for idx, uf in enumerate(uploaded_files):
            file_bytes = uf.getvalue()
            status.info(f"Extracting, chunking, and embedding {uf.name} ({round(len(file_bytes)/1024, 1)} KB)...")
            try:
                cached_path = _cache_upload(file_bytes, uf.name, upload_dir)
                outcome = context.document_service.ingest_bytes(
                    uf.name, file_bytes, source_path=cached_path
                )
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

st.markdown("<hr style='border-color:var(--border);margin:24px 0;'/>", unsafe_allow_html=True)

# ── Documents Table & Search/Filter ──────────────────────────────────────────
kb_doc_ids = context.user_registry.get_kb_document_ids(active_kb_id, user_id)
all_docs = context.registry.list_documents()
documents = [d for d in all_docs if d["document_id"] in set(kb_doc_ids)]

total_kb_chunks = sum(d.get("chunk_count", 0) for d in documents)
total_kb_entities = sum(d.get("entity_count", 0) for d in documents)
total_kb_rels = sum(d.get("relationship_count", 0) for d in documents)

render_html(
    f"""
    <div style="font-size:18px;font-weight:800;color:var(--text);margin-bottom:12px;">
      Documents in <span style="color:var(--accent);">{active_kb_name}</span>
      <span style="font-size:13px;font-weight:500;color:var(--text-muted);margin-left:6px;">
        ({len(documents)} total)
      </span>
    </div>
    """
)

if documents:
    kb_s1, kb_s2, kb_s3, kb_s4 = st.columns(4)
    with kb_s1:
        render_html(
            f"""<div class="stat-card" style="padding:14px 16px;">
              <div class="stat-value" style="font-size:22px;">{len(documents)}</div>
              <div class="stat-label">Documents</div>
            </div>"""
        )
    with kb_s2:
        render_html(
            f"""<div class="stat-card" style="padding:14px 16px;">
              <div class="stat-value" style="font-size:22px;color:var(--vector);">{total_kb_chunks}</div>
              <div class="stat-label">Vector Chunks</div>
            </div>"""
        )
    with kb_s3:
        render_html(
            f"""<div class="stat-card" style="padding:14px 16px;">
              <div class="stat-value" style="font-size:22px;color:var(--graph);">{total_kb_entities}</div>
              <div class="stat-label">Entities</div>
            </div>"""
        )
    with kb_s4:
        render_html(
            f"""<div class="stat-card" style="padding:14px 16px;">
              <div class="stat-value" style="font-size:22px;color:var(--hybrid);">{total_kb_rels}</div>
              <div class="stat-label">Relationships</div>
            </div>"""
        )
    render_html("<div style='height:14px;'></div>")

# Search & Status Filter Controls
f_col1, f_col2 = st.columns([3, 1.5], vertical_alignment="center")
with f_col1:
    doc_search = st.text_input(
        "Search documents",
        placeholder="Search documents by name...",
        key="doc_search_query",
        label_visibility="collapsed",
    )
with f_col2:
    status_filter = st.selectbox(
        "Filter by status",
        options=["All Statuses", "Ready / Indexed", "Partial", "Failed"],
        index=0,
        key="doc_status_filter",
        label_visibility="collapsed",
    )

# Apply in-memory filters
filtered_docs = documents
if doc_search.strip():
    q = doc_search.strip().lower()
    filtered_docs = [d for d in filtered_docs if q in d.get("document_name", "").lower()]

if status_filter != "All Statuses":
    if status_filter == "Ready / Indexed":
        filtered_docs = [d for d in filtered_docs if d.get("status", "").lower() in ("indexed", "ready")]
    elif status_filter == "Partial":
        filtered_docs = [d for d in filtered_docs if d.get("status", "").lower() == "partial"]
    elif status_filter == "Failed":
        filtered_docs = [d for d in filtered_docs if d.get("status", "").lower() == "failed"]

if not documents:
    render_html(
        """
        <div class="rag-card" style="text-align:center;padding:40px 20px;">
          <div style="font-size:16px;font-weight:700;color:var(--text);">No documents uploaded yet</div>
          <div style="font-size:13.5px;color:var(--text-muted);margin-top:6px;">
            Use the drag-and-drop zone above to ingest your first research document.
          </div>
        </div>
        """
    )
elif not filtered_docs:
    st.info("No documents match your search criteria.")
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
    for d in filtered_docs:
        status_raw = d.get("status", "").lower()
        df_rows.append(
            {
                "Document Name": d["document_name"],
                "Format": MIME_LABELS.get(d.get("mime_type", ""), d.get("mime_type", "TXT")),
                "File Size (KB)": round(float(d.get("file_size") or 0) / 1024, 1),
                "Status": STATUS_LABELS.get(status_raw, status_raw.capitalize()),
                "Vector Chunks": d.get("chunk_count", 0),
                "Entities": d.get("entity_count", 0),
                "Relationships": d.get("relationship_count", 0),
                "Upload Date": (d.get("created_at") or "")[:10],
            }
        )

    st.dataframe(pd.DataFrame(df_rows), use_container_width=True, hide_index=True)

    # ── Document Details Inspector & Actions ─────────────────────────────────
    st.markdown("<hr style='border-color:var(--border);margin:24px 0;'/>", unsafe_allow_html=True)
    st.markdown("### Document Inspector & Actions")

    doc_options = {
        d["document_id"]: f"{d['document_name']} [{STATUS_LABELS.get(d.get('status',''), d.get('status',''))}]"
        for d in filtered_docs
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
        status_badge_type = (
            "badge-success" if status_raw in ("indexed", "ready") else ("badge-warning" if status_raw == "partial" else "badge-error")
        )

        meta_col, action_col = st.columns([3.2, 1.3])
        with meta_col:
            render_html(
                f"""
                <div class="rag-card" style="padding:20px;">
                  <div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:16px;">
                    <div>
                      <div style="font-size:11px;color:var(--text-muted);text-transform:uppercase;font-weight:600;">Document</div>
                      <div style="font-size:14px;font-weight:700;color:var(--text);margin-top:4px;word-break:break-all;">{sel['document_name']}</div>
                    </div>
                    <div>
                      <div style="font-size:11px;color:var(--text-muted);text-transform:uppercase;font-weight:600;">Status</div>
                      <div style="margin-top:4px;"><span class="badge {status_badge_type}">{STATUS_LABELS.get(status_raw, status_raw.capitalize())}</span></div>
                    </div>
                    <div>
                      <div style="font-size:11px;color:var(--text-muted);text-transform:uppercase;font-weight:600;">Vector Chunks</div>
                      <div style="font-size:15px;font-weight:800;color:var(--vector);margin-top:4px;">{sel.get('chunk_count', 0)} chunks</div>
                    </div>
                    <div>
                      <div style="font-size:11px;color:var(--text-muted);text-transform:uppercase;font-weight:600;">Graph Provenance</div>
                      <div style="font-size:14px;font-weight:700;color:var(--graph);margin-top:4px;">
                        {sel.get('entity_count', 0)} ent · {sel.get('relationship_count', 0)} rel
                      </div>
                    </div>
                  </div>
                </div>
                """
            )
            if sel.get("error_message"):
                st.error(f"Processing Error: {sel['error_message']}")

        with action_col:
            st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
            if st.button("Re-index Document", icon=":material/refresh:", key="reindex_btn", use_container_width=True):
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

            if st.button("Delete Document", icon=":material/delete:", key="del_doc_btn", use_container_width=True):
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
            tab_chunks, tab_entities = st.tabs(
                [f"Vector Chunks ({len(chunks)})", f"Extracted Graph Entities ({len(doc_entities)})"]
            )
            with tab_chunks:
                if chunks:
                    for chunk in chunks[:15]:
                        with st.expander(f"Chunk #{chunk['chunk_index']} · Page {chunk.get('page_number') or 1}"):
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
                    st.info("No entities extracted for this document.")
