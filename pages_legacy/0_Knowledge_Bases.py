from __future__ import annotations

import pandas as pd
import streamlit as st

from rag_benchmark_lab.app import get_context
from rag_benchmark_lab.utils.auth import get_current_user_name, require_auth
from rag_benchmark_lab.utils.ui import inject_custom_css, render_lab_header

st.set_page_config(page_title="Knowledge Bases — RAG Benchmark Lab", page_icon="🗂️", layout="wide")
inject_custom_css()

user_id = require_auth()
context = get_context()

render_lab_header(
    title="Knowledge Bases",
    subtitle=f"Manage your private document collections. Each knowledge base maintains isolated vector and graph indexes.",
    badge_text="WORKSPACE",
)

user_name = get_current_user_name() or "User"

# Sidebar logout
with st.sidebar:
    st.markdown(f"**👤 {user_name}**")
    if st.button("🚪 Logout", use_container_width=True):
        from rag_benchmark_lab.utils.auth import logout_user
        logout_user()
        st.rerun()

# ─── Create New KB ─────────────────────────────────────────────────────────────
st.markdown("### ➕ Create New Knowledge Base")
with st.form("create_kb_form", clear_on_submit=True):
    col_name, col_desc, col_btn = st.columns([2, 3, 1])
    with col_name:
        kb_name = st.text_input("Knowledge Base Name", placeholder="e.g. Research Papers", label_visibility="collapsed")
    with col_desc:
        kb_desc = st.text_input("Description (optional)", placeholder="e.g. AI and ML papers from 2024", label_visibility="collapsed")
    with col_btn:
        create_submitted = st.form_submit_button("Create", type="primary", use_container_width=True)

if create_submitted:
    if not kb_name.strip():
        st.error("Please enter a name for the knowledge base.")
    else:
        kb = context.user_registry.create_knowledge_base(user_id, kb_name, kb_desc)
        st.success(f"✅ Knowledge base **{kb.name}** created successfully!")
        st.rerun()

st.markdown("---")

# ─── List Existing KBs ──────────────────────────────────────────────────────────
st.markdown("### 📚 Your Knowledge Bases")
kbs = context.user_registry.list_knowledge_bases(user_id)

if not kbs:
    st.info("You haven't created any knowledge bases yet. Create one above to start uploading documents.")
else:
    for kb in kbs:
        with st.container():
            col_info, col_actions = st.columns([5, 2])
            with col_info:
                st.markdown(f"""
                <div style="background:#fff;border:1px solid #e2e8f0;border-left:4px solid #2563eb;border-radius:10px;padding:16px 20px;margin-bottom:12px;">
                    <div style="font-size:17px;font-weight:700;color:#0f172a;margin-bottom:4px;">🗂️ {kb.name}</div>
                    <div style="font-size:13px;color:#64748b;margin-bottom:10px;">{kb.description or 'No description'}</div>
                    <div style="display:flex;gap:24px;font-size:12.5px;color:#475569;">
                        <span>📄 <strong>{kb.document_count}</strong> documents</span>
                        <span>📅 Created {kb.created_at[:10]}</span>
                        <span>🔄 Updated {kb.updated_at[:10]}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_actions:
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                open_btn = st.button("📂 Open", key=f"open_{kb.kb_id}", use_container_width=True, type="primary")
                rename_btn = st.button("✏️ Rename", key=f"rename_{kb.kb_id}", use_container_width=True)
                delete_btn = st.button("🗑️ Delete", key=f"delete_{kb.kb_id}", use_container_width=True)

                if open_btn:
                    st.session_state["active_kb_id"] = kb.kb_id
                    st.session_state["active_kb_name"] = kb.name
                    st.switch_page("pages/2_Corpus_Management.py")

                if rename_btn:
                    st.session_state[f"renaming_{kb.kb_id}"] = True

                if delete_btn:
                    st.session_state[f"confirming_delete_{kb.kb_id}"] = True

            # Rename form
            if st.session_state.get(f"renaming_{kb.kb_id}"):
                with st.form(f"rename_form_{kb.kb_id}"):
                    new_name = st.text_input("New name", value=kb.name)
                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        if st.form_submit_button("Save", type="primary"):
                            if new_name.strip():
                                context.user_registry.rename_knowledge_base(kb.kb_id, user_id, new_name)
                                del st.session_state[f"renaming_{kb.kb_id}"]
                                st.rerun()
                    with col_cancel:
                        if st.form_submit_button("Cancel"):
                            del st.session_state[f"renaming_{kb.kb_id}"]
                            st.rerun()

            # Delete confirmation
            if st.session_state.get(f"confirming_delete_{kb.kb_id}"):
                st.warning(f"⚠️ Are you sure you want to delete **{kb.name}**? This will remove all associated documents, vector indexes, and graph data.")
                col_confirm, col_cancel = st.columns(2)
                with col_confirm:
                    if st.button("🗑️ Yes, Delete", key=f"confirm_del_{kb.kb_id}", type="primary"):
                        # Get all document IDs in this KB and delete them
                        doc_ids = context.user_registry.get_kb_document_ids(kb.kb_id, user_id)
                        for doc_id in doc_ids:
                            try:
                                context.document_service.delete_document(doc_id)
                            except Exception:
                                pass
                        context.user_registry.delete_knowledge_base(kb.kb_id, user_id)
                        if st.session_state.get("active_kb_id") == kb.kb_id:
                            del st.session_state["active_kb_id"]
                            del st.session_state["active_kb_name"]
                        del st.session_state[f"confirming_delete_{kb.kb_id}"]
                        st.success(f"Deleted knowledge base: {kb.name}")
                        st.rerun()
                with col_cancel:
                    if st.button("Cancel", key=f"cancel_del_{kb.kb_id}"):
                        del st.session_state[f"confirming_delete_{kb.kb_id}"]
                        st.rerun()

    st.markdown("---")
    # Quick stats across all KBs
    total_docs = sum(kb.document_count for kb in kbs)
    st.markdown(f"**Total:** {len(kbs)} knowledge bases · {total_docs} documents")
