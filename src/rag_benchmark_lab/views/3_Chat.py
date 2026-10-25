from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime, timezone

import streamlit as st

from rag_benchmark_lab.app import get_context
from rag_benchmark_lab.schemas.auth import MessageRecord
from rag_benchmark_lab.schemas.documents import RetrievalStrategy
from rag_benchmark_lab.utils.auth import get_current_user_name, require_auth
from rag_benchmark_lab.utils.ui import inject_custom_css, render_dev_mode_banner, render_strategy_badge

# page config handled in main app
inject_custom_css()

# Extra chat-specific styles
st.markdown("""
<style>
.chat-bubble-user {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 75%;
    margin-left: auto;
    font-size: 14px;
    color: #1e293b;
}
.chat-bubble-assistant {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px 12px 12px 4px;
    padding: 14px 18px;
    margin: 8px 0;
    max-width: 85%;
    font-size: 14px;
    color: #1e293b;
    line-height: 1.65;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.chat-meta {
    font-size: 11px;
    color: #94a3b8;
    margin-bottom: 4px;
}
.chat-strategy-badge {
    font-size: 11px;
    margin-left: 6px;
}
.citation-pill {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 3px 8px;
    font-size: 11px;
    color: #475569;
    margin: 2px;
    cursor: pointer;
}
.conv-item {
    padding: 8px 12px;
    border-radius: 8px;
    cursor: pointer;
    margin-bottom: 4px;
    border: 1px solid transparent;
    font-size: 13px;
    color: #374151;
    transition: all 0.15s;
}
.conv-item:hover { background: #f1f5f9; border-color: #e2e8f0; }
.conv-item.active { background: #eff6ff; border-color: #bfdbfe; color: #1d4ed8; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

user_id = require_auth()
context = get_context()
user_name = get_current_user_name() or "User"

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"**👤 {user_name}**")

    # KB selector
    kbs = context.user_registry.list_knowledge_bases(user_id)
    if not kbs:
        st.info("No knowledge bases. Create one first.")
        st.page_link("pages/0_Knowledge_Bases.py", label="→ Knowledge Bases", icon="🗂️")
        st.stop()

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
        key="chat_kb_selector",
    )
    if selected_kb_id != active_kb_id:
        st.session_state["active_kb_id"] = selected_kb_id
        st.session_state["active_kb_name"] = kb_options[selected_kb_id]
        # Clear active conversation when switching KB
        if "active_conv_id" in st.session_state:
            del st.session_state["active_conv_id"]
        st.rerun()
    active_kb_id = selected_kb_id
    active_kb_name = kb_options[active_kb_id]

    st.markdown("---")

    # New Conversation button
    if st.button("➕ New Conversation", use_container_width=True, type="primary"):
        conv = context.user_registry.create_conversation(
            user_id=user_id,
            kb_id=active_kb_id,
            title="New Conversation",
        )
        st.session_state["active_conv_id"] = conv.conversation_id
        st.rerun()

    # List conversations for this KB
    conversations = context.user_registry.list_conversations(user_id, kb_id=active_kb_id)
    active_conv_id = st.session_state.get("active_conv_id")

    if conversations:
        st.markdown(f"**Conversations in '{active_kb_name}':**")
        for conv in conversations:
            is_active = conv.conversation_id == active_conv_id
            title = conv.title[:32] + ("..." if len(conv.title) > 32 else "")
            badge = "active" if is_active else ""
            if st.button(
                f"{'💬' if is_active else '○'} {title}",
                key=f"conv_{conv.conversation_id}",
                use_container_width=True,
                type="secondary" if not is_active else "primary",
            ):
                st.session_state["active_conv_id"] = conv.conversation_id
                st.rerun()

    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        from rag_benchmark_lab.utils.auth import logout_user
        logout_user()
        st.rerun()

# ─── Main Chat Area ────────────────────────────────────────────────────────────
active_conv_id = st.session_state.get("active_conv_id")

if not active_conv_id:
    # Show empty state with quick-start
    st.markdown("""
    <div style="text-align:center; padding: 80px 20px;">
        <div style="font-size: 48px; margin-bottom: 16px;">💬</div>
        <h2 style="font-size: 24px; font-weight: 700; color: #0f172a; margin: 0 0 8px 0;">Start a Conversation</h2>
        <p style="font-size: 15px; color: #64748b; max-width: 400px; margin: 0 auto 24px auto;">
            Click <strong>+ New Conversation</strong> in the sidebar to start asking questions about your documents.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Check if there are documents in this KB
    kb_doc_ids = context.user_registry.get_kb_document_ids(active_kb_id, user_id)
    if not kb_doc_ids:
        st.info(f"📄 No documents in '{active_kb_name}' yet. Upload documents to start chatting.")
        st.page_link("pages/2_Corpus_Management.py", label="→ Upload Documents", icon="📁")
    st.stop()

# Verify conversation belongs to this user
conv = context.user_registry.get_conversation(active_conv_id, user_id)
if not conv:
    st.error("Conversation not found or access denied.")
    if "active_conv_id" in st.session_state:
        del st.session_state["active_conv_id"]
    st.stop()

# ─── Conversation Header ───────────────────────────────────────────────────────
col_title, col_actions = st.columns([5, 2])
with col_title:
    render_dev_mode_banner(context.settings.llm_mock_mode)
    st.markdown(f"**💬 {conv.title}** · 🗂️ {active_kb_name}")

with col_actions:
    col_rename, col_delete = st.columns(2)
    with col_rename:
        if st.button("✏️ Rename", key="rename_conv_btn"):
            st.session_state["renaming_conv"] = True
    with col_delete:
        if st.button("🗑️ Delete", key="delete_conv_btn"):
            st.session_state["deleting_conv"] = True

if st.session_state.get("renaming_conv"):
    with st.form("rename_conv_form"):
        new_title = st.text_input("New conversation title", value=conv.title)
        c1, c2 = st.columns(2)
        with c1:
            if st.form_submit_button("Save", type="primary"):
                if new_title.strip():
                    context.user_registry.rename_conversation(active_conv_id, user_id, new_title)
                    del st.session_state["renaming_conv"]
                    st.rerun()
        with c2:
            if st.form_submit_button("Cancel"):
                del st.session_state["renaming_conv"]
                st.rerun()

if st.session_state.get("deleting_conv"):
    st.warning("Delete this conversation? This cannot be undone.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Confirm Delete", type="primary"):
            context.user_registry.delete_conversation(active_conv_id, user_id)
            del st.session_state["active_conv_id"]
            del st.session_state["deleting_conv"]
            st.rerun()
    with c2:
        if st.button("Cancel"):
            del st.session_state["deleting_conv"]
            st.rerun()

# ─── Strategy Selector ─────────────────────────────────────────────────────────
st.markdown("---")
strategy_col, topk_col = st.columns([3, 1])
with strategy_col:
    strategy_option = st.selectbox(
        "Retrieval Strategy",
        options=["AUTOMATIC (Query Router)", "VECTOR (Dense Vector RAG)", "GRAPH (GraphRAG)", "HYBRID (Vector + Graph RRF)"],
        index=0,
        key="chat_strategy",
        label_visibility="collapsed",
    )
with topk_col:
    top_k_input = st.number_input("Top-K", min_value=1, max_value=20, value=context.settings.default_top_k, key="chat_top_k", label_visibility="collapsed")

# ─── Message History ──────────────────────────────────────────────────────────
messages = context.user_registry.list_messages(active_conv_id, user_id)

# Display messages
for msg in messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div style="display:flex;justify-content:flex-end;margin:4px 0;">
            <div class="chat-bubble-user">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        strategy = msg.get("retrieval_strategy", "")
        badge = render_strategy_badge(strategy) if strategy else ""
        mock_notice = " · <em style='color:#f59e0b;font-size:10px;'>MOCK MODE</em>" if msg.get("mock_mode") else ""
        latency_info = ""
        if msg.get("total_latency_ms"):
            latency_info = f" · {msg['total_latency_ms']:.0f}ms"

        st.markdown(f"""
        <div style="display:flex;justify-content:flex-start;margin:4px 0;">
            <div style="max-width:85%;">
                <div class="chat-meta">🤖 Assistant {badge}{mock_notice}{latency_info}</div>
                <div class="chat-bubble-assistant">{msg["content"]}</div>
        """, unsafe_allow_html=True)

        # Citations
        citations = msg.get("citations", [])
        if citations:
            citation_html = " ".join(
                f'<span class="citation-pill">📄 {c.get("document_name","?")} [#{c.get("chunk_index","?")}]</span>'
                for c in citations[:5]
            )
            st.markdown(f'<div style="margin-left:0;margin-top:4px;">{citation_html}</div>', unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        # Expandable details
        if citations or msg.get("graph_evidence"):
            with st.expander(f"📋 Sources & Evidence ({len(citations)} citations)"):
                if citations:
                    for c in citations:
                        st.markdown(f"**{c.get('document_name','?')}** — Chunk #{c.get('chunk_index','?')} "
                                    f"(Page {c.get('page_number') or 1}) · Score: {c.get('score', 0):.3f}")
                if msg.get("graph_evidence"):
                    ge = msg["graph_evidence"]
                    rels = ge.get("relationships", [])
                    if rels:
                        st.markdown(f"**Graph paths ({len(rels)}):**")
                        for r in rels[:4]:
                            st.caption(f"• {r.get('source_entity_name','?')} → *{r.get('relationship_type','?')}* → {r.get('target_entity_name','?')}")

        if msg.get("warnings"):
            for w in msg["warnings"]:
                if "mock_mode" not in w:
                    st.caption(f"⚠️ {w}")

# ─── Input Area ───────────────────────────────────────────────────────────────
st.markdown("---")

# Quick suggestion chips
suggestions = [
    "What are the main topics in my documents?",
    "Summarize the key findings.",
    "What entities and relationships are discussed?",
    "Explain the methodology used.",
]
sug_cols = st.columns(len(suggestions))
for i, sug in enumerate(suggestions):
    with sug_cols[i]:
        if st.button(f"💡 {sug[:30]}...", key=f"sug_{i}", use_container_width=True, help=sug):
            st.session_state["chat_input_prefill"] = sug

# Chat input
question = st.chat_input("Ask a question about your documents...")

# Also handle suggestion clicks
if "chat_input_prefill" in st.session_state:
    prefilled = st.session_state.pop("chat_input_prefill")
    # Process this as a question
    question = prefilled

if question:
    # Rename conversation from first user message if still "New Conversation"
    if conv.title == "New Conversation" and len(messages) == 0:
        auto_title = question[:50] + ("..." if len(question) > 50 else "")
        context.user_registry.rename_conversation(active_conv_id, user_id, auto_title)

    # Save user message
    user_msg_id = str(uuid.uuid4())
    context.user_registry.save_message(MessageRecord(
        message_id=user_msg_id,
        conversation_id=active_conv_id,
        user_id=user_id,
        role="user",
        content=question,
        created_at=datetime.now(timezone.utc).isoformat(),
    ))

    # Get KB document IDs for scoped retrieval
    kb_doc_ids = context.user_registry.get_kb_document_ids(active_kb_id, user_id)
    doc_ids_filter = kb_doc_ids if kb_doc_ids else None

    # Map strategy
    manual_strategy = None
    if "VECTOR" in strategy_option:
        manual_strategy = RetrievalStrategy.VECTOR
    elif "GRAPH" in strategy_option:
        manual_strategy = RetrievalStrategy.GRAPH
    elif "HYBRID" in strategy_option:
        manual_strategy = RetrievalStrategy.HYBRID

    # Execute RAG pipeline
    with st.spinner("Retrieving evidence and generating answer..."):
        result = context.query_service.answer(
            question=question,
            manual_strategy=manual_strategy,
            document_ids=doc_ids_filter,
            top_k=int(top_k_input),
        )

    # Build citations list for storage
    citations_for_storage = [
        {
            "document_id": c.document_id,
            "document_name": c.document_name,
            "chunk_id": c.chunk_id,
            "chunk_index": c.chunk_index,
            "page_number": c.page_number,
            "score": float(c.score) if c.score is not None else None,
        }
        for c in result.citations
    ]

    # Build graph evidence for storage
    graph_ev_for_storage = None
    if result.graph_evidence:
        graph_ev_for_storage = {
            "entities": [
                {"entity_id": e.entity_id, "canonical_name": e.canonical_name, "entity_type": e.entity_type}
                for e in result.graph_evidence.entities[:10]
            ],
            "relationships": [
                {
                    "source_entity_name": r.source_entity_name,
                    "relationship_type": r.relationship_type,
                    "target_entity_name": r.target_entity_name,
                    "confidence": r.confidence,
                }
                for r in result.graph_evidence.relationships[:10]
            ],
        }

    # Save assistant message
    asst_msg_id = str(uuid.uuid4())
    context.user_registry.save_message(MessageRecord(
        message_id=asst_msg_id,
        conversation_id=active_conv_id,
        user_id=user_id,
        role="assistant",
        content=result.answer,
        retrieval_strategy=result.selected_strategy.value,
        retrieval_latency_ms=result.retrieval_latency_ms,
        generation_latency_ms=result.generation_latency_ms,
        total_latency_ms=result.total_latency_ms,
        citations=citations_for_storage,
        graph_evidence=graph_ev_for_storage,
        model_name=result.model_name,
        warnings=result.warnings or [],
        errors=result.errors or [],
        mock_mode=bool(result.warnings and "mock_mode_active" in result.warnings),
        created_at=datetime.now(timezone.utc).isoformat(),
    ))

    st.rerun()
