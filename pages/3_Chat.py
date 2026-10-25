"""RAG Benchmark Lab — Chat Workspace."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import streamlit as st

from rag_benchmark_lab.app import get_context
from rag_benchmark_lab.schemas.auth import MessageRecord
from rag_benchmark_lab.schemas.documents import RetrievalStrategy
from rag_benchmark_lab.utils.auth import get_current_user_name, require_auth
from rag_benchmark_lab.utils.ui import (
    inject_custom_css,
    render_dev_mode_banner,
    render_strategy_badge,
    render_topnav,
)

st.set_page_config(
    page_title="Chat — RAG Benchmark Lab",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_topnav(active_page="Chat")

user_id = require_auth()
context = get_context()
user_name = get_current_user_name() or "User"

render_dev_mode_banner(context.settings.llm_mock_mode)

# ── Page Header ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-bottom: 16px;">
      <h1 style="font-size: 24px; font-weight: 800; letter-spacing: -0.03em; color: #F5F7FA; margin: 0 0 4px 0;">
        Chat with Your Documents
      </h1>
      <p style="font-size: 13.5px; color: #A0AABC; margin: 0;">
        Ask questions and get answers grounded in your uploaded documents.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Knowledge Base Validation ────────────────────────────────────────────────
kbs = context.user_registry.list_knowledge_bases(user_id)
if not kbs:
    st.warning("No knowledge bases found. Create a knowledge base and upload documents first.")
    if st.button("Go to Documents", key="chat_goto_docs"):
        st.switch_page("pages/2_Documents.py")
    st.stop()

kb_options = {kb.kb_id: kb.name for kb in kbs}
active_kb_id = st.session_state.get("active_kb_id")
if active_kb_id not in kb_options:
    active_kb_id = list(kb_options.keys())[0]
    st.session_state["active_kb_id"] = active_kb_id
    st.session_state["active_kb_name"] = kb_options[active_kb_id]

# ── Toolbar: KB Selector + Strategy + Conversation Controls ─────────────────
tb_col1, tb_col2, tb_col3, tb_col4 = st.columns([2.2, 2.2, 1.2, 1.2], vertical_alignment="center")

with tb_col1:
    selected_kb_id = st.selectbox(
        "Active Knowledge Base",
        options=list(kb_options.keys()),
        format_func=lambda x: kb_options[x],
        index=list(kb_options.keys()).index(active_kb_id),
        key="chat_kb_sel",
        label_visibility="collapsed",
    )
    if selected_kb_id != active_kb_id:
        st.session_state["active_kb_id"] = selected_kb_id
        st.session_state["active_kb_name"] = kb_options[selected_kb_id]
        st.session_state.pop("active_conv_id", None)
        st.rerun()
    active_kb_id = selected_kb_id
    active_kb_name = kb_options[active_kb_id]

with tb_col2:
    STRATEGY_OPTIONS = {
        "vector": "Vector RAG",
        "graph": "GraphRAG",
        "hybrid": "Hybrid RAG",
        "auto": "Auto (Query Router)",
    }
    strategy_key = st.selectbox(
        "Retrieval Strategy",
        options=list(STRATEGY_OPTIONS.keys()),
        format_func=lambda k: STRATEGY_OPTIONS[k],
        key="chat_strategy",
        label_visibility="collapsed",
    )

with tb_col3:
    if st.button("New Chat", use_container_width=True, key="btn_chat_new"):
        conv = context.user_registry.create_conversation(
            user_id=user_id,
            kb_id=active_kb_id,
            title="New Conversation",
        )
        st.session_state["active_conv_id"] = conv.conversation_id
        st.rerun()

with tb_col4:
    if st.button("History", use_container_width=True, key="btn_chat_history"):
        st.session_state["show_conv_list"] = not st.session_state.get("show_conv_list", False)

# Strategy explanation (1 clear sentence)
STRATEGY_EXPLANATIONS = {
    "vector": "Retrieves semantically similar text passages.",
    "graph": "Retrieves related entities, relationships, and supporting passages.",
    "hybrid": "Combines vector and graph retrieval.",
    "auto": "Automatically selects the best retrieval strategy based on your question.",
}
st.caption(STRATEGY_EXPLANATIONS[strategy_key])

# ── Conversation History Drawer ──────────────────────────────────────────────
if st.session_state.get("show_conv_list"):
    conversations = context.user_registry.list_conversations(user_id, kb_id=active_kb_id)
    if conversations:
        st.markdown(f"**Conversations in '{active_kb_name}':**")
        for c_item in conversations:
            col_t, col_a = st.columns([4, 1], vertical_alignment="center")
            with col_t:
                is_active = c_item.conversation_id == st.session_state.get("active_conv_id")
                title_display = c_item.title if len(c_item.title) <= 50 else c_item.title[:48] + "…"
                if st.button(
                    title_display,
                    key=f"conv_{c_item.conversation_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state["active_conv_id"] = c_item.conversation_id
                    st.session_state["show_conv_list"] = False
                    st.rerun()
            with col_a:
                if st.button("Delete", key=f"del_conv_{c_item.conversation_id}", use_container_width=True):
                    context.user_registry.delete_conversation(c_item.conversation_id, user_id)
                    if st.session_state.get("active_conv_id") == c_item.conversation_id:
                        st.session_state.pop("active_conv_id", None)
                    st.rerun()
    else:
        st.info("No prior conversations. Click 'New Chat' to start.")
    st.markdown("<hr style='border-color: #293244; margin: 12px 0;'/>", unsafe_allow_html=True)

# ── Active Conversation Resolution ───────────────────────────────────────────
active_conv_id = st.session_state.get("active_conv_id")
kb_doc_ids = context.user_registry.get_kb_document_ids(active_kb_id, user_id)

if not active_conv_id:
    # Check if there is an existing conversation in this KB
    existing_convs = context.user_registry.list_conversations(user_id, kb_id=active_kb_id)
    if existing_convs:
        active_conv_id = existing_convs[0].conversation_id
        st.session_state["active_conv_id"] = active_conv_id
    elif kb_doc_ids:
        # Automatically initialize a new conversation for convenience
        conv = context.user_registry.create_conversation(
            user_id=user_id,
            kb_id=active_kb_id,
            title="New Conversation",
        )
        active_conv_id = conv.conversation_id
        st.session_state["active_conv_id"] = active_conv_id
    else:
        st.markdown(
            f"""
            <div class="rag-card" style="text-align: center; padding: 40px 20px; margin-top: 16px;">
              <h4 style="font-size: 17px; font-weight: 700; color: #F5F7FA; margin: 0 0 8px 0;">No Documents in Knowledge Base</h4>
              <p style="font-size: 13.5px; color: #A0AABC; max-width: 440px; margin: 0 auto 20px;">
                Upload documents to '{active_kb_name}' to start asking questions.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c_up1, c_up2, c_up3 = st.columns([1, 1.2, 1])
        with c_up2:
            if st.button("Upload Documents", type="primary", use_container_width=True, key="chat_upload_prompt"):
                st.switch_page("pages/2_Documents.py")
        st.stop()

conv = context.user_registry.get_conversation(active_conv_id, user_id)
if not conv:
    st.session_state.pop("active_conv_id", None)
    st.stop()

# ── Conversation Header & Actions ────────────────────────────────────────────
conv_hdr_col, ren_col, del_col = st.columns([5, 1, 1], vertical_alignment="center")
with conv_hdr_col:
    st.markdown(
        f'<div style="font-size: 13.5px; font-weight: 600; color: #A0AABC;">'
        f'Conversation: <span style="color: #F5F7FA;">{conv.title}</span> &nbsp;·&nbsp; {active_kb_name}</div>',
        unsafe_allow_html=True,
    )
with ren_col:
    if st.button("Rename", key="btn_ren_conv", use_container_width=True):
        st.session_state["renaming_conv"] = True
with del_col:
    if st.button("Clear", key="btn_del_conv", use_container_width=True):
        st.session_state["del_conv_confirm"] = True

if st.session_state.get("renaming_conv"):
    with st.form("rename_form"):
        new_title = st.text_input("New conversation title", value=conv.title)
        r1, r2 = st.columns(2)
        with r1:
            if st.form_submit_button("Save", type="primary"):
                if new_title.strip():
                    context.user_registry.rename_conversation(active_conv_id, user_id, new_title.strip())
                    st.session_state.pop("renaming_conv", None)
                    st.rerun()
        with r2:
            if st.form_submit_button("Cancel"):
                st.session_state.pop("renaming_conv", None)
                st.rerun()

if st.session_state.get("del_conv_confirm"):
    st.warning("Delete this conversation? All message history will be removed.")
    d1, d2 = st.columns(2)
    with d1:
        if st.button("Confirm Delete", type="primary", key="btn_conf_del_conv"):
            context.user_registry.delete_conversation(active_conv_id, user_id)
            st.session_state.pop("active_conv_id", None)
            st.session_state.pop("del_conv_confirm", None)
            st.rerun()
    with d2:
        if st.button("Cancel", key="btn_cancel_del_conv"):
            st.session_state.pop("del_conv_confirm", None)
            st.rerun()

st.markdown("<hr style='border-color: #293244; margin: 12px 0;'/>", unsafe_allow_html=True)

# ── Message History ──────────────────────────────────────────────────────────
messages = context.user_registry.list_messages(active_conv_id, user_id)

chat_container = st.container()
with chat_container:
    for msg in messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div style="display:flex;justify-content:flex-end;margin:8px 0;">'
                f'<div class="chat-user">{msg["content"]}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            strategy = msg.get("retrieval_strategy", "")
            badge_html = render_strategy_badge(strategy) if strategy else ""
            latency_str = ""
            if msg.get("total_latency_ms"):
                latency_str = f" · Response Time: {msg['total_latency_ms']:.0f}ms"
            mock_notice = (
                " · <em style='color:#F5B942;font-size:11px;'>Dev Mode</em>"
                if msg.get("mock_mode") else ""
            )

            st.markdown(
                f'<div style="display:flex;justify-content:flex-start;margin:8px 0;">'
                f'<div style="max-width:85%;">'
                f'<div style="font-size:12px;color:#A0AABC;margin-bottom:4px;">Assistant &nbsp;{badge_html}{mock_notice}{latency_str}</div>'
                f'<div class="chat-assistant">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

            # Citations
            citations = msg.get("citations", [])
            if citations:
                pills = " ".join(
                    f'<span style="display:inline-block;background:#1B2230;border:1px solid #293244;border-radius:4px;padding:2px 8px;font-size:11.5px;color:#A0AABC;margin:2px;">'
                    f'Source: {c.get("document_name","?")} [Chunk #{c.get("chunk_index","?")}]</span>'
                    for c in citations[:6]
                )
                st.markdown(f'<div style="margin:6px 0;">{pills}</div>', unsafe_allow_html=True)

            st.markdown("</div></div>", unsafe_allow_html=True)

            # Expandable evidence details
            if citations or msg.get("graph_evidence"):
                total_ev = len(citations)
                graph_ev = msg.get("graph_evidence")
                graph_rels = (graph_ev or {}).get("relationships", [])

                with st.expander(
                    f"Sources & Evidence ({total_ev} citations"
                    + (f", {len(graph_rels)} graph paths" if graph_rels else "")
                    + ")"
                ):
                    if citations:
                        st.markdown("**Document Sources & Relevance Scores:**")
                        for c in citations:
                            score_str = f" · Similarity Score: {c['score']:.3f}" if c.get("score") is not None else ""
                            st.caption(
                                f"• {c.get('document_name','?')} "
                                f"[Chunk #{c.get('chunk_index','?')}, Page {c.get('page_number') or 1}]{score_str}"
                            )
                    if graph_rels:
                        st.markdown("**Knowledge Graph Paths:**")
                        for r in graph_rels[:6]:
                            st.caption(
                                f"• {r.get('source_entity_name','?')} "
                                f"→ {r.get('relationship_type','?')} "
                                f"→ {r.get('target_entity_name','?')}"
                            )
                    if msg.get("retrieval_latency_ms") or msg.get("generation_latency_ms"):
                        st.markdown("**Performance Breakdown:**")
                        p_col1, p_col2, p_col3 = st.columns(3)
                        with p_col1:
                            st.metric("Retrieval Latency", f"{msg.get('retrieval_latency_ms',0):.0f} ms")
                        with p_col2:
                            st.metric("Generation Latency", f"{msg.get('generation_latency_ms',0):.0f} ms")
                        with p_col3:
                            st.metric("Total Response Time", f"{msg.get('total_latency_ms',0):.0f} ms")

# ── Suggested Questions Chips ────────────────────────────────────────────────
st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

SUGGESTIONS = [
    "What are the main topics discussed?",
    "Summarize the key findings.",
    "What entities and relationships exist?",
    "Explain the core methodology.",
]
sug_cols = st.columns(len(SUGGESTIONS))
for i, sug in enumerate(SUGGESTIONS):
    with sug_cols[i]:
        if st.button(sug, key=f"sug_{i}", use_container_width=True):
            st.session_state["chat_prefill"] = sug

# ── Message Input ────────────────────────────────────────────────────────────
question = st.chat_input("Ask a question about your documents...")

if "chat_prefill" in st.session_state:
    question = st.session_state.pop("chat_prefill")

if question:
    # Auto-rename if first message
    if conv.title == "New Conversation" and not messages:
        auto_title = question[:50] + ("…" if len(question) > 50 else "")
        context.user_registry.rename_conversation(active_conv_id, user_id, auto_title)

    # Save user message
    user_msg_id = str(uuid.uuid4())
    context.user_registry.save_message(
        MessageRecord(
            message_id=user_msg_id,
            conversation_id=active_conv_id,
            user_id=user_id,
            role="user",
            content=question,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    )

    # Map strategy
    manual_strategy = None
    if strategy_key == "vector":
        manual_strategy = RetrievalStrategy.VECTOR
    elif strategy_key == "graph":
        manual_strategy = RetrievalStrategy.GRAPH
    elif strategy_key == "hybrid":
        manual_strategy = RetrievalStrategy.HYBRID

    with st.spinner("Retrieving evidence and generating answer..."):
        result = context.query_service.answer(
            question=question,
            manual_strategy=manual_strategy,
            document_ids=kb_doc_ids if kb_doc_ids else None,
            top_k=context.settings.default_top_k,
        )

    # Groundedness notice if no evidence found
    answer_text = result.answer
    if not result.citations and not (result.graph_evidence and result.graph_evidence.entities):
        answer_text = "The uploaded documents do not contain sufficient evidence to answer this question."

    citations_storage = [
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

    graph_ev_storage = None
    if result.graph_evidence:
        graph_ev_storage = {
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

    context.user_registry.save_message(
        MessageRecord(
            message_id=str(uuid.uuid4()),
            conversation_id=active_conv_id,
            user_id=user_id,
            role="assistant",
            content=answer_text,
            retrieval_strategy=result.selected_strategy.value,
            retrieval_latency_ms=result.retrieval_latency_ms,
            generation_latency_ms=result.generation_latency_ms,
            total_latency_ms=result.total_latency_ms,
            citations=citations_storage,
            graph_evidence=graph_ev_storage,
            model_name=result.model_name,
            warnings=result.warnings or [],
            errors=result.errors or [],
            mock_mode=bool(result.warnings and "mock_mode_active" in result.warnings),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    )

    st.rerun()
