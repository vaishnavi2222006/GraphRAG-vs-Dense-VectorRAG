"""RAG Benchmark Lab — Chat Workspace."""
from __future__ import annotations

from datetime import datetime, timezone
import uuid

import streamlit as st

from rag_benchmark_lab.app import get_context
from rag_benchmark_lab.schemas.auth import MessageRecord
from rag_benchmark_lab.schemas.documents import RetrievalStrategy
from rag_benchmark_lab.utils.auth import get_current_user_name, require_auth
from rag_benchmark_lab.utils.ui import (
    format_chat_markdown,
    format_latency,
    format_metric,
    inject_custom_css,
    render_dev_mode_banner,
    render_html,
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

# ── Page Header ──────────────────────────────────────────────────────────────
render_html(
    """<div style="margin-top:20px;margin-bottom:18px;">
<div style="display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;color:var(--accent);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">
  <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--accent);"></span>
  Interactive Assistant
</div>
<h1 style="font-size:28px;font-weight:800;letter-spacing:-0.03em;color:var(--text);margin:0 0 4px 0;">
  Chat with Your Documents
</h1>
<p style="font-size:14px;color:var(--text-muted);margin:0;">
  Query your knowledge corpus conversationally using selectable retrieval strategies with verifiable grounded citations.
</p>
</div>"""
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
tb_col1, tb_col2, tb_col3, tb_col4 = st.columns([2.4, 2.2, 1.2, 1.2], vertical_alignment="center")

with tb_col1:
    selected_kb_id = st.selectbox(
        "Active Knowledge Base",
        options=list(kb_options.keys()),
        format_func=lambda x: f"KB: {kb_options[x]}",
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
        "vector": "Vector RAG (Semantic)",
        "graph": "GraphRAG (Relational)",
        "hybrid": "Hybrid RAG (RRF Fusion)",
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
    if st.button("New Chat", icon=":material/add:", use_container_width=True, key="btn_chat_new"):
        conv = context.user_registry.create_conversation(
            user_id=user_id,
            kb_id=active_kb_id,
            title="New Conversation",
        )
        st.session_state["active_conv_id"] = conv.conversation_id
        st.rerun()

with tb_col4:
    if st.button("History", icon=":material/history:", use_container_width=True, key="btn_chat_history"):
        st.session_state["show_conv_list"] = not st.session_state.get("show_conv_list", False)

# Strategy explanation line
STRATEGY_EXPLANATIONS = {
    "vector": "Retrieves semantically similar text passages via dense vector search.",
    "graph": "Traverses interconnected entity-relationship paths and supporting contexts.",
    "hybrid": "Fuses dense semantic retrieval with multi-hop graph paths via Reciprocal Rank Fusion.",
    "auto": "Automatically classifies query complexity and routes to the optimal retrieval approach.",
}
st.caption(STRATEGY_EXPLANATIONS[strategy_key])

# ── Conversation History Drawer ──────────────────────────────────────────────
if st.session_state.get("show_conv_list"):
    conversations = context.user_registry.list_conversations(user_id, kb_id=active_kb_id)
    with st.container(border=True):
        if conversations:
            st.markdown(f"**Conversations in '{active_kb_name}':**")
            for c_item in conversations:
                col_t, col_a = st.columns([4.2, 0.8], vertical_alignment="center")
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

# ── Active Conversation Resolution ───────────────────────────────────────────
active_conv_id = st.session_state.get("active_conv_id")
kb_doc_ids = context.user_registry.get_kb_document_ids(active_kb_id, user_id)

if not active_conv_id:
    existing_convs = context.user_registry.list_conversations(user_id, kb_id=active_kb_id)
    if existing_convs:
        active_conv_id = existing_convs[0].conversation_id
        st.session_state["active_conv_id"] = active_conv_id
    elif kb_doc_ids:
        conv = context.user_registry.create_conversation(
            user_id=user_id,
            kb_id=active_kb_id,
            title="New Conversation",
        )
        active_conv_id = conv.conversation_id
        st.session_state["active_conv_id"] = active_conv_id
    else:
        render_html(
            f"""
            <div class="rag-card" style="text-align:center;padding:48px 20px;margin-top:16px;">
              <h4 style="font-size:18px;font-weight:800;color:var(--text);margin:0 0 8px 0;">No Documents in Knowledge Base</h4>
              <p style="font-size:13.5px;color:var(--text-muted);max-width:460px;margin:0 auto 20px;">
                Upload documents to '{active_kb_name}' to start asking questions grounded in your research corpus.
              </p>
            </div>
            """
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
    render_html(
        f'<div style="font-size:13.5px;font-weight:600;color:var(--text-muted);">'
        f'Conversation: <strong style="color:var(--text);">{conv.title}</strong> &nbsp;·&nbsp; {active_kb_name}</div>'
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

st.markdown("<hr style='border-color:var(--border);margin:14px 0;'/>", unsafe_allow_html=True)

# ── Message History ──────────────────────────────────────────────────────────
messages = context.user_registry.list_messages(active_conv_id, user_id)

if not messages:
    # Modern empty-state assistant greeting with suggested prompt chips
    render_html(
        f"""
        <div class="rag-card" style="text-align:center;padding:40px 20px;margin-bottom:20px;">
          <div style="display:inline-flex;align-items:center;justify-content:center;width:44px;height:44px;border-radius:12px;background:var(--accent-light);border:1px solid var(--accent-border);margin-bottom:12px;">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
            </svg>
          </div>
          <h3 style="font-size:18px;font-weight:800;color:var(--text);margin:0 0 6px 0;">Research Assistant Ready</h3>
          <p style="font-size:13.5px;color:var(--text-muted);max-width:480px;margin:0 auto;line-height:1.55;">
            Ask questions about the documents in <strong style="color:var(--accent);">{active_kb_name}</strong>.
            Answers will be synthesized with grounded citations and multi-hop graph evidence.
          </p>
        </div>
        """
    )
else:
    for msg in messages:
        if msg["role"] == "user":
            render_html(
                f'<div style="display:flex;justify-content:flex-end;margin:10px 0;">'
                f'<div class="chat-user">{msg["content"]}</div></div>'
            )
        else:
            strategy = msg.get("retrieval_strategy", "")
            badge_html = render_strategy_badge(strategy) if strategy else ""
            latency_str = ""
            tot_lat = msg.get("total_latency_ms")
            if tot_lat is not None and isinstance(tot_lat, (int, float)) and tot_lat > 0:
                latency_str = f" · Latency: <strong>{tot_lat:.0f}ms</strong>"
            mock_notice = (
                " · <span class='badge badge-warning' style='font-size:10px;'>Dev Mode</span>"
                if msg.get("mock_mode") else ""
            )

            citations = msg.get("citations", [])
            citations_html = ""
            if citations:
                pills = " ".join(
                    f'<span class="badge" style="background:var(--card-hi);border:1px solid var(--border);color:var(--text-muted);font-size:11px;margin:2px;">'
                    f'📄 {c.get("document_name","?")} [Chunk #{c.get("chunk_index","?")}]</span>'
                    for c in citations[:6]
                )
                citations_html = f'<div style="margin:8px 0 4px 0;">{pills}</div>'

            # ── Per-Query Token & Cost Telemetry Resolution ──────────────────
            p_tok = msg.get("prompt_tokens")
            c_tok = msg.get("completion_tokens")
            t_tok = msg.get("total_tokens")
            if t_tok is None and p_tok is not None and c_tok is not None:
                t_tok = p_tok + c_tok

            model_name_disp = msg.get("model_name") or getattr(context.llm_service, "model_name", "Unknown model")
            provider_disp = msg.get("llm_provider") or getattr(context.llm_service, "provider_name", "Groq")

            cost_est = context.cost_service.calculate_cost(
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                model_name=model_name_disp,
                provider=provider_disp,
                total_tokens=t_tok,
            )

            # Format tokens
            in_tok_str = f"{p_tok:,}" if p_tok is not None else "Unavailable"
            out_tok_str = f"{c_tok:,}" if c_tok is not None else "Unavailable"
            tot_tok_str = f"{t_tok:,}" if t_tok is not None else "Unavailable"

            # Format latencies
            ret_lat = msg.get("retrieval_latency_ms")
            gen_lat = msg.get("generation_latency_ms")
            ret_lat_str = format_latency(ret_lat, fallback="Unavailable")
            gen_lat_str = format_latency(gen_lat, fallback="Unavailable", treat_zero_as_unavailable=True)
            tot_lat_str = format_latency(tot_lat, fallback="Unavailable", treat_zero_as_unavailable=True)

            # Format cost
            if cost_est.is_available and cost_est.cost_usd is not None and cost_est.cost_inr is not None:
                cost_badge_html = (
                    f'<span style="font-weight:700;color:var(--accent);font-size:12.5px;">{cost_est.formatted_usd}</span> '
                    f'<span style="color:var(--text-muted);font-size:11px;">({cost_est.formatted_inr})</span>'
                )
            else:
                cost_badge_html = '<span style="color:var(--text-muted);font-style:italic;">Cost unavailable</span>'

            # Clean, compact per-query telemetry card below the generated answer
            telemetry_card_html = f"""
            <div style="margin-top:10px;background:var(--card-hi);border:1px solid var(--border);border-radius:8px;padding:9px 13px;font-size:12px;color:var(--text);box-shadow:var(--shadow-sm);">
              <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:8px;padding-bottom:7px;margin-bottom:7px;border-bottom:1px solid var(--border);">
                <div style="display:flex;align-items:center;gap:6px;">
                  <span style="color:var(--text-muted);font-weight:600;">💰 Estimated Cost:</span>
                  {cost_badge_html}
                </div>
                <div style="display:flex;align-items:center;gap:7px;font-size:11.5px;color:var(--text-muted);">
                  <span>In: <strong style="color:var(--text);font-family:monospace;">{in_tok_str}</strong></span>
                  <span style="color:var(--border);">·</span>
                  <span>Out: <strong style="color:var(--text);font-family:monospace;">{out_tok_str}</strong></span>
                  <span style="color:var(--border);">·</span>
                  <span>Total Tokens: <strong style="color:var(--accent);font-family:monospace;">{tot_tok_str}</strong></span>
                </div>
              </div>
              <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:8px;font-size:11.5px;color:var(--text-muted);">
                <div style="display:flex;align-items:center;gap:6px;">
                  <span>🤖 <strong>Model:</strong> <code style="background:var(--card);padding:1px 5px;border-radius:4px;color:var(--text);font-size:11px;">{model_name_disp}</code></span>
                  <span>·</span>
                  <span>🏛️ <strong>Provider:</strong> <span style="color:var(--text);font-weight:600;">{provider_disp}</span></span>
                </div>
                <div style="display:flex;align-items:center;gap:7px;">
                  <span>⏱️ <strong>Retrieval:</strong> <span style="color:var(--text);">{ret_lat_str}</span></span>
                  <span>·</span>
                  <span>⚡ <strong>Generation:</strong> <span style="color:var(--text);">{gen_lat_str}</span></span>
                  <span>·</span>
                  <span>⌛ <strong>Total Response Time:</strong> <strong style="color:var(--text);">{tot_lat_str}</strong></span>
                </div>
              </div>
            </div>
            """

            render_html(
                f'<div style="display:flex;justify-content:flex-start;margin:10px 0;">'
                f'<div style="max-width:85%;width:100%;">'
                f'<div style="font-size:12px;color:var(--text-muted);margin-bottom:6px;display:flex;align-items:center;gap:6px;">'
                f'<strong>AI Assistant</strong> {badge_html}{mock_notice}{latency_str}'
                f'</div>'
                f'<div class="chat-assistant">{format_chat_markdown(msg["content"])}</div>'
                f'{citations_html}'
                f'{telemetry_card_html}'
                f'</div></div>'
            )

            # Technical error details if any
            errors = msg.get("errors") or []
            if errors:
                with st.expander("Technical Error Details", expanded=False):
                    for err in errors:
                        st.caption(f"• {err}")

            # Expandable evidence details and performance metrics
            has_evidence = bool(citations or msg.get("graph_evidence"))
            has_metrics = (ret_lat is not None) or (gen_lat is not None) or (tot_lat is not None) or (p_tok is not None)

            if has_evidence or has_metrics:
                total_ev = len(citations)
                graph_ev = msg.get("graph_evidence")
                graph_rels = (graph_ev or {}).get("relationships", [])

                exp_title = (
                    f"Sources & Telemetry ({total_ev} citations"
                    + (f", {len(graph_rels)} graph paths" if graph_rels else "")
                    + ")"
                ) if has_evidence else "Performance & Cost Telemetry"

                with st.expander(exp_title):
                    if citations:
                        st.markdown("**Document Sources & Relevance Scores:**")
                        for c in citations:
                            score_val = c.get("score")
                            score_str = (
                                f" · Score: {format_metric(score_val, fallback='', precision=3)}"
                                if score_val is not None
                                else ""
                            )
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

                    st.markdown("**Token Consumption & Cost:**")
                    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
                    with t_col1:
                        st.metric("Input Tokens", in_tok_str)
                    with t_col2:
                        st.metric("Output Tokens", out_tok_str)
                    with t_col3:
                        st.metric("Total Tokens", tot_tok_str)
                    with t_col4:
                        cost_metric_text = (
                            f"{cost_est.formatted_usd} ({cost_est.formatted_inr})"
                            if cost_est.is_available
                            else "Cost unavailable"
                        )
                        st.metric("Estimated Cost", cost_metric_text)

                    if has_metrics:
                        st.markdown("**Performance Breakdown:**")
                        p_col1, p_col2, p_col3 = st.columns(3)
                        with p_col1:
                            st.metric(
                                "Retrieval Latency",
                                ret_lat_str,
                            )
                        with p_col2:
                            st.metric(
                                "Generation Latency",
                                gen_lat_str,
                            )
                        with p_col3:
                            st.metric(
                                "Total Response Time",
                                tot_lat_str,
                            )

# ── Suggested Questions Chips ────────────────────────────────────────────────
st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
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

# ── Message Input Area ────────────────────────────────────────────────────────
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

    with st.spinner("Retrieving evidence and synthesizing grounded response..."):
        result = context.query_service.answer(
            question=question,
            manual_strategy=manual_strategy,
            document_ids=kb_doc_ids,
            top_k=context.settings.default_top_k,
        )

    # Groundedness notice if no evidence found
    answer_text = result.answer
    if not result.errors and not result.citations and not (result.graph_evidence and result.graph_evidence.entities):
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
            llm_provider=result.llm_provider,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cost_usd=result.cost_usd,
            cost_inr=result.cost_inr,
            warnings=result.warnings or [],
            errors=result.errors or [],
            mock_mode=bool(result.warnings and "mock_mode_active" in result.warnings),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    )

    st.rerun()
