"""RAG Benchmark Lab — Home Workspace & Product Overview."""
from __future__ import annotations

import streamlit as st

from rag_benchmark_lab.app import get_context
from rag_benchmark_lab.utils.auth import get_current_user_name, is_authenticated
from rag_benchmark_lab.utils.ui import (
    inject_custom_css,
    render_dev_mode_banner,
    render_topnav,
)

st.set_page_config(
    page_title="RAG Benchmark Lab",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_topnav(active_page="Home")

context = get_context()
is_auth = is_authenticated()
user_name = get_current_user_name() or "Researcher"
user_id = st.session_state.get("_rag_auth_user_id", "")

render_dev_mode_banner(context.settings.llm_mock_mode)

user_kbs = []
if is_auth and user_id:
    user_kbs = context.user_registry.list_knowledge_bases(user_id)

# ── 1. Hero Section ──────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align: center; padding: 36px 16px 20px; max-width: 860px; margin: 0 auto;">
      <h1 style="font-size: 44px; font-weight: 800; color: #F5F7FA; margin: 0 0 16px 0; letter-spacing: -0.035em; line-height: 1.15;">
        Make Your Documents Searchable.<br/>
        <span style="color: #8B7CFF;">Compare Your RAG.</span>
      </h1>
      <p style="font-size: 16px; color: #A0AABC; margin: 0 auto 28px; line-height: 1.6; max-width: 660px;">
        Explore how Vector RAG, GraphRAG, and Hybrid RAG retrieve evidence and answer questions from your documents.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Hero Buttons ─────────────────────────────────────────────────────────────
c_sp1, c_b1, c_b2, c_sp2 = st.columns([1.6, 1.4, 1.4, 1.6], vertical_alignment="center")
with c_b1:
    if not is_auth:
        if st.button("Get Started", type="primary", use_container_width=True, key="home_hero_get_started"):
            st.switch_page("pages/0_Login.py")
    elif not user_kbs:
        if st.button("Get Started", type="primary", use_container_width=True, key="home_hero_get_started"):
            st.switch_page("pages/2_Documents.py")
    else:
        if st.button("Get Started", type="primary", use_container_width=True, key="home_hero_get_started"):
            st.switch_page("pages/3_Chat.py")

with c_b2:
    if st.button("Explore RAG Methods", use_container_width=True, key="home_hero_explore"):
        st.switch_page("pages/4_Compare.py")

st.markdown("<div style='height: 36px;'></div>", unsafe_allow_html=True)

# ── 2. Product Illustration: Connected RAG Workflow ──────────────────────────
with st.container(border=True):
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
          <span style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #8B7CFF;">System Architecture</span>
          <h3 style="font-size: 18px; font-weight: 700; color: #F5F7FA; margin: 4px 0 0 0;">Unified RAG Retrieval & Synthesis Pipeline</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_doc, c_arr1, c_ret, c_arr2, c_ans = st.columns([1.2, 0.3, 3.4, 0.3, 1.2], vertical_alignment="center")

    with c_doc:
        with st.container(border=True):
            st.markdown(
                """
                <div style="text-align: center; padding: 8px 4px;">
                  <div style="display: inline-flex; align-items: center; justify-content: center; width: 36px; height: 36px; border-radius: 8px; background: rgba(57, 198, 181, 0.12); border: 1px solid rgba(57, 198, 181, 0.3); margin-bottom: 8px;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#39C6B5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                      <polyline points="14 2 14 8 20 8"></polyline>
                      <line x1="16" y1="13" x2="8" y2="13"></line>
                      <line x1="16" y1="17" x2="8" y2="17"></line>
                      <polyline points="10 9 9 9 8 9"></polyline>
                    </svg>
                  </div>
                  <div style="font-size: 13.5px; font-weight: 700; color: #39C6B5;">DOCUMENTS</div>
                  <div style="font-size: 11.5px; color: #A0AABC; margin-top: 4px;">PDF · DOCX · TXT</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with c_arr1:
        st.markdown("<div style='text-align: center; color: #3D4E6A; font-size: 20px; font-weight: 700;'>→</div>", unsafe_allow_html=True)

    with c_ret:
        r1, r2, r3 = st.columns(3)
        with r1:
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="text-align: center; padding: 6px 2px;">
                      <div style="display: inline-flex; align-items: center; justify-content: center; width: 32px; height: 32px; border-radius: 6px; background: rgba(96, 165, 250, 0.12); border: 1px solid rgba(96, 165, 250, 0.3); margin-bottom: 6px;">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#60A5FA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                          <circle cx="12" cy="12" r="10"></circle>
                          <line x1="2" y1="12" x2="22" y2="12"></line>
                          <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
                        </svg>
                      </div>
                      <div style="font-size: 13px; font-weight: 700; color: #60A5FA;">Vector RAG</div>
                      <div style="font-size: 11px; color: #A0AABC; margin-top: 4px;">Dense Embeddings</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with r2:
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="text-align: center; padding: 6px 2px;">
                      <div style="display: inline-flex; align-items: center; justify-content: center; width: 32px; height: 32px; border-radius: 6px; background: rgba(167, 139, 250, 0.12); border: 1px solid rgba(167, 139, 250, 0.3); margin-bottom: 6px;">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#A78BFA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                          <circle cx="18" cy="5" r="3"></circle>
                          <circle cx="6" cy="12" r="3"></circle>
                          <circle cx="18" cy="19" r="3"></circle>
                          <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line>
                          <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
                        </svg>
                      </div>
                      <div style="font-size: 13px; font-weight: 700; color: #A78BFA;">GraphRAG</div>
                      <div style="font-size: 11px; color: #A0AABC; margin-top: 4px;">Entity Traversal</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with r3:
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="text-align: center; padding: 6px 2px;">
                      <div style="display: inline-flex; align-items: center; justify-content: center; width: 32px; height: 32px; border-radius: 6px; background: rgba(57, 198, 181, 0.12); border: 1px solid rgba(57, 198, 181, 0.3); margin-bottom: 6px;">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#39C6B5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                          <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                          <polyline points="2 17 12 22 22 17"></polyline>
                          <polyline points="2 12 12 17 22 12"></polyline>
                        </svg>
                      </div>
                      <div style="font-size: 13px; font-weight: 700; color: #39C6B5;">Hybrid RAG</div>
                      <div style="font-size: 11px; color: #A0AABC; margin-top: 4px;">Rank Fusion (RRF)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with c_arr2:
        st.markdown("<div style='text-align: center; color: #3D4E6A; font-size: 20px; font-weight: 700;'>→</div>", unsafe_allow_html=True)

    with c_ans:
        with st.container(border=True):
            st.markdown(
                """
                <div style="text-align: center; padding: 8px 4px;">
                  <div style="display: inline-flex; align-items: center; justify-content: center; width: 36px; height: 36px; border-radius: 8px; background: rgba(139, 124, 255, 0.12); border: 1px solid rgba(139, 124, 255, 0.3); margin-bottom: 8px;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#8B7CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                    </svg>
                  </div>
                  <div style="font-size: 13.5px; font-weight: 700; color: #8B7CFF;">ANSWERS</div>
                  <div style="font-size: 11.5px; color: #A0AABC; margin-top: 4px;">Grounded Citations</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)

# ── 3. Feature Cards ─────────────────────────────────────────────────────────
st.markdown("### Retrieval Architectures")
st.caption("Three complementary approaches to finding ground-truth evidence in your documents.")

col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    with st.container(border=True):
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
              <span style="font-size: 16px; font-weight: 700; color: #F5F7FA;">VECTOR RAG</span>
              <span class="badge badge-vector">Semantic</span>
            </div>
            <p style="font-size: 13.5px; color: #A0AABC; line-height: 1.55; min-height: 48px; margin-bottom: 16px;">
              Retrieve semantically similar passages from your documents.
            </p>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explore Vector RAG", use_container_width=True, key="feat_btn_vector"):
            st.switch_page("pages/4_Compare.py")

with col_f2:
    with st.container(border=True):
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
              <span style="font-size: 16px; font-weight: 700; color: #F5F7FA;">GRAPHRAG</span>
              <span class="badge badge-graph">Relational</span>
            </div>
            <p style="font-size: 13.5px; color: #A0AABC; line-height: 1.55; min-height: 48px; margin-bottom: 16px;">
              Discover entities and relationships connecting information across documents.
            </p>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explore GraphRAG", use_container_width=True, key="feat_btn_graph"):
            st.switch_page("pages/4_Compare.py")

with col_f3:
    with st.container(border=True):
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
              <span style="font-size: 16px; font-weight: 700; color: #F5F7FA;">HYBRID RAG</span>
              <span class="badge badge-hybrid">Combined</span>
            </div>
            <p style="font-size: 13.5px; color: #A0AABC; line-height: 1.55; min-height: 48px; margin-bottom: 16px;">
              Combine vector retrieval and graph relationships to gather evidence.
            </p>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explore Hybrid RAG", use_container_width=True, key="feat_btn_hybrid"):
            st.switch_page("pages/4_Compare.py")

st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)

# ── 4. How It Works (01, 02, 03) ─────────────────────────────────────────────
st.markdown("### How It Works")
st.caption("A structured three-step workflow from raw documents to measurable benchmark insights.")

c_s1, c_s2, c_s3 = st.columns(3)

with c_s1:
    with st.container(border=True):
        st.markdown(
            """
            <div style="font-size: 20px; font-weight: 800; color: #8B7CFF; margin-bottom: 6px;">01 — Upload</div>
            <div style="font-size: 13.5px; color: #A0AABC; line-height: 1.5; min-height: 42px;">
              Add your PDF, DOCX, or TXT documents.
            </div>
            """,
            unsafe_allow_html=True,
        )

with c_s2:
    with st.container(border=True):
        st.markdown(
            """
            <div style="font-size: 20px; font-weight: 800; color: #39C6B5; margin-bottom: 6px;">02 — Ask</div>
            <div style="font-size: 13.5px; color: #A0AABC; line-height: 1.5; min-height: 42px;">
              Ask questions and inspect the supporting evidence.
            </div>
            """,
            unsafe_allow_html=True,
        )

with c_s3:
    with st.container(border=True):
        st.markdown(
            """
            <div style="font-size: 20px; font-weight: 800; color: #35C98A; margin-bottom: 6px;">03 — Evaluate</div>
            <div style="font-size: 13.5px; color: #A0AABC; line-height: 1.5; min-height: 42px;">
              Compare retrieval strategies and measure their performance.
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)

# ── 5. Existing Workspace Status ──────────────────────────────────────────────
st.markdown("### Research Workspace")
st.caption("Current status and quick navigation for your active research session.")

if is_auth:
    if user_kbs:
        active_kb_id = st.session_state.get("active_kb_id", user_kbs[0].kb_id)
        active_kb = next((kb for kb in user_kbs if kb.kb_id == active_kb_id), user_kbs[0])

        real_counts = context.health_service.get_real_counts()
        kb_doc_ids = context.user_registry.get_kb_document_ids(active_kb.kb_id, user_id)
        all_docs = context.registry.list_documents()
        docs = [d for d in all_docs if d["document_id"] in set(kb_doc_ids)]

        # Determine true processing status
        if not docs:
            processing_status = "No documents"
        elif any(d.get("status", "").lower() == "failed" for d in docs):
            processing_status = "Attention required"
        else:
            processing_status = "Ready"

        # Determine latest evaluation status
        eval_runs = context.result_store.list_runs()
        if eval_runs:
            latest_run = eval_runs[0]
            eval_date = latest_run.get("timestamp", "")[:10]
            strategies_run = ", ".join(latest_run.get("strategies", []))
            latest_eval_status = f"{len(eval_runs)} runs recorded ({eval_date})"
        else:
            latest_eval_status = "No runs yet"

        with st.container(border=True):
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; flex-wrap: wrap; gap: 8px;">
                  <div>
                    <span style="font-size: 11.5px; color: #A0AABC; text-transform: uppercase; letter-spacing: 0.05em;">Active Knowledge Base</span>
                    <div style="font-size: 18px; font-weight: 700; color: #F5F7FA;">{active_kb.name}</div>
                  </div>
                  <div>
                    <span class="badge badge-success">{processing_status}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Documents", len(docs))
            with m2:
                st.metric("Indexed Chunks", real_counts.get("chunks", 0))
            with m3:
                st.metric("Graph Entities", real_counts.get("entities", 0))
            with m4:
                st.metric("Evaluation Status", latest_eval_status)

        c_act1, c_act2, c_act3 = st.columns(3)
        with c_act1:
            if st.button("Manage Documents", use_container_width=True, key="home_ws_manage_docs"):
                st.switch_page("pages/2_Documents.py")
        with c_act2:
            if st.button("Start Chat", type="primary", use_container_width=True, key="home_ws_start_chat"):
                st.switch_page("pages/3_Chat.py")
        with c_act3:
            if st.button("Run Evaluation", use_container_width=True, key="home_ws_run_eval"):
                st.switch_page("pages/5_Evaluation.py")

    else:
        with st.container(border=True):
            st.markdown(
                """
                <div style="text-align: center; padding: 24px 16px;">
                  <h4 style="font-size: 16px; font-weight: 700; color: #F5F7FA; margin: 0 0 6px 0;">No Knowledge Bases Created</h4>
                  <p style="font-size: 13.5px; color: #A0AABC; max-width: 480px; margin: 0 auto 18px; line-height: 1.5;">
                    Create your first knowledge base in Documents to start uploading PDF, DOCX, or TXT documents.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            c_k1, c_k2, c_k3 = st.columns([1, 1.2, 1])
            with c_k2:
                if st.button("Create Knowledge Base", type="primary", use_container_width=True, key="home_btn_create_kb"):
                    st.switch_page("pages/2_Documents.py")

else:
    with st.container(border=True):
        st.markdown(
            """
            <div style="text-align: center; padding: 28px 16px;">
              <h4 style="font-size: 18px; font-weight: 700; color: #F5F7FA; margin: 0 0 8px 0;">Get Started with RAG Benchmark Lab</h4>
              <p style="font-size: 14px; color: #A0AABC; max-width: 520px; margin: 0 auto 20px; line-height: 1.6;">
                Sign in to upload research documents, build connected knowledge graphs, query using Vector RAG and GraphRAG, and evaluate empirical performance.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c_s_sp1, c_s_btn, c_s_sp2 = st.columns([1.5, 1.2, 1.5])
        with c_s_btn:
            if st.button("Sign In to Workspace", type="primary", use_container_width=True, key="home_btn_signin"):
                st.switch_page("pages/0_Login.py")
