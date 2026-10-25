from __future__ import annotations

import streamlit as st

from rag_benchmark_lab.app import get_context
from rag_benchmark_lab.utils.auth import is_authenticated, login_user, require_auth
from rag_benchmark_lab.utils.ui import inject_custom_css

st.set_page_config(
    page_title="RAG Benchmark Lab — Login",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="collapsed",
)

inject_custom_css()

# Extra auth-specific styles
st.markdown("""
<style>
.auth-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 40px 44px;
    box-shadow: 0 4px 24px rgba(15,23,42,0.08);
    margin: 0 auto;
    max-width: 480px;
}
.auth-logo {
    text-align: center;
    margin-bottom: 32px;
}
.auth-logo h1 {
    font-size: 28px;
    font-weight: 800;
    color: #0f172a;
    margin: 0 0 6px 0;
    letter-spacing: -0.03em;
}
.auth-logo p {
    font-size: 13.5px;
    color: #64748b;
    margin: 0;
}
.auth-divider {
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    margin: 20px 0;
    position: relative;
}
.auth-divider::before {
    content: '';
    position: absolute;
    left: 0;
    top: 50%;
    width: 40%;
    height: 1px;
    background: #e2e8f0;
}
.auth-divider::after {
    content: '';
    position: absolute;
    right: 0;
    top: 50%;
    width: 40%;
    height: 1px;
    background: #e2e8f0;
}
.feature-pill {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    color: #475569;
    margin: 3px;
}
.feature-pill.blue { background: #eff6ff; border-color: #bfdbfe; color: #1d4ed8; }
.feature-pill.green { background: #f0fdf4; border-color: #bbf7d0; color: #15803d; }
.feature-pill.purple { background: #f5f3ff; border-color: #ddd6fe; color: #7c3aed; }
</style>
""", unsafe_allow_html=True)

# If already logged in, redirect to workspace
if is_authenticated():
    st.success("✅ You are already logged in. Redirecting to your workspace...")
    st.page_link("pages/1_Overview.py", label="→ Go to Dashboard", icon="🏠")
    st.stop()

context = get_context()

# ─── Hero Section ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 40px 0 32px 0;">
    <div style="font-size: 48px; margin-bottom: 12px;">🧪</div>
    <h1 style="font-size: 36px; font-weight: 800; color: #0f172a; margin: 0 0 10px 0; letter-spacing: -0.04em;">
        RAG Benchmark Lab
    </h1>
    <p style="font-size: 16px; color: #64748b; margin: 0 0 20px 0; max-width: 560px; margin-left: auto; margin-right: auto; line-height: 1.6;">
        Empirical evaluation platform for <strong>Dense Vector RAG</strong>, <strong>GraphRAG</strong>,
        and <strong>Hybrid RAG</strong> — with real Qdrant and Neo4j backends.
    </p>
    <div style="margin-bottom: 8px;">
        <span class="feature-pill blue">⚡ Dense Vector RAG</span>
        <span class="feature-pill green">🕸️ GraphRAG (Neo4j)</span>
        <span class="feature-pill purple">🔀 Hybrid RRF Fusion</span>
    </div>
    <div>
        <span class="feature-pill">📄 PDF / TXT / DOCX</span>
        <span class="feature-pill">📊 Benchmark Evaluation</span>
        <span class="feature-pill">🔒 Private Workspaces</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Auth Tabs ────────────────────────────────────────────────────────────────
tab_login, tab_register = st.tabs(["🔐 Login", "✨ Create Account"])

with tab_login:
    with st.form("login_form", clear_on_submit=False):
        st.markdown("#### Sign in to your workspace")
        email_in = st.text_input("Email address", placeholder="you@example.com", key="login_email")
        password_in = st.text_input("Password", type="password", placeholder="••••••••", key="login_password")
        submitted = st.form_submit_button("🔐 Sign In", type="primary", use_container_width=True)

    if submitted:
        if not email_in or not password_in:
            st.error("Please enter both email and password.")
        else:
            user, err = context.user_registry.authenticate_user(email_in, password_in)
            if user:
                login_user(user.user_id, user.full_name, user.email)
                st.success(f"✅ Welcome back, **{user.full_name}**!")
                st.rerun()
            else:
                st.error(f"❌ {err}")

with tab_register:
    with st.form("register_form", clear_on_submit=False):
        st.markdown("#### Create your research account")
        full_name_in = st.text_input("Full name", placeholder="Jane Smith", key="reg_name")
        email_reg = st.text_input("Email address", placeholder="you@example.com", key="reg_email")
        password_reg = st.text_input("Password", type="password", placeholder="Min. 8 chars, upper + lower + digit", key="reg_password")
        password_confirm = st.text_input("Confirm password", type="password", placeholder="Repeat password", key="reg_password2")
        submitted_reg = st.form_submit_button("✨ Create Account", type="primary", use_container_width=True)

    if submitted_reg:
        errors: list[str] = []
        if password_reg != password_confirm:
            errors.append("Passwords do not match.")
        if not errors:
            user, reg_errors = context.user_registry.register_user(full_name_in, email_reg, password_reg)
            errors.extend(reg_errors)
        if errors:
            for err in errors:
                st.error(f"❌ {err}")
        else:
            login_user(user.user_id, user.full_name, user.email)
            st.success(f"🎉 Account created! Welcome, **{user.full_name}**!")
            st.balloons()
            st.rerun()

    st.caption("🔒 Passwords are hashed and never stored in plaintext. Your documents are private to your account.")

# ─── Architecture Overview ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🏛️ Research Architecture")

cols = st.columns(3)
with cols[0]:
    st.markdown("""
    **⚡ Dense Vector RAG**

    Retrieves passages using semantic similarity in high-dimensional embedding space.
    Ideal for direct factual queries and vocabulary mismatches.

    *Powered by: Qdrant + Sentence Transformers*
    """)
with cols[1]:
    st.markdown("""
    **🕸️ GraphRAG**

    Performs multi-hop entity and relationship traversal over structured knowledge graphs.
    Solves relational and multi-document reasoning.

    *Powered by: Neo4j + Entity Extraction*
    """)
with cols[2]:
    st.markdown("""
    **🔀 Hybrid RAG (RRF)**

    Fuses dense vector semantic candidates with graph paths using Reciprocal Rank Fusion.
    Optimizes recall for complex queries.

    *Powered by: Qdrant + Neo4j + RRF k=60*
    """)
