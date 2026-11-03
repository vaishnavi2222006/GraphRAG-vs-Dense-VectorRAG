"""RAG Benchmark Lab — Authentication (Sign In & Register)."""
from __future__ import annotations

import streamlit as st

from rag_benchmark_lab.app import get_context
from rag_benchmark_lab.utils.auth import is_authenticated, login_user
from rag_benchmark_lab.utils.ui import inject_custom_css, render_html, render_topnav

st.set_page_config(
    page_title="Sign In — RAG Benchmark Lab",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_topnav(active_page="Sign In")

# If already logged in, redirect directly to Home
if is_authenticated():
    st.success("You are already signed in.")
    try:
        st.switch_page("pages/1_Home.py")
    except Exception:
        try:
            st.switch_page("1_Home.py")
        except Exception:
            pass
    st.stop()

context = get_context()

col_l, col_center, col_r = st.columns([1.2, 2.2, 1.2])

with col_center:
    # ─── Geometric Logo & Hero Section ───────────────────────────────────────────
    render_html(
        """<div style="text-align:center;padding:24px 16px 16px;">
<div style="display:inline-flex;align-items:center;justify-content:center;width:48px;height:48px;background:var(--accent-light);border:1px solid var(--accent-border);border-radius:12px;margin-bottom:14px;">
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--accent);">
    <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
    <polyline points="2 17 12 22 22 17"></polyline>
    <polyline points="2 12 12 17 22 12"></polyline>
  </svg>
</div>
<h1 style="font-size:28px;font-weight:800;color:var(--text);margin:0 0 8px 0;letter-spacing:-0.03em;">
  RAG Benchmark Lab
</h1>
<p style="font-size:14px;color:var(--text-muted);margin:0 auto 16px;max-width:480px;line-height:1.55;">
  Empirical AI research workspace to explore, benchmark, and compare
  <span style="color:var(--vector);font-weight:600;">Vector RAG</span>,
  <span style="color:var(--graph);font-weight:600;">GraphRAG</span>, and
  <span style="color:var(--hybrid);font-weight:600;">Hybrid RAG</span>.
</p>
</div>"""
    )

    # ─── Auth Card ────────────────────────────────────────────────────────────────
    with st.container(border=True):
        tab_login, tab_register = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            with st.form("login_form", clear_on_submit=False):
                st.markdown("<h4 style='color:var(--text);margin-bottom:14px;font-size:15px;font-weight:700;'>Sign in to your research workspace</h4>", unsafe_allow_html=True)
                email_in = st.text_input("Email address", placeholder="researcher@example.com", key="login_email")
                password_in = st.text_input("Password", type="password", placeholder="••••••••", key="login_password")
                submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)

            if submitted:
                if not email_in or not password_in:
                    st.error("Please enter both email and password.")
                else:
                    user, err = context.user_registry.authenticate_user(email_in, password_in)
                    if user:
                        login_user(user.user_id, user.full_name, user.email)
                        st.success(f"Welcome back, {user.full_name}.")
                        st.switch_page("pages/1_Home.py")
                    else:
                        st.error(f"Authentication failed: {err}")

        with tab_register:
            with st.form("register_form", clear_on_submit=False):
                st.markdown("<h4 style='color:var(--text);margin-bottom:14px;font-size:15px;font-weight:700;'>Create your research account</h4>", unsafe_allow_html=True)
                full_name_in = st.text_input("Full name", placeholder="Dr. Jane Smith", key="reg_name")
                email_reg = st.text_input("Email address", placeholder="researcher@example.com", key="reg_email")
                password_reg = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Min. 8 characters (letters and digits)",
                    key="reg_password",
                )
                password_confirm = st.text_input("Confirm password", type="password", placeholder="Repeat password", key="reg_password2")
                submitted_reg = st.form_submit_button("Create Account", type="primary", use_container_width=True)

            if submitted_reg:
                errors: list[str] = []
                if password_reg != password_confirm:
                    errors.append("Passwords do not match.")
                if not errors:
                    user, reg_errors = context.user_registry.register_user(full_name_in, email_reg, password_reg)
                    errors.extend(reg_errors)
                if errors:
                    for err in errors:
                        st.error(f"Registration error: {err}")
                else:
                    login_user(user.user_id, user.full_name, user.email)
                    st.success(f"Account created. Welcome, {user.full_name}.")
                    st.switch_page("pages/1_Home.py")

        st.caption("All user documents, knowledge bases, and evaluation records are private to your account.")
