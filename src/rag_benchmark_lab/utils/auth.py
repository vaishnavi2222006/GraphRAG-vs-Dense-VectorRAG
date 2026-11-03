"""Authentication helpers for Streamlit session state management."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..schemas.auth import UserRecord


SESSION_USER_KEY = "_rag_auth_user_id"
SESSION_USER_NAME_KEY = "_rag_auth_user_name"
SESSION_USER_EMAIL_KEY = "_rag_auth_user_email"


def get_current_user_id() -> str | None:
    """Return the logged-in user ID from Streamlit session state, or None."""
    import streamlit as st
    return st.session_state.get(SESSION_USER_KEY)


def get_current_user_name() -> str | None:
    import streamlit as st
    return st.session_state.get(SESSION_USER_NAME_KEY)


def is_authenticated() -> bool:
    return get_current_user_id() is not None


def login_user(user_id: str, full_name: str, email: str) -> None:
    """Store user credentials in Streamlit session state."""
    import streamlit as st
    st.session_state[SESSION_USER_KEY] = user_id
    st.session_state[SESSION_USER_NAME_KEY] = full_name
    st.session_state[SESSION_USER_EMAIL_KEY] = email


def logout_user() -> None:
    """Clear user session."""
    import streamlit as st
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def require_auth() -> str:
    """
    Enforce authentication. If not logged in, redirects to the login page
    and stops execution. Returns the user_id if authenticated.
    """
    import streamlit as st
    user_id = get_current_user_id()
    if not user_id:
        st.warning("Please sign in to access this page.")
        # Attempt switch_page first
        for target in ("pages/0_Login.py", "0_Login.py"):
            try:
                st.switch_page(target)
                st.stop()
            except Exception:
                pass
        # Fallback to page_link
        for target in ("pages/0_Login.py", "0_Login.py"):
            try:
                st.page_link(target, label="Sign In", icon=":material/login:")
                st.stop()
            except Exception:
                pass
        if st.button("Sign In", type="primary", key="require_auth_signin_btn"):
            try:
                st.switch_page("pages/0_Login.py")
            except Exception:
                st.switch_page("0_Login.py")
        st.stop()
    return user_id
