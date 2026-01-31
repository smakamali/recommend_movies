"""
Session state helpers for Streamlit.
"""

import streamlit as st


def get_current_user_id() -> int | None:
    """Get current user ID from session state."""
    return st.session_state.get("user_id")


def set_current_user(user_id: int, user_data: dict | None = None) -> None:
    """Set current user in session state."""
    st.session_state["user_id"] = user_id
    if user_data:
        st.session_state["user_data"] = user_data


def clear_current_user() -> None:
    """Clear current user from session state."""
    if "user_id" in st.session_state:
        del st.session_state["user_id"]
    if "user_data" in st.session_state:
        del st.session_state["user_data"]


def init_session_state() -> None:
    """Initialize session state keys if not present."""
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None
    if "user_data" not in st.session_state:
        st.session_state["user_data"] = None
    if "recommendations_refresh" not in st.session_state:
        st.session_state["recommendations_refresh"] = 0
