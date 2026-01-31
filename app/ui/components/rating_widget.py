"""
Star rating widget component.
"""

import streamlit as st


def render_rating_widget(movie_id: int, on_rate: callable) -> None:
    """
    Render 5-star rating widget.

    Args:
        movie_id: Movie ID to rate
        on_rate: Callback(user_id, movie_id, rating) when user submits rating
    """
    user_id = st.session_state.get("user_id")
    if not user_id:
        st.caption("Sign in to rate")
        return

    rating = st.selectbox(
        "Rate",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: f"{x} ★",
        key=f"rate_{movie_id}",
        label_visibility="collapsed",
    )
    if st.button("Submit", key=f"submit_rate_{movie_id}"):
        try:
            on_rate(user_id, movie_id, float(rating))
            st.success("Rating saved!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to save rating: {e}")
