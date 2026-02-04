"""
Star rating widget component.
"""

import streamlit as st


def render_rating_widget(movie_id: int, on_rate: callable, current_rating: float | None = None) -> None:
    """
    Render 5-star rating widget.

    Args:
        movie_id: Movie ID to rate
        on_rate: Callback(user_id, movie_id, rating) when user submits rating
        current_rating: User's existing rating (if any) to pre-fill; None shows "Rate me!"
    """
    user_id = st.session_state.get("user_id")
    if not user_id:
        st.caption("Sign in to rate")
        return

    # Use 0 as sentinel for "Rate me!"; 1-5 for actual ratings
    options = [0, 1, 2, 3, 4, 5]

    def format_rating(x):
        return "Rate me!" if x == 0 else f"{x} ★"

    # When unrated: default to "Rate me!" (index 0). When rated: default to user's rating.
    if current_rating is not None:
        default_index = min(max(int(round(current_rating)), 1), 5)  # 1-5 maps to index 1-5
    else:
        default_index = 0

    rating = st.selectbox(
        "Your rating",
        options=options,
        index=default_index,
        format_func=format_rating,
        key=f"rate_{movie_id}",
        label_visibility="visible",
    )
    if st.button("Submit", key=f"submit_rate_{movie_id}"):
        if rating == 0:
            st.error("Please select a rating (1-5)")
        else:
            try:
                on_rate(user_id, movie_id, float(rating))
                st.success("Rating saved!")
            except Exception as e:
                st.error(f"Failed to save rating: {e}")
