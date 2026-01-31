"""
Movie display card component.
"""

import streamlit as st
import json


def render_movie_card(
    movie_id: int,
    title: str,
    release_year: int | None,
    genres: str,
    score: float,
    show_rating_widget: bool = False,
    on_rate: callable = None,
) -> None:
    """
    Render a movie card with optional rating widget.

    Args:
        movie_id: Movie ID
        title: Movie title
        release_year: Release year
        genres: Genres (JSON array string)
        score: Predicted/recommendation score
        show_rating_widget: Whether to show star rating widget
        on_rate: Callback(user_id, movie_id, rating) when user rates
    """
    try:
        genre_list = json.loads(genres) if genres else []
        genre_str = ", ".join(genre_list) if isinstance(genre_list, list) else str(genres)
    except (json.JSONDecodeError, TypeError):
        genre_str = str(genres)

    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{title}**")
            meta = []
            if release_year:
                meta.append(str(release_year))
            if genre_str:
                meta.append(genre_str)
            if meta:
                st.caption(" | ".join(meta))
            st.caption(f"Score: {score:.2f}")
        with col2:
            if show_rating_widget and on_rate:
                from app.ui.components.rating_widget import render_rating_widget
                render_rating_widget(movie_id, on_rate)
        st.divider()
