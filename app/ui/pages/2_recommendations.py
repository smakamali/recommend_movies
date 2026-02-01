"""
Recommendations page - personalized recommendations with inline rating.
"""

import streamlit as st
import sys
from pathlib import Path

# Ensure project root in path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ui.utils.api_client import (
    get_recommendations,
    refresh_recommendations,
    add_rating,
)
from app.ui.utils.session_state import get_current_user_id, init_session_state
from app.ui.components.movie_card import render_movie_card

init_session_state()

st.title("🎯 Recommendations")

user_id = get_current_user_id()

if not user_id:
    st.warning("Please create your profile first.")
    if st.button("Create Profile"):
        st.switch_page("pages/1_user_profile.py")
    st.stop()


def handle_rate(uid: int, mid: int, rating: float) -> None:
    """Callback when user submits a rating."""
    add_rating(uid, mid, rating)


# Initialize exclude_already_rated in session state if not present
if "exclude_already_rated" not in st.session_state:
    st.session_state["exclude_already_rated"] = False

# Options row: toggle + refresh button
col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
with col1:
    exclude_already_rated = st.toggle(
        "Exclude already rated",
        value=st.session_state["exclude_already_rated"],
        help="When on, movies you've rated are excluded from recommendations",
    )
    st.session_state["exclude_already_rated"] = exclude_already_rated
with col3:
    if st.button("🔄 Refresh Recommendations", use_container_width=True):
        try:
            refresh_recommendations(
                user_id,
                exclude_already_rated=st.session_state["exclude_already_rated"],
            )
            st.success("Recommendations refreshed!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to refresh: {e}")

st.divider()

try:
    data = get_recommendations(
        user_id,
        n=10,
        exclude_low_rated=True,
        exclude_already_rated=st.session_state["exclude_already_rated"],
    )
    recs = data.get("recommendations", [])
    if recs:
        st.subheader("Recommended for you")
        for r in recs:
            render_movie_card(
                movie_id=r["movie_id"],
                title=r["title"],
                release_year=r.get("release_year"),
                genres=r.get("genres", "[]"),
                score=r.get("score", 0),
                show_rating_widget=True,
                on_rate=handle_rate,
                user_rating=r.get("user_rating"),
            )
    else:
        st.info("No recommendations available. Rate some movies to improve your recommendations!")
except Exception as e:
    st.error(f"Failed to load recommendations: {e}")
    st.info("Make sure the API is running: uvicorn app.api.main:app --host 0.0.0.0 --port 8000")
