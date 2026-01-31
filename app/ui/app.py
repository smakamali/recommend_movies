"""
Streamlit main app for GraphSAGE Recommender System.

Run: streamlit run app/ui/app.py --server.port 8501
"""

import streamlit as st

from app.ui.utils.session_state import init_session_state

st.set_page_config(
    page_title="GraphSAGE Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()

st.title("🎬 GraphSAGE Movie Recommender")
st.markdown("Get personalized movie recommendations powered by Graph Neural Networks.")

# Check API health
try:
    from app.ui.utils.api_client import health_check
    health = health_check()
    if health.get("status") == "healthy":
        st.success("API connected")
    else:
        st.warning("API may not be fully ready")
except Exception as e:
    st.error(f"API not available: {e}")
    st.info("Start the API with: uvicorn app.api.main:app --host 0.0.0.0 --port 8000")

st.divider()

# Navigation
user_id = st.session_state.get("user_id")
if user_id:
    st.subheader(f"Welcome, User #{user_id}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 My Profile", use_container_width=True):
            st.switch_page("pages/1_user_profile.py")
    with col2:
        if st.button("🎯 Get Recommendations", use_container_width=True):
            st.switch_page("pages/2_recommendations.py")
else:
    st.info("Create your profile to get started.")
    if st.button("Create Profile"):
        st.switch_page("pages/1_user_profile.py")
