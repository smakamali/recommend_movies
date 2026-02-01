"""
User profile page - registration and profile view.
"""

import streamlit as st
import sys
from pathlib import Path

# Ensure project root in path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ui.utils.api_client import create_user, get_user, get_user_ratings, update_user
from app.ui.utils.session_state import get_current_user_id, set_current_user, init_session_state
from app.ui.components.user_form import render_user_form

init_session_state()
st.title("📋 My Profile")

user_id = get_current_user_id()
editing_profile = st.session_state.get("editing_profile", False)

if user_id:
    # Show profile and rating history (or edit form)
    try:
        user = get_user(user_id)

        if editing_profile:
            form_data = render_user_form(initial_data=user, is_edit=True)
            if form_data:
                try:
                    updated = update_user(
                        user_id,
                        name=form_data.get("name"),
                        age=form_data["age"],
                        gender=form_data["gender"],
                        occupation=form_data["occupation"],
                        zip_code=form_data.get("zip_code"),
                    )
                    set_current_user(user_id, updated)
                    st.session_state["editing_profile"] = False
                    st.success("Profile updated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to update profile: {e}")
            if st.button("Cancel"):
                st.session_state["editing_profile"] = False
                st.rerun()
        else:
            st.subheader("Profile")
            if user.get("name"):
                st.markdown(f"**{user['name']}**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Age", user.get("age", "-"))
            with col2:
                st.metric("Gender", "Male" if user.get("gender") == "M" else "Female")
            with col3:
                st.metric("Occupation", user.get("occupation", "-"))

            if st.button("Edit Profile"):
                st.session_state["editing_profile"] = True
                st.rerun()

            ratings_data = get_user_ratings(user_id)
            ratings = ratings_data.get("ratings", [])
            st.subheader("My Ratings")
            if ratings:
                for r in ratings[:20]:
                    title = r.get("title") or f"Movie #{r['movie_id']}"
                    st.write(f"{title}: {r['rating']} ★")
                if len(ratings) > 20:
                    st.caption(f"... and {len(ratings) - 20} more")
            else:
                st.info("No ratings yet. Go to Recommendations to rate movies!")

            st.divider()
            if st.button("Go to Recommendations"):
                st.switch_page("pages/2_recommendations.py")
    except Exception as e:
        st.error(f"Failed to load profile: {e}")
        user_id = None

if not user_id:
    # Show registration form
    form_data = render_user_form()
    if form_data:
        try:
            user = create_user(
                age=form_data["age"],
                gender=form_data["gender"],
                occupation=form_data["occupation"],
                zip_code=form_data.get("zip_code"),
                name=form_data.get("name"),
            )
            set_current_user(user["user_id"], user)
            st.success("Profile created! Redirecting to recommendations...")
            st.balloons()
            st.switch_page("pages/2_recommendations.py")
        except Exception as e:
            st.error(f"Failed to create profile: {e}")
