"""
User registration form component.
"""

import streamlit as st

from poc.data_loader import get_occupations_from_dataset

OCCUPATIONS = get_occupations_from_dataset()


def render_user_form() -> dict | None:
    """
    Render user registration form.
    Returns form data dict if submitted, else None.
    """
    with st.form("user_registration_form"):
        st.subheader("Create your profile")
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=25,
            help="Your age",
        )
        gender = st.radio("Gender", options=["M", "F"], format_func=lambda x: "Male" if x == "M" else "Female")
        occupation = st.selectbox("Occupation", options=OCCUPATIONS)
        zip_code = st.text_input("Zip code (optional)", placeholder="e.g. 12345", max_chars=10)
        submitted = st.form_submit_button("Create Profile")
        if submitted:
            return {
                "age": age,
                "gender": gender,
                "occupation": occupation,
                "zip_code": zip_code.strip() or None,
            }
    return None
