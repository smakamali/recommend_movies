"""
User registration form component.
"""

import streamlit as st

from poc.data_loader import get_occupations_from_dataset

OCCUPATIONS = get_occupations_from_dataset()


def render_user_form(initial_data: dict | None = None, is_edit: bool = False) -> dict | None:
    """
    Render user registration or edit form.

    Args:
        initial_data: Pre-fill form with this data (name, age, gender, occupation, zip_code)
        is_edit: If True, show "Update Profile" button; else "Create Profile"

    Returns:
        Form data dict if submitted, else None.
    """
    default_name = initial_data.get("name") or "" if initial_data else ""
    default_age = initial_data.get("age", 25) if initial_data else 25
    default_gender = initial_data.get("gender", "M") if initial_data else "M"
    default_occupation = initial_data.get("occupation", OCCUPATIONS[0]) if initial_data else OCCUPATIONS[0]
    default_zip = initial_data.get("zip_code") or "" if initial_data else ""

    with st.form("user_registration_form"):
        st.subheader("Update your profile" if is_edit else "Create your profile")
        name = st.text_input("Name", value=default_name, placeholder="Your display name", max_chars=100)
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=default_age,
            help="Your age",
        )
        gender = st.radio(
            "Gender",
            options=["M", "F"],
            index=0 if default_gender == "M" else 1,
            format_func=lambda x: "Male" if x == "M" else "Female",
        )
        occ_idx = OCCUPATIONS.index(default_occupation) if default_occupation in OCCUPATIONS else 0
        occupation = st.selectbox("Occupation", options=OCCUPATIONS, index=occ_idx)
        zip_code = st.text_input("Zip code (optional)", value=default_zip, placeholder="e.g. 12345", max_chars=10)
        submitted = st.form_submit_button("Update Profile" if is_edit else "Create Profile")
        if submitted:
            return {
                "name": name.strip() or None,
                "age": age,
                "gender": gender,
                "occupation": occupation,
                "zip_code": zip_code.strip() or None,
            }
    return None
