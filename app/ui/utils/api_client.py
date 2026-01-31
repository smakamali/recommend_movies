"""
FastAPI client wrapper for Streamlit UI.
"""

import os
import requests


def get_api_base_url() -> str:
    """Get API base URL from env or default."""
    return os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


def create_user(age: int, gender: str, occupation: str, zip_code: str | None = None) -> dict:
    """Create a new user."""
    r = requests.post(
        f"{get_api_base_url()}/api/users",
        json={"age": age, "gender": gender, "occupation": occupation, "zip_code": zip_code or ""},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def get_user(user_id: int) -> dict:
    """Get user profile."""
    r = requests.get(f"{get_api_base_url()}/api/users/{user_id}", timeout=10)
    r.raise_for_status()
    return r.json()


def get_user_ratings(user_id: int) -> dict:
    """Get user rating history."""
    r = requests.get(f"{get_api_base_url()}/api/users/{user_id}/ratings", timeout=10)
    r.raise_for_status()
    return r.json()


def add_rating(user_id: int, movie_id: int, rating: float) -> dict:
    """Add or update a rating."""
    r = requests.post(
        f"{get_api_base_url()}/api/ratings",
        json={"user_id": user_id, "movie_id": movie_id, "rating": rating},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def get_recommendations(
    user_id: int,
    n: int = 10,
    exclude_low_rated: bool = True,
) -> dict:
    """Get personalized recommendations."""
    r = requests.get(
        f"{get_api_base_url()}/api/recommendations/{user_id}",
        params={"n": n, "exclude_low_rated": exclude_low_rated},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def refresh_recommendations(user_id: int, n: int = 10) -> dict:
    """Refresh recommendations (invalidate cache)."""
    r = requests.post(
        f"{get_api_base_url()}/api/recommendations/{user_id}/refresh",
        params={"n": n},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def health_check() -> dict:
    """Check API health."""
    r = requests.get(f"{get_api_base_url()}/api/health", timeout=5)
    r.raise_for_status()
    return r.json()
