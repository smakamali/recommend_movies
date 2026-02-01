"""
API tests for user endpoints.

Uses FastAPI TestClient against the real app (default DB from config).
"""

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


class TestUserEndpoints:
    """Tests for POST /api/users, GET /api/users/{user_id}, GET /api/users/{user_id}/ratings."""

    def test_create_user(self):
        """POST /api/users creates a user and returns 200 with user data."""
        payload = {
            "age": 28,
            "gender": "F",
            "occupation": "programmer",
            "zip_code": "94043",
        }
        r = client.post("/api/users", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "user_id" in data
        assert data["age"] == 28
        assert data["gender"] == "F"
        assert data["occupation"] == "programmer"
        assert data["zip_code"] == "94043"

    def test_create_user_minimal(self):
        """POST /api/users accepts optional zip_code omitted."""
        payload = {
            "age": 25,
            "gender": "M",
            "occupation": "engineer",
        }
        r = client.post("/api/users", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["user_id"] is not None
        assert data["zip_code"] is None or data["zip_code"] == ""

    def test_create_user_invalid_gender(self):
        """POST /api/users with invalid gender returns 400."""
        payload = {
            "age": 25,
            "gender": "X",
            "occupation": "engineer",
        }
        r = client.post("/api/users", json=payload)
        assert r.status_code == 422  # Pydantic validation error

    def test_get_user(self):
        """GET /api/users/{user_id} returns user when exists."""
        # Create user first
        create_r = client.post(
            "/api/users",
            json={"age": 30, "gender": "M", "occupation": "doctor", "zip_code": "12345"},
        )
        assert create_r.status_code == 200
        user_id = create_r.json()["user_id"]

        r = client.get(f"/api/users/{user_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["user_id"] == user_id
        assert data["age"] == 30
        assert data["occupation"] == "doctor"

    def test_get_user_not_found(self):
        """GET /api/users/{user_id} returns 404 for non-existent user."""
        r = client.get("/api/users/999999")
        assert r.status_code == 404
        assert "not found" in r.json()["detail"].lower()

    def test_get_user_ratings(self):
        """GET /api/users/{user_id}/ratings returns list (empty or with ratings)."""
        # Create user first
        create_r = client.post(
            "/api/users",
            json={"age": 22, "gender": "F", "occupation": "student"},
        )
        assert create_r.status_code == 200
        user_id = create_r.json()["user_id"]

        r = client.get(f"/api/users/{user_id}/ratings")
        assert r.status_code == 200
        data = r.json()
        assert data["user_id"] == user_id
        assert "ratings" in data
        assert isinstance(data["ratings"], list)

    def test_get_user_ratings_not_found(self):
        """GET /api/users/{user_id}/ratings returns 404 for non-existent user."""
        r = client.get("/api/users/999999/ratings")
        assert r.status_code == 404
