"""
API tests for recommendation endpoints.

Uses FastAPI TestClient against the real app.
GET /api/recommendations/{user_id} returns 200 when model is loaded, 503 when not.
"""

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


class TestRecommendationEndpoints:
    """Tests for GET /api/recommendations/{user_id} and POST .../refresh."""

    def test_get_recommendations_requires_user(self):
        """GET /api/recommendations/{user_id} returns 404 for non-existent user."""
        r = client.get("/api/recommendations/999999?n=10")
        assert r.status_code == 404
        assert "not found" in r.json()["detail"].lower()

    def test_get_recommendations_response_structure(self):
        """
        GET /api/recommendations/{user_id} returns 200 with list or 503 if model not loaded.
        When 200, response has user_id, recommendations, n.
        """
        # Use a known user ID from MovieLens (e.g. 1) if DB is populated
        r = client.get("/api/recommendations/1?n=10&exclude_low_rated=true")
        if r.status_code == 503:
            assert "model" in r.json()["detail"].lower()
            return
        assert r.status_code == 200
        data = r.json()
        assert "user_id" in data
        assert "recommendations" in data
        assert "n" in data
        assert data["user_id"] == 1
        assert isinstance(data["recommendations"], list)
        assert data["n"] == len(data["recommendations"])
        for item in data["recommendations"]:
            assert "movie_id" in item
            assert "title" in item
            assert "score" in item

    def test_get_recommendations_query_params(self):
        """GET /api/recommendations/{user_id} accepts n and exclude_low_rated."""
        # Create a user so we have a valid user_id
        create_r = client.post(
            "/api/users",
            json={"age": 27, "gender": "M", "occupation": "engineer"},
        )
        assert create_r.status_code == 200
        user_id = create_r.json()["user_id"]

        r = client.get(f"/api/recommendations/{user_id}?n=5&exclude_low_rated=false")
        if r.status_code == 503:
            return  # Model not loaded
        assert r.status_code == 200
        data = r.json()
        assert len(data["recommendations"]) <= 5

    def test_refresh_recommendations_requires_user(self):
        """POST /api/recommendations/{user_id}/refresh returns 404 for non-existent user."""
        r = client.post("/api/recommendations/999999/refresh?n=10")
        assert r.status_code == 404

    def test_refresh_recommendations_response(self):
        """POST /api/recommendations/{user_id}/refresh returns 200 or 503."""
        create_r = client.post(
            "/api/users",
            json={"age": 29, "gender": "F", "occupation": "programmer"},
        )
        assert create_r.status_code == 200
        user_id = create_r.json()["user_id"]

        r = client.post(f"/api/recommendations/{user_id}/refresh?n=10")
        if r.status_code == 503:
            assert "model" in r.json()["detail"].lower()
            return
        assert r.status_code == 200
        data = r.json()
        assert data["user_id"] == user_id
        assert "recommendations" in data
        assert "n" in data
