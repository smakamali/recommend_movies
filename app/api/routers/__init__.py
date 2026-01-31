"""
API route handlers.
"""

from app.api.routers import users, movies, ratings, recommendations, system

__all__ = ["users", "movies", "ratings", "recommendations", "system"]
