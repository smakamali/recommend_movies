"""
Pydantic schemas for API request/response validation.
"""

from app.api.models.user import UserCreate, UserResponse
from app.api.models.movie import MovieResponse, MovieList
from app.api.models.rating import RatingCreate, RatingResponse, RatingStatsResponse
from app.api.models.recommendation import RecommendationResponse, RecommendationItem

__all__ = [
    "UserCreate",
    "UserResponse",
    "MovieResponse",
    "MovieList",
    "RatingCreate",
    "RatingResponse",
    "RatingStatsResponse",
    "RecommendationResponse",
    "RecommendationItem",
]
