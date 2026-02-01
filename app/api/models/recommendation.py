"""
Pydantic schemas for Recommendation API.
"""

from pydantic import BaseModel


class RecommendationItem(BaseModel):
    """Single recommendation item with movie and score."""

    movie_id: int
    title: str
    release_year: int | None
    genres: str
    score: float


class RecommendationResponse(BaseModel):
    """Response model for recommendations list."""

    user_id: int
    recommendations: list[RecommendationItem]
    n: int
