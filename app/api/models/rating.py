"""
Pydantic schemas for Rating API.
"""

from pydantic import BaseModel, Field


class RatingCreate(BaseModel):
    """Request body for creating a rating."""

    user_id: int = Field(..., gt=0)
    movie_id: int = Field(..., gt=0)
    rating: float = Field(..., ge=1.0, le=5.0)


class RatingResponse(BaseModel):
    """Response model for rating."""

    rating_id: int
    user_id: int
    movie_id: int
    rating: float

    class Config:
        from_attributes = True


class RatingStatsResponse(BaseModel):
    """Response model for rating statistics."""

    count: int
    average: float
    min: float
    max: float
