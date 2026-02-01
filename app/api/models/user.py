"""
Pydantic schemas for User API.
"""

from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    """Request body for creating a user."""

    age: int = Field(..., ge=1, le=120)
    gender: str = Field(..., pattern="^[MFO]$")
    occupation: str = Field(..., min_length=1, max_length=50)
    zip_code: str | None = Field(None, max_length=10)


class UserResponse(BaseModel):
    """Response model for user."""

    user_id: int
    age: int
    gender: str
    occupation: str
    zip_code: str | None

    class Config:
        from_attributes = True
