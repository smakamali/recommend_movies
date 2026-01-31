"""
User management API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.api.models.user import UserCreate, UserResponse
from app.database import crud

router = APIRouter(prefix="/api/users", tags=["users"])


@router.post("", response_model=UserResponse)
def create_user(user_in: UserCreate, db: Session = Depends(get_db)):
    """Create a new user with demographic info."""
    try:
        user = crud.create_user(
            db,
            age=user_in.age,
            gender=user_in.gender,
            occupation=user_in.occupation,
            zip_code=user_in.zip_code,
        )
        return user
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    """Get user profile by ID."""
    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.get("/{user_id}/ratings")
def get_user_ratings(user_id: int, db: Session = Depends(get_db)):
    """Get rating history for a user."""
    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    ratings = crud.get_user_ratings(db, user_id)
    return {"user_id": user_id, "ratings": [{"rating_id": r.rating_id, "movie_id": r.movie_id, "rating": r.rating} for r in ratings]}
