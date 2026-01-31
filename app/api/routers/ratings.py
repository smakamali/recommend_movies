"""
Rating API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_inference_engine
from app.api.models.rating import RatingCreate, RatingResponse, RatingStatsResponse
from app.core.inference.engine import InferenceEngine
from app.database import crud

router = APIRouter(prefix="/api/ratings", tags=["ratings"])


def _ensure_graph_initialized(engine: InferenceEngine, db: Session) -> None:
    """Initialize graph from database if not already done."""
    if engine.graph_manager is None:
        engine.initialize_graph(db)


@router.post("", response_model=RatingResponse)
def create_rating(
    rating_in: RatingCreate,
    db: Session = Depends(get_db),
    engine: InferenceEngine = Depends(get_inference_engine),
):
    """Add a new rating (and update inference graph)."""
    user = crud.get_user(db, rating_in.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    movie = crud.get_movie(db, rating_in.movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    existing = crud.get_rating_by_user_movie(db, rating_in.user_id, rating_in.movie_id)
    if existing:
        rating = crud.update_rating(db, existing.rating_id, rating_in.rating)
    else:
        try:
            rating = crud.create_rating(
                db,
                user_id=rating_in.user_id,
                movie_id=rating_in.movie_id,
                rating=rating_in.rating,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    # Update inference graph so recommendations reflect new rating
    try:
        _ensure_graph_initialized(engine, db)
        engine.add_rating(db, rating_in.user_id, rating_in.movie_id, float(rating_in.rating))
    except Exception:
        pass  # Don't fail the request if graph update fails
    return rating


@router.get("/stats", response_model=RatingStatsResponse)
def get_rating_stats(db: Session = Depends(get_db)):
    """Get global rating statistics."""
    stats = crud.get_global_rating_stats(db)
    return stats
