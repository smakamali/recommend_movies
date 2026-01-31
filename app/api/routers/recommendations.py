"""
Recommendation API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_inference_engine
from app.api.models.recommendation import RecommendationResponse, RecommendationItem
from app.core.inference.engine import InferenceEngine

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


def _ensure_graph_initialized(engine: InferenceEngine, db: Session) -> None:
    """Initialize graph from database if not already done."""
    if engine.graph_manager is None:
        engine.initialize_graph(db)


@router.get("/{user_id}", response_model=RecommendationResponse)
def get_recommendations(
    user_id: int,
    n: int = Query(10, ge=1, le=100),
    exclude_low_rated: bool = Query(True),
    db: Session = Depends(get_db),
    engine: InferenceEngine = Depends(get_inference_engine),
):
    """Get personalized recommendations for a user (cold-start and warm-start)."""
    _ensure_graph_initialized(engine, db)
    from app.database import crud
    user_obj = crud.get_user(db, user_id)
    if not user_obj:
        raise HTTPException(status_code=404, detail="User not found")
    try:
        recs = engine.get_recommendations(
            db,
            user_id=user_id,
            n=n,
            exclude_low_rated=exclude_low_rated,
            exclude_already_rated=False,  # Only exclude poorly rated (<=2)
            force_refresh=False,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    items = [
        RecommendationItem(
            movie_id=r["movie_id"],
            title=r["title"],
            release_year=r.get("release_year"),
            genres=r.get("genres", "[]"),
            score=float(r.get("score", 0)),
        )
        for r in recs
    ]
    return RecommendationResponse(user_id=user_id, recommendations=items, n=len(items))


@router.post("/{user_id}/refresh", response_model=RecommendationResponse)
def refresh_recommendations(
    user_id: int,
    n: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
    engine: InferenceEngine = Depends(get_inference_engine),
):
    """Refresh recommendations (invalidate cache and regenerate)."""
    _ensure_graph_initialized(engine, db)
    from app.database import crud
    user_obj = crud.get_user(db, user_id)
    if not user_obj:
        raise HTTPException(status_code=404, detail="User not found")
    try:
        recs = engine.refresh_recommendations(db, user_id=user_id, n=n)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    items = [
        RecommendationItem(
            movie_id=r["movie_id"],
            title=r["title"],
            release_year=r.get("release_year"),
            genres=r.get("genres", "[]"),
            score=float(r.get("score", 0)),
        )
        for r in recs
    ]
    return RecommendationResponse(user_id=user_id, recommendations=items, n=len(items))
