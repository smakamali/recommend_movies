"""
System API endpoints (health, model info).
"""

from fastapi import APIRouter, Depends
from app.api.dependencies import get_inference_engine, get_db
from app.core.inference.engine import InferenceEngine
from app.database import crud
from sqlalchemy.orm import Session

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health_check(
    db: Session = Depends(get_db),
    engine: InferenceEngine = Depends(get_inference_engine),
):
    """Health check: database and model loaded."""
    try:
        user_count = crud.get_user_count(db)
        movie_count = crud.get_movie_count(db)
    except Exception as e:
        return {"status": "unhealthy", "database": str(e), "model_loaded": engine.model is not None}
    return {
        "status": "healthy",
        "database": "connected",
        "users": user_count,
        "movies": movie_count,
        "model_loaded": engine.model is not None,
        "graph_initialized": engine.graph_manager is not None,
    }


@router.get("/model/info")
def model_info(engine: InferenceEngine = Depends(get_inference_engine)):
    """Get current model metadata."""
    if engine.metadata is None:
        return {"error": "Model not loaded"}
    return engine.metadata
