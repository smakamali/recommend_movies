"""
FastAPI dependency injection for database session and inference engine.
"""

import logging
from typing import Generator
from sqlalchemy.orm import Session

from app.database.connection import get_db_manager, get_session
from app.database import connection as db_connection
from app.core.inference.engine import InferenceEngine
from app.api.config import get_database_path, get_model_path

logger = logging.getLogger(__name__)


def get_db() -> Generator[Session, None, None]:
    """Yield database session for FastAPI Depends()."""
    db_path = get_database_path()
    if db_path.startswith("sqlite"):
        db_path = db_path.replace("sqlite:///", "")
    if not db_path.strip():
        db_path = None  # use connection default
    db_manager = get_db_manager(db_path=db_path) if db_path else get_db_manager()
    with db_manager.session_scope() as session:
        yield session


# Singleton inference engine
_inference_engine: InferenceEngine | None = None


def get_inference_engine() -> InferenceEngine:
    """Get or create singleton InferenceEngine."""
    global _inference_engine
    if _inference_engine is None:
        model_dir = get_model_path()
        _inference_engine = InferenceEngine(model_dir=model_dir)
        try:
            _inference_engine.load_model()
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning("Model not loaded (run train_model.py to train): %s", e)
            # Engine stays with model=None; health returns model_loaded=false
        # Graph will be initialized on first request that needs it
    return _inference_engine
