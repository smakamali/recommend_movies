"""
API configuration loaded from environment or defaults.
"""

import os
from pathlib import Path


def get_database_path() -> str:
    """Get database file path from env or default."""
    return os.getenv("DATABASE_URL", "sqlite:///").replace("sqlite:///", "") or str(
        Path(__file__).resolve().parents[2] / "data" / "recommender.db"
    )


def get_model_path() -> str:
    """Get model directory path from env or default."""
    return os.getenv("MODEL_PATH", "") or str(
        Path(__file__).resolve().parents[2] / "models" / "current"
    )


def get_log_level() -> str:
    """Get log level from env or default."""
    return os.getenv("LOG_LEVEL", "INFO").upper()


def get_api_host() -> str:
    """Get API host for binding."""
    return os.getenv("API_HOST", "0.0.0.0")


def get_api_port() -> int:
    """Get API port."""
    return int(os.getenv("API_PORT", "8000"))
