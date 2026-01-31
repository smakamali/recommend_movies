"""
Database module for the recommender system.

This module provides database models, connection management, and CRUD operations
for the SQLite database using SQLAlchemy ORM.
"""

from app.database.models import Base, User, Movie, Rating
from app.database.connection import DatabaseManager, get_db_manager, get_session
from app.database.init_db import init_database, verify_schema
from app.database import crud

__all__ = [
    # Models
    'Base',
    'User', 
    'Movie',
    'Rating',
    # Connection
    'DatabaseManager',
    'get_db_manager',
    'get_session',
    # Initialization
    'init_database',
    'verify_schema',
    # CRUD module
    'crud',
]
