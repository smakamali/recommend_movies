"""
Database connection management using SQLAlchemy.

This module handles SQLite database connection creation, session management,
and provides utilities for database operations.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from app.database.models import Base


# Default database path
DEFAULT_DB_PATH = "data/recommender.db"


def get_database_url(db_path: str = DEFAULT_DB_PATH) -> str:
    """
    Get SQLite database URL.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        SQLAlchemy database URL
    """
    # Ensure directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    
    # Convert to absolute path for SQLite
    abs_path = os.path.abspath(db_path)
    
    # SQLite URL format: sqlite:///path/to/database.db
    return f"sqlite:///{abs_path}"


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """
    Enable foreign key constraints for SQLite.
    
    SQLite disables foreign key constraints by default.
    This event listener enables them for all connections.
    """
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class DatabaseManager:
    """
    Database connection manager.
    
    Handles engine creation, session management, and database initialization.
    """
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH, echo: bool = False):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
            echo: If True, log all SQL statements (useful for debugging)
        """
        self.db_path = db_path
        self.database_url = get_database_url(db_path)
        
        # Create engine
        # Use StaticPool for SQLite to avoid threading issues
        self.engine = create_engine(
            self.database_url,
            echo=echo,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    def create_tables(self):
        """
        Create all tables defined in the models.
        
        This creates tables if they don't exist. Existing tables are not modified.
        """
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        """
        Drop all tables defined in the models.
        
        WARNING: This will delete all data in the database!
        """
        Base.metadata.drop_all(bind=self.engine)
    
    def reset_database(self):
        """
        Drop and recreate all tables.
        
        WARNING: This will delete all data in the database!
        """
        self.drop_tables()
        self.create_tables()
    
    def get_session(self) -> Session:
        """
        Get a new database session.
        
        Returns:
            SQLAlchemy Session object
            
        Note:
            Remember to close the session when done:
            session = db_manager.get_session()
            try:
                # Do database operations
                session.commit()
            except:
                session.rollback()
                raise
            finally:
                session.close()
        """
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.
        
        Automatically commits on success and rolls back on failure.
        
        Usage:
            with db_manager.session_scope() as session:
                # Do database operations
                session.add(user)
        
        Yields:
            SQLAlchemy Session object
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def close(self):
        """Close the database engine and all connections."""
        self.engine.dispose()


# Global database manager instance (singleton pattern)
_db_manager = None


def get_db_manager(db_path: str = DEFAULT_DB_PATH, echo: bool = False) -> DatabaseManager:
    """
    Get or create the global database manager instance.
    
    Args:
        db_path: Path to SQLite database file
        echo: If True, log all SQL statements
        
    Returns:
        DatabaseManager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(db_path=db_path, echo=echo)
    return _db_manager


def get_session() -> Generator[Session, None, None]:
    """
    Dependency function for getting database sessions.
    
    Useful for FastAPI dependency injection or similar patterns.
    
    Usage:
        def my_function(session: Session = Depends(get_session)):
            # Use session
            pass
    
    Yields:
        SQLAlchemy Session object
    """
    db_manager = get_db_manager()
    with db_manager.session_scope() as session:
        yield session
