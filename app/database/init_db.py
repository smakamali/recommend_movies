"""
Database initialization and schema creation.

This module provides functions to initialize the database schema and
populate it with initial data.
"""

from app.database.connection import DatabaseManager, get_db_manager
from app.database.models import Base


def init_database(db_path: str = "data/recommender.db", reset: bool = False) -> DatabaseManager:
    """
    Initialize the database and create all tables.
    
    Args:
        db_path: Path to SQLite database file
        reset: If True, drop existing tables before creating new ones
        
    Returns:
        DatabaseManager instance
    """
    db_manager = get_db_manager(db_path=db_path)
    
    if reset:
        print("Resetting database (dropping all tables)...")
        db_manager.reset_database()
        print("Database reset complete.")
    else:
        print("Creating database tables...")
        db_manager.create_tables()
        print("Database tables created.")
    
    return db_manager


def verify_schema(db_manager: DatabaseManager) -> bool:
    """
    Verify that all tables exist in the database.
    
    Args:
        db_manager: DatabaseManager instance
        
    Returns:
        True if all tables exist, False otherwise
    """
    from sqlalchemy import inspect
    
    inspector = inspect(db_manager.engine)
    existing_tables = set(inspector.get_table_names())
    
    # Expected tables
    expected_tables = {'users', 'movies', 'ratings'}
    
    missing_tables = expected_tables - existing_tables
    
    if missing_tables:
        print(f"Missing tables: {missing_tables}")
        return False
    
    print(f"All tables exist: {existing_tables}")
    return True


if __name__ == "__main__":
    # Initialize database when run as script
    print("Initializing database...")
    db_manager = init_database(reset=False)
    
    # Verify schema
    if verify_schema(db_manager):
        print("\n✅ Database initialization successful!")
    else:
        print("\n❌ Database initialization failed!")
