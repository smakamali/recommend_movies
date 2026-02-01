#!/usr/bin/env python
"""
Migration script: Add name column to users table.

Run this script for existing databases that were created before the name field
was added to the User model. New installations get the column automatically
via create_all().

Usage:
    python scripts/add_user_name_column.py
    python scripts/add_user_name_column.py --db-path data/recommender.db

Author: GraphSAGE Recommender System
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from app.database.connection import get_db_manager, DEFAULT_DB_PATH


def column_exists(engine, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    with engine.connect() as conn:
        result = conn.execute(text(f"PRAGMA table_info({table})"))
        columns = [row[1] for row in result.fetchall()]
        return column in columns


def add_name_column(db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Add name column to users table if it does not exist.

    Returns:
        True if column was added, False if it already existed.
    """
    db_manager = get_db_manager(db_path=db_path)
    engine = db_manager.engine

    if column_exists(engine, "users", "name"):
        print("Column 'name' already exists in users table. Nothing to do.")
        return False

    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE users ADD COLUMN name VARCHAR(100)"))
        conn.commit()

    print("Successfully added 'name' column to users table.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add name column to users table")
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"Path to database file (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()

    try:
        add_name_column(db_path=args.db_path)
    except Exception as e:
        print(f"Migration failed: {e}")
        sys.exit(1)
