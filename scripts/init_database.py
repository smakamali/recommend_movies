#!/usr/bin/env python
"""
Master database initialization script for MovieLens 100K dataset.

This script performs a complete database setup:
1. Creates database schema (tables, indexes, constraints)
2. Imports 1,682 movies with metadata
3. Imports 943 users with demographics
4. Imports 100,000 ratings (bulk insert)
5. Verifies data integrity

Usage:
    # Full import (recommended)
    python scripts/init_database.py --reset

    # Import only movies and users (no ratings)
    python scripts/init_database.py --reset --skip-ratings

    # Import only ratings (assumes movies/users exist)
    python scripts/init_database.py --ratings-only

    # Keep existing data (append mode)
    python scripts/init_database.py --no-reset

Author: Agent 1 - GraphSAGE Recommender System
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import init_database, get_db_manager, crud
from app.database.models import Rating


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)


def get_data_path():
    """
    Get path to MovieLens 100K data directory.
    
    Returns:
        Path object to data directory, or None if not found
    """
    # Try common locations
    possible_paths = [
        Path.home() / '.surprise_data' / 'ml-100k',
        Path('data') / 'ml-100k',
        Path('.') / 'data' / 'ml-100k',
    ]
    
    for path in possible_paths:
        if path.exists() and (path / 'u.item').exists():
            return path
    
    return None


def import_movies(db_manager, data_path, verbose=True):
    """
    Import movies from u.item file.
    
    Args:
        db_manager: DatabaseManager instance
        data_path: Path to MovieLens data directory
        verbose: Print progress information
        
    Returns:
        Number of movies imported
    """
    if verbose:
        print_section("Importing Movies")
    
    item_file = data_path / 'u.item'
    if not item_file.exists():
        print(f"[ERROR] Movies file not found: {item_file}")
        return 0
    
    # Genre names (19 genres in MovieLens 100k)
    genre_names = [
        'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    
    imported_count = 0
    skipped_count = 0
    
    session = db_manager.get_session()
    try:
        with open(item_file, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) < 24:
                    continue
                
                movie_id = int(parts[0])
                title = parts[1]
                release_date = parts[2]
                imdb_url = parts[4]
                
                # Check if movie exists
                existing = crud.get_movie(session, movie_id)
                if existing:
                    skipped_count += 1
                    continue
                
                # Extract genres from binary flags
                genres = []
                for i, genre_name in enumerate(genre_names):
                    if int(parts[5 + i]) == 1:
                        genres.append(genre_name)
                
                # Extract release year
                release_year = None
                if release_date:
                    try:
                        release_year = int(release_date.split('-')[-1])
                    except:
                        pass
                
                # Fallback: extract year from title (e.g., "Movie (1995)")
                if not release_year and '(' in title and ')' in title:
                    try:
                        year_str = title[title.rfind('(')+1:title.rfind(')')]
                        if year_str.isdigit() and len(year_str) == 4:
                            release_year = int(year_str)
                    except:
                        pass
                
                # Create movie
                crud.create_movie(
                    session,
                    movie_id=movie_id,
                    title=title,
                    genres=json.dumps(genres),
                    release_year=release_year,
                    imdb_url=imdb_url if imdb_url else None
                )
                
                imported_count += 1
                
                if verbose and (imported_count % 200 == 0):
                    print(f"  Imported {imported_count} movies...")
        
        session.commit()
        
        if verbose:
            print(f"\n[SUCCESS] Movie import complete!")
            print(f"  Imported: {imported_count} movies")
            if skipped_count > 0:
                print(f"  Skipped (already exist): {skipped_count} movies")
        
        return imported_count
        
    except Exception as e:
        session.rollback()
        print(f"[ERROR] Movie import failed: {e}")
        raise
    finally:
        session.close()


def import_users(db_manager, data_path, verbose=True):
    """
    Import users from u.user file.
    
    Args:
        db_manager: DatabaseManager instance
        data_path: Path to MovieLens data directory
        verbose: Print progress information
        
    Returns:
        Number of users imported
    """
    if verbose:
        print_section("Importing Users")
    
    user_file = data_path / 'u.user'
    if not user_file.exists():
        print(f"[ERROR] Users file not found: {user_file}")
        return 0
    
    imported_count = 0
    skipped_count = 0
    
    session = db_manager.get_session()
    try:
        with open(user_file, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) < 5:
                    continue
                
                user_id = int(parts[0])
                age = int(parts[1])
                gender = parts[2]
                occupation = parts[3]
                zip_code = parts[4]
                
                # Check if user exists
                existing = crud.get_user(session, user_id)
                if existing:
                    skipped_count += 1
                    continue
                
                # Create user
                crud.create_user(
                    session,
                    age=age,
                    gender=gender,
                    occupation=occupation,
                    zip_code=zip_code if zip_code else None
                )
                
                imported_count += 1
                
                if verbose and (imported_count % 200 == 0):
                    print(f"  Imported {imported_count} users...")
        
        session.commit()
        
        if verbose:
            print(f"\n[SUCCESS] User import complete!")
            print(f"  Imported: {imported_count} users")
            if skipped_count > 0:
                print(f"  Skipped (already exist): {skipped_count} users")
        
        return imported_count
        
    except Exception as e:
        session.rollback()
        print(f"[ERROR] User import failed: {e}")
        raise
    finally:
        session.close()


def import_ratings(db_manager, data_path, batch_size=5000, clear_existing=True, verbose=True):
    """
    Import ratings from u.data file using bulk insert.
    
    Args:
        db_manager: DatabaseManager instance
        data_path: Path to MovieLens data directory
        batch_size: Number of ratings to insert per batch
        clear_existing: Whether to delete existing ratings first
        verbose: Print progress information
        
    Returns:
        Number of ratings imported
    """
    if verbose:
        print_section("Importing Ratings")
    
    ratings_file = data_path / 'u.data'
    if not ratings_file.exists():
        print(f"[ERROR] Ratings file not found: {ratings_file}")
        return 0
    
    session = db_manager.get_session()
    
    try:
        # Get valid IDs from database
        if verbose:
            print("Loading valid user and movie IDs...")
        
        valid_user_ids = set()
        users = crud.get_users(session, skip=0, limit=10000)
        for user in users:
            valid_user_ids.add(user.user_id)
        
        valid_movie_ids = set()
        movies = crud.get_movies(session, skip=0, limit=10000)
        for movie in movies:
            valid_movie_ids.add(movie.movie_id)
        
        if verbose:
            print(f"  Valid user IDs: {len(valid_user_ids)}")
            print(f"  Valid movie IDs: {len(valid_movie_ids)}")
        
        # Clear existing ratings if requested
        if clear_existing:
            existing_count = crud.get_rating_count(session)
            if existing_count > 0:
                if verbose:
                    print(f"\nClearing {existing_count:,} existing ratings...")
                start_time = time.time()
                session.query(Rating).delete()
                session.commit()
                
                # Verify deletion
                remaining = crud.get_rating_count(session)
                if remaining > 0:
                    print(f"[WARNING] {remaining} ratings still exist after deletion!")
                    return 0
                
                if verbose:
                    elapsed = time.time() - start_time
                    print(f"[SUCCESS] Cleared in {elapsed:.2f}s")
        
        # Read ratings file
        if verbose:
            print(f"\nReading ratings from file...")
        
        ratings_data = []
        seen_pairs = set()
        duplicate_count = 0
        filtered_count = 0
        
        start_time = time.time()
        
        with open(ratings_file, 'r', encoding='latin-1') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                
                user_id = int(parts[0])
                movie_id = int(parts[1])
                rating = float(parts[2])
                
                # Check if IDs are valid
                if user_id not in valid_user_ids or movie_id not in valid_movie_ids:
                    filtered_count += 1
                    continue
                
                # Check for duplicates
                pair = (user_id, movie_id)
                if pair in seen_pairs:
                    duplicate_count += 1
                    continue
                
                seen_pairs.add(pair)
                ratings_data.append({
                    'user_id': user_id,
                    'movie_id': movie_id,
                    'rating': rating,
                    'created_at': datetime.now(),
                    'updated_at': datetime.now()
                })
                
                if verbose and (line_num % 20000 == 0):
                    print(f"  Read {line_num:,} lines...")
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"\n[SUCCESS] Read {len(ratings_data):,} valid ratings in {elapsed:.2f}s")
            if duplicate_count > 0:
                print(f"  Duplicates skipped: {duplicate_count}")
            if filtered_count > 0:
                print(f"  Invalid IDs filtered: {filtered_count}")
        
        # Bulk insert ratings
        if verbose:
            print(f"\nBulk inserting {len(ratings_data):,} ratings...")
        
        total_inserted = 0
        start_time = time.time()
        
        for i in range(0, len(ratings_data), batch_size):
            batch = ratings_data[i:i + batch_size]
            
            try:
                session.bulk_insert_mappings(Rating, batch)
                session.commit()
                total_inserted += len(batch)
                
                if verbose:
                    print(f"  Inserted {total_inserted:,}/{len(ratings_data):,} ratings...")
            except Exception as e:
                print(f"\n[ERROR] Failed at batch {i//batch_size + 1}: {e}")
                session.rollback()
                raise
        
        elapsed = time.time() - start_time
        rate = total_inserted / elapsed if elapsed > 0 else 0
        
        if verbose:
            print(f"\n[SUCCESS] Ratings import complete!")
            print(f"  Imported: {total_inserted:,} ratings")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Rate: {rate:,.0f} ratings/second")
        
        return total_inserted
        
    except Exception as e:
        session.rollback()
        print(f"[ERROR] Ratings import failed: {e}")
        raise
    finally:
        session.close()


def verify_import(db_manager, verbose=True):
    """
    Verify that data was imported correctly.
    
    Args:
        db_manager: DatabaseManager instance
        verbose: Print verification results
        
    Returns:
        True if verification passes, False otherwise
    """
    if verbose:
        print_section("Verification")
    
    session = db_manager.get_session()
    
    try:
        movie_count = crud.get_movie_count(session)
        user_count = crud.get_user_count(session)
        rating_count = crud.get_rating_count(session)
        
        if verbose:
            print(f"\nDatabase contents:")
            print(f"  Movies:  {movie_count:,}")
            print(f"  Users:   {user_count:,}")
            print(f"  Ratings: {rating_count:,}")
        
        # Expected counts
        expected_movies = 1682
        expected_users = 943
        expected_ratings = 100000
        
        success = True
        
        if movie_count != expected_movies:
            print(f"\n[WARNING] Expected {expected_movies} movies, found {movie_count}")
            success = False
        else:
            print(f"\n[SUCCESS] Movie count correct ({expected_movies})")
        
        if user_count != expected_users:
            print(f"[WARNING] Expected {expected_users} users, found {user_count}")
            success = False
        else:
            print(f"[SUCCESS] User count correct ({expected_users})")
        
        if rating_count > 0 and rating_count != expected_ratings:
            print(f"[WARNING] Expected {expected_ratings} ratings, found {rating_count}")
            success = False
        elif rating_count == expected_ratings:
            print(f"[SUCCESS] Rating count correct ({expected_ratings})")
        
        # Sample data
        if verbose and movie_count > 0:
            print(f"\nSample data:")
            sample_movie = crud.get_movie(session, 1)
            if sample_movie:
                genres = json.loads(sample_movie.genres)
                print(f"  Movie: {sample_movie.title} ({sample_movie.release_year})")
                print(f"    Genres: {', '.join(genres[:3])}")
            
            if user_count > 0:
                sample_user = crud.get_user(session, 1)
                if sample_user:
                    print(f"  User: {sample_user.age}yo {sample_user.gender}, {sample_user.occupation}")
        
        return success
        
    finally:
        session.close()


def main():
    """Main entry point for database initialization."""
    
    parser = argparse.ArgumentParser(
        description="Initialize MovieLens 100K database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full import (recommended for first-time setup)
  python scripts/init_database.py --reset

  # Import only ratings (if movies/users already exist)
  python scripts/init_database.py --ratings-only

  # Import without clearing existing data
  python scripts/init_database.py --no-reset

  # Skip ratings import
  python scripts/init_database.py --reset --skip-ratings
        """
    )
    
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Drop and recreate database tables (WARNING: deletes all data)'
    )
    parser.add_argument(
        '--no-reset',
        action='store_true',
        help='Keep existing database (append mode)'
    )
    parser.add_argument(
        '--ratings-only',
        action='store_true',
        help='Import only ratings (assumes movies/users exist)'
    )
    parser.add_argument(
        '--skip-ratings',
        action='store_true',
        help='Skip ratings import (import only movies and users)'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default='data/recommender.db',
        help='Path to SQLite database file (default: data/recommender.db)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to MovieLens data directory (auto-detected if not specified)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=5000,
        help='Batch size for ratings import (default: 5000)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    # Validate arguments
    if args.reset and args.no_reset:
        print("[ERROR] Cannot use both --reset and --no-reset")
        sys.exit(1)
    
    # Print header
    if verbose:
        print("="*60)
        print("MovieLens 100K Database Initialization")
        print("="*60)
        print(f"\nDatabase: {args.db_path}")
        print(f"Mode: {'Reset' if args.reset else 'Keep existing' if args.no_reset else 'Default'}")
        if args.ratings_only:
            print("Import: Ratings only")
        elif args.skip_ratings:
            print("Import: Movies and users only")
        else:
            print("Import: Movies, users, and ratings")
    
    try:
        # Find data path
        if args.data_path:
            data_path = Path(args.data_path)
        else:
            data_path = get_data_path()
        
        if not data_path or not data_path.exists():
            print("\n[ERROR] MovieLens 100K data not found!")
            print("Download from: http://files.grouplens.org/datasets/movielens/ml-100k.zip")
            print("Extract to: ~/.surprise_data/ml-100k/")
            print("Or specify path with: --data-path /path/to/ml-100k")
            sys.exit(1)
        
        if verbose:
            print(f"Data path: {data_path}")
        
        # Initialize database
        db_manager = init_database(db_path=args.db_path, reset=args.reset)
        
        # Import data
        start_time = time.time()
        
        if not args.ratings_only:
            # Import movies and users
            movie_count = import_movies(db_manager, data_path, verbose=verbose)
            user_count = import_users(db_manager, data_path, verbose=verbose)
        
        if not args.skip_ratings:
            # Import ratings
            clear_ratings = args.reset or not args.no_reset
            rating_count = import_ratings(
                db_manager, 
                data_path, 
                batch_size=args.batch_size,
                clear_existing=clear_ratings,
                verbose=verbose
            )
        
        total_time = time.time() - start_time
        
        # Verify
        success = verify_import(db_manager, verbose=verbose)
        
        # Summary
        if verbose:
            print_section("Summary")
            print(f"\n[SUCCESS] Database initialization complete!")
            print(f"Total time: {total_time:.2f}s")
            print(f"\nDatabase ready for:")
            print("  - Graph construction")
            print("  - Feature engineering")
            print("  - Model training")
            print("  - Recommendation generation")
            print("\nNext: python scripts/verify_database.py")
            print("="*60)
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n[ERROR] Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
