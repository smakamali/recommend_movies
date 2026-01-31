#!/usr/bin/env python
"""
Database demonstration script - showcases key functionality.

This script demonstrates:
- Database queries (users, movies, ratings)
- Search functionality
- Statistics and aggregations
- Relationships between entities

Usage:
    python scripts/demo_database.py

Author: Agent 1 - GraphSAGE Recommender System
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import get_db_manager, crud


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)


def demo_basic_queries(session):
    """Demonstrate basic database queries."""
    print_section("1. Basic Queries")
    
    # Get counts
    print(f"\nDatabase overview:")
    print(f"  Users:   {crud.get_user_count(session):,}")
    print(f"  Movies:  {crud.get_movie_count(session):,}")
    print(f"  Ratings: {crud.get_rating_count(session):,}")
    
    # Get a specific movie
    movie = crud.get_movie(session, movie_id=50)  # Star Wars
    if movie:
        genres = json.loads(movie.genres)
        print(f"\nMovie details (ID={movie.movie_id}):")
        print(f"  Title: {movie.title}")
        print(f"  Year: {movie.release_year}")
        print(f"  Genres: {', '.join(genres)}")
        print(f"  IMDB: {movie.imdb_url}")
    
    # Get a specific user
    user = crud.get_user(session, user_id=1)
    if user:
        print(f"\nUser details (ID={user.user_id}):")
        print(f"  Age: {user.age}")
        print(f"  Gender: {user.gender}")
        print(f"  Occupation: {user.occupation}")
        print(f"  Zip Code: {user.zip_code}")


def demo_search(session):
    """Demonstrate search functionality."""
    print_section("2. Search Functionality")
    
    # Search by title
    print(f"\nSearch for movies with 'Star' in title:")
    results = crud.search_movies(session, title="Star", limit=5)
    for movie in results:
        genres = json.loads(movie.genres)
        print(f"  [{movie.movie_id:>4}] {movie.title} ({movie.release_year})")
        print(f"        Genres: {', '.join(genres[:3])}")
    
    # Search by genre
    print(f"\nAction movies (first 5):")
    results = crud.search_movies(session, genre="Action", limit=5)
    for movie in results:
        print(f"  [{movie.movie_id:>4}] {movie.title} ({movie.release_year})")
    
    # Search by year
    print(f"\nMovies from 1977:")
    results = crud.search_movies(session, year=1977, limit=5)
    for movie in results:
        print(f"  [{movie.movie_id:>4}] {movie.title}")


def demo_ratings(session):
    """Demonstrate rating queries."""
    print_section("3. Ratings and Statistics")
    
    # User's ratings
    user_id = 1
    ratings = crud.get_ratings_by_user(session, user_id, limit=5)
    
    print(f"\nUser {user_id}'s ratings (first 5):")
    for rating in ratings:
        movie = crud.get_movie(session, rating.movie_id)
        if movie:
            stars = '*' * int(rating.rating)
            print(f"  {movie.title[:40]:<40} {rating.rating}/5.0 {stars}")
    
    # Movie statistics
    movie_id = 50  # Star Wars
    movie = crud.get_movie(session, movie_id)
    stats = crud.get_rating_stats(session, movie_id)
    
    if movie:
        print(f"\nRating statistics for '{movie.title}':")
        print(f"  Total ratings: {stats['count']}")
        print(f"  Average: {stats['average']:.2f}/5.0")
        print(f"  Range: {stats['min']:.1f} - {stats['max']:.1f}")


def demo_relationships(session):
    """Demonstrate entity relationships."""
    print_section("4. Relationships")
    
    # Get user with their ratings (using relationship)
    user = crud.get_user(session, user_id=1)
    if user:
        session.refresh(user)  # Load relationships
        print(f"\nUser {user.user_id} ({user.occupation}) has {len(user.ratings)} ratings")
        
        # Show rating distribution for this user
        rating_dist = {}
        for rating in user.ratings[:20]:  # First 20
            rating_dist[rating.rating] = rating_dist.get(rating.rating, 0) + 1
        
        print(f"  Rating distribution (first 20):")
        for rating_val in sorted(rating_dist.keys()):
            print(f"    {rating_val}: {rating_dist[rating_val]} movies")
    
    # Get movie with its ratings
    movie = crud.get_movie(session, movie_id=50)
    if movie:
        session.refresh(movie)
        print(f"\nMovie '{movie.title}' has {len(movie.ratings)} ratings")
        
        if movie.ratings:
            avg_rating = sum(r.rating for r in movie.ratings) / len(movie.ratings)
            print(f"  Average rating: {avg_rating:.2f}/5.0")


def demo_pagination(session):
    """Demonstrate pagination."""
    print_section("5. Pagination")
    
    print(f"\nMovies (page 1 - first 5):")
    movies = crud.get_movies(session, skip=0, limit=5)
    for movie in movies:
        print(f"  [{movie.movie_id:>4}] {movie.title}")
    
    print(f"\nMovies (page 2 - next 5):")
    movies = crud.get_movies(session, skip=5, limit=5)
    for movie in movies:
        print(f"  [{movie.movie_id:>4}] {movie.title}")


def main():
    """Run database demonstration."""
    
    print("="*60)
    print("GraphSAGE Recommender System - Database Demo")
    print("="*60)
    
    db_manager = get_db_manager()
    session = db_manager.get_session()
    
    try:
        # Run demonstrations
        demo_basic_queries(session)
        demo_search(session)
        demo_ratings(session)
        demo_relationships(session)
        demo_pagination(session)
        
        # Summary
        print_section("Summary")
        print("\n[SUCCESS] All database operations working correctly!")
        print("\nThe database layer provides:")
        print("  - Fast queries with indexed columns")
        print("  - Flexible search functionality")
        print("  - Comprehensive statistics")
        print("  - Clean ORM relationships")
        print("  - Pagination support")
        print("\nReady for graph construction and model training!")
        print("="*60)
        
    except Exception as e:
        print(f"\n[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        session.close()


if __name__ == "__main__":
    main()
