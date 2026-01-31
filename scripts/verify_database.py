#!/usr/bin/env python
"""
Comprehensive database verification and validation script.

This script performs thorough checks on the database:
1. Basic statistics (counts, distribution)
2. Data integrity (foreign keys, constraints)
3. Duplicate detection
4. Coverage analysis
5. Sample queries
6. Performance checks

Usage:
    # Full verification
    python scripts/verify_database.py

    # Quick check only
    python scripts/verify_database.py --quick

    # Check for duplicates
    python scripts/verify_database.py --check-duplicates

Author: Agent 1 - GraphSAGE Recommender System
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import get_db_manager, crud
from app.database.models import Rating, User, Movie
from sqlalchemy import func


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)


def check_basic_stats(session, verbose=True):
    """Check basic database statistics."""
    if verbose:
        print_section("1. Database Statistics")
    
    user_count = crud.get_user_count(session)
    movie_count = crud.get_movie_count(session)
    rating_count = crud.get_rating_count(session)
    
    print(f"\nDatabase contents:")
    print(f"  Users:   {user_count:,}")
    print(f"  Movies:  {movie_count:,}")
    print(f"  Ratings: {rating_count:,}")
    
    # Check against expected values
    expected = {'users': 943, 'movies': 1682, 'ratings': 100000}
    
    passed = True
    if user_count != expected['users']:
        print(f"  [WARNING] Expected {expected['users']} users")
        passed = False
    if movie_count != expected['movies']:
        print(f"  [WARNING] Expected {expected['movies']} movies")
        passed = False
    if rating_count > 0 and rating_count != expected['ratings']:
        print(f"  [WARNING] Expected {expected['ratings']} ratings")
        passed = False
    
    if passed:
        print("\n[SUCCESS] All counts match expected values")
    
    return passed


def check_rating_distribution(session, verbose=True):
    """Check rating value distribution."""
    if verbose:
        print_section("2. Rating Distribution")
    
    rating_dist = session.query(
        Rating.rating,
        func.count('*').label('count')
    ).group_by(Rating.rating).order_by(Rating.rating).all()
    
    total = sum(count for _, count in rating_dist)
    
    print(f"\nRating distribution:")
    for rating_val, count in rating_dist:
        pct = (count / total * 100) if total > 0 else 0
        bar = '#' * int(pct / 2)
        print(f"  {rating_val}: {count:>6,} ({pct:>5.1f}%) {bar}")
    
    print(f"\nTotal: {total:,}")
    
    return len(rating_dist) == 5  # Should have ratings 1-5


def check_foreign_keys(session, verbose=True):
    """Check foreign key integrity."""
    if verbose:
        print_section("3. Foreign Key Integrity")
    
    # Check for orphaned ratings (invalid user_id)
    orphaned_users = session.query(Rating).outerjoin(
        User, Rating.user_id == User.user_id
    ).filter(User.user_id == None).count()
    
    print(f"\nRatings with invalid user_id: {orphaned_users}")
    
    # Check for orphaned ratings (invalid movie_id)
    orphaned_movies = session.query(Rating).outerjoin(
        Movie, Rating.movie_id == Movie.movie_id
    ).filter(Movie.movie_id == None).count()
    
    print(f"Ratings with invalid movie_id: {orphaned_movies}")
    
    passed = (orphaned_users == 0 and orphaned_movies == 0)
    
    if passed:
        print("\n[SUCCESS] All foreign keys are valid")
    else:
        print("\n[ERROR] Found orphaned ratings!")
    
    return passed


def check_duplicates(session, verbose=True):
    """Check for duplicate (user_id, movie_id) pairs."""
    if verbose:
        print_section("4. Duplicate Check")
    
    duplicates = session.query(
        Rating.user_id,
        Rating.movie_id,
        func.count('*').label('count')
    ).group_by(
        Rating.user_id,
        Rating.movie_id
    ).having(
        func.count('*') > 1
    ).all()
    
    print(f"\nDuplicate (user_id, movie_id) pairs: {len(duplicates)}")
    
    if duplicates:
        print("\n[ERROR] Found duplicates:")
        for dup in duplicates[:10]:
            print(f"  User {dup[0]}, Movie {dup[1]}: {dup[2]} times")
        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more")
        return False
    else:
        print("[SUCCESS] No duplicates found")
        return True


def check_coverage(session, verbose=True):
    """Check user and movie coverage."""
    if verbose:
        print_section("5. Coverage Analysis")
    
    # User rating statistics
    rating_counts = session.query(
        Rating.user_id,
        func.count(Rating.rating_id).label('rating_count')
    ).group_by(Rating.user_id).subquery()
    
    user_stats = session.query(
        func.min(rating_counts.c.rating_count).label('min_ratings'),
        func.max(rating_counts.c.rating_count).label('max_ratings'),
        func.avg(rating_counts.c.rating_count).label('avg_ratings')
    ).first()
    
    print(f"\nUser rating statistics:")
    if user_stats and user_stats[0]:
        print(f"  Min: {int(user_stats[0])} ratings")
        print(f"  Max: {int(user_stats[1])} ratings")
        print(f"  Avg: {user_stats[2]:.1f} ratings")
    
    users_with_ratings = session.query(
        func.count(func.distinct(Rating.user_id))
    ).scalar()
    
    total_users = crud.get_user_count(session)
    print(f"  Users with ratings: {users_with_ratings}/{total_users}")
    
    # Movie rating statistics
    movies_with_ratings = session.query(
        func.count(func.distinct(Rating.movie_id))
    ).scalar()
    
    total_movies = crud.get_movie_count(session)
    print(f"  Movies with ratings: {movies_with_ratings}/{total_movies}")
    
    # Top rated movies
    top_movies = session.query(
        Movie.movie_id,
        Movie.title,
        func.count(Rating.rating_id).label('rating_count'),
        func.avg(Rating.rating).label('avg_rating')
    ).join(Rating).group_by(
        Movie.movie_id, Movie.title
    ).order_by(
        func.count(Rating.rating_id).desc()
    ).limit(5).all()
    
    if top_movies:
        print(f"\nTop 5 most rated movies:")
        for movie in top_movies:
            print(f"  [{movie[0]:>4}] {movie[1][:40]:<40}")
            print(f"        {movie[2]} ratings, avg: {movie[3]:.2f}")
    
    # Coverage is good if most users/movies have ratings
    coverage_ok = (
        users_with_ratings >= total_users * 0.95 and
        movies_with_ratings >= total_movies * 0.95
    )
    
    if coverage_ok:
        print("\n[SUCCESS] Good coverage (>95% for users and movies)")
    
    return coverage_ok


def check_sample_queries(session, verbose=True):
    """Run sample queries to verify functionality."""
    if verbose:
        print_section("6. Sample Queries")
    
    try:
        # Sample user's ratings
        sample_user = session.query(Rating.user_id).first()
        if sample_user:
            user_id = sample_user[0]
            user_ratings = crud.get_ratings_by_user(session, user_id, limit=5)
            
            print(f"\nSample: User {user_id}'s first 5 ratings:")
            for rating in user_ratings:
                movie = crud.get_movie(session, rating.movie_id)
                if movie:
                    print(f"  {movie.title[:40]:<40} {rating.rating}/5.0")
        
        # Sample movie's statistics
        sample_movie = session.query(Rating.movie_id).first()
        if sample_movie:
            movie_id = sample_movie[0]
            movie = crud.get_movie(session, movie_id)
            stats = crud.get_rating_stats(session, movie_id)
            
            if movie:
                print(f"\nSample: '{movie.title}' ratings:")
                print(f"  Count: {stats['count']}")
                print(f"  Average: {stats['average']:.2f}")
                print(f"  Range: {stats['min']:.1f} - {stats['max']:.1f}")
        
        # Search test
        results = crud.search_movies(session, title="Star", limit=3)
        if results:
            print(f"\nSearch for 'Star' (first 3 results):")
            for movie in results:
                print(f"  [{movie.movie_id}] {movie.title}")
        
        print("\n[SUCCESS] All sample queries executed successfully")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Sample queries failed: {e}")
        return False


def check_data_quality(session, verbose=True):
    """Check overall data quality."""
    if verbose:
        print_section("7. Data Quality Checks")
    
    issues = []
    
    # Check for NULL values in required fields
    null_movie_titles = session.query(Movie).filter(Movie.title == None).count()
    if null_movie_titles > 0:
        issues.append(f"Found {null_movie_titles} movies with NULL titles")
    
    null_user_ages = session.query(User).filter(User.age == None).count()
    if null_user_ages > 0:
        issues.append(f"Found {null_user_ages} users with NULL ages")
    
    # Check rating value constraints
    invalid_ratings = session.query(Rating).filter(
        (Rating.rating < 1) | (Rating.rating > 5)
    ).count()
    if invalid_ratings > 0:
        issues.append(f"Found {invalid_ratings} ratings outside 1-5 range")
    
    # Check gender constraints
    invalid_genders = session.query(User).filter(
        ~User.gender.in_(['M', 'F', 'O'])
    ).count()
    if invalid_genders > 0:
        issues.append(f"Found {invalid_genders} users with invalid gender")
    
    if issues:
        print("\n[WARNING] Data quality issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n[SUCCESS] No data quality issues found")
        return True


def run_full_verification(quick=False):
    """Run all verification checks."""
    
    print("="*60)
    print("Database Verification and Validation")
    print("="*60)
    
    db_manager = get_db_manager()
    session = db_manager.get_session()
    
    try:
        results = {}
        
        # Run checks
        results['basic_stats'] = check_basic_stats(session)
        results['rating_dist'] = check_rating_distribution(session)
        results['foreign_keys'] = check_foreign_keys(session)
        results['duplicates'] = check_duplicates(session)
        
        if not quick:
            results['coverage'] = check_coverage(session)
            results['sample_queries'] = check_sample_queries(session)
            results['data_quality'] = check_data_quality(session)
        
        # Summary
        print_section("Verification Summary")
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        print(f"\nChecks passed: {passed}/{total}")
        
        for check_name, result in results.items():
            status = "[PASS]" if result else "[FAIL]"
            check_display = check_name.replace('_', ' ').title()
            print(f"  {status} {check_display}")
        
        all_passed = all(results.values())
        
        if all_passed:
            print("\n[SUCCESS] All verification checks PASSED!")
            print("\nDatabase is ready for:")
            print("  - Graph construction")
            print("  - Feature engineering")
            print("  - Model training")
            print("  - Recommendation generation")
        else:
            print("\n[ERROR] Some verification checks FAILED!")
            print("Review the issues above and re-run import if needed.")
        
        print("="*60)
        
        return all_passed
        
    finally:
        session.close()


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Verify MovieLens 100K database integrity and quality"
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run only essential checks (faster)'
    )
    parser.add_argument(
        '--check-duplicates',
        action='store_true',
        help='Focus on duplicate detection'
    )
    
    args = parser.parse_args()
    
    try:
        if args.check_duplicates:
            db_manager = get_db_manager()
            session = db_manager.get_session()
            success = check_duplicates(session, verbose=True)
            session.close()
        else:
            success = run_full_verification(quick=args.quick)
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n[ERROR] Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
