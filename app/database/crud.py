"""
CRUD operations for User, Movie, and Rating models.

This module provides Create, Read, Update, Delete operations for all database models.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy import func, and_, or_
from sqlalchemy.orm import Session

from app.database.models import User, Movie, Rating


# ==================== USER CRUD OPERATIONS ====================

def create_user(
    session: Session,
    age: int,
    gender: str,
    occupation: str,
    zip_code: Optional[str] = None,
    name: Optional[str] = None,
) -> User:
    """
    Create a new user.
    
    Args:
        session: Database session
        age: User's age
        gender: User's gender ('M', 'F', or 'O')
        occupation: User's occupation
        zip_code: User's zip code (optional)
        name: User's display name (optional)
        
    Returns:
        Created User object
        
    Raises:
        ValueError: If gender is not 'M', 'F', or 'O'
    """
    if gender not in ('M', 'F', 'O'):
        raise ValueError("Gender must be 'M', 'F', or 'O'")
    
    user = User(
        age=age,
        gender=gender,
        occupation=occupation,
        zip_code=zip_code,
        name=name,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def get_user(session: Session, user_id: int) -> Optional[User]:
    """
    Get a user by ID.
    
    Args:
        session: Database session
        user_id: User ID
        
    Returns:
        User object or None if not found
    """
    return session.query(User).filter(User.user_id == user_id).first()


def get_users(
    session: Session,
    skip: int = 0,
    limit: int = 100
) -> List[User]:
    """
    Get a list of users with pagination.
    
    Args:
        session: Database session
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List of User objects
    """
    return session.query(User).offset(skip).limit(limit).all()


def get_user_count(session: Session) -> int:
    """
    Get total count of users.
    
    Args:
        session: Database session
        
    Returns:
        Total number of users
    """
    return session.query(func.count(User.user_id)).scalar()


def update_user(
    session: Session,
    user_id: int,
    **kwargs
) -> Optional[User]:
    """
    Update user information.
    
    Args:
        session: Database session
        user_id: User ID
        **kwargs: Fields to update (age, gender, occupation, zip_code)
        
    Returns:
        Updated User object or None if not found
    """
    user = get_user(session, user_id)
    if user:
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
        session.commit()
        session.refresh(user)
    return user


def delete_user(session: Session, user_id: int) -> bool:
    """
    Delete a user.
    
    Args:
        session: Database session
        user_id: User ID
        
    Returns:
        True if user was deleted, False if not found
    """
    user = get_user(session, user_id)
    if user:
        session.delete(user)
        session.commit()
        return True
    return False


def get_user_ratings(session: Session, user_id: int) -> List[Rating]:
    """
    Get all ratings by a user.
    
    Args:
        session: Database session
        user_id: User ID
        
    Returns:
        List of Rating objects
    """
    return session.query(Rating).filter(Rating.user_id == user_id).all()


# ==================== MOVIE CRUD OPERATIONS ====================

def create_movie(
    session: Session,
    movie_id: int,
    title: str,
    genres: str,
    release_year: Optional[int] = None,
    imdb_url: Optional[str] = None
) -> Movie:
    """
    Create a new movie.
    
    Args:
        session: Database session
        movie_id: Movie ID (from MovieLens dataset)
        title: Movie title
        genres: JSON string of genres
        release_year: Year the movie was released
        imdb_url: URL to IMDB page
        
    Returns:
        Created Movie object
    """
    movie = Movie(
        movie_id=movie_id,
        title=title,
        genres=genres,
        release_year=release_year,
        imdb_url=imdb_url
    )
    session.add(movie)
    session.commit()
    session.refresh(movie)
    return movie


def get_movie(session: Session, movie_id: int) -> Optional[Movie]:
    """
    Get a movie by ID.
    
    Args:
        session: Database session
        movie_id: Movie ID
        
    Returns:
        Movie object or None if not found
    """
    return session.query(Movie).filter(Movie.movie_id == movie_id).first()


def get_movies(
    session: Session,
    skip: int = 0,
    limit: int = 100
) -> List[Movie]:
    """
    Get a list of movies with pagination.
    
    Args:
        session: Database session
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List of Movie objects
    """
    return session.query(Movie).offset(skip).limit(limit).all()


def get_movie_count(session: Session) -> int:
    """
    Get total count of movies.
    
    Args:
        session: Database session
        
    Returns:
        Total number of movies
    """
    return session.query(func.count(Movie.movie_id)).scalar()


def search_movies(
    session: Session,
    title: Optional[str] = None,
    year: Optional[int] = None,
    genre: Optional[str] = None,
    limit: int = 100
) -> List[Movie]:
    """
    Search for movies by title, year, or genre.
    
    Args:
        session: Database session
        title: Search by title (partial match)
        year: Filter by release year
        genre: Filter by genre (partial match in genres JSON)
        limit: Maximum number of results
        
    Returns:
        List of Movie objects matching the criteria
    """
    query = session.query(Movie)
    
    if title:
        query = query.filter(Movie.title.ilike(f"%{title}%"))
    
    if year:
        query = query.filter(Movie.release_year == year)
    
    if genre:
        query = query.filter(Movie.genres.ilike(f"%{genre}%"))
    
    return query.limit(limit).all()


def get_movies_by_year(
    session: Session,
    year: int,
    skip: int = 0,
    limit: int = 100
) -> List[Movie]:
    """
    Get movies released in a specific year.
    
    Args:
        session: Database session
        year: Release year
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List of Movie objects
    """
    return session.query(Movie).filter(
        Movie.release_year == year
    ).offset(skip).limit(limit).all()


# ==================== RATING CRUD OPERATIONS ====================

def create_rating(
    session: Session,
    user_id: int,
    movie_id: int,
    rating: float
) -> Rating:
    """
    Create a new rating.
    
    Args:
        session: Database session
        user_id: User ID
        movie_id: Movie ID
        rating: Rating value (1.0 to 5.0)
        
    Returns:
        Created Rating object
        
    Raises:
        ValueError: If rating is not between 1 and 5
    """
    if not (1.0 <= rating <= 5.0):
        raise ValueError("Rating must be between 1.0 and 5.0")
    
    rating_obj = Rating(
        user_id=user_id,
        movie_id=movie_id,
        rating=rating
    )
    session.add(rating_obj)
    session.commit()
    session.refresh(rating_obj)
    return rating_obj


def get_rating(session: Session, rating_id: int) -> Optional[Rating]:
    """
    Get a rating by ID.
    
    Args:
        session: Database session
        rating_id: Rating ID
        
    Returns:
        Rating object or None if not found
    """
    return session.query(Rating).filter(Rating.rating_id == rating_id).first()


def get_rating_by_user_movie(
    session: Session,
    user_id: int,
    movie_id: int
) -> Optional[Rating]:
    """
    Get a rating by user and movie.
    
    Args:
        session: Database session
        user_id: User ID
        movie_id: Movie ID
        
    Returns:
        Rating object or None if not found
    """
    return session.query(Rating).filter(
        and_(Rating.user_id == user_id, Rating.movie_id == movie_id)
    ).first()


def get_ratings_by_user(
    session: Session,
    user_id: int,
    skip: int = 0,
    limit: int = 100
) -> List[Rating]:
    """
    Get all ratings by a specific user.
    
    Args:
        session: Database session
        user_id: User ID
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List of Rating objects
    """
    return session.query(Rating).filter(
        Rating.user_id == user_id
    ).offset(skip).limit(limit).all()


def get_ratings_by_movie(
    session: Session,
    movie_id: int,
    skip: int = 0,
    limit: int = 100
) -> List[Rating]:
    """
    Get all ratings for a specific movie.
    
    Args:
        session: Database session
        movie_id: Movie ID
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List of Rating objects
    """
    return session.query(Rating).filter(
        Rating.movie_id == movie_id
    ).offset(skip).limit(limit).all()


def get_rating_stats(session: Session, movie_id: int) -> Dict[str, Any]:
    """
    Get rating statistics for a movie.
    
    Args:
        session: Database session
        movie_id: Movie ID
        
    Returns:
        Dictionary with statistics:
        - count: Number of ratings
        - average: Average rating
        - min: Minimum rating
        - max: Maximum rating
    """
    stats = session.query(
        func.count(Rating.rating_id).label('count'),
        func.avg(Rating.rating).label('average'),
        func.min(Rating.rating).label('min'),
        func.max(Rating.rating).label('max')
    ).filter(Rating.movie_id == movie_id).first()
    
    return {
        'count': stats.count or 0,
        'average': float(stats.average) if stats.average else 0.0,
        'min': float(stats.min) if stats.min else 0.0,
        'max': float(stats.max) if stats.max else 0.0
    }


def update_rating(
    session: Session,
    rating_id: int,
    new_rating: float
) -> Optional[Rating]:
    """
    Update a rating value.
    
    Args:
        session: Database session
        rating_id: Rating ID
        new_rating: New rating value (1.0 to 5.0)
        
    Returns:
        Updated Rating object or None if not found
        
    Raises:
        ValueError: If rating is not between 1 and 5
    """
    if not (1.0 <= new_rating <= 5.0):
        raise ValueError("Rating must be between 1.0 and 5.0")
    
    rating = get_rating(session, rating_id)
    if rating:
        rating.rating = new_rating
        session.commit()
        session.refresh(rating)
    return rating


def delete_rating(session: Session, rating_id: int) -> bool:
    """
    Delete a rating.
    
    Args:
        session: Database session
        rating_id: Rating ID
        
    Returns:
        True if rating was deleted, False if not found
    """
    rating = get_rating(session, rating_id)
    if rating:
        session.delete(rating)
        session.commit()
        return True
    return False


def get_all_ratings(
    session: Session,
    skip: int = 0,
    limit: int = 1000
) -> List[Rating]:
    """
    Get all ratings with pagination.
    
    Args:
        session: Database session
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List of Rating objects
    """
    return session.query(Rating).offset(skip).limit(limit).all()


def get_rating_count(session: Session) -> int:
    """
    Get total count of ratings.
    
    Args:
        session: Database session
        
    Returns:
        Total number of ratings
    """
    return session.query(func.count(Rating.rating_id)).scalar()


def get_global_rating_stats(session: Session) -> Dict[str, Any]:
    """
    Get global rating statistics (all ratings).
    
    Args:
        session: Database session
        
    Returns:
        Dictionary with statistics:
        - count: Total number of ratings
        - average: Average rating
        - min: Minimum rating
        - max: Maximum rating
    """
    stats = session.query(
        func.count(Rating.rating_id).label('count'),
        func.avg(Rating.rating).label('average'),
        func.min(Rating.rating).label('min'),
        func.max(Rating.rating).label('max')
    ).first()
    
    return {
        'count': stats.count or 0,
        'average': float(stats.average) if stats.average else 0.0,
        'min': float(stats.min) if stats.min else 0.0,
        'max': float(stats.max) if stats.max else 0.0
    }
