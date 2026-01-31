"""
Unit tests for database CRUD operations.

Tests for User, Movie, and Rating CRUD operations using an in-memory
SQLite database for fast, isolated testing.
"""

import pytest
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base, User, Movie, Rating
from app.database import crud


@pytest.fixture
def engine():
    """Create an in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(engine):
    """Create a new database session for testing."""
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestUserCRUD:
    """Tests for User CRUD operations."""
    
    def test_create_user(self, session):
        """Test creating a new user."""
        user = crud.create_user(
            session,
            age=25,
            gender='M',
            occupation='engineer',
            zip_code='12345'
        )
        
        assert user.user_id is not None
        assert user.age == 25
        assert user.gender == 'M'
        assert user.occupation == 'engineer'
        assert user.zip_code == '12345'
    
    def test_create_user_invalid_gender(self, session):
        """Test that creating a user with invalid gender raises error."""
        with pytest.raises(ValueError):
            crud.create_user(
                session,
                age=25,
                gender='X',  # Invalid
                occupation='engineer'
            )
    
    def test_get_user(self, session):
        """Test retrieving a user by ID."""
        user = crud.create_user(
            session,
            age=30,
            gender='F',
            occupation='doctor'
        )
        
        retrieved = crud.get_user(session, user.user_id)
        assert retrieved is not None
        assert retrieved.user_id == user.user_id
        assert retrieved.age == 30
    
    def test_get_user_not_found(self, session):
        """Test that getting a non-existent user returns None."""
        user = crud.get_user(session, 999)
        assert user is None
    
    def test_get_users(self, session):
        """Test retrieving multiple users with pagination."""
        # Create 5 users
        for i in range(5):
            crud.create_user(
                session,
                age=20 + i,
                gender='M',
                occupation=f'job_{i}'
            )
        
        # Get first 3 users
        users = crud.get_users(session, skip=0, limit=3)
        assert len(users) == 3
        
        # Get next 2 users
        users = crud.get_users(session, skip=3, limit=3)
        assert len(users) == 2
    
    def test_update_user(self, session):
        """Test updating user information."""
        user = crud.create_user(
            session,
            age=25,
            gender='M',
            occupation='engineer'
        )
        
        updated = crud.update_user(
            session,
            user.user_id,
            age=26,
            occupation='senior engineer'
        )
        
        assert updated.age == 26
        assert updated.occupation == 'senior engineer'
        assert updated.gender == 'M'  # Unchanged
    
    def test_delete_user(self, session):
        """Test deleting a user."""
        user = crud.create_user(
            session,
            age=25,
            gender='M',
            occupation='engineer'
        )
        
        result = crud.delete_user(session, user.user_id)
        assert result is True
        
        # Verify user is deleted
        deleted = crud.get_user(session, user.user_id)
        assert deleted is None
    
    def test_get_user_count(self, session):
        """Test getting total user count."""
        assert crud.get_user_count(session) == 0
        
        crud.create_user(session, age=25, gender='M', occupation='engineer')
        crud.create_user(session, age=30, gender='F', occupation='doctor')
        
        assert crud.get_user_count(session) == 2


class TestMovieCRUD:
    """Tests for Movie CRUD operations."""
    
    def test_create_movie(self, session):
        """Test creating a new movie."""
        genres = json.dumps(['Action', 'Thriller'])
        movie = crud.create_movie(
            session,
            movie_id=1,
            title='Test Movie',
            genres=genres,
            release_year=1995,
            imdb_url='http://imdb.com/title/tt0000001'
        )
        
        assert movie.movie_id == 1
        assert movie.title == 'Test Movie'
        assert movie.genres == genres
        assert movie.release_year == 1995
    
    def test_get_movie(self, session):
        """Test retrieving a movie by ID."""
        genres = json.dumps(['Comedy'])
        movie = crud.create_movie(
            session,
            movie_id=1,
            title='Test Movie',
            genres=genres
        )
        
        retrieved = crud.get_movie(session, 1)
        assert retrieved is not None
        assert retrieved.movie_id == 1
        assert retrieved.title == 'Test Movie'
    
    def test_get_movies(self, session):
        """Test retrieving multiple movies with pagination."""
        # Create 5 movies
        for i in range(5):
            crud.create_movie(
                session,
                movie_id=i + 1,
                title=f'Movie {i}',
                genres=json.dumps(['Action'])
            )
        
        movies = crud.get_movies(session, skip=0, limit=3)
        assert len(movies) == 3
    
    def test_search_movies_by_title(self, session):
        """Test searching movies by title."""
        crud.create_movie(
            session,
            movie_id=1,
            title='The Matrix',
            genres=json.dumps(['Action'])
        )
        crud.create_movie(
            session,
            movie_id=2,
            title='The Matrix Reloaded',
            genres=json.dumps(['Action'])
        )
        crud.create_movie(
            session,
            movie_id=3,
            title='Inception',
            genres=json.dumps(['Action'])
        )
        
        results = crud.search_movies(session, title='Matrix')
        assert len(results) == 2
    
    def test_search_movies_by_year(self, session):
        """Test searching movies by release year."""
        crud.create_movie(
            session,
            movie_id=1,
            title='Movie 1995',
            genres=json.dumps(['Action']),
            release_year=1995
        )
        crud.create_movie(
            session,
            movie_id=2,
            title='Movie 1996',
            genres=json.dumps(['Action']),
            release_year=1996
        )
        
        results = crud.search_movies(session, year=1995)
        assert len(results) == 1
        assert results[0].release_year == 1995
    
    def test_search_movies_by_genre(self, session):
        """Test searching movies by genre."""
        crud.create_movie(
            session,
            movie_id=1,
            title='Action Movie',
            genres=json.dumps(['Action', 'Thriller'])
        )
        crud.create_movie(
            session,
            movie_id=2,
            title='Comedy Movie',
            genres=json.dumps(['Comedy'])
        )
        
        results = crud.search_movies(session, genre='Comedy')
        assert len(results) == 1
        assert 'Comedy' in results[0].genres
    
    def test_get_movie_count(self, session):
        """Test getting total movie count."""
        assert crud.get_movie_count(session) == 0
        
        crud.create_movie(session, movie_id=1, title='Movie 1', genres=json.dumps(['Action']))
        crud.create_movie(session, movie_id=2, title='Movie 2', genres=json.dumps(['Comedy']))
        
        assert crud.get_movie_count(session) == 2


class TestRatingCRUD:
    """Tests for Rating CRUD operations."""
    
    @pytest.fixture
    def setup_user_and_movie(self, session):
        """Create a user and movie for rating tests."""
        user = crud.create_user(
            session,
            age=25,
            gender='M',
            occupation='engineer'
        )
        movie = crud.create_movie(
            session,
            movie_id=1,
            title='Test Movie',
            genres=json.dumps(['Action'])
        )
        return user, movie
    
    def test_create_rating(self, session, setup_user_and_movie):
        """Test creating a new rating."""
        user, movie = setup_user_and_movie
        
        rating = crud.create_rating(
            session,
            user_id=user.user_id,
            movie_id=movie.movie_id,
            rating=4.5
        )
        
        assert rating.rating_id is not None
        assert rating.user_id == user.user_id
        assert rating.movie_id == movie.movie_id
        assert rating.rating == 4.5
    
    def test_create_rating_invalid_value(self, session, setup_user_and_movie):
        """Test that creating a rating with invalid value raises error."""
        user, movie = setup_user_and_movie
        
        with pytest.raises(ValueError):
            crud.create_rating(
                session,
                user_id=user.user_id,
                movie_id=movie.movie_id,
                rating=6.0  # Invalid (> 5.0)
            )
    
    def test_get_rating(self, session, setup_user_and_movie):
        """Test retrieving a rating by ID."""
        user, movie = setup_user_and_movie
        
        rating = crud.create_rating(
            session,
            user_id=user.user_id,
            movie_id=movie.movie_id,
            rating=4.0
        )
        
        retrieved = crud.get_rating(session, rating.rating_id)
        assert retrieved is not None
        assert retrieved.rating_id == rating.rating_id
    
    def test_get_rating_by_user_movie(self, session, setup_user_and_movie):
        """Test retrieving a rating by user and movie."""
        user, movie = setup_user_and_movie
        
        crud.create_rating(
            session,
            user_id=user.user_id,
            movie_id=movie.movie_id,
            rating=4.0
        )
        
        rating = crud.get_rating_by_user_movie(
            session,
            user_id=user.user_id,
            movie_id=movie.movie_id
        )
        
        assert rating is not None
        assert rating.rating == 4.0
    
    def test_get_ratings_by_user(self, session, setup_user_and_movie):
        """Test retrieving all ratings by a user."""
        user, movie = setup_user_and_movie
        
        # Create another movie
        movie2 = crud.create_movie(
            session,
            movie_id=2,
            title='Movie 2',
            genres=json.dumps(['Comedy'])
        )
        
        # Create ratings
        crud.create_rating(session, user.user_id, movie.movie_id, 4.0)
        crud.create_rating(session, user.user_id, movie2.movie_id, 5.0)
        
        ratings = crud.get_ratings_by_user(session, user.user_id)
        assert len(ratings) == 2
    
    def test_get_ratings_by_movie(self, session, setup_user_and_movie):
        """Test retrieving all ratings for a movie."""
        user, movie = setup_user_and_movie
        
        # Create another user
        user2 = crud.create_user(
            session,
            age=30,
            gender='F',
            occupation='doctor'
        )
        
        # Create ratings
        crud.create_rating(session, user.user_id, movie.movie_id, 4.0)
        crud.create_rating(session, user2.user_id, movie.movie_id, 5.0)
        
        ratings = crud.get_ratings_by_movie(session, movie.movie_id)
        assert len(ratings) == 2
    
    def test_get_rating_stats(self, session, setup_user_and_movie):
        """Test getting rating statistics for a movie."""
        user, movie = setup_user_and_movie
        
        # Create another user
        user2 = crud.create_user(
            session,
            age=30,
            gender='F',
            occupation='doctor'
        )
        
        # Create ratings
        crud.create_rating(session, user.user_id, movie.movie_id, 3.0)
        crud.create_rating(session, user2.user_id, movie.movie_id, 5.0)
        
        stats = crud.get_rating_stats(session, movie.movie_id)
        
        assert stats['count'] == 2
        assert stats['average'] == 4.0
        assert stats['min'] == 3.0
        assert stats['max'] == 5.0
    
    def test_update_rating(self, session, setup_user_and_movie):
        """Test updating a rating value."""
        user, movie = setup_user_and_movie
        
        rating = crud.create_rating(
            session,
            user_id=user.user_id,
            movie_id=movie.movie_id,
            rating=3.0
        )
        
        updated = crud.update_rating(session, rating.rating_id, 4.5)
        assert updated.rating == 4.5
    
    def test_delete_rating(self, session, setup_user_and_movie):
        """Test deleting a rating."""
        user, movie = setup_user_and_movie
        
        rating = crud.create_rating(
            session,
            user_id=user.user_id,
            movie_id=movie.movie_id,
            rating=4.0
        )
        
        result = crud.delete_rating(session, rating.rating_id)
        assert result is True
        
        # Verify rating is deleted
        deleted = crud.get_rating(session, rating.rating_id)
        assert deleted is None
    
    def test_unique_user_movie_constraint(self, session, setup_user_and_movie):
        """Test that a user can only rate a movie once."""
        user, movie = setup_user_and_movie
        
        crud.create_rating(
            session,
            user_id=user.user_id,
            movie_id=movie.movie_id,
            rating=4.0
        )
        
        # Try to create duplicate rating
        with pytest.raises(Exception):  # SQLAlchemy will raise IntegrityError
            crud.create_rating(
                session,
                user_id=user.user_id,
                movie_id=movie.movie_id,
                rating=5.0
            )
    
    def test_get_rating_count(self, session, setup_user_and_movie):
        """Test getting total rating count."""
        user, movie = setup_user_and_movie
        
        assert crud.get_rating_count(session) == 0
        
        crud.create_rating(session, user.user_id, movie.movie_id, 4.0)
        
        assert crud.get_rating_count(session) == 1


class TestRelationships:
    """Tests for model relationships and cascading deletes."""
    
    def test_user_ratings_relationship(self, session):
        """Test user-ratings relationship."""
        user = crud.create_user(
            session,
            age=25,
            gender='M',
            occupation='engineer'
        )
        movie = crud.create_movie(
            session,
            movie_id=1,
            title='Test Movie',
            genres=json.dumps(['Action'])
        )
        
        crud.create_rating(session, user.user_id, movie.movie_id, 4.0)
        
        # Refresh user to load relationships
        session.refresh(user)
        assert len(user.ratings) == 1
        assert user.ratings[0].rating == 4.0
    
    def test_movie_ratings_relationship(self, session):
        """Test movie-ratings relationship."""
        user = crud.create_user(
            session,
            age=25,
            gender='M',
            occupation='engineer'
        )
        movie = crud.create_movie(
            session,
            movie_id=1,
            title='Test Movie',
            genres=json.dumps(['Action'])
        )
        
        crud.create_rating(session, user.user_id, movie.movie_id, 4.0)
        
        # Refresh movie to load relationships
        session.refresh(movie)
        assert len(movie.ratings) == 1
        assert movie.ratings[0].rating == 4.0
    
    def test_cascade_delete_user(self, session):
        """Test that deleting a user cascades to ratings."""
        user = crud.create_user(
            session,
            age=25,
            gender='M',
            occupation='engineer'
        )
        movie = crud.create_movie(
            session,
            movie_id=1,
            title='Test Movie',
            genres=json.dumps(['Action'])
        )
        
        rating = crud.create_rating(session, user.user_id, movie.movie_id, 4.0)
        rating_id = rating.rating_id
        
        # Delete user
        crud.delete_user(session, user.user_id)
        
        # Verify rating is also deleted
        deleted_rating = crud.get_rating(session, rating_id)
        assert deleted_rating is None
    
    def test_cascade_delete_movie(self, session):
        """Test that deleting a movie cascades to ratings."""
        user = crud.create_user(
            session,
            age=25,
            gender='M',
            occupation='engineer'
        )
        movie = crud.create_movie(
            session,
            movie_id=1,
            title='Test Movie',
            genres=json.dumps(['Action'])
        )
        
        rating = crud.create_rating(session, user.user_id, movie.movie_id, 4.0)
        rating_id = rating.rating_id
        
        # Delete movie
        session.delete(movie)
        session.commit()
        
        # Verify rating is also deleted
        deleted_rating = crud.get_rating(session, rating_id)
        assert deleted_rating is None
