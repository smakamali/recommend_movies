"""
SQLAlchemy ORM models for the recommender system database.

This module defines the User, Movie, and Rating tables with proper relationships
and constraints as specified in ARCHITECTURE_MVP.md.
"""

from datetime import datetime
from typing import List
from sqlalchemy import (
    Column, Integer, String, Float, Text, ForeignKey, 
    CheckConstraint, UniqueConstraint, Index, TIMESTAMP
)
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class User(Base):
    """
    User table storing demographic information.
    
    Attributes:
        user_id: Primary key, auto-incremented
        age: User's age (required)
        gender: User's gender ('M', 'F', or 'O')
        occupation: User's occupation
        zip_code: User's zip code (optional)
        created_at: Timestamp when record was created
        updated_at: Timestamp when record was last updated
    """
    __tablename__ = 'users'
    
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[str] = mapped_column(String(1), nullable=False)
    occupation: Mapped[str] = mapped_column(String(50), nullable=False)
    zip_code: Mapped[str] = mapped_column(String(10), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP, 
        nullable=False, 
        server_default=func.current_timestamp()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP, 
        nullable=False, 
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp()
    )
    
    # Relationships
    ratings: Mapped[List["Rating"]] = relationship(
        "Rating", 
        back_populates="user",
        cascade="all, delete-orphan"
    )
    
    # Constraints
    __table_args__ = (
        CheckConstraint("gender IN ('M', 'F', 'O')", name='check_gender'),
    )
    
    def __repr__(self) -> str:
        return f"<User(user_id={self.user_id}, age={self.age}, gender='{self.gender}', occupation='{self.occupation}')>"


class Movie(Base):
    """
    Movie table storing movie information and metadata.
    
    Attributes:
        movie_id: Primary key (from MovieLens dataset)
        title: Movie title (required)
        release_year: Year the movie was released
        genres: JSON array of genres stored as text
        imdb_url: URL to IMDB page
        created_at: Timestamp when record was created
    """
    __tablename__ = 'movies'
    
    movie_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    release_year: Mapped[int] = mapped_column(Integer, nullable=True)
    genres: Mapped[str] = mapped_column(Text, nullable=False)  # JSON array as text
    imdb_url: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.current_timestamp()
    )
    
    # Relationships
    ratings: Mapped[List["Rating"]] = relationship(
        "Rating",
        back_populates="movie",
        cascade="all, delete-orphan"
    )
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_movies_title', 'title'),
        Index('idx_movies_year', 'release_year'),
    )
    
    def __repr__(self) -> str:
        return f"<Movie(movie_id={self.movie_id}, title='{self.title}', year={self.release_year})>"


class Rating(Base):
    """
    Rating table storing user ratings for movies.
    
    Attributes:
        rating_id: Primary key, auto-incremented
        user_id: Foreign key to users table
        movie_id: Foreign key to movies table
        rating: Rating value (1.0 to 5.0)
        created_at: Timestamp when rating was created
        updated_at: Timestamp when rating was last updated
    """
    __tablename__ = 'ratings'
    
    rating_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, 
        ForeignKey('users.user_id', ondelete='CASCADE'),
        nullable=False
    )
    movie_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey('movies.movie_id', ondelete='CASCADE'),
        nullable=False
    )
    rating: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.current_timestamp()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp()
    )
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="ratings")
    movie: Mapped["Movie"] = relationship("Movie", back_populates="ratings")
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("rating >= 1 AND rating <= 5", name='check_rating_range'),
        UniqueConstraint('user_id', 'movie_id', name='unique_user_movie'),
        Index('idx_ratings_user', 'user_id'),
        Index('idx_ratings_movie', 'movie_id'),
        Index('idx_ratings_timestamp', 'created_at'),
    )
    
    def __repr__(self) -> str:
        return f"<Rating(rating_id={self.rating_id}, user_id={self.user_id}, movie_id={self.movie_id}, rating={self.rating})>"
