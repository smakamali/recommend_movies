"""
Pydantic schemas for Movie API.
"""

from pydantic import BaseModel


class MovieResponse(BaseModel):
    """Response model for a single movie."""

    movie_id: int
    title: str
    release_year: int | None
    genres: str  # JSON array as string
    imdb_url: str | None

    class Config:
        from_attributes = True


class MovieList(BaseModel):
    """Response model for list of movies with total count."""

    movies: list[MovieResponse]
    total: int
