"""
Movie API endpoints.
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.api.models.movie import MovieResponse, MovieList
from app.database import crud

router = APIRouter(prefix="/api/movies", tags=["movies"])


@router.get("", response_model=MovieList)
def list_movies(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """List movies with pagination."""
    movies = crud.get_movies(db, skip=skip, limit=limit)
    total = crud.get_movie_count(db)
    return MovieList(
        movies=[MovieResponse.model_validate(m) for m in movies],
        total=total,
    )


@router.get("/search", response_model=MovieList)
def search_movies(
    title: str | None = Query(None),
    year: int | None = Query(None),
    genre: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Search movies by title, year, or genre."""
    movies = crud.search_movies(db, title=title, year=year, genre=genre, limit=limit)
    return MovieList(
        movies=[MovieResponse.model_validate(m) for m in movies],
        total=len(movies),
    )


@router.get("/{movie_id}", response_model=MovieResponse)
def get_movie(movie_id: int, db: Session = Depends(get_db)):
    """Get movie details by ID."""
    movie = crud.get_movie(db, movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    return movie
