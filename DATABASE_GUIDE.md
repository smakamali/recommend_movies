# GraphSAGE Recommender System - Database Documentation

Complete guide for the database layer of the GraphSAGE recommender system built on MovieLens 100K dataset.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Database Overview](#database-overview)
3. [Installation](#installation)
4. [Data Import](#data-import)
5. [API Reference](#api-reference)
6. [Scripts](#scripts)
7. [Troubleshooting](#troubleshooting)
8. [Architecture](#architecture)

---

## Quick Start

See [README.md](README.md) for environment setup. To import data and verify:

```bash
python scripts/init_database.py --reset
python scripts/verify_database.py
```

### Query the Database

```python
from app.database import get_db_manager, crud

db_manager = get_db_manager()
session = db_manager.get_session()

# Get statistics
print(f"Movies: {crud.get_movie_count(session):,}")
print(f"Users: {crud.get_user_count(session):,}")
print(f"Ratings: {crud.get_rating_count(session):,}")

# Search movies
movies = crud.search_movies(session, title="Star Wars")

# Get user ratings
ratings = crud.get_ratings_by_user(session, user_id=1)

session.close()
```

---

## Database Overview

### Dataset: MovieLens 100K

- **Users**: 943 with demographics (age, gender, occupation)
- **Movies**: 1,682 with metadata (title, year, genres, IMDB URL)
- **Ratings**: 100,000 user-movie interactions (1-5 stars)
- **Density**: ~6.3% (sparse matrix ideal for collaborative filtering)

### Schema

#### Users Table
```sql
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER NOT NULL,
    gender TEXT NOT NULL CHECK(gender IN ('M', 'F', 'O')),
    occupation TEXT NOT NULL,
    zip_code TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Movies Table
```sql
CREATE TABLE movies (
    movie_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    release_year INTEGER,
    genres TEXT NOT NULL,  -- JSON array
    imdb_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_movies_title ON movies(title);
CREATE INDEX idx_movies_year ON movies(release_year);
```

#### Ratings Table
```sql
CREATE TABLE ratings (
    rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    movie_id INTEGER NOT NULL,
    rating REAL NOT NULL CHECK(rating >= 1 AND rating <= 5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    UNIQUE(user_id, movie_id)
);
CREATE INDEX idx_ratings_user ON ratings(user_id);
CREATE INDEX idx_ratings_movie ON ratings(movie_id);
CREATE INDEX idx_ratings_timestamp ON ratings(created_at);
```

---

## Installation

See [README.md](README.md) for full setup. Prerequisites: Python 3.10, ~50 MB disk space. MovieLens 100K is auto-downloaded on first import.

### Dependencies

- **SQLAlchemy** 2.0+ - ORM and database toolkit
- **pandas** 2.0+ - Data manipulation
- **numpy** - Numerical computing
- **pytest** 9.0+ - Testing framework

---

## Data Import

### Full Import (Recommended)

Import movies, users, and ratings in one command:

```bash
python scripts/init_database.py --reset
```

**Output:**
```
============================================================
MovieLens 100K Database Initialization
============================================================

============================================================
Importing Movies
============================================================
  Imported 200 movies...
  Imported 400 movies...
  ...
[SUCCESS] Movie import complete!
  Imported: 1,682 movies

============================================================
Importing Users
============================================================
  Imported 200 users...
  ...
[SUCCESS] User import complete!
  Imported: 943 users

============================================================
Importing Ratings
============================================================
  Inserted 5,000/100,000 ratings...
  ...
[SUCCESS] Ratings import complete!
  Imported: 100,000 ratings
  Rate: 30,538 ratings/second

[SUCCESS] Database initialization complete!
Total time: 8.5s
```

### Partial Import Options

```bash
# Import only ratings (if movies/users already exist)
python scripts/init_database.py --ratings-only

# Import movies and users only (skip ratings)
python scripts/init_database.py --reset --skip-ratings

# Append mode (keep existing data)
python scripts/init_database.py --no-reset

# Custom batch size for ratings
python scripts/init_database.py --reset --batch-size 10000
```

### Performance

- **Movies**: ~0.5 seconds (1,682 records)
- **Users**: ~0.3 seconds (943 records)
- **Ratings**: ~3 seconds (100,000 records at 30K/sec)
- **Total**: ~8 seconds for complete import

---

## API Reference

### Database Connection

```python
from app.database import get_db_manager

# Get database manager (singleton)
db_manager = get_db_manager()

# Get session
session = db_manager.get_session()

# Use context manager (recommended)
with db_manager.session_scope() as session:
    # Your queries here
    pass  # Auto-commits on success, rolls back on error
```

### User Operations

```python
from app.database import crud

# Create user
user = crud.create_user(
    session,
    age=25,
    gender='M',
    occupation='engineer',
    zip_code='12345'
)

# Get user
user = crud.get_user(session, user_id=1)

# List users with pagination
users = crud.get_users(session, skip=0, limit=100)

# Update user
updated = crud.update_user(session, user_id=1, age=26)

# Delete user (cascades to ratings)
success = crud.delete_user(session, user_id=1)

# Get user's ratings
ratings = crud.get_user_ratings(session, user_id=1)

# Count users
count = crud.get_user_count(session)
```

### Movie Operations

```python
# Get movie
movie = crud.get_movie(session, movie_id=50)

# List movies
movies = crud.get_movies(session, skip=0, limit=100)

# Search movies
results = crud.search_movies(
    session,
    title="Star Wars",     # Partial match
    year=1977,             # Exact match
    genre="Action",        # Contains
    limit=10
)

# Get movies by year
movies = crud.get_movies_by_year(session, year=1995)

# Count movies
count = crud.get_movie_count(session)
```

### Rating Operations

```python
# Create rating
rating = crud.create_rating(
    session,
    user_id=1,
    movie_id=50,
    rating=5.0
)

# Get rating
rating = crud.get_rating(session, rating_id=1)

# Get specific user-movie rating
rating = crud.get_rating_by_user_movie(session, user_id=1, movie_id=50)

# Get user's ratings
ratings = crud.get_ratings_by_user(session, user_id=1, limit=20)

# Get movie's ratings
ratings = crud.get_ratings_by_movie(session, movie_id=50, limit=20)

# Get rating statistics for movie
stats = crud.get_rating_stats(session, movie_id=50)
# Returns: {'count': 583, 'average': 4.36, 'min': 1.0, 'max': 5.0}

# Update rating
updated = crud.update_rating(session, rating_id=1, new_rating=4.5)

# Delete rating
success = crud.delete_rating(session, rating_id=1)

# Count ratings
count = crud.get_rating_count(session)
```

---

## Scripts

### init_database.py

Master import script for complete database setup.

```bash
# Full import (recommended)
python scripts/init_database.py --reset

# Import only ratings
python scripts/init_database.py --ratings-only

# Skip ratings
python scripts/init_database.py --reset --skip-ratings

# Keep existing data
python scripts/init_database.py --no-reset

# Custom database path
python scripts/init_database.py --reset --db-path /path/to/db.db

# Quiet mode
python scripts/init_database.py --reset --quiet
```

### verify_database.py

Comprehensive verification and validation.

```bash
# Full verification (recommended)
python scripts/verify_database.py

# Quick check only
python scripts/verify_database.py --quick

# Check for duplicates
python scripts/verify_database.py --check-duplicates
```

**Checks performed:**
1. Basic statistics (counts)
2. Rating distribution
3. Foreign key integrity
4. Duplicate detection
5. Coverage analysis
6. Sample queries
7. Data quality

### demo_database.py

Interactive demonstration of database features.

```bash
python scripts/demo_database.py
```

**Demonstrates:**
- Basic queries
- Search functionality
- Rating statistics
- ORM relationships
- Pagination

---

## Troubleshooting

### Import Fails with "Dataset not found"

**Solution:** Download MovieLens 100K manually:

```bash
# Download
wget http://files.grouplens.org/datasets/movielens/ml-100k.zip

# Extract
unzip ml-100k.zip -d ~/.surprise_data/

# Or specify path
python scripts/init_database.py --reset --data-path /path/to/ml-100k
```

### "UNIQUE constraint failed" Error

**Cause:** Trying to import ratings when they already exist.

**Solution:** Use `--reset` to clear existing data:

```bash
python scripts/init_database.py --reset
```

### Import Errors

**Symptom:** Module not found or compatibility errors.

**Solution:** Ensure conda environment is activated and dependencies are installed:

```bash
conda activate recommender
pip install -r requirements.txt
```

### Foreign Key Violations

**Cause:** Ratings reference non-existent users/movies.

**Solution:** Import movies and users before ratings:

```bash
python scripts/init_database.py --reset --skip-ratings
python scripts/init_database.py --ratings-only
```

### Database Locked

**Cause:** Another process is using the database.

**Solution:** 
1. Close all connections
2. Delete `.db-journal` file if exists
3. Restart import

### Slow Import

**Cause:** Large batch size or slow disk.

**Solution:** Adjust batch size:

```bash
python scripts/init_database.py --reset --batch-size 1000
```

---

## Architecture

### Technology Stack

- **ORM**: SQLAlchemy 2.0 with declarative base
- **Database**: SQLite (embedded, file-based)
- **Language**: Python 3.10
- **Testing**: pytest with in-memory SQLite

### Design Principles

1. **Separation of Concerns**: Models, CRUD, Connection separated
2. **Type Safety**: Modern SQLAlchemy with type hints
3. **Data Integrity**: Foreign keys, constraints, indexes
4. **Performance**: Bulk operations, indexed queries
5. **Testability**: In-memory testing, fixtures

### Project Structure

```
recommender_system/
├── app/
│   └── database/
│       ├── __init__.py       # Package exports
│       ├── models.py         # ORM models
│       ├── connection.py     # DB manager
│       ├── crud.py           # CRUD operations
│       └── init_db.py        # Initialization
├── scripts/
│   ├── init_database.py      # Master import
│   ├── verify_database.py    # Verification
│   └── demo_database.py      # Demonstration
├── tests/
│   └── database/
│       └── test_crud.py      # Unit tests (30 tests)
├── data/
│   └── recommender.db        # SQLite database
├── environment.yml           # Conda spec
├── requirements.txt          # Pip dependencies
└── DATABASE_GUIDE.md         # This file
```

### Data Flow

```
MovieLens Files → Import Scripts → SQLite Database → CRUD API → Application
    (u.data)    → (init_database) →  (recommender.db) → (crud.py) → (Your Code)
```

### Performance Optimizations

1. **Indexes**: On frequently queried columns (title, year, user_id, movie_id)
2. **Bulk Insert**: 30K+ ratings/second with batch operations
3. **Connection Pooling**: StaticPool for SQLite
4. **Query Optimization**: Proper joins, subqueries
5. **Pagination**: Efficient LIMIT/OFFSET queries

---

## Advanced Usage

### Custom Database Path

```python
from app.database import DatabaseManager

db_manager = DatabaseManager(db_path='custom/path/db.db')
session = db_manager.get_session()
```

### Debug Mode (SQL Logging)

```python
db_manager = DatabaseManager(echo=True)  # Logs all SQL
```

### Batch Operations

```python
# Bulk insert ratings
ratings_data = [
    {'user_id': 1, 'movie_id': 50, 'rating': 5.0},
    {'user_id': 2, 'movie_id': 50, 'rating': 4.0},
    # ... more ratings
]

from app.database.models import Rating
session.bulk_insert_mappings(Rating, ratings_data)
session.commit()
```

### Complex Queries

```python
from sqlalchemy import and_, or_, func
from app.database.models import Movie, Rating

# Movies with high average rating
high_rated = session.query(
    Movie.title,
    func.avg(Rating.rating).label('avg_rating')
).join(Rating).group_by(
    Movie.title
).having(
    func.avg(Rating.rating) >= 4.0
).order_by(
    func.avg(Rating.rating).desc()
).limit(10).all()
```

---

## Testing

### Run Tests

```bash
# All tests
pytest tests/database/test_crud.py -v

# Specific test class
pytest tests/database/test_crud.py::TestUserCRUD -v

# With coverage
pytest tests/database/test_crud.py --cov=app.database --cov-report=html
```

### Test Coverage

- 30 unit tests
- 100% pass rate
- Coverage: All CRUD operations, relationships, constraints

---

## Next Steps

After database setup, proceed to:

1. **Phase 2**: Graph Construction
   - Build bipartite user-movie graph
   - Extract node features
   - Implement neighbor sampling

2. **Phase 3**: GraphSAGE Model
   - Define model architecture
   - Train on graph data
   - Generate embeddings

3. **Phase 4**: API Development
   - REST API for recommendations
   - Real-time inference
   - Monitoring and logging

---

## Support

### Documentation

- API Reference: See [API Reference](#api-reference) above
- Scripts: See [Scripts](#scripts) above
- Troubleshooting: See [Troubleshooting](#troubleshooting) above

### Files

- Architecture: `ARCHITECTURE_MVP.md`
- Test Examples: `tests/database/test_crud.py`
- Demo Script: `scripts/demo_database.py`

---

**Version**: 1.0  
**Last Updated**: 2026-01-31  
**Status**: Production Ready ✅  
**Phase**: 1 Complete - Ready for Phase 2
