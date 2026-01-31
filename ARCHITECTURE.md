# GraphSAGE Recommender System - High-Level Architecture

## Executive Summary

This document outlines the high-level architecture for a user-facing GraphSAGE-based movie recommender system. The system provides personalized movie recommendations using Graph Neural Networks (GNN), supporting both cold-start scenarios (demographic-only) and warm-start scenarios (with user ratings).

## System Overview

### Core Capabilities

1. **Cold-Start Recommendations**: Generate initial recommendations for new users based solely on demographic information (age, gender, occupation, zip_code)
2. **Warm-Start Recommendations**: Provide improved recommendations as users rate movies by incorporating their rating history
3. **Real-Time Updates**: Allow users to add ratings and see updated recommendations without system restart
4. **Interactive UI**: User-friendly web interface for rating movies and viewing recommendations
5. **Persistent Storage**: Maintain user profiles, ratings, and movie data in a lightweight database

### Design Principles

- **Modularity**: Separate concerns into distinct services (training, inference, UI, storage)
- **Scalability**: Design for future horizontal scaling and async processing
- **Simplicity**: Use lightweight components suitable for demonstration and development
- **Extensibility**: Allow easy addition of new features and model improvements

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                        │
│                        (Streamlit Web App)                          │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ HTTP/REST
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Application Layer                              │
│                       (FastAPI Server)                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │
│  │ User Management  │  │   Recommendation │  │  Rating Service  │ │
│  │    Endpoint      │  │     Endpoint     │  │     Endpoint     │ │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘ │
└──────────────────┬──────────────────┬────────────────────┬─────────┘
                   │                  │                    │
                   ▼                  ▼                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Business Logic Layer                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │
│  │   Inference      │  │  Graph Builder   │  │  Feature         │ │
│  │   Engine         │  │  & Preprocessor  │  │  Extractor       │ │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘ │
└──────────────────┬──────────────────┬────────────────────┬─────────┘
                   │                  │                    │
                   ▼                  ▼                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Model Layer                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │
│  │  Trained         │  │  Trained         │  │   Model          │ │
│  │  GraphSAGE       │  │  Preprocessor    │  │   Metadata       │ │
│  │  Model (.pth)    │  │  (.pkl)          │  │   (.json)        │ │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Data Layer                                   │
│                    (SQLite Database)                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │    Users     │  │   Ratings    │  │    Movies    │            │
│  │    Table     │  │    Table     │  │    Table     │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Training Pipeline (Offline)                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  1. Data Loading   →  2. Graph Building  →  3. Model Training │  │
│  │  4. Validation     →  5. Model Saving    →  6. Deployment     │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. User Interface Layer (Streamlit)

**Purpose**: Provide an interactive web interface for end users

**Components**:
- **User Registration Form**: Collect demographic information (age, gender, occupation, zip_code)
- **Movie Rating Panel**: Display movies with rating interface (1-5 stars)
- **Recommendation Display**: Show personalized movie recommendations
- **User Profile View**: Display user information and rating history

**Technology**: Streamlit (Python web framework for ML applications)

**Key Features**:
- Simple, intuitive UI with minimal learning curve
- Real-time updates when user adds ratings
- Responsive design for desktop and tablet
- Session state management for user context

### 2. Application Layer (FastAPI)

**Purpose**: Serve as the REST API backend for all business operations

**Endpoints**:

#### User Management
- `POST /api/users`: Create new user with demographic info
- `GET /api/users/{user_id}`: Retrieve user profile
- `GET /api/users/{user_id}/ratings`: Get user rating history

#### Ratings
- `POST /api/ratings`: Add a new rating
- `PUT /api/ratings/{rating_id}`: Update existing rating
- `DELETE /api/ratings/{rating_id}`: Remove rating
- `GET /api/ratings/stats`: Get rating statistics

#### Recommendations
- `GET /api/recommendations/{user_id}`: Get personalized recommendations (handles both cold-start and warm-start automatically)
  - Query params: `n` (default: 10), `exclude_rated` (default: true)
- `POST /api/recommendations/{user_id}/refresh`: Trigger recommendation refresh and cache invalidation

#### Movies
- `GET /api/movies`: List all movies with filters
- `GET /api/movies/{movie_id}`: Get movie details
- `GET /api/movies/search`: Search movies by title/genre

#### System
- `GET /api/health`: Health check endpoint
- `GET /api/model/info`: Get current model information

**Technology**: FastAPI (high-performance Python web framework)

**Key Features**:
- Async/await support for concurrent requests
- Automatic OpenAPI documentation
- Pydantic models for request/response validation
- CORS support for web clients
- Exception handling and logging

### 3. Business Logic Layer

#### 3.1 Inference Engine

**Purpose**: Generate recommendations using the trained GraphSAGE model

**Components**:
- **Model Loader**: Load trained model and preprocessor from disk
- **Graph Manager**: Maintain and update the user-item bipartite graph
- **Prediction Service**: Generate rating predictions and top-N recommendations (unified cold/warm-start)

**Key Operations**:
1. Load pre-trained GraphSAGE model and preprocessor
2. Add new user node to graph with demographic features (no edges initially)
3. Update graph edges when user adds ratings (incremental graph updates)
4. Generate user and item embeddings via forward pass (model handles cold/warm-start automatically)
5. Compute recommendation scores for all items
6. Filter out already-rated items
7. Return top-N recommendations

**Note**: The GraphSAGE model natively handles both cold-start (users with no ratings/edges) and warm-start (users with ratings/edges) scenarios through its message-passing architecture. No separate logic is needed.

**Caching Strategy**:
- Cache user embeddings after computation
- Invalidate cache when user adds new ratings
- Cache movie embeddings (rarely change)

#### 3.2 Graph Builder & Preprocessor

**Purpose**: Construct and maintain the bipartite user-item graph

**Components**:
- **Feature Preprocessor**: Normalize and encode user/item features
- **Graph Constructor**: Build PyTorch Geometric graph structure
- **Graph Updater**: Add nodes and edges dynamically
- **Index Manager**: Maintain user_id ↔ node_index mappings

**Key Operations**:
1. Load user and item features from database
2. Preprocess features (normalize age, one-hot encode categorical)
3. Build bipartite graph with bidirectional edges
4. Support incremental updates for new users/ratings
5. Maintain consistency between database and graph

#### 3.3 Feature Extractor

**Purpose**: Extract and transform features for users and movies

**Components**:
- **User Feature Extractor**: Age, gender, occupation, zip_code
- **Movie Feature Extractor**: Release year, genres
- **Feature Validator**: Ensure feature integrity
- **Default Feature Handler**: Provide defaults for missing values

**Feature Schema**:

**User Features**:
- `age`: Float (normalized to [0, 1])
- `gender`: One-hot encoded (M, F)
- `occupation`: One-hot encoded (21 categories)
- `zip_code`: Optional (not currently used in model)

**Movie Features**:
- `year`: Float (normalized to [0, 1])
- `genres`: Multi-hot encoded (19 genres)

### 4. Model Layer

**Purpose**: Store and version trained models and artifacts

**Artifacts**:

#### 4.1 Trained GraphSAGE Model (`graphsage_model.pth`)
- PyTorch state dict containing model weights
- Architecture: 2-layer GraphSAGE with 64-dim embeddings
- Includes rating prediction head

#### 4.2 Preprocessor (`preprocessor.pkl`)
- Fitted sklearn scalers and encoders
- User/item feature metadata
- Normalization parameters

#### 4.3 Model Metadata (`model_metadata.json`)
```json
{
  "model_version": "1.0.0",
  "training_date": "2026-01-30T00:00:00Z",
  "num_users_trained": 943,
  "num_movies": 1682,
  "hidden_dim": 64,
  "num_layers": 2,
  "loss_type": "combined",
  "val_rmse": 0.89,
  "val_precision@10": 0.42,
  "hyperparameters": {
    "learning_rate": 0.001,
    "dropout": 0.1,
    "batch_size": 512,
    "num_epochs": 50
  }
}
```

**Versioning Strategy**:
- Store models in `models/` directory with version subdirectories
- Use semantic versioning (MAJOR.MINOR.PATCH)
- Keep 3 most recent versions for rollback capability

### 5. Data Layer (SQLite)

**Purpose**: Persist all application data

**Database Schema**:

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

**Data Initialization**:
- Pre-populate `movies` table from MovieLens 100K dataset
- Import all 1,682 movies with metadata
- Users and ratings start empty

**Technology**: SQLite (embedded, serverless SQL database)

### 6. Training Pipeline (Offline)

**Purpose**: Train and update the GraphSAGE model periodically

**Stages**:

1. **Data Loading**: Load ratings from database
2. **Train/Validation Split**: 80/20 split with stratification
3. **Graph Construction**: Build bipartite graph from ratings
4. **Model Training**: Train GraphSAGE with early stopping
5. **Evaluation**: Compute metrics (RMSE, Precision@10, etc.)
6. **Model Saving**: Save model, preprocessor, and metadata
7. **Deployment**: Hot-swap the model in the inference engine

**Trigger Mechanisms**:
- **Scheduled**: Daily or weekly (cron job)
- **Manual**: Admin-triggered via CLI
- **Threshold-based**: When N new ratings accumulated

**Training Configuration**:
- Use existing `train_graphsage.py` as base
- Add model versioning and artifact management
- Include A/B testing support (future)

## Data Flow

### Cold-Start User Flow

1. User fills registration form (age, gender, occupation, zip_code)
2. Streamlit sends `POST /api/users` with demographic data
3. FastAPI creates user record in database
4. Feature Extractor extracts and preprocesses user features
5. Graph Builder adds new user node to graph (no edges yet)
6. Inference Engine generates embeddings for new user
7. Cold-Start Handler computes recommendations using demographic similarity
8. FastAPI returns top-10 movie recommendations
9. Streamlit displays movies with rating interface

### Rating Addition Flow

1. User rates a movie (1-5 stars) in UI
2. Streamlit sends `POST /api/ratings` with user_id, movie_id, rating
3. FastAPI validates and saves rating to database
4. Graph Manager adds edge between user and movie nodes
5. Inference Engine invalidates cached embeddings for user
6. FastAPI confirms rating saved
7. Streamlit shows success message

### Recommendation Refresh Flow

1. User clicks "Refresh Recommendations"
2. Streamlit sends `GET /api/recommendations/{user_id}/refresh`
3. Graph Manager ensures graph includes user's latest ratings
4. Inference Engine re-computes user embeddings with updated graph
5. Warm-Start Handler generates recommendations using rating history
6. FastAPI returns updated top-10 recommendations
7. Streamlit displays refreshed recommendations

## Project Directory Structure

```
gnn/
├── app/                                    # User-facing application
│   ├── __init__.py
│   │
│   ├── api/                                # FastAPI backend
│   │   ├── __init__.py
│   │   ├── main.py                         # FastAPI app entry point
│   │   ├── dependencies.py                 # Dependency injection
│   │   ├── config.py                       # Configuration management
│   │   │
│   │   ├── models/                         # Pydantic models (request/response)
│   │   │   ├── __init__.py
│   │   │   ├── user.py                     # User schemas
│   │   │   ├── movie.py                    # Movie schemas
│   │   │   ├── rating.py                   # Rating schemas
│   │   │   └── recommendation.py           # Recommendation schemas
│   │   │
│   │   ├── routers/                        # API route handlers
│   │   │   ├── __init__.py
│   │   │   ├── users.py                    # User endpoints
│   │   │   ├── movies.py                   # Movie endpoints
│   │   │   ├── ratings.py                  # Rating endpoints
│   │   │   ├── recommendations.py          # Recommendation endpoints
│   │   │   └── system.py                   # Health check, model info
│   │   │
│   │   └── middleware/                     # API middleware
│   │       ├── __init__.py
│   │       ├── logging.py                  # Request logging
│   │       └── error_handling.py           # Error handling
│   │
│   ├── ui/                                 # Streamlit frontend
│   │   ├── __init__.py
│   │   ├── app.py                          # Streamlit app entry point
│   │   ├── pages/                          # Multi-page app
│   │   │   ├── 1_user_profile.py           # User registration/profile
│   │   │   ├── 2_rate_movies.py            # Movie rating interface
│   │   │   └── 3_recommendations.py        # Recommendation display
│   │   │
│   │   ├── components/                     # Reusable UI components
│   │   │   ├── __init__.py
│   │   │   ├── movie_card.py               # Movie display card
│   │   │   ├── rating_widget.py            # Star rating widget
│   │   │   └── user_form.py                # User registration form
│   │   │
│   │   └── utils/                          # UI utilities
│   │       ├── __init__.py
│   │       ├── api_client.py               # FastAPI client wrapper
│   │       └── session_state.py            # Session state management
│   │
│   ├── core/                               # Core business logic
│   │   ├── __init__.py
│   │   │
│   │   ├── inference/                      # Inference engine
│   │   │   ├── __init__.py
│   │   │   ├── engine.py                   # Main inference engine
│   │   │   ├── graph_manager.py            # Graph construction & updates
│   │   │   ├── model_loader.py             # Model loading & caching
│   │   │   └── recommender.py              # Recommendation generation
│   │   │
│   │   ├── features/                       # Feature extraction
│   │   │   ├── __init__.py
│   │   │   ├── user_features.py            # User feature extraction
│   │   │   ├── movie_features.py           # Movie feature extraction
│   │   │   └── preprocessor.py             # Feature preprocessing
│   │   │
│   │   └── training/                       # Training pipeline
│   │       ├── __init__.py
│   │       ├── train.py                    # Main training script
│   │       ├── data_loader.py              # Load data from database
│   │       └── model_versioning.py         # Model versioning & artifacts
│   │
│   ├── database/                           # Database layer
│   │   ├── __init__.py
│   │   ├── connection.py                   # Database connection
│   │   ├── models.py                       # SQLAlchemy ORM models
│   │   ├── schemas.py                      # Database schemas
│   │   ├── crud.py                         # CRUD operations
│   │   └── init_db.py                      # Database initialization
│   │
│   └── utils/                              # Shared utilities
│       ├── __init__.py
│       ├── logging_config.py               # Logging configuration
│       └── validators.py                   # Input validators
│
├── tests/                                  # Tests for app
│   ├── __init__.py
│   ├── conftest.py                         # Pytest fixtures
│   │
│   ├── api/                                # API tests
│   │   ├── __init__.py
│   │   ├── test_users.py
│   │   ├── test_movies.py
│   │   ├── test_ratings.py
│   │   └── test_recommendations.py
│   │
│   ├── core/                               # Business logic tests
│   │   ├── __init__.py
│   │   ├── test_inference.py
│   │   ├── test_graph_manager.py
│   │   └── test_recommender.py
│   │
│   ├── database/                           # Database tests
│   │   ├── __init__.py
│   │   └── test_crud.py
│   │
│   └── integration/                        # End-to-end tests
│       ├── __init__.py
│       └── test_user_flow.py
│
├── data/                                   # Data directory (mounted volume)
│   ├── recommender.db                      # SQLite database
│   └── movielens/                          # MovieLens data cache
│
├── models/                                 # Model artifacts (mounted volume)
│   ├── current/                            # Current production model
│   │   ├── graphsage_model.pth             # Model weights
│   │   ├── preprocessor.pkl                # Fitted preprocessor
│   │   └── metadata.json                   # Model metadata
│   │
│   └── versions/                           # Model version history
│       ├── v1.0.0/
│       ├── v1.0.1/
│       └── ...
│
├── logs/                                   # Application logs (mounted volume)
│   ├── api.log
│   ├── inference.log
│   └── training.log
│
├── docker/                                 # Docker configuration
│   ├── Dockerfile                          # Main Dockerfile
│   ├── docker-compose.yml                  # Docker Compose config
│   └── entrypoint.sh                       # Container entry point
│
├── scripts/                                # Utility scripts
│   ├── init_database.py                    # Initialize and populate database
│   ├── train_model.py                      # Train model from CLI
│   ├── export_data.py                      # Export data for analysis
│   └── health_check.py                     # Health check script
│
├── config/                                 # Configuration files
│   ├── api_config.yaml                     # API configuration
│   ├── model_config.yaml                   # Model hyperparameters
│   └── logging_config.yaml                 # Logging configuration
│
├── docs/                                   # Additional documentation
│   ├── API.md                              # API documentation
│   └── DEPLOYMENT.md                       # Deployment guide
│
├── requirements.txt                        # Python dependencies
├── requirements-dev.txt                    # Development dependencies
├── pytest.ini                              # Pytest configuration
├── .env.example                            # Environment variables template
├── README.md                               # App-specific README
└── ARCHITECTURE.md                         # This file

# Existing GNN module files (unchanged)
├── graph_data_loader.py
├── graphsage_model.py
├── graphsage_recommender.py
├── train_graphsage.py
├── compare_fm_graphsage.py
├── param_tuning.py
├── find_threshold.py
├── environment.yml
├── README.md
└── ...
```

### Directory Organization Rationale

**`app/` - Main Application Package**
- **`api/`**: FastAPI backend with clear separation of concerns (routers, models, middleware)
- **`ui/`**: Streamlit frontend with page-based navigation and reusable components
- **`core/`**: Business logic independent of web frameworks (testable in isolation)
- **`database/`**: Database layer with ORM models and CRUD operations
- **`utils/`**: Shared utilities used across the application

**`tests/` - Comprehensive Test Suite**
- Mirrors the `app/` structure for easy navigation
- Separate integration tests for end-to-end flows
- Uses pytest with fixtures in `conftest.py`

**`data/` - Persistent Data (Volume Mount)**
- SQLite database file
- MovieLens dataset cache
- Survives container restarts

**`models/` - Model Artifacts (Volume Mount)**
- Current production model in `current/`
- Version history in `versions/` for rollback capability
- Persists across container updates

**`logs/` - Application Logs (Volume Mount)**
- Separate logs for different components
- Accessible from host for debugging

**`docker/` - Containerization**
- Dockerfile with multi-stage builds
- Docker Compose for orchestration
- Custom entrypoint for initialization

**`scripts/` - Utility Scripts**
- Database initialization and seeding
- Model training CLI
- Maintenance and admin tasks

**`config/` - Configuration Files**
- YAML-based configuration
- Environment-specific settings
- Separation of config from code

### Key Design Decisions

1. **Separation of Concerns**: API, UI, and business logic are clearly separated
2. **Framework Independence**: Core logic doesn't depend on FastAPI or Streamlit
3. **Testability**: Each layer can be tested independently
4. **Docker-Friendly**: Volume mounts for persistence, clear entry points
5. **Scalability**: Easy to extract components into separate services later
6. **Python Best Practices**: Proper package structure with `__init__.py` files

## Deployment Architecture

### Development Deployment

**Single Docker Container Setup**:
- All components run in one Docker container
- SQLite file mounted as volume for persistence
- Streamlit on port 8501 (exposed)
- FastAPI on port 8000 (exposed)
- Manual model training via CLI inside container
- Docker Compose for easy orchestration

**Container Structure**:
```
recommender-app/
├── FastAPI backend (port 8000)
├── Streamlit frontend (port 8501)
├── GraphSAGE model & inference engine
├── SQLite database (/data/recommender.db)
└── Model artifacts (/models/)
```

**Volume Mounts**:
- `./data:/app/data` - Database persistence
- `./models:/app/models` - Model artifacts persistence
- `./logs:/app/logs` - Application logs

### Production Deployment (Future)

**Multi-Container Setup**:
- FastAPI in Docker container (scalable)
- Streamlit in separate container
- PostgreSQL instead of SQLite
- Redis for caching embeddings
- Model serving via dedicated service
- Load balancer for horizontal scaling

## Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | Streamlit 1.31+ | Interactive web UI |
| Backend API | FastAPI 0.109+ | REST API server |
| Model Framework | PyTorch 2.0+ | Deep learning framework |
| Graph Framework | PyTorch Geometric 2.3+ | GNN implementation |
| Database | SQLite 3.x | Data persistence |
| ORM | SQLAlchemy 2.0+ | Database abstraction |
| Data Processing | Pandas 2.0+, NumPy 1.24+ | Data manipulation |
| ML Preprocessing | scikit-learn 1.3+ | Feature preprocessing |
| API Documentation | OpenAPI/Swagger | Auto-generated API docs |
| Logging | Python logging + structlog | Application logging |

## Security Considerations

1. **Input Validation**: Validate all user inputs via Pydantic models
2. **SQL Injection**: Use parameterized queries (SQLAlchemy ORM)
3. **Rate Limiting**: Limit API requests per user (future)
4. **Authentication**: Add API key or JWT authentication (future)
5. **Data Privacy**: No PII collection beyond demographics
6. **CORS**: Configure appropriate CORS policies

## Scalability Considerations

### Current Scope (MVP)
- Support 1,000+ users
- Handle 100+ concurrent requests
- Single-server deployment

### Future Scalability
- **Horizontal Scaling**: Load balance multiple FastAPI instances
- **Caching**: Redis for user/movie embeddings
- **Async Processing**: Celery for background tasks
- **Database**: PostgreSQL with connection pooling
- **Model Serving**: Separate inference service (TorchServe)
- **Batch Inference**: Pre-compute recommendations nightly

## Monitoring and Observability

### Metrics to Track
- API response times (p50, p95, p99)
- Model inference latency
- Recommendation quality metrics
- User engagement (ratings added, recommendations viewed)
- Error rates by endpoint
- Database query performance

### Logging Strategy
- Structured JSON logs with request IDs
- Separate logs for API, inference, and training
- Log levels: DEBUG (dev), INFO (prod), ERROR (always)

### Health Checks
- `/api/health`: Overall system health
- Model loaded and ready
- Database connectivity
- Graph in-memory state

## Testing Strategy

1. **Unit Tests**: Test individual components (FastAPI endpoints, inference logic)
2. **Integration Tests**: Test end-to-end flows (user registration → rating → recommendation)
3. **Model Tests**: Validate model outputs and performance
4. **Load Tests**: Stress test API with concurrent users
5. **UI Tests**: Streamlit interaction tests

## Migration Path

### Phase 1 (Current): Offline Demo
- Train model on MovieLens 100K
- Static recommendations
- No user management

### Phase 2 (This Project): Interactive System
- User registration and management
- Dynamic recommendations
- Rating collection and updates

### Phase 3 (Future): Production-Ready
- Authentication and authorization
- Horizontal scaling
- Advanced caching
- A/B testing framework
- Real-time model updates

## Open Questions and Future Enhancements

1. **Cold-Start Strategy**: Should we show popular movies or purely demographic-based?
2. **Rating Scale**: Keep 1-5 integer ratings or allow 0.5 increments?
3. **Recommendation Diversity**: Add diversity/exploration to avoid filter bubbles?
4. **Explanation**: Show why movies were recommended?
5. **Batch Recommendations**: Pre-compute for faster response times?
6. **Model Retraining**: Incremental training vs. full retraining?
7. **Feedback Loop**: Collect implicit feedback (clicks, watch time)?

## Success Criteria

### Functional Requirements
- ✅ Create new user with demographics
- ✅ Generate cold-start recommendations
- ✅ Add ratings for movies
- ✅ Generate updated recommendations with ratings
- ✅ View rating history

### Performance Requirements
- API response time < 500ms (p95)
- Recommendation generation < 1s
- Support 50 concurrent users

### Quality Requirements
- Test coverage > 80%
- Zero data loss
- Graceful error handling

## Conclusion

This architecture provides a solid foundation for a user-facing GraphSAGE recommender system. It balances simplicity (for development and demonstration) with extensibility (for future enhancements). The modular design allows each component to be developed, tested, and deployed independently.

**Next Steps**:
1. Review and approve architecture
2. Create detailed technical specifications
3. Begin implementation with data layer and training pipeline
4. Develop API endpoints
5. Build Streamlit UI
6. Integration testing
7. Deployment and documentation

---

**Document Version**: 1.0  
**Last Updated**: 2026-01-30  
**Authors**: AI Assistant  
**Status**: Draft - Awaiting Review
