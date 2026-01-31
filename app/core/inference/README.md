# GraphSAGE Inference Engine

Complete inference engine for generating real-time movie recommendations using trained GraphSAGE models.

## Quick Start

### Initialize Engine

```python
from app.core.inference.engine import InferenceEngine
from app.database.connection import get_db_manager

# Create engine
engine = InferenceEngine(
    model_dir="models/current",  # Directory with model artifacts
    cache_embeddings=True         # Enable embedding caching
)

# Load model
model_info = engine.load_model()
print(f"Loaded model version: {model_info['version']}")

# Initialize graph from database
db_manager = get_db_manager()
session = db_manager.get_session()
graph_stats = engine.initialize_graph(session)
print(f"Graph: {graph_stats['num_users']} users, {graph_stats['num_items']} items")
```

### Get Recommendations

```python
# Get recommendations for existing user
recommendations = engine.get_recommendations(
    session,
    user_id=1,
    n=10,                       # Number of recommendations
    exclude_low_rated=True,     # Exclude movies rated ≤2
    exclude_already_rated=True  # Exclude rated movies
)

# Display recommendations
for rec in recommendations:
    print(f"{rec['rank']}. {rec['title']} - Score: {rec['score']:.3f}")
```

### Handle New Users (Cold-Start)

```python
from app.database import crud

# Create new user
new_user = crud.create_user(
    session,
    age=25,
    gender='F',
    occupation='student',
    zip_code='12345'
)

# Get cold-start recommendations (based on demographics only)
recommendations = engine.get_recommendations(
    session,
    new_user.user_id,
    n=10
)
```

### Add Ratings

```python
# User rates a movie
crud.create_rating(session, user_id=1, movie_id=50, rating=5.0)

# Update graph
engine.add_rating(session, user_id=1, movie_id=50, rating=5.0)

# Get updated recommendations
new_recs = engine.refresh_recommendations(session, user_id=1, n=10)
```

## Architecture

### Components

```
InferenceEngine (engine.py)
    ├── ModelLoader (model_loader.py)
    │   └── Loads trained model, preprocessor, metadata
    ├── GraphManager (graph_manager.py)
    │   └── Builds and updates user-item bipartite graph
    ├── Recommender (recommender.py)
    │   └── Generates embeddings and recommendations
    └── FeatureProcessor (preprocessor.py)
        └── Extracts and transforms user/movie features
```

### Data Flow

1. **Model Loading**: Load GraphSAGE model and preprocessor
2. **Graph Building**: Construct bipartite graph from database
3. **Embedding Generation**: Run model forward pass
4. **Score Computation**: Calculate recommendation scores (dot product)
5. **Filtering**: Remove low-rated and already-rated movies
6. **Ranking**: Sort by score and return top-N

## API Reference

### InferenceEngine

#### `__init__(model_dir, device, cache_embeddings)`
Initialize inference engine.

**Args:**
- `model_dir` (str): Directory containing model artifacts (default: "models/current")
- `device` (str): 'cpu' or 'cuda' (default: auto-detect)
- `cache_embeddings` (bool): Enable embedding caching (default: True)

#### `load_model(force_reload=False)`
Load trained model and preprocessor.

**Returns:** `Dict` with model information

#### `initialize_graph(session)`
Build initial graph from database.

**Args:**
- `session`: SQLAlchemy database session

**Returns:** `Dict` with graph statistics

#### `add_user(session, user_id)`
Add new user node to graph (cold-start).

**Args:**
- `session`: Database session
- `user_id` (int): User ID

**Returns:** `int` - Node index

#### `add_rating(session, user_id, movie_id, rating)`
Update graph with new rating.

**Args:**
- `session`: Database session
- `user_id` (int): User ID
- `movie_id` (int): Movie ID
- `rating` (float): Rating value (1.0-5.0)

**Returns:** `bool` - True if edge was added

#### `get_recommendations(session, user_id, n=10, exclude_low_rated=True, exclude_already_rated=True, force_refresh=False)`
Generate recommendations for user.

**Args:**
- `session`: Database session
- `user_id` (int): User ID
- `n` (int): Number of recommendations (default: 10)
- `exclude_low_rated` (bool): Filter movies rated ≤2 (default: True)
- `exclude_already_rated` (bool): Filter rated movies (default: True)
- `force_refresh` (bool): Force recompute embeddings (default: False)

**Returns:** `List[Dict]` - Recommendations with keys:
- `movie_id` (int): Movie ID
- `title` (str): Movie title
- `release_year` (int): Release year
- `genres` (str): JSON array of genres
- `score` (float): Recommendation score
- `rank` (int): Rank (1-indexed)

#### `refresh_recommendations(session, user_id, n=10)`
Invalidate cache and regenerate recommendations.

**Args:**
- `session`: Database session
- `user_id` (int): User ID
- `n` (int): Number of recommendations

**Returns:** `List[Dict]` - Recommendations

#### `get_status()`
Get engine status.

**Returns:** `Dict` with status information:
- `model_loaded` (bool)
- `graph_initialized` (bool)
- `device` (str)
- `cache_enabled` (bool)
- `cache_valid` (bool)
- `model_info` (Dict)
- `graph_stats` (Dict)

#### `predict_rating(user_id, movie_id)`
Predict rating for user-movie pair.

**Args:**
- `user_id` (int): User ID
- `movie_id` (int): Movie ID

**Returns:** `float` - Predicted rating (1.0-5.0) or None

## Model Requirements

The inference engine expects the following artifacts in `model_dir`:

### `graphsage_model.pth`
PyTorch state dict with trained GraphSAGE weights.

**Expected Architecture:**
- 3 GraphSAGE layers
- 64-dimensional hidden embeddings
- Max pooling aggregator
- Dropout: 0.1

### `preprocessor.pkl`
Pickled `FeaturePreprocessor` fitted on training data.

**Contains:**
- Age scaler (StandardScaler)
- Gender encoder (LabelEncoder)
- Occupation encoder (LabelEncoder)
- User/item ID mappings
- Feature offsets

### `metadata.json`
Model configuration and metrics.

**Required Fields:**
```json
{
  "model_version": "1.0.0",
  "num_users_trained": 943,
  "num_movies": 1682,
  "hidden_dim": 64,
  "num_layers": 3,
  "aggregator": "max",
  "hyperparameters": {
    "dropout": 0.1
  }
}
```

## Performance

| Operation | Expected Time |
|-----------|---------------|
| Model loading | < 2s |
| Graph initialization (100K ratings) | < 5s |
| Embedding generation | < 1s |
| Recommendation generation | < 500ms |
| Add user | < 100ms |
| Add rating | < 50ms |

## Testing

### Run Unit Tests
```bash
pytest tests/core/test_inference.py -v
```

### Run Integration Tests
```bash
pytest tests/integration/test_inference_flow.py -v -s
```

### Test Components Individually

```python
# Test model loader
from app.core.inference.model_loader import ModelLoader
loader = ModelLoader()
model, preprocessor, metadata = loader.load_model()

# Test graph manager
from app.core.inference.graph_manager import GraphManager
from app.core.features.preprocessor import FeatureProcessor

feature_processor = FeatureProcessor(preprocessor)
manager = GraphManager(feature_processor)
graph_data = manager.build_graph_from_database(session)

# Test recommender
from app.core.inference.recommender import Recommender
recommender = Recommender(model)
user_emb, item_emb = recommender.generate_embeddings(graph_data)
```

## Troubleshooting

### Model Not Found
**Error:** `FileNotFoundError: Model weights file not found`

**Solution:** Ensure model artifacts exist in `models/current/`:
```bash
ls models/current/
# Should show: graphsage_model.pth, preprocessor.pkl, metadata.json
```

If missing, create mock model for testing:
```bash
python scripts/create_mock_model.py
```

### PyTorch Geometric Warnings
**Warning:** `Could not load torch_scatter / torch_sparse`

**Impact:** None (warnings only, functionality works)

**Cause:** Windows compatibility issues with PyTorch Geometric C++ extensions

**Solution:** Can be ignored or use Linux environment

### Empty Recommendations
**Issue:** `get_recommendations()` returns empty list

**Possible Causes:**
1. All movies already rated by user → Set `exclude_already_rated=False`
2. All movies rated ≤2 → Set `exclude_low_rated=False`
3. User not in graph → Automatically added as cold-start user

### Cache Issues
**Issue:** Recommendations not updating after adding rating

**Solution:** Use `refresh_recommendations()` to invalidate cache:
```python
engine.refresh_recommendations(session, user_id, n=10)
```

Or disable caching:
```python
engine = InferenceEngine(cache_embeddings=False)
```

## Logging

Configure logging for debugging:

```python
from app.utils.logging_config import configure_inference_logging

# Enable debug logging
configure_inference_logging(debug=True)

# Logs written to: logs/inference.log
```

## Advanced Usage

### Batch Recommendations
```python
# Get recommendations for multiple users
user_ids = [1, 2, 3, 4, 5]

for user_id in user_ids:
    recs = engine.get_recommendations(session, user_id, n=5)
    print(f"User {user_id}: {len(recs)} recommendations")
```

### Custom Filtering
```python
# Get all recommendations (no filtering)
all_recs = engine.get_recommendations(
    session,
    user_id,
    n=50,
    exclude_low_rated=False,
    exclude_already_rated=False
)

# Custom post-processing
genre_filtered = [r for r in all_recs if 'Action' in r['genres']]
```

### Rating Prediction
```python
# Predict rating for specific movie
predicted_rating = engine.predict_rating(user_id=1, movie_id=100)
print(f"Predicted rating: {predicted_rating:.2f}")
```

## Dependencies

- `torch>=2.0.0` - PyTorch
- `torch-geometric>=2.3.0` - PyTorch Geometric
- `sqlalchemy>=2.0.0` - Database ORM
- `pandas>=2.0.0` - Data processing
- `numpy>=1.24.0` - Numerical operations
- `scikit-learn>=1.3.0` - Feature preprocessing

## License

Part of the GraphSAGE Recommender System MVP.

## Support

For issues or questions, refer to:
- Main documentation: `ARCHITECTURE_MVP.md`
- Phase 2 completion report: `PHASE2_COMPLETE.md`
- Test examples: `tests/integration/test_inference_flow.py`
