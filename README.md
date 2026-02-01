# GraphSAGE Recommender System

User-facing movie recommender using GraphSAGE (GNN), with cold-start and warm-start support.

## Quick Start

### Environment

```bash
conda create -n recommender python=3.10 -y
conda activate recommender
pip install -r requirements.txt
# PyTorch with CUDA (optional): pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# Then: pip install torch-geometric
```

### Data & Model

```bash
# Import MovieLens 100K (movies, users, ratings)
python scripts/init_database.py --reset

# Train initial model (saves to models/current/)
python scripts/train_model.py
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Run API

```bash
# From project root
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

- API: http://localhost:8000  
- Docs: http://localhost:8000/docs  
- Health: http://localhost:8000/api/health  

### Run UI (Streamlit)

```bash
# Start API first (in one terminal)
uvicorn app.api.main:app --host 0.0.0.0 --port 8000

# Start Streamlit (in another terminal)
streamlit run app/ui/app.py --server.port 8501
```

- UI: http://localhost:8501  
- Create profile, get recommendations, rate movies inline  

### Docker

```bash
# Build and run API (mounts data/, models/, logs/)
docker-compose -f docker/docker-compose.yml up --build
```

- API: http://localhost:8000  

## Environment Variables

See `.env.example`:

- `DATABASE_URL` – SQLite path (e.g. `sqlite:///data/recommender.db`)
- `MODEL_PATH` – Model directory (e.g. `models/current`)
- `API_HOST`, `API_PORT` – API binding
- `LOG_LEVEL` – Logging level  

## Project Structure

- `app/api/` – FastAPI app, routers, Pydantic models
- `app/core/inference/` – Inference engine, graph manager, recommender
- `app/core/training/` – Training pipeline
- `app/database/` – SQLAlchemy models, CRUD, connection
- `poc/` – POC GraphSAGE model and training
- `config/` – YAML config (api, model)
- `docker/` – Dockerfile and docker-compose
- `scripts/` – init_database, train_model, verify_database  

## API Endpoints (MVP)

- `POST /api/users` – Create user (demographics)
- `GET /api/users/{user_id}` – Get user profile
- `GET /api/users/{user_id}/ratings` – User rating history
- `POST /api/ratings` – Add rating
- `GET /api/ratings/stats` – Global rating statistics
- `GET /api/recommendations/{user_id}?n=10&exclude_low_rated=true` – Get recommendations
- `POST /api/recommendations/{user_id}/refresh` – Refresh recommendations
- `GET /api/movies`, `GET /api/movies/{movie_id}`, `GET /api/movies/search` – Movies
- `GET /api/health` – Health check
- `GET /api/model/info` – Model metadata  

## Training Options

```bash
python scripts/train_model.py --epochs 30          # Custom epochs
python scripts/train_model.py --hidden-dim 128    # Larger model
python scripts/train_model.py --device cpu        # CPU training
python scripts/train_model.py --quiet             # Suppress output
```

Artifacts: `models/current/graphsage_model.pth`, `preprocessor.pkl`, `metadata.json`

## Documentation

- **ARCHITECTURE_MVP.md** – Full specification (schema, endpoints, data flows)
- **DATABASE_GUIDE.md** – Database CRUD reference, scripts, troubleshooting
