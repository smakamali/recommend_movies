# GraphSAGE Recommender - API & UI
FROM python:3.10-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python deps (PyTorch CPU for smaller image; override with cu124 if needed)
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn[standard] streamlit pyyaml

# Copy application
COPY app/ ./app/
COPY poc/ ./poc/
COPY config/ ./config/

# Create dirs for volumes
RUN mkdir -p /app/data /app/models /app/logs

ENV PYTHONPATH=/app
ENV DATABASE_URL=sqlite:///data/recommender.db
ENV MODEL_PATH=/app/models/current
ENV LOG_LEVEL=INFO

EXPOSE 8000 8501

# Default: run API only (use docker-compose or override for Streamlit)
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
