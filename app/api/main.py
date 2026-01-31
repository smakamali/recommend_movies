"""
FastAPI application entry point for GraphSAGE Recommender API.
"""
# OpenMP workaround: avoid libiomp5md.dll duplicate init (PyTorch/NumPy conflict)
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.config import get_api_host, get_api_port
from app.api.routers import users, movies, ratings, recommendations, system

app = FastAPI(
    title="GraphSAGE Recommender API",
    description="REST API for personalized movie recommendations using GraphSAGE",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users.router)
app.include_router(movies.router)
app.include_router(ratings.router)
app.include_router(recommendations.router)
app.include_router(system.router)


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "GraphSAGE Recommender API",
        "docs": "/docs",
        "health": "/api/health",
    }
