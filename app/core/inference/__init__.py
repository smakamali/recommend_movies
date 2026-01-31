"""
Inference engine for GraphSAGE-based recommendations.

This package contains:
- Model loading and caching
- Graph construction and management
- Recommendation generation
- Main inference engine orchestrator
"""

from app.core.inference.engine import InferenceEngine
from app.core.inference.model_loader import ModelLoader
from app.core.inference.graph_manager import GraphManager
from app.core.inference.recommender import Recommender

__all__ = ['InferenceEngine', 'ModelLoader', 'GraphManager', 'Recommender']
