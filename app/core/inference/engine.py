"""
Main inference engine orchestrator.

Combines model loading, graph management, and recommendation generation
into a high-level interface for the application.
"""

import torch
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from app.core.inference.model_loader import ModelLoader
from app.core.inference.graph_manager import GraphManager
from app.core.inference.recommender import Recommender
from app.core.features.preprocessor import FeatureProcessor
from app.database import crud

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    High-level inference engine for GraphSAGE recommendations.
    
    This class orchestrates:
    - Model loading and initialization
    - Graph construction and management
    - User and item embedding generation
    - Recommendation generation
    - Cache management
    
    Usage:
        engine = InferenceEngine()
        engine.load_model()
        engine.initialize_graph(session)
        recommendations = engine.get_recommendations(session, user_id=1, n=10)
    """
    
    def __init__(
        self,
        model_dir: str = "models/current",
        device: Optional[str] = None,
        cache_embeddings: bool = True
    ):
        """
        Initialize inference engine.
        
        Args:
            model_dir: Directory containing model artifacts
            device: Device for inference ('cpu' or 'cuda'). Auto-detects if None.
            cache_embeddings: Enable embedding caching (default: True)
        """
        self.model_dir = model_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_embeddings = cache_embeddings
        
        # Components (initialized in load_model and initialize_graph)
        self.model_loader: Optional[ModelLoader] = None
        self.feature_processor: Optional[FeatureProcessor] = None
        self.graph_manager: Optional[GraphManager] = None
        self.recommender: Optional[Recommender] = None
        
        # Model artifacts
        self.model = None
        self.preprocessor = None
        self.metadata = None
        
        # Embedding cache
        self._user_embeddings: Optional[torch.Tensor] = None
        self._item_embeddings: Optional[torch.Tensor] = None
        self._cache_valid = False
        
        logger.info(f"InferenceEngine initialized (device: {self.device}, caching: {cache_embeddings})")
    
    def load_model(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load trained model and preprocessor.
        
        Args:
            force_reload: Force reload from disk (default: False)
            
        Returns:
            Dictionary with model information
        """
        logger.info("Loading model...")
        
        # Initialize model loader
        self.model_loader = ModelLoader(model_dir=self.model_dir, device=self.device)
        
        # Load model artifacts
        self.model, self.preprocessor, self.metadata, rating_scaler = self.model_loader.load_model(force_reload)
        
        # Initialize feature processor
        self.feature_processor = FeatureProcessor(self.preprocessor)
        
        # Initialize recommender
        self.recommender = Recommender(self.model, rating_scaler, device=self.device)
        
        model_info = self.model_loader.get_model_info()
        logger.info(f"Model loaded: version {model_info.get('version', 'unknown')}")
        
        return model_info
    
    def initialize_graph(self, session: Session) -> Dict[str, int]:
        """
        Build initial graph from database.
        
        Args:
            session: Database session
            
        Returns:
            Dictionary with graph statistics
        """
        if self.feature_processor is None:
            raise RuntimeError("Model must be loaded before initializing graph. Call load_model() first.")
        
        logger.info("Initializing graph from database...")
        
        # Initialize graph manager
        self.graph_manager = GraphManager(self.feature_processor)
        
        # Build graph
        graph_data = self.graph_manager.build_graph_from_database(session)
        
        # Invalidate cache (new graph)
        self._invalidate_cache()
        
        stats = self.graph_manager.get_graph_stats()
        logger.info(f"Graph initialized: {stats}")
        
        return stats
    
    def add_user(self, session: Session, user_id: int) -> int:
        """
        Add new user node to graph (cold-start scenario).
        
        Args:
            session: Database session
            user_id: User ID to add
            
        Returns:
            Node index for the new user
        """
        if self.graph_manager is None:
            raise RuntimeError("Graph must be initialized first. Call initialize_graph().")
        
        logger.info(f"Adding user {user_id} to graph")
        
        node_idx = self.graph_manager.add_user(session, user_id)
        
        # Invalidate cache (graph changed)
        self._invalidate_cache()
        
        return node_idx
    
    def add_rating(
        self,
        session: Session,
        user_id: int,
        movie_id: int,
        rating: float
    ) -> bool:
        """
        Add or update rating edge in graph.
        
        Args:
            session: Database session
            user_id: User ID
            movie_id: Movie ID
            rating: Rating value
            
        Returns:
            True if edge was added, False if it already existed
        """
        if self.graph_manager is None:
            raise RuntimeError("Graph must be initialized first. Call initialize_graph().")
        
        logger.info(f"Adding rating: user {user_id} -> movie {movie_id} (rating={rating})")
        
        # Check if user exists in graph
        if user_id not in self.graph_manager.user_id_to_idx:
            logger.info(f"User {user_id} not in graph, adding...")
            self.add_user(session, user_id)
        
        # Add edge
        edge_added = self.graph_manager.add_rating(user_id, movie_id, rating)
        
        if edge_added:
            # Invalidate cache for this user
            self._invalidate_user_cache(user_id)
        
        return edge_added
    
    def get_recommendations(
        self,
        session: Session,
        user_id: int,
        n: int = 10,
        exclude_low_rated: bool = True,
        exclude_already_rated: bool = True,
        force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a user.
        
        Args:
            session: Database session
            user_id: User ID
            n: Number of recommendations (default: 10)
            exclude_low_rated: Filter out movies rated ≤2 (default: True)
            exclude_already_rated: Exclude movies already rated (default: True)
            force_refresh: Force recompute embeddings (default: False)
            
        Returns:
            List of recommendation dictionaries
        """
        if self.graph_manager is None or self.recommender is None:
            raise RuntimeError("Engine not initialized. Call load_model() and initialize_graph().")
        
        logger.info(f"Getting recommendations for user {user_id} (n={n})")
        
        # Check if user exists
        user_idx = self.graph_manager.get_user_node_index(user_id)
        if user_idx is None:
            # User not in graph - add as cold-start user
            logger.info(f"User {user_id} not in graph, adding as cold-start user")
            user_idx = self.add_user(session, user_id)
        
        # Get or compute embeddings
        user_emb, item_emb = self._get_embeddings(force_refresh)
        
        # Generate recommendations
        recommendations = self.recommender.get_recommendations(
            session=session,
            user_id=user_id,
            user_emb=user_emb,
            item_emb=item_emb,
            user_idx=user_idx,
            item_id_to_idx=self.graph_manager.item_id_to_idx,
            idx_to_item_id=self.graph_manager.idx_to_item_id,
            n=n,
            exclude_low_rated=exclude_low_rated,
            exclude_already_rated=exclude_already_rated
        )
        
        return recommendations
    
    def refresh_recommendations(
        self,
        session: Session,
        user_id: int,
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Refresh recommendations by invalidating cache and regenerating.
        
        Args:
            session: Database session
            user_id: User ID
            n: Number of recommendations (default: 10)
            
        Returns:
            List of recommendation dictionaries
        """
        logger.info(f"Refreshing recommendations for user {user_id}")
        
        # Invalidate cache
        self._invalidate_cache()
        
        # Generate new recommendations
        return self.get_recommendations(
            session,
            user_id,
            n=n,
            force_refresh=True
        )
    
    def _get_embeddings(
        self,
        force_refresh: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get user and item embeddings (from cache or compute).
        
        Args:
            force_refresh: Force recompute embeddings
            
        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        # Check cache
        if self.cache_embeddings and not force_refresh and self._cache_valid:
            logger.debug("Using cached embeddings")
            return self._user_embeddings, self._item_embeddings
        
        # Compute embeddings
        logger.debug("Computing embeddings...")
        graph_data = self.graph_manager.graph_data
        
        user_emb, item_emb = self.recommender.generate_embeddings(graph_data)
        
        # Cache if enabled
        if self.cache_embeddings:
            self._user_embeddings = user_emb
            self._item_embeddings = item_emb
            self._cache_valid = True
            logger.debug("Embeddings cached")
        
        return user_emb, item_emb
    
    def _invalidate_cache(self):
        """Invalidate embedding cache."""
        self._cache_valid = False
        logger.debug("Embedding cache invalidated")
    
    def _invalidate_user_cache(self, user_id: int):
        """
        Invalidate cache for a specific user (for now, invalidates entire cache).
        
        Args:
            user_id: User ID
        """
        # For simplicity, invalidate entire cache
        # Could be optimized to only recompute affected embeddings
        self._invalidate_cache()
        logger.debug(f"Cache invalidated for user {user_id}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get engine status.
        
        Returns:
            Dictionary with status information
        """
        model_loaded = self.model is not None
        graph_initialized = self.graph_manager is not None
        
        status = {
            'model_loaded': model_loaded,
            'graph_initialized': graph_initialized,
            'device': str(self.device),
            'cache_enabled': self.cache_embeddings,
            'cache_valid': self._cache_valid
        }
        
        if model_loaded:
            status['model_info'] = self.model_loader.get_model_info()
        
        if graph_initialized:
            status['graph_stats'] = self.graph_manager.get_graph_stats()
        
        return status
    
    def predict_rating(
        self,
        user_id: int,
        movie_id: int
    ) -> Optional[float]:
        """
        Predict rating for a user-movie pair.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating or None if user/movie not in graph
        """
        if self.graph_manager is None or self.recommender is None:
            raise RuntimeError("Engine not initialized.")
        
        user_idx = self.graph_manager.get_user_node_index(user_id)
        item_idx = self.graph_manager.item_id_to_idx.get(movie_id)
        
        if user_idx is None or item_idx is None:
            return None
        
        # Get embeddings
        user_emb, item_emb = self._get_embeddings()
        
        # Predict rating
        rating = self.recommender.predict_rating(user_emb, item_emb, user_idx, item_idx)
        
        return rating


if __name__ == "__main__":
    # Test inference engine
    from app.utils.logging_config import configure_inference_logging
    from app.database.connection import get_session
    
    configure_inference_logging(debug=True)
    
    print("Testing InferenceEngine...")
    
    try:
        # Initialize engine
        engine = InferenceEngine(model_dir="models/current")
        
        # Load model
        model_info = engine.load_model()
        print(f"\nModel loaded: {model_info}")
        
        # Initialize graph
        session = get_session()
        graph_stats = engine.initialize_graph(session)
        print(f"\nGraph initialized: {graph_stats}")
        
        # Get recommendations for a user
        user_id = 1
        recommendations = engine.get_recommendations(session, user_id, n=5)
        
        print(f"\nTop 5 recommendations for user {user_id}:")
        for rec in recommendations:
            print(f"  {rec['rank']}. {rec['title']} (score: {rec['score']:.3f})")
        
        # Get status
        status = engine.get_status()
        print(f"\nEngine status: {status}")
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nThis is expected if training hasn't been completed.")
        print("Creating mock model for testing...")
