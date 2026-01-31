"""
Recommendation generation using GraphSAGE embeddings.

Handles generating user and item embeddings, computing scores,
filtering, and returning top-N recommendations.
"""

import torch
import logging
from typing import List, Dict, Any, Optional, Set
from sqlalchemy.orm import Session

from poc.graphsage_model import GraphSAGERecommender
from app.database import crud
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class Recommender:
    """
    Generates recommendations using GraphSAGE model.
    
    This class handles:
    - Computing user and item embeddings via model forward pass
    - Calculating recommendation scores for all items
    - Filtering out poorly-rated items (rating ≤2)
    - Returning top-N recommendations with scores
    - Handling both cold-start and warm-start users
    """
    
    def __init__(
        self,
        model: GraphSAGERecommender,
        device: str = 'cpu'
    ):
        """
        Initialize recommender.
        
        Args:
            model: Trained GraphSAGE model
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        logger.info(f"Recommender initialized on device: {device}")
    
    def generate_embeddings(
        self,
        graph_data: Data
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate user and item embeddings from graph.
        
        Args:
            graph_data: PyTorch Geometric Data object
            
        Returns:
            Tuple of (user_embeddings, item_embeddings)
            - user_embeddings: Tensor of shape (num_users, hidden_dim)
            - item_embeddings: Tensor of shape (num_items, hidden_dim)
        """
        logger.debug("Generating embeddings...")
        
        # Move graph to device
        graph_data = graph_data.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            user_emb, item_emb = self.model(graph_data)
        
        logger.debug(f"  User embeddings: {user_emb.shape}")
        logger.debug(f"  Item embeddings: {item_emb.shape}")
        
        return user_emb, item_emb
    
    def compute_scores(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        user_idx: int
    ) -> torch.Tensor:
        """
        Compute recommendation scores for all items for a given user.
        
        Args:
            user_emb: User embeddings (num_users, hidden_dim)
            item_emb: Item embeddings (num_items, hidden_dim)
            user_idx: Index of the user to generate recommendations for
            
        Returns:
            Tensor of scores for all items (num_items,)
        """
        # Get user embedding
        user_embedding = user_emb[user_idx:user_idx+1]  # (1, hidden_dim)
        
        # Compute dot product with all item embeddings
        scores = torch.matmul(user_embedding, item_emb.t()).squeeze(0)  # (num_items,)
        
        return scores
    
    def get_recommendations(
        self,
        session: Session,
        user_id: int,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        user_idx: int,
        item_id_to_idx: Dict[int, int],
        idx_to_item_id: Dict[int, int],
        n: int = 10,
        exclude_low_rated: bool = True,
        exclude_already_rated: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate top-N recommendations for a user.
        
        Args:
            session: Database session
            user_id: User ID
            user_emb: User embeddings
            item_emb: Item embeddings
            user_idx: User node index
            item_id_to_idx: Mapping from movie_id to item index
            idx_to_item_id: Mapping from item index to movie_id
            n: Number of recommendations to return (default: 10)
            exclude_low_rated: Filter out movies rated ≤2 (default: True)
            exclude_already_rated: Exclude movies user has already rated (default: True)
            
        Returns:
            List of recommendation dictionaries with keys:
            - movie_id: Movie ID
            - title: Movie title
            - score: Recommendation score
            - rank: Rank in recommendations (1-indexed)
        """
        logger.info(f"Generating recommendations for user {user_id} (n={n})")
        
        # Compute scores for all items
        scores = self.compute_scores(user_emb, item_emb, user_idx)
        
        # Get movies to exclude
        excluded_movie_ids = self._get_excluded_movies(
            session,
            user_id,
            exclude_low_rated,
            exclude_already_rated
        )
        
        logger.debug(f"  Excluding {len(excluded_movie_ids)} movies")
        
        # Filter out excluded movies
        filtered_scores = []
        for item_idx, score in enumerate(scores.tolist()):
            movie_id = idx_to_item_id[item_idx]
            
            if movie_id not in excluded_movie_ids:
                filtered_scores.append((movie_id, score, item_idx))
        
        # Sort by score descending
        filtered_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N
        top_n = filtered_scores[:n]
        
        logger.debug(f"  Returning {len(top_n)} recommendations")
        
        # Build recommendation list with movie details
        recommendations = []
        for rank, (movie_id, score, item_idx) in enumerate(top_n, 1):
            movie = crud.get_movie(session, movie_id)
            
            if movie:
                recommendations.append({
                    'movie_id': movie_id,
                    'title': movie.title,
                    'release_year': movie.release_year,
                    'genres': movie.genres,
                    'score': float(score),
                    'rank': rank
                })
        
        logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
        
        return recommendations
    
    def _get_excluded_movies(
        self,
        session: Session,
        user_id: int,
        exclude_low_rated: bool,
        exclude_already_rated: bool
    ) -> Set[int]:
        """
        Get set of movie IDs to exclude from recommendations.
        
        Args:
            session: Database session
            user_id: User ID
            exclude_low_rated: Exclude movies rated ≤2
            exclude_already_rated: Exclude all movies already rated
            
        Returns:
            Set of movie IDs to exclude
        """
        excluded = set()
        
        # Get user's ratings
        user_ratings = crud.get_ratings_by_user(session, user_id, limit=10000)
        
        for rating in user_ratings:
            if exclude_already_rated:
                # Exclude all rated movies
                excluded.add(rating.movie_id)
            elif exclude_low_rated and rating.rating <= 2:
                # Only exclude poorly-rated movies
                excluded.add(rating.movie_id)
        
        return excluded
    
    def predict_rating(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        user_idx: int,
        item_idx: int
    ) -> float:
        """
        Predict rating for a specific user-item pair.
        
        Args:
            user_emb: User embeddings
            item_emb: Item embeddings
            user_idx: User node index
            item_idx: Item node index
            
        Returns:
            Predicted rating value
        """
        with torch.no_grad():
            score = self.model.predict(user_emb, item_emb, user_idx, item_idx, use_rating_head=True)
        
        return float(score.item())
    
    def batch_predict_ratings(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        user_indices: List[int],
        item_indices: List[int]
    ) -> List[float]:
        """
        Predict ratings for multiple user-item pairs.
        
        Args:
            user_emb: User embeddings
            item_emb: Item embeddings
            user_indices: List of user node indices
            item_indices: List of item node indices (must match length of user_indices)
            
        Returns:
            List of predicted ratings
        """
        if len(user_indices) != len(item_indices):
            raise ValueError("user_indices and item_indices must have same length")
        
        predictions = []
        with torch.no_grad():
            for user_idx, item_idx in zip(user_indices, item_indices):
                score = self.model.predict(user_emb, item_emb, user_idx, item_idx, use_rating_head=True)
                predictions.append(float(score.item()))
        
        return predictions


if __name__ == "__main__":
    # Test recommender
    from app.utils.logging_config import configure_inference_logging
    from app.database.connection import get_session
    from app.core.inference.model_loader import ModelLoader
    from app.core.features.preprocessor import FeatureProcessor
    from app.core.inference.graph_manager import GraphManager
    
    configure_inference_logging(debug=True)
    
    print("Testing Recommender...")
    
    try:
        # Load model
        loader = ModelLoader(model_dir="models/current")
        model, preprocessor, metadata = loader.load_model()
        
        # Build graph
        feature_processor = FeatureProcessor(preprocessor)
        manager = GraphManager(feature_processor)
        
        session = get_session()
        graph_data = manager.build_graph_from_database(session)
        
        # Create recommender
        recommender = Recommender(model)
        
        # Generate embeddings
        user_emb, item_emb = recommender.generate_embeddings(graph_data)
        
        # Get recommendations for first user
        user_id = 1
        user_idx = manager.get_user_node_index(user_id)
        
        if user_idx is not None:
            recommendations = recommender.get_recommendations(
                session,
                user_id,
                user_emb,
                item_emb,
                user_idx,
                manager.item_id_to_idx,
                manager.idx_to_item_id,
                n=5
            )
            
            print(f"\nTop 5 recommendations for user {user_id}:")
            for rec in recommendations:
                print(f"  {rec['rank']}. {rec['title']} (score: {rec['score']:.3f})")
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nThis is expected if training hasn't been completed.")
