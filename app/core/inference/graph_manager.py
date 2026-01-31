"""
Graph manager for constructing and updating the user-item bipartite graph.

Handles graph construction from database, adding new users, updating edges
when ratings are added, and maintaining node index mappings.
"""

import torch
import logging
from typing import Dict, List, Tuple, Optional, Set
from sqlalchemy.orm import Session
from torch_geometric.data import Data

from app.database import crud
from app.database.models import User, Movie, Rating
from app.core.features.preprocessor import FeatureProcessor

logger = logging.getLogger(__name__)


class GraphManager:
    """
    Manages the user-item bipartite graph for GraphSAGE inference.
    
    This class handles:
    - Building initial graph from database ratings
    - Adding new user nodes dynamically (for cold-start)
    - Adding edges when users rate movies
    - Maintaining user_id ↔ node_index mappings
    - Constructing PyTorch Geometric Data objects
    
    Graph Structure:
    - Nodes: [user_0, ..., user_N, item_0, ..., item_M]
    - Node indices: Users (0 to num_users-1), Items (num_users to num_users+num_items-1)
    - Edges: Bidirectional user-item interactions
    - Node features: User and item feature vectors
    """
    
    def __init__(self, feature_processor: FeatureProcessor):
        """
        Initialize graph manager.
        
        Args:
            feature_processor: FeatureProcessor for extracting features
        """
        self.feature_processor = feature_processor
        
        # Mappings
        self.user_id_to_idx: Dict[int, int] = {}
        self.item_id_to_idx: Dict[int, int] = {}
        self.idx_to_user_id: Dict[int, int] = {}
        self.idx_to_item_id: Dict[int, int] = {}
        
        # Graph data
        self.graph_data: Optional[Data] = None
        self.num_users = 0
        self.num_items = 0
        
        # Track edges (set of (user_idx, item_idx) tuples)
        self.edges: Set[Tuple[int, int]] = set()
        
        logger.info("GraphManager initialized")
    
    def build_graph_from_database(self, session: Session) -> Data:
        """
        Build initial graph from all ratings in database.
        
        Args:
            session: Database session
            
        Returns:
            PyTorch Geometric Data object
        """
        logger.info("Building graph from database...")
        
        # Get all users, movies, and ratings
        all_users = session.query(User).all()
        all_movies = session.query(Movie).all()
        all_ratings = session.query(Rating).all()
        
        logger.info(f"  Found {len(all_users)} users, {len(all_movies)} movies, {len(all_ratings)} ratings")
        
        # Build ID mappings
        self._build_id_mappings(all_users, all_movies)
        
        # Build edge list from ratings
        self._build_edges_from_ratings(all_ratings)
        
        # Build feature matrices
        user_features = self._build_user_feature_matrix(all_users)
        item_features = self._build_item_feature_matrix(all_movies)
        
        # Create graph data
        self.graph_data = self._construct_graph_data(user_features, item_features)
        
        logger.info(f"Graph built: {self.num_users} users, {self.num_items} items, {len(self.edges)} edges")
        
        return self.graph_data
    
    def _build_id_mappings(self, users: List[User], movies: List[Movie]):
        """Build user_id and item_id to node index mappings."""
        # User mappings (indices 0 to num_users-1)
        self.user_id_to_idx = {user.user_id: idx for idx, user in enumerate(users)}
        self.idx_to_user_id = {idx: user.user_id for idx, user in enumerate(users)}
        self.num_users = len(users)
        
        # Item mappings (indices num_users to num_users+num_items-1)
        self.item_id_to_idx = {movie.movie_id: idx for idx, movie in enumerate(movies)}
        self.idx_to_item_id = {idx: movie.movie_id for idx, movie in enumerate(movies)}
        self.num_items = len(movies)
        
        logger.debug(f"  Mapped {self.num_users} users and {self.num_items} items")
    
    def _build_edges_from_ratings(self, ratings: List[Rating]):
        """Build edge list from ratings."""
        self.edges = set()
        
        for rating in ratings:
            if rating.user_id not in self.user_id_to_idx:
                logger.warning(f"User {rating.user_id} not in mapping, skipping rating")
                continue
            if rating.movie_id not in self.item_id_to_idx:
                logger.warning(f"Movie {rating.movie_id} not in mapping, skipping rating")
                continue
            
            user_idx = self.user_id_to_idx[rating.user_id]
            item_idx = self.item_id_to_idx[rating.movie_id]
            
            # Add edge (user -> item)
            self.edges.add((user_idx, item_idx))
        
        logger.debug(f"  Built {len(self.edges)} unique edges")
    
    def _build_user_feature_matrix(self, users: List[User]) -> torch.Tensor:
        """
        Build user feature matrix.
        
        Args:
            users: List of User models
            
        Returns:
            Tensor of shape (num_users, user_feat_dim)
        """
        feat_dims = self.feature_processor.get_feature_dimensions()
        user_feat_dim = feat_dims['user_feat_dim']
        
        user_features = torch.zeros(self.num_users, user_feat_dim, dtype=torch.float)
        
        for user in users:
            user_idx = self.user_id_to_idx[user.user_id]
            
            # Extract and transform features
            features_dict = self.feature_processor.extract_and_transform_user(user)
            
            # Convert sparse dict to dense vector
            feature_offset = self.feature_processor.preprocessor.feature_offset['user_features']
            for feat_idx, feat_val in features_dict.items():
                local_idx = feat_idx - feature_offset
                if 0 <= local_idx < user_feat_dim:
                    user_features[user_idx, local_idx] = feat_val
        
        logger.debug(f"  Built user feature matrix: {user_features.shape}")
        return user_features
    
    def _build_item_feature_matrix(self, movies: List[Movie]) -> torch.Tensor:
        """
        Build item feature matrix.
        
        Args:
            movies: List of Movie models
            
        Returns:
            Tensor of shape (num_items, item_feat_dim)
        """
        feat_dims = self.feature_processor.get_feature_dimensions()
        item_feat_dim = feat_dims['item_feat_dim']
        
        item_features = torch.zeros(self.num_items, item_feat_dim, dtype=torch.float)
        
        for movie in movies:
            item_idx = self.item_id_to_idx[movie.movie_id]
            
            # Extract and transform features
            features_dict = self.feature_processor.extract_and_transform_movie(movie)
            
            # Convert sparse dict to dense vector
            feature_offset = self.feature_processor.preprocessor.feature_offset['item_features']
            for feat_idx, feat_val in features_dict.items():
                local_idx = feat_idx - feature_offset
                if 0 <= local_idx < item_feat_dim:
                    item_features[item_idx, local_idx] = feat_val
        
        logger.debug(f"  Built item feature matrix: {item_features.shape}")
        return item_features
    
    def _construct_graph_data(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor
    ) -> Data:
        """
        Construct PyTorch Geometric Data object.
        
        Args:
            user_features: User feature matrix (num_users, user_feat_dim)
            item_features: Item feature matrix (num_items, item_feat_dim)
            
        Returns:
            PyTorch Geometric Data object
        """
        # Pad features to same dimension
        user_feat_dim = user_features.size(1)
        item_feat_dim = item_features.size(1)
        max_feat_dim = max(user_feat_dim, item_feat_dim)
        
        if user_feat_dim < max_feat_dim:
            padding = torch.zeros(self.num_users, max_feat_dim - user_feat_dim)
            user_features = torch.cat([user_features, padding], dim=1)
        
        if item_feat_dim < max_feat_dim:
            padding = torch.zeros(self.num_items, max_feat_dim - item_feat_dim)
            item_features = torch.cat([item_features, padding], dim=1)
        
        # Concatenate user and item features
        x = torch.cat([user_features, item_features], dim=0)
        
        # Build edge index (bidirectional)
        edge_list = []
        for user_idx, item_idx in self.edges:
            # Offset item index by num_users
            item_node_idx = item_idx + self.num_users
            
            # Add bidirectional edges
            edge_list.append([user_idx, item_node_idx])
            edge_list.append([item_node_idx, user_idx])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Create node type mask (0=user, 1=item)
        node_type = torch.cat([
            torch.zeros(self.num_users, dtype=torch.long),
            torch.ones(self.num_items, dtype=torch.long)
        ])
        
        # Create Data object
        graph_data = Data(
            x=x,
            edge_index=edge_index,
            node_type=node_type,
            num_users=self.num_users,
            num_items=self.num_items
        )
        
        logger.debug(f"  Graph data: {graph_data.x.shape[0]} nodes, {graph_data.edge_index.shape[1]} edges")
        
        return graph_data
    
    def add_user(self, session: Session, user_id: int) -> int:
        """
        Add a new user node to the graph (cold-start scenario).
        
        Args:
            session: Database session
            user_id: User ID to add
            
        Returns:
            Node index for the new user
            
        Raises:
            ValueError: If user already exists in graph
        """
        if user_id in self.user_id_to_idx:
            logger.warning(f"User {user_id} already in graph")
            return self.user_id_to_idx[user_id]
        
        logger.info(f"Adding new user {user_id} to graph")
        
        # Get user from database
        user = crud.get_user(session, user_id)
        if user is None:
            raise ValueError(f"User {user_id} not found in database")
        
        # Assign new node index
        new_user_idx = self.num_users
        self.user_id_to_idx[user_id] = new_user_idx
        self.idx_to_user_id[new_user_idx] = user_id
        self.num_users += 1
        
        # Rebuild graph with new user
        self._rebuild_graph(session)
        
        logger.debug(f"  User {user_id} added with index {new_user_idx}")
        return new_user_idx
    
    def add_rating(
        self,
        user_id: int,
        movie_id: int,
        rating_value: float
    ) -> bool:
        """
        Add or update an edge when a user rates a movie.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            rating_value: Rating value (not used in graph structure, but logged)
            
        Returns:
            True if edge was added, False if it already existed
        """
        if user_id not in self.user_id_to_idx:
            raise ValueError(f"User {user_id} not in graph. Call add_user() first.")
        
        if movie_id not in self.item_id_to_idx:
            raise ValueError(f"Movie {movie_id} not in graph")
        
        user_idx = self.user_id_to_idx[user_id]
        item_idx = self.item_id_to_idx[movie_id]
        
        edge = (user_idx, item_idx)
        
        if edge in self.edges:
            logger.debug(f"Edge already exists: user {user_id} -> movie {movie_id}")
            return False
        
        # Add edge
        self.edges.add(edge)
        
        # Rebuild edge index in graph_data
        self._rebuild_edge_index()
        
        logger.info(f"Added edge: user {user_id} -> movie {movie_id} (rating={rating_value})")
        return True
    
    def _rebuild_edge_index(self):
        """Rebuild edge_index in graph_data after adding edges."""
        if self.graph_data is None:
            return
        
        edge_list = []
        for user_idx, item_idx in self.edges:
            item_node_idx = item_idx + self.num_users
            edge_list.append([user_idx, item_node_idx])
            edge_list.append([item_node_idx, user_idx])
        
        self.graph_data.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        logger.debug(f"  Edge index rebuilt: {self.graph_data.edge_index.shape[1]} edges")
    
    def _rebuild_graph(self, session: Session):
        """Rebuild entire graph (expensive, used when adding new users)."""
        logger.debug("Rebuilding entire graph...")
        
        # Get all users and movies
        all_users = session.query(User).filter(
            User.user_id.in_(list(self.user_id_to_idx.keys()))
        ).all()
        all_movies = session.query(Movie).all()
        
        # Rebuild feature matrices
        user_features = self._build_user_feature_matrix(all_users)
        item_features = self._build_item_feature_matrix(all_movies)
        
        # Reconstruct graph data
        self.graph_data = self._construct_graph_data(user_features, item_features)
        
        logger.debug("Graph rebuilt")
    
    def get_user_node_index(self, user_id: int) -> Optional[int]:
        """Get node index for a user ID."""
        return self.user_id_to_idx.get(user_id)
    
    def get_item_node_index(self, movie_id: int) -> Optional[int]:
        """Get node index for a movie ID (offset by num_users)."""
        item_idx = self.item_id_to_idx.get(movie_id)
        return item_idx + self.num_users if item_idx is not None else None
    
    def get_graph_stats(self) -> Dict[str, int]:
        """Get graph statistics."""
        return {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'num_edges': len(self.edges),
            'num_nodes': self.num_users + self.num_items
        }


if __name__ == "__main__":
    # Test graph manager
    from app.utils.logging_config import configure_inference_logging
    from app.database.connection import get_session
    import pickle
    
    configure_inference_logging(debug=True)
    
    print("Testing GraphManager...")
    
    try:
        # Load preprocessor
        with open("models/current/preprocessor.pkl", 'rb') as f:
            preprocessor = pickle.load(f)
        
        from app.core.features.preprocessor import FeatureProcessor
        feature_processor = FeatureProcessor(preprocessor)
        
        # Create graph manager
        manager = GraphManager(feature_processor)
        
        # Build graph from database
        session = get_session()
        graph_data = manager.build_graph_from_database(session)
        
        print(f"\nGraph statistics: {manager.get_graph_stats()}")
        print(f"Graph data shape: {graph_data.x.shape}")
        print(f"Edge index shape: {graph_data.edge_index.shape}")
        
    except FileNotFoundError:
        print("\nPreprocessor not found. This is expected if training hasn't been completed.")
