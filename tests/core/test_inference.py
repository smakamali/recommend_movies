"""
Unit tests for inference components.

Tests model loading, graph management, feature processing,
and recommendation generation.
"""

import pytest
import torch
from pathlib import Path
from sqlalchemy.orm import Session

from app.core.inference.model_loader import ModelLoader
from app.core.inference.graph_manager import GraphManager
from app.core.inference.recommender import Recommender
from app.core.inference.engine import InferenceEngine
from app.core.features.preprocessor import FeatureProcessor
from app.database.connection import get_session
from app.database import crud


@pytest.fixture
def session():
    """Create database session."""
    # Get database manager and create a session
    from app.database.connection import get_db_manager
    db_manager = get_db_manager()
    session = db_manager.get_session()
    yield session
    # Close session after test
    session.close()


@pytest.fixture
def model_loader():
    """Create model loader."""
    return ModelLoader(model_dir="models/current")


@pytest.fixture
def loaded_artifacts(model_loader):
    """Load model artifacts."""
    model, preprocessor, metadata = model_loader.load_model()
    return model, preprocessor, metadata


@pytest.fixture
def feature_processor(loaded_artifacts):
    """Create feature processor."""
    _, preprocessor, _ = loaded_artifacts
    return FeatureProcessor(preprocessor)


@pytest.fixture
def graph_manager(feature_processor):
    """Create graph manager."""
    return GraphManager(feature_processor)


@pytest.fixture
def built_graph(graph_manager, session):
    """Build graph from database."""
    return graph_manager.build_graph_from_database(session)


@pytest.fixture
def recommender(loaded_artifacts):
    """Create recommender."""
    model, _, _ = loaded_artifacts
    return Recommender(model)


class TestModelLoader:
    """Test model loading functionality."""
    
    def test_model_loader_initialization(self, model_loader):
        """Test that model loader initializes correctly."""
        assert model_loader is not None
        assert model_loader.model_dir == Path("models/current")
        assert model_loader.device in ['cpu', 'cuda']
    
    def test_load_model(self, model_loader):
        """Test loading model artifacts."""
        model, preprocessor, metadata = model_loader.load_model()
        
        assert model is not None
        assert preprocessor is not None
        assert metadata is not None
        
        # Check model is in eval mode
        assert not model.training
    
    def test_model_info(self, model_loader):
        """Test getting model info."""
        model_loader.load_model()
        info = model_loader.get_model_info()
        
        assert info['status'] == 'loaded'
        assert 'hidden_dim' in info
        assert 'num_layers' in info
    
    def test_cache(self, model_loader):
        """Test that model is cached after first load."""
        # First load
        model1, _, _ = model_loader.load_model()
        
        # Second load (should be cached)
        model2, _, _ = model_loader.load_model()
        
        # Should be same object
        assert model1 is model2


class TestFeatureProcessor:
    """Test feature extraction and preprocessing."""
    
    def test_extract_user_features(self, feature_processor, session):
        """Test extracting user features."""
        user = crud.get_user(session, user_id=1)
        
        if user:
            features = feature_processor.extract_user_features(user)
            
            assert 'user_id' in features
            assert 'age' in features
            assert 'gender' in features
            assert 'occupation' in features
    
    def test_extract_movie_features(self, feature_processor, session):
        """Test extracting movie features."""
        movie = crud.get_movie(session, movie_id=1)
        
        if movie:
            features = feature_processor.extract_movie_features(movie)
            
            assert 'item_id' in features
            assert 'release_year' in features
            # Check genre columns
            assert any(k.startswith('genre_') for k in features.keys())
    
    def test_feature_dimensions(self, feature_processor):
        """Test getting feature dimensions."""
        dims = feature_processor.get_feature_dimensions()
        
        assert 'user_feat_dim' in dims
        assert 'item_feat_dim' in dims
        assert dims['user_feat_dim'] > 0
        assert dims['item_feat_dim'] > 0


class TestGraphManager:
    """Test graph construction and management."""
    
    def test_build_graph(self, graph_manager, session):
        """Test building graph from database."""
        graph_data = graph_manager.build_graph_from_database(session)
        
        assert graph_data is not None
        assert graph_data.x.size(0) > 0  # Has nodes
        assert graph_data.edge_index.size(1) > 0  # Has edges
        assert graph_manager.num_users > 0
        assert graph_manager.num_items > 0
    
    def test_graph_statistics(self, built_graph, graph_manager):
        """Test graph statistics."""
        stats = graph_manager.get_graph_stats()
        
        assert stats['num_users'] > 0
        assert stats['num_items'] > 0
        assert stats['num_edges'] > 0
        assert stats['num_nodes'] == stats['num_users'] + stats['num_items']
    
    def test_node_mappings(self, built_graph, graph_manager):
        """Test user and item node mappings."""
        # Check that mappings exist
        assert len(graph_manager.user_id_to_idx) > 0
        assert len(graph_manager.item_id_to_idx) > 0
        
        # Check reverse mappings
        assert len(graph_manager.idx_to_user_id) == len(graph_manager.user_id_to_idx)
        assert len(graph_manager.idx_to_item_id) == len(graph_manager.item_id_to_idx)
    
    def test_add_user(self, built_graph, graph_manager, session):
        """Test adding a new user to graph."""
        # Create a new user
        new_user = crud.create_user(
            session,
            age=25,
            gender='M',
            occupation='student',
            zip_code='12345'
        )
        
        initial_num_users = graph_manager.num_users
        
        # Add user to graph
        user_idx = graph_manager.add_user(session, new_user.user_id)
        
        assert user_idx is not None
        assert new_user.user_id in graph_manager.user_id_to_idx
        
        # Clean up
        crud.delete_user(session, new_user.user_id)
    
    def test_add_rating(self, built_graph, graph_manager, session):
        """Test adding a rating edge."""
        # Get first user and movie
        user = crud.get_user(session, user_id=1)
        movie = crud.get_movie(session, movie_id=1)
        
        if user and movie:
            initial_edges = len(graph_manager.edges)
            
            # Add rating (may already exist)
            graph_manager.add_rating(user.user_id, movie.movie_id, 4.0)
            
            # Edge count should be >= initial
            assert len(graph_manager.edges) >= initial_edges


class TestRecommender:
    """Test recommendation generation."""
    
    def test_generate_embeddings(self, recommender, built_graph):
        """Test generating embeddings."""
        user_emb, item_emb = recommender.generate_embeddings(built_graph)
        
        assert user_emb is not None
        assert item_emb is not None
        assert user_emb.size(0) == built_graph.num_users
        assert item_emb.size(0) == built_graph.num_items
    
    def test_compute_scores(self, recommender, built_graph):
        """Test computing recommendation scores."""
        user_emb, item_emb = recommender.generate_embeddings(built_graph)
        
        # Compute scores for first user
        scores = recommender.compute_scores(user_emb, item_emb, user_idx=0)
        
        assert scores is not None
        assert scores.size(0) == item_emb.size(0)
    
    def test_get_recommendations(
        self,
        recommender,
        built_graph,
        graph_manager,
        session
    ):
        """Test getting recommendations."""
        user_emb, item_emb = recommender.generate_embeddings(built_graph)
        
        user_id = 1
        user_idx = graph_manager.get_user_node_index(user_id)
        
        if user_idx is not None:
            recommendations = recommender.get_recommendations(
                session,
                user_id,
                user_emb,
                item_emb,
                user_idx,
                graph_manager.item_id_to_idx,
                graph_manager.idx_to_item_id,
                n=5
            )
            
            assert len(recommendations) <= 5
            
            # Check recommendation structure
            if recommendations:
                rec = recommendations[0]
                assert 'movie_id' in rec
                assert 'title' in rec
                assert 'score' in rec
                assert 'rank' in rec


class TestInferenceEngine:
    """Test complete inference engine."""
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = InferenceEngine(model_dir="models/current")
        
        assert engine is not None
        assert engine.device in ['cpu', 'cuda']
    
    def test_load_model(self):
        """Test loading model through engine."""
        engine = InferenceEngine(model_dir="models/current")
        info = engine.load_model()
        
        assert info is not None
        assert engine.model is not None
        assert engine.preprocessor is not None
    
    def test_initialize_graph(self, session):
        """Test initializing graph through engine."""
        engine = InferenceEngine(model_dir="models/current")
        engine.load_model()
        
        stats = engine.initialize_graph(session)
        
        assert stats is not None
        assert stats['num_users'] > 0
        assert stats['num_items'] > 0
    
    def test_get_recommendations(self, session):
        """Test getting recommendations through engine."""
        engine = InferenceEngine(model_dir="models/current")
        engine.load_model()
        engine.initialize_graph(session)
        
        user_id = 1
        recommendations = engine.get_recommendations(session, user_id, n=5)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5
    
    def test_engine_status(self, session):
        """Test getting engine status."""
        engine = InferenceEngine(model_dir="models/current")
        
        # Before initialization
        status = engine.get_status()
        assert status['model_loaded'] == False
        assert status['graph_initialized'] == False
        
        # After loading model
        engine.load_model()
        status = engine.get_status()
        assert status['model_loaded'] == True
        
        # After initializing graph
        engine.initialize_graph(session)
        status = engine.get_status()
        assert status['graph_initialized'] == True
    
    def test_cache_invalidation(self, session):
        """Test cache invalidation after adding rating."""
        engine = InferenceEngine(model_dir="models/current", cache_embeddings=True)
        engine.load_model()
        engine.initialize_graph(session)
        
        # Get initial recommendations
        user_id = 1
        recs1 = engine.get_recommendations(session, user_id, n=5)
        
        # Cache should be valid
        assert engine._cache_valid == True
        
        # Add a rating (if possible)
        movie = crud.get_movie(session, movie_id=100)
        if movie:
            # Check if rating already exists
            existing = crud.get_rating_by_user_movie(session, user_id, movie.movie_id)
            if not existing:
                # Add new rating
                crud.create_rating(session, user_id, movie.movie_id, 4.0)
                engine.add_rating(session, user_id, movie.movie_id, 4.0)
                
                # Cache should be invalidated
                assert engine._cache_valid == False
                
                # Clean up
                rating = crud.get_rating_by_user_movie(session, user_id, movie.movie_id)
                if rating:
                    crud.delete_rating(session, rating.rating_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
