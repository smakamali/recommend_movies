"""
End-to-end integration test for inference flow.

Tests the complete user journey:
1. Load model and initialize graph
2. Add new user (cold-start)
3. Get initial recommendations
4. User rates movies
5. Refresh recommendations (warm-start)
6. Verify filtering works
"""

import pytest
from sqlalchemy.orm import Session

from app.core.inference.engine import InferenceEngine
from app.database.connection import get_session
from app.database import crud
from app.utils.logging_config import configure_inference_logging


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
def engine():
    """Create and initialize inference engine."""
    # Configure logging for tests
    configure_inference_logging(debug=True)
    
    engine = InferenceEngine(model_dir="models/current", cache_embeddings=True)
    return engine


class TestInferenceFlow:
    """Test complete inference flow."""
    
    def test_end_to_end_flow(self, engine, session):
        """
        Test complete end-to-end inference flow.
        
        This test simulates a real user journey through the system.
        """
        print("\n" + "="*60)
        print("INTEGRATION TEST: End-to-End Inference Flow")
        print("="*60)
        
        # Step 1: Load model
        print("\n[Step 1] Loading model...")
        model_info = engine.load_model()
        print(f"  ✓ Model loaded: {model_info.get('version', 'unknown')}")
        
        assert engine.model is not None
        assert engine.preprocessor is not None
        
        # Step 2: Initialize graph
        print("\n[Step 2] Initializing graph from database...")
        graph_stats = engine.initialize_graph(session)
        print(f"  ✓ Graph initialized:")
        print(f"    - Users: {graph_stats['num_users']}")
        print(f"    - Movies: {graph_stats['num_items']}")
        print(f"    - Edges: {graph_stats['num_edges']}")
        
        assert graph_stats['num_users'] > 0
        assert graph_stats['num_items'] > 0
        assert graph_stats['num_edges'] > 0
        
        # Step 3: Add new user (cold-start scenario)
        print("\n[Step 3] Creating new user...")
        new_user = crud.create_user(
            session,
            age=28,
            gender='F',
            occupation='programmer',
            zip_code='94043'
        )
        print(f"  ✓ Created user {new_user.user_id}")
        
        # Add user to graph
        user_idx = engine.add_user(session, new_user.user_id)
        print(f"  ✓ Added to graph with index {user_idx}")
        
        # Step 4: Get cold-start recommendations
        print(f"\n[Step 4] Getting cold-start recommendations for user {new_user.user_id}...")
        cold_start_recs = engine.get_recommendations(
            session,
            new_user.user_id,
            n=10,
            exclude_already_rated=False  # User has no ratings yet
        )
        
        print(f"  ✓ Generated {len(cold_start_recs)} recommendations")
        if cold_start_recs:
            print(f"\n  Top 5 cold-start recommendations:")
            for rec in cold_start_recs[:5]:
                print(f"    {rec['rank']}. {rec['title']} (score: {rec['score']:.3f})")
        
        assert len(cold_start_recs) <= 10
        
        # Step 5: User rates some movies
        print(f"\n[Step 5] User rates movies...")
        
        # Rate highly (good movies)
        highly_rated_movies = []
        if len(cold_start_recs) >= 3:
            for i in range(3):
                movie_id = cold_start_recs[i]['movie_id']
                rating = crud.create_rating(session, new_user.user_id, movie_id, 5.0)
                engine.add_rating(session, new_user.user_id, movie_id, 5.0)
                highly_rated_movies.append(movie_id)
                print(f"  ✓ Rated movie {movie_id} with 5.0")
        
        # Rate poorly (bad movies)
        poorly_rated_movies = []
        available_movies = [m.movie_id for m in crud.get_movies(session, limit=100) 
                          if m.movie_id not in highly_rated_movies]
        
        if len(available_movies) >= 2:
            for movie_id in available_movies[:2]:
                rating = crud.create_rating(session, new_user.user_id, movie_id, 1.0)
                engine.add_rating(session, new_user.user_id, movie_id, 1.0)
                poorly_rated_movies.append(movie_id)
                print(f"  ✓ Rated movie {movie_id} with 1.0")
        
        # Step 6: Get warm-start recommendations (with filtering)
        print(f"\n[Step 6] Getting warm-start recommendations (excluding low-rated)...")
        warm_start_recs = engine.get_recommendations(
            session,
            new_user.user_id,
            n=10,
            exclude_low_rated=True,  # Exclude movies rated ≤2
            exclude_already_rated=True  # Exclude all rated movies
        )
        
        print(f"  ✓ Generated {len(warm_start_recs)} recommendations")
        if warm_start_recs:
            print(f"\n  Top 5 warm-start recommendations:")
            for rec in warm_start_recs[:5]:
                print(f"    {rec['rank']}. {rec['title']} (score: {rec['score']:.3f})")
        
        assert len(warm_start_recs) <= 10
        
        # Step 7: Verify filtering works
        print(f"\n[Step 7] Verifying filtering...")
        
        # Check that poorly-rated movies are not in recommendations
        recommended_ids = {rec['movie_id'] for rec in warm_start_recs}
        
        for movie_id in poorly_rated_movies:
            assert movie_id not in recommended_ids, \
                f"Poorly-rated movie {movie_id} should not be recommended"
        print(f"  ✓ Poorly-rated movies correctly excluded")
        
        # Check that highly-rated movies are not in recommendations (already rated)
        for movie_id in highly_rated_movies:
            assert movie_id not in recommended_ids, \
                f"Already-rated movie {movie_id} should not be recommended"
        print(f"  ✓ Already-rated movies correctly excluded")
        
        # Step 8: Test refresh recommendations
        print(f"\n[Step 8] Testing refresh recommendations...")
        refreshed_recs = engine.refresh_recommendations(session, new_user.user_id, n=5)
        
        print(f"  ✓ Refreshed {len(refreshed_recs)} recommendations")
        
        assert len(refreshed_recs) <= 5
        
        # Step 9: Test engine status
        print(f"\n[Step 9] Checking engine status...")
        status = engine.get_status()
        
        print(f"  ✓ Engine status:")
        print(f"    - Model loaded: {status['model_loaded']}")
        print(f"    - Graph initialized: {status['graph_initialized']}")
        print(f"    - Cache enabled: {status['cache_enabled']}")
        
        assert status['model_loaded'] == True
        assert status['graph_initialized'] == True
        
        # Cleanup: Delete test user and ratings
        print(f"\n[Cleanup] Deleting test user and ratings...")
        crud.delete_user(session, new_user.user_id)
        print(f"  ✓ Test user deleted")
        
        print("\n" + "="*60)
        print("✅ INTEGRATION TEST PASSED")
        print("="*60)
    
    def test_multiple_users_concurrent(self, engine, session):
        """Test getting recommendations for multiple users."""
        print("\n[Test] Multiple users concurrent recommendations...")
        
        # Initialize engine
        engine.load_model()
        engine.initialize_graph(session)
        
        # Get recommendations for multiple existing users
        user_ids = [1, 2, 3]
        
        for user_id in user_ids:
            user = crud.get_user(session, user_id)
            if user:
                recs = engine.get_recommendations(session, user_id, n=5)
                print(f"  ✓ User {user_id}: {len(recs)} recommendations")
                assert len(recs) <= 5
        
        print("  ✓ Multiple users handled successfully")
    
    def test_cold_start_with_demographics_only(self, engine, session):
        """Test cold-start recommendation with only demographic info."""
        print("\n[Test] Cold-start with demographics only...")
        
        # Initialize engine
        engine.load_model()
        engine.initialize_graph(session)
        
        # Create user with minimal info
        new_user = crud.create_user(
            session,
            age=25,
            gender='M',
            occupation='student'
        )
        
        try:
            # Add to graph
            engine.add_user(session, new_user.user_id)
            
            # Get recommendations (cold-start)
            recs = engine.get_recommendations(
                session,
                new_user.user_id,
                n=10,
                exclude_already_rated=False
            )
            
            print(f"  ✓ Generated {len(recs)} cold-start recommendations")
            assert len(recs) <= 10
            
        finally:
            # Cleanup
            crud.delete_user(session, new_user.user_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements
