"""
Feature extraction and preprocessing wrapper.

Provides a convenient wrapper around the POC's FeaturePreprocessor
for use in the inference engine.
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional, List

from poc.data_loader import FeaturePreprocessor as POCFeaturePreprocessor
from app.database.models import User, Movie

logger = logging.getLogger(__name__)


class FeatureProcessor:
    """
    Wrapper around POC's FeaturePreprocessor for feature extraction.
    
    This class provides:
    - Extraction of user features from database models
    - Extraction of movie features from database models
    - Feature transformation using fitted preprocessor
    - Default value handling for missing features
    """
    
    def __init__(self, preprocessor: POCFeaturePreprocessor):
        """
        Initialize feature processor with a fitted preprocessor.
        
        Args:
            preprocessor: Fitted FeaturePreprocessor from POC
        """
        self.preprocessor = preprocessor
        
        # Default values for missing features
        self.default_age = 30
        self.default_gender = 'M'
        self.default_occupation = 'other'
        self.default_year = 1995
        
        logger.debug("FeatureProcessor initialized")
    
    def extract_user_features(
        self,
        user: User,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract features from User model.
        
        Args:
            user: User database model
            user_id: Optional user_id override (default: use user.user_id)
            
        Returns:
            Dictionary with user features ready for transformation
        """
        if user_id is None:
            user_id = str(user.user_id)
        
        return {
            'user_id': user_id,
            'age': user.age if user.age else self.default_age,
            'gender': user.gender if user.gender else self.default_gender,
            'occupation': user.occupation if user.occupation else self.default_occupation,
            'zip_code': user.zip_code
        }
    
    def extract_movie_features(
        self,
        movie: Movie,
        movie_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract features from Movie model.
        
        Args:
            movie: Movie database model
            movie_id: Optional movie_id override (default: use movie.movie_id)
            
        Returns:
            Dictionary with movie features ready for transformation
        """
        if movie_id is None:
            movie_id = str(movie.movie_id)
        
        # Parse genres from JSON string
        import json
        try:
            genres = json.loads(movie.genres) if isinstance(movie.genres, str) else movie.genres
        except:
            genres = []
        
        # Create genre feature dict (similar to POC's item features)
        feature_dict = {
            'item_id': movie_id,
            'title': movie.title,
            'release_year': movie.release_year if movie.release_year else self.default_year
        }
        
        # Add genre columns (19 genres from MovieLens)
        genre_names = [
            'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
            'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        
        for genre in genre_names:
            feature_dict[f'genre_{genre}'] = 1 if genre in genres else 0
        
        return feature_dict
    
    def transform_user_features(
        self,
        user_features: Dict[str, Any]
    ) -> Dict[int, float]:
        """
        Transform user features using fitted preprocessor.
        
        Args:
            user_features: Dictionary with raw user features
            
        Returns:
            Dictionary mapping feature_idx -> value (normalized/encoded)
        """
        # Convert to DataFrame format expected by preprocessor
        user_df = pd.DataFrame([user_features])
        
        # Transform using preprocessor
        transformed = self.preprocessor.transform_user_features(user_df)
        
        # Return features for this user
        user_id = user_features['user_id']
        return transformed.get(user_id, {})
    
    def transform_movie_features(
        self,
        movie_features: Dict[str, Any]
    ) -> Dict[int, float]:
        """
        Transform movie features using fitted preprocessor.
        
        Args:
            movie_features: Dictionary with raw movie features
            
        Returns:
            Dictionary mapping feature_idx -> value (normalized/encoded)
        """
        # Convert to DataFrame format expected by preprocessor
        movie_df = pd.DataFrame([movie_features])
        
        # Transform using preprocessor
        transformed = self.preprocessor.transform_item_features(movie_df)
        
        # Return features for this movie
        movie_id = movie_features['item_id']
        return transformed.get(movie_id, {})
    
    def extract_and_transform_user(self, user: User) -> Dict[int, float]:
        """
        Extract and transform user features in one call.
        
        Args:
            user: User database model
            
        Returns:
            Dictionary mapping feature_idx -> value
        """
        raw_features = self.extract_user_features(user)
        return self.transform_user_features(raw_features)
    
    def extract_and_transform_movie(self, movie: Movie) -> Dict[int, float]:
        """
        Extract and transform movie features in one call.
        
        Args:
            movie: Movie database model
            
        Returns:
            Dictionary mapping feature_idx -> value
        """
        raw_features = self.extract_movie_features(movie)
        return self.transform_movie_features(raw_features)
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        Get feature dimensions from preprocessor.
        
        Returns:
            Dictionary with feature dimension information
        """
        n_genders = len(self.preprocessor.gender_encoder.classes_)
        n_occupations = len(self.preprocessor.occupation_encoder.classes_)
        
        return {
            'user_feat_dim': 1 + n_genders + n_occupations,  # age + gender + occupation
            'item_feat_dim': 1 + 19,  # year + 19 genres
            'n_genders': n_genders,
            'n_occupations': n_occupations
        }


if __name__ == "__main__":
    # Test feature processor
    from app.utils.logging_config import configure_inference_logging
    from app.database.connection import get_session
    from app.database import crud
    
    configure_inference_logging(debug=True)
    
    print("Testing FeatureProcessor...")
    print("\nNote: This requires a fitted preprocessor from training.")
    print("If training hasn't been completed, this test will fail.")
    
    try:
        # This would normally come from ModelLoader
        import pickle
        with open("models/current/preprocessor.pkl", 'rb') as f:
            preprocessor = pickle.load(f)
        
        processor = FeatureProcessor(preprocessor)
        
        # Test with database user/movie
        session = get_session()
        user = crud.get_user(session, user_id=1)
        movie = crud.get_movie(session, movie_id=1)
        
        if user:
            user_features = processor.extract_and_transform_user(user)
            print(f"\nUser features extracted: {len(user_features)} features")
        
        if movie:
            movie_features = processor.extract_and_transform_movie(movie)
            print(f"Movie features extracted: {len(movie_features)} features")
        
        print(f"\nFeature dimensions: {processor.get_feature_dimensions()}")
        
    except FileNotFoundError:
        print("\nPreprocessor not found. This is expected if training hasn't been completed.")
