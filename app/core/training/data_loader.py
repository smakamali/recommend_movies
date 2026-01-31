"""
Data loader for training pipeline.

Loads data from SQLite database and converts to format compatible
with the POC GraphSAGE training code (Surprise Trainset format).
"""

import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Tuple, List, Dict
from surprise import Dataset, Reader, Trainset
from surprise.model_selection import train_test_split as surprise_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.database import get_db_manager, crud


def load_data_from_database(db_path: str = "data/recommender.db") -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple]]:
    """
    Load users, movies, and ratings from SQLite database.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        Tuple of (users_df, movies_df, ratings_list)
        - users_df: DataFrame with user demographics
        - movies_df: DataFrame with movie metadata
        - ratings_list: List of (user_id, movie_id, rating) tuples
    """
    db_manager = get_db_manager(db_path=db_path)
    session = db_manager.get_session()
    
    try:
        # Load users
        users = crud.get_users(session, skip=0, limit=10000)
        users_data = []
        for user in users:
            users_data.append({
                'user_id': str(user.user_id),
                'age': user.age,
                'gender': user.gender,
                'occupation': user.occupation,
                'zip_code': user.zip_code or ''
            })
        users_df = pd.DataFrame(users_data)
        
        # Load movies
        movies = crud.get_movies(session, skip=0, limit=10000)
        movies_data = []
        for movie in movies:
            genres = json.loads(movie.genres)
            movies_data.append({
                'item_id': str(movie.movie_id),
                'title': movie.title,
                'release_year': movie.release_year if movie.release_year else 1995,
                'genres': genres
            })
        movies_df = pd.DataFrame(movies_data)
        
        # Load ratings
        ratings = crud.get_all_ratings(session, skip=0, limit=200000)
        ratings_list = []
        for rating in ratings:
            ratings_list.append((
                str(rating.user_id),
                str(rating.movie_id),
                float(rating.rating)
            ))
        
        print(f"Loaded from database:")
        print(f"  Users: {len(users_df)}")
        print(f"  Movies: {len(movies_df)}")
        print(f"  Ratings: {len(ratings_list)}")
        
        return users_df, movies_df, ratings_list
        
    finally:
        session.close()


def create_surprise_trainset(ratings_list: List[Tuple], test_size: float = 0.2, random_state: int = 42) -> Tuple[Trainset, List]:
    """
    Convert ratings to Surprise Trainset format with train/test split.
    
    Args:
        ratings_list: List of (user_id, item_id, rating) tuples
        test_size: Fraction of data for testing
        random_state: Random seed
        
    Returns:
        Tuple of (trainset, testset)
    """
    # Create DataFrame for Surprise
    ratings_df = pd.DataFrame(ratings_list, columns=['user_id', 'item_id', 'rating'])
    
    # Create Surprise Dataset
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)
    
    # Split into train/test
    trainset, testset = surprise_split(data, test_size=test_size, random_state=random_state)
    
    print(f"Train/test split:")
    print(f"  Train ratings: {trainset.n_ratings}")
    print(f"  Test ratings: {len(testset)}")
    print(f"  Train users: {trainset.n_users}")
    print(f"  Train items: {trainset.n_items}")
    
    return trainset, testset


def extract_user_features(users_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract user demographic features.
    
    Args:
        users_df: DataFrame with user data
        
    Returns:
        DataFrame with user features (age, gender, occupation)
    """
    return users_df[['user_id', 'age', 'gender', 'occupation', 'zip_code']].copy()


def extract_movie_features(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract movie content features.
    
    Args:
        movies_df: DataFrame with movie data
        
    Returns:
        DataFrame with movie features (year, genres)
    """
    # Genre names (19 genres in MovieLens 100k)
    genre_names = [
        'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    
    # Create genre columns
    features_df = movies_df[['item_id', 'title', 'release_year']].copy()
    
    # Add binary genre columns
    for genre in genre_names:
        features_df[f'genre_{genre}'] = movies_df['genres'].apply(
            lambda x: 1 if genre in x else 0
        )
    
    return features_df


def load_training_data(db_path: str = "data/recommender.db", test_size: float = 0.2, random_state: int = 42):
    """
    Load complete training data from database.
    
    This is the main function to call for preparing training data.
    
    Args:
        db_path: Path to SQLite database
        test_size: Fraction of data for testing
        random_state: Random seed
        
    Returns:
        Dict with:
        - trainset: Surprise Trainset
        - testset: List of test samples
        - user_features_df: User features
        - item_features_df: Movie features
        - users_df: Original user data
        - movies_df: Original movie data
    """
    # Load from database
    users_df, movies_df, ratings_list = load_data_from_database(db_path)
    
    # Create train/test split
    trainset, testset = create_surprise_trainset(ratings_list, test_size, random_state)
    
    # Extract features
    user_features_df = extract_user_features(users_df)
    item_features_df = extract_movie_features(movies_df)
    
    return {
        'trainset': trainset,
        'testset': testset,
        'user_features_df': user_features_df,
        'item_features_df': item_features_df,
        'users_df': users_df,
        'movies_df': movies_df,
        'num_ratings': len(ratings_list)
    }


if __name__ == "__main__":
    # Test data loading
    print("Testing data loader...")
    data = load_training_data()
    print("\n[SUCCESS] Data loaded successfully!")
    print(f"Training data ready: {data['trainset'].n_ratings:,} train, {len(data['testset']):,} test")
