"""
Data loading and preprocessing for MovieLens 100K dataset.

This module handles downloading and loading the MovieLens 100K dataset,
including train/test splitting and data preprocessing.
Extended to support user and item features for Factorization Machines.
"""

import pandas as pd
import numpy as np
import os
import urllib.request
import zipfile
import shutil
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, LabelEncoder
import scipy.sparse as sp

# Note: Surprise removed - using SimpleTrainset and custom loaders


def load_movielens_100k(data_dir='data'):
    """
    Load MovieLens 100K dataset from u.data file.
    
    The dataset is downloaded if not present locally (via get_movielens_data_path).
    
    Args:
        data_dir (str): Directory hint (default: 'data')
        
    Returns:
        list: List of (user_id, item_id, rating) tuples
    """
    data_path = get_movielens_data_path()
    ratings_file = os.path.join(data_path, 'u.data')
    
    ratings_list = []
    with open(ratings_file, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                uid, iid, rating = parts[0], parts[1], float(parts[2])
                ratings_list.append((uid, iid, rating))
    
    return ratings_list


def get_train_test_split(ratings_list, test_size=0.2, random_state=42):
    """
    Split ratings into train/test sets.
    
    Args:
        ratings_list: List of (user_id, item_id, rating) tuples
        test_size (float): Proportion of data for testing (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
        
    Returns:
        tuple: (trainset, testset) where trainset is SimpleTrainset, testset is list of (uid, iid, rating)
    """
    from sklearn.model_selection import train_test_split
    from poc.trainset import SimpleTrainset
    
    train_ratings, test_ratings = train_test_split(
        ratings_list,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    trainset = SimpleTrainset(train_ratings)
    testset = [tuple(r) for r in test_ratings]
    return trainset, testset


def load_dataframe_from_builtin():
    """
    Load MovieLens 100K as pandas DataFrame for easier manipulation.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['user_id', 'item_id', 'rating', 'timestamp']
    """
    ratings_list = load_movielens_100k()
    df = pd.DataFrame(ratings_list, columns=['user_id', 'item_id', 'rating'])
    df['timestamp'] = 0  # Not used
    return df


def get_dataset_stats(data):
    """
    Print statistics about the dataset.
    
    Args:
        data: List of (uid, iid, rating) tuples
    """
    df = load_dataframe_from_builtin()
    
    print("=" * 60)
    print("MovieLens 100K Dataset Statistics")
    print("=" * 60)
    print(f"Total ratings: {len(df):,}")
    print(f"Number of users: {df['user_id'].nunique():,}")
    print(f"Number of items: {df['item_id'].nunique():,}")
    print(f"Rating scale: {df['rating'].min()} - {df['rating'].max()}")
    print(f"Sparsity: {(1 - len(df) / (df['user_id'].nunique() * df['item_id'].nunique())) * 100:.2f}%")
    print(f"Average ratings per user: {len(df) / df['user_id'].nunique():.2f}")
    print(f"Average ratings per item: {len(df) / df['item_id'].nunique():.2f}")
    print("=" * 60)


def get_movielens_data_path():
    """
    Get the path to MovieLens 100K data files.
    
    Surprise stores datasets in a cache directory. This function locates
    the ml-100k dataset files (u.user, u.item, etc.). If not found, downloads
    the full dataset.
    
    Returns:
        str: Path to directory containing MovieLens 100K files
    """
    # Check common locations for MovieLens 100K data
    possible_paths = [
        Path.home() / '.surprise_data' / 'ml-100k',
        Path('data') / 'ml-100k',
        Path('.') / 'data' / 'ml-100k',
    ]
    
    for path in possible_paths:
        if path.exists() and (path / 'u.user').exists():
            return str(path)
    
    # If not found, download the full dataset
    print("MovieLens 100K feature files not found. Downloading full dataset...")
    return _download_movielens_100k()


def _download_movielens_100k():
    """
    Download the full MovieLens 100K dataset including feature files.
    
    Returns:
        str: Path to directory containing MovieLens 100K files
    """
    # Use Surprise's data directory
    home_dir = Path.home()
    data_dir = home_dir / '.surprise_data' / 'ml-100k'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we already have the files
    if (data_dir / 'u.user').exists() and (data_dir / 'u.item').exists():
        return str(data_dir)
    
    # Download URL for MovieLens 100K
    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = data_dir / 'ml-100k.zip'
    
    print(f"  Downloading from {url}...")
    print(f"  Saving to {zip_path}...")
    
    try:
        # Download the zip file
        urllib.request.urlretrieve(url, zip_path)
        
        # Extract the zip file
        print("  Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract to a temporary directory
            temp_extract = data_dir / 'temp_extract'
            zip_ref.extractall(temp_extract)
            
            # Move files from ml-100k subdirectory to data_dir
            extracted_dir = temp_extract / 'ml-100k'
            if extracted_dir.exists():
                for file in extracted_dir.iterdir():
                    if file.is_file():
                        shutil.move(str(file), str(data_dir / file.name))
            
            # Clean up
            shutil.rmtree(temp_extract, ignore_errors=True)
        
        # Remove zip file
        zip_path.unlink()
        
        print("  Download complete!")
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to download MovieLens 100K dataset: {e}\n"
            f"Please download manually from: {url}\n"
            f"Extract to: {data_dir}"
        )
    
    return str(data_dir)


def load_user_features():
    """
    Load user demographic features from MovieLens 100k.
    
    Reads from u.user file with format:
    user_id|age|gender|occupation|zip_code
    
    Returns:
        pd.DataFrame with columns: ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    """
    data_path = get_movielens_data_path()
    user_file = os.path.join(data_path, 'u.user')
    
    # Read user file
    # Format: user_id|age|gender|occupation|zip_code
    user_df = pd.read_csv(
        user_file,
        sep='|',
        header=None,
        names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
        encoding='latin-1'
    )
    
    # Convert user_id to string for consistency
    user_df['user_id'] = user_df['user_id'].astype(str)
    
    return user_df


def load_item_features():
    """
    Load movie content features from MovieLens 100k.
    
    Reads from u.item file with format:
    movie_id|title|release_date|video_release_date|IMDb_URL|19 genre columns
    
    Returns:
        pd.DataFrame with columns: ['item_id', 'title', 'release_date', 
                                   'release_year', 'genre_*'] (19 genre columns)
    """
    data_path = get_movielens_data_path()
    item_file = os.path.join(data_path, 'u.item')
    
    # Genre names (19 genres in MovieLens 100k)
    genre_names = [
        'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    
    # Read item file
    # Format: movie_id|title|release_date|video_release_date|IMDb_URL|19 genre columns
    item_df = pd.read_csv(
        item_file,
        sep='|',
        header=None,
        names=['item_id', 'title', 'release_date', 'video_release_date', 
               'IMDb_URL'] + [f'genre_{g}' for g in genre_names],
        encoding='latin-1'
    )
    
    # Convert item_id to string for consistency
    item_df['item_id'] = item_df['item_id'].astype(str)
    
    # Extract release year from release_date
    # Format: DD-MMM-YYYY or empty
    def extract_year(date_str):
        if pd.isna(date_str) or date_str == '':
            return None
        try:
            # Try to parse date
            parts = str(date_str).split('-')
            if len(parts) == 3:
                return int(parts[2])
        except:
            pass
        return None
    
    item_df['release_year'] = item_df['release_date'].apply(extract_year)
    
    # Fill missing years with median
    median_year = item_df['release_year'].median()
    item_df['release_year'] = item_df['release_year'].fillna(median_year).astype(int)
    
    return item_df


def _get_schema_superset():
    """
    Load occupation and gender superset from MovieLens 100K u.user.
    Used to fit encoders on full schema so API-created users work at inference.
    Returns (occupations, genders) or (None, None) if u.user not available.
    """
    try:
        schema_df = load_user_features()
        return schema_df['occupation'].unique(), schema_df['gender'].unique()
    except (FileNotFoundError, OSError, Exception):
        return None, None


class FeaturePreprocessor:
    """
    Preprocess and normalize user/item features for Factorization Machines.
    
    Handles categorical encoding, normalization, missing values, and
    conversion to sparse matrices for myFM or libsvm format (deprecated).
    """
    
    def __init__(self):
        self.age_scaler = StandardScaler()
        self.year_scaler = StandardScaler()
        self.occupation_encoder = LabelEncoder()
        self.gender_encoder = LabelEncoder()
        self.user_id_map = {}  # Maps user_id to feature index
        self.item_id_map = {}  # Maps item_id to feature index
        self.feature_offset = {}  # Tracks feature index offsets
        self.fitted = False
        
    def fit(self, user_df, item_df):
        """
        Fit scalers and encoders on training data.
        
        Args:
            user_df: DataFrame with user features
            item_df: DataFrame with item features
        """
        # Fit age scaler (use .values to avoid feature name warnings)
        self.age_scaler.fit(user_df[['age']].values)
        
        # Fit year scaler (use .values to avoid feature name warnings)
        self.year_scaler.fit(item_df[['release_year']].values)
        
        # Fit encoders on superset of training data + schema from MovieLens u.user.
        # Ensures API-created users with any valid occupation/gender work at inference.
        occ_train = user_df['occupation'].unique()
        gen_train = user_df['gender'].unique()
        occ_schema, gen_schema = _get_schema_superset()
        occ_all = np.union1d(occ_train, occ_schema) if occ_schema is not None else occ_train
        gen_all = np.union1d(gen_train, gen_schema) if gen_schema is not None else gen_train
        self.occupation_encoder.fit(occ_all)
        self.gender_encoder.fit(gen_all)
        
        # Create feature index mappings
        # User IDs: indices 0 to n_users-1
        unique_users = sorted(user_df['user_id'].unique())
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        user_offset = len(unique_users)
        
        # Item IDs: indices n_users to n_users+n_items-1
        unique_items = sorted(item_df['item_id'].unique())
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        item_offset = len(unique_items)
        
        # Feature offsets
        self.feature_offset = {
            'user_id': 0,
            'item_id': user_offset,
            'user_features': user_offset + item_offset,
        }
        
        # Count user features: age (1) + gender (2) + occupation (n_occ)
        n_user_features = 1 + len(self.gender_encoder.classes_) + len(self.occupation_encoder.classes_)
        
        # Count item features: year (1) + genres (19)
        n_item_features = 1 + 19
        
        self.feature_offset['item_features'] = self.feature_offset['user_features'] + n_user_features
        self.total_features = self.feature_offset['item_features'] + n_item_features
        
        self.fitted = True
        
    def transform_user_features(self, user_df):
        """
        Transform user features to normalized format.
        
        Returns:
            dict: Dictionary mapping user_id -> feature vector
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        user_features = {}
        
        for _, row in user_df.iterrows():
            user_id = str(row['user_id'])
            features = {}
            
            # Age (normalized)
            age_norm = self.age_scaler.transform([[row['age']]])[0, 0]
            age_idx = self.feature_offset['user_features']
            features[age_idx] = age_norm
            
            # Gender (one-hot)
            gender_encoded = self.gender_encoder.transform([row['gender']])[0]
            gender_idx = self.feature_offset['user_features'] + 1 + gender_encoded
            features[gender_idx] = 1.0
            
            # Occupation (one-hot)
            occ_encoded = self.occupation_encoder.transform([row['occupation']])[0]
            n_genders = len(self.gender_encoder.classes_)
            occ_idx = self.feature_offset['user_features'] + 1 + n_genders + occ_encoded
            features[occ_idx] = 1.0
            
            user_features[user_id] = features
        
        return user_features
    
    def transform_item_features(self, item_df):
        """
        Transform item features to normalized format.
        
        Returns:
            dict: Dictionary mapping item_id -> feature vector
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        item_features = {}
        genre_cols = [col for col in item_df.columns if col.startswith('genre_')]
        
        for _, row in item_df.iterrows():
            item_id = str(row['item_id'])
            features = {}
            
            # Release year (normalized)
            year_norm = self.year_scaler.transform([[row['release_year']]])[0, 0]
            year_idx = self.feature_offset['item_features']
            features[year_idx] = year_norm
            
            # Genres (multi-hot)
            for i, genre_col in enumerate(genre_cols):
                if row[genre_col] == 1:
                    genre_idx = self.feature_offset['item_features'] + 1 + i
                    features[genre_idx] = 1.0
            
            item_features[item_id] = features
        
        return item_features
    
    def to_sparse_matrix(self, ratings_data, user_features_dict, item_features_dict):
        """
        Convert ratings with features to sparse matrix format for myFM.
        
        Args:
            ratings_data: Either a Surprise Trainset or list of (uid, iid, rating) tuples
            user_features_dict: Dict mapping user_id -> feature dict
            item_features_dict: Dict mapping item_id -> feature dict
            
        Returns:
            tuple: (X_sparse, y) where:
                - X_sparse: scipy.sparse.csr_matrix of shape (n_samples, n_features)
                - y: numpy array of ratings
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before converting to sparse matrix")
        
        # Collect all ratings and build sparse matrix
        rows = []
        cols = []
        data = []
        ratings = []
        
        # Handle both Trainset and list of tuples
        if hasattr(ratings_data, 'all_ratings'):
            # Surprise Trainset
            rating_list = list(ratings_data.all_ratings())
        else:
            # List of tuples (uid, iid, rating)
            rating_list = ratings_data
        
        for row_idx, rating_entry in enumerate(rating_list):
            if hasattr(ratings_data, 'all_ratings'):
                inner_uid, inner_iid, rating = rating_entry
                uid = ratings_data.to_raw_uid(inner_uid)
                iid = ratings_data.to_raw_iid(inner_iid)
            else:
                uid, iid, rating = rating_entry
                uid = str(uid)
                iid = str(iid)
            
            ratings.append(rating)
            
            # User ID (one-hot)
            if uid in self.user_id_map:
                user_idx = self.feature_offset['user_id'] + self.user_id_map[uid]
                rows.append(row_idx)
                cols.append(user_idx)
                data.append(1.0)
            
            # Item ID (one-hot)
            if iid in self.item_id_map:
                item_idx = self.feature_offset['item_id'] + self.item_id_map[iid]
                rows.append(row_idx)
                cols.append(item_idx)
                data.append(1.0)
            
            # User features
            if uid in user_features_dict:
                for feat_idx, feat_val in user_features_dict[uid].items():
                    rows.append(row_idx)
                    cols.append(feat_idx)
                    data.append(feat_val)
            
            # Item features
            if iid in item_features_dict:
                for feat_idx, feat_val in item_features_dict[iid].items():
                    rows.append(row_idx)
                    cols.append(feat_idx)
                    data.append(feat_val)
        
        # Build sparse matrix
        n_samples = len(ratings)
        n_features = self.total_features
        X_sparse = sp.csr_matrix((data, (rows, cols)), shape=(n_samples, n_features))
        y = np.array(ratings)
        
        return X_sparse, y
    
    def to_libsvm_format(self, ratings_data, user_features_dict, item_features_dict, output_file):
        """
        Convert ratings with features to libsvm format (deprecated).
        
        This method is kept for backward compatibility but is deprecated.
        Use to_sparse_matrix() instead for myFM.
        
        Format: label feature1:value1 feature2:value2 ...
        
        Args:
            ratings_data: Either a Surprise Trainset or list of (uid, iid, rating) tuples
            user_features_dict: Dict mapping user_id -> feature dict
            item_features_dict: Dict mapping item_id -> feature dict
            output_file: Path to output libsvm file
        """
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w') as f:
            # Handle both Trainset and list of tuples
            if hasattr(ratings_data, 'all_ratings'):
                # Surprise Trainset
                for inner_uid, inner_iid, rating in ratings_data.all_ratings():
                    uid = ratings_data.to_raw_uid(inner_uid)
                    iid = ratings_data.to_raw_iid(inner_iid)
                    self._write_libsvm_line(f, uid, iid, rating, user_features_dict, item_features_dict)
            else:
                # List of tuples (uid, iid, rating)
                for uid, iid, rating in ratings_data:
                    self._write_libsvm_line(f, str(uid), str(iid), rating, user_features_dict, item_features_dict)
    
    def _write_libsvm_line(self, f, uid, iid, rating, user_features_dict, item_features_dict):
        """Write a single line in libsvm format."""
        # Start with label (rating)
        line_parts = [str(rating)]
        
        # User ID (one-hot)
        if uid in self.user_id_map:
            user_idx = self.feature_offset['user_id'] + self.user_id_map[uid]
            line_parts.append(f"{user_idx}:1")
        
        # Item ID (one-hot)
        if iid in self.item_id_map:
            item_idx = self.feature_offset['item_id'] + self.item_id_map[iid]
            line_parts.append(f"{item_idx}:1")
        
        # User features
        if uid in user_features_dict:
            for feat_idx, feat_val in user_features_dict[uid].items():
                line_parts.append(f"{feat_idx}:{feat_val:.6f}")
        
        # Item features
        if iid in item_features_dict:
            for feat_idx, feat_val in item_features_dict[iid].items():
                line_parts.append(f"{feat_idx}:{feat_val:.6f}")
        
        f.write(' '.join(line_parts) + '\n')


def get_cold_start_split(data, test_size=0.2, random_state=42,
                         cold_user_threshold=17, cold_item_threshold=10):
    """
    Create train/test split with explicit cold start subset.
    
    Args:
        data: List of (uid, iid, rating) tuples
        user_features: DataFrame with user features
        item_features: DataFrame with item features
        cold_start_ratio: Fraction of test set to reserve for cold start (default: 0.1, not used for user-based definition)
        test_size: Fraction of data for testing (default: 0.2)
        random_state: Random seed (default: 42)
        cold_user_threshold: Maximum number of ratings in training to be considered cold start user (default: 5)
        cold_item_threshold: Maximum number of ratings in training to be considered cold start item (default: 10)
        
    Returns:
        tuple: (trainset, testset, cold_start_users, cold_start_items)
        - trainset: SimpleTrainset
        - testset: List of (uid, iid, rating) tuples
        - cold_start_users: Set of user IDs with <cold_user_threshold ratings in training
        - cold_start_items: Set of item IDs with <cold_item_threshold ratings in training
    """
    trainset, testset = get_train_test_split(data, test_size=test_size, random_state=random_state)
    
    # Get users and items in training set, count ratings per user/item
    train_users = set()
    train_items = set()
    user_rating_counts = defaultdict(int)
    item_rating_counts = defaultdict(int)
    
    for inner_uid, inner_iid, _ in trainset.all_ratings():
        uid = trainset.to_raw_uid(inner_uid)
        iid = trainset.to_raw_iid(inner_iid)
        train_users.add(uid)
        train_items.add(iid)
        user_rating_counts[uid] += 1
        item_rating_counts[iid] += 1
    
    # Identify cold start users (users with <cold_user_threshold ratings in training)
    cold_start_users = {uid for uid, count in user_rating_counts.items() 
                       if count < cold_user_threshold}
    
    # Intersect with test set users to only include users that appear in test
    all_test_users = set(uid for uid, _, _ in testset)
    cold_start_users = cold_start_users & all_test_users
    
    # Identify cold start items (few ratings in training)
    cold_start_items = {iid for iid, count in item_rating_counts.items() 
                       if count < cold_item_threshold}
    
    # Intersect with test items
    all_test_items = set(iid for _, iid, _ in testset)
    cold_start_items = cold_start_items & all_test_items
    
    return trainset, testset, cold_start_users, cold_start_items


def find_cold_start_threshold(data, user_features, item_features, 
                              target_count=500, test_size=0.2, random_state=42,
                              max_threshold=200, verbose=True):
    """
    Find the cold_user_threshold that yields approximately target_count cold start users.
    
    Args:
        data: List of (uid, iid, rating) tuples
        user_features: DataFrame with user features
        item_features: DataFrame with item features
        target_count (int): Target number of cold start users (default: 500)
        test_size (float): Fraction of data for testing (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
        max_threshold (int): Maximum threshold to test (default: 200)
        verbose (bool): Print detailed analysis (default: True)
        
    Returns:
        int: Recommended cold_user_threshold value
    """
    trainset, testset = get_train_test_split(data, test_size=test_size, random_state=random_state)
    
    # Count ratings per user in training set
    user_rating_counts = defaultdict(int)
    for inner_uid, inner_iid, _ in trainset.all_ratings():
        uid = trainset.to_raw_uid(inner_uid)
        user_rating_counts[uid] += 1
    
    # Get test set users
    all_test_users = set(uid for uid, _, _ in testset)
    
    # Test different thresholds
    if verbose:
        print("=" * 60)
        print("Finding Cold Start User Threshold")
        print("=" * 60)
        print(f"Target cold start users: {target_count}")
        print(f"Total users in training: {len(user_rating_counts)}")
        print(f"Total users in test: {len(all_test_users)}")
        print("\nTesting thresholds...")
        print("-" * 60)
    
    best_threshold = None
    best_count = None
    min_diff = float('inf')
    
    # Test thresholds from 1 to max_threshold
    results = []
    for threshold in range(1, max_threshold + 1):
        # Count users with < threshold ratings in training
        cold_candidates = {uid for uid, count in user_rating_counts.items() 
                          if count < threshold}
        # Intersect with test set users
        cold_start_users = cold_candidates & all_test_users
        count = len(cold_start_users)
        
        diff = abs(count - target_count)
        if diff < min_diff:
            min_diff = diff
            best_threshold = threshold
            best_count = count
        
        # Store results for every 5th threshold or near target
        if threshold % 5 == 0 or abs(count - target_count) < 50:
            results.append((threshold, count))
    
    if verbose:
        print(f"\nThreshold | Cold Start Users")
        print("-" * 30)
        for threshold, count in results[:20]:  # Show first 20 results
            marker = " <-- CLOSEST" if threshold == best_threshold else ""
            print(f"   {threshold:3d}   |      {count:4d}{marker}")
        if len(results) > 20:
            print(f"   ...")
            print(f"   {best_threshold:3d}   |      {best_count:4d} <-- BEST")
        
        print("\n" + "=" * 60)
        print(f"Recommended threshold: {best_threshold}")
        print(f"  -> Yields {best_count} cold start users")
        print(f"  -> Difference from target: {abs(best_count - target_count)} users")
        print("=" * 60)
        
        # Show distribution statistics
        rating_counts_list = list(user_rating_counts.values())
        print(f"\nRating Distribution Statistics:")
        print(f"  Min ratings per user: {min(rating_counts_list)}")
        print(f"  Max ratings per user: {max(rating_counts_list)}")
        print(f"  Mean ratings per user: {np.mean(rating_counts_list):.2f}")
        print(f"  Median ratings per user: {np.median(rating_counts_list):.2f}")
        print(f"  25th percentile: {np.percentile(rating_counts_list, 25):.2f}")
        print(f"  75th percentile: {np.percentile(rating_counts_list, 75):.2f}")
    
    return best_threshold


if __name__ == "__main__":
    # Example usage
    print("Loading MovieLens 100K dataset...")
    data = load_movielens_100k()
    
    # Display statistics
    get_dataset_stats(data)
    
    # Load features
    print("\nLoading user and item features...")
    user_features = load_user_features()
    item_features = load_item_features()
    print(f"User features shape: {user_features.shape}")
    print(f"Item features shape: {item_features.shape}")
    print(f"\nUser features columns: {list(user_features.columns)}")
    print(f"Item features columns: {list(item_features.columns[:10])}...")
    
    # Split into train/test
    trainset, testset = get_train_test_split(data, test_size=0.2, random_state=42)
    
    print(f"\nTraining set size: {trainset.n_ratings:,} ratings")
    print(f"Test set size: {len(testset):,} ratings")
    print(f"Number of users in training: {trainset.n_users}")
    print(f"Number of items in training: {trainset.n_items}")
    
    # Test feature preprocessing
    print("\nTesting feature preprocessing...")
    preprocessor = FeaturePreprocessor()
    preprocessor.fit(user_features, item_features)
    print(f"Total features: {preprocessor.total_features}")
    print(f"Feature offsets: {preprocessor.feature_offset}")
    
    # Test cold start split
    print("\nTesting cold start split...")
    trainset_cs, testset_cs, cold_users, cold_items = get_cold_start_split(data)
    print(f"Cold start users: {len(cold_users)}")
    print(f"Cold start items: {len(cold_items)}")
