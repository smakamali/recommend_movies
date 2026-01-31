"""
Graph Data Loader for GraphSAGE.

Transforms MovieLens data into PyTorch Geometric graph format.
Handles bipartite graph construction with user and item features.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from collections import defaultdict
from poc.data_loader import FeaturePreprocessor


def build_bipartite_graph(trainset, user_features_df, item_features_df, 
                          preprocessor=None):
    """
    Build bipartite user-item graph from MovieLens data.
    
    Creates a PyTorch Geometric Data object with:
    - Nodes: Users (0 to num_users-1) + Items (num_users to num_users+num_items-1)
    - Edges: User-item interactions (bidirectional for message passing)
    - Node features: User and item feature vectors
    
    Args:
        trainset: Surprise Trainset object
        user_features_df: DataFrame with user features (age, gender, occupation)
        item_features_df: DataFrame with item features (year, genres)
        preprocessor: Optional FeaturePreprocessor (will create if None)
        
    Returns:
        tuple: (graph_data, preprocessor, user_id_to_idx, item_id_to_idx)
            - graph_data: PyTorch Geometric Data object
            - preprocessor: FeaturePreprocessor instance
            - user_id_to_idx: Dict mapping user_id string -> node index
            - item_id_to_idx: Dict mapping item_id string -> node index
    """
    # Initialize or use provided preprocessor
    if preprocessor is None:
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(user_features_df, item_features_df)
    
    # Transform features to normalized format
    user_features_dict = preprocessor.transform_user_features(user_features_df)
    item_features_dict = preprocessor.transform_item_features(item_features_df)
    
    # Create ID mappings
    user_id_to_idx = preprocessor.user_id_map.copy()
    item_id_to_idx = preprocessor.item_id_map.copy()
    
    num_users = len(user_id_to_idx)
    num_items = len(item_id_to_idx)
    
    # Build edge list from trainset
    edge_list = []
    edge_weights = []  # Optional: can use rating values
    
    for inner_uid, inner_iid, rating in trainset.all_ratings():
        uid = trainset.to_raw_uid(inner_uid)
        iid = trainset.to_raw_iid(inner_iid)
        
        # Skip if user/item not in training set
        if uid not in user_id_to_idx or iid not in item_id_to_idx:
            continue
        
        user_idx = user_id_to_idx[uid]
        item_idx = item_id_to_idx[iid] + num_users  # Offset item indices
        
        # Add bidirectional edges for message passing
        # User -> Item
        edge_list.append([user_idx, item_idx])
        edge_weights.append(rating)
        
        # Item -> User (reverse direction)
        edge_list.append([item_idx, user_idx])
        edge_weights.append(rating)
    
    # Convert to tensor
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
    
    # Build feature matrices
    # User features: age (1) + gender (2) + occupation (n_occ)
    n_genders = len(preprocessor.gender_encoder.classes_)
    n_occupations = len(preprocessor.occupation_encoder.classes_)
    user_feat_dim = 1 + n_genders + n_occupations  # age + gender + occupation
    
    # Item features: year (1) + genres (19)
    item_feat_dim = 1 + 19  # year + genres
    
    # Initialize feature matrices
    user_feat_matrix = torch.zeros(num_users, user_feat_dim, dtype=torch.float)
    item_feat_matrix = torch.zeros(num_items, item_feat_dim, dtype=torch.float)
    
    # Fill user features
    for uid, features in user_features_dict.items():
        if uid in user_id_to_idx:
            user_idx = user_id_to_idx[uid]
            feat_vector = _dict_to_feature_vector(
                features, 
                preprocessor.feature_offset['user_features'],
                user_feat_dim
            )
            user_feat_matrix[user_idx] = feat_vector
    
    # Fill item features
    for iid, features in item_features_dict.items():
        if iid in item_id_to_idx:
            item_idx = item_id_to_idx[iid]
            feat_vector = _dict_to_feature_vector(
                features,
                preprocessor.feature_offset['item_features'],
                item_feat_dim
            )
            item_feat_matrix[item_idx] = feat_vector
    
    # Combine user and item features
    # Shape: (num_users + num_items, max(user_feat_dim, item_feat_dim))
    # Pad to same dimension if needed
    max_feat_dim = max(user_feat_dim, item_feat_dim)
    
    if user_feat_dim < max_feat_dim:
        padding = torch.zeros(num_users, max_feat_dim - user_feat_dim)
        user_feat_matrix = torch.cat([user_feat_matrix, padding], dim=1)
    
    if item_feat_dim < max_feat_dim:
        padding = torch.zeros(num_items, max_feat_dim - item_feat_dim)
        item_feat_matrix = torch.cat([item_feat_matrix, padding], dim=1)
    
    # Concatenate user and item features
    x = torch.cat([user_feat_matrix, item_feat_matrix], dim=0)
    
    # Create node type mask (0 for users, 1 for items)
    node_type = torch.cat([
        torch.zeros(num_users, dtype=torch.long),
        torch.ones(num_items, dtype=torch.long)
    ])
    
    # Create PyTorch Geometric Data object
    graph_data = Data(
        x=x,  # Node features
        edge_index=edge_index,  # Graph connectivity
        edge_attr=edge_attr,  # Edge weights (ratings)
        node_type=node_type,  # 0=user, 1=item
        num_users=num_users,
        num_items=num_items
    )
    
    return graph_data, preprocessor, user_id_to_idx, item_id_to_idx


def _dict_to_feature_vector(features_dict, feature_offset, feat_dim):
    """
    Convert feature dictionary to dense feature vector.
    
    Args:
        features_dict: Dict mapping feature_idx -> value
        feature_offset: Starting index for this feature group
        feat_dim: Dimension of feature vector
        
    Returns:
        torch.Tensor: Feature vector of shape (feat_dim,)
    """
    feat_vector = torch.zeros(feat_dim, dtype=torch.float)
    
    for feat_idx, feat_val in features_dict.items():
        # Map global feature index to local index
        local_idx = feat_idx - feature_offset
        if 0 <= local_idx < feat_dim:
            feat_vector[local_idx] = feat_val
    
    return feat_vector


def get_user_item_masks(num_users, num_items, device='cpu'):
    """
    Get boolean masks for users and items in the graph.
    
    Args:
        num_users: Number of users
        num_items: Number of items
        device: Device to place tensors on
        
    Returns:
        tuple: (user_mask, item_mask)
            - user_mask: Boolean tensor, True for user nodes
            - item_mask: Boolean tensor, True for item nodes
    """
    user_mask = torch.zeros(num_users + num_items, dtype=torch.bool, device=device)
    user_mask[:num_users] = True
    
    item_mask = torch.zeros(num_users + num_items, dtype=torch.bool, device=device)
    item_mask[num_users:] = True
    
    return user_mask, item_mask


def build_negative_samples(trainset, user_id_to_idx, item_id_to_idx, 
                          num_users, num_items, num_negatives=1):
    """
    Build negative samples for BPR loss.
    
    For each positive (user, item) pair, sample negative items
    that the user hasn't interacted with.
    
    Args:
        trainset: Surprise Trainset
        user_id_to_idx: Dict mapping user_id -> node index
        item_id_to_idx: Dict mapping item_id -> node index
        num_users: Number of users
        num_items: Number of items
        num_negatives: Number of negative samples per positive (default: 1)
        
    Returns:
        dict: Mapping user_idx -> set of item indices (positive items)
    """
    # Build set of positive items per user
    user_positive_items = defaultdict(set)
    
    for inner_uid, inner_iid, _ in trainset.all_ratings():
        uid = trainset.to_raw_uid(inner_uid)
        iid = trainset.to_raw_iid(inner_iid)
        
        if uid not in user_id_to_idx or iid not in item_id_to_idx:
            continue
        
        user_idx = user_id_to_idx[uid]
        item_idx = item_id_to_idx[iid]
        user_positive_items[user_idx].add(item_idx)
    
    return user_positive_items


if __name__ == "__main__":
    # Test graph construction
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from poc.data_loader import (load_movielens_100k, get_train_test_split,
                                load_user_features, load_item_features)
    
    print("Loading MovieLens 100K dataset...")
    data = load_movielens_100k()
    trainset, testset = get_train_test_split(data, test_size=0.2, random_state=42)
    
    print("Loading features...")
    user_features = load_user_features()
    item_features = load_item_features()
    
    print("Building bipartite graph...")
    graph_data, preprocessor, user_id_to_idx, item_id_to_idx = build_bipartite_graph(
        trainset, user_features, item_features
    )
    
    print(f"\nGraph Statistics:")
    print(f"  Number of nodes: {graph_data.x.size(0)}")
    print(f"  Number of users: {graph_data.num_users}")
    print(f"  Number of items: {graph_data.num_items}")
    print(f"  Number of edges: {graph_data.edge_index.size(1)}")
    print(f"  Feature dimension: {graph_data.x.size(1)}")
    print(f"  Edge attributes: {graph_data.edge_attr.size()}")
    
    print(f"\nUser ID mapping: {len(user_id_to_idx)} users")
    print(f"Item ID mapping: {len(item_id_to_idx)} items")
    
    print("\nGraph construction successful!")
