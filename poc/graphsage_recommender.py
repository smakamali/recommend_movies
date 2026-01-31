"""
GraphSAGE Recommender Interface.

Provides Surprise-compatible interface for evaluation.
Implements fit() and test() methods compatible with evaluation.py.
"""

import torch
import numpy as np
from surprise import Prediction

# Handle both relative imports (when used as module) and absolute imports (when run as script)
try:
    from .graph_data_loader import build_bipartite_graph
    from .graphsage_model import GraphSAGERecommender
    from .train_graphsage import train_graphsage_model
except ImportError:
    # Running as script, use absolute imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from gnn.graph_data_loader import build_bipartite_graph
    from gnn.graphsage_model import GraphSAGERecommender
    from gnn.train_graphsage import train_graphsage_model


class GraphSAGERecommenderWrapper:
    """
    High-level wrapper for GraphSAGE recommender.
    
    Provides interface compatible with Surprise and evaluation.py:
    - fit(trainset, user_features, item_features): Train model
    - predict(user_id, item_id): Predict single rating
    - test(testset): Return Surprise Prediction format
    """
    
    def __init__(self, hidden_dim=64, num_layers=2, dropout=0.1,
                 aggregator='max', num_epochs=20, batch_size=512,
                 learning_rate=0.001, num_negatives=1, device='cpu',
                 random_seed=42, loss_type='mse', mse_weight=1.0, bpr_weight=0.1,
                 val_ratio=0.1, early_stopping_patience=5, early_stopping_min_delta=1e-4):
        """
        Initialize GraphSAGE recommender.
        
        Args:
            hidden_dim: Hidden dimension for embeddings (default: 64)
            num_layers: Number of GraphSAGE layers (default: 2)
            dropout: Dropout rate (default: 0.1)
            aggregator: Aggregator type - 'mean' or 'max' (default: 'max')
            num_epochs: Number of training epochs (default: 20)
            batch_size: Batch size for training (default: 512)
            learning_rate: Learning rate (default: 0.001)
            num_negatives: Number of negative samples per positive (default: 1, used for BPR/combined)
            device: Device to train on (default: 'cpu')
            random_seed: Random seed for reproducibility (default: 42)
            loss_type: Loss function - 'mse', 'bpr', or 'combined' (default: 'mse')
            mse_weight: Weight for MSE component in combined loss (default: 1.0)
            bpr_weight: Weight for BPR component in combined loss (default: 0.1)
            val_ratio: Ratio of training data to use for validation (default: 0.1)
            early_stopping_patience: Number of epochs to wait without improvement before stopping (default: 5)
            early_stopping_min_delta: Minimum change in validation loss to qualify as improvement (default: 1e-4)
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregator = aggregator
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_negatives = num_negatives
        self.device = device
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.mse_weight = mse_weight
        self.bpr_weight = bpr_weight
        self.val_ratio = val_ratio
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        
        # Set random seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Model components (initialized in fit())
        self.model = None
        self.graph_data = None
        self.preprocessor = None
        self.user_id_to_idx = None
        self.item_id_to_idx = None
        self.idx_to_user_id = None
        self.idx_to_item_id = None
        self.trainset = None
        self.user_embeddings = None
        self.item_embeddings = None
        
    def fit(self, trainset, user_features, item_features, verbose=True):
        """
        Fit GraphSAGE model on training data.
        
        Args:
            trainset: Surprise Trainset
            user_features: DataFrame with user features
            item_features: DataFrame with item features
            verbose: Print training progress (default: True)
        """
        if verbose:
            print("Building bipartite graph...")
        
        # Build graph
        self.graph_data, self.preprocessor, self.user_id_to_idx, self.item_id_to_idx = \
            build_bipartite_graph(trainset, user_features, item_features)
        
        # Create reverse mappings
        self.idx_to_user_id = {idx: uid for uid, idx in self.user_id_to_idx.items()}
        self.idx_to_item_id = {idx: iid for iid, idx in self.item_id_to_idx.items()}
        
        self.trainset = trainset
        
        # Get feature dimensions
        user_feat_dim = self.graph_data.x[self.graph_data.node_type == 0].size(1)
        item_feat_dim = self.graph_data.x[self.graph_data.node_type == 1].size(1)
        
        # Initialize model
        if verbose:
            print("Initializing GraphSAGE model...")
        
        self.model = GraphSAGERecommender(
            num_users=self.graph_data.num_users,
            num_items=self.graph_data.num_items,
            user_feat_dim=user_feat_dim,
            item_feat_dim=item_feat_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            aggregator=self.aggregator
        )
        
        # Train model
        if verbose:
            print("Training GraphSAGE model...")
        
        history = train_graphsage_model(
            self.model,
            self.graph_data,
            trainset,
            self.user_id_to_idx,
            self.item_id_to_idx,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            num_negatives=self.num_negatives,
            device=self.device,
            verbose=verbose,
            loss_type=self.loss_type,
            mse_weight=self.mse_weight,
            bpr_weight=self.bpr_weight,
            val_ratio=self.val_ratio,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_min_delta=self.early_stopping_min_delta
        )
        
        # Get final embeddings for inference
        self.model.eval()
        with torch.no_grad():
            self.user_embeddings, self.item_embeddings = self.model(self.graph_data)
        
        if verbose:
            print("Model training completed!")
    
    def predict(self, user_id, item_id):
        """
        Predict rating for a single user-item pair.
        
        Args:
            user_id: User ID (string)
            item_id: Item ID (string)
            
        Returns:
            float: Predicted rating (clipped to [1, 5])
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert string IDs to indices
        user_idx = self.user_id_to_idx.get(str(user_id))
        item_idx = self.item_id_to_idx.get(str(item_id))
        
        if user_idx is None or item_idx is None:
            # Cold start: return average rating (3.0) or use feature-based prediction
            # For now, return neutral rating
            return 3.0
        
        # Compute prediction with rating head
        with torch.no_grad():
            score = self.model.predict(
                self.user_embeddings, 
                self.item_embeddings,
                user_idx, 
                item_idx,
                use_rating_head=True
            ).item()
        
        # Already clamped in model, but ensure valid range
        score = max(1.0, min(5.0, score))
        return float(score)
    
    def test(self, testset):
        """
        Test on testset (returns Surprise-compatible predictions).
        
        This method is critical for compatibility with evaluation.py.
        
        Args:
            testset: List of (uid, iid, true_rating) tuples
            
        Returns:
            list: List of Prediction objects compatible with Surprise
        """
        if self.model is None:
            raise ValueError("Model must be fitted before testing")
        
        predictions = []
        
        for uid, iid, true_r in testset:
            # Predict rating
            pred_r = self.predict(str(uid), str(iid))
            
            # Create Surprise Prediction object
            # Format: Prediction(uid, iid, r_ui, est, details)
            pred = Prediction(uid, iid, true_r, pred_r, {})
            predictions.append(pred)
        
        return predictions


def train_graphsage_recommender(trainset, user_features, item_features,
                                hidden_dim=64, num_layers=2, dropout=0.1,
                                aggregator='max', num_epochs=20, batch_size=512,
                                learning_rate=0.001, num_negatives=1, device='cuda',
                                random_seed=42, verbose=True, loss_type='mse',
                                mse_weight=1.0, bpr_weight=0.1,
                                val_ratio=0.1, early_stopping_patience=5, 
                                early_stopping_min_delta=1e-4):
    """
    Convenience function to train GraphSAGE recommender.
    
    Similar to train_fm_model() for consistency.
    
    Args:
        trainset: Surprise Trainset
        user_features: DataFrame with user features
        item_features: DataFrame with item features
        hidden_dim: Hidden dimension (default: 64)
        num_layers: Number of layers (default: 2)
        dropout: Dropout rate (default: 0.1)
        aggregator: Aggregator type (default: 'max')
        num_epochs: Training epochs (default: 20)
        batch_size: Batch size (default: 512)
        learning_rate: Learning rate (default: 0.001)
        num_negatives: Negative samples (default: 1, used for BPR/combined)
        device: Device (default: 'cuda')
        random_seed: Random seed (default: 42)
        verbose: Print progress (default: True)
        loss_type: Loss function - 'mse', 'bpr', or 'combined' (default: 'mse')
        mse_weight: Weight for MSE component in combined loss (default: 1.0)
        bpr_weight: Weight for BPR component in combined loss (default: 0.1)
        val_ratio: Ratio of training data to use for validation (default: 0.1)
        early_stopping_patience: Number of epochs to wait without improvement before stopping (default: 5)
        early_stopping_min_delta: Minimum change in validation loss to qualify as improvement (default: 1e-4)
        
    Returns:
        GraphSAGERecommenderWrapper: Trained recommender
    """
    recommender = GraphSAGERecommenderWrapper(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        aggregator=aggregator,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_negatives=num_negatives,
        device=device,
        random_seed=random_seed,
        loss_type=loss_type,
        mse_weight=mse_weight,
        bpr_weight=bpr_weight,
        val_ratio=val_ratio,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta
    )
    
    recommender.fit(trainset, user_features, item_features, verbose=verbose)
    
    return recommender


if __name__ == "__main__":
    # Example usage (run from repo root: python gnn/graphsage_recommender.py or python -m gnn.graphsage_recommender)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from gnn.graph_data_loader import build_bipartite_graph
    from gnn.graphsage_model import GraphSAGERecommender
    from gnn.train_graphsage import train_graphsage_model
    from poc.data_loader import (load_movielens_100k, get_train_test_split,
                                load_user_features, load_item_features)
    from poc.evaluation import evaluate_model
    
    print("Loading MovieLens 100K dataset...")
    data = load_movielens_100k()
    trainset, testset = get_train_test_split(data, test_size=0.2, random_state=42)
    
    print("Loading features...")
    user_features = load_user_features()
    item_features = load_item_features()
    
    print("Training GraphSAGE model...")
    model = train_graphsage_recommender(
        trainset, user_features, item_features,
        hidden_dim=64, num_layers=2, num_epochs=5,  # Short training for testing
        verbose=True
    )
    
    print("\nTesting predictions...")
    sample_user, sample_item, true_rating = testset[0]
    pred_rating = model.predict(sample_user, sample_item)
    print(f"User {sample_user}, Item {sample_item}:")
    print(f"  True rating: {true_rating:.2f}")
    print(f"  Predicted: {pred_rating:.2f}")
    print(f"  Error: {abs(true_rating - pred_rating):.2f}")
    
    print("\nTesting evaluation compatibility...")
    predictions = model.test(testset[:100])  # Test on first 100 for speed
    results = evaluate_model(predictions, k=10, threshold=4.0, verbose=True)
    
    print("\nGraphSAGE recommender test successful!")
