"""
Main training script for GraphSAGE recommender model.

Loads data from database, trains model using POC code, and saves artifacts.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.preprocessing import MinMaxScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import POC training components
from poc.graphsage_model import GraphSAGERecommender
from poc.graph_data_loader import build_bipartite_graph
from poc.data_loader import FeaturePreprocessor
from poc.train_graphsage import train_graphsage_model
from poc.evaluation import evaluate_model

# Import our training components
from app.core.training.data_loader import load_training_data
from app.core.training.model_versioning import save_model_artifacts, create_metadata


def train_model(
    db_path: str = "data/recommender.db",
    hidden_dim: int = 64,
    num_layers: int = 3,
    aggregator: str = 'max',
    dropout: float = 0.1,
    loss_type: str = 'mse',
    learning_rate: float = 0.001,
    batch_size: int = 512,
    num_epochs: int = 50,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 1e-4,
    output_dir: str = "models/current",
    device: str = None,
    verbose: bool = True
) -> Tuple[GraphSAGERecommender, Dict[str, Any]]:
    """
    Train GraphSAGE recommender model.
    
    Args:
        db_path: Path to SQLite database
        hidden_dim: Hidden dimension size
        num_layers: Number of GraphSAGE layers
        aggregator: Aggregator type ('mean', 'max', 'lstm')
        dropout: Dropout rate
        loss_type: Loss function ('mse', 'mae', 'bce')
        learning_rate: Learning rate
        batch_size: Batch size
        num_epochs: Number of training epochs
        early_stopping_patience: Patience for early stopping
        early_stopping_min_delta: Minimum delta for early stopping
        output_dir: Directory to save model artifacts
        device: Device to train on ('cpu', 'cuda', or None for auto)
        verbose: Print training progress
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    if verbose:
        print("="*60)
        print("GraphSAGE Recommender - Training Pipeline")
        print("="*60)
    
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if verbose:
        print(f"\nDevice: {device}")
        print(f"Database: {db_path}")
    
    # Step 1: Load data from database
    if verbose:
        print("\n" + "-"*60)
        print("Step 1: Loading data from database")
        print("-"*60)
    
    data = load_training_data(db_path=db_path, test_size=0.2, random_state=42)
    trainset = data['trainset']
    testset = data['testset']
    user_features_df = data['user_features_df']
    item_features_df = data['item_features_df']
    
    num_users = trainset.n_users
    num_items = trainset.n_items
    num_ratings = data['num_ratings']
    
    if verbose:
        print(f"\nData loaded:")
        print(f"  Users: {num_users}")
        print(f"  Movies: {num_items}")
        print(f"  Ratings: {num_ratings:,}")
        print(f"  Train ratings: {trainset.n_ratings:,}")
        print(f"  Test ratings: {len(testset):,}")
    
    # Step 2: Build graph and preprocess features
    if verbose:
        print("\n" + "-"*60)
        print("Step 2: Building bipartite graph and preprocessing features")
        print("-"*60)
    
    graph_data, preprocessor, user_id_map, item_id_map = build_bipartite_graph(
        trainset,
        user_features_df,
        item_features_df,
        None  # Will create new preprocessor
    )
    
    # Get feature dimensions from preprocessor
    n_genders = len(preprocessor.gender_encoder.classes_)
    n_occupations = len(preprocessor.occupation_encoder.classes_)
    user_feat_dim_original = 1 + n_genders + n_occupations  # age + gender + occupation
    item_feat_dim_original = 1 + 19  # year + 19 genres
    
    # The graph uses combined features with max dimension (both padded to same size)
    feat_dim = max(user_feat_dim_original, item_feat_dim_original)
    
    # Since features are padded in the graph, both user and item features have the same dimension
    user_feat_dim = feat_dim
    item_feat_dim = feat_dim
    
    # Fit rating scaler on training ratings (1-5 -> 0-1)
    train_ratings = np.array([r for (_, _, r) in trainset.all_ratings()]).reshape(-1, 1)
    rating_scaler = MinMaxScaler(feature_range=(0, 1))
    rating_scaler.fit(train_ratings)
    
    if verbose:
        print(f"\nGraph constructed:")
        print(f"  User nodes: {graph_data.num_users}")
        print(f"  Item nodes: {graph_data.num_items}")
        print(f"  Edges: {graph_data.edge_index.shape[1]}")
        print(f"  Original user feature dim: {user_feat_dim_original}")
        print(f"  Original item feature dim: {item_feat_dim_original}")
        print(f"  Padded feature dim (both): {feat_dim}")
    
    # Step 3: Initialize model
    if verbose:
        print("\n" + "-"*60)
        print("Step 3: Initializing GraphSAGE model")
        print("-"*60)
    
    model = GraphSAGERecommender(
        num_users=num_users,
        num_items=num_items,
        user_feat_dim=user_feat_dim,
        item_feat_dim=item_feat_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        aggregator=aggregator
    )
    
    if verbose:
        print(f"\nModel architecture:")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Layers: {num_layers}")
        print(f"  Aggregator: {aggregator}")
        print(f"  Dropout: {dropout}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 4: Train model
    if verbose:
        print("\n" + "-"*60)
        print("Step 4: Training model")
        print("-"*60)
        print(f"\nHyperparameters:")
        print(f"  Loss type: {loss_type}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Early stopping patience: {early_stopping_patience}")
    
    training_history = train_graphsage_model(
        model=model,
        graph_data=graph_data,
        trainset=trainset,
        user_id_to_idx=user_id_map,
        item_id_to_idx=item_id_map,
        rating_scaler=rating_scaler,
        loss_type=loss_type,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        verbose=verbose
    )
    
    # Model is trained in-place
    trained_model = model
    
    # Step 5: Evaluate on test set
    if verbose:
        print("\n" + "-"*60)
        print("Step 5: Evaluating on test set")
        print("-"*60)
    
    # Generate predictions on test set
    trained_model.eval()
    predictions = []
    
    with torch.no_grad():
        # Get embeddings for all nodes
        user_emb, item_emb = trained_model(graph_data)
        
        for uid, iid, true_rating in testset:
            if uid not in user_id_map or iid not in item_id_map:
                continue
            
            user_idx = user_id_map[uid]
            item_idx = item_id_map[iid]
            
            # Get prediction using precomputed embeddings (model outputs 0-1)
            pred_scaled = trained_model.predict(
                user_emb,
                item_emb,
                user_idx,
                item_idx
            )
            # Inverse transform to rating scale 1-5
            pred_rating = float(
                rating_scaler.inverse_transform([[pred_scaled.item()]])[0, 0]
            )
            predictions.append((uid, iid, true_rating, pred_rating))
    
    # Evaluate predictions
    test_metrics = evaluate_model(predictions, k=10, threshold=4.0, verbose=verbose)
    
    if verbose:
        print(f"\nTest metrics:")
        print(f"  RMSE: {test_metrics.get('rmse', 'N/A'):.4f}")
        print(f"  MAE: {test_metrics.get('mae', 'N/A'):.4f}")
        if 'precision@10' in test_metrics:
            print(f"  Precision@10: {test_metrics.get('precision@10', 'N/A'):.4f}")
        if 'recall@10' in test_metrics:
            print(f"  Recall@10: {test_metrics.get('recall@10', 'N/A'):.4f}")
    
    # Step 6: Save model artifacts
    if verbose:
        print("\n" + "-"*60)
        print("Step 6: Saving model artifacts")
        print("-"*60)
    
    # Prepare hyperparameters for metadata
    hyperparameters = {
        'num_users': num_users,
        'num_items': num_items,
        'user_feat_dim': user_feat_dim,
        'item_feat_dim': item_feat_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout,
        'aggregator': aggregator,
        'loss_type': loss_type,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'early_stopping_patience': early_stopping_patience
    }
    
    # Prepare metrics for metadata
    metrics = {
        'val_rmse': test_metrics.get('rmse', 0.0),
        'val_mae': test_metrics.get('mae', 0.0),
        'val_precision_10': test_metrics.get('precision@10', 0.0),
        'val_recall_10': test_metrics.get('recall@10', 0.0),
        'final_train_loss': training_history['train_loss'][-1] if training_history['train_loss'] else 0.0,
        'best_val_loss': min(training_history['val_loss']) if training_history['val_loss'] else 0.0,
        'epochs_trained': len(training_history['train_loss'])
    }
    
    # Create metadata
    metadata = create_metadata(
        model_version="1.0.0",
        num_users=num_users,
        num_movies=num_items,
        num_ratings=num_ratings,
        hyperparameters=hyperparameters,
        metrics=metrics,
        additional_info={
            'training_history': {
                'train_loss': [float(x) for x in training_history['train_loss']],
                'val_loss': [float(x) for x in training_history['val_loss']]
            },
            'device': device,
            'database_path': db_path
        }
    )
    
    # Save artifacts
    save_model_artifacts(
        model=trained_model,
        preprocessor=preprocessor,
        metadata=metadata,
        rating_scaler=rating_scaler,
        output_dir=output_dir
    )
    
    if verbose:
        print("\n" + "="*60)
        print("[SUCCESS] Training complete!")
        print("="*60)
        print(f"\nFinal metrics:")
        print(f"  Test RMSE: {metrics['val_rmse']:.4f}")
        print(f"  Test Precision@10: {metrics['val_precision_10']:.4f}")
        print(f"\nModel artifacts saved to: {output_dir}")
    
    return trained_model, {
        'metrics': metrics,
        'history': training_history,
        'hyperparameters': hyperparameters
    }


if __name__ == "__main__":
    # Test training with small configuration
    print("Testing training pipeline...")
    model, results = train_model(
        num_epochs=5,
        batch_size=256,
        device='cuda',
        verbose=True
    )
    print("\n[SUCCESS] Training pipeline test complete!")
