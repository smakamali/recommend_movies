"""
Model artifact management for versioning and deployment.

Handles saving and loading of trained models, preprocessors, and metadata.
"""

import os
import json
import pickle
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional


def save_model_artifacts(
    model: torch.nn.Module,
    preprocessor: Any,
    metadata: Dict[str, Any],
    output_dir: str = "models/current"
) -> None:
    """
    Save trained model and associated artifacts.
    
    Args:
        model: Trained PyTorch model
        preprocessor: Fitted FeaturePreprocessor
        metadata: Dictionary with training metadata
        output_dir: Directory to save artifacts
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model weights
    model_path = os.path.join(output_dir, "graphsage_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")
    
    # Save preprocessor
    preprocessor_path = os.path.join(output_dir, "preprocessor.pkl")
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Saved preprocessor to: {preprocessor_path}")
    
    # Add timestamp to metadata
    metadata['saved_at'] = datetime.now().isoformat()
    metadata['artifact_paths'] = {
        'model': model_path,
        'preprocessor': preprocessor_path,
        'metadata': os.path.join(output_dir, "metadata.json")
    }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")
    
    print(f"\n[SUCCESS] All artifacts saved to: {output_dir}")


def load_model_artifacts(
    model_class: type,
    model_dir: str = "models/current",
    device: str = 'cpu'
) -> Tuple[torch.nn.Module, Any, Dict[str, Any]]:
    """
    Load trained model and associated artifacts.
    
    Args:
        model_class: Model class to instantiate
        model_dir: Directory containing artifacts
        device: Device to load model on ('cpu' or 'cuda')
        
    Returns:
        Tuple of (model, preprocessor, metadata)
    """
    # Load metadata
    metadata_path = os.path.join(model_dir, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load preprocessor
    preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Initialize model with saved hyperparameters
    hyperparams = metadata.get('hyperparameters', {})
    
    # Filter to only model constructor arguments (exclude training-specific params)
    model_constructor_params = {
        'num_users': hyperparams.get('num_users'),
        'num_items': hyperparams.get('num_items'),
        'user_feat_dim': hyperparams.get('user_feat_dim'),
        'item_feat_dim': hyperparams.get('item_feat_dim'),
        'hidden_dim': hyperparams.get('hidden_dim'),
        'num_layers': hyperparams.get('num_layers'),
        'dropout': hyperparams.get('dropout'),
        'aggregator': hyperparams.get('aggregator')
    }
    
    model = model_class(**model_constructor_params)
    
    # Load model weights
    model_path = os.path.join(model_dir, "graphsage_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"[SUCCESS] Loaded model from: {model_dir}")
    print(f"  Version: {metadata.get('model_version', 'unknown')}")
    print(f"  Training date: {metadata.get('training_date', 'unknown')}")
    print(f"  Val RMSE: {metadata.get('metrics', {}).get('val_rmse', 'N/A')}")
    
    return model, preprocessor, metadata


def get_latest_model_path() -> str:
    """
    Get path to latest model directory.
    
    Returns:
        Path to models/current directory
    """
    return "models/current"


def create_metadata(
    model_version: str,
    num_users: int,
    num_movies: int,
    num_ratings: int,
    hyperparameters: Dict[str, Any],
    metrics: Dict[str, float],
    additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create metadata dictionary for model artifact.
    
    Args:
        model_version: Version string (e.g., "1.0.0")
        num_users: Number of users in training
        num_movies: Number of movies in training
        num_ratings: Number of ratings in training
        hyperparameters: Model hyperparameters
        metrics: Validation metrics
        additional_info: Optional additional metadata
        
    Returns:
        Complete metadata dictionary
    """
    metadata = {
        'model_version': model_version,
        'training_date': datetime.now().isoformat(),
        'num_users_trained': num_users,
        'num_movies': num_movies,
        'num_ratings': num_ratings,
        'hyperparameters': hyperparameters,
        'metrics': metrics
    }
    
    if additional_info:
        metadata.update(additional_info)
    
    return metadata


def verify_artifacts_exist(model_dir: str = "models/current") -> bool:
    """
    Verify that all required artifacts exist.
    
    Args:
        model_dir: Directory to check
        
    Returns:
        True if all artifacts exist, False otherwise
    """
    required_files = [
        "graphsage_model.pth",
        "preprocessor.pkl",
        "metadata.json"
    ]
    
    for filename in required_files:
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            print(f"[ERROR] Missing artifact: {path}")
            return False
    
    print(f"[SUCCESS] All artifacts verified in: {model_dir}")
    return True


if __name__ == "__main__":
    # Test artifact verification
    print("Checking for existing model artifacts...")
    if verify_artifacts_exist():
        print("Model artifacts found!")
    else:
        print("No model artifacts found. Train a model first.")
