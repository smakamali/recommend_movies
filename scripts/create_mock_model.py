"""
Create mock trained model artifacts for testing the inference engine.

This script creates placeholder model files when actual trained models
are not yet available from the training pipeline.
"""

import json
import pickle
import torch
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import from poc
sys.path.insert(0, str(Path(__file__).parent.parent))

from poc.graphsage_model import GraphSAGERecommender
from poc.data_loader import FeaturePreprocessor, load_user_features, load_item_features


def create_mock_model_artifacts(output_dir: str = "models/current"):
    """
    Create mock model artifacts for testing.
    
    Args:
        output_dir: Directory to save artifacts
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating mock model artifacts in {output_dir}...")
    
    # Load features to fit preprocessor
    print("  Loading features...")
    user_features = load_user_features()
    item_features = load_item_features()
    
    # Fit preprocessor
    print("  Fitting preprocessor...")
    preprocessor = FeaturePreprocessor()
    preprocessor.fit(user_features, item_features)
    
    # Save preprocessor
    preprocessor_path = output_path / "preprocessor.pkl"
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"    Saved: {preprocessor_path}")
    
    # Get feature dimensions
    n_genders = len(preprocessor.gender_encoder.classes_)
    n_occupations = len(preprocessor.occupation_encoder.classes_)
    user_feat_dim = 1 + n_genders + n_occupations
    item_feat_dim = 1 + 19  # year + 19 genres
    
    # Model configuration (from param tuning results)
    num_users = len(user_features)
    num_items = len(item_features)
    hidden_dim = 64
    num_layers = 3
    dropout = 0.1
    aggregator = 'max'
    
    print(f"  Creating model with:")
    print(f"    num_users={num_users}, num_items={num_items}")
    print(f"    user_feat_dim={user_feat_dim}, item_feat_dim={item_feat_dim}")
    print(f"    hidden_dim={hidden_dim}, num_layers={num_layers}")
    
    # Initialize model
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
    
    # Save model weights (random initialization)
    model_path = output_path / "graphsage_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"    Saved: {model_path}")
    print("    WARNING: Model uses random initialization (not trained)")
    
    # Create metadata
    metadata = {
        "model_version": "0.0.1-mock",
        "training_date": datetime.now().isoformat(),
        "num_users_trained": num_users,
        "num_movies": num_items,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "aggregator": aggregator,
        "loss_type": "MSE",
        "val_rmse": None,  # Not trained
        "val_precision@10": None,  # Not trained
        "hyperparameters": {
            "learning_rate": 0.001,
            "dropout": dropout,
            "batch_size": 512,
            "num_epochs": 0
        },
        "notes": "Mock model for testing - uses random initialization, NOT trained"
    }
    
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"    Saved: {metadata_path}")
    
    print("\nMock model artifacts created successfully!")
    print("\nNOTE: This is a MOCK model with random weights for testing only.")
    print("Recommendations will be random. Replace with trained model from Phase 5.")
    
    return {
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
        "metadata_path": str(metadata_path)
    }


if __name__ == "__main__":
    import sys
    
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "models/current"
    
    try:
        artifacts = create_mock_model_artifacts(output_dir)
        print(f"\nArtifacts created:")
        for name, path in artifacts.items():
            print(f"  {name}: {path}")
        
        print("\nYou can now test the inference engine!")
        
    except Exception as e:
        print(f"\nError creating mock artifacts: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
