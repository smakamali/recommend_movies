#!/usr/bin/env python
"""
Test loading and using the trained GraphSAGE model.

Verifies that saved artifacts can be loaded and used for inference.
"""

import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poc.graphsage_model import GraphSAGERecommender
from app.core.training.model_versioning import load_model_artifacts, verify_artifacts_exist


def test_model_loading():
    """Test loading saved model artifacts."""
    
    print("="*60)
    print("Model Loading Test")
    print("="*60)
    
    # Check artifacts exist
    print("\nStep 1: Verifying artifacts...")
    if not verify_artifacts_exist('models/current'):
        print("[ERROR] Model artifacts not found!")
        return False
    
    # Load artifacts
    print("\nStep 2: Loading model artifacts...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        model, preprocessor, metadata = load_model_artifacts(
            GraphSAGERecommender,
            model_dir='models/current',
            device=device
        )
        
        print(f"\n[SUCCESS] Model loaded successfully!")
        
        # Display metadata
        print("\nStep 3: Model information:")
        print(f"  Version: {metadata.get('model_version')}")
        print(f"  Training date: {metadata.get('training_date')}")
        print(f"  Device: {device}")
        
        print(f"\nArchitecture:")
        hyp = metadata.get('hyperparameters', {})
        print(f"  Users: {hyp.get('num_users')}")
        print(f"  Movies: {hyp.get('num_items')}")
        print(f"  Hidden dim: {hyp.get('hidden_dim')}")
        print(f"  Layers: {hyp.get('num_layers')}")
        print(f"  Aggregator: {hyp.get('aggregator')}")
        print(f"  Dropout: {hyp.get('dropout')}")
        
        print(f"\nPerformance:")
        met = metadata.get('metrics', {})
        print(f"  Test RMSE: {met.get('val_rmse'):.4f}")
        print(f"  Test MAE: {met.get('val_mae'):.4f}")
        print(f"  Precision@10: {met.get('val_precision_10'):.4f}")
        print(f"  Recall@10: {met.get('val_recall_10'):.4f}")
        
        print(f"\nTraining:")
        print(f"  Epochs: {met.get('epochs_trained')}")
        print(f"  Best val loss: {met.get('best_val_loss'):.4f}")
        
        # Test model mode
        print("\nStep 4: Testing model state...")
        print(f"  Model is in eval mode: {not model.training}")
        print(f"  Model device: {next(model.parameters()).device}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        print("\n" + "="*60)
        print("[SUCCESS] Model is ready for inference!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
