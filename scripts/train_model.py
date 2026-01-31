#!/usr/bin/env python
"""
CLI script for training GraphSAGE recommendation model.

Trains model on MovieLens 100K data from SQLite database and saves artifacts.

Usage:
    # Train with default parameters
    python scripts/train_model.py

    # Custom configuration
    python scripts/train_model.py --epochs 30 --batch-size 1024 --lr 0.001

    # CPU only
    python scripts/train_model.py --device cpu

Author: Agent 1 - GraphSAGE Recommender System
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.training.train import train_model


def main():
    """Main CLI entry point."""
    
    parser = argparse.ArgumentParser(
        description="Train GraphSAGE recommendation model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default parameters (recommended)
  python scripts/train_model.py

  # Custom epochs and batch size
  python scripts/train_model.py --epochs 30 --batch-size 1024

  # Force CPU training
  python scripts/train_model.py --device cpu

  # Adjust learning rate
  python scripts/train_model.py --lr 0.01

  # Save to different directory
  python scripts/train_model.py --output-dir models/v1.0.0
        """
    )
    
    # Database and data
    parser.add_argument(
        '--db-path',
        type=str,
        default='data/recommender.db',
        help='Path to SQLite database (default: data/recommender.db)'
    )
    
    # Model architecture
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=64,
        help='Hidden dimension size (default: 64)'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=3,
        help='Number of GraphSAGE layers (default: 3)'
    )
    parser.add_argument(
        '--aggregator',
        type=str,
        default='max',
        choices=['mean', 'max', 'lstm'],
        help='Aggregator type (default: max)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate (default: 0.1)'
    )
    
    # Training hyperparameters
    parser.add_argument(
        '--loss-type',
        type=str,
        default='mse',
        choices=['mse', 'mae', 'bce'],
        help='Loss function (default: mse)'
    )
    parser.add_argument(
        '--lr', '--learning-rate',
        type=float,
        default=0.001,
        dest='learning_rate',
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Batch size (default: 512)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=5,
        help='Early stopping patience (default: 5)'
    )
    parser.add_argument(
        '--early-stopping-min-delta',
        type=float,
        default=1e-4,
        help='Minimum delta for early stopping (default: 1e-4)'
    )
    
    # Output and device
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/current',
        help='Directory to save model artifacts (default: models/current)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda', 'auto'],
        help='Device to train on (default: auto-detect)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress training progress output'
    )
    
    args = parser.parse_args()
    
    # Convert 'auto' to None for auto-detection
    device = None if args.device == 'auto' else args.device
    
    try:
        print("="*60)
        print("GraphSAGE Recommender - Training CLI")
        print("="*60)
        print(f"\nConfiguration:")
        print(f"  Database: {args.db_path}")
        print(f"  Architecture: {args.num_layers} layers x {args.hidden_dim}d ({args.aggregator})")
        print(f"  Training: {args.epochs} epochs, batch={args.batch_size}, lr={args.learning_rate}")
        print(f"  Loss: {args.loss_type}")
        print(f"  Output: {args.output_dir}")
        print(f"  Device: {device if device else 'auto-detect'}")
        print("="*60)
        
        # Train model
        model, results = train_model(
            db_path=args.db_path,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            aggregator=args.aggregator,
            dropout=args.dropout,
            loss_type=args.loss_type,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            output_dir=args.output_dir,
            device=device,
            verbose=not args.quiet
        )
        
        # Print final summary
        metrics = results['metrics']
        
        print("\n" + "="*60)
        print("Training Complete - Final Results")
        print("="*60)
        print(f"\nModel Performance:")
        print(f"  Test RMSE: {metrics['val_rmse']:.4f}")
        print(f"  Test MAE: {metrics['val_mae']:.4f}")
        print(f"  Test Precision@10: {metrics['val_precision_10']:.4f}")
        print(f"  Test Recall@10: {metrics['val_recall_10']:.4f}")
        
        print(f"\nTraining Details:")
        print(f"  Epochs trained: {metrics['epochs_trained']}")
        print(f"  Final train loss: {metrics['final_train_loss']:.4f}")
        print(f"  Best val loss: {metrics['best_val_loss']:.4f}")
        
        print(f"\nArtifacts saved to: {args.output_dir}")
        print(f"  - graphsage_model.pth")
        print(f"  - preprocessor.pkl")
        print(f"  - metadata.json")
        
        print("\n" + "="*60)
        print("[SUCCESS] Training pipeline complete!")
        print("="*60)
        
        # Quick test load
        print("\nVerifying saved artifacts...")
        from app.core.training.model_versioning import verify_artifacts_exist
        if verify_artifacts_exist(args.output_dir):
            print("[SUCCESS] All artifacts verified and ready for deployment!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n[WARNING] Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
