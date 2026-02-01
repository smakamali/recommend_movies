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

# Mitigate OpenMP libiomp5md.dll conflict (PyTorch/NumPy/MKL)
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import warnings
warnings.filterwarnings("ignore", message=".*torch-scatter.*")

import sys
import argparse
from pathlib import Path

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.training.train import train_model

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.yaml"


def load_model_config(config_path: Path) -> dict:
    """Load model and training defaults from YAML. Missing file or keys use fallbacks."""
    fallback = {
        "model": {
            "hidden_dim": 64,
            "num_layers": 3,
            "aggregator": "max",
            "dropout": 0.1,
        },
        "training": {
            "learning_rate": 0.001,
            "batch_size": 512,
            "num_epochs": 50,
            "early_stopping_patience": 5,
            "early_stopping_min_delta": 1e-4,
            "loss_type": "mse",
        },
    }
    if not config_path.is_file():
        return fallback
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        model = {**fallback["model"], **(data.get("model") or {})}
        training = {**fallback["training"], **(data.get("training") or {})}
        return {"model": model, "training": training}
    except Exception:
        return fallback


def main():
    """Main CLI entry point. Defaults from config/model_config.yaml; CLI args override."""
    # Parse --config first so we can load defaults from the right file
    pre = argparse.ArgumentParser()
    pre.add_argument('--config', type=Path, default=None)
    pre_args, _ = pre.parse_known_args()
    config_path = pre_args.config if pre_args.config is not None else DEFAULT_CONFIG_PATH
    cfg = load_model_config(config_path)
    m, t = cfg["model"], cfg["training"]

    parser = argparse.ArgumentParser(
        description="Train GraphSAGE recommendation model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default parameters (from config/model_config.yaml)
  python scripts/train_model.py

  # Override epochs and batch size
  python scripts/train_model.py --epochs 30 --batch-size 1024

  # Use a different config file
  python scripts/train_model.py --config path/to/config.yaml

  # Force CPU training
  python scripts/train_model.py --device cpu

  # Adjust learning rate
  python scripts/train_model.py --lr 0.01

  # Save to different directory
  python scripts/train_model.py --output-dir models/v1.0.0
        """
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help=f'Path to model config YAML (default: {DEFAULT_CONFIG_PATH})'
    )

    # Database and data
    parser.add_argument(
        '--db-path',
        type=str,
        default='data/recommender.db',
        help='Path to SQLite database (default: data/recommender.db)'
    )

    # Model architecture (defaults from config)
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=m["hidden_dim"],
        help=f'Hidden dimension size (default from config: {m["hidden_dim"]})'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=m["num_layers"],
        help=f'Number of GraphSAGE layers (default from config: {m["num_layers"]})'
    )
    parser.add_argument(
        '--aggregator',
        type=str,
        default=m["aggregator"],
        choices=['mean', 'max', 'lstm'],
        help=f'Aggregator type (default from config: {m["aggregator"]})'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=m["dropout"],
        help=f'Dropout rate (default from config: {m["dropout"]})'
    )

    # Training hyperparameters (defaults from config)
    parser.add_argument(
        '--loss-type',
        type=str,
        default=t["loss_type"],
        choices=['mse', 'mae', 'bce'],
        help=f'Loss function (default from config: {t["loss_type"]})'
    )
    parser.add_argument(
        '--lr', '--learning-rate',
        type=float,
        default=t["learning_rate"],
        dest='learning_rate',
        help=f'Learning rate (default from config: {t["learning_rate"]})'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=t["batch_size"],
        help=f'Batch size (default from config: {t["batch_size"]})'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=t["num_epochs"],
        help=f'Number of training epochs (default from config: {t["num_epochs"]})'
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=t["early_stopping_patience"],
        help=f'Early stopping patience (default from config: {t["early_stopping_patience"]})'
    )
    parser.add_argument(
        '--early-stopping-min-delta',
        type=float,
        default=t["early_stopping_min_delta"],
        help=f'Minimum delta for early stopping (default from config: {t["early_stopping_min_delta"]})'
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
        print(f"  - rating_scaler.pkl")
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
