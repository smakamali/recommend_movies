"""
Model loader for GraphSAGE inference.

Handles loading trained models, preprocessors, and metadata from disk
with in-memory caching for efficient inference.
"""

import json
import pickle
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

from poc.graphsage_model import GraphSAGERecommender
from poc.data_loader import FeaturePreprocessor

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Loads and caches trained GraphSAGE model and preprocessor.
    
    This class handles:
    - Loading model weights from .pth files
    - Loading fitted preprocessor from .pkl files
    - Loading model metadata from .json files
    - Caching loaded models in memory
    - Model validation
    
    Expected directory structure:
    models/current/
        - graphsage_model.pth (model weights)
        - preprocessor.pkl (fitted preprocessor)
        - metadata.json (model configuration and metrics)
    """
    
    def __init__(
        self,
        model_dir: str = "models/current",
        device: Optional[str] = None
    ):
        """
        Initialize model loader.
        
        Args:
            model_dir: Directory containing model artifacts (default: "models/current")
            device: Device to load model on ('cpu' or 'cuda'). Auto-detects if None.
        """
        self.model_dir = Path(model_dir)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cache
        self._model: Optional[GraphSAGERecommender] = None
        self._preprocessor: Optional[FeaturePreprocessor] = None
        self._metadata: Optional[Dict[str, Any]] = None
        
        logger.info(f"ModelLoader initialized with device: {self.device}")
    
    def load_model(
        self,
        force_reload: bool = False
    ) -> Tuple[GraphSAGERecommender, FeaturePreprocessor, Dict[str, Any]]:
        """
        Load model, preprocessor, and metadata.
        
        Args:
            force_reload: Force reload from disk even if cached (default: False)
            
        Returns:
            Tuple of (model, preprocessor, metadata)
            
        Raises:
            FileNotFoundError: If model artifacts not found
            RuntimeError: If model loading fails
        """
        # Return cached if available and not forcing reload
        if not force_reload and self._model is not None:
            logger.info("Using cached model")
            return self._model, self._preprocessor, self._metadata
        
        logger.info(f"Loading model from {self.model_dir}")
        
        # Load metadata first to get model configuration
        metadata = self._load_metadata()
        
        # Load preprocessor
        preprocessor = self._load_preprocessor()
        
        # Load model with configuration from metadata
        model = self._load_model_weights(metadata, preprocessor)
        
        # Cache
        self._model = model
        self._preprocessor = preprocessor
        self._metadata = metadata
        
        logger.info("Model loaded successfully")
        logger.info(f"  Model version: {metadata.get('model_version', 'unknown')}")
        logger.info(f"  Hidden dim: {metadata.get('hidden_dim', 'unknown')}")
        logger.info(f"  Num layers: {metadata.get('num_layers', 'unknown')}")
        logger.info(f"  Val RMSE: {metadata.get('val_rmse', 'unknown')}")
        
        return model, preprocessor, metadata
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load model metadata from JSON file."""
        metadata_path = self.model_dir / "metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}\n"
                "Please ensure the model has been trained and saved."
            )
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.debug(f"Loaded metadata: {metadata}")
        return metadata
    
    def _load_preprocessor(self) -> FeaturePreprocessor:
        """Load fitted preprocessor from pickle file."""
        preprocessor_path = self.model_dir / "preprocessor.pkl"
        
        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor file not found: {preprocessor_path}\n"
                "Please ensure the model has been trained and saved."
            )
        
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        logger.debug("Loaded preprocessor")
        return preprocessor
    
    def _load_model_weights(
        self,
        metadata: Dict[str, Any],
        preprocessor: FeaturePreprocessor
    ) -> GraphSAGERecommender:
        """
        Load model architecture and weights.
        
        Args:
            metadata: Model metadata containing architecture configuration
            preprocessor: Fitted preprocessor with feature dimensions
            
        Returns:
            GraphSAGERecommender model loaded with trained weights
        """
        model_path = self.model_dir / "graphsage_model.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model weights file not found: {model_path}\n"
                "Please ensure the model has been trained and saved."
            )
        
        # Extract configuration from metadata (use saved dims to match trained weights)
        hp = metadata.get('hyperparameters', {})
        num_users = metadata.get('num_users_trained', hp.get('num_users', 943))
        num_items = metadata.get('num_movies', hp.get('num_items', 1682))
        hidden_dim = metadata.get('hidden_dim', hp.get('hidden_dim', 64))
        num_layers = metadata.get('num_layers', hp.get('num_layers', 3))
        dropout = hp.get('dropout', 0.1)
        aggregator = metadata.get('aggregator', hp.get('aggregator', 'max'))
        
        # Use feature dimensions from metadata (training pads both to max)
        user_feat_dim = hp.get('user_feat_dim')
        item_feat_dim = hp.get('item_feat_dim')
        if user_feat_dim is None or item_feat_dim is None:
            # Fallback: derive from preprocessor
            n_genders = len(preprocessor.gender_encoder.classes_)
            n_occupations = len(preprocessor.occupation_encoder.classes_)
            user_feat_raw = 1 + n_genders + n_occupations
            item_feat_raw = 1 + 19
            feat_dim = max(user_feat_raw, item_feat_raw)
            user_feat_dim = user_feat_dim if user_feat_dim is not None else feat_dim
            item_feat_dim = item_feat_dim if item_feat_dim is not None else feat_dim
        
        logger.debug(f"Initializing model with:")
        logger.debug(f"  num_users={num_users}, num_items={num_items}")
        logger.debug(f"  user_feat_dim={user_feat_dim}, item_feat_dim={item_feat_dim}")
        logger.debug(f"  hidden_dim={hidden_dim}, num_layers={num_layers}")
        logger.debug(f"  dropout={dropout}, aggregator={aggregator}")
        
        # Initialize model architecture
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
        
        # Load trained weights
        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()  # Set to evaluation mode
            logger.debug("Model weights loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}")
        
        return model
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self._model is None or self._metadata is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "version": self._metadata.get('model_version', 'unknown'),
            "device": str(self.device),
            "hidden_dim": self._metadata.get('hidden_dim'),
            "num_layers": self._metadata.get('num_layers'),
            "val_rmse": self._metadata.get('val_rmse'),
            "training_date": self._metadata.get('training_date'),
            "num_users_trained": self._metadata.get('num_users_trained'),
            "num_movies": self._metadata.get('num_movies')
        }
    
    def clear_cache(self):
        """Clear cached model and preprocessor."""
        self._model = None
        self._preprocessor = None
        self._metadata = None
        logger.info("Model cache cleared")


if __name__ == "__main__":
    # Test model loader
    from app.utils.logging_config import configure_inference_logging
    
    configure_inference_logging(debug=True)
    logger = logging.getLogger(__name__)
    
    try:
        loader = ModelLoader(model_dir="models/current")
        model, preprocessor, metadata = loader.load_model()
        
        print("\nModel loaded successfully!")
        print(f"Model info: {loader.get_model_info()}")
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nNote: This is expected if the model hasn't been trained yet.")
        print("The inference engine will create mock models for testing.")
