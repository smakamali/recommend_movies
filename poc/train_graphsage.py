"""
Training module for GraphSAGE with multiple loss functions.

Implements:
- BPR (Bayesian Personalized Ranking) loss for ranking
- MSE loss for rating prediction
- Combined BPR+MSE loss for joint optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from tqdm import tqdm


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking (BPR) loss.
    
    Formula: L = -log(σ(r_ui - r_uj)) + λ||θ||²
    where:
    - r_ui: score for positive pair (user i, item u)
    - r_uj: score for negative pair (user i, item j)
    - σ: sigmoid function
    - λ: regularization parameter
    
    Reference: GNN guide lines 379-389 (BPR Loss section).
    """
    
    def __init__(self, reg_lambda=0.01):
        """
        Initialize BPR loss.
        
        Args:
            reg_lambda: L2 regularization parameter (default: 0.01)
        """
        super(BPRLoss, self).__init__()
        self.reg_lambda = reg_lambda
    
    def forward(self, pos_scores, neg_scores, model_params=None):
        """
        Compute BPR loss.
        
        Args:
            pos_scores: Scores for positive pairs (batch_size,)
            neg_scores: Scores for negative pairs (batch_size,)
            model_params: Optional model parameters for regularization
            
        Returns:
            torch.Tensor: BPR loss value
        """
        # BPR loss: -log(σ(pos_score - neg_score))
        diff = pos_scores - neg_scores
        loss = -torch.mean(torch.log(torch.sigmoid(diff) + 1e-10))
        
        # Add L2 regularization if model parameters provided
        if model_params is not None and self.reg_lambda > 0:
            l2_reg = sum(p.pow(2.0).sum() for p in model_params)
            loss += self.reg_lambda * l2_reg
        
        return loss


class RatingMSELoss(nn.Module):
    """
    MSE loss for rating prediction.
    
    Predicts actual rating values in the range [1, 5] and optimizes
    for rating reconstruction accuracy using mean squared error.
    """
    
    def __init__(self, rating_range=(1.0, 5.0)):
        """
        Initialize rating MSE loss.
        
        Args:
            rating_range: Tuple of (min_rating, max_rating) (default: (1.0, 5.0))
        """
        super(RatingMSELoss, self).__init__()
        self.min_rating = rating_range[0]
        self.max_rating = rating_range[1]
        
    def forward(self, predicted_ratings, true_ratings):
        """
        Compute MSE loss for rating prediction.
        
        Args:
            predicted_ratings: Predicted ratings (batch_size,)
            true_ratings: True ratings (batch_size,)
            
        Returns:
            torch.Tensor: MSE loss value
        """
        # Clamp predictions to valid range
        predicted_ratings = torch.clamp(
            predicted_ratings, self.min_rating, self.max_rating
        )
        return nn.functional.mse_loss(predicted_ratings, true_ratings)


class CombinedLoss(nn.Module):
    """
    Combined BPR (ranking) + MSE (rating) loss.
    
    Jointly optimizes for both ranking accuracy (BPR) and rating prediction
    accuracy (MSE), allowing the model to learn embeddings suitable for both tasks.
    """
    
    def __init__(self, mse_weight=1.0, bpr_weight=0.1, reg_lambda=0.01):
        """
        Initialize combined loss.
        
        Args:
            mse_weight: Weight for MSE component (default: 1.0)
            bpr_weight: Weight for BPR component (default: 0.1)
            reg_lambda: L2 regularization parameter (default: 0.01)
        """
        super(CombinedLoss, self).__init__()
        self.mse_loss = RatingMSELoss()
        self.bpr_loss = BPRLoss(reg_lambda=reg_lambda)
        self.mse_weight = mse_weight
        self.bpr_weight = bpr_weight
        
    def forward(self, pred_ratings, true_ratings, pos_scores=None, 
                neg_scores=None, model_params=None):
        """
        Compute combined loss.
        
        Args:
            pred_ratings: Predicted ratings from rating head (batch_size,)
            true_ratings: True ratings (batch_size,)
            pos_scores: Optional scores for positive pairs (for BPR)
            neg_scores: Optional scores for negative pairs (for BPR)
            model_params: Optional model parameters for regularization
            
        Returns:
            torch.Tensor: Combined loss value
        """
        # MSE component (always used)
        mse = self.mse_loss(pred_ratings, true_ratings)
        loss = self.mse_weight * mse
        
        # BPR component (optional)
        if pos_scores is not None and neg_scores is not None:
            bpr = self.bpr_loss(pos_scores, neg_scores, model_params)
            loss += self.bpr_weight * bpr
            
        return loss


def train_graphsage_model(model, graph_data, trainset, user_id_to_idx, item_id_to_idx,
                          num_epochs=20, batch_size=512, learning_rate=0.001,
                          num_negatives=1, device='cpu', verbose=True,
                          loss_type='mse', mse_weight=1.0, bpr_weight=0.1,
                          val_ratio=0.1, early_stopping_patience=5, early_stopping_min_delta=1e-4):
    """
    Train GraphSAGE model with configurable loss function and early stopping.
    
    Args:
        model: GraphSAGERecommender instance
        graph_data: PyTorch Geometric Data object
        trainset: Surprise Trainset
        user_id_to_idx: Dict mapping user_id -> node index
        item_id_to_idx: Dict mapping item_id -> node index
        num_epochs: Number of training epochs (default: 20)
        batch_size: Batch size for training (default: 512)
        learning_rate: Learning rate (default: 0.001)
        num_negatives: Number of negative samples per positive (default: 1, used for BPR/combined)
        device: Device to train on (default: 'cpu')
        verbose: Print training progress (default: True)
        loss_type: Loss function - 'mse', 'bpr', or 'combined' (default: 'mse')
        mse_weight: Weight for MSE loss in combined mode (default: 1.0)
        bpr_weight: Weight for BPR loss in combined mode (default: 0.1)
        val_ratio: Ratio of training data to use for validation (default: 0.1)
        early_stopping_patience: Number of epochs to wait for improvement before stopping (default: 5)
        early_stopping_min_delta: Minimum change in validation loss to qualify as improvement (default: 1e-4)
        
    Returns:
        dict: Training history with losses per epoch
    """
    # Move model and data to device
    model = model.to(device)
    graph_data = graph_data.to(device)
    
    # Build training data structures
    positive_pairs = []
    user_positive_items = defaultdict(set)
    user_item_ratings = {}  # For MSE loss: (user_idx, item_idx) -> rating
    
    for inner_uid, inner_iid, rating in trainset.all_ratings():
        uid = trainset.to_raw_uid(inner_uid)
        iid = trainset.to_raw_iid(inner_iid)
        
        if uid not in user_id_to_idx or iid not in item_id_to_idx:
            continue
        
        user_idx = user_id_to_idx[uid]
        item_idx = item_id_to_idx[iid]
        
        positive_pairs.append((user_idx, item_idx, rating))
        user_positive_items[user_idx].add(item_idx)
        user_item_ratings[(user_idx, item_idx)] = rating
    
    # Split data into train and validation sets
    np.random.shuffle(positive_pairs)
    val_size = int(len(positive_pairs) * val_ratio)
    val_pairs = positive_pairs[:val_size]
    train_pairs = positive_pairs[val_size:]
    
    # Get all item indices for negative sampling (used in BPR/combined modes)
    all_item_indices = list(item_id_to_idx.values())
    num_items = len(all_item_indices)
    
    # Initialize optimizer and loss based on loss_type
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    if loss_type == 'bpr':
        criterion = BPRLoss(reg_lambda=0.01)
    elif loss_type == 'mse':
        criterion = RatingMSELoss()
    elif loss_type == 'combined':
        criterion = CombinedLoss(mse_weight=mse_weight, bpr_weight=bpr_weight, reg_lambda=0.01)
    else:
        raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'mse', 'bpr', or 'combined'")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    
    if verbose:
        print(f"\nTraining GraphSAGE model...")
        print(f"  Total samples: {len(train_pairs) + len(val_pairs):,}")
        print(f"  Training samples: {len(train_pairs):,}")
        print(f"  Validation samples: {len(val_pairs):,} ({val_ratio*100:.1f}%)")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Device: {device}")
        print(f"  Loss type: {loss_type}")
        if loss_type == 'combined':
            print(f"  MSE weight: {mse_weight}, BPR weight: {bpr_weight}")
        print(f"  Early stopping patience: {early_stopping_patience}")
        print(f"  Early stopping min delta: {early_stopping_min_delta}")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Shuffle training pairs
        indices = np.random.permutation(len(train_pairs))
        
        # Process in batches
        for batch_start in range(0, len(train_pairs), batch_size):
            batch_end = min(batch_start + batch_size, len(train_pairs))
            batch_indices = indices[batch_start:batch_end]
            
            # Get batch data
            batch_users = []
            batch_items = []
            batch_ratings = []
            batch_neg_items = []
            
            for idx in batch_indices:
                user_idx, item_idx, rating = train_pairs[idx]
                batch_users.append(user_idx)
                batch_items.append(item_idx)
                batch_ratings.append(rating)
                
                # Sample negative items (for BPR/combined loss)
                if loss_type in ['bpr', 'combined']:
                    neg_samples = []
                    attempts = 0
                    while len(neg_samples) < num_negatives and attempts < 100:
                        neg_item_idx = np.random.choice(all_item_indices)
                        if neg_item_idx not in user_positive_items[user_idx]:
                            neg_samples.append(neg_item_idx)
                        attempts += 1
                    
                    # If couldn't find enough negatives, use random (may include positives)
                    while len(neg_samples) < num_negatives:
                        neg_samples.append(np.random.choice(all_item_indices))
                    
                    batch_neg_items.extend(neg_samples[:num_negatives])
            
            # Forward pass: get embeddings
            user_emb, item_emb = model(graph_data)
            
            # Convert to tensors
            batch_users_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
            batch_items_tensor = torch.tensor(batch_items, dtype=torch.long, device=device)
            batch_ratings_tensor = torch.tensor(batch_ratings, dtype=torch.float32, device=device)
            
            # Compute loss based on loss_type
            if loss_type == 'mse':
                # MSE loss: predict ratings directly
                pred_ratings = model.predict(user_emb, item_emb, batch_users_tensor, 
                                            batch_items_tensor, use_rating_head=True)
                loss = criterion(pred_ratings, batch_ratings_tensor)
                
            elif loss_type == 'bpr':
                # BPR loss: ranking with negative sampling
                batch_neg_items_tensor = torch.tensor(batch_neg_items, dtype=torch.long, device=device)
                
                # Expand users for multiple negatives
                if num_negatives > 1:
                    batch_users_expanded = batch_users_tensor.repeat_interleave(num_negatives)
                else:
                    batch_users_expanded = batch_users_tensor
                
                pos_scores = model.predict(user_emb, item_emb, batch_users_tensor, 
                                          batch_items_tensor, use_rating_head=False)
                neg_scores = model.predict(user_emb, item_emb, batch_users_expanded, 
                                          batch_neg_items_tensor, use_rating_head=False)
                loss = criterion(pos_scores, neg_scores, model.parameters())
                
            elif loss_type == 'combined':
                # Combined loss: MSE for ratings + BPR for ranking
                batch_neg_items_tensor = torch.tensor(batch_neg_items, dtype=torch.long, device=device)
                
                # Get predicted ratings (with rating head)
                pred_ratings = model.predict(user_emb, item_emb, batch_users_tensor, 
                                            batch_items_tensor, use_rating_head=True)
                
                # Get raw scores for BPR (without rating head)
                pos_scores = (user_emb[batch_users_tensor] * item_emb[batch_items_tensor]).sum(dim=1)
                
                # Expand users for multiple negatives
                if num_negatives > 1:
                    batch_users_expanded = batch_users_tensor.repeat_interleave(num_negatives)
                else:
                    batch_users_expanded = batch_users_tensor
                    
                neg_scores = (user_emb[batch_users_expanded] * item_emb[batch_neg_items_tensor]).sum(dim=1)
                
                loss = criterion(pred_ratings, batch_ratings_tensor, pos_scores, 
                               neg_scores, model.parameters())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        history['train_loss'].append(avg_loss)
        history['epoch'].append(epoch + 1)
        
        # Compute validation loss
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            # Forward pass for validation
            user_emb, item_emb = model(graph_data)
            
            # Process validation set in batches
            for val_start in range(0, len(val_pairs), batch_size):
                val_end = min(val_start + batch_size, len(val_pairs))
                val_batch = val_pairs[val_start:val_end]
                
                val_users = []
                val_items = []
                val_ratings = []
                val_neg_items = []
                
                for user_idx, item_idx, rating in val_batch:
                    val_users.append(user_idx)
                    val_items.append(item_idx)
                    val_ratings.append(rating)
                    
                    # Sample negative items for BPR/combined loss
                    if loss_type in ['bpr', 'combined']:
                        neg_samples = []
                        attempts = 0
                        while len(neg_samples) < num_negatives and attempts < 100:
                            neg_item_idx = np.random.choice(all_item_indices)
                            if neg_item_idx not in user_positive_items[user_idx]:
                                neg_samples.append(neg_item_idx)
                            attempts += 1
                        
                        while len(neg_samples) < num_negatives:
                            neg_samples.append(np.random.choice(all_item_indices))
                        
                        val_neg_items.extend(neg_samples[:num_negatives])
                
                # Convert to tensors
                val_users_tensor = torch.tensor(val_users, dtype=torch.long, device=device)
                val_items_tensor = torch.tensor(val_items, dtype=torch.long, device=device)
                val_ratings_tensor = torch.tensor(val_ratings, dtype=torch.float32, device=device)
                
                # Compute validation loss based on loss_type
                if loss_type == 'mse':
                    pred_ratings = model.predict(user_emb, item_emb, val_users_tensor, 
                                                val_items_tensor, use_rating_head=True)
                    batch_val_loss = criterion(pred_ratings, val_ratings_tensor)
                    
                elif loss_type == 'bpr':
                    val_neg_items_tensor = torch.tensor(val_neg_items, dtype=torch.long, device=device)
                    
                    if num_negatives > 1:
                        val_users_expanded = val_users_tensor.repeat_interleave(num_negatives)
                    else:
                        val_users_expanded = val_users_tensor
                    
                    pos_scores = model.predict(user_emb, item_emb, val_users_tensor, 
                                              val_items_tensor, use_rating_head=False)
                    neg_scores = model.predict(user_emb, item_emb, val_users_expanded, 
                                              val_neg_items_tensor, use_rating_head=False)
                    batch_val_loss = criterion(pos_scores, neg_scores, model.parameters())
                    
                elif loss_type == 'combined':
                    val_neg_items_tensor = torch.tensor(val_neg_items, dtype=torch.long, device=device)
                    
                    pred_ratings = model.predict(user_emb, item_emb, val_users_tensor, 
                                                val_items_tensor, use_rating_head=True)
                    pos_scores = (user_emb[val_users_tensor] * item_emb[val_items_tensor]).sum(dim=1)
                    
                    if num_negatives > 1:
                        val_users_expanded = val_users_tensor.repeat_interleave(num_negatives)
                    else:
                        val_users_expanded = val_users_tensor
                        
                    neg_scores = (user_emb[val_users_expanded] * item_emb[val_neg_items_tensor]).sum(dim=1)
                    
                    batch_val_loss = criterion(pred_ratings, val_ratings_tensor, pos_scores, 
                                             neg_scores, model.parameters())
                
                val_loss += batch_val_loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        history['val_loss'].append(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model state
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_loss:.4f}, Val Loss = {avg_val_loss:.4f} (BEST)")
        else:
            patience_counter += 1
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_loss:.4f}, Val Loss = {avg_val_loss:.4f} (patience: {patience_counter}/{early_stopping_patience})")
            
            # Check if early stopping should trigger
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping triggered! No improvement in validation loss for {early_stopping_patience} epochs.")
                    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        if verbose:
            print(f"\nRestored model from epoch {best_epoch} with validation loss {best_val_loss:.4f}")
    
    if verbose:
        print("\nTraining completed!")
    
    return history


if __name__ == "__main__":
    # Test training
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from gnn.graph_data_loader import build_bipartite_graph
    from gnn.graphsage_model import GraphSAGERecommender
    from poc.data_loader import (load_movielens_100k, get_train_test_split,
                                load_user_features, load_item_features)
    
    print("Loading data...")
    data = load_movielens_100k()
    trainset, testset = get_train_test_split(data, test_size=0.2, random_state=42)
    user_features = load_user_features()
    item_features = load_item_features()
    
    print("Building graph...")
    graph_data, preprocessor, user_id_to_idx, item_id_to_idx = build_bipartite_graph(
        trainset, user_features, item_features
    )
    
    print("Initializing model...")
    user_feat_dim = graph_data.x[graph_data.node_type == 0].size(1)
    item_feat_dim = graph_data.x[graph_data.node_type == 1].size(1)
    
    model = GraphSAGERecommender(
        num_users=graph_data.num_users,
        num_items=graph_data.num_items,
        user_feat_dim=user_feat_dim,
        item_feat_dim=item_feat_dim,
        hidden_dim=64,
        num_layers=2
    )
    
    print("Training model with early stopping (max 50 epochs)...")
    history = train_graphsage_model(
        model, graph_data, trainset, user_id_to_idx, item_id_to_idx,
        num_epochs=50, batch_size=256, learning_rate=0.001,
        device='cuda', verbose=True,
        val_ratio=0.10, early_stopping_patience=5, early_stopping_min_delta=1e-4
    )
    
    print(f"\nFinal train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"Trained for {len(history['epoch'])} epochs")
    print("Training test successful!")
