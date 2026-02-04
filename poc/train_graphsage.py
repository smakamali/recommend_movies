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
    
    def forward(self, pos_scores, neg_scores, model_params=None, weights=None):
        """
        Compute BPR loss (optionally weighted for graded BPR).
        
        Args:
            pos_scores: Scores for positive pairs (batch_size,)
            neg_scores: Scores for negative pairs (batch_size,)
            model_params: Optional model parameters for regularization
            weights: Optional per-pair weights (batch_size,) for graded BPR; larger rating gap -> higher weight
            
        Returns:
            torch.Tensor: BPR loss value
        """
        # BPR loss: -log(σ(pos_score - neg_score))
        diff = pos_scores - neg_scores
        log_sigmoid = torch.log(torch.sigmoid(diff) + 1e-10)
        if weights is not None:
            loss = -(weights * log_sigmoid).mean()
        else:
            loss = -torch.mean(log_sigmoid)
        
        # Add L2 regularization if model parameters provided
        if model_params is not None and self.reg_lambda > 0:
            l2_reg = sum(p.pow(2.0).sum() for p in model_params)
            loss += self.reg_lambda * l2_reg
        
        return loss


class RatingMSELoss(nn.Module):
    """
    MSE loss for rating prediction.
    
    With scaled labels, predicts values in [0, 1] (sigmoid output) and optimizes
    for rating reconstruction using mean squared error.
    """
    
    def __init__(self, rating_range=(0.0, 1.0)):
        """
        Initialize rating MSE loss.
        
        Args:
            rating_range: Tuple of (min_rating, max_rating) (default: (0.0, 1.0) for scaled labels)
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
        self.mse_loss = RatingMSELoss(rating_range=(0.0, 1.0))
        self.bpr_loss = BPRLoss(reg_lambda=reg_lambda)
        self.mse_weight = mse_weight
        self.bpr_weight = bpr_weight
        
    def forward(self, pred_ratings, true_ratings, pos_scores=None, 
                neg_scores=None, model_params=None, bpr_weights=None):
        """
        Compute combined loss.
        
        Args:
            pred_ratings: Predicted ratings from rating head (batch_size,)
            true_ratings: True ratings (batch_size,)
            pos_scores: Optional scores for positive pairs (for BPR)
            neg_scores: Optional scores for negative pairs (for BPR)
            model_params: Optional model parameters for regularization
            bpr_weights: Optional per-pair weights for graded BPR (batch_size,)
            
        Returns:
            torch.Tensor: Combined loss value
        """
        # MSE component (always used)
        mse = self.mse_loss(pred_ratings, true_ratings)
        loss = self.mse_weight * mse
        
        # BPR component (optional)
        if pos_scores is not None and neg_scores is not None:
            bpr = self.bpr_loss(pos_scores, neg_scores, model_params, weights=bpr_weights)
            loss += self.bpr_weight * bpr
            
        return loss


def _compute_mae(model, graph_data, pairs, rating_scaler, device, batch_size, loss_type='mse'):
    """
    Compute MAE (1-5 scale) for a list of (user_idx, item_idx, true_rating) pairs.
    true_rating is in original 1-5 scale.
    For MSE/combined use rating head; for BPR use dot-product-derived [1,5] (proxy metric).
    """
    if not pairs:
        return 0.0
    model.eval()
    lt = (str(loss_type).lower() if loss_type else 'mse')
    lt = lt if lt in ('mse', 'bpr', 'combined') else 'mse'
    errors = []
    with torch.no_grad():
        user_emb, item_emb = model(graph_data)
        for start in range(0, len(pairs), batch_size):
            end = min(start + batch_size, len(pairs))
            batch = pairs[start:end]
            users = torch.tensor([p[0] for p in batch], dtype=torch.long, device=device)
            items = torch.tensor([p[1] for p in batch], dtype=torch.long, device=device)
            true_ratings = np.array([p[2] for p in batch], dtype=np.float64)  # 1-5
            if lt in ('mse', 'combined'):
                pred_scaled = model.predict(user_emb, item_emb, users, items, use_rating_head=True)
                pred_1_5 = rating_scaler.inverse_transform(
                    pred_scaled.cpu().numpy().reshape(-1, 1)
                ).flatten()
            else:
                dots = model.predict(user_emb, item_emb, users, items, use_rating_head=False)
                pred_1_5 = (1.0 + 4.0 * torch.sigmoid(dots).cpu().numpy()).astype(np.float64)
            errors.extend(np.abs(true_ratings - pred_1_5))
    return float(np.mean(errors))


def _compute_mape(model, graph_data, pairs, rating_scaler, device, batch_size, loss_type='mse', epsilon=1e-10):
    """
    Compute MAPE in percent (1-5 scale) for a list of (user_idx, item_idx, true_rating) pairs.
    MAPE = mean(|true - pred| / |true|) * 100.
    For MSE/combined use rating head; for BPR use dot-product-derived [1,5] (proxy metric).
    """
    if not pairs:
        return 0.0
    model.eval()
    lt = (str(loss_type).lower() if loss_type else 'mse')
    lt = lt if lt in ('mse', 'bpr', 'combined') else 'mse'
    pct_errors = []
    with torch.no_grad():
        user_emb, item_emb = model(graph_data)
        for start in range(0, len(pairs), batch_size):
            end = min(start + batch_size, len(pairs))
            batch = pairs[start:end]
            users = torch.tensor([p[0] for p in batch], dtype=torch.long, device=device)
            items = torch.tensor([p[1] for p in batch], dtype=torch.long, device=device)
            true_ratings = np.array([p[2] for p in batch], dtype=np.float64)  # 1-5
            if lt in ('mse', 'combined'):
                pred_scaled = model.predict(user_emb, item_emb, users, items, use_rating_head=True)
                pred_1_5 = rating_scaler.inverse_transform(
                    pred_scaled.cpu().numpy().reshape(-1, 1)
                ).flatten()
            else:
                dots = model.predict(user_emb, item_emb, users, items, use_rating_head=False)
                pred_1_5 = (1.0 + 4.0 * torch.sigmoid(dots).cpu().numpy()).astype(np.float64)
            denom = np.maximum(np.abs(true_ratings), epsilon)
            pct_errors.extend(100.0 * np.abs(true_ratings - pred_1_5) / denom)
    return float(np.mean(pct_errors))


def _sample_graded_bpr_pair(user_rated_items, users_with_graded):
    """
    Sample one graded BPR pair: (user, pos_item, neg_item) where both items are rated
    by the user and rating(pos_item) > rating(neg_item).
    
    Args:
        user_rated_items: dict user_idx -> list of (item_idx, rating)
        users_with_graded: list of user_idx that have at least two rated items with different ratings
        
    Returns:
        tuple: (user_idx, pos_item, neg_item, weight) or None if no valid pair possible.
        weight = 1 + (r_pos - r_neg) / 4.0 so that 5 vs 1 is penalized more than 4 vs 3.
    """
    if not users_with_graded:
        return None
    user_idx = np.random.choice(users_with_graded)
    rated = user_rated_items[user_idx]
    if len(rated) < 2:
        return None
    # Sample two distinct (item, rating) with r_pos > r_neg
    i, j = np.random.choice(len(rated), size=2, replace=False)
    (pos_item, r_pos), (neg_item, r_neg) = rated[i], rated[j]
    if r_pos <= r_neg:
        pos_item, neg_item = neg_item, pos_item
        r_pos, r_neg = r_neg, r_pos
    weight = 1.0 + (r_pos - r_neg) / 4.0
    return (user_idx, pos_item, neg_item, weight)


def train_graphsage_model(model, graph_data, trainset, user_id_to_idx, item_id_to_idx,
                          rating_scaler,
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
        rating_scaler: Fitted MinMaxScaler to transform ratings from [1,5] to [0,1]
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
    user_rated_items = defaultdict(list)  # user_idx -> [(item_idx, rating), ...] for graded BPR
    
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
        user_rated_items[user_idx].append((item_idx, float(rating)))
    
    # Users with at least two rated items and at least one pair with different ratings (for graded BPR)
    users_with_graded = [
        u for u, rated in user_rated_items.items()
        if len(rated) >= 2 and len(set(r for _, r in rated)) >= 2
    ]
    if loss_type in ['bpr', 'combined'] and not users_with_graded:
        raise ValueError(
            "No users with at least two differently-rated items for graded BPR. "
            "Graded BPR requires (pos, neg) pairs where both are rated and rating(pos) > rating(neg)."
        )
    
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
        criterion = RatingMSELoss(rating_range=(0.0, 1.0))
    elif loss_type == 'combined':
        criterion = CombinedLoss(mse_weight=mse_weight, bpr_weight=bpr_weight, reg_lambda=0.01)
    else:
        raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'mse', 'bpr', or 'combined'")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
        'train_mape': [],
        'val_mape': [],
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
            batch_bpr_weights = []
            
            if loss_type in ['bpr', 'combined']:
                # Graded BPR: sample (user, pos_item, neg_item) where both rated and rating(pos) > rating(neg)
                max_attempts = batch_size * 10
                attempts = 0
                while len(batch_users) < batch_size and attempts < max_attempts:
                    pair = _sample_graded_bpr_pair(user_rated_items, users_with_graded)
                    if pair is not None:
                        user_idx, pos_item, neg_item, weight = pair
                        batch_users.append(user_idx)
                        batch_items.append(pos_item)
                        batch_neg_items.append(neg_item)
                        batch_bpr_weights.append(weight)
                        if loss_type == 'combined':
                            r_pos = user_item_ratings[(user_idx, pos_item)]
                            batch_ratings.append(
                                float(rating_scaler.transform([[r_pos]])[0, 0])
                            )
                    attempts += 1
                if len(batch_users) == 0:
                    continue
                # Pad to batch_size if we didn't get enough (rare)
                while len(batch_users) < batch_size:
                    pair = _sample_graded_bpr_pair(user_rated_items, users_with_graded)
                    if pair is not None:
                        user_idx, pos_item, neg_item, weight = pair
                        batch_users.append(user_idx)
                        batch_items.append(pos_item)
                        batch_neg_items.append(neg_item)
                        batch_bpr_weights.append(weight)
                        if loss_type == 'combined':
                            r_pos = user_item_ratings[(user_idx, pos_item)]
                            batch_ratings.append(
                                float(rating_scaler.transform([[r_pos]])[0, 0])
                            )
                    if len(batch_users) >= batch_size:
                        break
            else:
                # MSE-only: use train_pairs as before
                for idx in batch_indices:
                    user_idx, item_idx, rating = train_pairs[idx]
                    batch_users.append(user_idx)
                    batch_items.append(item_idx)
                    batch_ratings.append(
                        float(rating_scaler.transform([[rating]])[0, 0])
                    )
            
            # Forward pass: get embeddings
            user_emb, item_emb = model(graph_data)
            
            # Convert to tensors
            batch_users_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
            batch_items_tensor = torch.tensor(batch_items, dtype=torch.long, device=device)
            if batch_ratings:
                batch_ratings_tensor = torch.tensor(batch_ratings, dtype=torch.float32, device=device)
            if loss_type in ['bpr', 'combined']:
                batch_neg_items_tensor = torch.tensor(batch_neg_items, dtype=torch.long, device=device)
                batch_bpr_weights_tensor = torch.tensor(batch_bpr_weights, dtype=torch.float32, device=device)
            
            # Compute loss based on loss_type
            if loss_type == 'mse':
                # MSE loss: predict ratings directly
                pred_ratings = model.predict(user_emb, item_emb, batch_users_tensor, 
                                            batch_items_tensor, use_rating_head=True)
                loss = criterion(pred_ratings, batch_ratings_tensor)
                
            elif loss_type == 'bpr':
                # Graded BPR: one (pos, neg) per sample, weighted by rating gap
                pos_scores = (user_emb[batch_users_tensor] * item_emb[batch_items_tensor]).sum(dim=1)
                neg_scores = (user_emb[batch_users_tensor] * item_emb[batch_neg_items_tensor]).sum(dim=1)
                loss = criterion(pos_scores, neg_scores, model.parameters(), weights=batch_bpr_weights_tensor)
                
            elif loss_type == 'combined':
                # Combined: MSE on (user, pos) rating + graded BPR with weights
                pred_ratings = model.predict(user_emb, item_emb, batch_users_tensor, 
                                            batch_items_tensor, use_rating_head=True)
                pos_scores = (user_emb[batch_users_tensor] * item_emb[batch_items_tensor]).sum(dim=1)
                neg_scores = (user_emb[batch_users_tensor] * item_emb[batch_neg_items_tensor]).sum(dim=1)
                loss = criterion(pred_ratings, batch_ratings_tensor, pos_scores, 
                               neg_scores, model.parameters(), bpr_weights=batch_bpr_weights_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        history['train_loss'].append(avg_loss)
        history['epoch'].append(epoch + 1)
        
        # Compute training MAE and MAPE (1-5 scale; loss-dependent predictions)
        train_mae = _compute_mae(model, graph_data, train_pairs, rating_scaler, device, batch_size, loss_type=loss_type)
        train_mape = _compute_mape(model, graph_data, train_pairs, rating_scaler, device, batch_size, loss_type=loss_type)
        history['train_mae'].append(train_mae)
        history['train_mape'].append(train_mape)
        
        # Compute validation loss, validation MAE and MAPE
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            # Forward pass for validation
            user_emb, item_emb = model(graph_data)
            
            if loss_type == 'mse':
                # MSE: iterate val_pairs in batches
                for val_start in range(0, len(val_pairs), batch_size):
                    val_end = min(val_start + batch_size, len(val_pairs))
                    val_batch = val_pairs[val_start:val_end]
                    val_users = [p[0] for p in val_batch]
                    val_items = [p[1] for p in val_batch]
                    val_ratings = [float(rating_scaler.transform([[p[2]]])[0, 0]) for p in val_batch]
                    val_users_tensor = torch.tensor(val_users, dtype=torch.long, device=device)
                    val_items_tensor = torch.tensor(val_items, dtype=torch.long, device=device)
                    val_ratings_tensor = torch.tensor(val_ratings, dtype=torch.float32, device=device)
                    pred_ratings = model.predict(user_emb, item_emb, val_users_tensor,
                                                val_items_tensor, use_rating_head=True)
                    batch_val_loss = criterion(pred_ratings, val_ratings_tensor)
                    val_loss += batch_val_loss.item()
                    val_batches += 1
            else:
                # BPR/combined: graded BPR validation batches (sample graded pairs)
                num_val_batches = max(1, len(val_pairs) // batch_size)
                for _ in range(num_val_batches):
                    val_users = []
                    val_items = []
                    val_ratings = []
                    val_neg_items = []
                    val_bpr_weights = []
                    filled = 0
                    attempts = 0
                    while filled < batch_size and attempts < batch_size * 10:
                        pair = _sample_graded_bpr_pair(user_rated_items, users_with_graded)
                        if pair is not None:
                            user_idx, pos_item, neg_item, weight = pair
                            val_users.append(user_idx)
                            val_items.append(pos_item)
                            val_neg_items.append(neg_item)
                            val_bpr_weights.append(weight)
                            if loss_type == 'combined':
                                r_pos = user_item_ratings[(user_idx, pos_item)]
                                val_ratings.append(float(rating_scaler.transform([[r_pos]])[0, 0]))
                            filled += 1
                        attempts += 1
                    if filled == 0:
                        continue
                    val_users_tensor = torch.tensor(val_users, dtype=torch.long, device=device)
                    val_items_tensor = torch.tensor(val_items, dtype=torch.long, device=device)
                    val_neg_items_tensor = torch.tensor(val_neg_items, dtype=torch.long, device=device)
                    val_bpr_weights_tensor = torch.tensor(val_bpr_weights, dtype=torch.float32, device=device)
                    if loss_type == 'bpr':
                        pos_scores = (user_emb[val_users_tensor] * item_emb[val_items_tensor]).sum(dim=1)
                        neg_scores = (user_emb[val_users_tensor] * item_emb[val_neg_items_tensor]).sum(dim=1)
                        batch_val_loss = criterion(pos_scores, neg_scores, model.parameters(),
                                                   weights=val_bpr_weights_tensor)
                    else:
                        val_ratings_tensor = torch.tensor(val_ratings, dtype=torch.float32, device=device)
                        pred_ratings = model.predict(user_emb, item_emb, val_users_tensor,
                                                     val_items_tensor, use_rating_head=True)
                        pos_scores = (user_emb[val_users_tensor] * item_emb[val_items_tensor]).sum(dim=1)
                        neg_scores = (user_emb[val_users_tensor] * item_emb[val_neg_items_tensor]).sum(dim=1)
                        batch_val_loss = criterion(pred_ratings, val_ratings_tensor, pos_scores,
                                                   neg_scores, model.parameters(), bpr_weights=val_bpr_weights_tensor)
                    val_loss += batch_val_loss.item()
                    val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        history['val_loss'].append(avg_val_loss)
        
        # Compute validation MAE and MAPE (1-5 scale; loss-dependent predictions)
        val_mae = _compute_mae(model, graph_data, val_pairs, rating_scaler, device, batch_size, loss_type=loss_type)
        val_mape = _compute_mape(model, graph_data, val_pairs, rating_scaler, device, batch_size, loss_type=loss_type)
        history['val_mae'].append(val_mae)
        history['val_mape'].append(val_mape)
        
        # Early stopping check
        if avg_val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model state
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_loss:.4f}, Val Loss = {avg_val_loss:.4f} | MAE T={train_mae:.4f}/V={val_mae:.4f} MAPE T={train_mape:.2f}%/V={val_mape:.2f}% (BEST)")
        else:
            patience_counter += 1
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_loss:.4f}, Val Loss = {avg_val_loss:.4f} | MAE T={train_mae:.4f}/V={val_mae:.4f} MAPE T={train_mape:.2f}%/V={val_mape:.2f}% (patience: {patience_counter}/{early_stopping_patience})")
            
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
    
    # Record best epoch in history for downstream use
    history['best_epoch'] = best_epoch
    
    if verbose:
        print("\nTraining completed!")
    
    return history


if __name__ == "__main__":
    # Test training
    import sys
    from pathlib import Path
    from sklearn.preprocessing import MinMaxScaler
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from poc.graph_data_loader import build_bipartite_graph
    from poc.graphsage_model import GraphSAGERecommender
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
    
    # Fit rating scaler
    train_ratings = np.array([r for (_, _, r) in trainset.all_ratings()]).reshape(-1, 1)
    rating_scaler = MinMaxScaler(feature_range=(0, 1))
    rating_scaler.fit(train_ratings)
    
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
        rating_scaler,
        num_epochs=50, batch_size=256, learning_rate=0.001,
        device='cuda', verbose=True,
        val_ratio=0.10, early_stopping_patience=5, early_stopping_min_delta=1e-4
    )
    
    print(f"\nFinal train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"Trained for {len(history['epoch'])} epochs")
    print("Training test successful!")
