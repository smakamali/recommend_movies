"""
GraphSAGE Model for Recommendation.

Implements GraphSAGE architecture with feature support for MovieLens.
Uses pooling aggregator for expressive feature learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data


class GraphSAGERecommender(nn.Module):
    """
    GraphSAGE model for recommendation with user and item features.
    
    Architecture:
    - Input projection: Features -> hidden_dim
    - 2-3 GraphSAGE layers with pooling aggregator
    - Output: User and item embeddings
    
    Following GNN guide lines 663-729 (GraphSAGE section).
    """
    
    def __init__(self, num_users, num_items, user_feat_dim, item_feat_dim,
                 hidden_dim=64, num_layers=2, dropout=0.1, aggregator='max'):
        """
        Initialize GraphSAGE model.
        
        Args:
            num_users: Number of users
            num_items: Number of items
            user_feat_dim: Dimension of user features
            item_feat_dim: Dimension of item features
            hidden_dim: Hidden dimension for embeddings (default: 64)
            num_layers: Number of GraphSAGE layers (default: 2)
            dropout: Dropout rate (default: 0.1)
            aggregator: Aggregator type - 'mean' or 'max' (default: 'max')
        """
        super(GraphSAGERecommender, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Feature dimensions (after padding to same size)
        self.feat_dim = max(user_feat_dim, item_feat_dim)
        
        # Input projection: project features to hidden dimension
        self.user_proj = nn.Linear(user_feat_dim, hidden_dim)
        self.item_proj = nn.Linear(item_feat_dim, hidden_dim)
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # First layer: hidden_dim -> hidden_dim
                self.convs.append(
                    SAGEConv(hidden_dim, hidden_dim, aggr=aggregator)
                )
            else:
                # Subsequent layers: hidden_dim -> hidden_dim
                self.convs.append(
                    SAGEConv(hidden_dim, hidden_dim, aggr=aggregator)
                )
        
        # Rating prediction head: maps dot product to rating scale [1, 5]
        self.rating_head = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize model parameters."""
        # Xavier initialization for projection layers
        nn.init.xavier_uniform_(self.user_proj.weight)
        nn.init.xavier_uniform_(self.item_proj.weight)
        nn.init.zeros_(self.user_proj.bias)
        nn.init.zeros_(self.item_proj.bias)
    
    def forward(self, graph_data):
        """
        Forward pass through GraphSAGE.
        
        Args:
            graph_data: PyTorch Geometric Data object with:
                - x: Node features (num_nodes, feat_dim)
                - edge_index: Graph connectivity (2, num_edges)
                - node_type: Node type mask (0=user, 1=item)
                
        Returns:
            tuple: (user_embeddings, item_embeddings)
                - user_embeddings: (num_users, hidden_dim)
                - item_embeddings: (num_items, hidden_dim)
        """
        x = graph_data.x  # (num_nodes, feat_dim)
        edge_index = graph_data.edge_index
        node_type = graph_data.node_type
        
        # Split into user and item features
        user_mask = (node_type == 0)
        item_mask = (node_type == 1)
        
        user_feat = x[user_mask]  # (num_users, feat_dim)
        item_feat = x[item_mask]  # (num_items, feat_dim)
        
        # Project features to hidden dimension
        user_h = self.user_proj(user_feat)  # (num_users, hidden_dim)
        item_h = self.item_proj(item_feat)  # (num_items, hidden_dim)
        
        # Combine for message passing
        h = torch.cat([user_h, item_h], dim=0)  # (num_nodes, hidden_dim)
        
        # Apply GraphSAGE layers
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index)
            
            # Residual connection (if dimensions match)
            if h.size(1) == h_new.size(1):
                h = h + h_new
            else:
                h = h_new
            
            # Activation and dropout (except last layer)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Split back into user and item embeddings
        user_emb = h[user_mask]  # (num_users, hidden_dim)
        item_emb = h[item_mask]  # (num_items, hidden_dim)
        
        return user_emb, item_emb
    
    def predict(self, user_emb, item_emb, user_idx, item_idx, use_rating_head=True):
        """
        Predict rating for user-item pair.
        
        Args:
            user_emb: User embeddings (num_users, hidden_dim)
            item_emb: Item embeddings (num_items, hidden_dim)
            user_idx: User index (scalar or tensor)
            item_idx: Item index (scalar or tensor)
            use_rating_head: Whether to use rating prediction head (default: True)
            
        Returns:
            torch.Tensor: Predicted rating scores
        """
        # Get embeddings for specific user/item
        if isinstance(user_idx, int):
            user_emb_selected = user_emb[user_idx:user_idx+1]
        else:
            user_emb_selected = user_emb[user_idx]
        
        if isinstance(item_idx, int):
            item_emb_selected = item_emb[item_idx:item_idx+1]
        else:
            item_emb_selected = item_emb[item_idx]
        
        # Dot product for score
        scores = (user_emb_selected * item_emb_selected).sum(dim=1, keepdim=True)
        
        if use_rating_head and hasattr(self, 'rating_head'):
            # Use learned mapping to [1, 5]
            scores = self.rating_head(scores).squeeze(-1)
            scores = torch.clamp(scores, 1.0, 5.0)
        else:
            # Fallback: simple clipping (for BPR-only mode)
            scores = scores.squeeze(-1)
        
        return scores


if __name__ == "__main__":
    # Test GraphSAGE model
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from gnn.graph_data_loader import build_bipartite_graph
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
    
    print("Initializing GraphSAGE model...")
    user_feat_dim = graph_data.x[graph_data.node_type == 0].size(1)
    item_feat_dim = graph_data.x[graph_data.node_type == 1].size(1)
    
    model = GraphSAGERecommender(
        num_users=graph_data.num_users,
        num_items=graph_data.num_items,
        user_feat_dim=user_feat_dim,
        item_feat_dim=item_feat_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        aggregator='max'
    )
    
    print(f"\nModel Architecture:")
    print(f"  Input feature dim: {user_feat_dim} (users), {item_feat_dim} (items)")
    print(f"  Hidden dimension: {model.hidden_dim}")
    print(f"  Number of layers: {model.num_layers}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(graph_data)
    
    print(f"  User embeddings shape: {user_emb.shape}")
    print(f"  Item embeddings shape: {item_emb.shape}")
    
    # Test prediction
    print("\nTesting prediction...")
    user_idx = 0
    item_idx = 0
    score = model.predict(user_emb, item_emb, user_idx, item_idx)
    print(f"  Prediction for user {user_idx}, item {item_idx}: {score.item():.4f}")
    
    print("\nModel test successful!")
