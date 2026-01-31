# GraphSAGE for Recommender Systems

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Architecture](#architecture)
4. [Inductive Learning Capability](#inductive-learning-capability)
5. [Implementation Considerations](#implementation-considerations)
6. [When to Use GraphSAGE](#when-to-use-graphsage)
7. [Comparison with Other Approaches](#comparison-with-other-approaches)
8. [Code Examples](#code-examples)
9. [References](#references)

---

## Overview

**GraphSAGE: Inductive Representation Learning on Large Graphs** (NeurIPS 2017)

GraphSAGE (Graph Sample and Aggregate) is designed for inductive learning, making it particularly useful for cold start scenarios in recommendations. Unlike transductive methods (LightGCN, NGCF), GraphSAGE can generate embeddings for nodes not seen during training.

### Key Innovation

**Inductive Learning**: Can generate embeddings for nodes not seen during training by learning aggregation functions that operate on node features and neighborhood structure.

### Main Characteristics

- **Neighborhood Sampling**: Samples fixed-size neighborhoods instead of using all neighbors
- **Aggregator Functions**: Multiple options for aggregating neighbor information
- **Feature-Based**: Incorporates node features for better cold start handling
- **Scalable**: Sampling enables handling large graphs efficiently

---

## Theoretical Foundation

### Neighborhood Sampling

Instead of using all neighbors, GraphSAGE samples fixed-size neighborhoods:
- Sample $S$ neighbors at each layer
- Reduces computation from $O(d^k)$ to $O(S^k)$ for k layers
- Enables inductive learning and scalability

**Benefits**:
- Fixed computational budget per node
- Memory efficient
- Enables mini-batch training
- Better generalization

### Aggregator Functions

GraphSAGE supports multiple aggregator functions for combining neighbor information:

#### 1. Mean Aggregator

$$h_{\mathcal{N}(v)} = \text{MEAN}(\{h_u : u \in \mathcal{N}(v)\})$$

**Properties**:
- Simple and efficient
- Permutation invariant
- Works well in practice
- Most commonly used

**When to Use**:
- Default choice for most applications
- When all neighbors should be weighted equally
- Computational efficiency is important

#### 2. LSTM Aggregator

- Treats neighbors as a sequence
- Applies LSTM to neighbor embeddings
- More expressive but less permutation-invariant

**Properties**:
- Higher capacity
- Order-dependent (requires random ordering)
- Slower than mean aggregator

**When to Use**:
- When neighbor ordering matters
- Need maximum expressiveness
- Have sufficient training data

#### 3. Pooling Aggregator

$$h_{\mathcal{N}(v)} = \text{MAX}(\{\sigma(W_{pool} h_u + b) : u \in \mathcal{N}(v)\})$$

**Properties**:
- Element-wise max pooling
- More expressive than mean
- Permutation invariant
- Captures distinctive features

**When to Use**:
- Want more expressiveness than mean
- Need permutation invariance
- Moderate computational budget

### Node Update Rule

After aggregating neighbor information, GraphSAGE updates the node embedding:

$$h_v^{(k)} = \sigma(W^{(k)} \cdot \text{CONCAT}(h_v^{(k-1)}, h_{\mathcal{N}(v)}^{(k-1)}))$$

where:
- $h_v^{(k-1)}$: node's own embedding from previous layer
- $h_{\mathcal{N}(v)}^{(k-1)}$: aggregated neighbor embeddings
- $W^{(k)}$: learnable weight matrix at layer $k$
- $\sigma$: non-linear activation (ReLU, LeakyReLU, etc.)
- CONCAT: concatenation of self and neighborhood

**Key Insight**: Concatenating self and neighborhood embeddings allows the model to learn how much to weight:
- The node's own features
- Information from its neighbors

---

## Architecture

### Layer Structure

A typical GraphSAGE layer performs:

1. **Sample Neighbors**: Sample $S$ neighbors for each node
2. **Aggregate**: Apply aggregator function to neighbor embeddings
3. **Combine**: Concatenate node's own embedding with aggregated neighbors
4. **Transform**: Apply linear transformation and activation

### Multi-Layer Architecture

**Typical Setup**:
- **2 layers**: Most common, captures 2-hop neighborhood
- **3 layers**: For larger graphs or more complex patterns
- **Embedding dimension**: 64-256 (typically 64 or 128)

**Layer-wise Sampling**:
- Layer 1: Sample $S_1$ neighbors (e.g., 25)
- Layer 2: Sample $S_2$ neighbors (e.g., 10)
- Total neighborhood size: $S_1 \times S_2$ (e.g., 250)

**Receptive Field**:
- After $k$ layers, each node aggregates information from its $k$-hop neighborhood
- Sampling controls the size of this neighborhood

### Forward Propagation

For a 2-layer GraphSAGE on a user-item bipartite graph:

**Layer 1**:
- For each user, sample items they interacted with
- For each item, sample users who interacted with it
- Aggregate and update

**Layer 2**:
- For each user, aggregate from Layer 1 item embeddings
- For each item, aggregate from Layer 1 user embeddings
- Final embeddings

**Prediction**:
$$\hat{y}_{ui} = h_u^{(2)T} h_i^{(2)}$$

---

## Inductive Learning Capability

### Transductive vs. Inductive

**Transductive Learning** (LightGCN, NGCF):
- Learns embeddings for specific nodes seen during training
- Cannot handle new nodes without retraining
- Stores lookup table of embeddings

**Inductive Learning** (GraphSAGE):
- Learns aggregation functions, not fixed embeddings
- Can generate embeddings for new nodes
- Uses node features and graph structure

### How Inductive Learning Works

**Training Phase**:
1. Learn aggregator functions $\text{AGG}^{(k)}$
2. Learn weight matrices $W^{(k)}$
3. Use known nodes and their features

**Inference Phase for New Node**:
1. Extract node features
2. Sample neighborhood from graph
3. Apply learned aggregator functions
4. Generate embedding on-the-fly

### Cold Start Handling

**New Users**:
- Initialize with user features (demographics, profile)
- Sample from items they've interacted with (even if just 1-2)
- Generate embedding using learned functions

**New Items**:
- Initialize with item features (metadata, category, description)
- Sample from users who interacted (if any)
- Generate embedding using learned functions

**Zero Interaction Case**:
- Use only node features
- Apply feature transformation
- Still get reasonable embedding (better than random)

### Benefits for Recommendations

1. **Real-time Onboarding**: New users get personalized recommendations immediately
2. **Dynamic Catalog**: New items can be recommended without retraining
3. **Reduced Retraining**: Model stays relevant longer
4. **Feature Utilization**: Leverages rich metadata effectively

---

## Implementation Considerations

### Sampling Size Selection

**Layer 1 (Closer to Target)**:
- Typical: $S_1 = 25$ neighbors
- Captures immediate connections
- More important neighbors

**Layer 2 (Further from Target)**:
- Typical: $S_2 = 10$ neighbors
- Captures broader context
- Less direct influence

**Trade-offs**:
- **Larger $S$**: 
  - More information
  - Better accuracy
  - Slower training/inference
  - More memory
- **Smaller $S$**: 
  - Faster
  - Less memory
  - May miss important neighbors
  - Lower accuracy

**Recommendations**:
- Start with $S_1=25, S_2=10$
- Reduce if memory/speed issues
- Increase if accuracy not satisfactory
- Validate on held-out data

### Aggregator Choice for RecSys

**Mean Aggregator**:
- **Pros**: Simple, fast, works well
- **Cons**: Treats all neighbors equally
- **Best for**: Most recommendation tasks, default choice

**Pooling Aggregator**:
- **Pros**: More expressive, captures distinctive features
- **Cons**: Slightly slower, more parameters
- **Best for**: When diversity in neighbors matters

**LSTM Aggregator**:
- **Pros**: Most expressive, can capture order
- **Cons**: Slowest, not permutation invariant, needs more data
- **Best for**: Sequential data, temporal patterns

**Empirical Finding**: Mean aggregator often performs comparably to more complex ones in RecSys tasks, while being simpler and faster.

### Node Features

**For Users**:
- Demographics (age, gender, location)
- Profile information
- Historical statistics (avg rating, activity level)
- Engagement features

**For Items**:
- Metadata (category, brand, price)
- Content features (text embeddings, image embeddings)
- Popularity statistics
- Temporal features (release date, trending score)

**Feature Engineering**:
- Normalize numerical features
- Embed categorical features
- Combine multiple feature types
- Pre-compute complex features

**Feature Dimension**:
- Typical: 64-256 dimensions after embedding
- Balance between expressiveness and efficiency
- Can use feature projection layer

### Training Strategy

**Mini-Batch Training**:
1. Sample batch of user-item pairs
2. For each node in batch, sample neighborhoods
3. Compute embeddings via aggregation
4. Calculate loss and update

**Negative Sampling**:
- Sample negative items for each positive pair
- Ratio: 1:1 to 1:4 (positive:negative)
- Can use uniform or popularity-based sampling

**Loss Functions**:
- **BPR Loss**: Most common for implicit feedback
- **Cross-Entropy**: For binary classification
- **Margin Loss**: Maximize separation between positive and negative

**Optimization**:
- Adam optimizer typically works well
- Learning rate: 0.001-0.01
- L2 regularization: 0.0001-0.001
- Batch size: 512-2048

### Memory and Computation

**Memory Requirements**:
- Smaller than full graph methods (due to sampling)
- Stores: model parameters, sampled subgraphs, features
- Scales with batch size and sampling size

**Computation Complexity**:
- Per node: $O(S^k \cdot d^2)$ for $k$ layers, $d$ dimensions, $S$ samples
- More expensive than LightGCN due to:
  - Feature transformations
  - Concatenation operations
  - Non-linear activations

**Optimization Tips**:
- Use sparse operations for features
- Pre-compute and cache frequent neighborhoods
- Batch neighborhood sampling
- GPU acceleration for aggregations

---

## When to Use GraphSAGE

### Ideal Use Cases

✅ **Cold Start is Critical**:
- Frequent new users/items
- Need immediate recommendations
- Can't afford retraining

✅ **Rich Node Features Available**:
- Have user demographics, item metadata
- Want to leverage content information
- Hybrid content-collaborative filtering

✅ **Inductive Learning Needed**:
- Graph evolves rapidly
- New nodes added constantly
- Want to generalize to unseen nodes

✅ **Dynamic Graphs**:
- User preferences change
- Catalog updates frequently
- Real-time recommendations

### When NOT to Use GraphSAGE

❌ **Pure Collaborative Filtering**:
- No node features available
- Only have interaction data
- LightGCN simpler and often better

❌ **Static Small Graphs**:
- Few new users/items
- Inductive capability not needed
- Simpler methods sufficient

❌ **Extreme Efficiency Requirements**:
- Need fastest possible inference
- Sampling overhead not acceptable
- Pre-computed embeddings preferred

❌ **Limited Feature Engineering Resources**:
- Can't create/maintain node features
- Features are poor quality
- Feature-less methods better

---

## Comparison with Other Approaches

### GraphSAGE vs. LightGCN

| Aspect | GraphSAGE | LightGCN |
|--------|-----------|----------|
| **Learning Type** | Inductive | Transductive |
| **Node Features** | Required | Not used |
| **Cold Start** | Excellent | Poor |
| **Simplicity** | Moderate | Very Simple |
| **Efficiency** | Moderate (sampling) | High |
| **New Nodes** | Handle without retrain | Require retrain |
| **Feature Engineering** | Important | Not applicable |
| **Best For** | Cold start, dynamic graphs | Pure CF, static graphs |

**When to Choose**:
- **GraphSAGE**: Cold start critical, have features, dynamic setting
- **LightGCN**: Pure CF, efficiency priority, static setting

### GraphSAGE vs. NGCF

| Aspect | GraphSAGE | NGCF |
|--------|-----------|------|
| **Learning Type** | Inductive | Transductive |
| **Node Features** | Required | Optional |
| **Interaction Modeling** | Via features | Explicit (element-wise product) |
| **Cold Start** | Good | Poor |
| **Complexity** | Moderate | Higher |
| **Parameters** | Moderate | More |

**When to Choose**:
- **GraphSAGE**: Inductive learning, cold start, have features
- **NGCF**: Explicit interaction modeling, static graph

### GraphSAGE vs. PinSage

| Aspect | GraphSAGE | PinSage |
|--------|-----------|---------|
| **Sampling** | Uniform/Fixed-size | Random walk-based |
| **Scale** | Millions | Billions |
| **Complexity** | Moderate | High |
| **Infrastructure** | Single machine | Distributed |
| **Industry Proven** | Research | Production (Pinterest) |

**When to Choose**:
- **GraphSAGE**: Up to millions of nodes, single machine
- **PinSage**: Billions of nodes, distributed infrastructure

### Performance Trade-offs

**Accuracy**: GraphSAGE typically competitive with other GNNs, especially in cold start scenarios

**Training Time**: 
- Slower than LightGCN (feature transformations)
- Faster than full-graph methods (sampling)
- Comparable to NGCF

**Inference Speed**:
- Slower than pre-computed embeddings
- Faster than full aggregation
- Can cache for frequent nodes

**Memory Usage**:
- Less than full-graph storage
- More than pure embedding lookup
- Scales with sampling size

---

## Code Examples

### Basic GraphSAGE Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSAGELayer(nn.Module):
    """
    Single GraphSAGE layer with mean aggregator.
    """
    def __init__(self, in_dim, out_dim):
        super(GraphSAGELayer, self).__init__()
        # Weight matrix for combining self and neighbor embeddings
        self.linear = nn.Linear(in_dim * 2, out_dim)
        
    def forward(self, self_feats, neighbor_feats):
        """
        Args:
            self_feats: [batch_size, in_dim] - node's own features
            neighbor_feats: [batch_size, num_neighbors, in_dim] - neighbor features
        Returns:
            [batch_size, out_dim] - updated node embeddings
        """
        # Mean aggregation over neighbors
        agg_neighbor = neighbor_feats.mean(dim=1)  # [batch_size, in_dim]
        
        # Concatenate self and aggregated neighbor
        combined = torch.cat([self_feats, agg_neighbor], dim=1)  # [batch_size, in_dim*2]
        
        # Linear transformation and activation
        output = F.relu(self.linear(combined))  # [batch_size, out_dim]
        
        # L2 normalization (common in GraphSAGE)
        output = F.normalize(output, p=2, dim=1)
        
        return output


class GraphSAGE(nn.Module):
    """
    Multi-layer GraphSAGE model for recommendations.
    """
    def __init__(self, feature_dim, hidden_dim=64, num_layers=2):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        
        # GraphSAGE layers
        self.layers = nn.ModuleList()
        self.layers.append(GraphSAGELayer(feature_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GraphSAGELayer(hidden_dim, hidden_dim))
    
    def forward(self, node_features, sampled_neighbors):
        """
        Args:
            node_features: [batch_size, feature_dim] - features of target nodes
            sampled_neighbors: List of [batch_size, num_samples, feature_dim] 
                              for each layer
        Returns:
            [batch_size, hidden_dim] - final node embeddings
        """
        h = node_features
        
        for i, layer in enumerate(self.layers):
            h = layer(h, sampled_neighbors[i])
        
        return h


class GraphSAGERecommender(nn.Module):
    """
    Complete GraphSAGE-based recommender system.
    """
    def __init__(self, num_users, num_items, user_feature_dim, 
                 item_feature_dim, hidden_dim=64, num_layers=2):
        super(GraphSAGERecommender, self).__init__()
        
        # User and item GraphSAGE models
        self.user_model = GraphSAGE(user_feature_dim, hidden_dim, num_layers)
        self.item_model = GraphSAGE(item_feature_dim, hidden_dim, num_layers)
        
        # Feature embeddings (if using ID features)
        self.user_feature_embedding = nn.Embedding(num_users, user_feature_dim)
        self.item_feature_embedding = nn.Embedding(num_items, item_feature_dim)
        
    def forward(self, user_ids, item_ids, user_neighbors, item_neighbors):
        """
        Args:
            user_ids: [batch_size] - user IDs
            item_ids: [batch_size] - item IDs
            user_neighbors: List of neighbor features for users
            item_neighbors: List of neighbor features for items
        Returns:
            [batch_size] - prediction scores
        """
        # Get user and item features
        user_features = self.user_feature_embedding(user_ids)
        item_features = self.item_feature_embedding(item_ids)
        
        # Get embeddings via GraphSAGE
        user_emb = self.user_model(user_features, user_neighbors)
        item_emb = self.item_model(item_features, item_neighbors)
        
        # Dot product for prediction
        scores = (user_emb * item_emb).sum(dim=1)
        
        return scores
    
    def predict_for_user(self, user_id, all_item_ids, user_neighbors, 
                        all_item_neighbors):
        """
        Generate recommendations for a single user.
        """
        batch_size = len(all_item_ids)
        user_ids = torch.full((batch_size,), user_id, dtype=torch.long)
        
        scores = self.forward(user_ids, all_item_ids, user_neighbors, 
                             all_item_neighbors)
        
        return scores
```

### Neighborhood Sampling

```python
import random
import numpy as np

class NeighborhoodSampler:
    """
    Samples fixed-size neighborhoods for GraphSAGE.
    """
    def __init__(self, graph, sample_sizes):
        """
        Args:
            graph: Dictionary mapping node_id -> list of neighbor_ids
            sample_sizes: List of sample sizes for each layer [S1, S2, ...]
        """
        self.graph = graph
        self.sample_sizes = sample_sizes
    
    def sample_neighbors(self, nodes, num_samples):
        """
        Sample neighbors for given nodes.
        
        Args:
            nodes: List of node IDs
            num_samples: Number of neighbors to sample
        Returns:
            Dictionary mapping node_id -> sampled neighbor IDs
        """
        sampled = {}
        
        for node in nodes:
            neighbors = self.graph.get(node, [])
            
            if len(neighbors) >= num_samples:
                # Sample without replacement
                sampled[node] = random.sample(neighbors, num_samples)
            else:
                # Sample with replacement if not enough neighbors
                sampled[node] = random.choices(neighbors, k=num_samples)
        
        return sampled
    
    def sample_multi_layer(self, target_nodes):
        """
        Sample neighborhoods for multiple layers.
        
        Args:
            target_nodes: List of target node IDs
        Returns:
            List of sampled neighborhoods for each layer
        """
        all_samples = []
        current_nodes = target_nodes
        
        for num_samples in self.sample_sizes:
            # Sample neighbors for current nodes
            sampled = self.sample_neighbors(current_nodes, num_samples)
            all_samples.append(sampled)
            
            # Next layer samples from these neighbors
            current_nodes = []
            for neighbors in sampled.values():
                current_nodes.extend(neighbors)
            current_nodes = list(set(current_nodes))  # Remove duplicates
        
        return all_samples


# Usage example
def create_user_item_graph(interactions):
    """
    Create bipartite graph from user-item interactions.
    
    Args:
        interactions: List of (user_id, item_id) tuples
    Returns:
        user_graph: user_id -> list of item_ids
        item_graph: item_id -> list of user_ids
    """
    user_graph = {}
    item_graph = {}
    
    for user_id, item_id in interactions:
        # Add to user graph
        if user_id not in user_graph:
            user_graph[user_id] = []
        user_graph[user_id].append(item_id)
        
        # Add to item graph
        if item_id not in item_graph:
            item_graph[item_id] = []
        item_graph[item_id].append(user_id)
    
    return user_graph, item_graph


# Example usage
interactions = [(0, 10), (0, 11), (1, 10), (1, 12), (2, 11), (2, 13)]
user_graph, item_graph = create_user_item_graph(interactions)

sampler = NeighborhoodSampler(user_graph, sample_sizes=[25, 10])
target_users = [0, 1]
samples = sampler.sample_multi_layer(target_users)
print("Sampled neighborhoods:", samples)
```

### Training Loop

```python
import torch.optim as optim

def train_graphsage(model, train_data, sampler, num_epochs=10, 
                   batch_size=256, lr=0.001):
    """
    Training loop for GraphSAGE recommender.
    
    Args:
        model: GraphSAGERecommender model
        train_data: List of (user_id, item_id) positive pairs
        sampler: NeighborhoodSampler for sampling neighborhoods
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Shuffle training data
        random.shuffle(train_data)
        
        # Mini-batch training
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            # Extract user and item IDs
            user_ids = torch.tensor([u for u, _ in batch], dtype=torch.long).to(device)
            pos_item_ids = torch.tensor([i for _, i in batch], dtype=torch.long).to(device)
            
            # Sample negative items
            neg_item_ids = torch.randint(0, model.item_model.num_items, 
                                        (len(batch),), dtype=torch.long).to(device)
            
            # Sample neighborhoods (simplified - should sample actual neighbors)
            user_neighbors = sampler.sample_multi_layer(user_ids.cpu().tolist())
            pos_item_neighbors = sampler.sample_multi_layer(pos_item_ids.cpu().tolist())
            neg_item_neighbors = sampler.sample_multi_layer(neg_item_ids.cpu().tolist())
            
            # Forward pass for positive pairs
            pos_scores = model(user_ids, pos_item_ids, user_neighbors, pos_item_neighbors)
            
            # Forward pass for negative pairs
            neg_scores = model(user_ids, neg_item_ids, user_neighbors, neg_item_neighbors)
            
            # BPR loss
            loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model
```

### Handling Cold Start

```python
def handle_cold_start_user(model, user_features, sampler, all_items, k=10):
    """
    Generate recommendations for a cold start user.
    
    Args:
        model: Trained GraphSAGE model
        user_features: Feature vector for new user
        sampler: Neighborhood sampler
        all_items: List of all item IDs
        k: Number of recommendations
    Returns:
        Top-K recommended item IDs
    """
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # For cold start, use only user features (no neighbors yet)
        # Create dummy user ID
        user_id = torch.tensor([0], dtype=torch.long).to(device)
        
        # Get user embedding using features only
        user_emb = model.user_model(
            user_features.unsqueeze(0).to(device), 
            [torch.zeros(1, 25, user_features.size(0)).to(device),  # No real neighbors
             torch.zeros(1, 10, user_features.size(0)).to(device)]
        )
        
        # Get all item embeddings
        item_ids = torch.tensor(all_items, dtype=torch.long).to(device)
        item_neighbors = sampler.sample_multi_layer(all_items)
        
        # Batch process items
        batch_size = 256
        all_scores = []
        
        for i in range(0, len(all_items), batch_size):
            batch_item_ids = item_ids[i:i+batch_size]
            batch_neighbors = [n[i:i+batch_size] for n in item_neighbors]
            
            # Get item embeddings
            item_emb = model.item_model(
                model.item_feature_embedding(batch_item_ids),
                batch_neighbors
            )
            
            # Compute scores
            scores = torch.matmul(user_emb, item_emb.t()).squeeze()
            all_scores.append(scores)
        
        # Combine all scores
        all_scores = torch.cat(all_scores)
        
        # Get top-K
        top_k_scores, top_k_indices = torch.topk(all_scores, k)
        top_k_items = [all_items[idx] for idx in top_k_indices.cpu().tolist()]
    
    return top_k_items, top_k_scores.cpu().tolist()


# Example usage for cold start
def demo_cold_start():
    # Assume we have a trained model
    # model = trained_graphsage_model
    # sampler = trained_neighborhood_sampler
    
    # New user features (e.g., demographics)
    new_user_features = torch.randn(64)  # 64-dim feature vector
    
    # All available items
    all_items = list(range(1000))  # 1000 items
    
    # Get recommendations
    # recommended_items, scores = handle_cold_start_user(
    #     model, new_user_features, sampler, all_items, k=10
    # )
    
    # print("Recommended items:", recommended_items)
    # print("Scores:", scores)
    pass
```

---

## References

### Original Paper

- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). "Inductive Representation Learning on Large Graphs". *NeurIPS 2017*. https://arxiv.org/abs/1706.02216

### Related Papers

- **FastGCN**: Chen, J., Ma, T., & Xiao, C. (2018). "FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling". *ICLR 2018*.

- **PinSage**: Ying, R., et al. (2018). "Graph Convolutional Neural Networks for Web-Scale Recommender Systems". *KDD 2018*. (Industrial application of GraphSAGE ideas)

### Implementation Resources

- **Official Implementation**: https://github.com/williamleif/GraphSAGE
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv
- **DGL**: https://docs.dgl.ai/en/latest/generated/dgl.nn.pytorch.conv.SAGEConv.html

### Tutorials and Guides

- PyTorch Geometric Tutorial: https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
- DGL GraphSAGE Tutorial: https://docs.dgl.ai/en/latest/tutorials/blitz/4_link_predict.html

### Related Documentation

- Main GNN Guide: [`GNN_RECOM_SYS_GUIDE.md`](GNN_RECOM_SYS_GUIDE.md)
- Recommender Systems Guide: [`../recom_sys_guide.md`](../recom_sys_guide.md)
- GraphSAGE Implementation: [`graphsage_model.py`](graphsage_model.py)

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Extracted From**: GNN_RECOM_SYS_GUIDE.md

---

*This document provides comprehensive coverage of GraphSAGE for recommender systems, focusing on its unique inductive learning capability and cold start handling. For broader context on GNN-based recommendations, refer to the main GNN guide.*
