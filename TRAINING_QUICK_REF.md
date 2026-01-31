# GraphSAGE Model Training - Quick Reference

## One-Command Training

```bash
conda activate recommender
python scripts/train_model.py
```

**Result**: Trained model in `models/current/` (~30 seconds on GPU)

---

## Model Performance

```
Test RMSE:        1.0321
Test MAE:         0.8547
Precision@10:     67.3%
Recall@10:        73.1%
NDCG@10:          83.7%
```

---

## Model Details

**Architecture:**
- 3-layer GraphSAGE
- 64-dim embeddings
- Max pooling aggregator
- 28,017 parameters

**Training:**
- 17 epochs (early stopped)
- 512 batch size
- 0.001 learning rate
- MSE loss

**Data:**
- 943 users
- 1,651 movies
- 100,000 ratings
- 80/20 train/test split

---

## Artifacts

```
models/current/
├── graphsage_model.pth   (109 KB) - Model weights
├── preprocessor.pkl      (15 KB)  - Feature preprocessor
└── metadata.json         (2 KB)   - Training metadata
```

---

## Load Model

```python
import torch
from poc.graphsage_model import GraphSAGERecommender
from app.core.training.model_versioning import load_model_artifacts

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocessor, metadata = load_model_artifacts(
    GraphSAGERecommender,
    model_dir='models/current',
    device=device
)

print(f"Model loaded: v{metadata['model_version']}")
print(f"Test RMSE: {metadata['metrics']['val_rmse']:.4f}")
```

---

## Verify Model

```bash
python scripts/test_model.py
```

---

## Training Options

```bash
# Custom epochs
python scripts/train_model.py --epochs 30

# Larger model
python scripts/train_model.py --hidden-dim 128 --num-layers 4

# Different aggregator
python scripts/train_model.py --aggregator mean

# CPU training
python scripts/train_model.py --device cpu

# Quiet mode
python scripts/train_model.py --quiet
```

---

## Success Metrics

| Metric | Value | Status |
|--------|-------|--------|
| RMSE | 1.0321 | ✅ Good |
| Precision@10 | 67.3% | ✅ Strong |
| Recall@10 | 73.1% | ✅ Strong |
| Training Time | 31.5s | ✅ Fast |
| Model Size | 126 KB | ✅ Compact |

---

**Status**: Ready for Inference ✅  
**Version**: 1.0.0  
**Date**: 2026-01-31
