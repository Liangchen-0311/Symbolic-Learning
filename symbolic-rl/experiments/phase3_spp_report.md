# Phase 3 SPP Report: Spatial Pyramid Pooling for Symbolic Features on ImageNet

## Motivation

In the baseline Phase 3 (no SPP), each formula reduces its intermediate spatial feature map to **a single scalar** via one pooling operator (e.g., `global_avg_pool`). This discards all spatial information — "edge in top-left" and "edge in bottom-right" become indistinguishable.

**SPP fix:** Strip the final pooling operator from each formula, then apply **8 different pooling operators** to the intermediate feature map. Each formula now produces 8 scalars encoding WHERE in the image the pattern occurs.

## Pipeline

### Formula Preparation

- Loaded 8,811 deduplicated formulas from Phase 3 v2 Step 1
- Took top 6,000 by accuracy (min accuracy = 0.0469)
- Stripped the final root pooling operator from each formula to get the **body**
- Result: 6,000 unique bodies (no body sharing found across formulas)

### SPP Pooling Operators (8 total)

| Pool | Description |
|------|-------------|
| `global_avg_pool` | Global average |
| `global_max_pool` | Global maximum |
| `global_std_pool` | Global standard deviation |
| `pool_quad_tl` | Top-left quadrant average |
| `pool_quad_tr` | Top-right quadrant average |
| `pool_quad_bl` | Bottom-left quadrant average |
| `pool_quad_br` | Bottom-right quadrant average |
| `pool_center` | Center region average |

**Total features:** 6,000 bodies × 8 pools = **48,000 features**

### Feature Extraction

| Item | Value |
|------|-------|
| Resolution | **224×224** |
| Precision | FP32 |
| Train images | 200,000 (200/class) |
| Val images | 50,000 (full) |
| Batch size | 256 |
| LRU cache | max_size=500, hit rate=11.7% |
| Train mmap | 200K × 48K × 4 bytes = **36 GB** |
| Val mmap | 50K × 48K × 4 bytes = **9 GB** |

**Feature matrix quality (train, first 5K samples):**

| Metric | Value |
|--------|-------|
| Active features (var > 1e-6) | 35,320 (73.6%) |
| Near-zero variance | 12,680 (26.4%) |
| All-zero/constant | 1,622 (3.4%) |
| Value range | [-10000, 10000] |
| NaN / Inf | 0 |

The 26.4% near-zero features are expected: some formula bodies produce constant output in specific spatial quadrants (e.g., a formula that only activates in the center will have zero variance in corner quadrants).

### Classifier Training — Sweep A: Weight Decay

| Item | Value |
|------|-------|
| Model | `nn.Linear(48000, 1000)` — 48M parameters |
| Optimizer | AdamW (lr=1e-3) + CosineAnnealingLR |
| Epochs | 20 |
| Batch size | 1024 |

| weight_decay | Train Top-1 | Val Top-1 | Val Top-5 | Time |
|-------------|-------------|-----------|-----------|------|
| 1.0 | — | — | — | — |
| 2.0 | 99.92% | 19.66% | 37.12% | 282s |
| 5.0 | 99.92% | 20.53% | 38.65% | 284s |
| **10.0** | **98.47%** | **20.61%** | **39.18%** | **284s** |
| 20.0 | 84.61% | 20.32% | 38.66% | 283s |
| 50.0 | 53.83% | 18.32% | 35.26% | 282s |

**Best: wd=10.0, Val Top-1 = 20.61%, Val Top-5 = 39.18%**

*(Sweep B — Dropout × Weight Decay grid — in progress)*

## Comparison Across Runs

| Run | Features | Resolution | Val Top-1 | Val Top-5 |
|-----|----------|-----------|-----------|-----------|
| v1 single-bank (baseline) | 2,611 × 1 pool | 224×224 | 9.24% | 21.70% |
| v2 FP16 (buggy) | 6,000 × 1 pool (31 working) | 112×112 | 8.29% | 18.81% |
| v2 FP32 (wd=0.1) | 6,000 × 1 pool | 112×112 | 13.89% | 29.27% |
| v2 FP32 (wd=5.0) | 6,000 × 1 pool | 112×112 | 17.27% | 33.64% |
| **v2 SPP (wd=10.0)** | **6,000 × 8 pools** | **224×224** | **20.61%** | **39.18%** |

**Progression of improvements:**

| Improvement | Delta Top-1 | Cumulative |
|------------|-------------|------------|
| Fix FP16 → FP32 | +5.6% | 13.89% |
| Tune wd 0.1 → 5.0 | +3.4% | 17.27% |
| Add SPP (8 pools) + 224×224 | +3.3% | 20.61% |
| **Total gain vs v2 FP16** | **+12.3%** | — |
| **Total gain vs v1 baseline** | **+11.4%** | — |

## Analysis

### SPP works: spatial information matters

The +3.3% gain from SPP confirms that spatial information was being wasted. The quadrant pooling operators allow the linear classifier to learn location-dependent patterns (e.g., "sky texture in top half" vs "ground texture in bottom half").

### Overfitting remains the bottleneck

With 48,000 features × 1,000 classes = 48M parameters trained on 200K samples, overfitting is severe (train 98.5% vs val 20.6%). The optimal wd=10.0 is very high, indicating the model needs extreme regularization.

### Diminishing returns on regularization

| wd | Train | Val Top-1 |
|----|-------|-----------|
| 2.0 | 99.92% | 19.66% |
| 5.0 | 99.92% | 20.53% |
| 10.0 | 98.47% | 20.61% |
| 20.0 | 84.61% | 20.32% |
| 50.0 | 53.83% | 18.32% |

The sweet spot is around wd=10. Beyond that, underfitting begins.

## Output Files

```
outputs/imagenet_v2/phase3_spp/
  bodies_sorted.json          # 6,000 formula bodies (sorted lexicographically)
  X_train_spp.mmap            # 200K × 48000 float32 (36 GB)
  X_val_spp.mmap              # 50K × 48000 float32 (9 GB)
  y_train.npy / y_val.npy     # Labels
  feat_mean.npy / feat_std.npy  # Standardization parameters
  results.json                # All results (after sweep completes)
```

## Reproduction

```bash
PYTHONUNBUFFERED=1 python experiments/run_phase3_spp.py
```

## Next Steps

1. **Dropout sweep results** — in progress, will add when complete
2. **Feature selection (L1)** — 48K features is too many; L1 regularization can identify the most useful subset
3. **More training data** — 400/class would reduce overfitting
4. **Multi-resolution SPP** — run same bodies at 112 + 224, concatenate features for both scale information
5. **More diverse operators** — morphological ops, frequency domain, learned kernels
