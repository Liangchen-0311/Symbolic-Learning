# Phase 3 Report: Symbolic Feature Extraction + Linear Classification on ImageNet

## Overview

Phase 3 takes the 10,039 symbolic formulas discovered in Phase 1 (RL-based formula discovery across 4 independent banks), deduplicates them, extracts scalar features from ImageNet images, and trains a linear classifier (`nn.Linear`, no MLP) for 1000-class classification.

**Script:** `experiments/run_phase3_v2.py`
**Run date:** 2026-03-30
**Hardware:** 1× NVIDIA A100-SXM4-80GB

---

## Pipeline

### Step 1: Fast Deduplication (~14 min)

| Item | Value |
|------|-------|
| Input | 10,039 unique formulas (4 banks × ~2,500 each) |
| Resolution | 64×64 |
| Eval images | 1,000 (1 per class, stratified) |
| Method | Greedy Pearson correlation (threshold=0.88, accuracy-descending) |

**Process:**
1. Loaded formulas from 4 Phase-1 banks (`outputs/imagenet_v2/phase1/bank_{0..3}/feature_bank/`).
2. Executed all 10,039 formulas on 1,000 images at 64×64 (28 seconds on GPU).
3. Removed 10 invalid + 303 constant-output formulas → 9,726 remaining.
4. Greedy Pearson dedup: 9,726 → **8,811 formulas**.
5. Capped to **top 6,000 by accuracy** (min accuracy = 0.0469).

**Output:** `outputs/imagenet_v2/phase3_v2/dedup_formulas.json`

### Step 2: Feature Matrix Extraction

| Item | Value |
|------|-------|
| Formulas | 6,000 (sorted lexicographically) |
| Resolution | 112×112 |
| Precision | **FP32** |
| Train images | 200,000 (200 per class) |
| Val images | 50,000 (full val set) |
| LRU cache | max_size=500, hit rate=10.9% |
| Train extraction time | 163.5 min |
| Val extraction time | ~40 min |
| **Total Step 2 time** | **~3.4 hours** |

**Process:**
1. Formulas sorted lexicographically to maximize LRU sub-expression cache hits.
2. Image preprocessing: `Resize(144) → CenterCrop(112) → ToTensor()` (raw [0,1] pixels, no ImageNet mean/std normalization).
3. For each batch (512 images): executed 6,000 formulas, accumulated outputs on GPU, single GPU→CPU transfer.
4. Features saved as memory-mapped float32 files with checkpoint/resume support.

**Key optimizations applied:**
- **No intermediate NaN/Inf GPU-CPU sync:** Replaced per-operator `torch.isnan().any()` checks (GPU-CPU sync points, ~3.7x overhead) with `torch.nan_to_num()` (GPU-only).
- **GPU-side accumulation:** All 6,000 formula outputs accumulated in GPU tensor, single `.cpu().numpy()` per batch.
- **LRU sub-expression cache:** Cached intermediate spatial tensors for prefix-sharing formulas.

**Output files:**
- `X_train.mmap`: 200,000 × 6,000 float32 (4.80 GB)
- `X_val.mmap`: 50,000 × 6,000 float32 (1.20 GB)

### Step 3: Linear Classifier Training (~4 min)

| Item | Value |
|------|-------|
| Model | `nn.Linear(6000, 1000)` — 6M parameters |
| Optimizer | AdamW (lr=1e-3) + CosineAnnealingLR |
| Epochs | 20 |
| Batch size | 1024 |
| Standardization | Online mean/std from training mmap |

---

## Results

### Weight Decay Sweep

| weight_decay | Train Top-1 | Val Top-1 | Val Top-5 | Time |
|-------------|-------------|-----------|-----------|------|
| 1e-4 | 99.92% | 13.84% | 28.94% | 61.0s |
| 1e-3 | 99.92% | 13.87% | 29.00% | 66.1s |
| 1e-2 | 99.92% | 13.87% | 28.95% | 54.1s |
| **1e-1** | **99.92%** | **13.89%** | **29.27%** | **53.2s** |

**Best: weight_decay=0.1, Val Top-1 = 13.89%, Val Top-5 = 29.27%**

### Per-Superclass Accuracy (best model, wd=0.1)

| Superclass | Accuracy |
|-----------|----------|
| mammal_wild | 17.40% |
| fish_aquatic | 17.00% |
| musical_instrument | 16.08% |
| food_fruit | 15.92% |
| tool_implement | 14.36% |
| insect_arthropod | 14.08% |
| clothing_fabric | 14.04% |
| food_other | 13.96% |
| bird | 13.92% |
| mammal_pet | 13.56% |
| electronic_device | 13.40% |
| container_vessel | 13.40% |
| primate | 13.28% |
| furniture_indoor | 13.28% |
| natural_scene_misc | 13.00% |
| plant_flower | 12.60% |
| vehicle_land | 12.52% |
| vehicle_water_air | 12.44% |
| structure_building | 11.92% |
| reptile_amphibian | 11.56% |

### Comparison with v1 (single-bank) and FP16 buggy run

| Run | Formulas | Resolution | Train Top-1 | Val Top-1 | Val Top-5 |
|-----|----------|-----------|-------------|-----------|-----------|
| v1 (single bank, 224×224) | 2,611 | 224×224 | 13.28% | 9.24% | 21.70% |
| v2 FP16 (buggy — 99.5% zeros) | 6,000 (31 working) | 112×112 | 12.32% | 8.29% | 18.81% |
| **v2 FP32** | **6,000** | **112×112** | **99.92%** | **13.89%** | **29.27%** |

---

## Analysis

### FP16 Bug Postmortem

The first v2 run used FP16 for operator execution. This caused two fatal issues:

1. **Conv2d dtype mismatch:** Operator kernels (Sobel, Gaussian, Gabor) are created in FP32. When fed FP16 input, `F.conv2d` raised `RuntimeError`, caught silently by the try/except in the extraction loop → output set to 0.
2. **`_safe_binary` clamp overflow:** The decorator clamps to ±1e6, but FP16 max = 65504. The clamp value itself overflows to inf in FP16.

**Impact:** 5,969 of 6,000 formulas (99.5%) used convolution operators and produced all-zero features. Only 31 pure-arithmetic formulas worked. Despite this, the model achieved 8.29% val top-1, showing that even a handful of diverse features carry significant discriminative power.

### Overfitting is the Main Problem

The FP32 run reveals a clear overfitting pattern:

| Metric | Value |
|--------|-------|
| Train Top-1 | 99.92% |
| Val Top-1 | 13.89% |
| Gap | **86 percentage points** |

The model perfectly memorizes the training set but fails to generalize. With 6,000 features × 1,000 classes = 6M parameters trained on 200K samples (~33 samples/parameter), this is expected.

**Evidence that stronger regularization helps:** The best weight_decay (0.1) is 1000× larger than the v1 best (0.001). At wd=0.1, val top-5 is 29.27% vs 28.94% at wd=1e-4. The model still benefits from more regularization.

### Feature Redundancy

Comparing the FP16 buggy run (31 working formulas → 8.29%) with v1 (2,611 formulas → 9.24%) shows that adding ~2,600 more formulas only gains ~1% accuracy. This suggests massive information redundancy — Pearson dedup at r=0.88 removes linear correlation but not functional redundancy (many formulas capture similar visual concepts through different operator chains).

---

## Next Steps to Improve Accuracy

### 1. Stronger Regularization (immediate, minutes)
- Sweep larger weight_decay: [0.2, 0.5, 1.0, 2.0]
- Add dropout before the linear layer
- Try Elastic Net (L1 + L2) to simultaneously regularize and select features

### 2. L1 Feature Selection (immediate, minutes)
- Train with L1 penalty to zero out redundant features
- Identify the effective feature subset (expect 500–2000 out of 6000 to be non-zero)
- Retrain with only selected features

### 3. More Training Data
- Increase to 400/class (400K images) — reduces overfitting by doubling the sample/parameter ratio
- Requires re-running Step 2 (~6 hours at FP32)

### 4. Higher Resolution
- Use 224×224 instead of 112×112 — spatial operators (gabor, edge, blur) have more detail to work with
- v1 used 224×224 and may benefit from this

### 5. Training Improvements
- More epochs (50–80) with proper LR warmup
- Label smoothing to prevent overconfident predictions
- Mixup / CutMix style augmentation on the feature vectors

---

## Configuration Summary

```
Phase 1 banks:      outputs/imagenet_v2/phase1/bank_{0..3}/feature_bank/
Dedup threshold:    0.88 (Pearson correlation)
Max formulas:       6,000 (top by accuracy)
Resolution:         112×112
Precision:          FP32
Train images:       200,000 (200/class)
Val images:         50,000 (full)
LRU cache:          max_size=500, hit_rate=10.9%
Classifier:         nn.Linear(6000, 1000), no hidden layers
Optimizer:          AdamW, lr=1e-3, CosineAnnealingLR, 20 epochs
Best weight_decay:  0.1
```

## Output Files

```
outputs/imagenet_v2/phase3_v2/
  dedup_formulas.json        # 8,811 deduped formulas (full list)
  formula_list_sorted.json   # 6,000 formulas used (sorted lexicographically)
  X_train.mmap               # 200K × 6000 float32 (4.80 GB)
  X_val.mmap                 # 50K × 6000 float32 (1.20 GB)
  y_train.npy / y_val.npy    # Labels
  feat_mean.npy / feat_std.npy  # Standardization parameters
  final_results.json         # All results
```

## Reproduction

```bash
# Full pipeline (Step 1 + 2 + 3)
PYTHONUNBUFFERED=1 python experiments/run_phase3_v2.py --device cuda

# From Step 2 (dedup already saved)
PYTHONUNBUFFERED=1 python experiments/run_phase3_v2.py --device cuda --start_step 2

# From Step 3 only (features already extracted)
PYTHONUNBUFFERED=1 python experiments/run_phase3_v2.py --device cuda --start_step 3
```
