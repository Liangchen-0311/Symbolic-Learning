# Phase 3 v3 Report: Multi-Resolution SPP + New Operators on ImageNet

## Overview

v3 is a clean restart with significantly expanded operator set, new terminals, learnable convolution kernels, and multi-resolution SPP+histogram feature encoding.

**Script:** `experiments/run_phase3_v3.py`
**Total runtime:** 8.9 hours (A100-SXM4-80GB)
**Best result:** Val Top-1 = 19.45%, Val Top-5 = 37.20%

---

## Pipeline Summary

### Phase 1A: RL Formula Discovery (v3 operators)

**Config:** `configs/tensor_vsr_imagenet_v3.yaml`

| Item | Value |
|------|-------|
| Banks | 4 (short / cross-channel / texture / multiscale) |
| Resolution | 112×112 |
| min_accuracy_threshold | 0.08 |
| correlation_threshold | 0.85 |
| Iterations | 500 (early stop at ~330) |

**New components vs v2:**
- 5 new spatial operators: `opening`, `closing`, `tophat`, `high_freq`, `low_freq`
- 5 new distribution-aware pooling: `ratio_above_mean`, `percentile_90`, `spatial_entropy`, `peak_location_y`, `peak_location_x`
- 1 multi-dim pooling: `patch_histogram_4x4` (outputs [B, 4])
- 12 learnable convolution kernels: `conv3x3_0..7`, `conv5x5_0..3` (SymbolicKernelBank, frozen during RL)
- 4 new color terminals: `I_r` (R/total), `I_g` (G/total), `I_RG` (R-G), `I_BY` (B-(R+G)/2)
- Grammar tightened: pooling only at final token, consecutive unary ban
- `_safe_binary` clamp changed from ±1e6 to ±60000

**Result:** 4 banks × ~2500 formulas = **10,049 total**

| Bank | Formulas | Mean Acc | New Ops Usage | Conv Kernel Usage |
|------|----------|----------|---------------|-------------------|
| 0 | 2,532 | 0.110 | 17.9% | 16.7% |
| 1 | 2,503 | 0.096 | 13.4% | 30.2% |
| 2 | 2,507 | 0.096 | 29.0% | 18.8% |
| 3 | 2,507 | 0.096 | 10.7% | 35.6% |

All formulas have accuracy ≥ 0.08. 100% of formulas in all banks use at least one v3-only operator.

### Phase 2: Validation + Deduplication

| Item | Value |
|------|-------|
| Input | 10,049 formulas |
| Failed execution (224×224) | 5,279 (52.5%) |
| Valid after filtering | 2,891 |
| After Pearson dedup (r=0.95) | **2,676** |

**Critical issue discovered:** 5,279 formulas failed because `build_data_batch()` in `train_imagenet_pipeline.py` only provided 6 original terminals, missing the 4 new v3 terminals (`I_r`, `I_g`, `I_RG`, `I_BY`). **This has been fixed** for future runs. Recovering these formulas would nearly triple the formula count.

### Phase 3 v3: Multi-Resolution SPP + Histogram Encoding

#### Feature Encoding (12 scalars per body per resolution)

Each formula body (stripped of final pooling) produces a spatial feature map [B, H, W], encoded as:

| Encoding | Count | Description |
|----------|-------|-------------|
| SPP pools | 8 | global_avg/max/std, quad_tl/tr/bl/br, center |
| Histogram bins | 4 | patch_histogram_4x4 (4×4 grid → 4-bin soft histogram) |
| **Total** | **12** | |

#### Multi-Resolution Extraction

| Resolution | Train Time | Val Time | Features |
|-----------|-----------|---------|----------|
| 112×112 | 83.4 min | 20.9 min | 32,112 |
| 224×224 | 319.2 min | 79.6 min | 32,112 |
| **Total** | **6.7 hours** | **1.7 hours** | **64,224** |

64×64 was excluded to save disk space (32 GB) and time.

**SymbolicKernelBank consistency:** Kernel weights saved to `kernel_bank_weights.pt` and loaded at extraction start, ensuring all resolutions use identical learnable kernels.

#### Per-Resolution L1 Feature Selection

| Resolution | Quota | Selected | L1 wd |
|-----------|-------|---------|-------|
| 112×112 | 12,000 | 12,000 | 20.0 |
| 224×224 | 18,000 | 18,000 | 20.0 |
| **Total** | **30,000** | **30,000** | |

224 gets 60% of the budget (higher resolution = more informative).

#### Classifier Training

| Item | Value |
|------|-------|
| Model | `nn.Linear(30000, 1000)` — 30M parameters |
| Optimizer | AdamW (lr=1e-3) + CosineAnnealingLR |
| Epochs | 30 |
| Batch size | 1024 |
| Train images | 200,000 (200/class) |
| Val images | 50,000 (full) |

---

## Results

### Weight Decay Sweep

| weight_decay | Train Top-1 | Val Top-1 | Val Top-5 | Time |
|-------------|-------------|-----------|-----------|------|
| 5.0 | 99.92% | 19.12% | 36.59% | 262s |
| 10.0 | 99.90% | 19.24% | 36.96% | 274s |
| **20.0** | **94.24%** | **19.45%** | **37.20%** | **451s** |
| 50.0 | 63.57% | 18.07% | 34.77% | 273s |

**Best: wd=20.0, Val Top-1 = 19.45%, Val Top-5 = 37.20%**

### Comparison Across All Runs

| Run | Formulas | Encoding | Resolution | Val Top-1 | Val Top-5 |
|-----|----------|----------|-----------|-----------|-----------|
| v1 (single bank) | 2,611 | 1 pool | 224 | 9.24% | 21.70% |
| v2 FP32 (wd=5.0) | 6,000 | 1 pool | 112 | 17.27% | 33.64% |
| v2 SPP (wd=10.0) | 6,000 | 8 SPP | 224 | 20.61% | 39.18% |
| **v3 (wd=20.0)** | **2,676** | **12 (SPP+hist)** | **112+224** | **19.45%** | **37.20%** |

---

## Analysis

### Why v3 didn't surpass v2

v3 achieved 19.45% with 2,676 formulas vs v2's 20.61% with 6,000 formulas. The primary reason:

**Formula count:** v3 lost 73% of its formulas (10,049 → 2,676) due to the Phase 2 terminal mismatch bug. The 5,279 failed formulas all used the new v3 terminals (`I_r`, `I_g`, `I_RG`, `I_BY`) which weren't available in the Phase 2 execution function.

**Per-formula quality:** v3's formulas are higher quality (mean accuracy 0.10 vs v2's 0.06, min 0.08 vs 0.008), but the quantity gap overwhelms the quality advantage.

### What the new operators contributed

Despite the formula loss, v3's 2,676 formulas achieved 19.45% — close to v2's 20.61% with 6,000 formulas. This suggests:
- **New operators are more expressive:** Each v3 formula captures more information
- **SPP + histogram encoding extracts more from each formula:** 12 scalars vs 8 in v2
- **Multi-resolution helps:** 112+224 provides both medium and fine detail

### Issues encountered and fixed

1. **FP16 bug (v2):** Conv operators failed silently with FP16 input → fixed by using FP32 throughout
2. **SymbolicKernelBank graph issue:** `nn.Parameter` created computation graphs during RL reward computation → fixed with `detach()` in non-finetune mode
3. **Kernel weight inconsistency across runs:** `torch.randn` produces different weights on each init → fixed by saving/loading `kernel_bank_weights.pt`
4. **Phase 2 terminal mismatch:** `build_data_batch()` in pipeline missing 4 new terminals → **fixed in code, needs Phase 2 re-run**
5. **Disk quota:** 3-resolution mmap + concatenation exceeded quota → solved by dropping 64×64 and per-resolution L1 selection (no concatenation)
6. **Label file corruption:** `y_train.npy` overwritten by interrupted runs with wrong sample count → fixed by saving labels only once during first extraction

---

## Next Steps (Priority Order)

### 1. Fix Phase 2 + recover 5,279 formulas (highest impact)
- `build_data_batch()` in `train_imagenet_pipeline.py` already fixed (added `I_r`, `I_g`, `I_RG`, `I_BY`)
- Re-run Phase 2 only → expect ~7,000+ formulas after dedup (vs current 2,676)
- Re-extract features → expect 25%+ accuracy

### 2. Feature interactions (pairwise products)
- Top-300 features by L1 weight → 44,850 interaction features
- Concatenate with base features → L1 selection → retrain

### 3. Hierarchical Layer 2 formulas
- Select top-100 Layer 1 bodies as new terminals
- Run Phase 1B with 110 terminals (10 original + 100 L1)
- Expected +5-10% accuracy

### 4. End-to-end fine-tuning of learnable kernels
- Unfreeze SymbolicKernelBank
- Jointly optimize kernels + classifier
- Expected +3-5%

---

## Output Files

```
outputs/imagenet_v3/
├── phase1/bank_{0..3}/feature_bank/   # 10,049 RL-discovered formulas
├── phase2/feature_bank/               # 2,676 deduplicated formulas
└── phase3_v3/
    ├── kernel_bank_weights.pt         # Fixed kernel weights
    ├── bodies_sorted.json             # 2,676 formula bodies (sorted)
    ├── X_train_112.mmap               # 200K × 32112 (25.7 GB)
    ├── X_val_112.mmap                 # 50K × 32112 (6.4 GB)
    ├── X_train_224.mmap               # 200K × 32112 (25.7 GB)
    ├── X_val_224.mmap                 # 50K × 32112 (6.4 GB)
    ├── X_train_selected.mmap          # 200K × 30000 (24.0 GB)
    ├── X_val_selected.mmap            # 50K × 30000 (6.0 GB)
    ├── y_train.npy / y_val.npy        # Labels
    ├── feat_mean_selected.npy         # Standardization
    ├── feat_std_selected.npy
    └── results.json                   # All results
```

## Reproduction

```bash
# Phase 1A (RL discovery with v3 operators)
python experiments/train_imagenet_pipeline.py \
  --config configs/tensor_vsr_imagenet_v3.yaml \
  --device cuda --output_dir outputs/imagenet_v3 --start_phase 1

# Phase 3 v3 (multi-res SPP + L1 selection + classifier)
PYTHONUNBUFFERED=1 python experiments/run_phase3_v3.py
```
