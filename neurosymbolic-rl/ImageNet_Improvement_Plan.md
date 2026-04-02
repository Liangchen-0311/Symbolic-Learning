# Symbolic Feature Evolution — ImageNet Scaling Improvement Plan

> **Context**: We have a working symbolic feature discovery framework (PPO + RPN + competitive feature bank + linear classifier) that achieves 76.84% on CIFAR-10 and 55.8% on CIFAR-100. This document details all code-level changes needed to scale the system to ImageNet (224×224, 1000 classes).
>
> **Core idea**: Use reinforcement learning to discover thousands of interpretable symbolic (math) formulas. Each formula extracts a scalar feature from an image. Then linearly combine all features via logistic regression to get classification accuracy.
>
> **Target**: Achieve competitive accuracy on ImageNet using **only** symbolic formulas + logistic regression, no MLP/CNN.
>
> **Hardware**: 1–4× NVIDIA A100 (80 GB), 100+ GB system RAM.
>
> **Last updated**: 2026-03-25

---

## Implementation Status Overview

| Section | Status | Summary |
|---|---|---|
| §1 RL Improvements | ✅ Complete | Loss-based reward, top-5, hierarchical eval, entropy schedule, LR warmup all implemented |
| §2 Tensor Operators | ✅ Complete | All 46 operators implemented (27 non-root + 19 root), vocabulary = 55 tokens |
| §3 Formula Complexity | ✅ Complete | max_depth=7, multi-bank strategy configured |
| §4 Training Pipeline | ✅ Complete | 3-phase pipeline + ImageNet loader done; single A100 80GB sufficient |
| §5 Engineering | ✅ Complete | _safe_binary + checkpointing + mmap feature extraction + PyTorch linear classifier |
| §6 Operator Implementation | ✅ Complete | All 18 new operators implemented, vocabulary = 55 tokens |
| §7 ImageNet-Specific Scaling | ✅ Complete | Memory-mapped extraction, PyTorch Linear classifier, per-superclass reporting, baselines |

---

## 1. Reinforcement Learning Improvements ✅ COMPLETE

### 1.1 Reward Signal Redesign ✅

**Implemented in `tensor_environment_large_bank.py`** (`_compute_loss_based_reward()`):

1. **Loss-based reward** ✅ — `reward = max(0, 1 - loss / log(num_classes))`
2. **Top-k accuracy** ✅ — `reward = 0.6 * normalized_loss + 0.3 * top5_accuracy + 0.1 * top1_accuracy`
3. **Hierarchical evaluation** ✅ — 20 ImageNet superclasses defined in `imagenet_loader.py`, switches to full 1000-class at 50% bank capacity

### 1.2 Formula Evaluation Efficiency ✅

1. **Stratified mini-batch evaluation** ✅ — `eval_batch_size` configurable, two-stage eval (quick + full)
2. **Resolution-adaptive evaluation** ✅ — `eval_resolution_quick=64`, `eval_resolution_full=224`, uses `F.interpolate`
3. **Sub-expression caching** ✅ — `_subexpr_cache` dict with batch-change invalidation

### 1.3 PPO Hyperparameter Adjustments ✅

1. **Entropy coefficient schedule** ✅ — `entropy_coef_start=0.05` → `entropy_coef_end=0.005`, linear decay over 50%
2. **Episodes per iteration** ✅ — `episodes_per_iteration=150` in ImageNet config
3. **Learning rate warm-up** ✅ — `lr_warmup_iterations=100`, linear warmup in `_update_schedule()`

### 1.4 Feature Bank Scaling ✅

1. **Bank capacity** ✅ — `feature_bank_size=12000` per bank (configurable up to 50K)
2. **Adaptive accuracy threshold** ✅ — `adaptive_threshold=true`, raises to `mean_accuracy * 0.8` after warmup
3. **Correlation threshold** ✅ — `correlation_threshold=0.85`, `correlation_threshold_full=0.78`
4. **Multi-bank training** ✅ — 4 banks with different max_depth configs

---

## 2. Tensor Operators Improvements ⚠️ PARTIAL

### 2.1 Operators Added ✅

| Operator | Status | Location |
|---|---|---|
| `blur_7x7` | ✅ Done | `tensor_operators.py` |
| `sigmoid` | ✅ Done | `tensor_operators.py` |
| `global_min_pool` | ✅ Done | `tensor_operators.py` |
| Gabor kernel caching | ✅ Done | `_gabor_cache` class-level dict |

### 2.2 Operators Removed ✅

`sharpen`, `erode`, `rotate_90`, `translate_x`, `translate_y` — all absent from current codebase (either removed or never added). Action space is clean.

### 2.3 Input Terminals ✅

| Terminal | Status |
|---|---|
| `I_R`, `I_G`, `I_B`, `I_GRAY` | ✅ Original |
| `I_H` (Hue) | ✅ Added — full RGB→HSV math in `_prepare_data_batch()` |
| `I_S` (Saturation) | ✅ Added |

### 2.4 Current Operator Inventory (as of now)

**Current vocabulary: 36 tokens** (3 special + 27 operators + 6 terminals)

| Category | Count | Operators |
|---|---|---|
| Channel Arithmetic | 3 | add, subtract, multiply |
| Activations | 3 | relu, abs, sigmoid |
| Spatial / Structural | 7 | blur, blur_7x7, edge_x, edge_y, laplacian, dilate, normalize |
| Texture / Frequency | 4 | gabor_0, gabor_45, gabor_90, local_std_5x5 |
| Global Pooling (root) | 5 | global_avg/max/min/std/l2_pool |
| Spatial Pooling (root) | 6 | pool_top/bottom/left/right_half, pool_center, pool_corners |
| **Total operators** | **28** | |
| Terminals | 6 | I_R, I_G, I_B, I_GRAY, I_H, I_S |
| Special | 3 | START, END, PAD |
| **Grand total** | **36** | |

### 2.5 Operators Still Missing ❌ — NEEDS IMPLEMENTATION

**Target: 55 tokens. Gap: 19 operators to add.**

#### 2.5.1 Arithmetic: `div` (P0)

**Rationale**: Division enables ratio features (e.g., `I_R / I_G` for color ratios). Critical for fine-grained discrimination — many ImageNet classes differ only by color distribution ratios (different bird species, flower types). Cannot be composed from existing operators.

```python
@_safe_binary
def div(x, y):
    """Safe division with epsilon to prevent div-by-zero."""
    return x / (y + 1e-8 * y.sign().clamp(min=1e-8))
```

#### 2.5.2 Pointwise: `negate` (P0)

**Rationale**: Fundamental building block. Enables "absence of feature" detection (e.g., `negate(edge_x)` = smooth regions). Also needed to compose `erode` from `negate → dilate → negate`.

```python
@staticmethod
def negate(x):
    return -x
```

#### 2.5.3 Pointwise: `pow2` (P1)

**Rationale**: Squared values emphasize large activations, suppress small ones. Enables energy features (e.g., `pow2(edge_x) + pow2(edge_y)` = gradient magnitude squared). Important for texture discrimination.

```python
@staticmethod
def pow2(x):
    return x * x
```

#### 2.5.4 Pointwise: `sqrt_abs` (P1)

**Rationale**: Compresses dynamic range — opposite of `pow2`. On ImageNet with 224×224, spatial operators can produce very large values; `sqrt_abs` brings them into a useful range. Also a standard component of Hellinger distance features.

```python
@staticmethod
def sqrt_abs(x):
    return torch.sqrt(torch.abs(x) + 1e-8)
```

#### 2.5.5 Pointwise: `log1p_abs` (P1)

**Rationale**: Logarithmic compression. Useful for features spanning multiple orders of magnitude (common in frequency-domain-like features from Gabor filters). `log(1 + |x|)` is numerically stable.

```python
@staticmethod
def log1p_abs(x):
    return torch.log1p(torch.abs(x))
```

#### 2.5.6 Geometric: `flip_h`, `flip_v` (P1)

**Rationale**: Symmetry detection. `x - flip_h(x)` extracts left-right asymmetry, a strong feature for many ImageNet categories (faces, animals, vehicles). Two operators for horizontal and vertical symmetry.

```python
@staticmethod
def flip_h(x):
    return x.flip(-1)  # flip width dimension

@staticmethod
def flip_v(x):
    return x.flip(-2)  # flip height dimension
```

#### 2.5.7 Multi-scale: `downsample_2x`, `downsample_4x`, `stride_pool_4` (P0)

**Rationale**: **Critical for ImageNet**. On 224×224 images, a 3×3 edge detector only sees 1.3% of the image. Downsampling creates multi-scale representations — `downsample_4x → edge_x` effectively detects edges at 4× the spatial scale. `stride_pool_4` provides a different downsampling strategy (max-pooling preserves strong features).

```python
@staticmethod
def downsample_2x(x):
    """2x spatial downsampling via average pooling: [B,H,W] → [B,H/2,W/2]"""
    return F.avg_pool2d(x.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)

@staticmethod
def downsample_4x(x):
    """4x spatial downsampling: [B,H,W] → [B,H/4,W/4]"""
    return F.avg_pool2d(x.unsqueeze(1), kernel_size=4, stride=4).squeeze(1)

@staticmethod
def stride_pool_4(x):
    """4x max pooling (preserves strong activations): [B,H,W] → [B,H/4,W/4]"""
    return F.max_pool2d(x.unsqueeze(1), kernel_size=4, stride=4).squeeze(1)
```

#### 2.5.8 Spatial Pooling: `pool_thirds_*` (3), `pool_quadrants_*` (4), `pool_surround`, `spp_pool` (P2)

**Rationale**: Finer-grained spatial features. ImageNet has strong spatial priors (sky at top, ground at bottom, object centered). Current halves + center + corners cover coarse regions, but thirds and quadrants provide intermediate granularity.

```python
# Thirds: divide image into 3 horizontal strips
@staticmethod
def pool_thirds_top(x):     # top 1/3
    H = x.shape[-2]; return x[..., :H//3, :].mean(dim=[-2,-1])

@staticmethod
def pool_thirds_mid(x):     # middle 1/3
    H = x.shape[-2]; return x[..., H//3:2*H//3, :].mean(dim=[-2,-1])

@staticmethod
def pool_thirds_bot(x):     # bottom 1/3
    H = x.shape[-2]; return x[..., 2*H//3:, :].mean(dim=[-2,-1])

# Quadrants: 4 quadrants of the image
@staticmethod
def pool_quad_tl(x):        # top-left
    H,W = x.shape[-2], x.shape[-1]; return x[..., :H//2, :W//2].mean(dim=[-2,-1])

@staticmethod
def pool_quad_tr(x):        # top-right
    H,W = x.shape[-2], x.shape[-1]; return x[..., :H//2, W//2:].mean(dim=[-2,-1])

@staticmethod
def pool_quad_bl(x):        # bottom-left
    H,W = x.shape[-2], x.shape[-1]; return x[..., H//2:, :W//2].mean(dim=[-2,-1])

@staticmethod
def pool_quad_br(x):        # bottom-right
    H,W = x.shape[-2], x.shape[-1]; return x[..., H//2:, W//2:].mean(dim=[-2,-1])

# Surround: mean of image border (8px ring)
@staticmethod
def pool_surround(x):
    B = 8  # border width
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[..., B:-B, B:-B] = False
    return (x * mask).sum(dim=[-2,-1]) / mask.sum(dim=[-2,-1])

# SPP: spatial pyramid pooling (concatenates 1×1, 2×2, 4×4 grid averages → 21-dim)
@staticmethod
def spp_pool(x):
    """Simplified SPP → scalar: weighted average of multi-scale grids."""
    g1 = x.mean(dim=[-2,-1])                         # 1×1 global
    g2 = F.adaptive_avg_pool2d(x.unsqueeze(1), 2).squeeze(1).mean(dim=[-2,-1])  # 2×2
    g4 = F.adaptive_avg_pool2d(x.unsqueeze(1), 4).squeeze(1).mean(dim=[-2,-1])  # 4×4
    return (g1 + g2 + g4) / 3.0
```

### 2.6 Target Operator Count (after full implementation)

| Category | Count | Operators |
|---|---|---|
| Channel Arithmetic | **4** | add, subtract, multiply, **div** |
| Pointwise Primitives | **7** | relu, abs, sigmoid, **negate**, **pow2**, **sqrt_abs**, **log1p_abs** |
| Spatial / Structural | 7 | blur, blur_7x7, edge_x, edge_y, laplacian, dilate, normalize |
| Texture / Frequency | 4 | gabor_0, gabor_45, gabor_90, local_std_5x5 |
| Geometric | **2** | **flip_h**, **flip_v** |
| Multi-scale | **3** | **downsample_2x**, **downsample_4x**, **stride_pool_4** |
| **Non-root total** | **27** | |
| Global Pooling | 5 | global_avg/max/min/std/l2_pool |
| Spatial Pooling (existing) | 6 | halves(4), center, corners |
| Spatial Pooling (new) | **8** | **thirds(3)**, **quadrants(4)**, **pool_surround** |
| Other Pooling | **1** | **spp_pool** |
| **Root total** | **20** | |
| Terminals | 6 | I_R, I_G, I_B, I_GRAY, I_H, I_S |
| Special | 3 | START, END, PAD |
| **Grand total** | **56** | |

This is within the ~55–60 token budget. **Do not exceed 60 tokens** — beyond that, PPO exploration efficiency degrades significantly.

---

## 3. Formula Complexity Adjustments ✅ COMPLETE

### 3.1 Max Depth ✅
- ImageNet config: `max_depth=7`, `max_sequence_length=18`

### 3.2 Complexity Strategy ✅
- Width over depth approach — more operator types rather than deeper formulas
- Multi-scale shortcuts: `downsample_4x → edge_x` captures large-scale structure at depth 3
- Target formula length: 7–10 tokens; max: 18–20

### 3.3 Multi-Bank Strategy ✅

| Bank | max_depth | max_sequence_length | Focus |
|---|---|---|---|
| Bank A | 6 | 15 | Simple + spatial |
| Bank B | 8 | 20 | Complex multi-scale |
| Bank C | 6 | 15 | Texture-focused |
| Bank D | 7 | 18 | Cross-channel |

---

## 4. Training Pipeline Design ⚠️ PARTIAL

### 4.1 Three-Phase Training Pipeline ✅

Implemented in `experiments/train_imagenet_pipeline.py`:
- **Phase 1**: 64×64 fast discovery → 30K–40K candidate formulas
- **Phase 2**: 224×224 validation + deduplication → 15K–25K validated formulas
- **Phase 3**: Full feature extraction + LogisticRegression (L-BFGS, C-value sweep)

### 4.2 Data Loading ✅

Implemented in `src/data/imagenet_loader.py`:
- Resolution-adaptive loading (64 / 224)
- Stratified sampling, no mean/std normalization
- 20 WordNet superclasses for hierarchical eval

Config `data_dir` already updated to `/workspace/neurosymbolic-rl/data/imagenet`.

### 4.3 Multi-GPU Parallelization ❌ NOT IMPLEMENTED

**Current state**: Single-GPU only. No `torch.distributed` setup.

**Plan** (for 4× A100):
1. **Bank-parallel training**: 1 bank per GPU, fully independent
2. **Feature extraction parallelization**: distribute 1.28M images across GPUs in Phase 3
3. Use `torch.distributed.launch` or `torchrun`
4. Each GPU: own PPO policy + environment + bank
5. Synchronize only at merge/deduplication stage

**Implementation priority**: P2 — can proceed with single-GPU for initial experiments; multi-GPU needed for full-scale runs.

---

## 5. Engineering / Code Quality ⚠️ PARTIAL

### 5.1 Numerical Stability ✅
- `_safe_binary` decorator: clamps to `[-1e6, 1e6]`, `nan_to_num`
- Applied to all 4 binary operators: `add`, `subtract`, `multiply`, `div`
- `_safe_binary` also contains `_match_size` logic for cross-scale binary ops (e.g., when `downsample_2x` output is combined with full-resolution tensor)

### 5.2 Feature Extraction Speed ✅

Implemented in `train_imagenet_pipeline.py`:
1. **Memory-mapped output**: `extract_features_mmap()` writes feature matrix directly to `.mmap` file — supports 128 GB+ feature matrices without RAM pressure
2. **Batched formula execution**: 512-image batches processed through all formulas
3. **Sub-expression caching**: `_execute_with_cache()` caches shared prefixes within each batch
4. **Online StandardScaler**: `_online_mean_std()` computes mean/std in chunks from mmap without loading full matrix

### 5.3 Checkpointing and Resumability ✅
- `--resume_from` flag in `train_tensor_vsr_large_bank.py`
- Saves bank + PPO state every 500 iterations

### 5.4 Config ✅
- `configs/tensor_vsr_imagenet.yaml` — complete template
- `data_dir` updated to `/workspace/neurosymbolic-rl/data/imagenet`

---

## 6. Operator Implementation ✅ COMPLETE

All planned operators are now implemented and registered. The vocabulary grew from 36 → **55 tokens**.

### 6.1 Operators Added (18 new)

| Category | Operators | Notes |
|---|---|---|
| Arithmetic | `div` | Wrapped with `_safe_binary` (includes `_match_size` for cross-scale ops) |
| Pointwise | `negate`, `pow2`, `sqrt_abs`, `log1p_abs` | All numerically stable |
| Geometric | `flip_h`, `flip_v` | Symmetry detection |
| Multi-scale | `downsample_2x`, `downsample_4x`, `stride_pool_4` | Critical for 224×224 |
| Spatial pooling | `pool_thirds_top/mid/bot`, `pool_quad_tl/tr/bl/br`, `pool_surround` | 8 new root operators |

### 6.2 Final Vocabulary (55 tokens)

| Category | Count | Operators |
|---|---|---|
| Channel Arithmetic | 4 | add, subtract, multiply, div |
| Pointwise Primitives | 7 | relu, abs, sigmoid, negate, pow2, sqrt_abs, log1p_abs |
| Spatial / Structural | 7 | blur, blur_7x7, edge_x, edge_y, laplacian, dilate, normalize |
| Geometric | 2 | flip_h, flip_v |
| Multi-scale | 3 | downsample_2x, downsample_4x, stride_pool_4 |
| Texture / Frequency | 4 | gabor_0, gabor_45, gabor_90, local_std_5x5 |
| **Non-root total** | **27** | |
| Global Pooling | 5 | global_avg/max/min/std/l2_pool |
| Spatial Pooling | 14 | halves(4), center, corners, thirds(3), quadrants(4), surround |
| **Root total** | **19** | |
| Terminals | 6 | I_R, I_G, I_B, I_GRAY, I_H, I_S |
| Special | 3 | START, END, PAD |
| **Grand total** | **55** | Within the ≤60 budget |

### 6.3 Key Design Decisions

- **`spp_pool` removed**: outputs multi-dimensional (21-dim) which breaks the scalar-per-formula architecture. The quadrant + thirds pooling operators provide equivalent spatial coverage with scalar outputs.
- **`_safe_binary` includes `_match_size`**: all 4 binary operators (add/subtract/multiply/div) automatically handle cross-scale operands (e.g., 64×64 op 32×32 after downsample_2x). This is critical now that `downsample_2x/4x` are in the vocabulary.
- **Vocabulary auto-generates from `TENSOR_OPERATORS`**: `TensorTokenVocabulary` iterates the registry, so no manual vocabulary updates needed when operators are added/removed.

### 6.4 Verification

All 46 operators pass:
- [x] Correct input/output shapes (unary `[B,H,W]→[B,H,W]`, root `[B,H,W]→[B]`)
- [x] Numerically stable on random, zero, and large inputs
- [x] Works on both 64×64 and 224×224
- [x] Cross-scale binary ops (e.g., `add(64×64, 32×32)`) handled correctly
- [x] Vocabulary size = 55 ≤ 60

---

## 7. ImageNet-Specific Scaling Challenges (NEW) ❌

### 7.1 Memory Management for Feature Extraction

**Problem**: The feature matrix for Phase 3 is enormous.
- 25,000 formulas × 1,281,167 images × 4 bytes (float32) = **128 GB**
- Cannot fit in RAM all at once

**Solution**: Chunked feature extraction
```python
# Pseudo-code for Phase 3 feature extraction
chunk_size = 5000  # images per chunk
n_chunks = ceil(1_281_167 / chunk_size)  # ~257 chunks

# Memory-mapped output file
features_mmap = np.memmap('features_train.mmap', dtype='float32',
                          mode='w+', shape=(1_281_167, n_formulas))

for i, chunk in enumerate(data_loader(chunk_size)):
    chunk_features = evaluate_all_formulas(chunk)  # [chunk_size, n_formulas]
    features_mmap[i*chunk_size : (i+1)*chunk_size] = chunk_features
    features_mmap.flush()
```

### 7.2 Linear Classifier at ImageNet Scale

**Problem**: sklearn's L-BFGS and SAGA solvers will almost certainly OOM or be extremely slow at 1.28M samples × 25K features × 1000 classes (the weight matrix alone is 25K × 1000 × 4 bytes = 100 MB, but the optimizer states and batch gradients push well beyond GPU/RAM limits for sklearn).

**Recommended approach** (PyTorch Linear first):
1. **Primary**: PyTorch `nn.Linear(n_features, 1000)` + CrossEntropyLoss + AdamW. Trains on GPU with mini-batches, handles arbitrary scale. Still a purely linear model — fully interpretable. Use L2 weight decay for regularization, sweep `weight_decay ∈ {1e-4, 1e-3, 1e-2, 1e-1}`.
2. **If GPU unavailable**: `sklearn.linear_model.SGDClassifier` with `loss='log_loss'` — processes mini-batches on CPU, can handle large scale with `partial_fit()`.
3. **Feature selection** (optional): If 25K features is too many for the linear layer to generalize, use L1 regularization (sparse linear) or mutual information to select top 5K–10K features before final training.
4. **Avoid**: sklearn L-BFGS / SAGA at this scale — they attempt to materialize the full gradient matrix in memory.

### 7.3 Evaluation Metrics

For ImageNet, report:
- **Top-1 accuracy** (primary metric)
- **Top-5 accuracy** (standard ImageNet metric)
- **Per-superclass accuracy** (diagnostic — which categories are weak?)
- **Number of formulas used** (interpretability metric)
- **Comparison**: random feature baseline, PCA baseline, linear probe on pretrained features

### 7.4 Data Path Configuration

The ImageNet data is being downloaded to: `/workspace/neurosymbolic-rl/data/imagenet/`

After download completes, the directory structure will be:
```
/workspace/neurosymbolic-rl/data/imagenet/
├── train/          # ~1.28M images in 1000 class folders
│   ├── class_0000/
│   ├── class_0001/
│   └── ...
├── val/            # 50K images in 1000 class folders
│   ├── class_0000/
│   └── ...
└── cache/          # HuggingFace download cache (can be deleted after extraction)
```

Config already updated: `data_dir: /workspace/neurosymbolic-rl/data/imagenet`

---

## 8. Revised Priority-Ordered Task List

### ✅ Completed Tasks

| Task | Files | Status |
|---|---|---|
| Reward signal: loss-based + top-5 + hierarchical eval | `tensor_environment_large_bank.py` | ✅ Done |
| Add `blur_7x7`, `sigmoid`, `global_min_pool` | `tensor_operators.py` | ✅ Done |
| Add HSV terminals (`I_H`, `I_S`) | `tensor_environment_large_bank.py` | ✅ Done |
| Resolution-adaptive evaluation (64→224) | `tensor_environment_large_bank.py` | ✅ Done |
| Gabor kernel caching | `tensor_operators.py` | ✅ Done |
| Entropy coefficient schedule + LR warmup | `ppo_trainer.py` | ✅ Done |
| Increase bank capacity (12K configurable) | `large_feature_bank.py`, config | ✅ Done |
| Adaptive accuracy threshold | `large_feature_bank.py` | ✅ Done |
| Numerical stability `_safe_binary` + `_match_size` | `tensor_operators.py` | ✅ Done (all 4 binary ops) |
| Sub-expression caching | `tensor_environment_large_bank.py` | ✅ Done |
| Checkpointing + resume support | `train_tensor_vsr_large_bank.py` | ✅ Done |
| ImageNet data loader + superclasses | `imagenet_loader.py` | ✅ Done |
| Three-phase training script | `train_imagenet_pipeline.py` | ✅ Done |
| ImageNet config YAML + data_dir fix | `tensor_vsr_imagenet.yaml` | ✅ Done |
| Remove sharpen/erode/rotate_90/translate | `tensor_operators.py` | ✅ Done (absent) |
| Add `div`, `negate` operators | `tensor_operators.py` | ✅ Done |
| Add `pow2`, `sqrt_abs`, `log1p_abs` | `tensor_operators.py` | ✅ Done |
| Add `flip_h`, `flip_v` | `tensor_operators.py` | ✅ Done |
| Add `downsample_2x`, `downsample_4x`, `stride_pool_4` | `tensor_operators.py` | ✅ Done |
| Add spatial pooling: thirds(3), quadrants(4), surround | `tensor_operators.py` | ✅ Done |
| Vocabulary auto-updates from TENSOR_OPERATORS | `tensor_environment_large_bank.py` | ✅ Verified (55 tokens) |

| Memory-mapped feature extraction (`extract_features_mmap`) | `train_imagenet_pipeline.py` | ✅ Done |
| Online StandardScaler (`_online_mean_std`) | `train_imagenet_pipeline.py` | ✅ Done |
| PyTorch Linear classifier (AdamW + cosine LR + weight_decay sweep) | `train_imagenet_pipeline.py` | ✅ Done |
| Per-superclass accuracy reporting | `train_imagenet_pipeline.py` | ✅ Done |
| Baseline comparisons (random, PCA, pixel-mean) | `train_imagenet_pipeline.py` | ✅ Done |

### Deferred Tasks

| Task | Reason |
|---|---|
| Multi-GPU parallelization | Single A100 80GB is sufficient for all phases. Multi-GPU only needed if training time becomes a bottleneck (>1 week for Phase 1). |

### Hardware Assessment

Single NVIDIA A100-SXM4-80GB is sufficient for all phases:
- **Phase 1** (RL): 10K images × 64×64 cached on GPU = ~470 MB. Peak < 2 GB.
- **Phase 2** (validation): 10K images × 224×224 = ~14 GB. Fits in 80 GB.
- **Phase 3** (feature extraction): 512-image batches × formula execution = ~1 GB peak GPU.
- **Phase 3** (linear classifier): nn.Linear(25K, 1000) = ~100 MB. Mini-batch training = ~50 MB.
- **Feature matrix**: 1.28M × 25K × 4 bytes = 128 GB → stored on disk as memory-mapped file, never loaded fully into RAM or GPU.
