# ImageNet v3 — Clean Restart Plan (From Scratch to 50%+)

> **Current best (v2 reference)**: Val Top-1 = 20.61%, Val Top-5 = 39.18%
> (6,000 formulas × 8 SPP pools, 224×224, wd=10.0, 200K train images)
>
> **This is a CLEAN RESTART.** We are not building on v2 outputs. All formulas, feature matrices, and models will be regenerated from scratch with a significantly improved operator set, grammar, encoding, and pipeline.
>
> **Target**: Val Top-1 ≥ 50%
>
> **Hardware**: Multiple A100-80GB available (rent as needed). Memory/disk is NOT a constraint.
>
> **Core principle**: Structure is symbolic (RL-discovered), parameters can be learned. Every feature must be traceable to an explicit formula. Final classifier is `nn.Linear` (no MLP). FP32 throughout (no FP16).
>
> **Key insight from experiments so far**:
> - FP16 bug fix: 8.29% → 13.89% (+5.6%) — correctness matters most
> - Regularization: 13.89% → 17.27% (+3.4%) — diminishing returns
> - SPP (8 pools): 17.27% → 20.61% (+3.3%) — spatial info matters
> - The bottleneck is NOT regularization or training data — it's **feature expressiveness**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    Input Image                       │
│              I_R, I_G, I_B, I_GRAY, I_H, I_S        │
│              + I_r, I_g, I_RG, I_BY (color ratios)  │
│              at 3 resolutions: 64, 112, 224          │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│              Layer 1: Base Formulas                   │
│   ~5000 RL-discovered formulas using:                │
│   - Fixed operators (edge_x, blur, gabor, etc.)      │
│   - Learnable 3×3 and 5×5 kernels (8+4=12 new)      │
│   Each formula → feature map [H, W]                  │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│              Layer 2: Hierarchical Formulas           │
│   Top-100 Layer 1 feature maps become new terminals  │
│   ~3000 RL-discovered Layer 2 formulas               │
│   Each formula → feature map [H, W]                  │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│              Feature Encoding                        │
│   Each feature map (Layer 1 + Layer 2) encoded by:   │
│   - SPP: 8 fixed pooling ops → 8 scalars            │
│   - Patch Histogram: 4×4 grid → 4-bin hist → 4 vals │
│   = 12 scalars per formula                           │
│                                                      │
│   5000 L1 + 3000 L2 = 8000 formulas × 12 = 96,000  │
│   × 3 resolutions = 288,000 base features           │
│   + top-500 pairwise interactions = ~125,000         │
│   Total: ~400,000 raw features                       │
│   → L1 selection to ~50,000 effective features       │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│              Classification                          │
│   nn.Linear(n_features, 1000)                        │
│   Train on 500K images, wd=10–20                     │
│                                                      │
│   End-to-end fine-tuning phase:                      │
│   Learnable kernels + classifier jointly optimized   │
└─────────────────────────────────────────────────────┘
```

---

## Step 1: Feature Interactions (Degree-2 Polynomial)

**Goal**: +3–5% (→ ~24%)
**Training mode**: Offline (feature matrix already exists)

**What**: Compute pairwise products of top-K important features. This gives the linear classifier access to feature conjunctions ("red edges AND green texture") which it currently cannot express.

**Implementation**:

```python
"""
Modify run_phase3_spp.py to add interaction features after Step 2.
"""

# 1. Load existing model weights to find important features
model = load_trained_model('nn.Linear(48000, 1000), wd=10.0')
weight_importance = model.weight.abs().sum(dim=0)  # [48000]
top_k = 300
top_indices = weight_importance.topk(top_k).indices

# 2. Extract top-K features from existing mmap
X_train_topk = X_train_mmap[:, top_indices.cpu().numpy()]  # [200K, 300]
X_val_topk = X_val_mmap[:, top_indices.cpu().numpy()]

# 3. Compute pairwise products
# 300 choose 2 = 44,850 interaction features
import itertools
pairs = list(itertools.combinations(range(top_k), 2))

X_train_interact = np.empty((n_train, len(pairs)), dtype=np.float32)
X_val_interact = np.empty((n_val, len(pairs)), dtype=np.float32)

for idx, (i, j) in enumerate(pairs):
    X_train_interact[:, idx] = X_train_topk[:, i] * X_train_topk[:, j]
    X_val_interact[:, idx] = X_val_topk[:, i] * X_val_topk[:, j]

# 4. Concatenate: original 48K + interactions 45K = 93K features
X_train_full = np.concatenate([
    np.array(X_train_mmap),
    X_train_interact
], axis=1)

# 5. Retrain with strong regularization
# model = nn.Linear(93000, 1000)
# sweep wd = [10.0, 20.0, 50.0]
```

**Interpretability**: Each interaction feature = "formula_i output × formula_j output". Fully traceable.

---

## Step 2: Multi-Resolution Feature Extraction

**Goal**: +3–5% (→ ~27%)
**Training mode**: Offline (store 3 feature matrices)

**What**: Run same 6,000 formula bodies × 8 SPP pools at 3 resolutions: 64×64, 112×112, 224×224. A 3×3 edge detector "sees" different spatial extents at each resolution.

**Implementation**:

```python
"""
New script: run_phase3_multires.py
"""

resolutions = [64, 112, 224]
formula_bodies = load_formulas('bodies_sorted.json')  # 6000 bodies
spp_pools = 8

for res in resolutions:
    extract_spp_features(
        formulas=formula_bodies,
        train_loader=ImageNetLoader(samples_per_class=200, resolution=res),
        val_loader=ImageNetLoader(split='val', resolution=res),
        output_prefix=f'features_{res}'
    )

# Concatenate all resolutions: 48,000 × 3 = 144,000 features
# Disk: 200K × 144K × 4 bytes = 115 GB (feasible with large disk)
# Train: nn.Linear(144000, 1000), wd=20–50
```

---

## Step 3: New Operators — Distribution-Aware Pooling + Color Terminals

**Goal**: +2–3% (→ ~30%)
**Training mode**: Offline

### 3a. New Pooling Operators

Add to `tensor_operators.py` and register in `TENSOR_OPERATORS` + `ROOT_OPERATORS`:

```python
@staticmethod
def ratio_above_mean(x):
    """Fraction of pixels above the mean: [B,H,W] → [B]
    Measures spatial EXTENT of a feature — unlike avg_pool which measures intensity.
    """
    mean = x.mean(dim=[-2,-1], keepdim=True)
    return (x > mean).float().mean(dim=[-2,-1])

@staticmethod
def percentile_90(x):
    """90th percentile value: [B,H,W] → [B]
    Robust peak measurement (unlike global_max_pool which is noise-sensitive).
    """
    flat = x.flatten(1)
    k = max(1, flat.shape[1] // 10)
    return flat.topk(k, dim=1).values[:, -1]

@staticmethod
def spatial_entropy(x):
    """Entropy of spatial distribution: [B,H,W] → [B]
    High = feature spread evenly. Low = feature concentrated in one spot.
    """
    flat = torch.abs(x).flatten(1) + 1e-8
    p = flat / flat.sum(dim=1, keepdim=True)
    return -(p * torch.log(p)).sum(dim=1)

@staticmethod
def peak_location_y(x):
    """Vertical position of max activation (normalized 0-1): [B,H,W] → [B]"""
    H = x.shape[1]
    col_max = x.max(dim=2).values
    return col_max.argmax(dim=1).float() / H

@staticmethod
def peak_location_x(x):
    """Horizontal position of max activation (normalized 0-1): [B,H,W] → [B]"""
    W = x.shape[2]
    row_max = x.max(dim=1).values
    return row_max.argmax(dim=1).float() / W
```

### 3b. Symbolic Bag-of-Words Pooling (KEY NEW IDEA)

```python
@staticmethod
def patch_histogram_4x4(x):
    """Spatial Bag-of-Words: [B,H,W] → [B, 4]
    
    Divide feature map into 4×4 grid (16 patches).
    Compute mean of each patch.
    Build 4-bin soft histogram of these 16 patch means.
    
    Encodes the DISTRIBUTION of a feature across space.
    "3 out of 16 patches have strong edges" ≠ "average edge is moderate"
    Analogous to Bag-of-Visual-Words (ILSVRC 2010 winner concept).
    """
    B, H, W = x.shape
    x4d = x.unsqueeze(1)
    patch_means = F.adaptive_avg_pool2d(x4d, output_size=4).view(B, 16)

    vmin = patch_means.min(dim=1, keepdim=True).values
    vmax = patch_means.max(dim=1, keepdim=True).values
    normalized = (patch_means - vmin) / (vmax - vmin + 1e-8)

    n_bins = 4
    bins = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        in_bin = torch.sigmoid(20 * (normalized - lo)) * torch.sigmoid(20 * (hi - normalized))
        bins.append(in_bin.sum(dim=1))
    return torch.stack(bins, dim=1)  # [B, 4]
```

Register in `MULTI_DIM_OPERATORS = {'patch_histogram_4x4': 4}`.

### 3c. New Input Terminals (Color Ratios)

Add to `build_data_batch()`:

```python
total = I_R + I_G + I_B + 1e-8
data_batch['I_r'] = I_R / total       # red ratio (illumination invariant)
data_batch['I_g'] = I_G / total       # green ratio
data_batch['I_RG'] = I_R - I_G        # red-green opponent
data_batch['I_BY'] = I_B - (I_R + I_G) / 2  # blue-yellow opponent
```

Total terminals: 6 → 10.

### 3d. New Spatial Operators

```python
@staticmethod
def opening(x):
    """Morphological opening (erode→dilate). Removes small bright spots."""
    x4 = x.unsqueeze(1)
    eroded = -F.max_pool2d(-x4, kernel_size=3, stride=1, padding=1)
    opened = F.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)
    return opened.squeeze(1)

@staticmethod
def closing(x):
    """Morphological closing (dilate→erode). Fills small dark holes."""
    x4 = x.unsqueeze(1)
    dilated = F.max_pool2d(x4, kernel_size=3, stride=1, padding=1)
    closed = -F.max_pool2d(-dilated, kernel_size=3, stride=1, padding=1)
    return closed.squeeze(1)

@staticmethod
def tophat(x):
    """Top-hat: x - opening(x). Extracts small bright details on dark background."""
    return x - TensorOperators.opening(x)

@staticmethod
def high_freq(x):
    """High-frequency residual: x - blur_7x7(x). Fine texture and detail."""
    blurred = F.avg_pool2d(x.unsqueeze(1), kernel_size=7, stride=1, padding=3).squeeze(1)
    return x - blurred

@staticmethod
def low_freq(x):
    """Low-frequency: 15×15 blur. Overall color/brightness gradients."""
    return F.avg_pool2d(x.unsqueeze(1), kernel_size=15, stride=1, padding=7).squeeze(1)
```

After adding all Step 3 operators: **rerun Phase 1** (4 banks, ~30 min total) to discover formulas using new operators.

---

## Step 4: Learnable Convolution Kernels

**Goal**: +5% (→ ~35%)
**Training mode**: Phase A (RL discovery) = offline. Phase B (fine-tuning) = online.

### 4a. SymbolicKernelBank

```python
class SymbolicKernelBank(nn.Module):
    def __init__(self):
        super().__init__()
        # Classic kernels — initialized to known values, fine-tunable
        self.edge_x = nn.Parameter(torch.tensor([[-1.,0,1],[-2,0,2],[-1,0,1]]).view(1,1,3,3))
        self.edge_y = nn.Parameter(torch.tensor([[-1.,-2,-1],[0,0,0],[1,2,1.]]).view(1,1,3,3))
        self.laplacian = nn.Parameter(torch.tensor([[0.,1,0],[1,-4,1],[0,1,0.]]).view(1,1,3,3))
        self.gabor_0 = nn.Parameter(_make_gabor_kernel(0.0))
        self.gabor_45 = nn.Parameter(_make_gabor_kernel(math.pi/4))
        self.gabor_90 = nn.Parameter(_make_gabor_kernel(math.pi/2))
        
        # New learnable kernels — random init, fully data-driven
        self.conv3x3 = nn.ParameterList([nn.Parameter(torch.randn(1,1,3,3)*0.1) for _ in range(8)])
        self.conv5x5 = nn.ParameterList([nn.Parameter(torch.randn(1,1,5,5)*0.1) for _ in range(4)])
    
    def apply_kernel(self, name, x):
        kernel = getattr(self, name, None)
        if name.startswith('conv3x3_'):
            kernel = self.conv3x3[int(name.split('_')[-1])]
        elif name.startswith('conv5x5_'):
            kernel = self.conv5x5[int(name.split('_')[-1])]
        padding = kernel.shape[-1] // 2
        return F.conv2d(x.unsqueeze(1), kernel, padding=padding).squeeze(1)
```

### 4b. Two-Phase Training

```
Phase A: RL discovery (fixed kernels — fast, offline)
  - Freeze learnable kernels at random init values
  - PPO discovers formula structures as usual
  - Output: ~5000 formulas that use both classic and new kernel operators

Phase B: End-to-end fine-tuning (online — kernels change each step)
  - Jointly optimize:
      classifier: lr=1e-3, wd=10.0
      classic kernels: lr=1e-5 (small — stay close to Sobel/Gabor)
      learned kernels: lr=1e-4 (larger — free to learn)
  - 500K train images, 10 epochs
```

### 4c. Post-Training Kernel Analysis (for paper)

```python
# Visualize learned kernels
for i, kernel in enumerate(kernel_bank.conv3x3):
    plt.imshow(kernel.detach().cpu().squeeze(), cmap='RdBu')
    plt.title(f'Learned conv3x3_{i}')
    
# Compare fine-tuned classic kernels with originals
print("edge_x drift:", (kernel_bank.edge_x - original_sobel_x).abs().mean())
```

---

## Step 5: Hierarchical Two-Layer Formulas

**Goal**: +10% (→ ~45%)
**Training mode**: Offline

### 5a. Select Layer 1 Bases (Forward Selection)

```python
# From 6000 formula bodies, greedily select top-100 most complementary
selected = []
for round in range(100):
    best_gain, best_body = 0, None
    candidates = random.sample(remaining, min(500, len(remaining)))
    
    for body in candidates:
        trial_features = current_features + extract_body_features(body)
        trial_acc = quick_eval(trial_features, val_subset)
        if trial_acc - current_acc > best_gain:
            best_gain = trial_acc - current_acc
            best_body = body
    
    if best_gain < 0.001: break
    selected.append(best_body)

# These 100 bodies → Layer 2 terminals: L1_0 ... L1_99
```

### 5b. Run Phase 1 for Layer 2

```python
# Layer 2 terminals = 10 original channels + 100 Layer 1 feature maps = 110
# Layer 2 operators = same as Layer 1
# Shorter formulas: max_depth=5, max_seq_len=12

config_L2 = {
    'terminals': ['I_R',...,'I_BY', 'L1_0',...,'L1_99'],
    'max_depth': 5,
    'max_sequence_length': 12,
    'feature_bank_size': 5000,
    'multi_bank': {'num_banks': 4}
}
```

### 5c. Combine Layer 1 + Layer 2 Features

```python
# Layer 1: 5000 × 12 encodings = 60,000 features
# Layer 2: 3000 × 12 encodings = 36,000 features
# × 3 resolutions = 288,000 features
# + interactions = ~125,000
# → L1 selection to ~50,000 effective features
# Train nn.Linear(50000, 1000)
```

---

## Step 6: End-to-End Fine-Tuning

**Goal**: +5% (→ ~50%)
**Training mode**: Online (required — kernels change each step)

```python
kernel_bank = SymbolicKernelBank().cuda()
classifier = nn.Linear(n_features, 1000).cuda()
classifier.load_state_dict(pretrained_weights)  # warm start

optimizer = AdamW([
    {'params': classifier.parameters(), 'lr': 1e-3, 'weight_decay': 10.0},
    {'params': kernel_bank.classic_params(), 'lr': 1e-5},
    {'params': kernel_bank.learned_params(), 'lr': 1e-4},
])

for epoch in range(10):
    for images, labels in train_loader_500k:
        data_batch = build_data_batch(images.cuda())
        feats = extract_all_features_learnable(formulas, data_batch, kernel_bank)
        loss = criterion(classifier(normalize(feats)), labels.cuda())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## Step 7: Scale Up Training Data

**Goal**: +5–10% (→ ~55–60%)
**Training mode**: Online (full 1.18M dataset)

Same as Step 6 but with `samples_per_class=None` (full dataset), 10 epochs.

---

## Predicted Accuracy Progression (Clean Restart)

| Milestone | What | Estimated Top-1 | Training Mode |
|------|------|----------------|---------------|
| Batch 1 baseline | L1 formulas + new ops + SPP + BoW + multi-res + interactions + 500K train | ~32% | Offline |
| + Batch 2 | Hierarchical L2 formulas added | ~42% | Offline |
| + Batch 3 (fine-tune) | End-to-end learnable kernels | ~48% | Online |
| + Batch 3 (scale up) | Full 1.18M training data | ~53% | Online |

v1/v2 reference: v2 SPP reached 20.61% without new operators, BoW, multi-resolution, interactions, or hierarchical formulas. All of these are now included from the start.

Online training is ONLY needed in Batch 3 (learnable kernel fine-tuning). Batches 1 and 2 use offline feature matrices.

---

## Codebase State

The following already exist from v1/v2 and should be reused/modified:
- `tensor_operators.py` — has basic operators (edge_x, blur, gabor, etc.), needs new operators added
- `train_imagenet_pipeline.py` — has Phase 1 (RL), Phase 3 (extraction + classifier), needs updates for new encoding
- `imagenet_loader.py` — has ImageNet loading with stratified sampling, HSV conversion
- `RPNGrammarMask` — has basic RPN grammar, needs grammar fixes (pooling-at-end, no-repeat rules)
- `large_feature_bank.py` — has competitive evolutionary bank, works as-is
- `ppo_trainer.py` — has PPO with entropy schedule, LR warmup, binary_op_bias, works as-is

What needs to be NEW or significantly changed:
- New operators and terminals in `tensor_operators.py` (Step 3 + Step 4)
- `patch_histogram_4x4` multi-dim pooling (Step 3b)
- `SymbolicKernelBank` module (Step 4)
- Grammar fixes in `RPNGrammarMask` (Step 3.6)
- Multi-resolution feature extraction logic
- Layer 2 formula discovery (Phase 1B)
- Feature interaction computation
- L1 feature selection
- Forward selection for Layer 1 bases

---

## Implementation Order for Claude Code

> **IMPORTANT: This is a clean restart.** We are NOT using any previously generated formulas or feature matrices. All previous v1/v2 outputs can be ignored. Start from scratch with the full set of new operators, terminals, and grammar rules.
>
> The codebase and ImageNet data already exist on the server. The main files to modify are `tensor_operators.py`, `build_data_batch()` in the environment, and `RPNGrammarMask`.

### Batch 1: Foundation — implement everything + run Phase 1A

1. **Add all new components to `tensor_operators.py`:**
   - 5 new spatial operators (opening, closing, tophat, high_freq, low_freq)
   - 5 new distribution-aware pooling operators (ratio_above_mean, percentile_90, spatial_entropy, peak_location_y, peak_location_x) — register in ROOT_OPERATORS
   - `patch_histogram_4x4` — register as multi-dim root operator (output dim=4)
   - `SymbolicKernelBank` with 6 classic (initialized to Sobel/Gabor values) + 8 learnable 3×3 + 4 learnable 5×5 kernels — register conv3x3_0..7 and conv5x5_0..3 as operators
2. **Add 4 color ratio terminals** to `build_data_batch()`: I_r, I_g, I_RG, I_BY (total terminals: 10)
3. **Enforce grammar rules** in `RPNGrammarMask`:
   - Pooling/root operators ONLY allowed as final token when stack_depth=1
   - Ban consecutive identical unary operators
4. **Run smoke test**: load 1000 images, execute a few test formulas with new operators, verify no NaN/Inf, verify grammar rules work
5. **Run Phase 1A** (Layer 1 formula discovery):
   - 4 banks, 112×112, 20K training images (20/class)
   - Learnable kernels FROZEN at random init during RL
   - Early stopping (50 iters without bank growth)
   - Target: ~10,000–15,000 formulas total across 4 banks
6. **Extract Layer 1 features** at 3 resolutions (64, 112, 224):
   - Each formula body × (8 SPP pools + 4 histogram bins) = 12 encodings per resolution
   - Store as float32 mmap. Train: 500K images (500/class). Val: 50K (full).
7. **L1 feature selection** → reduce to ~50K features
8. **Add feature interactions**: top-500 features by L1 weight → pairwise products → L1 selection again
9. **Train classifier** (nn.Linear, sweep wd=[5, 10, 20, 50], 30 epochs)
10. **Report results** — this is our v3 Layer-1-only baseline

### Batch 2: Hierarchical formulas — Layer 2

11. **Forward selection**: pick top-100 Layer 1 bodies (most complementary, not just highest accuracy)
12. **Run Phase 1B** (Layer 2 formula discovery): 110 terminals (10 original + 100 L1 bodies), 4 banks, max_depth=5, max_seq_len=12
13. **Extract Layer 2 features** at 3 resolutions, same encoding (12 per body per resolution)
14. **Combine L1 + L2 features** + interactions, L1 selection → ~50K
15. **Train classifier**, report results

### Batch 3: End-to-end fine-tuning + scale up

16. **End-to-end fine-tune** learnable kernels + classifier (online training, 500K images, 10 epochs). This is the ONLY step that requires online training.
17. **Re-extract features** with updated kernels, retrain classifier offline for clean numbers
18. **Scale to full 1.18M training set** (online training, 10 epochs)
19. **Final report** with per-superclass accuracy, ablation study, kernel visualization

---

## Key Constraints (MUST follow)

1. ** Final classifier = `nn.Linear` only.
2. **Every feature is traceable.** Each scalar = an explicit symbolic formula (or product of two formulas).
3. **FP32 throughout.** No FP16 for operator execution.
4. **Learnable parameters OK** if formula STRUCTURE is symbolic/RL-discovered.
5. **`_safe_binary` clamp range: ±60000** (future-proofing for FP16 compatibility).
6. **No ImageNet mean/std normalization.** Formulas operate on raw [0,1] pixel values.
