 # v3.2 Complete Restart Plan

> **Current best**: v2 SPP = 20.61% Top-1, v3 = 19.45% (lost 73% formulas to Phase 2 bug).
> **Target**: 50%+ Top-1 on ImageNet-1K.
> **Reference**: SIFT + Fisher Vector + linear SVM = 54.3% Top-1 (ILSVRC 2010 winner).
> **Core principle**: Every operation is deterministic math. No hidden layers, no learned nonlinearities. Formula structure is symbolic (RL-discovered). Conv kernel parameters can be learned. Final classifier = nn.Linear. FP32 throughout.
>
> **Key insight from research**: The gap between our 20% and the classical 54% is NOT feature quality — it's feature ENCODING. Our formulas produce spatial response maps comparable to Gabor/Sobel filter banks, but we collapse them to single scalars via pooling. The 2010 ILSVRC winner encoded local feature DISTRIBUTIONS via Fisher Vectors. Power normalization alone (signed square root) historically contributed +8–12% absolute improvement. We must adopt these encoding techniques.

---

## Part 1: Lessons from v1/v2/v3

### What worked
| Discovery | Evidence |
|---|---|
| Multi-bank RL (4 banks, different configs) | 10,039 diverse formulas vs single-bank 3,614 |
| SPP (8 pooling ops per body) | +3.3% (17.27% → 20.61%) |
| Grammar fixes (pooling at end only, no consecutive repeat unary) | Eliminated broken/redundant formulas |
| Strong weight decay (wd=5–10) | +3.4% (13.89% → 17.27%) |
| 224×224 for feature extraction | Better than 112×112 for spatial operators |
| Loss-based reward + hierarchical eval | RL converges faster on 1000 classes |
| LRU sub-expression cache | Prevents OOM while sharing computation |

### What went wrong and fixes
| Problem | Root Cause | Fix in v3.2 |
|---|---|---|
| FP16 killed 99.5% formulas | Conv kernels FP32 vs FP16 input | **FP32 everywhere, no exceptions** |
| 96% formulas hit max_length with junk | RL pads `relu relu sigmoid sigmoid` | **Ban consecutive identical unary ops in grammar** |
| Phase 2 killed 73% formulas | `build_data_batch()` missing 4 new terminals | **Skip Phase 2 entirely** — use Phase 1 correlation gate + Phase 3 L1 selection |
| min_accuracy=0.08 too strict | Weak-but-unique formulas discarded | **min_accuracy=0.002** |
| correlation=0.85 too strict | Bank saturated at ~2,500/bank | **correlation=0.92** — target 4,000–5,000/bank |
| Learnable kernels frozen at random values | Wasted RL action space | **Pretrain kernels first** |
| Scalar pooling loses distribution info | Only captures mean/max/std | **Fisher Vector + distribution statistics + power normalization** |
| No rotation-invariant operators | RL must discover `edge_x² + edge_y²` itself | **Add edge_mag, gabor_mag as primitives** |
| Missing classic CV operators | No DoG, Harris, LBP, local contrast | **Add all** |

---

## Part 2: New Operators to Add

### 2.1 Rotation-Invariant Operators (HIGHEST PRIORITY)

These correspond to the local geometric operator Mλ(f) = |νλ[f](c)| from the Geometric MetaFormer paper — taking the modulus across orientations to achieve rotation invariance.

```python
@staticmethod
def edge_mag(x):
    """Gradient magnitude (rotation invariant): sqrt(edge_x² + edge_y²).
    Corresponds to |∇f| — the fundamental rotation-invariant edge measure.
    This is what RL currently has to discover as 'edge_x pow2 edge_y pow2 add sqrt_abs'."""
    ex = TensorOperators.edge_x(x)
    ey = TensorOperators.edge_y(x)
    return torch.sqrt(ex * ex + ey * ey + 1e-8)

@staticmethod
def edge_orient(x):
    """Gradient orientation (0 to π): atan2(edge_y, edge_x).
    Combined with edge_mag, this is the basis of HOG descriptors."""
    ex = TensorOperators.edge_x(x)
    ey = TensorOperators.edge_y(x)
    return torch.atan2(ey, ex + 1e-8)

@staticmethod
def gabor_mag(x):
    """Gabor energy (rotation invariant): sqrt(gabor_0² + gabor_45² + gabor_90²).
    Total texture energy regardless of orientation."""
    g0 = TensorOperators.gabor_0(x)
    g45 = TensorOperators.gabor_45(x)
    g90 = TensorOperators.gabor_90(x)
    return torch.sqrt(g0*g0 + g45*g45 + g90*g90 + 1e-8)
```

### 2.2 Local Structure Operators (HIGH PRIORITY)

These are the building blocks of SIFT and HOG — the features that powered the 54% classical result.

```python
@staticmethod
def local_contrast(x):
    """Local contrast normalization: (x - local_mean) / local_std.
    SIFT's first step — removes illumination variation, preserves structure.
    An image in shadow and in sunlight have the same local_contrast output."""
    x4d = x.unsqueeze(1)
    local_mean = F.avg_pool2d(x4d, kernel_size=7, stride=1, padding=3)
    local_sq_mean = F.avg_pool2d(x4d ** 2, kernel_size=7, stride=1, padding=3)
    local_var = (local_sq_mean - local_mean ** 2).clamp(min=0)
    local_std = torch.sqrt(local_var + 1e-8)
    return ((x4d - local_mean) / local_std).squeeze(1)

@staticmethod
def dog(x):
    """Difference of Gaussians: blur_3x3(x) - blur_7x7(x).
    SIFT's core — detects blobs and keypoints at a specific scale.
    Approximates the Laplacian of Gaussian but is more numerically stable."""
    x4d = x.unsqueeze(1)
    fine = F.avg_pool2d(x4d, kernel_size=3, stride=1, padding=1)
    coarse = F.avg_pool2d(x4d, kernel_size=7, stride=1, padding=3)
    return (fine - coarse).squeeze(1)

@staticmethod
def corner_harris(x):
    """Harris corner response: det(M) - 0.04 * trace(M)².
    M = structure tensor (smoothed outer product of gradients).
    High response = corner/junction, low = flat or edge.
    Key for detecting object shape features."""
    ix = TensorOperators.edge_x(x)
    iy = TensorOperators.edge_y(x)
    ix2 = F.avg_pool2d((ix*ix).unsqueeze(1), 5, 1, 2).squeeze(1)
    iy2 = F.avg_pool2d((iy*iy).unsqueeze(1), 5, 1, 2).squeeze(1)
    ixiy = F.avg_pool2d((ix*iy).unsqueeze(1), 5, 1, 2).squeeze(1)
    det = ix2 * iy2 - ixiy * ixiy
    trace = ix2 + iy2
    return det - 0.04 * trace * trace

@staticmethod
def lbp_like(x):
    """LBP approximation: sigmoid(center - local_mean).
    Output ∈ [0,1]. 1 = center brighter than neighbors, 0 = darker.
    Classic texture descriptor, complementary to local_std."""
    x4d = x.unsqueeze(1)
    center = x4d
    neighbors = F.avg_pool2d(x4d, kernel_size=3, stride=1, padding=1)
    return torch.sigmoid(10 * (center - neighbors)).squeeze(1)
```

### 2.3 Second-Order Operators (MEDIUM PRIORITY)

```python
@staticmethod
def edge_xx(x):
    """Second horizontal derivative — detects vertical ridges and valleys."""
    kernel = torch.tensor([[[[1., -2., 1.]]]], device=x.device)
    return F.conv2d(x.unsqueeze(1), kernel, padding=(0, 1)).squeeze(1)

@staticmethod
def edge_yy(x):
    """Second vertical derivative — detects horizontal ridges and valleys."""
    kernel = torch.tensor([[[[1.], [-2.], [1.]]]], device=x.device)
    return F.conv2d(x.unsqueeze(1), kernel, padding=(1, 0)).squeeze(1)
```

### 2.4 Variance-Aware Spatial Pooling (Register in ROOT_OPERATORS)

```python
@staticmethod
def std_center(x):
    """Center region standard deviation: [B,H,W] → [B].
    Captures texture variation in the main object region."""
    h, w = x.shape[1], x.shape[2]
    center = x[:, h//4:3*h//4, w//4:3*w//4]
    return torch.std(center, dim=[-2, -1])

@staticmethod
def std_top_half(x):
    """Top half standard deviation: [B,H,W] → [B]."""
    return torch.std(x[:, :x.shape[1]//2, :], dim=[-2, -1])

@staticmethod
def std_bottom_half(x):
    """Bottom half standard deviation: [B,H,W] → [B]."""
    return torch.std(x[:, x.shape[1]//2:, :], dim=[-2, -1])
```

### 2.5 Carry forward from v3 (already implemented, keep these)

- 5 spatial ops: opening, closing, tophat, high_freq, low_freq
- 5 distribution pooling: ratio_above_mean, percentile_90, spatial_entropy, peak_location_y, peak_location_x
- patch_histogram_4x4 (multi-dim, output=4)
- SymbolicKernelBank (12 learnable kernels)
- 4 color terminals: I_r, I_g, I_RG, I_BY

### 2.6 Updated Operator Registry Summary

After all additions, the full operator set:

**Terminals (10):** I_R, I_G, I_B, I_GRAY, I_H, I_S, I_r, I_g, I_RG, I_BY

**Unary tensor operators (~40):**
- Existing: blur, blur_7x7, edge_x, edge_y, laplacian, dilate, normalize, flip_h, flip_v, downsample_2x, downsample_4x, stride_pool_4, gabor_0, gabor_45, gabor_90, local_std_5x5, relu, abs, sigmoid, negate, pow2, sqrt_abs, log1p_abs, opening, closing, tophat, high_freq, low_freq
- New v3.2: edge_mag, edge_orient, gabor_mag, local_contrast, dog, corner_harris, lbp_like, edge_xx, edge_yy
- Learnable: conv3x3_0..7, conv5x5_0..3

**Binary tensor operators (4):** add, subtract, multiply, div

**Scalar pooling / root operators (~25):**
- Existing: global_avg/max/min/std/l2_pool, pool_top/bottom/left/right_half, pool_center, pool_corners, pool_thirds_top/mid/bot, pool_quad_tl/tr/bl/br, pool_surround, ratio_above_mean, percentile_90, spatial_entropy, peak_location_y/x
- New v3.2: std_center, std_top_half, std_bottom_half

**Multi-dim root (1):** patch_histogram_4x4 (output=4)

---

## Part 3: Feature Encoding (THE CORE CHANGE)

This is where the biggest accuracy gains come from. Research shows the gap between 20% and 54% is primarily an encoding problem, not a feature quality problem.

### 3.1 Power Normalization + L2 Normalization

Apply to ALL feature vectors before classification. Historically worth +8–12% absolute.

```python
# After standardization, before classifier:
feats = (feats - mean) / std                           # standardize
feats = torch.sign(feats) * torch.sqrt(torch.abs(feats) + 1e-8)  # power norm (α=0.5)
feats = F.normalize(feats, p=2, dim=1)                 # L2 norm
```

Every step is deterministic math. Power norm suppresses "bursty" features that dominate the linear classifier.

### 3.2 Distribution Statistics Encoding (replaces simple SPP)

Instead of collapsing each formula's spatial response map to 1 scalar via pooling, compute rich distribution statistics.

For each formula body, in each of 5 spatial regions (1×1 global + 2×2 quadrants):

| Statistic | Count | What it captures |
|---|---|---|
| mean | 1 | Average intensity (= current global_avg_pool) |
| std | 1 | Variation (= current global_std_pool) |
| max | 1 | Peak strength (= current global_max_pool) |
| skewness | 1 | Distribution asymmetry — positive = few strong activations |
| kurtosis | 1 | Peakedness — high = concentrated response |
| quantiles (10,25,50,75,90%) | 5 | Non-parametric distribution shape |
| ratio_above_mean | 1 | Feature spatial coverage/extent |
| **Total per region** | **12** | |

12 stats × 5 regions = **60 scalars per formula body per resolution**.

```python
def encode_body_distribution(feature_map):
    """Encode a formula body's spatial response as distribution statistics.
    Input: [B, H, W] feature map. Output: [B, 60] statistics.
    All operations are deterministic math — no learnable parameters."""
    
    regions = {
        'global': feature_map,
        'quad_tl': feature_map[:, :H//2, :W//2],
        'quad_tr': feature_map[:, :H//2, W//2:],
        'quad_bl': feature_map[:, H//2:, :W//2],
        'quad_br': feature_map[:, H//2:, W//2:],
    }
    
    all_stats = []
    for region_pixels in regions.values():
        flat = region_pixels.flatten(1)  # [B, N]
        mean = flat.mean(dim=1)
        std = flat.std(dim=1)
        maximum = flat.max(dim=1).values
        skewness = ((flat - mean.unsqueeze(1))**3).mean(dim=1) / (std**3 + 1e-8)
        kurtosis = ((flat - mean.unsqueeze(1))**4).mean(dim=1) / (std**4 + 1e-8) - 3
        quantiles = torch.quantile(flat, torch.tensor([0.1,0.25,0.5,0.75,0.9]), dim=1)
        ratio = (flat > mean.unsqueeze(1)).float().mean(dim=1)
        
        all_stats.append(torch.stack([mean, std, maximum, skewness, kurtosis,
                                       *quantiles, ratio], dim=1))  # [B, 12]
    
    return torch.cat(all_stats, dim=1)  # [B, 60]
```

### 3.3 Symbolic Fisher Vector Encoding (BIGGEST POTENTIAL GAIN)

This is what powered the 54% classical result. Fully deterministic, no hidden state.

**Concept**: Instead of asking "how strong is this feature on average?", ask "how does the distribution of this feature across the image differ from the typical distribution?"

**Step A: Construct local descriptors.**
- Divide image into 8×8 = 64 overlapping patches
- For each patch, evaluate top-100 formula bodies (selected by forward selection) → average response per patch
- Each patch → a 100-dimensional local descriptor vector
- PCA reduce to 32 dimensions (deterministic linear projection, fitted once on training set)

**Step B: Fit visual vocabulary (GMM, one-time offline).**
- Collect ~1M local descriptors from 15K training images
- Fit GMM with K=64 components on PCA-reduced descriptors
- Save GMM parameters: 64 means (μ_k), 64 variances (σ²_k), 64 weights (π_k)
- This is a fixed statistical model, not a neural network

**Step C: Compute Fisher Vector per image.**
```python
def symbolic_fisher_vector(patches, gmm):
    """
    patches: [N_patches, D] — local descriptors (N=64, D=32)
    gmm: fitted GMM with K=64 components
    Returns: [2 × D × K] = [4096] dimensional vector
    
    Every operation is deterministic math. The GMM is a fixed statistical
    reference, not a learnable model — same role as correlation_threshold
    in the current pipeline.
    """
    K, D = gmm.K, gmm.D  # 64, 32
    
    # Posterior: how much does each patch belong to each Gaussian?
    # gamma[t,k] = π_k × N(patch_t | μ_k, σ²_k) / Σ_j π_j × N(patch_t | μ_j, σ²_j)
    gamma = compute_posterior(patches, gmm)  # [N_patches, K]
    
    # First-order: how do patches assigned to component k deviate from μ_k?
    G_mu = zeros(K, D)
    for k in range(K):
        residuals = (patches - gmm.mu[k]) / gmm.sigma[k]
        G_mu[k] = (gamma[:, k:k+1] * residuals).sum(0) / sqrt(gmm.pi[k])
    
    # Second-order: how does variance within component k deviate from σ²_k?
    G_sigma = zeros(K, D)
    for k in range(K):
        sq_residuals = ((patches - gmm.mu[k]) / gmm.sigma[k])**2 - 1
        G_sigma[k] = (gamma[:, k:k+1] * sq_residuals).sum(0) / sqrt(2 * gmm.pi[k])
    
    fv = cat([G_mu.flatten(), G_sigma.flatten()])  # [4096]
    
    # Power + L2 normalization (critical for performance)
    fv = sign(fv) * sqrt(abs(fv) + 1e-8)
    fv = fv / (fv.norm() + 1e-8)
    
    return fv
```

**Interpretability**: FV dimension [k, d] (first-order) = "patches belonging to visual pattern k deviate from the average pattern k by this much in formula dimension d". Fully traceable.

### 3.4 Homogeneous Kernel Map (Deterministic Kernel Approximation)

Apply to final features before linear classifier. Gives the linear classifier chi-squared kernel SVM power.

```python
def homogeneous_kernel_map(x, order=1):
    """Maps each scalar feature to 2n+1 values.
    order=1 → each feature becomes 3 values.
    This is an EXACT, DETERMINISTIC, CLOSED-FORM feature map.
    Linear classifier on mapped features = chi-squared kernel SVM."""
    
    abs_x = torch.abs(x) + 1e-8
    sqrt_x = torch.sqrt(abs_x)
    log_x = torch.log(abs_x)
    
    features = [sqrt_x]
    for j in range(1, order + 1):
        features.append(sqrt_x * torch.cos(j * log_x) * math.sqrt(2 / math.pi))
        features.append(sqrt_x * torch.sin(j * log_x) * math.sqrt(2 / math.pi))
    
    return torch.stack(features, dim=-1)  # [batch, d, 3]
```

50K features × 3 = 150K mapped features. Linear classifier in this space ≈ chi-squared kernel SVM.

---

## Part 4: Training Pipeline

### Step 0: Kernel Pretraining (10 min)

Train 12 learnable kernels via a simple supervised task BEFORE RL.

```
Model: 10 terminals × 12 kernels → adaptive_avg_pool → Linear(120, 1000)
Data: 20K images (20/class), 112×112
Train: ~10 epochs
Output: kernel_bank_pretrained.pt
```

Phase 1 loads these pretrained values (frozen). This ensures conv3x3_0..7 and conv5x5_0..3 are meaningful filters, not random noise.

### Step 1: Phase 1A — Layer 1 RL (~30 min)

```
- Resolution: 112×112
- Training data: 20K images (20/class)
- 4 banks, each capacity=5,000
- min_accuracy_threshold: 0.002
- correlation_threshold: 0.92
- Learnable kernels: FROZEN at pretrained values
- Early stopping: 50 iters without bank growth
- Grammar: pooling only at final token (stack_depth=1), ban consecutive identical unary ops
- Target: 16,000–20,000 total formulas across 4 banks
```

### Step 2: Forward Selection (~1 hour)

Select top-100 most COMPLEMENTARY Layer 1 formula bodies.

This is NOT accuracy ranking — it's greedy selection by complementary information gain:

```
Prepare: 5K val images (5/class), encode each body with 8 SPP pools

Round 1: Pick body with highest standalone val accuracy → add to selected set
Round 2-100: 
  For each candidate (sample 500 random from remaining):
    Temporarily add candidate's features to selected set
    Quick eval: 3-5 SGD steps on val set
    Record accuracy gain
  Pick the candidate with largest gain
  
  Stop early if best gain < 0.1%
```

**Why this matters (Geometric MetaFormer theory)**: Forward selection serves as the "learnable channel mixer T" in the interleaved structure. It performs adaptive pointwise channel selection on Layer 1 feature maps before Layer 2 applies new geometry-aware spatial operations. This satisfies Proposition 1: spatial discrimination must be renewed by interleaving M (geometry) and T (learnable).

Output: `layer1_bases.json` — 100 formula bodies.

### Step 3: Phase 1B — Layer 2 RL (~30 min)

```
Terminals: 10 original channels + 100 Layer 1 feature maps = 110 terminals
4 banks, max_depth=5, max_seq_len=12 (shorter than L1 — L1 already did spatial transforms)
min_accuracy=0.002, correlation=0.92
Target: ~8,000–10,000 Layer 2 formulas
```

**Theoretical justification**: This realizes the composition principle from the Geometric MetaFormer paper:
```
F[l+1] = M_cw( T(F[l]) )

Where:
  F[l] = Layer 1 feature maps
  T    = forward selection (learnable channel mixing)
  M_cw = Layer 2 formulas (geometry-aware spatial operators applied channelwise)
```

Layer 2 formulas can express concepts like "L1_edge_map × L1_texture_map averaged over center region" — cross-formula spatial co-occurrence that Layer 1 alone cannot capture.

### Step 4: Phase 3 — Feature Extraction + Encoding (~8 hours)

**No Phase 2.** Phase 1's correlation gate + Phase 3's L1 selection handle quality filtering.

**4a. Distribution Statistics Extraction**

For each formula body (L1 + L2), at each resolution (112, 224):
- Execute body → feature map [B, H, W]
- Apply distribution encoding (12 stats × 5 regions = 60 per body per resolution)
- Store as float32 mmap

Total: ~(15K L1 + 8K L2) × 60 × 2 resolutions = ~2.76M raw features

**4b. L1 Feature Selection → ~50K**

Use L1-regularized linear classifier to select the most useful ~50K features from the 2.76M raw features.

**4c. Feature Interactions**

Take top-500 features by L1 weight magnitude.
Compute pairwise products: 500 × 499 / 2 = ~125K interaction features.
Each interaction = "formula_i stat × formula_j stat" — fully interpretable.
Run L1 selection again on (50K base + 125K interactions) → final ~50K.

**4d. Symbolic Fisher Vector**

Separately from the above:
- Use the 100 forward-selected bodies from Step 2
- Extract local descriptors: 8×8 patches × 100 bodies → 64 patches × 100-dim per image
- PCA reduce to 32-dim
- Fit GMM (K=64) on training descriptors
- Compute Fisher Vector per image: 4,096-dim
- Power + L2 normalize

**4e. Combine All Features**

```
Distribution stats (L1 selected): ~50K features
Fisher Vector: 4,096 features
Total: ~54K features
```

**4f. Apply Homogeneous Kernel Map**

54K × 3 (order=1 chi-squared map) = 162K features.
Or: apply only to top-20K features → 60K mapped + 34K unmapped = 94K.

**4g. Power Normalization + L2 Normalization**

Apply signed-sqrt + L2 norm to final feature vector.

### Step 5: Train Classifier

```
nn.Linear(n_features, 1000)
500K training images (500/class)
AdamW + CosineAnnealingLR
30 epochs
Sweep wd = [10, 20, 50, 100]
```

### Step 6: End-to-End Kernel Fine-Tuning (ONLY step needing online training)

```
Unfreeze SymbolicKernelBank
Jointly optimize: classifier (lr=1e-3, wd=10) + classic kernels (lr=1e-5) + learned kernels (lr=1e-4)
500K images, 10 epochs, online (features recomputed each step because kernels change)
After fine-tuning: re-extract features with updated kernels, retrain classifier offline
```

### Step 7: Scale to Full Dataset

Same as Step 6 but with full 1.18M training images.

---

## Part 5: Expected Accuracy Progression

| After | What was added | Estimated Top-1 |
|---|---|---|
| Phase 1A + distribution stats + power norm | Expanded operators, rich encoding, normalization | ~30% |
| + Layer 2 hierarchical formulas | Cross-formula spatial co-occurrence | ~37% |
| + Symbolic Fisher Vector | Local feature distribution encoding | ~43% |
| + Homogeneous kernel map + interactions | Nonlinear feature combinations | ~47% |
| + End-to-end kernel fine-tuning | Data-optimized filters | ~50% |
| + Full 1.18M training data | Reduced overfitting | ~53% |

---

## Part 6: Execution Batches for Claude Code

### Batch 1: Operators + Phase 1A (~1 day)

1. Add ALL new operators to `tensor_operators.py`:
   - Rotation-invariant: `edge_mag`, `edge_orient`, `gabor_mag`
   - Local structure: `local_contrast`, `dog`, `corner_harris`, `lbp_like`
   - Second-order: `edge_xx`, `edge_yy`
   - Variance pooling: `std_center`, `std_top_half`, `std_bottom_half`
   - Register all in TENSOR_OPERATORS, register pooling variants in ROOT_OPERATORS
2. Keep all v3 operators (opening, closing, tophat, high_freq, low_freq, distribution pooling, patch_histogram, SymbolicKernelBank, color terminals)
3. Ensure grammar rules: pooling only at final token (stack_depth=1), ban consecutive identical unary ops
4. Pretrain learnable kernels → `kernel_bank_pretrained.pt`
5. Run Phase 1A: 4 banks, 112×112, min_acc=0.002, corr=0.92, early stopping
6. Report: formula count per bank, operator usage stats, mean accuracy

### Batch 2: Layer 2 + Feature Extraction (~2 days)

7. Forward selection: pick top-100 most complementary L1 bodies
8. Run Phase 1B: Layer 2 RL, 110 terminals, 4 banks, max_depth=5
9. Implement distribution statistics encoding (12 stats × 5 regions × 2 resolutions)
10. Extract features for all L1+L2 bodies → mmap files
11. L1 feature selection → ~50K
12. Feature interactions (top-500 pairwise) → L1 selection again → ~50K
13. Apply power normalization + L2 normalization
14. Train classifier, sweep wd=[10, 20, 50, 100], 30 epochs
15. Report results — **this is our distribution-stats baseline**

### Batch 3: Fisher Vector + Kernel Map (~2 days)

16. Implement Symbolic Fisher Vector:
    - Extract 8×8 patch descriptors using 100 forward-selected bodies
    - PCA (32-dim) + GMM (K=64), fitted on training set
    - Compute FV per image (4,096-dim) + power+L2 norm
17. Combine: distribution stats features (~50K) + Fisher Vector (4,096)
18. Implement homogeneous kernel map (chi-squared, order=1)
19. Apply to combined features → train classifier
20. Report results — **this is our full-encoding baseline**

### Batch 4: End-to-End Fine-Tuning + Scale Up (~3 days)

21. End-to-end fine-tune learnable kernels + classifier (online, 500K images)
22. Re-extract features with updated kernels
23. Retrain classifier offline
24. Scale to full 1.18M training set
25. Final report with per-superclass accuracy, ablation study

---

## Key Constraints (MUST follow)

1. **No MLP, no hidden layers.** Classifier = `nn.Linear` only.
2. **Every feature traceable.** Each scalar = deterministic math applied to the image.
3. **FP32 throughout.** No FP16 for any operator execution.
4. **Learnable kernel parameters OK** — formula STRUCTURE is symbolic/RL-discovered, only conv kernel values are learned.
5. **No ImageNet mean/std normalization.** Formulas operate on raw [0,1] pixels.
6. **`_safe_binary` clamp: ±60000.**
7. **Grammar enforced:** pooling only at end, no consecutive identical unary ops.
8. **build_data_batch() must include ALL 10 terminals.** (v3 Phase 2 bug must not recur.)
9. **Skip Phase 2.** Use Phase 1 correlation gate + Phase 3 L1 selection instead.
10. **GMM for Fisher Vector is a fixed statistical model** — fitted once, never updated during classification training. Same conceptual role as PCA or StandardScaler.
