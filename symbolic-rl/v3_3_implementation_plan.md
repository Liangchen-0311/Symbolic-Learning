# v3.3 Implementation Plan — for Claude Code

> **Audience**: This document is an execution spec for Claude Code. It describes eight sections of upgrades to an existing symbolic-feature-discovery codebase for image classification. Implement them in the order given (see "Suggested Implementation Order"). Each section states **what to build**, **where it goes**, **acceptance criteria**, and **constraints**.
>
> **Project recap**: An RL agent (currently PPO) discovers symbolic formulas (RPN token sequences of image operators). Each formula maps an image → a scalar feature. An **interpretable, neuron-free classifier** is trained on these features. **Primary delivered classifier: HistGB + MI + sample_weight** (a collaborator-validated method that beats linear on small problems). **Reference: `nn.Linear`** (kept as the exact-additive-decomposition interpretability baseline and regression guard). See Section 6 for the full pluggable design. Everything is fully interpretable and deterministic. Current results (linear): CIFAR-10 76.84%, CIFAR-100 55.8%. Target: ImageNet-1K 50%+.

---

## Validated Empirical Findings (the evidence base for v3.3's design choices)

A collaborator applied this exact method to a **bone-fracture X-ray classification** task (10 classes, authoritative dataset, fracture and normal images drawn from the **same distribution/source** — so device/source shortcut is ruled out) and observed:

- **Linear classifier: 72% test accuracy. HistGB classifier: 95% test accuracy.** A +23-point jump on identical symbolic features.
- **Conclusion (well-supported): the symbolic features encode genuinely discriminative information *non-linearly*.** Linear reads only part of it (72%); a neuron-free non-linear classifier (HistGB) unlocks the rest (95%). This is real signal, not a shortcut (same-source data rules out device fingerprinting; `I_BONE`/`I_EDGE_PRIOR` are deterministic filters, not external models).
- **Top single formulas were weak (univariate acc 0.34–0.40); 95% comes from the *combination*.** This validates the "many weak symbolic learners + strong neuron-free ensemble" paradigm.
- **Feature importance was dominated by high-order statistics** (`.range`, `.std`, `.median`, `.skew`, `.q90`, `.q10`, `.kurtosis`) applied to formula feature-maps — the same formula under different statistical pooling (e.g. `formula[619].range`, `.std`, `.median`) all ranked in the top-7. This directly validates the Section 1A.2 statistical operators and the "one formula × many statistical pools" default (Section 1A.6).

These findings drive several v3.3 design decisions, flagged inline as **[Evidence: fracture study]**:
1. The line that "symbolic features are linearly separable" is **wrong** — they are non-linearly encoded; HistGB is primary, linear is the interpretability reference (Sections 6, methodology).
2. Pareto anti-bloat must be **classifier-dependent**: long formulas may encode useful non-linear structure under HistGB, so the length penalty is strict under linear but relaxed under HistGB (Section 2).
3. Feature selection / bank admission must use **MI (non-linear), not linear accuracy** — weak-linear formulas (0.34–0.40) are valuable in combination (Sections 6B, 7).
4. **Deterministic pre-computed prior terminals** (like the collaborator's `I_EDGE_PRIOR`) shorten formulas and help (Section 1A.0).
5. **One formula × many statistical pools** as a default expansion (Section 1A.6).
6. Interpretability's real value is **trust-verification** (occlusion / cross-source / probe tests), not pretty formulas — light decided the fracture result's validity, not formula readability (Section 8).

---

## Hard Constraints (NEVER violate these)

1. **No neurons in the classifier, and the classifier must stay interpretable.** The constraint is *not* "linear only" — it is "no artificial neurons (no weight-activation-backprop units, no MLP, no deep nets) **and** the decision process must remain human-interpretable." Permitted classifier families: **HistGB (primary delivered, with MI feature selection + `class_weight='balanced'`)**, `nn.Linear` (kept as the exact-additive-decomposition reference and regression guard), and **EBM / GA²M** (optional middle ground). All are neuron-free. Section 6 specifies parameter budgets and the multiclass tree-explosion caveat that forces HistGB to be combined with WordNet superclass decomposition. Large black-box GBDT ensembles (hundreds of deep trees on 1000-way flat) are allowed **only as an accuracy upper-bound reference**, never as the delivered model.
2. **Formula STRUCTURE is symbolic** (discovered by search). Only convolution-kernel *parameters* may be learned (the existing `SymbolicKernelBank`).
3. **FP32 for all computation.** FP16 is allowed *only* for on-disk/CPU feature-map caching (Section 4), and such cached values must be cast back to FP32 before any computation.
4. **Every feature must be traceable** to an explicit, deterministic mathematical expression on the input pixels.
5. **No external pretrained knowledge** (no CLIP, no GPT, no Grounding DINO, no word embeddings). WordNet is used *only* as a class-hierarchy lookup (an index of which ImageNet classes share a parent), never as a text/semantic feature source.
6. **Grammar rules preserved**: pooling operators only at the final token; ban consecutive identical unary operators.
7. Do **not** delete or rewrite existing working modules wholesale. Add new modules and extend registries. Keep backward compatibility so the existing PPO Layer-1 pipeline still runs.

---

## Codebase Reference (existing files)

These files already exist and define the current system. Read them before editing.

| File | Role |
|---|---|
| `tensor_operators.py` | Operator definitions + `TENSOR_OPERATORS`, `ROOT_OPERATORS`, `MULTI_DIM_OPERATORS` registries |
| `tensor_environment_large_bank.py` | RL environment, `TensorTokenVocabulary`, formula execution, action masking |
| `policy_agent.py` | `PolicyAgent` (LSTM controller, `action_head`, `value_head`, `sample_action`, `evaluate_actions`) |
| `ppo_trainer.py` | `PPOTrainer` (`collect_trajectories`, `compute_gae`, PPO update) |
| `reward.py` | `RewardShaper.compute_reward` (accuracy + parsimony + diversity) |
| `tensor_evaluator.py` | `TensorProgramEvaluator` (`execute_formula`, `evaluate_single_formula`, `evaluate_feature_bank`) |
| `tensor_action_masking.py`, `rpn_grammar_mask.py`, `action_mask.py` | Grammar/stack-based action masks |
| `l1_selected_bodies.json` | **Already produced**: top-100 Layer-1 formula bodies (RPN strings). Layer-2 will build on these. |
| `tensor_vsr_imagenet_v2.yaml` | Config (operators, thresholds, multi-bank settings) |
| `train_tensor_vsr_large_bank.py` | Main training entrypoint |

Operator registry format (in `tensor_operators.py`):
```python
TENSOR_OPERATORS = {
    'add': (TensorOperators.add, 2, 'tensor'),     # (fn, arity, output_type)
    'relu': (TensorOperators.relu, 1, 'tensor'),
    'global_avg_pool': (TensorOperators.global_avg_pool, 1, 'scalar'),
    ...
}
ROOT_OPERATORS = { 'global_avg_pool', ... }   # set of pooling ops that must be the final token
```

Terminal tokens currently: `['I_R', 'I_G', 'I_B', 'I_GRAY']` (in `TensorTokenVocabulary`), plus color terminals and learnable kernels `conv3x3_0..7`, `conv5x5_0..3` referenced in formula strings.

---

## Section 1 — New Operators (deterministic prior terminals + semantic + high-order statistical + fuzzy logic)

**Goal**: Increase the *expressiveness* of single formulas by adding (1A.0) deterministic pre-computed prior terminals, (1A.1) mid-level semantic operators, (1A.2) high-order statistical pooling operators, (1A.3) asymmetry/spatial-localization pooling, (1A.4) directional line operators, (1A.5) fuzzy-logic operators, and (1A.6) a one-formula-many-pools default. This is the single highest-leverage change for feature quality (current operators are mostly low-level edges/textures, and current pooling only captures first-order moments). The statistical pooling operators are especially synergistic with the HistGB classifier (Section 6): a distribution statistic like skewness or entropy gives HistGB exactly the kind of threshold-able signal it splits on (e.g. `if pool_skewness > 0.3 → bright-background scene`).

### 1A.0. Deterministic pre-computed prior terminals — **[Evidence: fracture study]**

**Motivation**: The fracture study's most-used input channel was `I_EDGE_PRIOR` = "pre-computed gradient magnitude" (it appeared in ~half of the top-20 formulas and top feature-importance list). A prior terminal is a **new leaf/input** for the RPN grammar — a feature map computed once by a fixed deterministic transform, added alongside `I_R/I_G/I_B/I_GRAY`. Benefits: (1) **shorter formulas** — the agent starts from "edge map" instead of re-deriving `... edge_mag ...` every time; (2) **frees RL budget** for higher-level composition instead of re-discovering that edges matter.

**CRITICAL constraint — these MUST be deterministic transforms, NOT learned/external models.** `I_EDGE_PRIOR` is just Sobel/gradient magnitude — a fixed formula, no training, no external weights. This keeps Hard Constraint #5 (no external pretrained knowledge) intact. A terminal computed by a pretrained network would violate the whole interpretability premise and is forbidden.

Add these terminals (computed once per image, cached as input channels, FP32):

```python
# In the terminal/channel construction (TensorTokenVocabulary / data pipeline).
# Each is a DETERMINISTIC transform of the grayscale (or per-channel) image — no learning.
def make_prior_terminals(img_gray):   # img_gray: [B, H, W]
    # I_EDGE: gradient magnitude (Sobel) — pre-computed edge prior (the collaborator's I_EDGE_PRIOR)
    gx = TensorOperators.edge_x(img_gray)
    gy = TensorOperators.edge_y(img_gray)
    I_EDGE = torch.sqrt(gx*gx + gy*gy + 1e-8)

    # I_FREQ: high-frequency residual = image - blurred (detail/texture prior), deterministic
    blurred = F.avg_pool2d(img_gray.unsqueeze(1), 5, 1, 2).squeeze(1)
    I_FREQ = img_gray - blurred

    # I_LAPLACIAN (optional): second-derivative edge prior, deterministic
    I_LAPLACIAN = TensorOperators.laplacian(img_gray)   # if available; else 4-neighbor laplacian
    return {'I_EDGE': I_EDGE, 'I_FREQ': I_FREQ, 'I_LAPLACIAN': I_LAPLACIAN}
```

Register `I_EDGE`, `I_FREQ` (and optionally `I_LAPLACIAN`) as **terminal tokens** in `TensorTokenVocabulary` (the same place `I_R/I_G/I_B/I_GRAY` are defined), so the RL grammar can start formulas from them. They behave exactly like existing terminals (leaf nodes, output a 2D map). Make them **toggleable in config** (`prior_terminals: [I_EDGE, I_FREQ]`) so their contribution can be ablated.

> Note: this is a lightweight cousin of the Layer-2 idea (Section 4) — Layer-2 promotes *discovered* formulas to terminals; 1A.0 promotes *known deterministic transforms* to terminals. Both shorten formulas by providing higher-level starting points.

### 1A.1. Mid-level semantic operators (HIGH PRIORITY)

Add the following `@staticmethod` methods to the `TensorOperators` class in `tensor_operators.py`. All operate on `[B, H, W]` tensors and return `[B, H, W]` (unary, `'tensor'` output) unless noted. All must be FP32 and numerically safe (add `1e-8` under sqrt/div).

```python
@staticmethod
def blob_detector(x):
    """Multi-scale blob response via Difference-of-Gaussians normalized by scale.
    Detects roughly circular/compact regions (objects, heads, fruit).
    LoG approximation: blur_small - blur_large, then abs for scale-invariant response."""
    x4d = x.unsqueeze(1)
    fine   = F.avg_pool2d(x4d, kernel_size=3, stride=1, padding=1)
    medium = F.avg_pool2d(x4d, kernel_size=7, stride=1, padding=3)
    coarse = F.avg_pool2d(x4d, kernel_size=11, stride=1, padding=5)
    # Two DoG bands → blob energy across scales
    dog1 = fine - medium
    dog2 = medium - coarse
    return torch.sqrt(dog1 * dog1 + dog2 * dog2 + 1e-8).squeeze(1)

@staticmethod
def symmetry_v(x):
    """Vertical-axis (left-right) mirror symmetry score per pixel.
    High where the local neighborhood is symmetric about the image's vertical axis.
    Output: similarity in [0,1]-ish via exp(-|x - flip(x)|). Objects (faces, cars,
    animals seen frontally) are often left-right symmetric."""
    flipped = torch.flip(x, dims=[-1])
    return torch.exp(-torch.abs(x - flipped))

@staticmethod
def symmetry_h(x):
    """Horizontal-axis (top-bottom) mirror symmetry score per pixel.
    Reflections in water, symmetric scenes."""
    flipped = torch.flip(x, dims=[-2])
    return torch.exp(-torch.abs(x - flipped))

@staticmethod
def contour(x):
    """Contour / closed-boundary strength: gradient magnitude gated by local
    coherence. Approximates 'how strongly does a closed object boundary pass here'.
    = edge_mag * sigmoid(local structure-tensor coherence)."""
    ex = TensorOperators.edge_x(x)
    ey = TensorOperators.edge_y(x)
    mag = torch.sqrt(ex * ex + ey * ey + 1e-8)
    # Structure-tensor coherence: how oriented (edge-like vs flat/corner) the region is
    ex2 = F.avg_pool2d((ex*ex).unsqueeze(1), 5, 1, 2).squeeze(1)
    ey2 = F.avg_pool2d((ey*ey).unsqueeze(1), 5, 1, 2).squeeze(1)
    exy = F.avg_pool2d((ex*ey).unsqueeze(1), 5, 1, 2).squeeze(1)
    trace = ex2 + ey2 + 1e-8
    diff  = torch.sqrt((ex2 - ey2)**2 + 4*exy*exy + 1e-8)
    coherence = diff / trace          # in [0,1], 1 = strongly oriented edge
    return mag * coherence

@staticmethod
def elongation(x):
    """Local elongation / anisotropy from the structure tensor eigenvalue ratio.
    High for elongated structures (limbs, poles, stems), low for blobs/flat."""
    ex = TensorOperators.edge_x(x)
    ey = TensorOperators.edge_y(x)
    ex2 = F.avg_pool2d((ex*ex).unsqueeze(1), 5, 1, 2).squeeze(1)
    ey2 = F.avg_pool2d((ey*ey).unsqueeze(1), 5, 1, 2).squeeze(1)
    exy = F.avg_pool2d((ex*ey).unsqueeze(1), 5, 1, 2).squeeze(1)
    tmp = torch.sqrt((ex2 - ey2)**2 + 4*exy*exy + 1e-8)
    lam1 = 0.5 * (ex2 + ey2 + tmp)    # larger eigenvalue
    lam2 = 0.5 * (ex2 + ey2 - tmp)    # smaller eigenvalue
    return (lam1 - lam2) / (lam1 + lam2 + 1e-8)   # anisotropy in [0,1]

@staticmethod
def radial_gradient(x):
    """Radial-vs-tangential gradient alignment about the image center.
    Captures 'pointing toward/away from center' structure — useful for centered
    objects vs background. Deterministic, center-referenced."""
    B, H, W = x.shape
    ys = torch.linspace(-1, 1, H, device=x.device).view(1, H, 1).expand(B, H, W)
    xs = torch.linspace(-1, 1, W, device=x.device).view(1, 1, W).expand(B, H, W)
    norm = torch.sqrt(xs*xs + ys*ys + 1e-8)
    rx, ry = xs / norm, ys / norm     # unit radial vectors
    ex = TensorOperators.edge_x(x)
    ey = TensorOperators.edge_y(x)
    return ex * rx + ey * ry          # projection of gradient onto radial direction
```

If `edge_mag`, `dog`, `corner_harris`, `lbp_like`, `local_contrast`, `edge_xx`, `edge_yy`, `gabor_mag` are **not yet present** (they were specced in v3.2), add them too — definitions are in the v3.2 plan; reuse those exact implementations.

### 1A.2. High-order statistical pooling operators (HIGH PRIORITY)

**Motivation**: Every existing pooling operator captures only *first-order* moments (mean/max/min/std/L2). Two images with identical mean and std can have completely different distribution *shapes* (normal vs bimodal vs skewed) — current operators cannot tell them apart. Image-class discrimination on ImageNet leans heavily on global intensity/color *distribution shape* and *texture statistics*, which these operators expose directly. They are also the strongest synergy with HistGB (threshold-able distribution statistics → readable split rules).

These are **pooling operators**: they take `[B, H, W]` and return `[B]` (scalar, `'scalar'` output), and they **must be added to `ROOT_OPERATORS`** (they may appear only as the final token, like the existing pooling ops). All FP32, numerically safe.

**Tier 1 — MUST ADD (6 ops): distribution-shape moments + quantiles.** These fill the biggest gap (no shape information beyond first order).

```python
@staticmethod
def pool_skewness(x):
    """3rd standardized moment — distribution asymmetry.
    e.g. bright-sky scenes (bright pixels dominate) vs night scenes have opposite sign."""
    flat = x.reshape(x.shape[0], -1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True) + 1e-8
    return (((flat - mean) / std) ** 3).mean(dim=1)

@staticmethod
def pool_kurtosis(x):
    """4th standardized moment minus 3 — tail heaviness.
    sparse strong edges (high) vs uniform texture (low)."""
    flat = x.reshape(x.shape[0], -1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True) + 1e-8
    return (((flat - mean) / std) ** 4).mean(dim=1) - 3.0

@staticmethod
def pool_q10(x):
    """10th percentile — representative dark-region value."""
    return torch.quantile(x.reshape(x.shape[0], -1), 0.10, dim=1)

@staticmethod
def pool_q90(x):
    """90th percentile — representative bright-region value."""
    return torch.quantile(x.reshape(x.shape[0], -1), 0.90, dim=1)

@staticmethod
def pool_iqr(x):
    """Inter-quartile range q75-q25 — robust spread, outlier-resistant vs std."""
    flat = x.reshape(x.shape[0], -1)
    return torch.quantile(flat, 0.75, dim=1) - torch.quantile(flat, 0.25, dim=1)

@staticmethod
def pool_above_mean_ratio(x):
    """Fraction of pixels above the mean — 'bright-area occupancy'.
    high for sky/snow scenes, low for dark scenes."""
    flat = x.reshape(x.shape[0], -1)
    mean = flat.mean(dim=1, keepdim=True)
    return (flat > mean).float().mean(dim=1)
```

**Tier 2 — RECOMMENDED (3 ops): information-theoretic statistics.** Entropy/uniformity are classic texture discriminators (GLCM, Tamura). Use soft (differentiable) histograms.

```python
@staticmethod
def pool_entropy(x):
    """Shannon entropy of a 32-bin soft histogram (per-sample min-max normalized).
    high = complex texture/diverse content; low = flat/single-color region."""
    flat = x.reshape(x.shape[0], -1)
    mn = flat.min(dim=1, keepdim=True).values
    mx = flat.max(dim=1, keepdim=True).values
    norm = (flat - mn) / (mx - mn + 1e-8)
    bins = 32
    centers = torch.linspace(0, 1, bins, device=x.device).view(1, bins, 1)
    soft_hist = torch.exp(-100.0 * (norm.unsqueeze(1) - centers) ** 2)   # [B, bins, N]
    hist = soft_hist.sum(dim=2)
    hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)
    return -(hist * torch.log(hist + 1e-8)).sum(dim=1)

@staticmethod
def pool_energy(x):
    """Mean squared value — signal energy normalized by area."""
    return (x ** 2).mean(dim=(-2, -1))

@staticmethod
def pool_uniformity(x):
    """Sum of squared histogram probabilities (Σ p_i^2) — inverse of dispersion.
    high = pixel values concentrated (single color); low = spread out."""
    flat = x.reshape(x.shape[0], -1)
    mn = flat.min(dim=1, keepdim=True).values
    mx = flat.max(dim=1, keepdim=True).values
    norm = (flat - mn) / (mx - mn + 1e-8)
    bins = 16
    centers = torch.linspace(0, 1, bins, device=x.device).view(1, bins, 1)
    soft_hist = torch.exp(-100.0 * (norm.unsqueeze(1) - centers) ** 2)
    hist = soft_hist.sum(dim=2)
    hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)
    return (hist ** 2).sum(dim=1)
```

**Tier 3 — OPTIONAL (2 ops): co-occurrence / spatial-correlation statistics.** Simplified GLCM-style texture statistics. Add only if Tier 1+2 prove useful in the CIFAR ablation.

```python
@staticmethod
def pool_neighbor_diff_var(x):
    """Variance of adjacent-pixel differences — local contrast statistic.
    high = high-frequency texture (fur, fabric), low = flat."""
    diff_x = x[:, :, 1:] - x[:, :, :-1]
    diff_y = x[:, 1:, :] - x[:, :-1, :]
    return diff_x.var(dim=(-2, -1)) + diff_y.var(dim=(-2, -1))

@staticmethod
def pool_autocorr_lag1(x):
    """Lag-1 horizontal autocorrelation — spatial smoothness.
    high = smooth, low = noisy/high-frequency."""
    fx = x[:, :, 1:].reshape(x.shape[0], -1)
    fy = x[:, :, :-1].reshape(x.shape[0], -1)
    mx = fx.mean(dim=1, keepdim=True); my = fy.mean(dim=1, keepdim=True)
    cov = ((fx - mx) * (fy - my)).mean(dim=1)
    return cov / (fx.std(dim=1) * fy.std(dim=1) + 1e-8)
```

**Do NOT add** (deliberately excluded to avoid bloat / output-type problems): raw histograms (multi-dim output — violates scalar-output contract; their discriminative power is already captured indirectly by entropy + skewness + kurtosis + quantiles), Fourier spectra (overlaps Gabor, poor human readability), Hu moments (strong invariance not needed for ImageNet; low priority).

### 1A.3. Asymmetry + spatial-localization pooling operators (RECOMMENDED) — **[Evidence: fracture study]**

The fracture study's operator set included `lr_asymmetry` (left-right asymmetry → unilateral fractures) and fine-grained spatial pools (`pool_quad_tr`, `pool_thirds_mid`, `pool_surround`, `peak_location_y`). These are **general-purpose** (not domain-specific) and worth adding: asymmetry is a strong signal across tasks (faces/cars/animals are often symmetric; symmetry *breaks* flag anomalies), and spatial-localization pools let a formula express **where** a feature is, not just how much.

**Important — add only general operators, not the collaborator's domain-specific ones.** Do NOT port `cortical_cont`, `discont_map`, `bone_enhance`, `displace_ind`, `threshold_bone` — those are fracture-specific and meaningless for CIFAR/ImageNet. (Note the irony observed in the study: the *most* important formulas leaned on general texture/direction operators like `lbp_like`, `local_range`, `gabor`, `line`, not the hand-designed pathology operators — general composable operators outperform narrow ones because RL can compose specialized effects from them.)

**Asymmetry pooling ops (SCALAR output → ROOT ops):**
```python
@staticmethod
def pool_lr_asymmetry(x):
    """Mean absolute left-right difference: |x - fliplr(x)| pooled.
    high = left/right halves differ (unilateral structure); low = symmetric."""
    flipped = torch.flip(x, dims=[-1])
    return torch.abs(x - flipped).mean(dim=(-2, -1))

@staticmethod
def pool_tb_asymmetry(x):
    """Mean absolute top-bottom difference: |x - flipud(x)| pooled."""
    flipped = torch.flip(x, dims=[-2])
    return torch.abs(x - flipped).mean(dim=(-2, -1))
```

**Spatial-localization pooling ops (SCALAR output → ROOT ops):**
```python
@staticmethod
def pool_surround(x):
    """Mean of the border region minus mean of the central region.
    high = energy concentrated at edges/periphery; low = centered."""
    B, H, W = x.shape
    cy0, cy1 = H // 4, 3 * H // 4
    cx0, cx1 = W // 4, 3 * W // 4
    center = x[:, cy0:cy1, cx0:cx1].mean(dim=(-2, -1))
    total  = x.mean(dim=(-2, -1))
    # surround mean ≈ (total*area - center*area_c)/(area - area_c); use simple proxy:
    return total - center      # >0 when periphery brighter than center

@staticmethod
def pool_quadrant_tr(x):
    """Mean of the top-right quadrant (example fine-grained spatial pool).
    Add the other three quadrants analogously (tl, bl, br) if useful."""
    B, H, W = x.shape
    return x[:, :H // 2, W // 2:].mean(dim=(-2, -1))

@staticmethod
def pool_thirds_mid(x):
    """Mean of the central horizontal third (band) of the image."""
    B, H, W = x.shape
    return x[:, H // 3: 2 * H // 3, :].mean(dim=(-2, -1))

@staticmethod
def peak_location_y(x):
    """Normalized vertical position (0=top,1=bottom) of the max-response pixel.
    Returns *where* the strongest response is, not its magnitude — a positional feature."""
    B, H, W = x.shape
    flat_idx = x.reshape(B, -1).argmax(dim=1)
    row = (flat_idx // W).float() / max(H - 1, 1)
    return row

@staticmethod
def peak_location_x(x):
    """Normalized horizontal position (0=left,1=right) of the max-response pixel."""
    B, H, W = x.shape
    flat_idx = x.reshape(B, -1).argmax(dim=1)
    col = (flat_idx % W).float() / max(W - 1, 1)
    return col
```

### 1A.4. Directional line/edge operators (OPTIONAL) — **[Evidence: fracture study]**

The study used explicit multi-orientation detectors (`line_h`, `line_v`, `line_45`, `line_135`, `edge_diag_45`, `edge_diag_135`) that gave finer orientation selectivity than a single `gabor_mag`. Add a small oriented-line family (unary, tensor output) if the CIFAR ablation shows orientation matters; otherwise defer.

```python
@staticmethod
def line_h(x):
    """Horizontal line/ridge response (second derivative across rows)."""
    k = torch.tensor([[-1.,-1.,-1.],[2.,2.,2.],[-1.,-1.,-1.]], device=x.device).view(1,1,3,3)
    return F.conv2d(x.unsqueeze(1), k, padding=1).squeeze(1)

@staticmethod
def line_v(x):
    """Vertical line/ridge response."""
    k = torch.tensor([[-1.,2.,-1.],[-1.,2.,-1.],[-1.,2.,-1.]], device=x.device).view(1,1,3,3)
    return F.conv2d(x.unsqueeze(1), k, padding=1).squeeze(1)

@staticmethod
def line_diag45(x):
    """45° diagonal line/ridge response."""
    k = torch.tensor([[-1.,-1.,2.],[-1.,2.,-1.],[2.,-1.,-1.]], device=x.device).view(1,1,3,3)
    return F.conv2d(x.unsqueeze(1), k, padding=1).squeeze(1)

@staticmethod
def line_diag135(x):
    """135° diagonal line/ridge response."""
    k = torch.tensor([[2.,-1.,-1.],[-1.,2.,-1.],[-1.,-1.,2.]], device=x.device).view(1,1,3,3)
    return F.conv2d(x.unsqueeze(1), k, padding=1).squeeze(1)
```

### 1A.5. Fuzzy-logic operators (MEDIUM PRIORITY)

Add fuzzy operators. **Use the product/probabilistic form, not min/max**, because product forms are differentiable everywhere (needed for the optional end-to-end kernel fine-tuning) and avoid sparse-gradient issues. Inputs are squashed to [0,1] via sigmoid first so the fuzzy semantics ("degree of truth") are meaningful.

```python
@staticmethod
def fuzzy_not(x):
    """Fuzzy NOT: 1 - σ(x). Output in (0,1)."""
    return 1.0 - torch.sigmoid(x)

@staticmethod
def fuzzy_and(x, y):
    """Fuzzy AND (probabilistic t-norm): σ(x) * σ(y). Output in (0,1).
    'This region satisfies BOTH feature x AND feature y.'"""
    return torch.sigmoid(x) * torch.sigmoid(y)

@staticmethod
def fuzzy_or(x, y):
    """Fuzzy OR (probabilistic t-conorm): σ(x)+σ(y)-σ(x)σ(y). Output in (0,1).
    'This region satisfies feature x OR feature y.'"""
    sx, sy = torch.sigmoid(x), torch.sigmoid(y)
    return sx + sy - sx * sy
```

### 1A.6. Default expansion: one formula × many statistical pools — **[Evidence: fracture study]**

In the fracture study, the *same* formula feature-map appeared multiple times in the top feature-importance list under *different* statistical pools (`formula[619].range`, `.std`, `.median` all in top-7). The different statistics carry **non-redundant** information (range = dynamic span, std = dispersion, median = typical value), and the classifier (HistGB) picks whichever it needs.

**Make this a default, automatic expansion** — independent of, and complementary to, the RL-selected pooling token (1A.2):

- When a formula body (the pre-final-pool feature map) enters the bank, **automatically compute a fixed battery of statistical pools** over it: `{mean, std, range, median, q10, q90, iqr, skewness, kurtosis, l1_norm, l2_norm, max, min, above_mean_ratio}`. Each produces one scalar feature; one formula thus yields ~14 features.
- This does **not** consume RL search budget (the RL agent still picks one pooling token per formula as before; this battery is applied *post-hoc* to the body, cheaply, for all bank formulas).
- Implement as a deterministic function `expand_formula_statistics(feature_map) -> dict[str, scalar]`, applied during feature extraction (Section 4D) before the classifier. Feature names follow the study's convention `formula[i].<stat>` so they remain fully traceable.
- The downstream MI selection (Section 6B) and bank reshuffle (Section 7) operate on this expanded feature set, so useless statistics are pruned automatically — keep admission wide, let selection/HistGB decide which stats matter.

### 1A.7. Register the new operators

In `tensor_operators.py`, add entries to `TENSOR_OPERATORS`:

```python
# --- v3.3 semantic operators (unary, tensor output) ---
'blob_detector':  (TensorOperators.blob_detector, 1, 'tensor'),
'symmetry_v':     (TensorOperators.symmetry_v, 1, 'tensor'),
'symmetry_h':     (TensorOperators.symmetry_h, 1, 'tensor'),
'contour':        (TensorOperators.contour, 1, 'tensor'),
'elongation':     (TensorOperators.elongation, 1, 'tensor'),
'radial_gradient':(TensorOperators.radial_gradient, 1, 'tensor'),

# --- v3.3 directional line operators (1A.4, unary tensor; OPTIONAL) ---
'line_h':       (TensorOperators.line_h, 1, 'tensor'),
'line_v':       (TensorOperators.line_v, 1, 'tensor'),
'line_diag45':  (TensorOperators.line_diag45, 1, 'tensor'),
'line_diag135': (TensorOperators.line_diag135, 1, 'tensor'),

# --- v3.3 high-order statistical pooling (unary, SCALAR output → ROOT ops) ---
# Tier 1 (must add)
'pool_skewness':        (TensorOperators.pool_skewness, 1, 'scalar'),
'pool_kurtosis':        (TensorOperators.pool_kurtosis, 1, 'scalar'),
'pool_q10':             (TensorOperators.pool_q10, 1, 'scalar'),
'pool_q90':             (TensorOperators.pool_q90, 1, 'scalar'),
'pool_iqr':             (TensorOperators.pool_iqr, 1, 'scalar'),
'pool_above_mean_ratio':(TensorOperators.pool_above_mean_ratio, 1, 'scalar'),
# Tier 2 (recommended)
'pool_entropy':         (TensorOperators.pool_entropy, 1, 'scalar'),
'pool_energy':          (TensorOperators.pool_energy, 1, 'scalar'),
'pool_uniformity':      (TensorOperators.pool_uniformity, 1, 'scalar'),
# Tier 3 (optional — gate on CIFAR ablation)
'pool_neighbor_diff_var':(TensorOperators.pool_neighbor_diff_var, 1, 'scalar'),
'pool_autocorr_lag1':   (TensorOperators.pool_autocorr_lag1, 1, 'scalar'),

# --- v3.3 asymmetry + spatial-localization pooling (1A.3, SCALAR → ROOT ops) ---
'pool_lr_asymmetry': (TensorOperators.pool_lr_asymmetry, 1, 'scalar'),
'pool_tb_asymmetry': (TensorOperators.pool_tb_asymmetry, 1, 'scalar'),
'pool_surround':     (TensorOperators.pool_surround, 1, 'scalar'),
'pool_quadrant_tr':  (TensorOperators.pool_quadrant_tr, 1, 'scalar'),
'pool_thirds_mid':   (TensorOperators.pool_thirds_mid, 1, 'scalar'),
'peak_location_y':   (TensorOperators.peak_location_y, 1, 'scalar'),
'peak_location_x':   (TensorOperators.peak_location_x, 1, 'scalar'),

# --- v3.3 fuzzy logic (tensor output) ---
'fuzzy_not': (TensorOperators.fuzzy_not, 1, 'tensor'),
'fuzzy_and': (TensorOperators.fuzzy_and, 2, 'tensor'),
'fuzzy_or':  (TensorOperators.fuzzy_or, 2, 'tensor'),
```

**Add ALL scalar-output ops to `ROOT_OPERATORS`** (statistical + asymmetry + spatial-localization — they output scalars and may appear only as the final token):

```python
ROOT_OPERATORS |= {
    # statistical (1A.2)
    'pool_skewness', 'pool_kurtosis', 'pool_q10', 'pool_q90', 'pool_iqr',
    'pool_above_mean_ratio', 'pool_entropy', 'pool_energy', 'pool_uniformity',
    'pool_neighbor_diff_var', 'pool_autocorr_lag1',
    # asymmetry + spatial localization (1A.3)
    'pool_lr_asymmetry', 'pool_tb_asymmetry', 'pool_surround',
    'pool_quadrant_tr', 'pool_thirds_mid', 'peak_location_y', 'peak_location_x',
}
```

`fuzzy_and`/`fuzzy_or` are binary and must be wrapped with the existing `_safe_binary` decorator/size-matching path, exactly like `add`/`subtract`.

Tensor-output ops (semantic 1A.1, directional 1A.4, fuzzy 1A.5) are **not** pooling ops → do **not** add to `ROOT_OPERATORS`. All scalar-output ops (statistical 1A.2, asymmetry/spatial 1A.3) **are** pooling ops → they **must** be in `ROOT_OPERATORS`.

### Acceptance criteria — Section 1
- [ ] All new operators importable. With everything added: `TENSOR_OPERATORS` grows by 33 (6 semantic + 4 directional + 11 statistical + 7 asymmetry/spatial + 3 fuzzy + 2 extra statistical pools already counted). Deferring optional tiers (1A.4 directional, Tier-3 statistical) reduces this; log the exact count.
- [ ] `ROOT_OPERATORS` gains all scalar-output ops (11 statistical + 7 asymmetry/spatial = 18, fewer if Tier-3 deferred); no tensor-output op is in `ROOT_OPERATORS`.
- [ ] Unit test: each tensor-output unary op maps `[4, 32, 32]` → `[4, 32, 32]`; each scalar-output pooling op maps `[4, 32, 32]` → `[4]`; `peak_location_*` returns values in [0,1]; no NaN/Inf; FP32 in/out.
- [ ] Binary fuzzy ops handle mismatched spatial sizes via `_safe_binary`.
- [ ] Grammar check: every scalar-output op is accepted **only** as the final token; rejected mid-formula.
- [ ] `expand_formula_statistics(feature_map)` (1A.6) returns the ~14-stat battery as named scalars `formula[i].<stat>`; deterministic; used in Section 4D feature extraction.
- [ ] Formula strings `"I_R blob_detector pool_center"`, `"I_R edge_x pool_skewness"`, `"I_R pool_lr_asymmetry"`, `"I_R edge_y peak_location_y"` execute end-to-end through `TensorProgramEvaluator.execute_formula`.
- [ ] Existing Layer-1 PPO run still launches without errors (backward compatible).

---

## Section 2 — GRPO with Pareto Group-Relative Advantage

**Goal**: Replace PPO's critic-based advantage with **Group Relative Policy Optimization (GRPO)**, and make the group-relative ranking **Pareto-based** over `(accuracy↑, length↓, depth↓)` to control formula bloat. This removes the critic (which is hard to train on the discontinuous symbolic reward landscape) and biases toward short, shallow formulas. **Critical caveat — the length penalty must be classifier-dependent (see 2E): [Evidence: fracture study] showed symbolic features encode useful *non-linear* structure, so long formulas are not always bloat under a non-linear classifier (HistGB).**

### 2A. New module: `grpo_trainer.py`

Create `grpo_trainer.py` modeled on `ppo_trainer.py` but with these differences:

1. **No value/critic.** Ignore `policy.value_head` (leave it in `PolicyAgent` for backward compat, but GRPO does not use or train it). Advantage comes from group-relative Pareto rank, not GAE.

2. **Group sampling.** Each update step samples a *group* of `G` complete formulas (default `G = 16`) from the current policy (all from the empty-stack start state).

3. **Pareto non-dominated sort** over the group. For each formula collect:
   - `accuracy` (from `TensorProgramEvaluator.evaluate_single_formula`, maximize)
   - `length` = number of tokens (minimize)
   - `depth`  = expression-tree depth of the RPN (minimize)

   Implement `fast_non_dominated_sort(objectives)` (standard NSGA-II routine). A formula A dominates B iff A is ≥ on every objective and strictly > on at least one (with sign per the min/max direction). Output an integer `rank` per formula (0 = Pareto front, 1 = next front, …).

4. **Crowding distance** within each rank (standard NSGA-II) to preserve diversity and break ties toward more unique formulas.

5. **Group-relative advantage** from rank + crowding:
   ```python
   raw_score = -(rank.float()) + 0.1 * normalized_crowding   # higher = better
   advantage = (raw_score - raw_score.mean()) / (raw_score.std() + 1e-8)
   ```
   Every token in a formula's trajectory receives that formula's scalar `advantage` (trajectory-level credit assignment, as in standard GRPO).

6. **GRPO policy loss** — same clipped surrogate as PPO but with the group-relative advantage and no value loss:
   ```python
   ratio = exp(new_logprob - old_logprob)            # per token
   L_clip = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
   loss   = L_clip.mean() - entropy_coef * entropy    # NO value_coef term
   ```
   Keep `clip_epsilon` (0.2), `entropy_coef` (reuse existing schedule), `max_grad_norm` (0.5) from config.

7. **Parsimony tie-break (the user's "强行卡死" rule), with a classifier-dependent strength `λ_len` (see 2E).** Within a single Pareto front (same rank), if two formulas have `accuracy` within `acc_tol = 0.003` of each other, the shorter/shallower one receives a higher `raw_score`. Implement as a configurable lexicographic nudge:
   ```python
   raw_score -= lambda_len * length + lambda_len * depth   # lambda_len from config (2E)
   ```
   Under a **linear** classifier set `λ_len` strict (e.g. 1e-3); under **HistGB** relax it (e.g. 2e-4 or 0) so useful long non-linear formulas survive.

### 2B. Depth computation

Add a helper (in `grpo_trainer.py` or a small `formula_utils.py`) that computes RPN expression-tree depth from a token list, using arity from `TENSOR_OPERATORS` (terminals = depth 1; unary = child+1; binary = max(children)+1; pooling = child+1).

### 2C. Config + entrypoint wiring

In `tensor_vsr_imagenet_v2.yaml` add:
```yaml
search_algorithm: grpo        # one of: ppo | grpo  (default grpo for v3.3)
grpo:
  group_size: 16
  acc_tol: 0.003              # accuracy tolerance for parsimony tie-break
  pareto_objectives: [accuracy, length, depth]
  crowding_weight: 0.1
  lambda_len: 1.0e-3          # parsimony strength; SET BY CLASSIFIER (see 2E):
                              #   linear → 1e-3 (strict anti-bloat)
                              #   histgb → 2e-4 or 0 (relaxed; keep useful long non-linear formulas)
```
In `train_tensor_vsr_large_bank.py`, branch on `search_algorithm`: if `grpo`, instantiate `GRPOTrainer`; else keep `PPOTrainer`. **Both must remain runnable.**

### 2D. Why GRPO + Pareto (record in code comments)
- GRPO drops the critic, which on symbolic search has a near-impossible job (it must predict the value of half-finished, non-executable formulas in a highly discontinuous reward landscape). Group-relative ranking sidesteps this.
- Pareto dominance over (accuracy, length, depth) controls **bloat** without a hand-tuned weight λ in the *dominance* test: a 51%-accuracy / length-50 formula does NOT dominate a 50%-accuracy / length-10 formula. The only tunable knob is the **tie-break** strength `λ_len` (2E).

### 2E. Classifier-dependent length penalty — **[Evidence: fracture study]**

**The correction**: our earlier assumption was "long formulas are bloat; shortening them won't hurt accuracy." The fracture study refutes this *for non-linear classifiers*. There, symbolic features encoded discriminative information **non-linearly** (linear 72% → HistGB 95%), and the top formulas were long (15–18 tokens). A long formula like `if A and B but not C` has low *linear* univariate accuracy yet provides a valuable non-linear building block that HistGB exploits. Aggressively shortening such formulas would discard real signal.

**Therefore the length penalty `λ_len` must depend on the downstream classifier:**

| Downstream classifier | `λ_len` | Rationale |
|---|---|---|
| **Linear** (additive) | strict, e.g. `1e-3` | A linear model can only sum features; long formulas that merely stack terms ARE redundant (the classifier could replicate them), so anti-bloat is free and improves readability. |
| **HistGB** (non-linear, primary) | relaxed, e.g. `2e-4` or `0` | Long formulas may encode non-linear structure the classifier genuinely uses; over-penalizing length would cost accuracy. Keep a *mild* penalty only to break exact ties and curb runaway length. |

**Wiring**: `λ_len` is read from `grpo.lambda_len`, and the orchestrator (Section 5) sets it from `classifier.type` (linear → strict, histgb → relaxed). Hard length/depth caps from the grammar (`max_seq_len`) still apply in both cases as a safety ceiling — relaxing `λ_len` loosens the *soft* preference, not the *hard* limit.

**Required ablation** (CIFAR-10, record in report): for HistGB, compare final accuracy under strict vs relaxed `λ_len`. If relaxed wins, it confirms long formulas carry non-linear signal under HistGB; if strict wins or ties, shorter formulas suffice and we keep them for interpretability. This ablation directly tests the fracture-study hypothesis on our own data.

### Acceptance criteria — Section 2
- [ ] `grpo_trainer.py` exists; `GRPOTrainer` has the same outward interface as `PPOTrainer` (so the entrypoint can swap them).
- [ ] `fast_non_dominated_sort` unit-tested on a toy set of 5 (acc, len, depth) triples with a hand-checked expected ranking.
- [ ] Depth function unit-tested: `"I_R edge_x pool_center"` → depth 3; `"I_R I_G add pool_center"` → depth 3; nested cases correct.
- [ ] `λ_len` is read from config and set by classifier type (linear→strict, histgb→relaxed); both settings run.
- [ ] A short GRPO run (e.g. 50 groups) on CIFAR-10 completes, produces formulas; under strict `λ_len` mean formula length does **not** grow monotonically (bloat check); under relaxed `λ_len` longer formulas are permitted.
- [ ] The `λ_len` strict-vs-relaxed accuracy ablation (HistGB, CIFAR-10) is run and logged.
- [ ] `search_algorithm: ppo` still works unchanged.

---

## Section 3 — WordNet Hierarchical Class Decomposition

**Goal**: On ImageNet, decompose the 1000-way problem into a coarse superclass classifier plus per-superclass fine classifiers, using the WordNet hierarchy. Each classifier is a separate **interpretable neuron-free model** (pluggable per Section 6 via `classifier_type`). This converts one intractable 1000-way problem into ~1 coarse + ~20 fine problems, each near CIFAR-100 scale. **This decomposition is *mandatory* when the classifier is HistGB**, because multiclass GBDT trains one tree per class per round; without superclass decomposition a flat 1000-way HistGB explodes to `max_iter × 1000` trees (Section 6.0).

### 3A. Build the hierarchy: `wordnet_hierarchy.py`

- Input: ImageNet-1K class list (WordNet synset IDs, e.g. `n01440764`). These ship with the dataset; do **not** download external embeddings.
- Use the offline WordNet graph (via `nltk.corpus.wordnet`, which is a local lexical database, not a pretrained model — this is allowed; it is a class-index lookup, not a semantic feature source). If `nltk` WordNet data is unavailable in the environment, fall back to a **static hand-mapped table** committed to the repo (`imagenet_superclasses.json`) mapping each of the 1000 synsets to one of ~20 superclasses.
- Produce ~20 superclasses by cutting the WordNet tree at a depth that yields balanced groups (target 15–130 leaf classes per superclass). Typical groups: dog, other-mammal, bird, reptile, fish, invertebrate, vehicle, instrument, tool, container, furniture, electronic-device, structure, food, clothing, plant, scene, material, sport, misc.
- Output `imagenet_superclasses.json`:
  ```json
  {
    "superclasses": {
      "dog":    {"id": 0, "classes": [151, 152, ...]},   // 1000-space class indices
      "bird":   {"id": 1, "classes": [...]},
      ...
    },
    "class_to_superclass": {"151": 0, "152": 0, ...}
  }
  ```

### 3B. Two-stage classifier: `hierarchical_classifier.py`

- **Coarse classifier**: an interpretable classifier (configurable; **recommended default `nn.Linear(n_global_features, n_superclasses)`** — 20-way routing is easy and benefits from the strongest, exact-additive interpretability) trained on the *global* feature set (the existing Layer-1/Layer-2 features) with superclass labels.
- **Fine classifiers**: a dict `{superclass_id: classifier}`. Each is trained only on images of that superclass, using features discovered/selected for that superclass (see 3C). **Recommended default: HistGB + MI + `class_weight='balanced'`** (Section 6) — intra-superclass separation is the harder ~50-way problem and benefits from nonlinear interactions. Both stages are configurable via `coarse_classifier_type` / `fine_classifier_type`.
- **Soft-cascade inference** (robust to coarse errors):
  ```python
  coarse_logits = coarse_clf(global_feats)                # [B, S]
  coarse_prob   = softmax(coarse_logits, dim=1)           # [B, S]
  final_logits  = full of -inf, shape [B, 1000]
  for s in superclasses:
      fine_logits_s = fine_clf[s](feats_s)                # [B, n_s]
      for local_i, global_c in enumerate(classes_of[s]):
          final_logits[:, global_c] = log(coarse_prob[:, s] + 1e-9) + log_softmax(fine_logits_s)[:, local_i]
  pred = argmax(final_logits, dim=1)
  ```
  All operations are linear + softmax; remains fully interpretable and contains no MLP.

### 3C. Per-superclass search (ties into Sections 2 & 4)
- For each superclass `s`, run the GRPO Layer-1 search **using only images in `s`** (subset the dataset by `classes_of[s]`). Reward = fine-class accuracy *within* `s`. This densifies the reward signal (e.g. 130-way dog classification instead of 1000-way).
- Then run the Layer-2 enumeration (Section 4) on `s`'s selected Layer-1 bodies.
- Persist each superclass's discovered bodies to `layer_outputs/superclass_{id}_bodies.json`.

### 3D. Config
```yaml
hierarchical:
  enabled: true               # set false to fall back to flat 1000-way
  superclass_file: imagenet_superclasses.json
  coarse_only_warmup: true    # train coarse clf first, then fine clfs
  soft_cascade: true
```

### Acceptance criteria — Section 3
- [ ] `imagenet_superclasses.json` generated; every one of the 1000 classes maps to exactly one superclass; superclass sizes within [15, 130] (allow a "misc" overflow bucket if needed).
- [ ] Coarse classifier trains and reports superclass top-1 (expect 70–80%).
- [ ] Soft-cascade inference returns a valid 1000-way prediction for every image.
- [ ] `hierarchical.enabled: false` reproduces the original flat pipeline.
- [ ] No neurons / no MLP in any classifier path: grep for `ReLU`/multi-layer `Sequential` and confirm none. The classifier is one of the Section 6 allowed families (`nn.Linear`, EBM, or shallow GBDT within the interpretability budget).

---

## Section 4 — Feature-Map Layer-2 by Caching + Enumeration (top-30)

**Goal**: Build Layer-2 features that operate on Layer-1 **feature maps** (2D, pre-pooling), capturing cross-formula spatial interactions a linear classifier cannot. Make it fast via (a) caching Layer-1 maps and (b) **enumerating** a constrained Layer-2 form instead of RL. Keep **top-30** Layer-1 bodies.

### 4A. Select top-30 Layer-1 bodies
- From `l1_selected_bodies.json` (100 bodies), take the **top-30** by L1-classifier weight magnitude. Save to `layer1_top30.json`.
- Rationale: Layer-2 combination count is `C(k,2)`; k=30 → 435 pairs (vs 100 → 4950). Redundant near-duplicates among the 100 contribute little.

### 4B. Cache Layer-1 feature maps: `layer1_cache.py`
- For each of the 30 bodies, execute it on every training image but **stop before the final pooling** so the output is the 2D map `[H, W]`. (Reuse `TensorProgramEvaluator.execute_formula` but capture the pre-pool tensor; if the body already ends in a pooling token, evaluate the body minus its final root op.)
- **CIFAR-10**: cache at 16×16, FP32, on GPU. Size ≈ `30 × 50,000 × 16 × 16 × 4 B ≈ 1.5 GB`. Fits in GPU.
- **ImageNet**: cache at 28×28, **FP16 on CPU/disk** (`numpy.memmap` or pinned CPU tensor), cast to FP32 on read. Size ≈ `30 × 1.28M × 28 × 28 × 2 B ≈ 60 GB` (CPU/disk). For the **per-superclass** path (Section 3C) each subset is far smaller (e.g. dog ≈ 170k images → ≈ 8 GB FP16), so prefer caching **per superclass**.
- API:
  ```python
  class Layer1Cache:
      def build(self, bodies, dataset, resolution): ...   # populates cache
      def get(self, body_idx, image_indices) -> Tensor:   # returns [N, H, W] FP32
  ```

### 4C. Enumerate Layer-2 formulas: `layer2_enumerate.py`
Constrained form (keeps search tiny and bloat-free):
```
[ f_i  f_j  BinOp  {0–2 UnaryOps}  Pool ]
  f_i, f_j ∈ top-30 Layer-1 bodies, i < j   (dedupe symmetric ops)
  BinOp ∈ {subtract, multiply, fuzzy_and, fuzzy_or}     # informative, bounded
  UnaryOps ∈ subsets (size 0–2) of {abs, relu, sigmoid, normalize, blob_detector, contour}
  Pool ∈ ROOT_OPERATORS  (all pooling ops — now including the Section 1A.2 statistical pools:
                          pool_skewness, pool_entropy, pool_q90, … — so Layer-2 features can
                          capture the *distribution shape* of cross-formula interactions, not
                          just their mean)
```
- **Two-stage evaluation** (multi-fidelity):
  - **Stage A (coarse)**: evaluate every candidate on a subsample (CIFAR: 5k; ImageNet/superclass: 20/class) using cached maps → univariate accuracy. Keep top-2000.
  - **Stage B (precise)**: re-evaluate the top-2000 on the full set (or 100/class) → final univariate accuracy. Keep top-K (default `K = 500` per superclass, `K = 2000` for CIFAR-10 flat).
- **Dedup rules**: `multiply`/`fuzzy_and`/`fuzzy_or` are symmetric → only `i < j`. `subtract` is antisymmetric → keep both orders OR keep `i<j` and let the linear classifier's sign handle it (prefer the latter to halve the count). Drop `i == j`.
- Each surviving Layer-2 formula is stored as a full RPN string referencing the Layer-1 body indices, so it remains fully interpretable and re-executable from pixels.
- **No RL here.** This space is ~10^4; enumeration is faster and exhaustive. (RL/GRPO stays for Layer-1, whose space is ~10^25.)

### 4D. Integrate Layer-2 features
- Concatenate Layer-1 selected features + Layer-2 enumerated features, then feed the existing encoding stack (distribution statistics, power+L2 norm, optional Fisher Vector) and the linear (or hierarchical) classifier.

### Acceptance criteria — Section 4
- [ ] `layer1_top30.json` produced (exactly 30 bodies).
- [ ] `Layer1Cache` builds on CIFAR-10 in < 3 min, total GPU cache ≤ 2 GB at 16×16.
- [ ] `layer2_enumerate.py` runs the full CIFAR-10 enumeration (Stage A + B) in < 10 min and outputs top-2000 Layer-2 formulas with accuracies.
- [ ] Re-executing a saved Layer-2 formula from its RPN string reproduces (within 1e-4) the cached evaluation value (verifies traceability).
- [ ] **Ablation logged**: CIFAR-10 accuracy with Layer-1 only vs Layer-1 + Layer-2. (Decision gate: if Layer-2 adds < 1%, flag it in the report; if ≥ 2%, proceed to ImageNet.)
- [ ] ImageNet path uses per-superclass FP16 CPU/disk caching and the two-stage evaluator without OOM.

---

## Section 5 — Run Order, Reporting, and Decision Gates

Implement a top-level script `run_v3_3.py` (or extend `train_tensor_vsr_large_bank.py`) that executes:

1. **(Section 1)** Load extended operator registry (prior terminals 1A.0 + semantic + statistical + asymmetry/spatial + directional + fuzzy) and apply the one-formula-many-pools default (1A.6). Sanity-test all new operators and terminals.
2. **CIFAR-10 validation of the whole stack first** (cheap, fast, decisive):
   a. GRPO Layer-1 search (Section 2, classifier-dependent `λ_len`) **with Section 7 statistical gating** (wide admission + periodic reshuffle) → Layer-1 bodies.
   b. Select top-30 (Section 4A), cache maps (4B), enumerate Layer-2 (4C).
   c. Train the classifier(s) on Layer-1+Layer-2 features. **Run the Section 6 classifier comparison here** (linear / EBM / HistGB / large-GBDT reference). Log accuracy + mean formula length/depth over iterations (bloat check).
   d. **Run Section 8 interpretability tools** on the resulting features: probe images for the top formulas, formula→text translation, and (if applicable) occlusion test. Record the level-2/level-5 metrics.
   e. **Gate**: report CIFAR-10 (i) Layer-1 only, (ii) +Layer-2, (iii) +semantic operators, (iv) +statistical operators (1A.2 × HistGB synergy), (v) +prior terminals (1A.0) ablation, (vi) classifier comparison (Section 6), (vii) strict-vs-relaxed `λ_len` (Section 2E), (viii) admission-gate ablation (legacy `min_acc` vs Section 7 wide+reshuffle). Proceed to ImageNet only if the stack is healthy.
3. **(Section 3)** ImageNet: build WordNet superclasses; for each superclass run GRPO Layer-1 (with Section 7 gating) + Layer-2 enumeration on its image subset; train **coarse classifier (default: linear) + fine classifiers (default: HistGB + MI + balanced)**; soft-cascade evaluate. If Section 6's CIFAR-10 comparison shows a different best classifier, swap accordingly.
4. **Final report** `v3_3_report.md` with: per-section ablations, classifier comparison table, statistical-operator / prior-terminal / gating / `λ_len` ablations, total formula counts, bank rejection/pruning counts by category (Section 7), the five-level interpretability self-assessment + quantitative interpretability metrics (Section 8.4), CIFAR-10/100 and ImageNet top-1, mean formula length/depth (bloat metric), wall-clock per stage, and example discovered formulas with their plain-English reading and probe images.

### Global acceptance criteria
- [ ] End-to-end CIFAR-10 run reproduces ≥ current 76.84% with Layer-1 only + linear classifier (no regression from adding the new code paths), and reports the Layer-2 / semantic-operator / classifier deltas.
- [ ] All eight sections individually unit-tested as specced above.
- [ ] `ppo` and flat (`hierarchical.enabled: false`) paths still run — nothing is removed, only added.
- [ ] No constraint from the "Hard Constraints" list is violated (spot-check: classifier is neuron-free and within Section 6's interpretability budget, FP32 compute, no external pretrained models, fuzzy ops differentiable, grammar rules intact).

---

## Section 6 — Classifier: HistGB + MI + sample_weight (primary), with Linear/EBM interpretability reference

**Goal**: The constraint is "no neurons + interpretable," **not** "linear only." A collaborator's pipeline — **HistGradientBoostingClassifier (HistGB) + mutual-information feature selection (MI) + sample_weight**, with CV hyperparameter search — empirically **beat linear on a 10-class problem**. We adopt this as the **primary delivered classifier**, because HistGB is neuron-free (decision trees + additive boosting, no weights/activations/backprop) and stays interpretable at path level. We additionally keep **Linear** (and optionally **EBM**) as an **interpretability reference**, because they give *exact additive decomposition* — the strongest interpretability claim — and let us report the accuracy/interpretability trade-off with real numbers.

This section adds a **pluggable classifier interface** and a **comparison harness**, and specifies how to scale the collaborator's 10-class recipe to our 20-way (coarse) and ~50-way (fine) problems. It changes no features and no search — only what sits on top of the existing feature matrix. Run on CIFAR-10 first (features already exist), then carry the configuration into the ImageNet hierarchical pipeline (Section 3).

### 6.0 — CRITICAL: the multiclass tree-explosion problem (why HistGB *forces* superclass decomposition)

This is the single most important fact for scaling the collaborator's method from 10 to 1000 classes, and it is the reason Section 3's WordNet decomposition is mandatory, not optional, when using HistGB.

GBDT (including HistGB) trains multiclass via the one-vs-all / multinomial mechanism: **each boosting round fits one tree *per class*.** Therefore:

```
total trees = max_iter × n_classes

Collaborator (10 classes), max_iter=300:  300 × 10   =   3,000 trees   (manageable)
Flat ImageNet (1000 classes), max_iter=300: 300 × 1000 = 300,000 trees  (explodes:
                                              untrainable in practice, and totally
                                              uninterpretable — nobody reads 300k trees)
```

A flat 1000-way HistGB is therefore **both** computationally impractical **and** non-interpretable. The fix is structural, not a hyperparameter:

- **Coarse classifier**: one HistGB over ~20 superclasses → `max_iter × 20` trees.
- **Fine classifiers**: one HistGB per superclass over its ~50 classes, trained **only on that superclass's image subset**.
- **Inference**: coarse routes (soft-cascade, Section 3B) → for any single prediction you only inspect *one* ~20-way coarse model + *one* ~50-way fine model, never a 1000-way ensemble.

So when `classifier.type` is any GBDT variant, the pipeline **must** run with `hierarchical.enabled: true`. A flat GBDT over 1000 classes is allowed only as a never-delivered reference and only if it actually fits in memory/time (it usually will not — document the failure if so).

### 6A. Pluggable classifier interface: `classifiers.py`

```python
class BaseSymbolicClassifier:
    """Consumes a feature matrix X [N, D] (D symbolic features) and integer labels y [N].
    All backends must expose a per-prediction explanation and accept sample weights."""
    def fit(self, X_train, y_train, X_val, y_val, sample_weight=None): ...
    def predict(self, X) -> np.ndarray: ...           # class indices
    def predict_proba(self, X) -> np.ndarray: ...      # [N, n_classes]
    def explain(self, x_single) -> dict: ...           # per-prediction attribution (see 6D)
    def global_importance(self) -> np.ndarray: ...     # [D] global feature importance
    def interpretability_report(self) -> dict: ...     # size metrics (see 6C)
```

Backends:

1. **`HistGBClassifier` (PRIMARY / delivered)** — wraps `sklearn.ensemble.HistGradientBoostingClassifier`, preceded by **MI feature selection** and trained with **sample_weight / `class_weight='balanced'`**. Use HistGB specifically (not XGBoost/LightGBM) because it is in sklearn (no extra dependency), histogram-based (fast), and is exactly what the collaborator validated. `explain` returns the decision paths on the dominant trees rendered as readable conjunctions of formula thresholds (see 6D). Path-level interpretable.

2. **`LinearClassifier` (interpretability reference)** — wraps the existing `nn.Linear` + softmax. `explain` returns the per-class weight·feature contribution (exact additive decomposition). Interpretability gold standard; also the regression guard for the refactor.

3. **`EBMClassifier` (optional middle ground)** — wraps `interpret.glassbox.ExplainableBoostingClassifier` (InterpretML; pure Python, no neurons). Additive with learned per-feature shape functions + a few pairwise interactions → keeps exact additive decomposition while modeling nonlinearity. `max_bins=256`, `interactions=10`.

4. **`ReferenceGBDTClassifier` (black-box, reference-only)** — full-strength HistGB/LightGBM (e.g. `max_iter=500, max_depth=8`), **never delivered**, used only to measure the accuracy ceiling. Reported in a separate "upper-bound reference" row, never mixed into interpretable results.

> Dependency note: HistGB is in `sklearn` (already a dependency). `interpret` (EBM) is optional. None contain neural networks → all satisfy "no neurons." If an optional package is missing, skip that backend gracefully with a logged note; do not fail the harness.

> **Synergy with the statistical operators (Section 1A.2)**: HistGB splits on feature thresholds, so the high-order statistical features (skewness, kurtosis, entropy, quantiles) are unusually valuable to it — each gives a directly threshold-able, semantically meaningful signal (e.g. `if pool_entropy > τ → textured-surface class`). Under a linear classifier a single statistic is just one more weighted term; under HistGB it can anchor an entire decision rule. The Section 1A.2 operators and the HistGB classifier are therefore co-designed: expect a larger combined gain than either change alone, and verify this interaction explicitly in the Section 6F comparison (report HistGB accuracy with vs without the 1A.2 statistical features).

### 6B. Feature selection: MI vs L1, and the two-stage rule

The collaborator selects features with `SelectKBest(mutual_info_classif, k=K)`. Match the selector to the downstream classifier:

- **GBDT/EBM (nonlinear) → MI.** Mutual information captures *any* dependency (including nonlinear/non-monotonic), so it keeps features a tree can exploit even when their *linear* weight is ~0. Using L1 here would discard exactly those nonlinear-but-useful features (L1 judges features through a linear model). **[Evidence: fracture study]** — there the top single formulas had univariate *linear* accuracy of only 0.34–0.40, yet combined under HistGB they reached 95%. Selecting by linear accuracy would have under-ranked exactly the formulas that carry the non-linear signal; MI ranks them correctly. This is concrete proof that bank admission and feature selection must use MI, not linear accuracy, whenever the downstream classifier is non-linear.
- **Linear → L1.** Same family as the classifier; keeps linearly-separating features and auto-removes redundancy.

**Two-stage selection (recommended for our redundant formula bank).** MI's weakness is that it scores each feature independently and does **not** remove redundancy — given 100 near-duplicate formulas it keeps them all. Our bank is highly correlated (corr 0.6–0.9). So:

```
Stage 1 (de-dup): existing correlation gate (corr_threshold=0.92) + L1/Lasso pruning  → removes redundant formulas
Stage 2 (match classifier): SelectKBest(mutual_info_classif, k=K) for GBDT/EBM
                             (or L1 top-K for linear)
```

**MI at scale — subsample the estimate.** `mutual_info_classif` (kNN-based) is slow on 1.28M images × many features. Estimate MI on a **subsample** (50k–100k images); it is only used to *rank* features for top-K selection, so it is robust to subsampling. Under superclass decomposition each fine problem has ~50 classes, so a 50k subsample gives ~1000 images/class — a stable MI estimate (another reason the decomposition helps).

### 6C. Scaling the collaborator's parameters from 10 → 20 (coarse) and ~50 (fine) classes

Collaborator's original (10-class) grid:
```python
hgb_K_values = [100, 200, 300]
hgb_configs  = [(100, 0.1, 3), (200, 0.1, 5), (200, 0.05, 3), (300, 0.1, 5)]  # (max_iter, lr, max_depth)
```
**Do not reuse `max_iter` directly** — remember `total_trees = max_iter × n_classes`. Scaled, neuron-free, budget-aware configs:

**Coarse classifier (≈20-way superclass, full data):**
```python
coarse_K_values = [300, 500]                       # MI top-K; ~15-25 features/class
coarse_configs  = [(150, 0.1, 4), (150, 0.1, 5), (200, 0.05, 4)]   # (max_iter, lr, max_depth)
coarse_clf = HistGradientBoostingClassifier(
    max_iter=200, learning_rate=0.1, max_depth=4,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,  # auto-set tree count
    class_weight='balanced',        # superclasses are very imbalanced (dog 130 vs reptile 36)
    l2_regularization=1.0,          # regularize at scale
    random_state=42,
)
```

**Fine classifier (per superclass, ≈50-way, subset data):**
```python
fine_K_values = [200, 400]                         # MI top-K
fine_configs  = [(150, 0.1, 5), (200, 0.05, 5), (200, 0.1, 6)]
fine_clf = HistGradientBoostingClassifier(
    max_iter=200, learning_rate=0.05, max_depth=5,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
    class_weight='balanced',
    l2_regularization=1.0,
    random_state=42,
)
```

Rationale for the deltas from the 10-class recipe:
- **K grows with class count** (~10–30 features/class) but stays bounded because each sub-problem has few classes.
- **max_iter kept modest (100–200)** precisely because it multiplies by n_classes; do **not** scale it up to compensate for more classes.
- **max_depth slightly larger (4–6)** since more classes need finer boundaries; still shallow enough to read paths.
- **`early_stopping=True`** replaces grid-searching `max_iter` — let validation decide the tree count (usually fewer trees, less overfit, faster).
- **`class_weight='balanced'`** (equivalently `compute_sample_weight('balanced', y)`) — handles imbalance, mandatory for the coarse stage.

**CV cost control.** The collaborator's full grid × 5-fold CV is 100× costlier per fit at our scale. Search hyperparameters on a *subset* (e.g. 100 images/class, or on 2–3 representative superclasses), rely on `early_stopping` instead of searching `max_iter`, then retrain the chosen `(K, lr, max_depth)` on full data.

### 6D. Interpretability budget + per-prediction explanation

- **Delivered HistGB budget.** Keep `max_depth ≤ 6` and prefer `early_stopping` to cap tree count; report `total_trees = effective_iters × n_classes_in_node`. NOTE the collaborator's best configs (100–300 iters, depth 3–5) put 100–300 trees/class — readable per-tree (depth ≤5) but many trees, so single-prediction explanation is **path-level + feature-importance**, not exact additive. Record this honestly; only `reference_gbdt` may use depth 8 / 500 iters.
- **`explain(x_single)`** uniform output rendered to plain English (all formulas resolve to their RPN + plain-English reading, reusing the interpretability-slide mapping):
  - **HistGB**: conjunction of splits on the dominant trees, e.g. `"(中心红色横边强度 > 0.30) AND (顶部蓝色圆形物 ≤ 0.10) → airplane"`, plus how many trees voted and the margin; also report top-k `global_importance` features.
  - **Linear / EBM**: top-k features by signed additive contribution; sum = logit (exact).
- **`interpretability_report()`** returns: total split/parameter count, mean decision-path length (GBDT), whether exact additive decomposition holds (linear/EBM yes; HistGB no), and a one-line sample explanation.

### 6E. Config

```yaml
classifier:
  type: histgb                 # histgb (PRIMARY) | linear | ebm | reference_gbdt
  feature_selection: mi        # mi (for histgb/ebm) | l1 (for linear)
  mi_subsample: 50000          # images used to estimate MI ranking
  histgb:
    coarse: {max_iter: 200, learning_rate: 0.1,  max_depth: 4, K: 500}
    fine:   {max_iter: 200, learning_rate: 0.05, max_depth: 5, K: 400}
    early_stopping: true
    n_iter_no_change: 10
    l2_regularization: 1.0
    class_weight: balanced
  ebm_interactions: 10
hierarchical:
  enabled: true                # MUST be true for any GBDT type (see 6.0)
  coarse_classifier_type: linear   # keep coarse linear for strongest top-level interpretability...
  fine_classifier_type: histgb     # ...and use HistGB for the harder intra-superclass split
```

Recommended delivered architecture (the collaborator's method + our hierarchy): **coarse = linear** (20-way is easy; strongest, exact interpretability for the routing decision) **+ fine = HistGB+MI+balanced** (intra-superclass separation is harder and benefits from nonlinear interactions). The harness still evaluates all-linear and all-HistGB variants for comparison.

### 6F. Comparison harness: `compare_classifiers.py`

On a fixed feature matrix (CIFAR-10 Layer-1+Layer-2 features), identical splits, report:

| Classifier | Feature select | Test Acc | Interpretable? | Exact additive? | Notes |
|---|---|---|---|---|---|
| Linear | L1 | … | yes | yes | reference / regression guard |
| EBM | MI | … | yes | yes (+few interactions) | optional middle ground |
| **HistGB + MI + balanced (PRIMARY)** | MI | … | yes (path-level) | no | collaborator's method, scaled |
| Reference GBDT (500×8) | MI | … | **no — reference only** | no | accuracy ceiling, never delivered |

Also run, for HistGB specifically, a **budget sweep** reporting accuracy at the collaborator's settings (100–300 iters, depth 3–5) vs a tighter readable budget — so we can see how much accuracy depends on tree count.

Two questions the report must answer:
1. How much does HistGB+MI+balanced beat linear on our features? (the upside of the collaborator's method)
2. How much accuracy does the delivered (interpretable) model leave vs the black-box reference? (cost of staying interpretable)

### 6G. Wiring into the existing pipeline

- Refactor the current `nn.Linear(...)` + train loop to call `make_classifier(config.classifier.type)` → `BaseSymbolicClassifier`. The linear path must reproduce current CIFAR-10 results within ±0.1% (regression guard).
- Apply the existing **standardization + power-norm + L2-norm** before all backends identically (note: tree models are scale-invariant, so this neither helps nor hurts them — keeps the comparison controlled).
- `fit(..., sample_weight=...)` must thread sample weights through to `HistGradientBoostingClassifier.fit(clf__sample_weight=...)` exactly as in the collaborator's `Pipeline`.

### Acceptance criteria — Section 6
- [ ] `classifiers.py` exposes `HistGBClassifier` (primary), `LinearClassifier`, `EBMClassifier` (optional), `ReferenceGBDTClassifier` behind `BaseSymbolicClassifier`; `make_classifier(type)` factory works from config.
- [ ] `HistGBClassifier` pipeline = `SelectKBest(mutual_info_classif, k=K)` → `HistGradientBoostingClassifier(...)`, fit with `sample_weight`/`class_weight='balanced'`, MI estimated on `mi_subsample` images.
- [ ] Any GBDT type asserts `hierarchical.enabled == true` (enforces the 6.0 decomposition); a flat 1000-way GBDT is refused except as `reference_gbdt`.
- [ ] `LinearClassifier` reproduces current CIFAR-10 number within ±0.1% (regression guard).
- [ ] Coarse and fine HistGB use the scaled configs (6C) with `early_stopping` and `class_weight='balanced'`; total-tree count logged as `max_iter × n_classes`.
- [ ] `explain()` for every backend renders human-readable attributions referencing real formulas (RPN + plain-English).
- [ ] `compare_classifiers.py` outputs the table + the HistGB budget sweep + the two summary deltas on CIFAR-10.
- [ ] Reference GBDT segregated as black-box / upper-bound, never in "interpretable" claims.
- [ ] Missing optional dep (`interpret`) → backend skipped gracefully; HistGB (sklearn) and linear always run.

### Note for the advisor discussion (record in the report)
The delivered model must stay neuron-free **and** interpretable. **HistGB + MI + sample_weight is neuron-free** (trees + boosting, no neurons) and the collaborator showed it beats linear on 10 classes — we adopt it as primary, scaled to 1000 classes via WordNet superclass decomposition (without which multiclass GBDT explodes to `max_iter × 1000` trees, Section 6.0). Its interpretability is **path-level** ("prediction = a readable conjunction of formula thresholds," cf. ProtoTree) plus global feature importance — genuine, but weaker than the **exact additive decomposition** of Linear/EBM ("prediction = sum of per-formula contributions"). We therefore deliver HistGB for accuracy while reporting Linear/EBM as the exact-interpretability reference, letting the advisor pick the operating point on the accuracy/interpretability trade-off with real numbers. Recommended default: **coarse = linear (exact, easy 20-way) + fine = HistGB (harder ~50-way)**.

---

## Section 7 — Statistical Gating for Bank Admission and Periodic Reshuffle

**Goal**: Use statistics to control which formulas enter the bank and which get pruned — but split the responsibility correctly. The current admission rule is a single test (univariate accuracy > `min_acc=0.002`), which has three blind spots: (1) it admits **degenerate** formulas (near-constant output, NaN/inf, saturated range) that are just noise; (2) it misses **non-linear redundancy** (formulas with low linear correlation but near-identical distributions); (3) it judges discriminative power by **linear** separability, which mis-fits the HistGB endgame (a formula can have low linear accuracy yet strong non-linear MI that HistGB could exploit).

**Core principle — wide admission, strict reshuffle.** Do **not** make admission strict. Strict admission would kill diversity, starve the early bank (when the policy is still weak and most formulas look mediocre), and fight the Pareto/parsimony pressure (short formulas naturally have lower variance/MI). Instead:

- **Admission gate = degeneracy rejection only** (cheap, per-formula, generous). Reject only formulas that are statistically *broken*, never formulas that are merely *weak*. Discriminative-power judgment is deliberately **not** done at admission.
- **Periodic reshuffle = group-level discriminative + redundancy pruning** (the user's earlier "宽进 + 定期大洗牌 Lasso" idea, now generalized). Every `reshuffle_interval` iterations, evaluate the *whole* bank together and prune what is genuinely useless or redundant. Group evaluation is far more accurate than per-formula admission judgments, because usefulness is contextual (a weak-alone formula can be complementary).

This section unifies three distinct roles statistics play in the pipeline; keep them conceptually separate in the code:

| Role | Where | Statistics used |
|---|---|---|
| **As features** | formula pooling ops (Section 1A.2) | skewness, kurtosis, entropy, quantiles … |
| **As admission gate** | bank entry (7A) | variance, finite-ratio, dynamic-range (degeneracy only) |
| **As reshuffle criterion** | periodic group pruning (7B) | MI (HistGB-matched), Lasso/HistGB importance, Wasserstein redundancy |

### 7A. Admission gate (degeneracy rejection — wide)

New module `bank_admission.py`. Called when a formula completes, **before** it is added to the bank. Operates on the formula's output vector `v = feature_values [N]` over the current evaluation batch. **Reject only on statistical degeneracy**, with generous thresholds:

```python
def admission_gate(v, cfg):
    # 1. finite-ratio: reject formulas producing many NaN/inf
    finite = torch.isfinite(v).float().mean()
    if finite < cfg.finite_min:            # e.g. 0.95
        return False, "nonfinite"
    v = v[torch.isfinite(v)]

    # 2. variance floor: reject near-constant (no information) outputs
    if v.var() < cfg.var_min:              # e.g. 1e-5  (generous — only kills true constants)
        return False, "degenerate_constant"

    # 3. dynamic-range floor: reject saturated/clamped outputs (collapsed by ±60000 clamp etc.)
    iqr = torch.quantile(v, 0.75) - torch.quantile(v, 0.25)
    if iqr < cfg.iqr_min:                  # e.g. 1e-4
        return False, "saturated"

    return True, "admit"
```

**Deliberately NOT in the admission gate**: any accuracy/MI threshold on discriminative power. Admission keeps the existing very-low `min_acc=0.002` as a floor at most (or drops it entirely in favor of degeneracy-only gating — A/B both in the CIFAR ablation). The point is to stay **wide**: a formula that is statistically healthy but weak still gets in, because its value may only appear in combination.

### 7B. Periodic reshuffle (group-level pruning — strict)

New module `bank_reshuffle.py`. Every `reshuffle_interval` GRPO iterations (default 100), evaluate the entire current bank jointly and prune:

```python
def reshuffle(bank, X_bank, y, cfg):
    # X_bank: [N_images, n_formulas]  (cached feature values for all bank formulas)

    # Step 1 — MI ranking (HistGB-matched discriminative power).
    #   Estimate MI on a subsample (Section 6B) to keep it cheap; MI captures
    #   non-linear dependence, so it does NOT mis-kill formulas HistGB could use.
    mi = mutual_info_per_feature(X_bank, y, subsample=cfg.mi_subsample)
    keep_mi = mi > cfg.mi_floor             # drop genuinely uninformative formulas

    # Step 2 — Lasso / HistGB-importance pruning (the user's "Lasso reshuffle").
    #   Fit an L1-regularized linear model (or read HistGB feature_importances_)
    #   on the whole bank; formulas with zero/near-zero importance are dropped.
    importance = lasso_or_histgb_importance(X_bank[:, keep_mi], y)
    keep_imp = importance > cfg.imp_floor

    # Step 3 — distribution-redundancy pruning (non-linear dedup).
    #   Existing corr>0.92 gate catches linear redundancy; add a 1-D Wasserstein
    #   check ONLY among medium-correlation pairs (0.7–0.92) to catch
    #   "low linear correlation but near-identical distribution" duplicates.
    survivors = wasserstein_dedup(X_bank[:, keep_imp],
                                  corr_band=(0.70, 0.92),
                                  w_min=cfg.wasserstein_min)
    return survivors
```

Notes:
- **Wasserstein only on the 0.70–0.92 correlation band** — below 0.70 the formulas are clearly different (skip, save compute); above 0.92 the existing linear-correlation gate already removed them.
- Reshuffle uses **MI**, not linear accuracy, for discriminative pruning — consistent with the HistGB endgame (Section 6B's "match the selector to the classifier").
- The reshuffle is where strictness lives; admission stays wide.

### 7C. Guardrails against over-pruning (do not let the gates kill diversity)

Implement and log these safety checks; they encode the "wide admission, careful reshuffle" principle:

- **Floor on bank size**: never let reshuffle drop the bank below `min_bank_size` (e.g. 2000). If pruning would go below, keep the top-`min_bank_size` by MI rather than applying hard floors.
- **Diversity quota**: when pruning redundant formulas, preserve at least one representative per correlation cluster (do not collapse a whole cluster to nothing).
- **Pareto consistency**: the admission gate must not reject a formula merely for being short/low-variance if it is on the accuracy/length Pareto front (Section 2) — degeneracy rejection (constant/NaN) is fine, but a short, healthy, weak-alone formula must survive admission.
- **Early-training leniency**: for the first `warmup_iters` iterations, apply admission only (no reshuffle pruning), so the early bank can fill up while the policy is still weak.
- **Log every rejection reason** (nonfinite / degenerate_constant / saturated / low_mi / low_importance / wasserstein_redundant) and report counts per category in `v3_3_report.md`, so over-aggressive gating is visible.

### 7D. Config

```yaml
bank_admission:
  finite_min: 0.95
  var_min: 1.0e-5            # generous: kills only true constants
  iqr_min: 1.0e-4
  keep_min_acc: 0.002        # optional legacy floor; set null to use degeneracy-only gating
bank_reshuffle:
  enabled: true
  reshuffle_interval: 100    # GRPO iterations between reshuffles
  mi_floor: 0.005            # drop formulas with near-zero MI to labels
  mi_subsample: 50000
  imp_floor: 1.0e-4          # Lasso/HistGB importance threshold
  wasserstein_min: 0.05      # distribution-redundancy threshold (0.70–0.92 corr band)
  min_bank_size: 2000        # never prune below this
  warmup_iters: 500          # admission-only before this many iterations
```

### Acceptance criteria — Section 7
- [ ] `bank_admission.py` rejects only degenerate formulas (constant / non-finite / saturated); a healthy-but-weak formula is admitted.
- [ ] `bank_reshuffle.py` runs every `reshuffle_interval` iters; prunes by MI floor → importance floor → Wasserstein dedup (0.70–0.92 band only).
- [ ] Reshuffle never drops bank below `min_bank_size`; diversity quota keeps ≥1 representative per correlation cluster.
- [ ] No reshuffle pruning before `warmup_iters`.
- [ ] All rejection/pruning reasons logged with per-category counts in the final report.
- [ ] Ablation: bank size and final accuracy with (i) legacy single `min_acc` gate vs (ii) Section 7 wide-admission + reshuffle — report both so the change is justified by data, not assumption.
- [ ] MI in reshuffle uses subsampled estimate (Section 6B); confirm it is non-linear MI, not linear accuracy.

---

## Section 8 — Interpretability: Definition, Measurement, and Trust-Verification — **[Evidence: fracture study]**

**Goal**: Stop treating "interpretability" as a vague binary claim and make it a *defined, measured, enhanced* property. The fracture study made the core lesson concrete: a top formula like `I_r local_range I_BONE multiply I_NEG multiply lbp_like flip_h I_NEG multiply normalize` is **fully traceable but not human-readable** — and crucially, *reading the formula could not tell us whether the 95% was real signal or a shortcut*. What settled it was **same-source data provenance** (and, ideally, an occlusion test) — not formula readability. This reframes where our interpretability value actually lives.

### 8.0 The traceable-vs-understandable distinction (state this explicitly in the paper)

| | Traceable | Understandable |
|---|---|---|
| Meaning | every computation is deterministic and reproducible | a human can see *what it measures and why it helps* |
| Our status | ✅ 100% (pixels → prediction, all explicit math) | ⚠️ partial (short formulas yes; long ones no) |
| Neural nets | ❌ | ❌ |

We have **100% traceability**; **understandability is partial**. The selling point must be built on what we actually have, not on "every formula is readable" (which a single 17-token formula refutes).

### 8.1 Five-level interpretability spectrum (locate ourselves honestly)

1. **Mechanistic transparency** — no black box; every value from explicit math. **Us: ~100%** (our true strength).
2. **Per-unit semantics** — each formula maps to a human-readable visual concept. **Us: ~50%** (short formulas only; the long-formula gap).
3. **Decision attribution** — for a given prediction, quantify which formulas contributed. **Us: ~85%** (linear exact; HistGB path-level + importance).
4. **Concept alignment** — features align with expert-recognized concepts. **Us: ~15%** (we deliberately use no external concept knowledge).
5. **Causal verifiability** — verify the model uses causally-relevant features, not dataset shortcuts. **Us: strong-but-unsystematic; this section makes it systematic.**

**Recalibrated selling point**: build the claim on **levels 1 + 3 + 5** (mechanistic transparency + decision attribution + causal verifiability), where we genuinely dominate black boxes — *not* on level 2 (which long formulas break) or level 4 (which we forgo by design). State level 2/4 limits honestly; reviewers respect a precise interpretability claim far more than a hand-wavy "we are interpretable."

### 8.2 Trust-verification tools (the real value — black boxes cannot do these) — implement these

This is the heart of Section 8: the fracture case proved interpretability's payoff is **verifying whether high accuracy is trustworthy**, which our deterministic features make possible and a CNN does not.

- **8.2a Occlusion test** (`interpretability/occlusion_test.py`): mask the task-relevant region (e.g. via a deterministic foreground mask — for the fracture case, the existing `threshold_bone` operator generates a bone mask) and re-evaluate. If accuracy holds, the signal is in the object; if it collapses, the model was using background/shortcut. Report accuracy-vs-occlusion curve.
- **8.2b Cross-source test** (`interpretability/cross_source_test.py`): if data has source/site/device metadata, train on one source and test on another. A large drop reveals source-shortcut dependence. (For the fracture study this was implicitly passed via same-source data; make it an explicit, optional harness for any dataset with source labels.)
- **8.2c Probe images** (`interpretability/probe_images.py`): for any formula — *especially* an unreadable long one — render the top-k and bottom-k activating images from the training set. A human instantly sees what the formula responds to ("ah, it fires on textured bone-cortex breaks"), recovering level-2 understanding for formulas whose token-chain is opaque. This is the single most effective tool for the long-formula problem.
- **8.2d Deterministic formula→text translation** (`interpretability/formula_to_text.py`): a rule-based (NOT LLM, no external knowledge) translator that maps an RPN token chain to a structured phrase, e.g. `I_R edge_x blur pool_center → "smoothed horizontal-edge strength of the red channel, central region"`. For long formulas it produces nested phrasing — verbose but consistent and traceable. Reuse the channel/operator names already used in the interpretability slides.

### 8.3 Enhancing per-unit semantics (level 2) — link to existing mechanisms

- **Pareto length penalty (Section 2E) IS an interpretability mechanism**, not just anti-overfit: shorter formulas are more readable. State this linkage explicitly. (But respect 2E's classifier-dependent caveat: under HistGB we relax length to keep useful non-linear structure — so probe images (8.2c), not shortness alone, carry level-2 understanding there.)
- **DAG naming / sub-expression abstraction (future, DreamCoder-style)**: when a sub-chain (e.g. `edge_x blur`) recurs across high-importance formulas, abstract it into a named composite (`smooth_h_edge`), so long formulas read as compositions of a few named mid-level concepts. Note as future work; not required for v3.3.

### 8.4 Quantitative interpretability metrics (report these — make interpretability measurable)

In `v3_3_report.md`, report interpretability as numbers, not adjectives:
- **mean formula length / depth** (level-2 proxy; lower = more readable);
- **fraction of delivered features with length ≤ L** (e.g. ≤ 6 tokens) — the "readable fraction";
- **probe-image consistency**: do the top-k activators of a formula share an obvious visual theme (can be a quick human rating on a sample);
- **occlusion robustness**: accuracy retained under object-region-only evaluation (level-5 metric);
- For HistGB: mean decision-path length and number of formulas covering 80% of total importance.

### 8.5 Framing the fracture result in the paper (decided by verification, not readability)

Write the 72%→95% result as a **dual win**: (i) it demonstrates symbolic features encode genuine *non-linear* discriminative signal that a neuron-free classifier unlocks; (ii) it demonstrates that our method's interpretability lets us *verify* that 95% is trustworthy (same-source provenance + occlusion test) — a check a black-box CNN at the same accuracy cannot provide. The headline is not "95%"; it is "95% **that we can audit**."

### Acceptance criteria — Section 8
- [ ] `occlusion_test.py`, `cross_source_test.py`, `probe_images.py`, `formula_to_text.py` implemented; each runs on CIFAR-10 and (where applicable) the ImageNet/superclass pipeline.
- [ ] Probe images render top-k/bottom-k activators for an arbitrary formula (including a long, unreadable one) — verify on at least one 12+-token formula.
- [ ] `formula_to_text.py` is purely rule-based (no LLM, no external data); every channel/operator token has a phrase mapping; nested phrasing for compound formulas.
- [ ] The five-level self-assessment and the recalibrated (levels 1+3+5) selling point are written into `v3_3_report.md`.
- [ ] Quantitative interpretability metrics (8.4) computed and reported.
- [ ] No external pretrained model used anywhere in Section 8 (Hard Constraint #5 preserved).

---

## Suggested Implementation Order (for Claude Code)

1. **Section 1** (operators: prior terminals + semantic + statistical + asymmetry/spatial + fuzzy + multi-pool default) — smallest, unblocks everything, immediately testable.
2. **Section 4** (cache + Layer-2 enumeration) on **CIFAR-10** — validates the feature-map idea cheaply; depends only on existing Layer-1 bodies in `l1_selected_bodies.json`.
3. **Section 6** (HistGB primary + classifier comparison) on **CIFAR-10** — cheap, runs on existing features; validates HistGB+MI+balanced beats linear at our feature scale (and the 1A.2-statistics × HistGB synergy), exercises the budget sweep before committing the ImageNet fine-stage choice.
4. **Section 7** (statistical gating + reshuffle) — wire into the search loop; cheap to validate on CIFAR-10 with the ablation (legacy gate vs wide-admission+reshuffle).
5. **Section 2** (GRPO + Pareto, classifier-dependent length penalty) — swap-in trainer; validate bloat control + the strict-vs-relaxed `λ_len` ablation on CIFAR-10.
6. **Section 8** (interpretability tools) — probe images / occlusion / formula-to-text run on existing CIFAR-10 features; needed to frame results and verify trust before scaling.
7. **Section 3** (WordNet hierarchy) — mandatory for ImageNet scale when using HistGB (see 6.0); uses the winning classifier from Section 6 for the fine stage (default: HistGB).
8. **Section 5** (orchestration + report).

Keep every change behind a config flag so the original PPO / flat / Layer-1-only pipeline remains runnable for comparison. Commit each section separately with its unit tests.
