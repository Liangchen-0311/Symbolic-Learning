# ImageNet First Try — Experiment Report

> **Date**: 2026-03-25 ~ 2026-03-26
>
> **Goal**: Run the symbolic feature discovery pipeline end-to-end on ImageNet-1K for the first time. Assess feasibility and get a baseline accuracy number.
>
> **Method**: PPO discovers symbolic (math) formulas → each formula extracts a scalar feature from an image → linear classifier (nn.Linear) maps features to 1000 classes.

---

## 1. Setup

- **Hardware**: 1× NVIDIA A100-SXM4-80GB, RunPod instance (~125 GB disk quota)
- **Dataset**: ImageNet-1K downloaded via HuggingFace streaming
  - Train: 1,180,660 images across 1000 classes (92% of full 1.28M — some images skipped during streaming download, 1 corrupted JPEG repaired)
  - Val: 50,000 images (complete)
- **Operator vocabulary**: 55 tokens (27 non-root operators + 19 root operators + 6 terminals + 3 special)
- **Config**: Single bank, max_depth=7, max_sequence_length=18, loss-based reward, hierarchical eval (20 superclasses)

---

## 2. Phase 1: Formula Discovery (RL)

**Config**:
- Resolution: 64×64 (downscaled for speed)
- Training data: 20,000 images (20 per class, stratified)
- Bank capacity: 12,000
- PPO: 150 episodes/iteration, entropy schedule 0.05→0.005, LR warmup 100 iters
- Reward: composite loss-based (0.6 × normalized_loss + 0.3 × top5 + 0.1 × top1)
- Hierarchical eval: 20 superclasses during early training

**Observations**:

| Iteration | Bank Size | Avg Reward | Notes |
|---|---|---|---|
| 10 | 49 | -0.607 | Mostly invalid formulas, bank filling slowly |
| 20 | 116 | -0.582 | Starting to find valid formulas |
| 30 | 281 | -0.532 | Rapid discovery phase begins |
| 40 | 529 | -0.495 | ~250 new formulas per 10 iters |
| 50 | 999 | -0.337 | Reward improving, bank growing fast |
| 60 | 1,983 | -0.131 | Peak discovery rate: ~1000 formulas in 10 iters |
| 70 | 3,241 | -0.098 | Discovery rate slowing |
| **73** | **3,614** | — | **Stopped: reached target of 3,500** |

**Key finding**: Formula discovery hit a wall around iteration 60–70. After the bank reached ~3,000 formulas, the correlation threshold (0.85) started rejecting most new formulas as too similar to existing ones. In the earlier full run (10,000 iterations, killed after many hours), the bank plateaued at 3,842 formulas — only 228 more formulas discovered in thousands of additional iterations beyond iteration 73.

**Phase 1 time**: 14.4 minutes (73 iterations × 150 episodes)

---

## 3. Phase 2: Full-Resolution Validation & Deduplication

**Process**:
1. Loaded all 3,614 Phase 1 formulas
2. Re-evaluated each formula at 224×224 on 50,000 images (50 per class)
3. Accuracy drop filter: **disabled** — Phase 1 used hierarchical eval (20 superclasses) while Phase 2 evaluated on full 1000 classes, making accuracy values incomparable
4. Cross-formula deduplication: Pearson correlation threshold 0.78, greedy (keep highest-accuracy formula, reject any new formula with |r| ≥ 0.78 against any kept formula)

**Results**:
- Input: 3,614 formulas
- After accuracy filter: 3,614 (all passed — filter disabled due to hierarchical/full-class incomparability)
- After deduplication: **2,611 unique formulas** (1,003 removed as too correlated)

**Phase 2 time**: ~40 minutes (dominated by per-formula evaluation at 224×224)

---

## 4. Phase 3: Feature Extraction & Classification

### 4.1 Feature Extraction

**Disk quota constraint**: The RunPod instance has ~125 GB disk quota. ImageNet data occupies 111 GB, leaving only ~14 GB for feature matrices. Full train extraction (1.18M × 2,611 × 4 bytes = 12.3 GB) would exhaust the quota. Solution: **use a train subset of 200,000 images (200 per class)** for classifier training.

| Split | Images | Formulas | Matrix Size | Time |
|---|---|---|---|---|
| Train (subset) | 200,000 | 2,611 | 200K × 2,611 (~2 GB) | 49.9 min |
| Val (full) | 50,000 | 2,611 | 50K × 2,611 (~500 MB) | 12.6 min |

Features written to memory-mapped files with checkpoint/resume support (progress saved every 10 batches — crash-safe).

### 4.2 Linear Classifier Training

**Method**: PyTorch `nn.Linear(2611, 1000)` + AdamW + cosine LR schedule, 20 epochs, batch_size=1024. Features standardized online (zero-mean, unit-variance).

**Weight decay sweep**:

| weight_decay | Train Top-1 | Val Top-1 | Val Top-5 | Time |
|---|---|---|---|---|
| 0.0001 | 13.26% | 9.22% | 21.64% | 20.1s |
| **0.001** | **13.28%** | **9.24%** | **21.70%** | **20.0s** |
| 0.01 | 13.25% | 9.14% | 21.59% | 20.0s |
| 0.1 | 13.04% | 9.09% | 21.52% | 19.7s |

**Best**: weight_decay=0.001, **Top-1 = 9.24%, Top-5 = 21.70%**

### 4.3 Baselines

| Method | Top-1 | Top-5 | Notes |
|---|---|---|---|
| **Our method (2,611 formulas)** | **9.24%** | **21.70%** | Symbolic features + linear classifier |
| PCA (100 components of our features) | 8.68% | 20.67% | Dimensionality reduction of our features |
| Pixel channel means (6 features) | 0.13% | 0.63% | Just R/G/B/Gray/H/S global averages |
| Random features (1000 dims) | 0.10% | 0.51% | Random Gaussian features |
| Random guessing | 0.10% | 0.50% | 1/1000 and 5/1000 |

### 4.4 Per-Superclass Accuracy

| Superclass | Top-1 | Notes |
|---|---|---|
| fish_aquatic | 11.96% | Best — distinctive colors/textures |
| mammal_wild | 10.72% | |
| primate | 10.28% | |
| electronic_device | 10.12% | |
| musical_instrument | 10.08% | |
| mammal_pet | 9.84% | |
| food_fruit | 9.76% | |
| natural_scene_misc | 9.68% | |
| tool_implement | 9.60% | |
| food_other | 9.52% | |
| container_vessel | 9.32% | |
| plant_flower | 9.20% | |
| reptile_amphibian | 9.08% | |
| clothing_fabric | 8.64% | |
| furniture_indoor | 8.60% | |
| insect_arthropod | 8.48% | |
| bird | 8.28% | |
| vehicle_land | 7.60% | |
| structure_building | 7.20% | |
| vehicle_water_air | 6.84% | Worst — lacks spatial structure features |

**Pattern**: Animal/natural classes (fish, mammals, primates) outperform man-made object classes (vehicles, buildings). This makes sense — our symbolic operators (color channels, edges, textures, Gabor filters) capture biological visual properties better than architectural/structural features.

---

## 5. Issues Encountered

### 5.1 Disk Quota (125 GB limit)
RunPod's per-user storage quota (~125 GB) was the primary constraint. ImageNet data (111 GB) + HF cache (18 GB) + feature matrices nearly exhausted it. Mitigations:
- Deleted HF download cache after dataset extraction
- Used train subset (200K instead of 1.18M) for classifier training
- Implemented memory-mapped feature extraction to avoid RAM pressure

### 5.2 Corrupted JPEG
One image (`class_0316/01180659.JPEG`) was corrupted during streaming download, crashing the DataLoader. Fixed by:
- Adding `_SafeImageFolder` class that returns a black image on load failure
- Re-downloading the corrupted file from HuggingFace

### 5.3 GPU OOM in Phase 3
Phase 2 cached 50K × 224×224 images on GPU (~78 GB). When Phase 3 started in the same process, GPU memory was exhausted. Fixed by running Phase 3 as a separate process.

### 5.4 Formula Discovery Plateau
Bank growth stalled at ~3,500 formulas (iteration ~70). The correlation threshold (0.85) increasingly rejected new formulas as redundant. With 10,000 iterations configured, the extra ~9,900 iterations would have added fewer than 300 formulas — diminishing returns.

### 5.5 Phase 2 Accuracy Filter Mismatch
Phase 1 used hierarchical eval (20 superclasses, baseline ~5%) while Phase 2 evaluated on full 1000 classes (baseline ~0.1%). The "30% relative accuracy drop" filter rejected 100% of formulas because the two accuracy scales are incomparable. Fixed by disabling this filter when hierarchical eval was used in Phase 1.

---

## 6. Analysis

### What worked
- **Pipeline runs end-to-end**: From RL formula discovery through to final ImageNet accuracy, the entire 3-phase pipeline works
- **92× random baseline**: 9.24% vs 0.1% shows symbolic formulas capture real visual information
- **Better than PCA**: Our 2,611 RL-discovered formulas (9.24%) outperform the top 100 principal components of the same features (8.68%), suggesting RL finds complementary features that PCA misses
- **Fast Phase 1**: 3,500+ formulas discovered in just 14 minutes on a single A100

### What limits accuracy
1. **Formula plateau at ~3,500**: The single-bank correlation threshold creates a diversity bottleneck. Multi-bank training (4 banks with different configs) should yield 10K–15K formulas
2. **Train subset (200K vs 1.18M)**: Disk quota forced us to use only 17% of training data. Full train data would improve the linear classifier
3. **Single bank, single config**: Only one max_depth/entropy schedule was explored. Multi-bank strategy with different depths (6/7/8) and focus areas (spatial, texture, channel) would discover more diverse formulas
4. **Linear classifier only**: nn.Linear with 2,611 features on 1000 classes is severely underfitting (train accuracy only 13.28%). More features would help

### Next steps (priority order)
1. **Multi-bank training**: 4 banks → merge → deduplicate → expect 8K–12K formulas
2. **More disk space or external storage**: Use full 1.18M train images for classifier
3. **Longer Phase 1**: Run each bank for 500+ iterations instead of 73
4. **Feature selection**: L1 regularization to identify the most discriminative formulas
5. **Non-linear classifier experiment**: 1-hidden-layer MLP as an upper-bound comparison (sacrifices full interpretability)

---

## 7. Formula Quality Analysis

### 7.1 Overall Statistics

| Metric | Value |
|---|---|
| Total formulas (after dedup) | 2,611 |
| Individual formula accuracy range | 0.02% – 0.33% |
| Mean individual accuracy | 0.12% |
| Combined accuracy (linear classifier) | 9.24% |

Individual formula accuracy is very low (best single formula: 0.33% top-1) — this is expected for 1000-class classification. The power comes from **linear combination**: 2,611 weak features combined by the linear classifier yield 9.24%, a 28× amplification over the best single formula.

### 7.2 Formula Length Distribution

| Length | Count | Percentage |
|---|---|---|
| 2–3 (simple) | 14 | 0.5% |
| 4–6 (medium) | 10 | 0.4% |
| 7–17 (complex) | 24 | 0.9% |
| **18 (max length)** | **2,563** | **98.2%** |

**98% of formulas hit the max sequence length (18 tokens).** This is a strong signal that the RL agent learned to maximize formula complexity — longer formulas tend to extract more specific features that survive the correlation-based dedup. The few short formulas that made it into the bank capture basic global statistics that are uncorrelated with complex formulas.

### 7.3 Operator Usage

Most frequently used operators (appearing in % of all formulas):

| Operator | Usage | Role |
|---|---|---|
| `subtract` | 105% | Cross-channel differences (e.g., I_R - I_G for color contrast) |
| `multiply` | 100% | Feature interactions |
| `local_std_5x5` | 90% | Texture roughness — the most popular spatial operator |
| `laplacian` | 76% | Edge/blob detection |
| `sigmoid` | 71% | Bounding values to [0,1] after complex operations |
| `log1p_abs` | 65% | Dynamic range compression |
| `flip_h` | 65% | Symmetry detection (new operator) |
| `relu` | 59% | Thresholding |
| `flip_v` | 54% | Vertical symmetry (new operator) |
| `negate` | 54% | Feature inversion (new operator) |
| `gabor_45` | 54% | Diagonal texture detection |
| `edge_y` | 51% | Vertical edges |

**Key observations**:
- `subtract` and `multiply` appear in virtually every formula — cross-channel arithmetic is the backbone of discriminative features
- `local_std_5x5` is the dominant spatial operator (90%) — texture variation is extremely informative for ImageNet classes
- The **new operators** (`flip_h`, `flip_v`, `negate`, `log1p_abs`, `pow2`, `sqrt_abs`) are heavily used (34–65%), confirming they were necessary additions
- `downsample_2x/4x` appear less frequently (~15%) but enable critical multi-scale features

### 7.4 Terminal (Input Channel) Usage

| Terminal | Usage | Notes |
|---|---|---|
| `I_GRAY` | 83% | Grayscale — structure/texture backbone |
| `I_H` | 76% | Hue — heavily used for color-based discrimination |
| `I_G` | 68% | Green channel |
| `I_B` | 46% | Blue channel |
| `I_R` | 40% | Red channel |
| `I_S` | 23% | Saturation — least used |

**HSV channels (I_H, I_S) are validated**: Hue is the 2nd most used terminal (76%), confirming that color-based discrimination is critical for ImageNet fine-grained classes. The newly added HSV terminals were not wasted.

### 7.5 Example Formulas (Interpreted)

**Simple formulas** (basic statistics that provide global context):
- `I_H pool_center` — average Hue of the center region (object color)
- `I_GRAY global_std_pool` — overall contrast of the image
- `I_R pool_thirds_mid` — red intensity in the middle strip (body of object)

**Medium formulas** (one or two transformations):
- `I_G edge_x pool_quad_bl` — horizontal edges of green channel in bottom-left (ground texture)
- `I_S gabor_90 abs pool_quad_tl` — vertical texture energy of saturation in top-left

**Complex formulas** (deep multi-step pipelines — 98% of all formulas):
- `I_GRAY I_G multiply gabor_45 negate local_std_5x5 gabor_90 abs flip_v sigmoid edge_y blur_7x7 normalize sigmoid edge_y relu relu pool_right_half`
  → Luminance × Green → diagonal texture → invert → texture variation → vertical texture → flip → threshold → vertical edges → blur → normalize → sigmoid → edges → relu → right half average.
  This detects **right-side vertical edge texture patterns** modulated by luminance-green interaction.

- `I_H gabor_45 I_H multiply abs I_H I_G multiply subtract I_GRAY dilate abs normalize subtract gabor_90 log1p_abs edge_y global_l2_pool`
  → Multiple hue-based texture features subtracted against grayscale morphological features → compressed → vertical edges → L2 norm.
  This computes a **color-texture contrast feature** combining hue patterns with structural edges.

### 7.6 Quality Assessment

**Do the formulas actually help?** Yes — three lines of evidence:

1. **vs. random features**: 9.24% vs 0.10%. Random features are useless; our formulas carry real visual information.
2. **vs. pixel means only**: 9.24% vs 0.13%. The 6 channel averages alone are nearly worthless. The spatial/texture/cross-channel operations in our formulas are doing the heavy lifting.
3. **vs. PCA of our features**: 9.24% vs 8.68%. Even after PCA reduces our 2,611 features to the best 100 linear combinations, we still beat it — suggesting our features contain complementary non-redundant information.

**Concerns**:
- Individual formula accuracy (0.02–0.33%) is very low. Each formula is an extremely weak classifier — the system relies entirely on combining thousands of them.
- 98% of formulas are max-length (18 tokens). The RL agent may be padding formulas with redundant operations (e.g., `relu relu`, `sigmoid sigmoid`) to reach max length rather than discovering genuinely complex features.
- The operator composition is often hard to interpret semantically despite being fully symbolic — a 18-token RPN formula is technically interpretable but not easily human-readable.

---

## 8. Reproducibility

```bash
# Phase 1 (formula discovery) + Phase 2 (validation & dedup)
python experiments/run_phase2_3.py

# Phase 3 only (from existing Phase 2 output)
python experiments/run_phase3_only.py

# Config
configs/tensor_vsr_imagenet_single_bank.yaml

# Output
outputs/imagenet_single_bank/
├── phase1/bank_0/feature_bank/    # 3,614 discovered formulas
├── phase2/feature_bank/           # 2,611 deduplicated formulas
└── phase3/
    ├── X_train.mmap               # Train feature matrix (200K × 2,611)
    ├── X_val.mmap                 # Val feature matrix (50K × 2,611)
    └── final_results.json         # All metrics
```
