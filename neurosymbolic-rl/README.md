# Neurosymbolic Visual Feature Learning

Discovering interpretable mathematical formulas for image classification via reinforcement learning. Instead of learning opaque neural network weights, this system uses RL to search over a symbolic grammar of image operators and composes them into explicit, human-readable feature extractors. A linear classifier on top of these symbolic features achieves competitive accuracy on ImageNet-1K.

## Core Idea

```
Image  -->  Symbolic Formulas (RL-discovered)  -->  Feature Vectors  -->  Linear Classifier  -->  Prediction
        e.g. "I_GRAY edge_x pow2                  deterministic          nn.Linear(d, 1000)
              I_GRAY edge_y pow2 add                statistics
              sqrt_abs global_avg_pool"             (no hidden layers)
```

Every feature is a deterministic mathematical expression applied to raw pixels. There are no hidden layers, no learned nonlinearities -- only:

1. **Symbolic formula structure** discovered by RL (e.g., `edge_mag relu local_std_5x5 pool_center`)
2. **Convolutional kernel parameters** that can be learned (12 kernels in a `SymbolicKernelBank`)
3. **A single `nn.Linear` layer** as the final classifier

This makes every prediction fully traceable: you can inspect exactly which mathematical operations on which pixel regions led to a classification decision.

## How It Works

### Phase 1: Formula Discovery (Reinforcement Learning)

An RL agent (PPO) generates formulas in Reverse Polish Notation by selecting tokens from a grammar of ~70 tensor operators and ~28 root (pooling) operators. Each formula is evaluated on a subset of ImageNet, and the agent receives a reward based on the formula's discriminative power (classification accuracy). A feature bank stores the best non-redundant formulas, using correlation gating (threshold=0.92) to ensure diversity.

Four parallel banks with different configurations (short formulas, cross-channel, texture-focused, multi-scale) encourage exploration of complementary feature types.

**Example discovered formulas:**
```
I_GRAY edge_mag local_std_5x5 global_avg_pool       # texture strength
I_RG gabor_0 relu pool_center                        # color-oriented texture in center
I_S corner_harris global_max_pool                     # strongest corner in saturation channel
I_r I_g subtract abs blur global_std_pool             # color contrast variation
```

### Phase 2: Hierarchical Composition (Layer 2)

The top-100 most complementary Layer 1 formulas (selected via forward selection for maximum information gain) become new "terminals" for a second round of RL. Layer 2 formulas can express cross-formula spatial co-occurrence patterns that Layer 1 alone cannot capture -- for example, "the edge map is strong where the texture map is also strong."

### Phase 3: Feature Encoding

Raw formula outputs (spatial feature maps) are encoded into rich statistical descriptors:

| Encoding | Dim per formula | What it captures |
|----------|----------------|------------------|
| Distribution statistics | 60 | 12 stats (mean, std, max, skewness, kurtosis, 5 quantiles, ratio) x 5 spatial regions |
| Symbolic Fisher Vector | 4,096 (total) | How local feature distributions deviate from a learned visual vocabulary (GMM) |
| Homogeneous kernel map | 3x per feature | Deterministic approximation of chi-squared kernel SVM |

All encodings are deterministic math -- no neural networks.

### Phase 4: Classification

A linear classifier (`nn.Linear`) is trained on the encoded features with strong weight decay and power + L2 normalization. The entire pipeline from pixels to prediction remains interpretable.

## Operator Library

The grammar includes classical computer vision operators as building blocks:

| Category | Operators |
|----------|-----------|
| **Edge detection** | `edge_x`, `edge_y`, `edge_mag`, `edge_orient`, `edge_xx`, `edge_yy`, `laplacian` |
| **Texture** | `gabor_0`, `gabor_45`, `gabor_90`, `gabor_mag`, `local_std_5x5`, `lbp_like` |
| **Local structure** | `local_contrast`, `dog` (Difference of Gaussians), `corner_harris` |
| **Morphological** | `opening`, `closing`, `tophat`, `dilate` |
| **Frequency** | `high_freq`, `low_freq`, `blur`, `blur_7x7` |
| **Pointwise** | `relu`, `abs`, `sigmoid`, `negate`, `pow2`, `sqrt_abs`, `log1p_abs` |
| **Spatial** | `flip_h`, `flip_v`, `downsample_2x`, `downsample_4x`, `normalize` |
| **Binary** | `add`, `subtract`, `multiply`, `div` |
| **Learnable kernels** | `conv3x3_0..7`, `conv5x5_0..3` (pretrained, then frozen during RL) |
| **Color terminals** | `I_R`, `I_G`, `I_B`, `I_GRAY`, `I_H`, `I_S`, `I_r`, `I_g`, `I_RG`, `I_BY` |
| **Spatial pooling** | `global_avg/max/min/std`, `pool_center`, `pool_quad_*`, `pool_thirds_*`, etc. |

## Results

| Version | Formulas | Encoding | Top-1 | Top-5 |
|---------|----------|----------|-------|-------|
| v2 | 6,000 | 1 scalar pool | 17.27% | 33.64% |
| v2 + SPP | 6,000 | 8 pools | 20.61% | 39.18% |
| v3 | 2,676 | 12 stats | 19.45% | 37.20% |
| **v3.2** (in progress) | ~20K+ | 60 stats + FV + kernel map | target 50%+ | -- |

**Reference:** SIFT + Fisher Vector + linear SVM = 54.3% (ILSVRC 2010 winner). This is the classical, non-deep-learning benchmark we aim to match using RL-discovered (rather than hand-designed) features.

## Project Structure

```
src/
  symbolic/
    tensor_operators.py      # All image operators (edge, texture, morphological, etc.)
    feature_encoding.py      # Distribution stats, Fisher Vector, kernel map
    tensor_evaluator.py      # Execute RPN formulas on image batches
    large_feature_bank.py    # Formula storage with correlation gating
  rl/
    ppo_trainer.py           # PPO training loop
    tensor_environment_large_bank.py  # RL environment: formula generation + evaluation
    rpn_grammar_mask.py      # Grammar enforcement (valid RPN, no degenerate formulas)
  data/
    imagenet_loader.py       # ImageNet data loading with multi-resolution support
  models/
    policy_agent.py          # RL policy network (token-by-token formula generation)

experiments/
    run_v3_2_pipeline.py     # Full v3.2 pipeline (Steps 0-7)
    train_imagenet_pipeline.py  # Core RL training + feature extraction

configs/
    tensor_vsr_imagenet_v3_2.yaml  # Current configuration
```

## Running

```bash
# Full pipeline (from scratch)
python experiments/run_v3_2_pipeline.py --start_step 0

# Resume from a specific step
python experiments/run_v3_2_pipeline.py --start_step 4   # from feature extraction

# Monitor progress
tail -f logs/v3_2_pipeline.log
```

### Requirements

- Python 3.10+
- PyTorch 2.x with CUDA
- ~80 GB GPU memory (A100 recommended)
- ~7 TB disk for feature mmap files (full pipeline)
- See `requirements.txt` for Python dependencies

## Design Constraints

1. **No MLP, no hidden layers.** The classifier is strictly `nn.Linear`.
2. **Every feature is traceable.** Each scalar = a deterministic math expression on pixels.
3. **FP32 throughout.** No mixed precision during operator execution.
4. **Formula structure is symbolic.** Only conv kernel values are learned; the composition is RL-discovered.
5. **No ImageNet mean/std normalization.** Formulas operate on raw [0, 1] pixels.
