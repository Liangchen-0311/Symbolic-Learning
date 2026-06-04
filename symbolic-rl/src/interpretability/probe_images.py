"""v3.3 Section 8.2c — probe images.

For any formula — *especially* an unreadable long one — render the top-k and bottom-k
activating images from the training set. A human instantly sees what the formula responds
to ("ah, it fires on textured fur"), recovering level-2 understanding for formulas whose
token-chain is opaque. This is the single most effective tool for the long-formula problem.

Deterministic; no external model. Reuses the exact terminal channels + RPN execution used
during Layer-1 search.
"""

import torch

from src.symbolic.layer1_cache import build_data_batch, execute_body_map
from src.symbolic.tensor_operators import TENSOR_OPERATORS, ROOT_OPERATORS


def formula_scalar_values(formula_str, images, device='cpu', batch_size=512):
    """Compute the scalar feature value of a formula for every image.

    If the formula ends in a pooling (ROOT) op, that pool is used; otherwise the body map
    is reduced by its mean (so any body, pooled or not, yields one scalar per image).

    Args:
        formula_str: RPN token chain.
        images: [N, 3, H, W] in [0,1].
        device: torch device.
        batch_size: chunk size to bound memory.
    Returns:
        Tensor [N] of scalar activations (FP32, finite).
    """
    tokens = formula_str.strip().split()
    pool_tok = tokens[-1] if (tokens and tokens[-1] in ROOT_OPERATORS) else None
    out = []
    n = images.shape[0]
    for s in range(0, n, batch_size):
        chunk = images[s:s + batch_size]
        db = build_data_batch(chunk, device)
        body = execute_body_map(formula_str, db)            # [B,H,W] pre-pool map
        if body is None:
            out.append(torch.zeros(chunk.shape[0]))
            continue
        if pool_tok is not None:
            val = TENSOR_OPERATORS[pool_tok][0](body)
            if val.dim() > 1:                               # multi-dim pool -> reduce
                val = val.mean(dim=1)
        else:
            val = body.mean(dim=(-2, -1))
        out.append(torch.nan_to_num(val, nan=0.0, posinf=1e4, neginf=-1e4).cpu())
    return torch.cat(out)[:n]


def top_bottom_activators(formula_str, images, k=8, device='cpu'):
    """Return indices (and values) of the top-k and bottom-k activating images.

    Returns:
        dict with 'top_idx','bottom_idx' (LongTensor [k]) and 'top_val','bottom_val'.
    """
    vals = formula_scalar_values(formula_str, images, device=device)
    k = min(k, vals.shape[0])
    top = torch.topk(vals, k, largest=True)
    bot = torch.topk(vals, k, largest=False)
    return {
        'top_idx': top.indices, 'top_val': top.values,
        'bottom_idx': bot.indices, 'bottom_val': bot.values,
    }


def save_probe_montage(formula_str, images, out_path, k=8, device='cpu', title=None):
    """Render a top-k (row 1) / bottom-k (row 2) montage PNG for a formula.

    Returns the out_path on success, or None if matplotlib is unavailable (logged, never
    fatal — keeps the harness running headless).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [probe_images] matplotlib unavailable, skipping montage: {e}")
        return None

    act = top_bottom_activators(formula_str, images, k=k, device=device)
    k = act['top_idx'].shape[0]
    fig, axes = plt.subplots(2, k, figsize=(1.4 * k, 3.2))
    if k == 1:
        axes = axes.reshape(2, 1)
    for col in range(k):
        for row, (key, vkey) in enumerate([('top_idx', 'top_val'), ('bottom_idx', 'bottom_val')]):
            idx = int(act[key][col])
            img = images[idx].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            ax = axes[row, col]
            ax.imshow(img)
            ax.set_title(f"{float(act[vkey][col]):.2f}", fontsize=7)
            ax.axis('off')
    axes[0, 0].set_ylabel('TOP', fontsize=9)
    axes[1, 0].set_ylabel('BOTTOM', fontsize=9)
    fig.suptitle(title or f"probe: {formula_str[:70]}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return out_path
