"""
Layer-1 feature-map cache (v3.3 Section 4B).

Executes each Layer-1 *body* (an RPN string WITHOUT a final pooling token) on every
image but stops before pooling, yielding the 2D feature map [H, W]. These cached maps
feed the Layer-2 enumerator (Section 4C), which builds cross-formula spatial interactions
a pooled linear classifier cannot see.

Storage (Section 4 / Hard-Constraint 3 — FP32 compute, FP16 only for caching):
  - CIFAR-10:  FP32 on GPU,  16x16  (~1.5 GB for 30 bodies x 50k).
  - ImageNet:  FP16 on CPU/disk (numpy memmap), 28x28, cast to FP32 on read. Prefer
               caching PER SUPERCLASS (each subset is far smaller).
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from src.symbolic.tensor_operators import TENSOR_OPERATORS, ROOT_OPERATORS, TensorOperators


def build_data_batch(images, device):
    """Build the terminal channels (RGB + grayscale + HSV + ratios + opponent + the v3.3
    Section 1A.0 deterministic prior terminals I_EDGE/I_FREQ/I_LAPLACIAN) from a batch of
    [B, 3, H, W] images in [0,1]. Matches the channels available during Layer-1 search so
    any discovered body (including ones that reference a prior terminal) stays re-executable
    downstream (traceability — Hard Constraint #4). Prior terminals are cheap and added
    unconditionally; unused keys are simply never popped by `execute_body_map`."""
    images = images.to(device, dtype=torch.float32)
    I_R, I_G, I_B = images[:, 0], images[:, 1], images[:, 2]
    I_GRAY = 0.2989 * I_R + 0.5870 * I_G + 0.1140 * I_B
    Cmax, _ = images.max(dim=1)
    Cmin, _ = images.min(dim=1)
    delta = Cmax - Cmin + 1e-8
    Hh = torch.zeros_like(I_R)
    mr = (Cmax == I_R)
    mg = (Cmax == I_G) & ~mr
    mb = ~mr & ~mg
    Hh[mr] = (((I_G[mr] - I_B[mr]) / delta[mr]) % 6)
    Hh[mg] = ((I_B[mg] - I_R[mg]) / delta[mg]) + 2
    Hh[mb] = ((I_R[mb] - I_G[mb]) / delta[mb]) + 4
    Hh = Hh / 6.0
    S = torch.where(Cmax > 1e-8, delta / (Cmax + 1e-8), torch.zeros_like(Cmax))
    total = I_R + I_G + I_B + 1e-8
    db = {
        'I_R': I_R, 'I_G': I_G, 'I_B': I_B, 'I_GRAY': I_GRAY,
        'I_H': Hh, 'I_S': S, 'I_r': I_R / total, 'I_g': I_G / total,
        'I_RG': I_R - I_G, 'I_BY': I_B - (I_R + I_G) / 2,
    }
    # v3.3 Section 1A.0 deterministic prior terminals (keep downstream re-executable).
    db.update(TensorOperators.make_prior_terminals(I_GRAY))
    return db


def execute_body_map(body_str, data_batch):
    """Execute a Layer-1 body (no final pool) -> 2D feature map [B, H, W] FP32.

    Returns None if the body is malformed or does not reduce to a single 2D tensor.
    If the body accidentally ends in a pooling token, that token is dropped first
    (we want the pre-pool map, per Section 4B)."""
    tokens = body_str.strip().split()
    if tokens and tokens[-1] in ROOT_OPERATORS:
        tokens = tokens[:-1]
    stack = []
    for token in tokens:
        if token in data_batch:
            stack.append(data_batch[token])
        elif token in TENSOR_OPERATORS:
            op_func, arity, _ = TENSOR_OPERATORS[token]
            if len(stack) < arity:
                return None
            operands = [stack.pop() for _ in range(arity)]
            operands.reverse()
            result = op_func(*operands)
            result = torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)
            stack.append(result)
        else:
            return None
    if len(stack) != 1:
        return None
    out = torch.clamp(stack[0], -1e4, 1e4)
    return out if out.dim() >= 2 else None


def _resize_map(m, resolution):
    """Resize a [B, H, W] map to [B, resolution, resolution] via adaptive avg pool."""
    if m.shape[-1] == resolution and m.shape[-2] == resolution:
        return m
    return F.adaptive_avg_pool2d(m.unsqueeze(1), output_size=resolution).squeeze(1)


class Layer1Cache:
    """Cache of pre-pool Layer-1 feature maps for a fixed set of bodies.

    build(images) populates the cache; get(body_idx, image_indices) returns [N, R, R] FP32.
    """

    def __init__(self, bodies, device='cpu', resolution=16, storage='gpu',
                 memmap_path=None):
        """
        Args:
            bodies: list of Layer-1 body RPN strings (no final pool).
            device: torch device for execution.
            resolution: cached map side length (16 CIFAR, 28 ImageNet).
            storage: 'gpu' (FP32 tensor on device) | 'cpu' (FP32 CPU tensor)
                     | 'memmap' (FP16 numpy memmap on disk, cast to FP32 on read).
            memmap_path: required if storage == 'memmap'.
        """
        self.bodies = list(bodies)
        self.device = device
        self.resolution = int(resolution)
        self.storage = storage
        self.memmap_path = memmap_path
        self.n_bodies = len(self.bodies)
        self.n_images = 0
        self._store = None        # tensor [n_bodies, N, R, R] or np.memmap [n_bodies, N, R, R]
        self.valid_mask = None    # bool[n_bodies] — bodies that executed cleanly

    def build(self, images, batch_size=512, verbose=False):
        """Execute every body on every image (batched), storing the resized pre-pool maps.

        Args:
            images: [N, 3, H, W] float tensor in [0,1] (CPU or GPU).
            batch_size: image batch size for execution.
        """
        N = images.shape[0]
        self.n_images = N
        R = self.resolution
        nb = self.n_bodies
        self.valid_mask = np.ones(nb, dtype=bool)

        if self.storage == 'memmap':
            assert self.memmap_path is not None, "memmap storage requires memmap_path"
            os.makedirs(os.path.dirname(os.path.abspath(self.memmap_path)), exist_ok=True)
            self._store = np.memmap(self.memmap_path, dtype=np.float16, mode='w+',
                                    shape=(nb, N, R, R))
        elif self.storage == 'cpu':
            self._store = torch.zeros((nb, N, R, R), dtype=torch.float32)
        else:  # gpu
            self._store = torch.zeros((nb, N, R, R), dtype=torch.float32, device=self.device)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = images[start:end]
            db = build_data_batch(batch, self.device)
            for bi, body in enumerate(self.bodies):
                m = execute_body_map(body, db)
                if m is None:
                    self.valid_mask[bi] = False
                    continue
                m = _resize_map(m, R).to(torch.float32)   # [b, R, R]
                if self.storage == 'memmap':
                    self._store[bi, start:end] = m.detach().cpu().numpy().astype(np.float16)
                else:
                    self._store[bi, start:end] = m.detach().to(self._store.device)
            if verbose:
                print(f"  [Layer1Cache] {end}/{N} images")
        if self.storage == 'memmap':
            self._store.flush()
        return self

    def get(self, body_idx, image_indices=None):
        """Return [N, R, R] FP32 maps for a body (optionally a subset of images)."""
        if self.storage == 'memmap':
            arr = self._store[body_idx]
            if image_indices is not None:
                arr = arr[np.asarray(image_indices)]
            return torch.from_numpy(np.asarray(arr, dtype=np.float32))
        t = self._store[body_idx]
        if image_indices is not None:
            idx = torch.as_tensor(image_indices, device=t.device, dtype=torch.long)
            t = t.index_select(0, idx)
        return t.to(torch.float32)

    def get_pair(self, i, j, image_indices=None, device=None):
        """Convenience: return (map_i, map_j) on the requested device."""
        mi = self.get(i, image_indices)
        mj = self.get(j, image_indices)
        if device is not None:
            mi, mj = mi.to(device), mj.to(device)
        return mi, mj
