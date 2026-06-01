"""
Tensor/Spatial Operators for Image Processing

These operators work on batched image tensors [batch, height, width]
instead of scalar features.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_gabor_kernel(theta, sigma=2.0, freq=0.5, kernel_size=7, device='cpu'):
    """Build a single Gabor filter kernel.

    Args:
        theta: Orientation in radians.
        sigma: Gaussian envelope width.
        freq: Spatial frequency of the sinusoid.
        kernel_size: Size of the square kernel.
        device: Torch device.

    Returns:
        [1, 1, kernel_size, kernel_size] float32 tensor.
    """
    half = kernel_size // 2
    y, x = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32, device=device),
        torch.arange(-half, half + 1, dtype=torch.float32, device=device),
        indexing='ij',
    )
    # Rotate coordinates
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x_theta = x * cos_t + y * sin_t
    y_theta = -x * sin_t + y * cos_t

    gaussian = torch.exp(-0.5 * (x_theta ** 2 + y_theta ** 2) / (sigma ** 2))
    sinusoid = torch.cos(2 * math.pi * freq * x_theta)
    kernel = gaussian * sinusoid

    # Zero-mean normalisation so that the filter has no DC response
    kernel = kernel - kernel.mean()
    return kernel.view(1, 1, kernel_size, kernel_size)


def _safe_binary(func):
    """Decorator that clamps output and handles NaN after binary operations."""
    def wrapper(x, y):
        # Match spatial sizes if needed (scalar vs tensor)
        if x.dim() != y.dim():
            if x.dim() < y.dim():
                x = x.unsqueeze(-1).unsqueeze(-1).expand_as(y)
            else:
                y = y.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        elif x.dim() >= 2 and x.shape != y.shape:
            # Both are tensors but different spatial sizes — use smaller
            min_h = min(x.shape[-2], y.shape[-2])
            min_w = min(x.shape[-1], y.shape[-1])
            x = x[..., :min_h, :min_w]
            y = y[..., :min_h, :min_w]
        result = func(x, y)
        return torch.clamp(torch.nan_to_num(result, nan=0.0), -60000, 60000)
    return wrapper


class TensorOperators:
    """
    Spatial operators for image processing.
    All methods work on batched tensors: [batch, H, W]
    """

    # ============================================
    # Element-wise Operations
    # ============================================

    @staticmethod
    @_safe_binary
    def add(x, y):
        """Element-wise addition: x + y"""
        return x + y

    @staticmethod
    @_safe_binary
    def subtract(x, y):
        """Element-wise subtraction: x - y"""
        return x - y

    @staticmethod
    @_safe_binary
    def multiply(x, y):
        """Hadamard product (element-wise multiplication)."""
        return x * y

    @staticmethod
    @_safe_binary
    def div(x, y):
        """Safe element-wise division: x / y (with epsilon for stability)."""
        return x / (y + 1e-8 * torch.sign(y).clamp(min=1e-8))

    # ============================================
    # Non-linear Activations / Pointwise
    # ============================================

    @staticmethod
    def relu(x):
        """ReLU activation"""
        return F.relu(x)

    @staticmethod
    def negate(x):
        """Negation: -x. Enables 'absence of feature' detection."""
        return -x

    @staticmethod
    def pow2(x):
        """Square: x². Emphasizes large activations, useful for energy features."""
        return x * x

    @staticmethod
    def sqrt_abs(x):
        """sqrt(|x|). Compresses dynamic range — opposite of pow2."""
        return torch.sqrt(torch.abs(x) + 1e-8)

    @staticmethod
    def log1p_abs(x):
        """log(1 + |x|). Logarithmic compression for large-range features."""
        return torch.log1p(torch.abs(x))

    # ============================================
    # Spatial/Structural Operators (Key!)
    # ============================================

    @staticmethod
    def blur(x):
        """
        Local averaging (3×3 blur).
        Mimics receptive field.

        Args:
            x: [batch, H, W]
        Returns:
            [batch, H, W] blurred image
        """
        x_unsq = x.unsqueeze(1)  # [batch, 1, H, W]
        blurred = F.avg_pool2d(x_unsq, kernel_size=3, stride=1, padding=1)
        return blurred.squeeze(1)

    @staticmethod
    def edge_x(x):
        """
        Horizontal edge detection (Sobel X).
        Fixed kernel: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

        Args:
            x: [batch, H, W]
        Returns:
            [batch, H, W] horizontal edges
        """
        kernel = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ], device=x.device).view(1, 1, 3, 3)

        x_unsq = x.unsqueeze(1)  # [batch, 1, H, W]
        edge = F.conv2d(x_unsq, kernel, padding=1)
        return edge.squeeze(1)

    @staticmethod
    def edge_y(x):
        """
        Vertical edge detection (Sobel Y).
        Fixed kernel: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        """
        kernel = torch.tensor([
            [-1., -2., -1.],
            [ 0.,  0.,  0.],
            [ 1.,  2.,  1.]
        ], device=x.device).view(1, 1, 3, 3)

        x_unsq = x.unsqueeze(1)
        edge = F.conv2d(x_unsq, kernel, padding=1)
        return edge.squeeze(1)

    @staticmethod
    def blur_7x7(x):
        """7x7 average blur (larger receptive field for ImageNet-scale images)."""
        x_4d = x.unsqueeze(1)
        blurred = F.avg_pool2d(x_4d, kernel_size=7, stride=1, padding=3)
        return blurred.squeeze(1)

    @staticmethod
    def dilate(x):
        """
        Morphological dilation (max pooling).
        """
        x_unsq = x.unsqueeze(1)
        dilated = F.max_pool2d(x_unsq, kernel_size=3, stride=1, padding=1)
        return dilated.squeeze(1)

    # ============================================
    # Morphological Operators
    # ============================================

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

    # ============================================
    # Frequency Separation Operators
    # ============================================

    @staticmethod
    def high_freq(x):
        """High-frequency residual: x - blur_7x7(x). Fine texture and detail."""
        blurred = F.avg_pool2d(x.unsqueeze(1), kernel_size=7, stride=1, padding=3).squeeze(1)
        return x - blurred

    @staticmethod
    def low_freq(x):
        """Low-frequency: 15×15 blur. Overall color/brightness gradients."""
        return F.avg_pool2d(x.unsqueeze(1), kernel_size=15, stride=1, padding=7).squeeze(1)

    # ============================================
    # New: Utility Operators
    # ============================================

    @staticmethod
    def abs_val(x):
        """Absolute value, extremely useful after edge detection or subtraction."""
        return torch.abs(x)

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation. Bounds output to [0,1], stabilizing multiply chains."""
        return torch.sigmoid(x)

    @staticmethod
    def laplacian(x):
        """
        2nd derivative filter (Laplacian).
        Excellent for extracting fine textures, blobs, and omni-directional corners.
        """
        weight = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]], device=x.device)
        x_unsqueezed = x.unsqueeze(1)
        out = F.conv2d(x_unsqueezed, weight, padding=1)
        return out.squeeze(1)

    @staticmethod
    def normalize(x):
        """
        Instance normalization (zero mean, unit variance per image).
        Provides contrast and illumination invariance.
        """
        mean = x.mean(dim=[-1, -2], keepdim=True)
        std = x.std(dim=[-1, -2], keepdim=True) + 1e-5
        return (x - mean) / std

    # ============================================
    # Geometric Operators
    # ============================================

    @staticmethod
    def flip_h(x):
        """Horizontal flip. x - flip_h(x) extracts left-right asymmetry."""
        return x.flip(-1)

    @staticmethod
    def flip_v(x):
        """Vertical flip. x - flip_v(x) extracts top-bottom asymmetry."""
        return x.flip(-2)

    # ============================================
    # Multi-scale Operators
    # ============================================

    @staticmethod
    def downsample_2x(x):
        """2x spatial downsampling via average pooling: [B,H,W] -> [B,H/2,W/2]"""
        return F.avg_pool2d(x.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)

    @staticmethod
    def downsample_4x(x):
        """4x spatial downsampling: [B,H,W] -> [B,H/4,W/4]"""
        return F.avg_pool2d(x.unsqueeze(1), kernel_size=4, stride=4).squeeze(1)

    @staticmethod
    def stride_pool_4(x):
        """4x max pooling (preserves strong activations): [B,H,W] -> [B,H/4,W/4]"""
        return F.max_pool2d(x.unsqueeze(1), kernel_size=4, stride=4).squeeze(1)

    # ============================================
    # Texture / Frequency Operators
    # ============================================

    # Class-level Gabor kernel cache — avoids rebuilding per call
    _gabor_cache: dict = {}

    @staticmethod
    def _get_gabor_kernel(theta, device):
        """Retrieve (or build & cache) a Gabor kernel for *theta* on *device*."""
        key = (theta, str(device))
        if key not in TensorOperators._gabor_cache:
            TensorOperators._gabor_cache[key] = _make_gabor_kernel(
                theta, device=device
            )
        return TensorOperators._gabor_cache[key]

    @staticmethod
    def gabor_0(x):
        """Gabor filter at 0° — detects horizontal texture/edges."""
        kernel = TensorOperators._get_gabor_kernel(0.0, x.device)
        return F.conv2d(x.unsqueeze(1), kernel, padding=3).squeeze(1)

    @staticmethod
    def gabor_45(x):
        """Gabor filter at 45° — detects diagonal texture/edges."""
        kernel = TensorOperators._get_gabor_kernel(math.pi / 4, x.device)
        return F.conv2d(x.unsqueeze(1), kernel, padding=3).squeeze(1)

    @staticmethod
    def gabor_90(x):
        """Gabor filter at 90° — detects vertical texture/edges."""
        kernel = TensorOperators._get_gabor_kernel(math.pi / 2, x.device)
        return F.conv2d(x.unsqueeze(1), kernel, padding=3).squeeze(1)

    @staticmethod
    def local_std_5x5(x):
        """Local standard deviation over 5×5 patches.

        Computed as sqrt(E[X²] - E[X]²).
        Effective for detecting texture roughness / homogeneous vs busy regions.
        """
        x_4d = x.unsqueeze(1)  # [B, 1, H, W]
        local_mean = F.avg_pool2d(x_4d, kernel_size=5, stride=1, padding=2)
        local_sq_mean = F.avg_pool2d(x_4d ** 2, kernel_size=5, stride=1, padding=2)
        variance = (local_sq_mean - local_mean ** 2).clamp(min=0.0)
        return variance.sqrt().squeeze(1)

    # ============================================
    # Rotation-Invariant Operators (v3.2)
    # ============================================

    @staticmethod
    def edge_mag(x):
        """Gradient magnitude (rotation invariant): sqrt(edge_x² + edge_y²).
        Corresponds to |∇f| — the fundamental rotation-invariant edge measure."""
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
        return torch.sqrt(g0 * g0 + g45 * g45 + g90 * g90 + 1e-8)

    # ============================================
    # Local Structure Operators (v3.2)
    # ============================================

    @staticmethod
    def local_contrast(x):
        """Local contrast normalization: (x - local_mean) / local_std.
        SIFT's first step — removes illumination variation, preserves structure."""
        x4d = x.unsqueeze(1)
        local_mean = F.avg_pool2d(x4d, kernel_size=7, stride=1, padding=3)
        local_sq_mean = F.avg_pool2d(x4d ** 2, kernel_size=7, stride=1, padding=3)
        local_var = (local_sq_mean - local_mean ** 2).clamp(min=0)
        local_std = torch.sqrt(local_var + 1e-8)
        return ((x4d - local_mean) / local_std).squeeze(1)

    @staticmethod
    def dog(x):
        """Difference of Gaussians: blur_3x3(x) - blur_7x7(x).
        SIFT's core — detects blobs and keypoints at a specific scale."""
        x4d = x.unsqueeze(1)
        fine = F.avg_pool2d(x4d, kernel_size=3, stride=1, padding=1)
        coarse = F.avg_pool2d(x4d, kernel_size=7, stride=1, padding=3)
        return (fine - coarse).squeeze(1)

    @staticmethod
    def corner_harris(x):
        """Harris corner response: det(M) - 0.04 * trace(M)².
        M = structure tensor (smoothed outer product of gradients).
        High response = corner/junction, low = flat or edge."""
        ix = TensorOperators.edge_x(x)
        iy = TensorOperators.edge_y(x)
        ix2 = F.avg_pool2d((ix * ix).unsqueeze(1), 5, 1, 2).squeeze(1)
        iy2 = F.avg_pool2d((iy * iy).unsqueeze(1), 5, 1, 2).squeeze(1)
        ixiy = F.avg_pool2d((ix * iy).unsqueeze(1), 5, 1, 2).squeeze(1)
        det = ix2 * iy2 - ixiy * ixiy
        trace = ix2 + iy2
        return det - 0.04 * trace * trace

    @staticmethod
    def lbp_like(x):
        """LBP approximation: sigmoid(center - local_mean).
        Output ∈ [0,1]. 1 = center brighter than neighbors, 0 = darker."""
        x4d = x.unsqueeze(1)
        neighbors = F.avg_pool2d(x4d, kernel_size=3, stride=1, padding=1)
        return torch.sigmoid(10 * (x4d - neighbors)).squeeze(1)

    # ============================================
    # Second-Order Operators (v3.2)
    # ============================================

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

    # ============================================
    # Semantic Operators (v3.3) — mid-level structure
    # ============================================

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

    # ============================================
    # Fuzzy-Logic Operators (v3.3) — probabilistic t-norms
    # ============================================
    # Product/probabilistic form (not min/max): differentiable everywhere (needed for
    # optional end-to-end kernel fine-tuning) and avoids sparse-gradient issues. Inputs
    # are squashed to [0,1] via sigmoid first so the "degree of truth" semantics hold.

    @staticmethod
    def fuzzy_not(x):
        """Fuzzy NOT: 1 - σ(x). Output in (0,1)."""
        return 1.0 - torch.sigmoid(x)

    @staticmethod
    @_safe_binary
    def fuzzy_and(x, y):
        """Fuzzy AND (probabilistic t-norm): σ(x) * σ(y). Output in (0,1).
        'This region satisfies BOTH feature x AND feature y.'"""
        return torch.sigmoid(x) * torch.sigmoid(y)

    @staticmethod
    @_safe_binary
    def fuzzy_or(x, y):
        """Fuzzy OR (probabilistic t-conorm): σ(x)+σ(y)-σ(x)σ(y). Output in (0,1).
        'This region satisfies feature x OR feature y.'"""
        sx, sy = torch.sigmoid(x), torch.sigmoid(y)
        return sx + sy - sx * sy

    # ============================================
    # Pooling/Reduction Operators (CRITICAL!)
    # ============================================

    @staticmethod
    def global_avg_pool(x):
        """
        Global average pooling: [batch, H, W] → [batch]

        This is a MANDATORY operator for the root node.
        Converts spatial tensor to scalar feature.
        """
        return torch.mean(x, dim=[-2, -1])

    @staticmethod
    def global_max_pool(x):
        """
        Global max pooling: [batch, H, W] → [batch]

        This is a MANDATORY operator for the root node.
        Converts spatial tensor to scalar feature.
        """
        return torch.amax(x, dim=[-2, -1])

    @staticmethod
    def global_std_pool(x):
        """
        Global std pooling: [batch, H, W] → [batch]

        Can be used as root node.
        Measures variation in the image.
        """
        return torch.std(x, dim=[-2, -1])

    @staticmethod
    def global_l2_pool(x):
        """
        Global L2 norm: [batch, H, W] → [batch]

        Can be used as root node.
        """
        return torch.norm(x, p=2, dim=[-2, -1])

    @staticmethod
    def global_min_pool(x):
        """
        Global min pooling: [batch, H, W] → [batch]

        Detects absence of feature — useful for distinguishing
        indoor/outdoor, dark regions, etc.
        """
        return x.amin(dim=[-2, -1])

    # ============================================
    # Distribution-Aware Pooling (Root)
    # ============================================

    @staticmethod
    def ratio_above_mean(x):
        """Fraction of pixels above the mean: [B,H,W] → [B].
        Measures spatial EXTENT of a feature — unlike avg_pool which measures intensity."""
        mean = x.mean(dim=[-2, -1], keepdim=True)
        return (x > mean).float().mean(dim=[-2, -1])

    @staticmethod
    def percentile_90(x):
        """90th percentile value: [B,H,W] → [B].
        Robust peak measurement (unlike global_max_pool which is noise-sensitive)."""
        flat = x.flatten(1)
        k = max(1, flat.shape[1] // 10)
        return flat.kthvalue(flat.shape[1] - k + 1, dim=1).values

    @staticmethod
    def spatial_entropy(x):
        """Approximate spatial entropy via histogram: [B,H,W] → [B].
        High = feature spread evenly. Low = feature concentrated in one spot."""
        flat = x.flatten(1)
        mins = flat.min(dim=1, keepdim=True).values
        maxs = flat.max(dim=1, keepdim=True).values
        normalized = (flat - mins) / (maxs - mins + 1e-8)
        n_bins = 16
        bin_idx = (normalized * (n_bins - 1e-4)).long().clamp(0, n_bins - 1)
        counts = torch.zeros(flat.shape[0], n_bins, device=x.device)
        counts.scatter_add_(1, bin_idx, torch.ones_like(flat))
        probs = counts / counts.sum(dim=1, keepdim=True).clamp(min=1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=1)
        return entropy

    @staticmethod
    def peak_location_y(x):
        """Vertical position of max activation (normalized 0-1): [B,H,W] → [B]."""
        flat = x.flatten(1)
        max_idx = flat.argmax(dim=1)
        H, W = x.shape[-2], x.shape[-1]
        return (max_idx // W).float() / max(H - 1, 1)

    @staticmethod
    def peak_location_x(x):
        """Horizontal position of max activation (normalized 0-1): [B,H,W] → [B]."""
        flat = x.flatten(1)
        max_idx = flat.argmax(dim=1)
        W = x.shape[-1]
        return (max_idx % W).float() / max(W - 1, 1)

    # ============================================
    # Multi-Dim Pooling (Root)
    # ============================================

    @staticmethod
    def patch_histogram_4x4(x):
        """Spatial Bag-of-Words: [B,H,W] → [B, 4].
        Divide into 4×4 grid (16 patches), compute patch means,
        build 4-bin soft histogram of patch activations."""
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
        return torch.stack(bins, dim=1)

    # ============================================
    # Spatial Pooling Operators (Position-Aware)
    # ============================================

    @staticmethod
    def pool_top_half(x):
        """
        Pool top half of image: [batch, H, W] → [batch]

        Captures "sky region" information (birds, planes).
        Useful for objects typically in upper part of image.
        """
        top_half = x[:, :x.shape[1]//2, :]  # Top half [batch, H/2, W]
        return torch.mean(top_half, dim=[-2, -1])

    @staticmethod
    def pool_bottom_half(x):
        """
        Pool bottom half of image: [batch, H, W] → [batch]

        Captures "ground region" information (cars, animals).
        Useful for objects typically in lower part of image.
        """
        bottom_half = x[:, x.shape[1]//2:, :]  # Bottom half [batch, H/2, W]
        return torch.mean(bottom_half, dim=[-2, -1])

    @staticmethod
    def pool_left_half(x):
        """
        Pool left half of image: [batch, H, W] → [batch]

        Captures left region information.
        """
        left_half = x[:, :, :x.shape[2]//2]  # Left half [batch, H, W/2]
        return torch.mean(left_half, dim=[-2, -1])

    @staticmethod
    def pool_right_half(x):
        """
        Pool right half of image: [batch, H, W] → [batch]

        Captures right region information.
        """
        right_half = x[:, :, x.shape[2]//2:]  # Right half [batch, H, W/2]
        return torch.mean(right_half, dim=[-2, -1])

    @staticmethod
    def pool_center(x):
        """
        Pool center region: [batch, H, W] → [batch]

        Captures central region (typically contains main object).
        Center is defined as middle 50% of image (25% border removed).
        """
        h, w = x.shape[1], x.shape[2]
        h_start, h_end = h // 4, 3 * h // 4
        w_start, w_end = w // 4, 3 * w // 4
        center = x[:, h_start:h_end, w_start:w_end]  # Center [batch, H/2, W/2]
        return torch.mean(center, dim=[-2, -1])

    @staticmethod
    def pool_corners(x):
        """
        Pool corner regions: [batch, H, W] → [batch]

        Captures background/context information from corners.
        """
        h, w = x.shape[1], x.shape[2]
        corner_size_h = h // 4
        corner_size_w = w // 4

        # Extract four corners
        corners = torch.cat([
            x[:, :corner_size_h, :corner_size_w].flatten(1),      # Top-left
            x[:, :corner_size_h, -corner_size_w:].flatten(1),     # Top-right
            x[:, -corner_size_h:, :corner_size_w].flatten(1),     # Bottom-left
            x[:, -corner_size_h:, -corner_size_w:].flatten(1),    # Bottom-right
        ], dim=1)
        return torch.mean(corners, dim=1)

    # ============================================
    # Spatial Pooling: Thirds (horizontal strips)
    # ============================================

    @staticmethod
    def pool_thirds_top(x):
        """Pool top 1/3 of image: [B,H,W] -> [B]"""
        H = x.shape[-2]
        return x[..., :H // 3, :].mean(dim=[-2, -1])

    @staticmethod
    def pool_thirds_mid(x):
        """Pool middle 1/3 of image: [B,H,W] -> [B]"""
        H = x.shape[-2]
        return x[..., H // 3:2 * H // 3, :].mean(dim=[-2, -1])

    @staticmethod
    def pool_thirds_bot(x):
        """Pool bottom 1/3 of image: [B,H,W] -> [B]"""
        H = x.shape[-2]
        return x[..., 2 * H // 3:, :].mean(dim=[-2, -1])

    # ============================================
    # Spatial Pooling: Quadrants
    # ============================================

    @staticmethod
    def pool_quad_tl(x):
        """Pool top-left quadrant: [B,H,W] -> [B]"""
        H, W = x.shape[-2], x.shape[-1]
        return x[..., :H // 2, :W // 2].mean(dim=[-2, -1])

    @staticmethod
    def pool_quad_tr(x):
        """Pool top-right quadrant: [B,H,W] -> [B]"""
        H, W = x.shape[-2], x.shape[-1]
        return x[..., :H // 2, W // 2:].mean(dim=[-2, -1])

    @staticmethod
    def pool_quad_bl(x):
        """Pool bottom-left quadrant: [B,H,W] -> [B]"""
        H, W = x.shape[-2], x.shape[-1]
        return x[..., H // 2:, :W // 2].mean(dim=[-2, -1])

    @staticmethod
    def pool_quad_br(x):
        """Pool bottom-right quadrant: [B,H,W] -> [B]"""
        H, W = x.shape[-2], x.shape[-1]
        return x[..., H // 2:, W // 2:].mean(dim=[-2, -1])

    # ============================================
    # Spatial Pooling: Surround (border ring)
    # ============================================

    @staticmethod
    def pool_surround(x):
        """Pool border ring (8px): [B,H,W] -> [B]. Captures background/context."""
        B_px = 8
        H, W = x.shape[-2], x.shape[-1]
        # If image too small for 8px border, fall back to global mean
        if H <= 2 * B_px or W <= 2 * B_px:
            return x.mean(dim=[-2, -1])
        mask = torch.ones(H, W, device=x.device, dtype=torch.bool)
        mask[B_px:-B_px, B_px:-B_px] = False
        border_pixels = x[..., mask]  # [B, num_border_pixels]
        return border_pixels.mean(dim=-1)

    # ============================================
    # Variance-Aware Spatial Pooling (v3.2)
    # ============================================

    @staticmethod
    def std_center(x):
        """Center region standard deviation: [B,H,W] → [B].
        Captures texture variation in the main object region."""
        h, w = x.shape[1], x.shape[2]
        center = x[:, h // 4:3 * h // 4, w // 4:3 * w // 4]
        return torch.std(center, dim=[-2, -1])

    @staticmethod
    def std_top_half(x):
        """Top half standard deviation: [B,H,W] → [B]."""
        return torch.std(x[:, :x.shape[1] // 2, :], dim=[-2, -1])

    @staticmethod
    def std_bottom_half(x):
        """Bottom half standard deviation: [B,H,W] → [B]."""
        return torch.std(x[:, x.shape[1] // 2:, :], dim=[-2, -1])

    # ============================================
    # High-Order Statistical Pooling (v3.3 Section 1A.2, root-only)
    # ============================================
    # Existing pooling captures only first-order moments (mean/max/min/std/L2). Two images
    # with identical mean+std can have very different distribution SHAPE. These expose shape
    # and texture statistics directly — and are the strongest synergy with HistGB, since each
    # is a directly threshold-able, semantically meaningful split signal.

    # ---- Tier 1: distribution-shape moments + quantiles ----
    @staticmethod
    def pool_skewness(x):
        """3rd standardized moment — distribution asymmetry: [B,H,W] → [B].
        e.g. bright-sky scenes (bright pixels dominate) vs night scenes have opposite sign."""
        flat = x.reshape(x.shape[0], -1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True) + 1e-8
        return (((flat - mean) / std) ** 3).mean(dim=1)

    @staticmethod
    def pool_kurtosis(x):
        """4th standardized moment minus 3 — tail heaviness: [B,H,W] → [B].
        sparse strong edges (high) vs uniform texture (low)."""
        flat = x.reshape(x.shape[0], -1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True) + 1e-8
        return (((flat - mean) / std) ** 4).mean(dim=1) - 3.0

    @staticmethod
    def pool_q10(x):
        """10th percentile — representative dark-region value: [B,H,W] → [B]."""
        return torch.quantile(x.reshape(x.shape[0], -1), 0.10, dim=1)

    @staticmethod
    def pool_q90(x):
        """90th percentile — representative bright-region value: [B,H,W] → [B]."""
        return torch.quantile(x.reshape(x.shape[0], -1), 0.90, dim=1)

    @staticmethod
    def pool_iqr(x):
        """Inter-quartile range q75-q25 — robust spread, outlier-resistant vs std: [B,H,W] → [B]."""
        flat = x.reshape(x.shape[0], -1)
        return torch.quantile(flat, 0.75, dim=1) - torch.quantile(flat, 0.25, dim=1)

    @staticmethod
    def pool_above_mean_ratio(x):
        """Fraction of pixels above the mean — 'bright-area occupancy': [B,H,W] → [B].
        high for sky/snow scenes, low for dark scenes."""
        flat = x.reshape(x.shape[0], -1)
        mean = flat.mean(dim=1, keepdim=True)
        return (flat > mean).float().mean(dim=1)

    # ---- Tier 2: information-theoretic statistics (soft histograms) ----
    @staticmethod
    def pool_entropy(x):
        """Shannon entropy of a 32-bin soft histogram (per-sample min-max normalized): [B,H,W] → [B].
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
        """Mean squared value — signal energy normalized by area: [B,H,W] → [B]."""
        return (x ** 2).mean(dim=(-2, -1))

    @staticmethod
    def pool_uniformity(x):
        """Sum of squared histogram probabilities (Σ p_i²) — inverse of dispersion: [B,H,W] → [B].
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

    # ---- Tier 3: co-occurrence / spatial-correlation statistics ----
    @staticmethod
    def pool_neighbor_diff_var(x):
        """Variance of adjacent-pixel differences — local contrast statistic: [B,H,W] → [B].
        high = high-frequency texture (fur, fabric), low = flat."""
        diff_x = x[:, :, 1:] - x[:, :, :-1]
        diff_y = x[:, 1:, :] - x[:, :-1, :]
        return diff_x.var(dim=(-2, -1)) + diff_y.var(dim=(-2, -1))

    @staticmethod
    def pool_autocorr_lag1(x):
        """Lag-1 horizontal autocorrelation — spatial smoothness: [B,H,W] → [B].
        high = smooth, low = noisy/high-frequency."""
        fx = x[:, :, 1:].reshape(x.shape[0], -1)
        fy = x[:, :, :-1].reshape(x.shape[0], -1)
        mx = fx.mean(dim=1, keepdim=True); my = fy.mean(dim=1, keepdim=True)
        cov = ((fx - mx) * (fy - my)).mean(dim=1)
        return cov / (fx.std(dim=1) * fy.std(dim=1) + 1e-8)

    # ============================================
    # Spatial Pyramid Pooling (SPP) — kept as utility, NOT registered
    # ============================================

    @staticmethod
    def spp_pool(x):
        """
        Spatial Pyramid Pooling: [batch, H, W] → [batch, 21]

        Pools input at three pyramid levels:
          - 1x1 grid → 1 value  (global summary)
          - 2x2 grid → 4 values (quadrant summaries)
          - 4x4 grid → 16 values (fine-grained spatial layout)

        Total output: 1 + 4 + 16 = 21 dimensions per formula.
        """
        x_4d = x.unsqueeze(1)  # [batch, 1, H, W]

        # Level 1: 1x1 → [batch, 1, 1, 1] → [batch, 1]
        pool_1x1 = F.adaptive_avg_pool2d(x_4d, output_size=1).flatten(1)

        # Level 2: 2x2 → [batch, 1, 2, 2] → [batch, 4]
        pool_2x2 = F.adaptive_avg_pool2d(x_4d, output_size=2).flatten(1)

        # Level 3: 4x4 → [batch, 1, 4, 4] → [batch, 16]
        pool_4x4 = F.adaptive_avg_pool2d(x_4d, output_size=4).flatten(1)

        # Concatenate: [batch, 1+4+16] = [batch, 21]
        return torch.cat([pool_1x1, pool_2x2, pool_4x4], dim=1)


# ============================================
# Learnable Kernel Bank
# ============================================

class SymbolicKernelBank(nn.Module):
    """Learnable convolution kernels registered as symbolic operators.

    Classic kernels (6): initialized to Sobel/Gabor values, fine-tunable.
      - classic_edge_x, classic_edge_y, classic_laplacian (3×3)
      - classic_gabor_0, classic_gabor_45, classic_gabor_90 (7×7)
    Learnable kernels (12): random init, fully data-driven.
      - conv3x3_0..7 (8 × 3×3)
      - conv5x5_0..3 (4 × 5×5)

    During RL discovery (Phase A): all kernels frozen (detached).
    During fine-tuning (Phase B): jointly optimized with classifier.
    """

    def __init__(self, device='cpu'):
        super().__init__()
        # 6 classic kernels — initialized to known filter values
        self.classic_3x3 = nn.Parameter(torch.stack([
            torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]),   # Sobel X
            torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]),   # Sobel Y
            torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]),     # Laplacian
        ], dim=0).unsqueeze(1).to(device))  # [3, 1, 3, 3]

        self.classic_7x7 = nn.Parameter(torch.stack([
            _make_gabor_kernel(0.0, device=device).squeeze(0).squeeze(0),           # Gabor 0°
            _make_gabor_kernel(math.pi / 4, device=device).squeeze(0).squeeze(0),   # Gabor 45°
            _make_gabor_kernel(math.pi / 2, device=device).squeeze(0).squeeze(0),   # Gabor 90°
        ], dim=0).unsqueeze(1).to(device))  # [3, 1, 7, 7]

        # 8 learnable 3×3 + 4 learnable 5×5
        self.conv3x3 = nn.Parameter(torch.randn(8, 1, 3, 3, device=device) * 0.1)
        self.conv5x5 = nn.Parameter(torch.randn(4, 1, 5, 5, device=device) * 0.1)

        # During RL (Phase A), kernels are frozen — use detached weights
        self.finetune_mode = False

        # Names for classic kernels
        self._classic_3x3_names = ['classic_edge_x', 'classic_edge_y', 'classic_laplacian']
        self._classic_7x7_names = ['classic_gabor_0', 'classic_gabor_45', 'classic_gabor_90']

    def get_operator(self, name):
        """Return a callable for the named kernel operator."""
        if name in self._classic_3x3_names:
            idx = self._classic_3x3_names.index(name)
            def op(x, _idx=idx, _self=self):
                kernel = _self.classic_3x3[_idx:_idx + 1]
                if not _self.finetune_mode:
                    kernel = kernel.detach()
                return F.conv2d(x.unsqueeze(1), kernel, padding=1).squeeze(1)
            return op
        elif name in self._classic_7x7_names:
            idx = self._classic_7x7_names.index(name)
            def op(x, _idx=idx, _self=self):
                kernel = _self.classic_7x7[_idx:_idx + 1]
                if not _self.finetune_mode:
                    kernel = kernel.detach()
                return F.conv2d(x.unsqueeze(1), kernel, padding=3).squeeze(1)
            return op
        elif name.startswith('conv3x3_'):
            idx = int(name.split('_')[-1])
            def op(x, _idx=idx, _self=self):
                kernel = _self.conv3x3[_idx:_idx + 1]
                if not _self.finetune_mode:
                    kernel = kernel.detach()
                return F.conv2d(x.unsqueeze(1), kernel, padding=1).squeeze(1)
            return op
        elif name.startswith('conv5x5_'):
            idx = int(name.split('_')[-1])
            def op(x, _idx=idx, _self=self):
                kernel = _self.conv5x5[_idx:_idx + 1]
                if not _self.finetune_mode:
                    kernel = kernel.detach()
                return F.conv2d(x.unsqueeze(1), kernel, padding=2).squeeze(1)
            return op
        raise ValueError(f"Unknown kernel: {name}")

    def register_operators(self, registry):
        """Add all kernel operators to a TENSOR_OPERATORS-style registry."""
        for name in self._classic_3x3_names + self._classic_7x7_names:
            registry[name] = (self.get_operator(name), 1, 'tensor')
        for i in range(self.conv3x3.shape[0]):
            name = f'conv3x3_{i}'
            registry[name] = (self.get_operator(name), 1, 'tensor')
        for i in range(self.conv5x5.shape[0]):
            name = f'conv5x5_{i}'
            registry[name] = (self.get_operator(name), 1, 'tensor')


# ============================================
# Operator Registry
# ============================================

TENSOR_OPERATORS = {
    # Arithmetic (wrapped with _safe_binary for numerical stability + _match_size)
    'add': (TensorOperators.add, 2, 'tensor'),
    'subtract': (TensorOperators.subtract, 2, 'tensor'),
    'multiply': (TensorOperators.multiply, 2, 'tensor'),
    'div': (TensorOperators.div, 2, 'tensor'),

    # Pointwise primitives
    'relu': (TensorOperators.relu, 1, 'tensor'),
    'abs': (TensorOperators.abs_val, 1, 'tensor'),
    'sigmoid': (TensorOperators.sigmoid, 1, 'tensor'),
    'negate': (TensorOperators.negate, 1, 'tensor'),
    'pow2': (TensorOperators.pow2, 1, 'tensor'),
    'sqrt_abs': (TensorOperators.sqrt_abs, 1, 'tensor'),
    'log1p_abs': (TensorOperators.log1p_abs, 1, 'tensor'),

    # Spatial / structural operators
    'blur': (TensorOperators.blur, 1, 'tensor'),
    'blur_7x7': (TensorOperators.blur_7x7, 1, 'tensor'),
    'edge_x': (TensorOperators.edge_x, 1, 'tensor'),
    'edge_y': (TensorOperators.edge_y, 1, 'tensor'),
    'dilate': (TensorOperators.dilate, 1, 'tensor'),
    'laplacian': (TensorOperators.laplacian, 1, 'tensor'),
    'normalize': (TensorOperators.normalize, 1, 'tensor'),

    # Morphological
    'opening': (TensorOperators.opening, 1, 'tensor'),
    'closing': (TensorOperators.closing, 1, 'tensor'),
    'tophat': (TensorOperators.tophat, 1, 'tensor'),

    # Frequency separation
    'high_freq': (TensorOperators.high_freq, 1, 'tensor'),
    'low_freq': (TensorOperators.low_freq, 1, 'tensor'),

    # Geometric
    'flip_h': (TensorOperators.flip_h, 1, 'tensor'),
    'flip_v': (TensorOperators.flip_v, 1, 'tensor'),

    # Multi-scale
    'downsample_2x': (TensorOperators.downsample_2x, 1, 'tensor'),
    'downsample_4x': (TensorOperators.downsample_4x, 1, 'tensor'),
    'stride_pool_4': (TensorOperators.stride_pool_4, 1, 'tensor'),

    # Texture / frequency
    'gabor_0': (TensorOperators.gabor_0, 1, 'tensor'),
    'gabor_45': (TensorOperators.gabor_45, 1, 'tensor'),
    'gabor_90': (TensorOperators.gabor_90, 1, 'tensor'),
    'local_std_5x5': (TensorOperators.local_std_5x5, 1, 'tensor'),

    # Rotation-invariant (v3.2)
    'edge_mag': (TensorOperators.edge_mag, 1, 'tensor'),
    'edge_orient': (TensorOperators.edge_orient, 1, 'tensor'),
    'gabor_mag': (TensorOperators.gabor_mag, 1, 'tensor'),

    # Local structure (v3.2)
    'local_contrast': (TensorOperators.local_contrast, 1, 'tensor'),
    'dog': (TensorOperators.dog, 1, 'tensor'),
    'corner_harris': (TensorOperators.corner_harris, 1, 'tensor'),
    'lbp_like': (TensorOperators.lbp_like, 1, 'tensor'),

    # Second-order (v3.2)
    'edge_xx': (TensorOperators.edge_xx, 1, 'tensor'),
    'edge_yy': (TensorOperators.edge_yy, 1, 'tensor'),

    # --- v3.3 semantic operators (unary, tensor output) ---
    'blob_detector':  (TensorOperators.blob_detector, 1, 'tensor'),
    'symmetry_v':     (TensorOperators.symmetry_v, 1, 'tensor'),
    'symmetry_h':     (TensorOperators.symmetry_h, 1, 'tensor'),
    'contour':        (TensorOperators.contour, 1, 'tensor'),
    'elongation':     (TensorOperators.elongation, 1, 'tensor'),
    'radial_gradient':(TensorOperators.radial_gradient, 1, 'tensor'),

    # --- v3.3 fuzzy logic ---
    'fuzzy_not': (TensorOperators.fuzzy_not, 1, 'tensor'),
    'fuzzy_and': (TensorOperators.fuzzy_and, 2, 'tensor'),
    'fuzzy_or':  (TensorOperators.fuzzy_or, 2, 'tensor'),

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

    # Global pooling (root-only)
    'global_avg_pool': (TensorOperators.global_avg_pool, 1, 'scalar'),
    'global_max_pool': (TensorOperators.global_max_pool, 1, 'scalar'),
    'global_min_pool': (TensorOperators.global_min_pool, 1, 'scalar'),
    'global_std_pool': (TensorOperators.global_std_pool, 1, 'scalar'),
    'global_l2_pool': (TensorOperators.global_l2_pool, 1, 'scalar'),

    # Distribution-aware pooling (root-only)
    'ratio_above_mean': (TensorOperators.ratio_above_mean, 1, 'scalar'),
    'percentile_90': (TensorOperators.percentile_90, 1, 'scalar'),
    'spatial_entropy': (TensorOperators.spatial_entropy, 1, 'scalar'),
    'peak_location_y': (TensorOperators.peak_location_y, 1, 'scalar'),
    'peak_location_x': (TensorOperators.peak_location_x, 1, 'scalar'),

    # Multi-dim pooling (root-only)
    'patch_histogram_4x4': (TensorOperators.patch_histogram_4x4, 1, 'scalar'),

    # Spatial pooling — halves (root-only, position-aware)
    'pool_top_half': (TensorOperators.pool_top_half, 1, 'scalar'),
    'pool_bottom_half': (TensorOperators.pool_bottom_half, 1, 'scalar'),
    'pool_left_half': (TensorOperators.pool_left_half, 1, 'scalar'),
    'pool_right_half': (TensorOperators.pool_right_half, 1, 'scalar'),
    'pool_center': (TensorOperators.pool_center, 1, 'scalar'),
    'pool_corners': (TensorOperators.pool_corners, 1, 'scalar'),

    # Spatial pooling — thirds (root-only)
    'pool_thirds_top': (TensorOperators.pool_thirds_top, 1, 'scalar'),
    'pool_thirds_mid': (TensorOperators.pool_thirds_mid, 1, 'scalar'),
    'pool_thirds_bot': (TensorOperators.pool_thirds_bot, 1, 'scalar'),

    # Spatial pooling — quadrants (root-only)
    'pool_quad_tl': (TensorOperators.pool_quad_tl, 1, 'scalar'),
    'pool_quad_tr': (TensorOperators.pool_quad_tr, 1, 'scalar'),
    'pool_quad_bl': (TensorOperators.pool_quad_bl, 1, 'scalar'),
    'pool_quad_br': (TensorOperators.pool_quad_br, 1, 'scalar'),

    # Spatial pooling — surround (root-only)
    'pool_surround': (TensorOperators.pool_surround, 1, 'scalar'),

    # Variance-aware spatial pooling (v3.2, root-only)
    'std_center': (TensorOperators.std_center, 1, 'scalar'),
    'std_top_half': (TensorOperators.std_top_half, 1, 'scalar'),
    'std_bottom_half': (TensorOperators.std_bottom_half, 1, 'scalar'),
}

# Root-only operators (must output scalar)
ROOT_OPERATORS = {
    # Global pooling
    'global_avg_pool',
    'global_max_pool',
    'global_min_pool',
    'global_std_pool',
    'global_l2_pool',
    # Spatial pooling — halves
    'pool_top_half',
    'pool_bottom_half',
    'pool_left_half',
    'pool_right_half',
    'pool_center',
    'pool_corners',
    # Spatial pooling — thirds
    'pool_thirds_top',
    'pool_thirds_mid',
    'pool_thirds_bot',
    # Spatial pooling — quadrants
    'pool_quad_tl',
    'pool_quad_tr',
    'pool_quad_bl',
    'pool_quad_br',
    # Spatial pooling — surround
    'pool_surround',
    # Distribution-aware pooling
    'ratio_above_mean',
    'percentile_90',
    'spatial_entropy',
    'peak_location_y',
    'peak_location_x',
    # Multi-dim pooling
    'patch_histogram_4x4',
    # Variance-aware spatial pooling (v3.2)
    'std_center',
    'std_top_half',
    'std_bottom_half',
    # High-order statistical pooling (v3.3 Section 1A.2)
    'pool_skewness',
    'pool_kurtosis',
    'pool_q10',
    'pool_q90',
    'pool_iqr',
    'pool_above_mean_ratio',
    'pool_entropy',
    'pool_energy',
    'pool_uniformity',
    'pool_neighbor_diff_var',
    'pool_autocorr_lag1',
}

# Multi-dimensional root operators (output more than 1 value per sample)
MULTI_DIM_OPERATORS = {
    'patch_histogram_4x4': 4,
}
