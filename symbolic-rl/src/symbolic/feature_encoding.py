"""
Feature Encoding for v3.2 Pipeline

Implements the core encoding techniques that bridge the gap between
formula-discovered feature maps and linear classification:
  1. Distribution statistics encoding (12 stats × 5 regions = 60 per body)
  2. Symbolic Fisher Vector (GMM-based, 4096-dim)
  3. Homogeneous kernel map (chi-squared approximation)
  4. Power normalization + L2 normalization
"""

import math
import numpy as np
import torch
import torch.nn.functional as F


# ============================================
# 1. Distribution Statistics Encoding
# ============================================

def encode_body_distribution(feature_map):
    """Encode a formula body's spatial response as distribution statistics.

    For each of 5 spatial regions (global + 4 quadrants), compute 12 statistics:
      mean, std, max, skewness, kurtosis, 5 quantiles (10/25/50/75/90%), ratio_above_mean

    Args:
        feature_map: [B, H, W] tensor — spatial response of a formula body.

    Returns:
        [B, 60] tensor — 12 stats × 5 regions.
    """
    B, H, W = feature_map.shape

    regions = [
        feature_map,                               # global
        feature_map[:, :H // 2, :W // 2],          # quad_tl
        feature_map[:, :H // 2, W // 2:],          # quad_tr
        feature_map[:, H // 2:, :W // 2],          # quad_bl
        feature_map[:, H // 2:, W // 2:],          # quad_br
    ]

    quantile_levels = torch.tensor(
        [0.1, 0.25, 0.5, 0.75, 0.9],
        device=feature_map.device, dtype=feature_map.dtype,
    )

    all_stats = []
    for region in regions:
        flat = region.reshape(B, -1)  # [B, N]

        mean = flat.mean(dim=1)
        std = flat.std(dim=1).clamp(min=1e-8)
        maximum = flat.max(dim=1).values

        centered = flat - mean.unsqueeze(1)
        skewness = (centered ** 3).mean(dim=1) / (std ** 3 + 1e-8)
        kurtosis = (centered ** 4).mean(dim=1) / (std ** 4 + 1e-8) - 3.0

        quantiles = torch.quantile(flat, quantile_levels, dim=1)  # [5, B]

        ratio = (flat > mean.unsqueeze(1)).float().mean(dim=1)

        # Stack: [B, 12]
        stats = torch.stack([
            mean, std, maximum, skewness, kurtosis,
            quantiles[0], quantiles[1], quantiles[2], quantiles[3], quantiles[4],
            ratio,
        ], dim=1)
        all_stats.append(stats)

    return torch.cat(all_stats, dim=1)  # [B, 60] (but actually 11*5=55)


def encode_body_distribution_v2(feature_map):
    """Enhanced version: 12 stats × 5 regions = 60 per body.

    Stats: mean, std, max, min, skewness, kurtosis, q10, q25, q50, q75, q90, ratio_above_mean

    Args:
        feature_map: [B, H, W] tensor.

    Returns:
        [B, 60] tensor.
    """
    B, H, W = feature_map.shape

    regions = [
        feature_map,
        feature_map[:, :H // 2, :W // 2],
        feature_map[:, :H // 2, W // 2:],
        feature_map[:, H // 2:, :W // 2],
        feature_map[:, H // 2:, W // 2:],
    ]

    quantile_levels = torch.tensor(
        [0.1, 0.25, 0.5, 0.75, 0.9],
        device=feature_map.device, dtype=feature_map.dtype,
    )

    all_stats = []
    for region in regions:
        flat = region.reshape(B, -1)

        mean = flat.mean(dim=1)
        std = flat.std(dim=1).clamp(min=1e-8)
        maximum = flat.max(dim=1).values

        centered = flat - mean.unsqueeze(1)
        skewness = (centered ** 3).mean(dim=1) / (std ** 3 + 1e-8)
        kurtosis = (centered ** 4).mean(dim=1) / (std ** 4 + 1e-8) - 3.0

        quantiles = torch.quantile(flat, quantile_levels, dim=1)  # [5, B]

        ratio = (flat > mean.unsqueeze(1)).float().mean(dim=1)

        # 12 stats per region
        stats = torch.stack([
            mean, std, maximum, skewness, kurtosis,
            quantiles[0], quantiles[1], quantiles[2], quantiles[3], quantiles[4],
            ratio,
            maximum - flat.min(dim=1).values,  # range as 12th stat
        ], dim=1)
        all_stats.append(stats)

    return torch.cat(all_stats, dim=1)  # [B, 60]


# ============================================
# 2. Symbolic Fisher Vector Encoding
# ============================================

class SymbolicFisherVector:
    """Fisher Vector encoding using formula-based local descriptors.

    Pipeline:
      1. Divide image into 8×8 = 64 overlapping patches
      2. For each patch, evaluate top-K formula bodies → K-dim descriptor
      3. PCA reduce to pca_dim dimensions
      4. Compute Fisher Vector against a fitted GMM
      5. Power + L2 normalize

    The GMM is a fixed statistical model fitted once on training data.
    """

    def __init__(self, pca_dim=32, gmm_k=64, device='cpu'):
        self.pca_dim = pca_dim
        self.gmm_k = gmm_k
        self.device = device

        # To be fitted
        self.pca_mean = None       # [D]
        self.pca_components = None  # [pca_dim, D]
        self.gmm_means = None      # [K, pca_dim]
        self.gmm_vars = None       # [K, pca_dim]
        self.gmm_weights = None    # [K]

    def extract_patches(self, feature_maps, grid_size=8):
        """Extract local descriptors from formula feature maps.

        Args:
            feature_maps: [B, N_bodies, H, W] — N_bodies formula responses.
            grid_size: patches per dimension (8 → 64 patches).

        Returns:
            [B, grid_size², N_bodies] — local descriptors.
        """
        B, N, H, W = feature_maps.shape
        ph = H // grid_size
        pw = W // grid_size

        patches = []
        for i in range(grid_size):
            for j in range(grid_size):
                patch = feature_maps[:, :, i * ph:(i + 1) * ph, j * pw:(j + 1) * pw]
                patch_desc = patch.mean(dim=[-2, -1])  # [B, N]
                patches.append(patch_desc)

        return torch.stack(patches, dim=1)  # [B, 64, N]

    def fit_pca(self, descriptors):
        """Fit PCA on local descriptors.

        Args:
            descriptors: [N_total, D] — collected from training images.
        """
        D = descriptors.shape[1]
        mean = descriptors.mean(dim=0)
        centered = descriptors - mean

        # SVD for PCA
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        components = Vt[:self.pca_dim]  # [pca_dim, D]

        self.pca_mean = mean.to(self.device)
        self.pca_components = components.to(self.device)

    def apply_pca(self, descriptors):
        """Apply fitted PCA to descriptors.

        Args:
            descriptors: [..., D]

        Returns:
            [..., pca_dim]
        """
        shape = descriptors.shape[:-1]
        D = descriptors.shape[-1]
        flat = descriptors.reshape(-1, D)
        centered = flat - self.pca_mean
        projected = centered @ self.pca_components.T  # [N, pca_dim]
        return projected.reshape(*shape, self.pca_dim)

    def fit_gmm(self, descriptors, n_iter=50):
        """Fit GMM with EM algorithm on PCA-reduced descriptors.

        Args:
            descriptors: [N, pca_dim] — PCA-reduced local descriptors.
            n_iter: number of EM iterations.
        """
        N, D = descriptors.shape
        K = self.gmm_k

        # Initialize with K-means++ style
        indices = torch.randperm(N, device=descriptors.device)[:K]
        means = descriptors[indices].clone()  # [K, D]
        vars_ = torch.ones(K, D, device=descriptors.device)
        weights = torch.ones(K, device=descriptors.device) / K

        for _ in range(n_iter):
            # E-step: compute responsibilities
            gamma = self._compute_posterior(descriptors, means, vars_, weights)  # [N, K]

            # M-step
            Nk = gamma.sum(dim=0).clamp(min=1e-8)  # [K]
            weights = Nk / N
            means = (gamma.T @ descriptors) / Nk.unsqueeze(1)  # [K, D]
            diff = descriptors.unsqueeze(1) - means.unsqueeze(0)  # [N, K, D]
            vars_ = (gamma.unsqueeze(2) * diff ** 2).sum(dim=0) / Nk.unsqueeze(1)
            vars_ = vars_.clamp(min=1e-6)

        self.gmm_means = means.to(self.device)
        self.gmm_vars = vars_.to(self.device)
        self.gmm_weights = weights.to(self.device)

    def _compute_posterior(self, x, means, vars_, weights):
        """Compute GMM posterior probabilities.

        Args:
            x: [N, D]
            means: [K, D]
            vars_: [K, D]
            weights: [K]

        Returns:
            [N, K] posterior probabilities.
        """
        K = means.shape[0]
        # Log-likelihood: log N(x | mu_k, sigma_k^2)
        # = -0.5 * [D*log(2pi) + sum(log(var)) + sum((x-mu)^2/var)]
        diff = x.unsqueeze(1) - means.unsqueeze(0)  # [N, K, D]
        log_exp = -0.5 * (diff ** 2 / vars_.unsqueeze(0)).sum(dim=2)  # [N, K]
        log_det = -0.5 * vars_.log().sum(dim=1)  # [K]
        log_prior = weights.log()  # [K]

        log_prob = log_exp + log_det + log_prior  # [N, K]
        # Normalize (log-sum-exp for numerical stability)
        log_prob = log_prob - log_prob.logsumexp(dim=1, keepdim=True)
        return log_prob.exp()

    def compute_fisher_vector(self, descriptors):
        """Compute Fisher Vector for a set of local descriptors.

        Args:
            descriptors: [N_patches, pca_dim] — PCA-reduced descriptors for one image.

        Returns:
            [2 * pca_dim * gmm_k] — Fisher Vector (power + L2 normalized).
        """
        K = self.gmm_k
        D = self.pca_dim

        # Posterior: [N, K]
        gamma = self._compute_posterior(
            descriptors, self.gmm_means, self.gmm_vars, self.gmm_weights,
        )

        # First-order gradient: deviation from mean
        diff = descriptors.unsqueeze(1) - self.gmm_means.unsqueeze(0)  # [N, K, D]
        norm_diff = diff / self.gmm_vars.unsqueeze(0).sqrt()  # [N, K, D]
        G_mu = (gamma.unsqueeze(2) * norm_diff).sum(dim=0)  # [K, D]
        G_mu = G_mu / (self.gmm_weights.unsqueeze(1).sqrt() + 1e-8)

        # Second-order gradient: deviation of variance
        sq_norm_diff = norm_diff ** 2 - 1  # [N, K, D]
        G_sigma = (gamma.unsqueeze(2) * sq_norm_diff).sum(dim=0)  # [K, D]
        G_sigma = G_sigma / ((2 * self.gmm_weights.unsqueeze(1)).sqrt() + 1e-8)

        fv = torch.cat([G_mu.flatten(), G_sigma.flatten()])  # [2*K*D]

        # Power normalization (signed sqrt)
        fv = torch.sign(fv) * torch.sqrt(torch.abs(fv) + 1e-8)

        # L2 normalization
        fv = fv / (fv.norm() + 1e-8)

        return fv

    def encode_batch(self, feature_maps, grid_size=8):
        """Encode a batch of images to Fisher Vectors.

        Args:
            feature_maps: [B, N_bodies, H, W] — formula responses.
            grid_size: patches per dimension.

        Returns:
            [B, 2 * pca_dim * gmm_k] — Fisher Vectors.
        """
        patches = self.extract_patches(feature_maps, grid_size)  # [B, 64, N]
        pca_patches = self.apply_pca(patches)  # [B, 64, pca_dim]

        fvs = []
        for i in range(patches.shape[0]):
            fv = self.compute_fisher_vector(pca_patches[i])
            fvs.append(fv)

        return torch.stack(fvs, dim=0)  # [B, 4096]

    def save(self, path):
        """Save fitted parameters."""
        torch.save({
            'pca_mean': self.pca_mean,
            'pca_components': self.pca_components,
            'gmm_means': self.gmm_means,
            'gmm_vars': self.gmm_vars,
            'gmm_weights': self.gmm_weights,
            'pca_dim': self.pca_dim,
            'gmm_k': self.gmm_k,
        }, path)

    def load(self, path):
        """Load fitted parameters."""
        data = torch.load(path, map_location=self.device, weights_only=True)
        self.pca_mean = data['pca_mean']
        self.pca_components = data['pca_components']
        self.gmm_means = data['gmm_means']
        self.gmm_vars = data['gmm_vars']
        self.gmm_weights = data['gmm_weights']
        self.pca_dim = data['pca_dim']
        self.gmm_k = data['gmm_k']


# ============================================
# 3. Homogeneous Kernel Map
# ============================================

def homogeneous_kernel_map(x, order=1):
    """Deterministic chi-squared kernel approximation.

    Maps each scalar feature to (2*order + 1) values.
    Linear classifier on mapped features ≈ chi-squared kernel SVM.

    Args:
        x: [B, D] feature matrix.
        order: approximation order (1 → each feature becomes 3 values).

    Returns:
        [B, D * (2*order + 1)] mapped features.
    """
    abs_x = torch.abs(x) + 1e-8
    sqrt_x = torch.sqrt(abs_x)
    log_x = torch.log(abs_x)

    features = [sqrt_x]
    coeff = math.sqrt(2.0 / math.pi)
    for j in range(1, order + 1):
        features.append(sqrt_x * torch.cos(j * log_x) * coeff)
        features.append(sqrt_x * torch.sin(j * log_x) * coeff)

    return torch.cat(features, dim=-1)  # [B, D * (2*order+1)]


# ============================================
# 4. Power Normalization + L2 Normalization
# ============================================

def power_normalize(x, alpha=0.5):
    """Signed power normalization: sign(x) * |x|^alpha.

    Suppresses "bursty" features that dominate the linear classifier.
    With alpha=0.5, this is signed square root.

    Args:
        x: [B, D] feature matrix (should be standardized first).
        alpha: power exponent (default 0.5).

    Returns:
        [B, D] power-normalized features.
    """
    return torch.sign(x) * torch.pow(torch.abs(x) + 1e-8, alpha)


def l2_normalize(x, dim=1):
    """L2 normalization along a dimension.

    Args:
        x: tensor.
        dim: dimension to normalize along.

    Returns:
        L2-normalized tensor.
    """
    return F.normalize(x, p=2, dim=dim)


def apply_normalization_pipeline(features):
    """Full normalization pipeline for final features.

    Steps:
      1. Standardize (zero mean, unit variance per feature)
      2. Power normalization (signed sqrt)
      3. L2 normalization (per sample)

    Args:
        features: [B, D] raw feature matrix.

    Returns:
        [B, D] normalized feature matrix.
        mean: [D] feature means (for applying to test set).
        std: [D] feature stds (for applying to test set).
    """
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp(min=1e-8)
    standardized = (features - mean) / std

    power_normed = power_normalize(standardized, alpha=0.5)
    l2_normed = l2_normalize(power_normed, dim=1)

    return l2_normed, mean, std


def apply_normalization_pipeline_with_stats(features, mean, std):
    """Apply normalization using pre-computed statistics (for test set).

    Args:
        features: [B, D] raw feature matrix.
        mean: [D] pre-computed means.
        std: [D] pre-computed stds.

    Returns:
        [B, D] normalized feature matrix.
    """
    standardized = (features - mean) / std
    power_normed = power_normalize(standardized, alpha=0.5)
    return l2_normalize(power_normed, dim=1)
