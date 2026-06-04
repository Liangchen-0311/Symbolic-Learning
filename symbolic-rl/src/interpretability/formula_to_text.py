"""v3.3 Section 8.2d — deterministic formula→text translation.

A rule-based (NOT LLM, NO external knowledge) translator mapping an RPN token chain to a
structured English phrase, e.g.
    "I_R edge_x blur pool_center" -> "smoothed horizontal-edge strength of the red channel,
                                      central region"
For long formulas it produces nested phrasing — verbose but consistent and fully traceable.
Reuses the channel/operator names already used in the interpretability slides.

This recovers a *consistent* description for every formula (including unreadable long ones),
which is the level-2 (per-unit semantics) complement to probe images (Section 8.2c).
"""

from src.symbolic.tensor_operators import TENSOR_OPERATORS, ROOT_OPERATORS

# --- Terminal channel phrases (RGB + grayscale + HSV + ratios + opponent + prior) ---
TERMINAL_PHRASES = {
    'I_R': 'the red channel', 'I_G': 'the green channel', 'I_B': 'the blue channel',
    'I_GRAY': 'the grayscale image', 'I_H': 'the hue channel', 'I_S': 'the saturation channel',
    'I_r': 'the red-ratio (illumination-invariant)', 'I_g': 'the green-ratio (illumination-invariant)',
    'I_RG': 'the red-minus-green opponent channel', 'I_BY': 'the blue-minus-yellow opponent channel',
    # Section 1A.0 prior terminals
    'I_EDGE': 'the edge-magnitude prior (Sobel)', 'I_FREQ': 'the high-frequency detail prior',
    'I_LAPLACIAN': 'the laplacian edge prior',
}

# --- Unary operator phrase templates ("{X}" = the child phrase) ---
UNARY_PHRASES = {
    'relu': 'the positive part of {X}', 'abs': 'the magnitude of {X}',
    'sigmoid': 'the soft-gated {X}', 'negate': 'the negation of {X}',
    'pow2': 'the squared {X}', 'sqrt_abs': 'the square-root magnitude of {X}',
    'log1p_abs': 'the log-magnitude of {X}', 'normalize': 'the normalized {X}',
    'blur': 'the smoothed {X}', 'blur_7x7': 'the heavily-smoothed {X}',
    'edge_x': 'the horizontal-edge strength of {X}', 'edge_y': 'the vertical-edge strength of {X}',
    'edge_xx': 'the horizontal second-derivative of {X}', 'edge_yy': 'the vertical second-derivative of {X}',
    'edge_mag': 'the edge magnitude of {X}', 'edge_orient': 'the edge orientation of {X}',
    'laplacian': 'the laplacian of {X}', 'dilate': 'the dilated {X}',
    'opening': 'the morphologically-opened {X}', 'closing': 'the morphologically-closed {X}',
    'tophat': 'the top-hat (bright-detail) of {X}',
    'high_freq': 'the high-frequency content of {X}', 'low_freq': 'the low-frequency content of {X}',
    'flip_h': 'the horizontally-flipped {X}', 'flip_v': 'the vertically-flipped {X}',
    'downsample_2x': 'the 2x-downsampled {X}', 'downsample_4x': 'the 4x-downsampled {X}',
    'stride_pool_4': 'the strided-pooled {X}',
    'gabor_0': 'the 0°-Gabor response of {X}', 'gabor_45': 'the 45°-Gabor response of {X}',
    'gabor_90': 'the 90°-Gabor response of {X}', 'gabor_mag': 'the Gabor magnitude of {X}',
    'local_std_5x5': 'the local texture (5x5 std) of {X}', 'local_contrast': 'the local contrast of {X}',
    'dog': 'the difference-of-Gaussians of {X}', 'corner_harris': 'the corner response of {X}',
    'lbp_like': 'the local-binary-pattern texture of {X}',
    # v3.3 semantic
    'blob_detector': 'the multi-scale blob response of {X}',
    'symmetry_v': 'the left-right symmetry of {X}', 'symmetry_h': 'the top-bottom symmetry of {X}',
    'contour': 'the contour/boundary strength of {X}', 'elongation': 'the local elongation of {X}',
    'radial_gradient': 'the radial gradient (about center) of {X}',
    # v3.3 directional lines
    'line_h': 'the horizontal-line response of {X}', 'line_v': 'the vertical-line response of {X}',
    'line_diag45': 'the 45°-line response of {X}', 'line_diag135': 'the 135°-line response of {X}',
    'fuzzy_not': 'the fuzzy-NOT of {X}',
}

# --- Binary operator phrase templates ("{X}", "{Y}" = the two child phrases) ---
BINARY_PHRASES = {
    'add': '({X} plus {Y})', 'subtract': '({X} minus {Y})',
    'multiply': '({X} times {Y})', 'div': '({X} divided by {Y})',
    'fuzzy_and': '({X} AND {Y})', 'fuzzy_or': '({X} OR {Y})',
}

# --- Pooling (ROOT) operator phrase suffixes ("{X}" = the child phrase) ---
POOL_PHRASES = {
    'global_avg_pool': 'the average of {X}', 'global_max_pool': 'the maximum of {X}',
    'global_min_pool': 'the minimum of {X}', 'global_std_pool': 'the spread (std) of {X}',
    'global_l2_pool': 'the energy (L2) of {X}',
    'pool_top_half': 'the average over the top half of {X}',
    'pool_bottom_half': 'the average over the bottom half of {X}',
    'pool_left_half': 'the average over the left half of {X}',
    'pool_right_half': 'the average over the right half of {X}',
    'pool_center': 'the average over the central region of {X}',
    'pool_corners': 'the average over the corners of {X}',
    'pool_thirds_top': 'the average over the top third of {X}',
    'pool_thirds_mid': 'the average over the central band of {X}',
    'pool_thirds_bot': 'the average over the bottom third of {X}',
    'pool_quad_tl': 'the average over the top-left quadrant of {X}',
    'pool_quad_tr': 'the average over the top-right quadrant of {X}',
    'pool_quad_bl': 'the average over the bottom-left quadrant of {X}',
    'pool_quad_br': 'the average over the bottom-right quadrant of {X}',
    'pool_surround': 'the periphery-minus-center contrast of {X}',
    'std_center': 'the texture (std) of the central region of {X}',
    'std_top_half': 'the texture (std) of the top half of {X}',
    'std_bottom_half': 'the texture (std) of the bottom half of {X}',
    'ratio_above_mean': 'the bright-area fraction of {X}',
    'percentile_90': 'the 90th-percentile value of {X}',
    'spatial_entropy': 'the spatial entropy of {X}',
    'peak_location_y': 'the vertical position of the peak of {X}',
    'peak_location_x': 'the horizontal position of the peak of {X}',
    'patch_histogram_4x4': 'the 4x4 spatial activation histogram of {X}',
    # v3.3 statistical
    'pool_skewness': 'the distribution skewness of {X}', 'pool_kurtosis': 'the distribution kurtosis of {X}',
    'pool_q10': 'the 10th-percentile value of {X}', 'pool_q90': 'the 90th-percentile value of {X}',
    'pool_iqr': 'the inter-quartile spread of {X}', 'pool_above_mean_ratio': 'the bright-area fraction of {X}',
    'pool_entropy': 'the intensity entropy of {X}', 'pool_energy': 'the energy of {X}',
    'pool_uniformity': 'the intensity uniformity of {X}',
    'pool_neighbor_diff_var': 'the local-contrast variance of {X}',
    'pool_autocorr_lag1': 'the spatial smoothness (autocorrelation) of {X}',
    # v3.3 asymmetry
    'pool_lr_asymmetry': 'the left-right asymmetry of {X}',
    'pool_tb_asymmetry': 'the top-bottom asymmetry of {X}',
}


def _humanize(token):
    """Fallback phrase for any token without an explicit mapping (keeps coverage total)."""
    return token.replace('_', ' ')


# Unified lookup used by the test's coverage check.
TOKEN_PHRASES = {}
TOKEN_PHRASES.update({k: v for k, v in TERMINAL_PHRASES.items()})
TOKEN_PHRASES.update({k: v for k, v in UNARY_PHRASES.items()})
TOKEN_PHRASES.update({k: v for k, v in BINARY_PHRASES.items()})
TOKEN_PHRASES.update({k: v for k, v in POOL_PHRASES.items()})


def formula_to_text(formula_str, extra_terminals=None):
    """Translate an RPN formula string to a structured English phrase.

    Args:
        formula_str: space-separated RPN token chain (e.g. "I_R edge_x pool_center").
        extra_terminals: optional dict {token: phrase} for L1_*/custom terminals.
    Returns:
        str: a nested, deterministic plain-English reading. Unknown tokens fall back to a
        humanized token name, so the function is total (never raises on vocab tokens).
    """
    terminals = dict(TERMINAL_PHRASES)
    if extra_terminals:
        terminals.update(extra_terminals)
    tokens = formula_str.strip().split()
    stack = []
    for tok in tokens:
        if tok in terminals:
            stack.append(terminals[tok])
        elif tok.startswith('I_') or tok.startswith('L1_'):
            stack.append(_humanize(tok))                       # unmapped terminal
        elif tok in ROOT_OPERATORS:
            child = stack.pop() if stack else 'the input'
            tmpl = POOL_PHRASES.get(tok, 'the ' + _humanize(tok) + ' of {X}')
            stack.append(tmpl.format(X=child))
        elif tok in TENSOR_OPERATORS:
            _, arity, _ = TENSOR_OPERATORS[tok]
            if arity == 1:
                child = stack.pop() if stack else 'the input'
                tmpl = UNARY_PHRASES.get(tok, 'the ' + _humanize(tok) + ' of {X}')
                stack.append(tmpl.format(X=child))
            else:
                y = stack.pop() if stack else 'the input'
                x = stack.pop() if stack else 'the input'
                tmpl = BINARY_PHRASES.get(tok, '(' + '{X} ' + _humanize(tok) + ' {Y})')
                stack.append(tmpl.format(X=x, Y=y))
        else:
            stack.append(_humanize(tok))
    if not stack:
        return '(empty formula)'
    return stack[-1] if len(stack) == 1 else ' ; '.join(stack)


if __name__ == '__main__':
    for f in ["I_R edge_x blur pool_center",
              "I_R edge_x I_G blur subtract lbp_like flip_h fuzzy_and normalize pool_skewness",
              "I_EDGE pool_lr_asymmetry"]:
        print(f"{f}\n  -> {formula_to_text(f)}\n")
