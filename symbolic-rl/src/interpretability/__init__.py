"""v3.3 Section 8 — interpretability & trust-verification tools.

The fracture study's lesson: interpretability's real payoff is *verifying whether high
accuracy is trustworthy* (occlusion / cross-source / probe), not pretty formulas. These
tools make that systematic. All are deterministic and use NO external pretrained model
(Hard Constraint #5).
"""

from .formula_to_text import formula_to_text, TOKEN_PHRASES
from .probe_images import formula_scalar_values, top_bottom_activators, save_probe_montage
from .occlusion_test import occlusion_curve, region_mask
from .cross_source_test import cross_source_eval

__all__ = [
    'formula_to_text', 'TOKEN_PHRASES',
    'formula_scalar_values', 'top_bottom_activators', 'save_probe_montage',
    'occlusion_curve', 'region_mask',
    'cross_source_eval',
]
