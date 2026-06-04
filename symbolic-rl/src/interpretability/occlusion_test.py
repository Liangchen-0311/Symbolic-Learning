"""v3.3 Section 8.2a — occlusion test.

Mask a region of the image and re-evaluate. If accuracy holds when the BACKGROUND is
occluded (and drops when the OBJECT region is occluded), the model relies on object signal,
not a background/shortcut. The fracture study made this concrete: light (provenance +
occlusion), not formula readability, decided whether 95% was trustworthy.

General + deterministic (no segmentation model needed): we occlude either the central
region (a proxy for the object, since dataset objects are typically centered) or the
periphery (a proxy for background), sweeping the occluded fraction, and report the
accuracy-vs-occlusion curve. A model using genuine object signal degrades fast under
center-occlusion and stays flat under background-occlusion.
"""

import numpy as np
import torch


def region_mask(images, fraction, mode='center', fill='mean'):
    """Return a copy of `images` with a region occluded.

    Args:
        images: [N, 3, H, W] in [0,1].
        fraction: linear side fraction of the occluded square (0..1). For 'center', a
                  fraction f occludes the central f·H × f·W square; for 'background' it
                  KEEPS that central square and occludes everything else.
        mode: 'center' (occlude middle) | 'background' (occlude periphery).
        fill: 'mean' (per-image mean colour) | 'zero'.
    Returns:
        occluded copy [N, 3, H, W].
    """
    out = images.clone()
    N, C, H, W = out.shape
    fh, fw = int(round(fraction * H)), int(round(fraction * W))
    y0, x0 = (H - fh) // 2, (W - fw) // 2
    y1, x1 = y0 + fh, x0 + fw
    if fill == 'mean':
        fillv = out.mean(dim=(-2, -1), keepdim=True)        # [N,C,1,1]
    else:
        fillv = torch.zeros(N, C, 1, 1, dtype=out.dtype, device=out.device)
    if mode == 'center':
        out[:, :, y0:y1, x0:x1] = fillv
    elif mode == 'background':
        keep = torch.zeros_like(out)
        keep[:, :, y0:y1, x0:x1] = 1.0
        out = out * keep + fillv * (1.0 - keep)
    else:
        raise ValueError(f"mode must be 'center' or 'background', got {mode}")
    return out


def occlusion_curve(images, labels, feature_fn, classifier,
                    fractions=(0.0, 0.25, 0.5, 0.75), mode='center'):
    """Accuracy as a function of occluded fraction.

    Args:
        images: [N, 3, H, W] test images in [0,1].
        labels: [N] int labels (tensor or ndarray).
        feature_fn: callable images[N,3,H,W] -> feature matrix X [N, D] (ndarray).
                    Must apply the SAME normalization the classifier was fit with.
        classifier: a fitted object exposing .score(X, y) (Section 6 BaseSymbolicClassifier).
        fractions: occluded side-fractions to sweep (0.0 = clean baseline).
        mode: 'center' (occlude object) | 'background' (occlude background).
    Returns:
        dict: {'mode', 'fractions', 'accuracy', 'baseline', 'retained',
               'verdict'} where retained = acc/baseline and verdict interprets the trend.
    """
    y = labels.numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    accs = []
    for f in fractions:
        imgs = images if f == 0.0 else region_mask(images, f, mode=mode)
        X = feature_fn(imgs)
        accs.append(float(classifier.score(X, y)))
    baseline = accs[0] if accs else 0.0
    retained = [a / baseline if baseline > 0 else 0.0 for a in accs]

    # Interpret: under center-occlusion a trustworthy (object-using) model loses a lot;
    # under background-occlusion it should stay near baseline.
    last_ret = retained[-1] if retained else 1.0
    if mode == 'center':
        verdict = ('object-driven (accuracy collapses when object occluded)'
                   if last_ret < 0.7 else
                   'WARNING: accuracy survives object-occlusion — possible background shortcut')
    else:
        verdict = ('object-driven (accuracy survives background-occlusion)'
                   if last_ret > 0.85 else
                   'WARNING: accuracy drops when only background removed — background dependence')
    return {
        'mode': mode,
        'fractions': list(fractions),
        'accuracy': [round(a, 4) for a in accs],
        'baseline': round(baseline, 4),
        'retained': [round(r, 4) for r in retained],
        'verdict': verdict,
    }
