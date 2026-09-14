"""Shared SLIC segmentation on the RGB image before model normalization."""
from numbers import Integral

import numpy as np
from skimage.segmentation import slic


def slic_segments(rgb: np.ndarray, n_segments: int = 100,
                  compactness: float = 10.0, sigma: float = 1.0) -> np.ndarray:
    """Accept aligned HWC RGB floats in [0, 1]; return zero-based HW labels.

    n_segments is nominal: always use the actual label count for sampling.
    Resizing/cropping belongs to the model's preprocessing, not this function.
    """
    rgb = np.asarray(rgb)
    if (rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape) < 1
            or not np.issubdtype(rgb.dtype, np.floating)
            or not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1):
        raise ValueError("rgb must be a finite floating-point HWC RGB array in [0, 1]")
    if isinstance(n_segments, bool) or not isinstance(n_segments, Integral) or n_segments < 1:
        raise ValueError("n_segments must be a positive integer")
    if not np.isfinite(compactness) or compactness <= 0:
        raise ValueError("compactness must be finite and positive")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    return slic(rgb, n_segments=n_segments, compactness=compactness,
                sigma=sigma, start_label=0, enforce_connectivity=True,
                convert2lab=True, channel_axis=-1)
