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


def regular_grid_segments(image_shape: tuple[int, int], patch_size: int = 16) -> np.ndarray:
    """Return row-major labels for a non-overlapping square patch grid.

    ViT-B/16 receives 224 x 224 inputs, so ``patch_size=16`` produces the
    model-aligned 14 x 14 = 196 feature partition. Requiring exact divisibility
    avoids silently creating partial boundary features that do not correspond
    to ViT tokens.
    """
    if (not isinstance(image_shape, (tuple, list)) or len(image_shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, Integral)
                   or value < 1 for value in image_shape)):
        raise ValueError("image_shape must contain two positive integers")
    if isinstance(patch_size, bool) or not isinstance(patch_size, Integral) or patch_size < 1:
        raise ValueError("patch_size must be a positive integer")
    height, width = map(int, image_shape)
    if height % patch_size or width % patch_size:
        raise ValueError("image dimensions must be divisible by patch_size")
    rows = np.arange(height, dtype=np.int64) // int(patch_size)
    columns = np.arange(width, dtype=np.int64) // int(patch_size)
    grid_width = width // int(patch_size)
    return rows[:, None] * grid_width + columns[None, :]
