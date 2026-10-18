"""Shared baseline validation for region-perturbation attribution methods."""

from __future__ import annotations

import numpy as np


def prepare_baseline(image: np.ndarray, baseline: np.ndarray | None) -> np.ndarray:
    """Return an HWC reference image, defaulting to normalized-space zero."""
    image = np.asarray(image)
    if baseline is None:
        return np.zeros_like(image)
    reference = np.asarray(baseline)
    if reference.ndim == 1 and reference.shape == (image.shape[-1],):
        reference = np.broadcast_to(reference, image.shape)
    if (reference.shape != image.shape
            or not np.issubdtype(reference.dtype, np.floating)
            or not np.isfinite(reference).all()):
        raise ValueError("baseline must be a finite floating HWC image or channel vector")
    return np.asarray(reference, dtype=image.dtype)
