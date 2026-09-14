"""Leave-one-region-out attribution using the same input contract as LIME.

Each region is independently replaced with zero in normalized input space.
No pixel attribution postprocessing, model loading or visualization is done here.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class AblationConfig:
    batch_size: int = 32

    def __post_init__(self):
        if (isinstance(self.batch_size, bool)
                or not isinstance(self.batch_size, Integral) or self.batch_size < 1):
            raise ValueError("batch_size must be an integer >= 1")


@dataclass(frozen=True)
class AblationResult:
    region_ids: np.ndarray  # Sorted original labels, as in LimeResult.
    coefficients: np.ndarray  # Signed original_logit - ablated_logits.
    target: int
    original_logit: float
    ablated_logits: np.ndarray  # Same order as region_ids and coefficients.
    forward_samples: int
    forward_batches: int
    used_cached_original: bool


def fit_ablation(
    image: np.ndarray,
    segments: np.ndarray,
    predict_logits: Callable[[np.ndarray], np.ndarray],
    target: int,
    config: AblationConfig | None = None,
    *,
    original_logit: float | None = None,
) -> AblationResult:
    """Compute f_target(image) - f_target(image with region i zeroed).

    image: finite floating-point normalized HWC input, already resized/cropped.
    segments: shared integer HW labels aligned with image; noncontiguous and
        negative labels are supported. Reuse SLIC from pre-normalization RGB.
    predict_logits: deterministic callable taking normalized NumPy NHWC batches
        and returning finite real (N, K) pre-Softmax logits. The caller owns
        checkpoint loading, device transfer, eval mode and disabling gradients.
    target: explicit zero-based correct class, never selected by extra inference.
    original_logit: optional public cached score for this exact input, target,
        model and preprocessing; the caller is responsible for cache validity.

    With C actual regions, evaluate C+1 inputs by default (original first), or C
    when reusing original_logit. Count inputs separately from batches and never
    pad to a sampling budget. Every perturbation starts from the original image,
    not the previously ablated one. Preserve negative contributions; these are
    deletion effects, not Shapley values and need not sum to the total effect.
    """
    config = config if config is not None else AblationConfig()
    image, segments = np.asarray(image), np.asarray(segments)
    if (image.ndim != 3 or min(image.shape) < 1
            or not np.issubdtype(image.dtype, np.floating)
            or not np.isfinite(image).all()):
        raise ValueError("image must be a finite floating-point HWC array")
    if (segments.shape != image.shape[:2]
            or not np.issubdtype(segments.dtype, np.integer)):
        raise ValueError("segments must be an integer HW array aligned with image")
    if isinstance(target, bool) or not isinstance(target, Integral) or target < 0:
        raise ValueError("target must be a nonnegative integer class index")
    if original_logit is not None and (
        isinstance(original_logit, bool) or not isinstance(original_logit, Real)
        or not np.isfinite(original_logit)
    ):
        raise ValueError("original_logit must be a finite real scalar or None")

    region_ids, inverse = np.unique(segments, return_inverse=True)
    inverse = inverse.reshape(segments.shape)
    used_cached_original = original_logit is not None
    offset = int(not used_cached_original)
    forward_samples = len(region_ids) + offset
    values = np.empty(forward_samples, dtype=np.float64)
    batches = 0
    n_classes = None
    for start in range(0, forward_samples, config.batch_size):
        stop = min(start + config.batch_size, forward_samples)
        # Internal region index -1 denotes the original (keep every region).
        removed = np.arange(start - offset, stop - offset)
        keep = inverse[None, ...] != removed[:, None, None]
        # Only materialize one image batch, not all C perturbed images.
        perturbed = image[None, ...] * keep[..., None]
        logits = np.asarray(predict_logits(perturbed))
        if (logits.ndim != 2 or logits.shape[0] != stop - start
                or logits.shape[1] <= target
                or not (np.issubdtype(logits.dtype, np.floating)
                        or np.issubdtype(logits.dtype, np.integer))
                or not np.isfinite(logits).all()):
            raise ValueError("predict_logits must return finite real (batch, classes) logits including target")
        if n_classes is not None and logits.shape[1] != n_classes:
            raise ValueError("predict_logits changed its class count between batches")
        n_classes = logits.shape[1]
        values[start:stop] = logits[:, target]
        batches += 1

    original = float(original_logit) if used_cached_original else float(values[0])
    ablated_logits = values[offset:].copy()
    return AblationResult(
        region_ids=region_ids,
        coefficients=original - ablated_logits,
        target=int(target),
        original_logit=original,
        ablated_logits=ablated_logits,
        forward_samples=forward_samples,
        forward_batches=batches,
        used_cached_original=used_cached_original,
    )
