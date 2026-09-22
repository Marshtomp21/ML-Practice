"""Image LIME baseline using all regions and a weighted Ridge surrogate.

Reference: https://github.com/marcotcr/lime/blob/master/lime/lime_image.py
Project adaptations: caller supplies logits, shared SLIC regions and normalized
inputs; disabled regions are zero, rather than the reference's regional mean.
No pixel attribution postprocessing or visualization is performed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Callable

import numpy as np
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class LimeConfig:
    num_samples: int = 1024  # Includes the unperturbed input in row zero.
    batch_size: int = 32
    mask_rate: float = 0.5
    kernel_width: float = 0.25
    ridge_alpha: float = 1.0
    seed: int = 0

    def __post_init__(self):
        for name, minimum in (("num_samples", 2), ("batch_size", 1), ("seed", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not np.isfinite(self.mask_rate) or not 0 < self.mask_rate < 1:
            raise ValueError("mask_rate must be finite and strictly between 0 and 1")
        if not np.isfinite(self.kernel_width) or self.kernel_width <= 0:
            raise ValueError("kernel_width must be finite and positive")
        if not np.isfinite(self.ridge_alpha) or self.ridge_alpha < 0:
            raise ValueError("ridge_alpha must be finite and nonnegative")


@dataclass(frozen=True)
class LimeResult:
    region_ids: np.ndarray  # Sorted original labels; coefficients have this order.
    coefficients: np.ndarray
    intercept: float
    target: int
    original_logit: float
    local_prediction: float
    weighted_r2: float
    forward_samples: int
    forward_batches: int
    realized_mask_rate: float  # Excludes the forced original row.
    unique_masks: int


def sample_masks(n_regions: int, config: LimeConfig) -> np.ndarray:
    """Bernoulli keep masks; row zero is always the original image.

    mask_rate is an expected region fraction, not an exact pixel fraction.
    Duplicate and empty masks are valid and count toward the forward budget.
    """
    if isinstance(n_regions, bool) or not isinstance(n_regions, Integral) or n_regions < 1:
        raise ValueError("n_regions must be a positive integer")
    rng = np.random.default_rng(config.seed)
    masks = (rng.random((config.num_samples, n_regions)) >= config.mask_rate)
    masks[0] = True
    return masks


def locality_weights(masks: np.ndarray, kernel_width: float) -> np.ndarray:
    """Reference image-LIME cosine distance and sqrt exponential kernel.

    For binary z, cos(z, 1) = sqrt(kept / C); an empty mask has distance 1.
    """
    masks = np.asarray(masks)
    if (masks.ndim != 2 or min(masks.shape) < 1
            or not np.all((masks == 0) | (masks == 1))):
        raise ValueError("masks must be a nonempty binary matrix")
    if not np.isfinite(kernel_width) or kernel_width <= 0:
        raise ValueError("kernel_width must be finite and positive")
    distance = 1.0 - np.sqrt(masks.mean(axis=1))
    return np.exp(-0.5 * (distance / kernel_width) ** 2)


def fit_lime(
    image: np.ndarray,
    segments: np.ndarray,
    predict_logits: Callable[[np.ndarray], np.ndarray],
    target: int,
    config: LimeConfig | None = None,
) -> LimeResult:
    """Fit one LIME surrogate, without producing or saving an attribution map.

    image: finite floating-point normalized HWC input (already resized/cropped).
    segments: shared HW integer labels aligned with image, including noncontiguous
        labels. Segment RGB pixels before normalization, then reuse these labels.
    predict_logits: deterministic inference callable receiving normalized NHWC
        batches, returning finite (N, K) pre-Softmax logits as a NumPy array.
        The caller owns checkpoint loading, device transfer and eval mode.
    target: explicit zero-based correct class; never inferred via an extra call.

    Uses every region (feature_selection='none'), an unpenalized intercept and
    sum_i w_i*(y_i-b-z_i@beta)^2 + alpha*||beta||^2. Exactly num_samples
    inputs are evaluated; no hidden target-selection or endpoint forwards.
    """
    config = config if config is not None else LimeConfig()
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

    region_ids, inverse = np.unique(segments, return_inverse=True)
    inverse = inverse.reshape(segments.shape)
    masks = sample_masks(len(region_ids), config)
    values = np.empty(config.num_samples, dtype=np.float64)
    batches = 0
    n_classes = None
    for start in range(0, config.num_samples, config.batch_size):
        stop = min(start + config.batch_size, config.num_samples)
        keep = masks[start:stop, inverse]
        # Allocate only one batch of images; never materialize B x H x W x C.
        perturbed = image[None, ...] * keep[..., None]
        logits = np.asarray(predict_logits(perturbed))
        if (logits.ndim != 2 or logits.shape[0] != stop - start
                or logits.shape[1] <= target or not np.isfinite(logits).all()):
            raise ValueError("predict_logits must return finite (batch, classes) logits including target")
        if n_classes is not None and logits.shape[1] != n_classes:
            raise ValueError("predict_logits changed its class count between batches")
        n_classes = logits.shape[1]
        values[start:stop] = logits[:, target]
        batches += 1

    weights = locality_weights(masks, config.kernel_width)
    surrogate = Ridge(alpha=config.ridge_alpha, fit_intercept=True, solver="svd")
    surrogate.fit(masks.astype(np.float64), values, sample_weight=weights)
    return LimeResult(
        region_ids=region_ids,
        coefficients=surrogate.coef_.copy(),
        intercept=float(surrogate.intercept_),
        target=int(target),
        original_logit=float(values[0]),
        local_prediction=float(surrogate.predict(masks[:1])[0]),
        weighted_r2=float(surrogate.score(masks, values, sample_weight=weights)),
        forward_samples=config.num_samples,
        forward_batches=batches,
        realized_mask_rate=float(1 - masks[1:].mean()),
        unique_masks=int(np.unique(masks, axis=0).shape[0]),
    )
