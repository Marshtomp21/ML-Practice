"""Partition-aware RISE using SLIC regions or a fixed patch grid as switches.

This is an explicit region-based variant of RISE, not the standard soft-mask
RISE estimator. A Bernoulli variable controls every supplied feature region;
the same implementation therefore supports SLIC-RISE and ViT Patch-RISE.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class RegionRiseConfig:
    num_masks: int = 1024
    batch_size: int = 32
    mask_rate: float = 0.5
    seed: int = 0

    def __post_init__(self):
        for name, minimum in (("num_masks", 1), ("batch_size", 1), ("seed", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not np.isfinite(self.mask_rate) or not 0 < self.mask_rate < 1:
            raise ValueError("mask_rate must be finite and strictly between 0 and 1")

    @property
    def keep_probability(self) -> float:
        return 1.0 - float(self.mask_rate)


@dataclass(frozen=True)
class RegionRiseResult:
    region_ids: np.ndarray
    coefficients: np.ndarray
    saliency: np.ndarray
    target: int
    forward_samples: int
    forward_batches: int
    keep_probability: float
    realized_keep_probability: float
    unique_masks: int
    mean_target_logit: float


def sample_region_masks(n_regions: int, config: RegionRiseConfig) -> np.ndarray:
    """Sample deterministic Bernoulli keep masks with a local RNG."""
    if isinstance(n_regions, bool) or not isinstance(n_regions, Integral) or n_regions < 1:
        raise ValueError("n_regions must be a positive integer")
    rng = np.random.default_rng(config.seed)
    return rng.random((config.num_masks, n_regions)) < config.keep_probability


def explain_region_rise(
    image: np.ndarray,
    segments: np.ndarray,
    predict_logits: Callable[[np.ndarray], np.ndarray],
    target: int,
    config: RegionRiseConfig | None = None,
) -> RegionRiseResult:
    """Estimate per-region RISE scores for an arbitrary aligned partition."""
    config = config if config is not None else RegionRiseConfig()
    image, segments = np.asarray(image), np.asarray(segments)
    if (image.ndim != 3 or min(image.shape) < 1
            or not np.issubdtype(image.dtype, np.floating)
            or not np.isfinite(image).all()):
        raise ValueError("image must be a finite floating-point HWC array")
    if segments.shape != image.shape[:2] or not np.issubdtype(segments.dtype, np.integer):
        raise ValueError("segments must be an integer HW array aligned with image")
    if isinstance(target, bool) or not isinstance(target, Integral) or target < 0:
        raise ValueError("target must be a nonnegative integer class index")

    region_ids, inverse = np.unique(segments, return_inverse=True)
    inverse = inverse.reshape(segments.shape)
    masks = sample_region_masks(len(region_ids), config)
    score_sum = np.zeros(len(region_ids), dtype=np.float64)
    target_logit_sum = 0.0
    batches = 0
    n_classes = None
    for start in range(0, config.num_masks, config.batch_size):
        stop = min(start + config.batch_size, config.num_masks)
        keep = masks[start:stop, inverse]
        logits = np.asarray(predict_logits(image[None, ...] * keep[..., None]))
        if (logits.ndim != 2 or logits.shape[0] != stop - start
                or logits.shape[1] <= target or not np.isfinite(logits).all()):
            raise ValueError(
                "predict_logits must return finite (batch, classes) logits including target"
            )
        if n_classes is not None and logits.shape[1] != n_classes:
            raise ValueError("predict_logits changed its class count between batches")
        n_classes = logits.shape[1]
        values = logits[:, target].astype(np.float64, copy=False)
        score_sum += values @ masks[start:stop]
        target_logit_sum += float(values.sum())
        batches += 1

    coefficients = score_sum / (config.num_masks * config.keep_probability)
    saliency = coefficients[inverse].reshape(segments.shape)
    return RegionRiseResult(
        region_ids=region_ids,
        coefficients=coefficients,
        saliency=saliency,
        target=int(target),
        forward_samples=config.num_masks,
        forward_batches=batches,
        keep_probability=config.keep_probability,
        realized_keep_probability=float(masks.mean(dtype=np.float64)),
        unique_masks=int(np.unique(masks, axis=0).shape[0]),
        mean_target_logit=target_logit_sum / config.num_masks,
    )
