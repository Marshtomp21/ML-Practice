"""KernelSHAP over image superpixels with an explicit forward budget.

The caller supplies the shared SLIC segmentation and target-class logits. A
coalition keeps its selected regions and replaces all other normalized pixels
with zero. Empty and full coalitions are evaluated as budgeted anchor points.

Reference: Lundberg and Lee, NeurIPS 2017, "A Unified Approach to
Interpreting Model Predictions". This implementation is project-specific and
does not depend on the ``shap`` package.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from numbers import Integral
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class KernelShapConfig:
    num_samples: int = 1024  # Total forwards, including empty/full anchors.
    batch_size: int = 32
    ridge_alpha: float = 0.0
    paired_sampling: bool = True
    seed: int = 0

    def __post_init__(self):
        for name, minimum in (("num_samples", 2), ("batch_size", 1), ("seed", 0)):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not np.isfinite(self.ridge_alpha) or self.ridge_alpha < 0:
            raise ValueError("ridge_alpha must be finite and nonnegative")
        if not isinstance(self.paired_sampling, bool):
            raise ValueError("paired_sampling must be a boolean")


@dataclass(frozen=True)
class CoalitionSample:
    masks: np.ndarray
    regression_weights: np.ndarray
    sampling_mode: str


@dataclass(frozen=True)
class KernelShapResult:
    region_ids: np.ndarray  # Sorted original labels; coefficients use this order.
    coefficients: np.ndarray
    base_value: float
    target: int
    original_logit: float
    local_prediction: float
    efficiency_residual: float
    weighted_r2: float
    forward_samples: int
    forward_batches: int
    unique_coalitions: int
    sampling_mode: str
    design_rank: int
    condition_number: float


def shapley_kernel_weight(n_features: int, coalition_size: int) -> float:
    """Return the KernelSHAP weight; endpoints are exact anchors (infinite)."""
    if (
        isinstance(n_features, bool)
        or not isinstance(n_features, Integral)
        or n_features < 1
    ):
        raise ValueError("n_features must be a positive integer")
    if (
        isinstance(coalition_size, bool)
        or not isinstance(coalition_size, Integral)
        or not 0 <= coalition_size <= n_features
    ):
        raise ValueError("coalition_size must be an integer in [0, n_features]")
    if coalition_size in (0, n_features):
        return float("inf")
    return (n_features - 1) / (
        comb(n_features, coalition_size)
        * coalition_size
        * (n_features - coalition_size)
    )


def sample_coalitions(
    n_regions: int, config: KernelShapConfig | None = None
) -> CoalitionSample:
    """Sample binary coalitions while reserving rows 0/1 for empty/full.

    When the budget covers all ``2**C`` coalitions, each is enumerated once.
    Otherwise a coalition size ``s`` is drawn with mass proportional to
    ``(C-1)/(s*(C-s))`` and a subset of that size is uniform. Consequently the
    proposal probability of an individual coalition is proportional to its
    SHAP kernel weight, so sampled interior rows have constant importance
    weight. Optional antithetic complements reduce Monte Carlo variance.
    Duplicates remain valid and count toward the forward budget.
    """
    config = config if config is not None else KernelShapConfig()
    if (
        isinstance(n_regions, bool)
        or not isinstance(n_regions, Integral)
        or n_regions < 1
    ):
        raise ValueError("n_regions must be a positive integer")

    # This avoids constructing 2**C merely to decide whether enumeration fits.
    exact = n_regions <= (config.num_samples.bit_length() - 1)
    if exact:
        rows = [np.zeros(n_regions, dtype=bool), np.ones(n_regions, dtype=bool)]
        for encoded in range(1, (1 << n_regions) - 1):
            rows.append(
                np.fromiter(
                    ((encoded >> index) & 1 for index in range(n_regions)),
                    dtype=bool,
                    count=n_regions,
                )
            )
        masks = np.stack(rows)
        weights = np.zeros(len(masks), dtype=np.float64)
        if len(masks) > 2:
            sizes = masks[2:].sum(axis=1)
            weights[2:] = [
                shapley_kernel_weight(n_regions, int(size)) for size in sizes
            ]
            weights[2:] /= weights[2:].mean()
        return CoalitionSample(masks, weights, "exact")

    masks = np.empty((config.num_samples, n_regions), dtype=bool)
    masks[0] = False
    masks[1] = True
    weights = np.zeros(config.num_samples, dtype=np.float64)
    if n_regions == 1:
        return CoalitionSample(masks[:2], weights[:2], "exact")

    rng = np.random.default_rng(config.seed)
    sizes = np.arange(1, n_regions)
    size_mass = (n_regions - 1) / (sizes * (n_regions - sizes))
    size_probability = size_mass / size_mass.sum()
    row = 2
    while row < config.num_samples:
        size = int(rng.choice(sizes, p=size_probability))
        coalition = np.zeros(n_regions, dtype=bool)
        coalition[rng.choice(n_regions, size=size, replace=False)] = True
        masks[row] = coalition
        weights[row] = 1.0
        row += 1
        if config.paired_sampling and row < config.num_samples:
            masks[row] = ~coalition
            weights[row] = 1.0
            row += 1
    mode = "paired_kernel" if config.paired_sampling else "kernel"
    return CoalitionSample(masks, weights, mode)


def _solve_constrained(
    masks: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    delta: float,
    ridge_alpha: float,
) -> tuple[np.ndarray, int, float, float]:
    """Weighted least squares with the exact efficiency constraint sum(phi)=delta."""
    n_regions = masks.shape[1]
    if n_regions == 1:
        return np.array([delta]), 1, 1.0, 1.0

    interior = weights > 0
    design = masks[interior].astype(np.float64)
    response = values[interior] - values[0]
    sqrt_weight = np.sqrt(weights[interior])
    weighted_design = design * sqrt_weight[:, None]
    weighted_response = response * sqrt_weight

    gram = weighted_design.T @ weighted_design
    gram.flat[:: n_regions + 1] += ridge_alpha
    kkt = np.block(
        [[gram, np.ones((n_regions, 1))], [np.ones((1, n_regions)), np.zeros((1, 1))]]
    )
    rhs = np.r_[weighted_design.T @ weighted_response, delta]
    coefficients = np.linalg.lstsq(kkt, rhs, rcond=None)[0][:-1]
    # Remove the last few ulps of constraint error without another model call.
    coefficients[-1] += delta - coefficients.sum()

    constrained_design = np.vstack([weighted_design, np.ones((1, n_regions))])
    singular_values = np.linalg.svd(constrained_design, compute_uv=False)
    tolerance = (
        (max(constrained_design.shape) * singular_values[0] * np.finfo(np.float64).eps)
        if singular_values.size
        else 0.0
    )
    rank = int(np.sum(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if rank == n_regions
        else float("inf")
    )

    if len(response) == 0:
        weighted_r2 = 1.0
    else:
        residual = response - design @ coefficients
        residual_sum = float(np.sum(weights[interior] * residual**2))
        mean = float(np.average(response, weights=weights[interior]))
        total_sum = float(np.sum(weights[interior] * (response - mean) ** 2))
        weighted_r2 = (
            1.0
            if total_sum <= np.finfo(float).eps and residual_sum <= 1e-24
            else (
                0.0
                if total_sum <= np.finfo(float).eps
                else 1.0 - residual_sum / total_sum
            )
        )
    return coefficients, rank, condition, weighted_r2


def fit_kernel_shap(
    image: np.ndarray,
    segments: np.ndarray,
    predict_logits: Callable[[np.ndarray], np.ndarray],
    target: int,
    config: KernelShapConfig | None = None,
) -> KernelShapResult:
    """Fit KernelSHAP region values without saving a pixel attribution map.

    ``image`` is a finite normalized floating-point HWC input and ``segments``
    is its aligned integer HW SLIC map. ``predict_logits`` receives NHWC
    batches and must return finite pre-Softmax ``(N, K)`` logits. The explicit
    target is normally the correct zero-based class.
    """
    config = config if config is not None else KernelShapConfig()
    image, segments = np.asarray(image), np.asarray(segments)
    if (
        image.ndim != 3
        or min(image.shape) < 1
        or not np.issubdtype(image.dtype, np.floating)
        or not np.isfinite(image).all()
    ):
        raise ValueError("image must be a finite floating-point HWC array")
    if segments.shape != image.shape[:2] or not np.issubdtype(
        segments.dtype, np.integer
    ):
        raise ValueError("segments must be an integer HW array aligned with image")
    if isinstance(target, bool) or not isinstance(target, Integral) or target < 0:
        raise ValueError("target must be a nonnegative integer class index")

    region_ids, inverse = np.unique(segments, return_inverse=True)
    inverse = inverse.reshape(segments.shape)
    sample = sample_coalitions(len(region_ids), config)
    masks = sample.masks
    values = np.empty(len(masks), dtype=np.float64)
    batches = 0
    n_classes = None
    for start in range(0, len(masks), config.batch_size):
        stop = min(start + config.batch_size, len(masks))
        keep = masks[start:stop, inverse]
        perturbed = image[None, ...] * keep[..., None]
        logits = np.asarray(predict_logits(perturbed))
        if (
            logits.ndim != 2
            or logits.shape[0] != stop - start
            or logits.shape[1] <= target
            or not np.isfinite(logits).all()
        ):
            raise ValueError(
                "predict_logits must return finite (batch, classes) logits including target"
            )
        if n_classes is not None and logits.shape[1] != n_classes:
            raise ValueError("predict_logits changed its class count between batches")
        n_classes = logits.shape[1]
        values[start:stop] = logits[:, target]
        batches += 1

    base_value, original_logit = float(values[0]), float(values[1])
    delta = original_logit - base_value
    coefficients, rank, condition, weighted_r2 = _solve_constrained(
        masks, values, sample.regression_weights, delta, config.ridge_alpha
    )
    local_prediction = base_value + float(coefficients.sum())
    return KernelShapResult(
        region_ids=region_ids,
        coefficients=coefficients,
        base_value=base_value,
        target=int(target),
        original_logit=original_logit,
        local_prediction=local_prediction,
        efficiency_residual=local_prediction - original_logit,
        weighted_r2=weighted_r2,
        forward_samples=len(masks),
        forward_batches=batches,
        unique_coalitions=int(np.unique(masks, axis=0).shape[0]),
        sampling_mode=sample.sampling_mode,
        design_rank=rank,
        condition_number=condition,
    )
