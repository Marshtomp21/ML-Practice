"""OrdShap value and position importance over an ordered image patch grid.

Least-squares estimator of Hill et al., "OrdShap: Feature Position Importance
for Sequential Black-Box Models", NeurIPS 2025, following the official
``ordshap/ordshap_v2.py`` (https://github.com/davinhill/ordshap, commit
aaccc3e). For features ``i`` and a position permutation ``sigma`` (slot ``j``
shows feature ``sigma[j]``; feature ``i`` sits at slot ``pos_i``):

1. Coalitions ``S`` are drawn with the KernelSHAP kernel (empty and full
   coalitions are anchors). Each interior coalition is evaluated under
   ``num_permutations`` independent uniformly random permutations.
2. Value importance (OrdShap-VI) solves the KernelSHAP weighted least squares
   on permutation-averaged values ``v(S) = E_sigma f(S, sigma)``, constrained
   to sum to ``E_sigma f(N, sigma) - f(empty)``.
3. Position importance (OrdShap-PI) is the weighted least-squares slope of the
   per-permutation residual ``f(S, sigma) - f(empty) - sum_{i in S} VI_i`` on
   the centred positions ``(pos_i + 1 - (d + 1) / 2) * 1[i in S]``.

Image adaptation (M1 design of optional experiment 5): the image is cut into a
``grid x grid`` lattice of equal patches numbered row-major; patch contents are
features and lattice slots are positions. Patches move whole without scaling,
and absent patches take a uniform baseline, which makes absent features
permutation invariant, so the empty coalition costs a single forward. As a
supplementary 2-D reading, PI is also refit with separate centred row and
column coordinates. This module does not depend on the ``shap`` package.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import factorial
from numbers import Integral
from typing import Callable

import numpy as np

from attribution.baseline import prepare_baseline
from attribution.game_theoretic.kernel_shap import (
    KernelShapConfig,
    _solve_constrained,
    sample_coalitions,
)


@dataclass(frozen=True)
class OrdShapConfig:
    num_coalitions: int = 256  # Coalition rows including empty/full anchors.
    num_permutations: int = 16  # Permutations per interior coalition.
    full_permutations: int = 160  # Permutations of the full input for E f(N).
    batch_size: int = 32
    paired_sampling: bool = True
    seed: int = 0

    def __post_init__(self):
        for name, minimum in (("num_coalitions", 2), ("num_permutations", 1),
                              ("full_permutations", 1), ("batch_size", 1), ("seed", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not isinstance(self.paired_sampling, bool):
            raise ValueError("paired_sampling must be a boolean")


@dataclass(frozen=True)
class OrdShapResult:
    value_importance: np.ndarray  # (d,) OrdShap-VI.
    position_importance: np.ndarray  # (d,) OrdShap-PI on the row-major index.
    position_importance_2d: np.ndarray  # (d, 2) supplementary row/column slopes.
    base_value: float  # f(empty coalition).
    original_logit: float  # f(x) in its original arrangement.
    permuted_full_mean: float  # E_sigma f(N, sigma).
    target: int
    efficiency_residual: float
    forward_samples: int
    forward_batches: int
    unique_coalitions: int
    sampling_mode: str
    permutation_mode: str
    vi_design_rank: int
    pi_design_rank: int


def _permutation_sampler(n_features: int, count: int, rng: np.random.Generator):
    """Return ``count`` permutations, or all ``d!`` when they fit in ``count``."""
    if factorial(n_features) <= count:
        return np.asarray(list(permutations(range(n_features))), dtype=np.int64), "exact"
    return np.argsort(rng.random((count, n_features)), axis=1), "sampled"


def estimate_ordshap(
    evaluate: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_features: int,
    config: OrdShapConfig | None = None,
    *,
    position_coordinates: np.ndarray | None = None,
) -> dict:
    """Run the estimator on ``evaluate(masks, perms) -> values``.

    ``masks`` is an ``(n, d)`` boolean coalition array over features and
    ``perms`` an ``(n, d)`` integer array whose row ``r`` places feature
    ``perms[r, j]`` at slot ``j``. ``position_coordinates`` optionally gives a
    ``(d, k)`` coordinate per slot for the supplementary multi-axis PI fit.
    """
    config = config if config is not None else OrdShapConfig()
    d = int(n_features)
    rng = np.random.default_rng([config.seed, 1])
    sample = sample_coalitions(d, KernelShapConfig(
        num_samples=config.num_coalitions, batch_size=config.batch_size,
        paired_sampling=config.paired_sampling, seed=config.seed))
    masks, weights = sample.masks, sample.regression_weights
    identity = np.arange(d, dtype=np.int64)[None]
    forward = 0

    def run(mask_rows, perm_rows):
        nonlocal forward
        values = np.asarray(evaluate(mask_rows, perm_rows), dtype=np.float64)
        if values.shape != (len(mask_rows),) or not np.isfinite(values).all():
            raise ValueError("evaluate must return one finite value per row")
        forward += len(mask_rows)
        return values

    base_value = float(run(masks[:1], identity)[0])
    original = float(run(masks[1:2], identity)[0])
    full_perms, mode = _permutation_sampler(d, config.full_permutations, rng)
    full_mean = float(run(np.repeat(masks[1:2], len(full_perms), axis=0), full_perms).mean())

    interior = np.flatnonzero(weights > 0)
    perm_blocks, value_blocks = [], []
    for row in interior:
        perms, _ = _permutation_sampler(d, config.num_permutations, rng)
        perm_blocks.append(perms)
        value_blocks.append(run(np.repeat(masks[row:row + 1], len(perms), axis=0), perms))

    averaged = np.zeros(len(masks), dtype=np.float64)
    averaged[0], averaged[1] = base_value, full_mean
    for row, values in zip(interior, value_blocks):
        averaged[row] = values.mean()
    vi, vi_rank, _, _ = _solve_constrained(masks, averaged, weights,
                                           full_mean - base_value, 0.0)

    # Per-permutation rows for the position regression.
    row_masks = np.concatenate([np.repeat(masks[row:row + 1], len(p), axis=0)
                                for row, p in zip(interior, perm_blocks)]) if len(interior) else np.zeros((0, d), bool)
    row_perms = np.concatenate(perm_blocks) if perm_blocks else np.zeros((0, d), np.int64)
    row_values = np.concatenate(value_blocks) if value_blocks else np.zeros(0)
    row_weights = np.concatenate([np.full(len(p), weights[row]) for row, p in
                                  zip(interior, perm_blocks)]) if len(interior) else np.zeros(0)
    slot_of = np.argsort(row_perms, axis=1)  # slot index of each feature
    response = row_values - base_value - row_masks.astype(np.float64) @ vi
    sqrt_weight = np.sqrt(row_weights)

    def slopes(coordinates):
        centred = coordinates - coordinates.mean(axis=0, keepdims=True)  # (d, k)
        design = centred[slot_of] * row_masks[..., None]  # (n, d, k)
        design = design.reshape(len(row_masks), -1)
        solution, _, rank, _ = np.linalg.lstsq(design * sqrt_weight[:, None],
                                               response * sqrt_weight, rcond=None)
        solution[np.abs(solution) < 1e-10] = 0.0
        return solution.reshape(d, coordinates.shape[1]), int(rank)

    pi, pi_rank = slopes(np.arange(1, d + 1, dtype=np.float64)[:, None])
    pi_2d = None
    if position_coordinates is not None:
        pi_2d, _ = slopes(np.asarray(position_coordinates, dtype=np.float64))
    return {
        "value_importance": vi, "position_importance": pi[:, 0],
        "position_importance_2d": pi_2d, "base_value": base_value,
        "original_logit": original, "permuted_full_mean": full_mean,
        "efficiency_residual": float(base_value + vi.sum() - full_mean),
        "forward_samples": forward,
        "unique_coalitions": int(np.unique(masks, axis=0).shape[0]),
        "sampling_mode": sample.sampling_mode, "permutation_mode": mode,
        "vi_design_rank": vi_rank, "pi_design_rank": pi_rank,
    }


def patch_labels(height: int, width: int, grid: int) -> np.ndarray:
    """Row-major ``grid x grid`` labels of equal patches."""
    if height % grid or width % grid:
        raise ValueError("image size must be divisible by the patch grid")
    rows = np.repeat(np.arange(grid), height // grid)
    cols = np.repeat(np.arange(grid), width // grid)
    return (rows[:, None] * grid + cols[None, :]).astype(np.int64)


def compose_patches(patches: np.ndarray, fill: np.ndarray, masks: np.ndarray,
                    perms: np.ndarray, grid: int) -> np.ndarray:
    """Place patch ``perms[r, j]`` (or the uniform fill if absent) at slot ``j``."""
    placed = patches[perms]  # (n, d, ph, pw, C)
    present = np.take_along_axis(masks, perms, axis=1)  # (n, d)
    placed = np.where(present[..., None, None, None], placed, fill)
    n, _, ph, pw, channels = placed.shape
    placed = placed.reshape(n, grid, grid, ph, pw, channels).transpose(0, 1, 3, 2, 4, 5)
    return placed.reshape(n, grid * ph, grid * pw, channels)


def split_patches(image: np.ndarray, grid: int) -> np.ndarray:
    height, width, channels = image.shape
    ph, pw = height // grid, width // grid
    return (image.reshape(grid, ph, grid, pw, channels).transpose(0, 2, 1, 3, 4)
            .reshape(grid * grid, ph, pw, channels))


def fit_ordshap(
    image: np.ndarray,
    grid: int,
    predict_logits: Callable[[np.ndarray], np.ndarray],
    target: int,
    config: OrdShapConfig | None = None,
    *,
    baseline: np.ndarray | None = None,
) -> OrdShapResult:
    """Explain one normalized HWC image on a ``grid x grid`` ordered patch lattice."""
    config = config if config is not None else OrdShapConfig()
    image = np.asarray(image)
    if (image.ndim != 3 or not np.issubdtype(image.dtype, np.floating)
            or not np.isfinite(image).all()):
        raise ValueError("image must be a finite floating-point HWC array")
    if isinstance(grid, bool) or not isinstance(grid, Integral) or grid < 2:
        raise ValueError("grid must be an integer >= 2")
    if isinstance(target, bool) or not isinstance(target, Integral) or target < 0:
        raise ValueError("target must be a nonnegative integer class index")
    reference = prepare_baseline(image, baseline)
    if not np.all(reference == reference[:1, :1]):
        raise ValueError("OrdShap patch moves require a spatially uniform baseline")
    patches = split_patches(image, grid)
    fill = reference[0, 0]
    batches = 0

    def evaluate(masks, perms):
        nonlocal batches
        values = []
        for start in range(0, len(masks), config.batch_size):
            stop = start + config.batch_size
            batch = compose_patches(patches, fill, masks[start:stop], perms[start:stop], grid)
            logits = np.asarray(predict_logits(batch))
            if (logits.ndim != 2 or logits.shape[0] != len(batch)
                    or logits.shape[1] <= target or not np.isfinite(logits).all()):
                raise ValueError("predict_logits must return finite (batch, classes) logits")
            values.append(logits[:, target])
            batches += 1
        return np.concatenate(values)

    coordinates = np.stack(np.divmod(np.arange(grid * grid), grid), axis=1)
    out = estimate_ordshap(evaluate, grid * grid, config, position_coordinates=coordinates)
    return OrdShapResult(
        value_importance=out["value_importance"],
        position_importance=out["position_importance"],
        position_importance_2d=out["position_importance_2d"],
        base_value=out["base_value"], original_logit=out["original_logit"],
        permuted_full_mean=out["permuted_full_mean"], target=int(target),
        efficiency_residual=out["efficiency_residual"],
        forward_samples=out["forward_samples"], forward_batches=batches,
        unique_coalitions=out["unique_coalitions"], sampling_mode=out["sampling_mode"],
        permutation_mode=out["permutation_mode"],
        vi_design_rank=out["vi_design_rank"], pi_design_rank=out["pi_design_rank"],
    )
