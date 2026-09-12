"""RISE image attribution with streamed random-mask evaluation.

Reference: Petsiuk, Das and Saenko, BMVC 2018, and the authors' code at
https://github.com/eclique/RISE/blob/master/explanations.py.

Project adaptations: the caller supplies normalized NHWC inputs and target
class pre-Softmax logits. Multiplication by a mask therefore replaces hidden
pixels with the project's zero baseline in normalized input space. The native
low-resolution random-grid masks are retained; RISE is not converted into a
superpixel method.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Callable

import numpy as np
from skimage.transform import resize


@dataclass(frozen=True)
class RiseConfig:
    num_masks: int = 1024
    batch_size: int = 32
    grid_size: int = 7
    mask_rate: float = 0.5
    seed: int = 0

    def __post_init__(self):
        for name, minimum in (
            ("num_masks", 1),
            ("batch_size", 1),
            ("grid_size", 2),
            ("seed", 0),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not np.isfinite(self.mask_rate) or not 0 < self.mask_rate < 1:
            raise ValueError("mask_rate must be finite and strictly between 0 and 1")

    @property
    def keep_probability(self) -> float:
        return 1.0 - float(self.mask_rate)


@dataclass(frozen=True)
class RiseResult:
    saliency: np.ndarray  # Signed pixel map with shape (H, W).
    target: int
    forward_samples: int
    forward_batches: int
    keep_probability: float
    realized_grid_keep_probability: float
    realized_pixel_keep_probability: float
    unique_low_resolution_masks: int
    mean_target_logit: float


def _validate_image_shape(image_shape: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(image_shape, (tuple, list))
        or len(image_shape) != 2
        or any(
            isinstance(v, bool) or not isinstance(v, Integral) or v < 1
            for v in image_shape
        )
    ):
        raise ValueError("image_shape must contain two positive integers")
    return int(image_shape[0]), int(image_shape[1])


def _sample_mask_parameters(
    image_shape: tuple[int, int], config: RiseConfig
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Sample low-resolution grids and crop shifts with a local RNG."""
    height, width = _validate_image_shape(image_shape)
    rng = np.random.default_rng(config.seed)
    grids = (
        rng.random((config.num_masks, config.grid_size, config.grid_size))
        < config.keep_probability
    ).astype(np.float32)
    cell_size = (
        int(np.ceil(height / config.grid_size)),
        int(np.ceil(width / config.grid_size)),
    )
    shifts = rng.integers(0, cell_size, size=(config.num_masks, 2))
    return grids, shifts, cell_size


def _render_mask_batch(
    grids: np.ndarray,
    shifts: np.ndarray,
    image_shape: tuple[int, int],
    cell_size: tuple[int, int],
) -> np.ndarray:
    """Bilinearly upsample low-resolution grids and apply random crops."""
    height, width = image_shape
    grid_size = grids.shape[1]
    up_size = ((grid_size + 1) * cell_size[0], (grid_size + 1) * cell_size[1])
    masks = np.empty((len(grids), height, width), dtype=np.float32)
    for index, (grid, shift) in enumerate(zip(grids, shifts)):
        upsampled = resize(
            grid,
            up_size,
            order=1,
            mode="reflect",
            anti_aliasing=False,
            preserve_range=True,
        )
        row, column = int(shift[0]), int(shift[1])
        masks[index] = upsampled[row : row + height, column : column + width]
    return masks


def generate_rise_masks(
    image_shape: tuple[int, int], config: RiseConfig | None = None
) -> np.ndarray:
    """Generate all native RISE soft masks as a finite float32 (N, H, W) array.

    This helper is intended for tests, inspection, or explicit mask reuse. The
    main explainer renders only one batch at a time to bound peak memory.
    """
    config = config if config is not None else RiseConfig()
    image_shape = _validate_image_shape(image_shape)
    grids, shifts, cell_size = _sample_mask_parameters(image_shape, config)
    return _render_mask_batch(grids, shifts, image_shape, cell_size)


def region_scores_from_saliency(
    saliency: np.ndarray, segments: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sum a pixel saliency map inside each (possibly noncontiguous) region."""
    saliency, segments = np.asarray(saliency), np.asarray(segments)
    if (
        saliency.ndim != 2
        or min(saliency.shape) < 1
        or not np.issubdtype(saliency.dtype, np.floating)
        or not np.isfinite(saliency).all()
    ):
        raise ValueError("saliency must be a finite floating-point HW array")
    if segments.shape != saliency.shape or not np.issubdtype(
        segments.dtype, np.integer
    ):
        raise ValueError("segments must be an integer HW array aligned with saliency")

    region_ids, inverse = np.unique(segments, return_inverse=True)
    scores = np.bincount(
        inverse.ravel(), weights=saliency.ravel().astype(np.float64), minlength=len(region_ids)
    )
    return region_ids, scores


def explain_rise(
    image: np.ndarray,
    predict_logits: Callable[[np.ndarray], np.ndarray],
    target: int,
    config: RiseConfig | None = None,
) -> RiseResult:
    """Explain one image with RISE while evaluating exactly ``num_masks`` inputs.

    ``image`` is a finite floating-point normalized HWC input. ``predict_logits``
    receives normalized NHWC batches and must return finite pre-Softmax logits
    with shape (batch, classes). ``target`` is the explicit zero-based correct
    class and is never inferred through an extra model call.

    Low-resolution Bernoulli grids are bilinearly upsampled, randomly shifted,
    and cropped as in native RISE. The target-logit-weighted masks are divided
    by the nominal ``num_masks * keep_probability``. No unperturbed image is
    forced into the sample because that would change both the estimator and the
    fixed forward budget.
    """
    config = config if config is not None else RiseConfig()
    image = np.asarray(image)
    if (
        image.ndim != 3
        or min(image.shape) < 1
        or not np.issubdtype(image.dtype, np.floating)
        or not np.isfinite(image).all()
    ):
        raise ValueError("image must be a finite floating-point HWC array")
    if isinstance(target, bool) or not isinstance(target, Integral) or target < 0:
        raise ValueError("target must be a nonnegative integer class index")

    image_shape = (image.shape[0], image.shape[1])
    grids, shifts, cell_size = _sample_mask_parameters(image_shape, config)
    saliency_sum = np.zeros(image_shape, dtype=np.float64)
    target_logit_sum = 0.0
    pixel_keep_sum = 0.0
    batches = 0
    n_classes = None

    for start in range(0, config.num_masks, config.batch_size):
        stop = min(start + config.batch_size, config.num_masks)
        masks = _render_mask_batch(
            grids[start:stop], shifts[start:stop], image_shape, cell_size
        )
        perturbed = image[None, ...] * masks[..., None]
        logits = np.asarray(predict_logits(perturbed))
        try:
            finite_logits = bool(np.isfinite(logits).all())
        except TypeError:
            finite_logits = False
        if (
            logits.ndim != 2
            or logits.shape[0] != stop - start
            or logits.shape[1] <= target
            or not finite_logits
        ):
            raise ValueError(
                "predict_logits must return finite (batch, classes) logits including target"
            )
        if n_classes is not None and logits.shape[1] != n_classes:
            raise ValueError("predict_logits changed its class count between batches")
        n_classes = logits.shape[1]

        values = logits[:, target].astype(np.float64, copy=False)
        saliency_sum += np.einsum(
            "b,bhw->hw", values, masks, dtype=np.float64, optimize=True
        )
        target_logit_sum += float(values.sum())
        pixel_keep_sum += float(masks.sum(dtype=np.float64))
        batches += 1

    saliency = saliency_sum / (config.num_masks * config.keep_probability)
    return RiseResult(
        saliency=saliency,
        target=int(target),
        forward_samples=config.num_masks,
        forward_batches=batches,
        keep_probability=config.keep_probability,
        realized_grid_keep_probability=float(grids.mean(dtype=np.float64)),
        realized_pixel_keep_probability=(
            pixel_keep_sum / (config.num_masks * image_shape[0] * image_shape[1])
        ),
        unique_low_resolution_masks=int(np.unique(grids, axis=0).shape[0]),
        mean_target_logit=target_logit_sum / config.num_masks,
    )
