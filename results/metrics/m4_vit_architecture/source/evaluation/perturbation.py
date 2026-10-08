"""Region-wise insertion/deletion logit AUC and seed stability diagnostics."""
import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import spearmanr


def insertion_deletion(image, segments, scores, predict, target, batch_size=32, steps=20,
                       fraction_mode="regions"):
    """Evaluate insertion/deletion and integrate on region or pixel fractions.

    Scores correspond to sorted unique labels. Ties use ascending region ID.
    Keep signed contributions. Raw logit AUC is not bounded to [0,1]. Evaluation
    forwards are returned separately and must not count as attribution budget.

    ``fraction_mode='regions'`` preserves the M2/M3 protocol. ``'pixels'``
    selects checkpoints nearest to equally spaced cumulative pixel fractions;
    this is the fair comparison for unequal SLIC regions versus equal ViT
    patches. The returned x-axis always records the actual changed fraction.
    """
    image, segments, scores = map(np.asarray, (image, segments, scores))
    ids, inverse = np.unique(segments, return_inverse=True)
    inverse = inverse.reshape(segments.shape)
    if (segments.shape != image.shape[:2] or scores.shape != (len(ids),)
            or not np.isfinite(scores).all() or batch_size < 1 or steps < 1
            or fraction_mode not in ("regions", "pixels")):
        raise ValueError("Invalid shapes, scores, batch_size or steps")
    order = np.argsort(-scores, kind="stable")
    rank = np.empty(len(ids), dtype=int)
    rank[order] = np.arange(len(ids))
    if fraction_mode == "regions":
        counts = np.unique(np.rint(np.linspace(0, len(ids), steps + 1)).astype(int))
        fractions = counts / len(ids)
    else:
        region_pixels = np.bincount(inverse.ravel(), minlength=len(ids))[order]
        cumulative = np.r_[0, np.cumsum(region_pixels)] / segments.size
        targets = np.linspace(0.0, 1.0, steps + 1)
        counts = np.unique(np.abs(cumulative[:, None] - targets[None, :]).argmin(axis=0))
        fractions = cumulative[counts]
    curves = {}
    forward_samples = 0
    for name in ("insertion", "deletion"):
        values = []
        for start in range(0, len(counts), batch_size):
            keep = rank[inverse][None] < counts[start:start+batch_size, None, None]
            if name == "deletion":
                keep = ~keep
            logits = predict(image[None] * keep[..., None])
            values.extend(logits[:, target].tolist())
            forward_samples += len(keep)
        curves[name] = values
    # The two endpoint predictions are each evaluated twice (once per curve).
    # Average the duplicates to reduce tiny batch-dependent numerical variation.
    baseline_logit = float((curves["insertion"][0] + curves["deletion"][-1]) / 2)
    full_logit = float((curves["insertion"][-1] + curves["deletion"][0]) / 2)
    normalization_delta = full_logit - baseline_logit
    tolerance = 32 * np.finfo(np.float64).eps * max(
        1.0, abs(baseline_logit), abs(full_logit)
    )
    normalized_insertion = normalized_deletion = None
    normalized_insertion_auc = normalized_deletion_auc = None
    if abs(normalization_delta) > tolerance:
        normalized_insertion = (
            (np.asarray(curves["insertion"], dtype=np.float64) - baseline_logit)
            / normalization_delta
        )
        normalized_deletion = (
            (np.asarray(curves["deletion"], dtype=np.float64) - baseline_logit)
            / normalization_delta
        )
        normalized_insertion_auc = float(trapezoid(normalized_insertion, fractions))
        normalized_deletion_auc = float(trapezoid(normalized_deletion, fractions))

    return {
        "fractions": fractions.tolist(),
        "fraction_mode": fraction_mode,
        **curves,
        "insertion_auc": float(trapezoid(curves["insertion"], fractions)),
        "deletion_auc": float(trapezoid(curves["deletion"], fractions)),
        "baseline_logit": baseline_logit,
        "full_logit": full_logit,
        "normalization_delta": normalization_delta,
        "normalized_insertion": (
            None if normalized_insertion is None else normalized_insertion.tolist()
        ),
        "normalized_deletion": (
            None if normalized_deletion is None else normalized_deletion.tolist()
        ),
        "normalized_insertion_auc": normalized_insertion_auc,
        "normalized_deletion_auc": normalized_deletion_auc,
        "evaluation_forward_samples": forward_samples,
    }


def seed_stability(left, right):
    """Region Spearman and top-20% Jaccard; constant rank correlation is undefined."""
    left, right = np.asarray(left), np.asarray(right)
    if left.ndim != 1 or left.shape != right.shape or not len(left):
        raise ValueError("Expected aligned nonempty region vectors")
    constant = np.ptp(left) == 0 or np.ptp(right) == 0
    rho = None if constant else float(spearmanr(left, right).statistic)
    k = max(1, int(np.ceil(0.2 * len(left))))
    a = set(np.argsort(-left, kind="stable")[:k])
    b = set(np.argsort(-right, kind="stable")[:k])
    return {"spearman": rho, "top20_jaccard": len(a & b) / len(a | b)}


def input_stability(reference, perturbed):
    """Compute finite-neighborhood stability on aligned signed vectors."""
    reference = np.asarray(reference, dtype=np.float64)
    perturbed = np.asarray(perturbed, dtype=np.float64)
    if (reference.ndim != 1 or not len(reference)
            or perturbed.ndim != 2 or perturbed.shape[1:] != reference.shape
            or not len(perturbed) or not np.isfinite(reference).all()
            or not np.isfinite(perturbed).all()):
        raise ValueError("Expected finite aligned reference and perturbed vectors")

    reference_norm = float(np.linalg.norm(reference))
    perturbed_norms = np.linalg.norm(perturbed, axis=1)
    zero_reference = reference_norm == 0.0
    zero_perturbed = int(np.count_nonzero(perturbed_norms == 0.0))
    max_sensitivity = None
    if not zero_reference:
        distances = np.linalg.norm(perturbed - reference[None, :], axis=1)
        max_sensitivity = float(np.max(distances / reference_norm))
    cosine_similarity = None
    if not zero_reference and zero_perturbed == 0:
        cosine = (perturbed @ reference) / (perturbed_norms * reference_norm)
        cosine_similarity = float(np.mean(np.clip(cosine, -1.0, 1.0)))
    return {
        "max_sensitivity": max_sensitivity,
        "cosine_similarity": cosine_similarity,
        "zero_reference": zero_reference,
        "zero_perturbed": zero_perturbed,
        "stability_repeats": len(perturbed),
    }


def region_attribution_map(segments, scores):
    """Spread each signed region contribution uniformly over its pixels."""
    segments, scores = np.asarray(segments), np.asarray(scores, dtype=np.float64)
    if segments.ndim != 2 or not np.issubdtype(segments.dtype, np.integer):
        raise ValueError("segments must be an integer HW array")
    region_ids, inverse = np.unique(segments, return_inverse=True)
    if scores.shape != (len(region_ids),) or not np.isfinite(scores).all():
        raise ValueError("scores must align with sorted unique segment labels")
    counts = np.bincount(inverse.ravel(), minlength=len(region_ids)).astype(np.float64)
    return (scores / counts)[inverse].reshape(segments.shape)


def localization_metrics(attribution, target_mask):
    """Compute positive-energy fraction and row-major Pointing Game hit."""
    attribution = np.asarray(attribution, dtype=np.float64)
    target_mask = np.asarray(target_mask, dtype=bool)
    if (attribution.ndim != 2 or attribution.shape != target_mask.shape
            or not np.isfinite(attribution).all() or not target_mask.any()):
        raise ValueError("Expected a finite HW attribution and nonempty aligned mask")
    positive = np.maximum(attribution, 0.0)
    total = float(positive.sum())
    zero_energy = total == 0.0
    energy = 0.0 if zero_energy else float(positive[target_mask].sum() / total)
    pointing = int(target_mask.ravel()[int(np.argmax(positive))]) if not zero_energy else 0
    return {
        "localization_energy": energy,
        "pointing_game": pointing,
        "zero_positive_energy": zero_energy,
    }
