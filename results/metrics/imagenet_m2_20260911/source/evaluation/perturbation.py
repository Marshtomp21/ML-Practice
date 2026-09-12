"""Region-wise insertion/deletion logit AUC and seed stability diagnostics."""
import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import spearmanr


def insertion_deletion(image, segments, scores, predict, target, batch_size=32, steps=20):
    """21 nominal checkpoints; integrate over actual changed-region fractions.

    Scores correspond to sorted unique labels. Ties use ascending region ID.
    Keep signed contributions. Raw logit AUC is not bounded to [0,1]. Evaluation
    forwards are returned separately and must not count as attribution budget.
    """
    image, segments, scores = map(np.asarray, (image, segments, scores))
    ids, inverse = np.unique(segments, return_inverse=True)
    inverse = inverse.reshape(segments.shape)
    if (segments.shape != image.shape[:2] or scores.shape != (len(ids),)
            or not np.isfinite(scores).all() or batch_size < 1 or steps < 1):
        raise ValueError("Invalid shapes, scores, batch_size or steps")
    order = np.argsort(-scores, kind="stable")
    rank = np.empty(len(ids), dtype=int)
    rank[order] = np.arange(len(ids))
    counts = np.unique(np.rint(np.linspace(0, len(ids), steps+1)).astype(int))
    fractions = counts / len(ids)
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
    return {"fractions": fractions.tolist(), **curves,
            "insertion_auc": float(trapezoid(curves["insertion"], fractions)),
            "deletion_auc": float(trapezoid(curves["deletion"], fractions)),
            "evaluation_forward_samples": forward_samples}


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
