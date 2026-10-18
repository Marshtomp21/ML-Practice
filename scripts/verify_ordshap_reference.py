"""Check attribution.game_theoretic.ordshap against the official OrdShap code.

Requires a checkout of https://github.com/davinhill/ordshap (verified at commit
aaccc3e) and the ``shap`` package, neither of which the project depends on:

    git clone https://github.com/davinhill/ordshap /path/to/ordshap
    python scripts/verify_ordshap_reference.py /path/to/ordshap

With full coalition and permutation enumeration both estimators are
deterministic, so value and position importance must agree to rounding error.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from attribution.game_theoretic.ordshap import OrdShapConfig, estimate_ordshap  # noqa: E402


def main() -> None:
    sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
    from ordshap.ordshap_v2 import ordshap

    rng = np.random.default_rng(3)
    report = []
    for d in (3, 4, 5):
        slot_weights = rng.normal(size=d)

        def model(x):
            x = np.asarray(x, dtype=float)
            return (x @ slot_weights + 0.7 * x[:, 0] * x[:, 1]
                    + 0.3 * x[:, -1] ** 2).reshape(-1, 1)

        sample = rng.normal(size=(1, d))
        pi_ref, vi_ref = ordshap(model=model, data=np.zeros((1, d))).explain(
            sample, n_perm_samples=1000)

        def evaluate(masks, perms):
            present = np.where(masks, sample, 0.0)
            return model(np.take_along_axis(present, perms, axis=1)).ravel()

        ours = estimate_ordshap(evaluate, d, OrdShapConfig(
            num_coalitions=2 ** d, num_permutations=1000, full_permutations=1000))
        report.append({
            "features": d,
            "max_abs_vi_difference": float(np.abs(ours["value_importance"] - np.ravel(vi_ref)).max()),
            "max_abs_pi_difference": float(np.abs(ours["position_importance"] - np.ravel(pi_ref)).max()),
        })
    print(json.dumps(report, indent=2))
    if any(row["max_abs_vi_difference"] > 1e-10 or row["max_abs_pi_difference"] > 1e-10
           for row in report):
        raise SystemExit("OrdShap does not match the reference implementation")


if __name__ == "__main__":
    main()
