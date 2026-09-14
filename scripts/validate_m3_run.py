"""Validate a completed M3 default run and its per-sample accounting."""

import argparse
import json
from pathlib import Path

import numpy as np

METHODS = {"lime", "rise", "kernel_shap", "ablation"}
RANDOM_METHODS = METHODS - {"ablation"}


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
    selection_path = run / "selected_shard.json"
    if not selection_path.exists():
        selection_path = run / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    records = rows(run / "metrics.jsonl")
    config = metadata["config"]
    seeds = set(config["seeds"])
    expected = len(config["models"]) * len(selection["selected"]) * (3 * len(seeds) + 1)
    assert len(records) == expected
    keys = [(row["model"], row["id"], row["method"], row["seed"]) for row in records]
    assert len(keys) == len(set(keys))
    assert {row["method"] for row in records} == METHODS

    for row in records:
        assert row["method"] in METHODS
        assert row["model"] in config["models"]
        if row["method"] in RANDOM_METHODS:
            assert row["seed"] in seeds
            assert row["forward_samples"] == config["num_samples"]
        else:
            assert row["seed"] is None
            assert row["forward_samples"] == row["actual_regions"] + 1
        assert row["stability_forward_samples"] == (
            row["forward_samples"] * config["stability_repeats"]
        )
        assert row["stability_prediction_forward_samples"] == config["stability_repeats"]
        assert row["stability_repeats"] == config["stability_repeats"]
        assert 0 <= row["prediction_changes"] <= config["stability_repeats"]
        assert len(row["perturbed_predictions"]) == config["stability_repeats"]
        assert len(row["fractions"]) == len(row["insertion"]) == len(row["deletion"])
        assert np.isfinite(row["insertion"] + row["deletion"]).all()
        assert np.isfinite(row["insertion_auc"])
        assert np.isfinite(row["deletion_auc"])
        if row["max_sensitivity"] is None:
            assert row["zero_reference"]
        else:
            assert np.isfinite(row["max_sensitivity"])
            assert row["max_sensitivity"] >= 0
        if row["cosine_similarity"] is None:
            assert row["zero_reference"] or row["zero_perturbed"] > 0
        else:
            assert -1.0000001 <= row["cosine_similarity"] <= 1.0000001
        if row["localization_eligible"]:
            assert 0 <= row["localization_energy"] <= 1
            assert row["pointing_game"] in (0, 1)
        else:
            assert row["localization_energy"] is None
            assert row["pointing_game"] is None
        if row["method"] == "kernel_shap":
            assert abs(row["diagnostics"]["efficiency_residual"]) < 1e-8
            assert row["diagnostics"]["design_rank"] == row["actual_regions"]

    complete = json.loads((run / "complete.json").read_text(encoding="utf-8"))
    assert complete["rows"] == complete["expected"] == expected
    print(json.dumps({
        "run": run.name,
        "domain": metadata["domain"],
        "images": len(selection["selected"]),
        "models": len(config["models"]),
        "rows": len(records),
        "methods": sorted(METHODS),
        "valid": True,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
