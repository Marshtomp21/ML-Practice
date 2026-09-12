"""Check a completed pilot's coverage, budgets, endpoints and numeric artifacts."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def stored_path(value):
    """Resolve artifacts written on either Windows or POSIX."""
    return ROOT/Path(str(value).replace("\\", "/"))


def validate(directory):
    def read(name):
        return json.loads((directory/name).read_text(encoding="utf-8"))
    metadata, selection, complete = read("run.json"), read("selection.json"), read("complete.json")
    config = metadata["config"]
    rows = [json.loads(line) for line in (directory/"metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = set()
    for model, sample in product(config["models"], selection["selected"]):
        expected.add((model, sample["id"], "ablation", None))
        for method, seed in product(("lime", "rise"), config["seeds"]):
            expected.add((model, sample["id"], method, seed))
    observed = [(r["model"], r["id"], r["method"], r["seed"]) for r in rows]
    assert len(set(observed)) == len(observed), "Duplicate run records"
    assert set(observed) == expected, "Missing or unexpected method runs"
    assert complete["rows"] == complete["expected"] == len(expected)
    original_logits = {}
    for row in rows:
        assert row["run_fingerprint"] == metadata["fingerprint"]
        n = row["actual_regions"]
        budget = n+1 if row["method"] == "ablation" else config["num_samples"]
        assert row["forward_samples"] == budget
        assert row["forward_batches"] == (budget+config["batch_size"]-1)//config["batch_size"]
        x = np.asarray(row["fractions"])
        assert x[0] == 0 and x[-1] == 1 and np.all(np.diff(x) > 0)
        assert len(row["insertion"]) == len(row["deletion"]) == len(x)
        assert row["evaluation_forward_samples"] == 2*len(x)
        assert np.isfinite(row["insertion"]+row["deletion"]).all()
        assert abs(row["insertion"][0]-row["deletion"][-1]) < 1e-5
        assert abs(row["insertion"][-1]-row["deletion"][0]) < 1e-5
        original_logits.setdefault((row["model"], row["id"]), []).append(row["insertion"][-1])
        with np.load(stored_path(row["artifact"]), allow_pickle=False) as array:
            np.testing.assert_array_equal(np.unique(array["segments"]), array["region_ids"])
            assert array["region_scores"].shape == (n,)
            assert np.isfinite(array["region_scores"]).all()
            if row["method"] == "rise":
                assert array["saliency"].shape == array["segments"].shape
                assert np.isfinite(array["saliency"]).all()
    assert max(np.ptp(scores) for scores in original_logits.values()) < 1e-4
    pairs_per_method = len(config["seeds"])*(len(config["seeds"])-1)//2
    assert len(read("seed_stability.json")) == len(config["models"])*len(selection["selected"])*2*pairs_per_method
    return {"validated_rows": len(rows), "models": list(config["models"]),
            "images": len(selection["selected"]), "status": "passed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.run_dir), indent=2))
