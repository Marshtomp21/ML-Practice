"""Freeze stratified CHNCXR configuration-selection and final-evaluation IDs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("data/splits/chncxr_full_main/chncxr_selection.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits/chncxr_m3_config_selection.json"),
    )
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.fraction < 1:
        parser.error("fraction must be between zero and one")
    source = json.loads(args.selection.read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    config_rows, final_rows = [], []
    by_class = {}
    for target in sorted({row["target"] for row in source["joint_correct"]}):
        rows = sorted(
            (row for row in source["joint_correct"] if row["target"] == target),
            key=lambda row: row["id"],
        )
        count = int(round(len(rows) * args.fraction))
        chosen = set(int(index) for index in rng.choice(len(rows), count, replace=False))
        config_rows.extend(row for index, row in enumerate(rows) if index in chosen)
        final_rows.extend(row for index, row in enumerate(rows) if index not in chosen)
        by_class[str(target)] = {
            "joint_correct": len(rows),
            "configuration_selection": count,
            "final_evaluation": len(rows) - count,
        }
    config_rows.sort(key=lambda row: row["id"])
    final_rows.sort(key=lambda row: row["id"])
    canonical = json.dumps({
        "configuration_selection": [row["id"] for row in config_rows],
        "final_evaluation": [row["id"] for row in final_rows],
    }, sort_keys=True).encode()
    payload = {
        "source_selection_fingerprint": source["run_fingerprint"],
        "strategy": "stratified_random_holdout_by_image",
        "configuration_fraction": args.fraction,
        "seed": args.seed,
        "analysis_unit": "image",
        "by_class": by_class,
        "configuration_selection_count": len(config_rows),
        "final_evaluation_count": len(final_rows),
        "configuration_selection": config_rows,
        "final_evaluation": final_rows,
        "split_fingerprint": hashlib.sha256(canonical).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "configuration_selection": len(config_rows),
        "final_evaluation": len(final_rows),
        "by_class": by_class,
        "split_fingerprint": payload["split_fingerprint"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
