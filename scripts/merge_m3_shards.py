"""Merge completed, disjoint M3 sample shards into one validated run directory."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m3_default import RANDOM_METHODS, summarize_m3
from experiments.perturbation_pilot import read_rows, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("shards", type=Path, nargs="+")
    args = parser.parse_args()
    output = args.output.resolve()
    shards = [path.resolve() for path in args.shards]
    if output.exists():
        raise ValueError(f"Output already exists: {output}")

    metadata = [json.loads((path / "run.json").read_text(encoding="utf-8")) for path in shards]
    fingerprints = {item["fingerprint"] for item in metadata}
    if len(fingerprints) != 1:
        raise ValueError("Shard run fingerprints differ")
    selections = [
        json.loads((path / "selected_shard.json").read_text(encoding="utf-8"))
        for path in shards
    ]
    shard_counts = {item["shard_count"] for item in selections}
    shard_indices = {item["shard_index"] for item in selections}
    if shard_counts != {len(shards)} or shard_indices != set(range(len(shards))):
        raise ValueError("Shard indices do not form a complete partition")
    selected_ids = [row["id"] for item in selections for row in item["selected"]]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Sample shards overlap")
    full_selection = json.loads((shards[0] / "selection.json").read_text(encoding="utf-8"))
    if set(selected_ids) != {row["id"] for row in full_selection["selected"]}:
        raise ValueError("Sample shards do not cover the fixed selection")

    records = [row for path in shards for row in read_rows(path / "metrics.jsonl")]
    keys = [(row["model"], row["id"], row["method"], row["seed"]) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Metric shards contain duplicate method runs")
    config = metadata[0]["config"]
    expected = len(config["models"]) * len(full_selection["selected"]) * (
        len(RANDOM_METHODS) * len(config["seeds"]) + 1
    )
    if len(records) != expected:
        raise ValueError(f"Expected {expected} metric rows, found {len(records)}")

    output.mkdir(parents=True)
    merged_metadata = {
        **metadata[0],
        "merged_utc": datetime.now(timezone.utc).isoformat(),
        "merged_shards": [path.name for path in shards],
    }
    write_json(output / "run.json", merged_metadata)
    write_json(output / "selection.json", full_selection)
    for name in config["models"]:
        shutil.copy2(shards[0] / f"screen_{name}.json", output / f"screen_{name}.json")
    shutil.copytree(shards[0] / "source", output / "source")
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in sorted(records, key=lambda x: (
            x["model"], x["id"], x["method"], -1 if x["seed"] is None else x["seed"]
        )):
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    stability = [
        row
        for path in shards
        for row in json.loads((path / "seed_stability.json").read_text(encoding="utf-8"))
    ]
    write_json(output / "seed_stability.json", stability)
    summarize_m3(output)
    write_json(output / "complete.json", {
        "rows": len(records),
        "expected": expected,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "shards": len(shards),
    })
    write_json(output / "shards.json", {
        "fingerprint": metadata[0]["fingerprint"],
        "shards": [
            {"name": path.name, "sample_ids": [row["id"] for row in selection["selected"]]}
            for path, selection in zip(shards, selections)
        ],
    })
    print(json.dumps({
        "output": str(output),
        "domain": metadata[0]["domain"],
        "samples": len(selected_ids),
        "rows": len(records),
        "shards": len(shards),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
