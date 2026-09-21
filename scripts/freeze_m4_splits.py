"""Freeze disjoint M4 tuning and final-evaluation image lists from M3 screening."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def canonical_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "id": row["id"],
            "path": row["path"].replace("\\", "/"),
            "target": row["target"],
            "sha256": row["sha256"],
        }
        for row in sorted(rows, key=lambda item: item["id"])
    ]


def freeze_imagenet(seed: int, fraction: float) -> dict:
    source = read_json("data/splits/m3_imagenet_default_20260913/imagenet_selection.json")
    pool = canonical_rows(source["joint_correct"])
    seen_ids = {row["id"] for row in source["selected"]}
    if not seen_ids.issubset({row["id"] for row in pool}):
        raise ValueError("M3 ImageNet images are not all in the jointly correct pool")
    desired = round(len(pool) * fraction)
    if desired < len(seen_ids) or desired >= len(pool):
        raise ValueError("Invalid ImageNet tuning fraction")
    unseen = [row for row in pool if row["id"] not in seen_ids]
    indices = np.random.default_rng(seed).choice(
        len(unseen), desired - len(seen_ids), replace=False
    )
    selected = seen_ids | {unseen[int(index)]["id"] for index in indices}
    return {
        "source_selection_fingerprint": source["run_fingerprint"],
        "configuration_selection": [row for row in pool if row["id"] in selected],
        "final_evaluation": [row for row in pool if row["id"] not in selected],
        "m3_seen_ids": sorted(seen_ids),
        "m3_seen_excluded_from_final": sorted(seen_ids),
    }


def freeze_chncxr() -> dict:
    source = read_json("data/splits/chncxr_m3_config_selection.json")
    m3 = read_json("data/splits/m3_chncxr_default_20260913/chncxr_selection.json")
    seen_ids = {row["id"] for row in m3["selected"]}
    config = canonical_rows(source["configuration_selection"])
    final = canonical_rows(source["final_evaluation"])
    excluded = [row for row in final if row["id"] in seen_ids]
    return {
        "source_selection_fingerprint": source["source_selection_fingerprint"],
        "source_split_fingerprint": source["split_fingerprint"],
        "configuration_selection": config,
        "final_evaluation": [row for row in final if row["id"] not in seen_ids],
        "m3_seen_ids": sorted(seen_ids),
        "m3_seen_excluded_from_final": sorted(row["id"] for row in excluded),
        "exploratory_only": excluded,
    }


def validate(payload: dict) -> None:
    groups = [payload["configuration_selection"], payload["final_evaluation"]]
    groups.append(payload.get("exploratory_only", []))
    ids = [[row["id"] for row in rows] for rows in groups]
    if any(len(group) != len(set(group)) for group in ids):
        raise ValueError("Duplicate image ID within a split")
    if any(set(ids[left]) & set(ids[right]) for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("Split groups overlap")
    if set(payload["m3_seen_ids"]) & set(ids[1]):
        raise ValueError("An M3-inspected image entered the final set")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--imagenet-config-fraction", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=ROOT / "data/splits/m4_frozen_20260917.json")
    args = parser.parse_args()
    if not 0 < args.imagenet_config_fraction < 1:
        parser.error("imagenet-config-fraction must be between zero and one")

    domains = {
        "imagenet": freeze_imagenet(args.seed, args.imagenet_config_fraction),
        "chncxr": freeze_chncxr(),
    }
    for domain in domains.values():
        validate(domain)
    payload = {
        "stage": "m4",
        "analysis_unit": "image",
        "selection_seed": args.seed,
        "imagenet_config_fraction": args.imagenet_config_fraction,
        "rule": "M3-inspected images may be used for tuning but not final evaluation",
        "domains": domains,
    }
    identity = {
        name: {key: [row["id"] for row in value[key]]
               for key in ("configuration_selection", "final_evaluation")}
        for name, value in domains.items()
    }
    payload["split_fingerprint"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    if args.output.exists():
        current = json.loads(args.output.read_text(encoding="utf-8"))
        if current != payload:
            raise ValueError(f"Frozen split already exists and differs: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({
        "output": str(args.output),
        "fingerprint": payload["split_fingerprint"],
        "counts": {name: {key: len(value.get(key, [])) for key in
                             ("configuration_selection", "final_evaluation", "exploratory_only")}
                   for name, value in domains.items()},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
