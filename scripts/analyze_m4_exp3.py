"""Quantify the LIME -> +RISE -> +KernelSHAP candidate-set gains from M4 E1."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("lime", "rise", "kernel_shap")
BUDGETS = (256, 512, 1024, 2048)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def dominates(left: dict, right: dict) -> bool:
    """Use insertion high, deletion low and measured attribution time low."""
    good = (left["insertion_auc"] >= right["insertion_auc"]
            and left["deletion_auc"] <= right["deletion_auc"]
            and left["attribution_seconds"] <= right["attribution_seconds"])
    strict = (left["insertion_auc"] > right["insertion_auc"]
              or left["deletion_auc"] < right["deletion_auc"]
              or left["attribution_seconds"] < right["attribution_seconds"])
    return good and strict


def analyze(rows: list[dict], seeds: tuple[int, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["domain"], row["model"], row["id"], row["variant"])].append(row)
    per_image = []
    images = sorted({(row["domain"], row["model"], row["id"]) for row in rows})
    for domain, model, image_id in images:
        ablation_rows = grouped[domain, model, image_id, "ablation_default"]
        if len(ablation_rows) != 1:
            raise ValueError(f"Missing or duplicate Ablation result: {image_id}")
        ablation = ablation_rows[0]
        reference = {field: float(ablation[field]) for field in
                     ("insertion_auc", "deletion_auc", "attribution_seconds", "forward_samples")}
        for budget in BUDGETS:
            options = {}
            for method in METHODS:
                key = domain, model, image_id, f"{method}_budget_{budget}"
                selected = grouped[key]
                if len(selected) != len(seeds) or {r["seed"] for r in selected} != set(seeds):
                    raise ValueError(f"Incomplete seed set for {key}")
                options[method] = {field: float(np.mean([r[field] for r in selected]))
                                   for field in ("insertion_auc", "deletion_auc",
                                                 "attribution_seconds", "forward_samples")}
            stage = []
            available = []
            for method in METHODS:
                available.append(method)
                current = [options[name] for name in available]
                stage.append({
                    "candidate_methods": available.copy(),
                    "best_insertion": max(item["insertion_auc"] for item in current),
                    "best_deletion": min(item["deletion_auc"] for item in current),
                    "total_forward_samples_all_seeds": int(sum(
                        sum(row["forward_samples"] for row in
                            grouped[domain, model, image_id, f"{name}_budget_{budget}"])
                        for name in available
                    ) + reference["forward_samples"]),
                    "added_method_nondominated": not any(
                        dominates(other, options[method])
                        for other in [reference] + [options[name] for name in available[:-1]]
                    ),
                })
            per_image.append({
                "domain": domain, "model": model, "id": image_id, "budget": budget,
                "ablation": reference,
                "method_metrics": options,
                "stages": stage,
                "rise_insertion_gain": stage[1]["best_insertion"] - stage[0]["best_insertion"],
                "rise_deletion_gain": stage[0]["best_deletion"] - stage[1]["best_deletion"],
                "shap_insertion_gain": stage[2]["best_insertion"] - stage[1]["best_insertion"],
                "shap_deletion_gain": stage[1]["best_deletion"] - stage[2]["best_deletion"],
            })
    return per_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/metrics/m4_exp3_config_20260917")
    args = parser.parse_args()
    sources, rows, fingerprints = [], [], {}
    for domain in ("imagenet", "chncxr"):
        for shard in (0, 1):
            path = ROOT / "results/metrics" / f"m4_{domain}_config_exp1_shard{shard}"
            complete = json.loads((path / "complete_main.json").read_text(encoding="utf-8"))
            if complete["rows"] != complete["expected"]:
                raise ValueError(f"Incomplete E1 shard: {path}")
            sources.append(str(path.relative_to(ROOT)))
            fingerprints[str(path.relative_to(ROOT))] = json.loads(
                (path / "run.json").read_text(encoding="utf-8")
            )["fingerprint"]
            rows.extend(read_jsonl(path / "metrics.jsonl"))
    per_image = analyze(rows, (0, 1, 2, 3, 4))
    grouped = defaultdict(list)
    for row in per_image:
        grouped[row["domain"], row["model"], row["budget"]].append(row)
    summary = []
    for (domain, model, budget), selected in sorted(grouped.items()):
        summary.append({
            "domain": domain, "model": model, "budget": budget, "images": len(selected),
            **{f"median_{metric}": median([row[metric] for row in selected])
               for metric in ("rise_insertion_gain", "rise_deletion_gain",
                              "shap_insertion_gain", "shap_deletion_gain")},
            "rise_new_frontier_fraction": float(np.mean([
                row["stages"][1]["added_method_nondominated"] for row in selected
            ])),
            "shap_new_frontier_fraction": float(np.mean([
                row["stages"][2]["added_method_nondominated"] for row in selected
            ])),
        })
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "per_image.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in per_image),
        encoding="utf-8",
    )
    (args.output / "summary.json").write_text(
        json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
                    "scope": "configuration_selection", "sources": sources,
                    "source_fingerprints": fingerprints, "rows": len(per_image),
                    "summary": summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"per_image": len(per_image), "summary_rows": len(summary),
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
