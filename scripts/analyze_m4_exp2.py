"""Summarize the M4 experiment-2 parameter scans.

The image is the analysis unit: five seed rows are averaged within each image
before aggregate summaries and paired bootstrap comparisons are computed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("imagenet", "chncxr")
SEEDS = (0, 1, 2, 3, 4)
METRICS = (
    "insertion_auc", "deletion_auc", "attribution_seconds", "forward_samples",
    "evaluation_forward_samples", "actual_regions", "localization_energy",
    "pointing_game",
)
VARIANT_INFO = {
    "lime_slic_50": ("lime_slic", 50),
    "lime_slic_200": ("lime_slic", 200),
    "lime_slic_300": ("lime_slic", 300),
    "rise_grid_4": ("rise_grid", 4),
    "rise_grid_10": ("rise_grid", 10),
    "rise_grid_14": ("rise_grid", 14),
    "lime_mask_0.25": ("lime_mask", 0.25),
    "lime_mask_0.75": ("lime_mask", 0.75),
    "rise_mask_0.25": ("rise_mask", 0.25),
    "rise_mask_0.75": ("rise_mask", 0.75),
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def bootstrap(values: list[float], statistic: str, seed: int,
              repeats: int = 2000) -> list[float] | None:
    finite = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(finite):
        return None
    rng = np.random.default_rng(seed)
    sampled = rng.choice(finite, size=(repeats, len(finite)), replace=True)
    if statistic == "mean":
        estimates = sampled.mean(axis=1)
    else:
        estimates = np.median(sampled, axis=1)
    return [float(np.percentile(estimates, 2.5)),
            float(np.percentile(estimates, 97.5))]


def aggregate(values: list[float | None], seed: int) -> dict:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return {"n": 0, "mean": None, "median": None, "iqr": None,
                "mean_ci95": None, "median_ci95": None}
    array = np.asarray(finite, dtype=float)
    return {
        "n": len(finite),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "iqr": [float(np.percentile(array, 25)), float(np.percentile(array, 75))],
        "mean_ci95": bootstrap(finite, "mean", seed),
        "median_ci95": bootstrap(finite, "median", seed + 1),
    }


def load_rows() -> tuple[list[dict], list[str]]:
    rows, sources = [], []
    for domain in DOMAINS:
        for shard in (0, 1):
            directory = ROOT / "results/metrics" / f"m4_{domain}_config_exp2_shard{shard}"
            complete = json.loads((directory / "complete_main.json").read_text())
            if complete["rows"] != complete["expected"]:
                raise ValueError(f"Incomplete shard: {directory}")
            sources.append(str(directory.relative_to(ROOT)))
            rows.extend(read_jsonl(directory / "metrics.jsonl"))
    expected_methods = {"lime", "rise"}
    if {row["method"] for row in rows} != expected_methods:
        raise ValueError("Unexpected methods in experiment-2 results")
    if {row["variant"] for row in rows} != set(VARIANT_INFO):
        raise ValueError("Unexpected or missing experiment-2 variants")
    return rows, sources


def image_level(rows: list[dict]) -> dict[tuple[str, str, str, str], dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["domain"], row["model"], row["id"], row["variant"])].append(row)
    result = {}
    for key, selected in grouped.items():
        if {row["seed"] for row in selected} != set(SEEDS) or len(selected) != 5:
            raise ValueError(f"Invalid seed set for {key}")
        values = {}
        for metric in METRICS:
            available = [row.get(metric) for row in selected]
            finite = [float(value) for value in available if value is not None]
            values[metric] = float(np.mean(finite)) if finite else None
        result[key] = values
    return result


def summarize(images: dict[tuple[str, str, str, str], dict]) -> list[dict]:
    output = []
    for domain in DOMAINS:
        for model in ("resnet50", "vgg13"):
            for variant, (family, value) in VARIANT_INFO.items():
                selected = [metrics for (d, m, _, v), metrics in images.items()
                            if (d, m, v) == (domain, model, variant)]
                row = {"domain": domain, "model": model, "variant": variant,
                       "family": family, "parameter": value, "images": len(selected)}
                stable = int(hashlib.sha256(
                    f"{domain}:{model}:{variant}".encode("utf-8")
                ).hexdigest()[:8], 16)
                for index, metric in enumerate(METRICS):
                    row[metric] = aggregate(
                        [item[metric] for item in selected],
                        seed=20260917 + index + stable % 100000,
                    )
                output.append(row)
    return output


def comparisons(images: dict[tuple[str, str, str, str], dict]) -> list[dict]:
    output = []
    for domain in DOMAINS:
        for model in ("resnet50", "vgg13"):
            families = defaultdict(list)
            for variant, (family, value) in VARIANT_INFO.items():
                families[family].append((value, variant))
            for family, levels in families.items():
                levels.sort()
                for left_index, (left_value, left_variant) in enumerate(levels):
                    for right_value, right_variant in levels[left_index + 1:]:
                        deltas = {}
                        common_ids = sorted({image_id for d, m, image_id, v in images
                                             if (d, m, v) == (domain, model, left_variant)}
                                            & {image_id for d, m, image_id, v in images
                                               if (d, m, v) == (domain, model, right_variant)})
                        for metric in METRICS:
                            values = [
                                images[(domain, model, image_id, right_variant)][metric]
                                - images[(domain, model, image_id, left_variant)][metric]
                                for image_id in common_ids
                                if images[(domain, model, image_id, left_variant)][metric] is not None
                                and images[(domain, model, image_id, right_variant)][metric] is not None
                            ]
                            deltas[metric] = aggregate(
                                values,
                                seed=20260917 + len(output) * 31 + len(metric),
                            )
                        output.append({"domain": domain, "model": model, "family": family,
                                       "low_parameter": left_value,
                                       "high_parameter": right_value,
                                       "low_variant": left_variant,
                                       "high_variant": right_variant,
                                       "images": len(common_ids), "delta_high_minus_low": deltas})
    return output


def write_report(path: Path, summary: list[dict], pairs: list[dict]) -> None:
    lines = ["# M4 实验 2 汇总分析", "",
             "分析单位为图像：每张图像先对 5 个随机种子取均值，再进行跨图像汇总。",
             "均值和中位数的 95% bootstrap 区间均以图像为重采样单位。", ""]
    for domain in DOMAINS:
        lines.append(f"## {domain}")
        for model in ("resnet50", "vgg13"):
            lines.append(f"### {model}")
            lines.append("| 配置 | 图像数 | Insertion 均值 | Deletion 均值 | 耗时中位数 | 前向次数中位数 |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for row in [item for item in summary
                        if item["domain"] == domain and item["model"] == model]:
                lines.append("| {variant} | {images} | {ins:.4f} | {dele:.4f} | {time:.3f} | {forward:.1f} |".format(
                    variant=row["variant"], images=row["images"],
                    ins=row["insertion_auc"]["mean"], dele=row["deletion_auc"]["mean"],
                    time=row["attribution_seconds"]["median"],
                    forward=row["forward_samples"]["median"],))
            lines.append("")
    lines.extend(["## 参数两两配对差值", "",
                  "差值定义为高参数配置减低参数配置；区间为图像级 bootstrap 95% CI。", ""])
    lines.append("| 域 | 模型 | 参数族 | 低值 | 高值 | Δ Insertion | Δ Deletion | Δ 耗时 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for row in pairs:
        delta = row["delta_high_minus_low"]
        lines.append("| {domain} | {model} | {family} | {low_parameter} | {high_parameter} | {ins:.4f} | {dele:.4f} | {time:.3f} |".format(
            domain=row["domain"], model=row["model"], family=row["family"],
            low_parameter=row["low_parameter"], high_parameter=row["high_parameter"],
            ins=delta["insertion_auc"]["mean"], dele=delta["deletion_auc"]["mean"],
            time=delta["attribution_seconds"]["mean"],))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/metrics/m4_exp2_config_20260917")
    args = parser.parse_args()
    rows, sources = load_rows()
    images = image_level(rows)
    summary = summarize(images)
    pairs = comparisons(images)
    args.output.mkdir(parents=True, exist_ok=True)
    per_image = [
        {"domain": domain, "model": model, "id": image_id, "variant": variant,
         "family": VARIANT_INFO[variant][0], "parameter": VARIANT_INFO[variant][1], **metrics}
        for (domain, model, image_id, variant), metrics in sorted(images.items())
    ]
    (args.output / "per_image.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in per_image),
        encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "configuration_selection", "sources": sources,
        "rows": len(rows), "image_variant_rows": len(per_image),
        "summary": summary, "paired_comparisons": pairs,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.output / "REPORT.md", summary, pairs)
    print(json.dumps({"rows": len(rows), "image_variant_rows": len(per_image),
                      "summary_rows": len(summary), "comparison_rows": len(pairs),
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
