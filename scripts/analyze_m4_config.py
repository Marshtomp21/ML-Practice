"""Summarize M4 configuration-selection runs and freeze final-evaluation cells.

Experiments 1 and 2 are combined into one cell table per domain and model. The
default LIME/RISE cell of experiment 2 is the experiment-1 1024-budget cell, so
every one-factor scan includes its reference level. The image is the analysis
unit: seed rows are averaged within an image before any summary or bootstrap.

Final-evaluation selection rule (fixed before any final-evaluation run):
within each domain, model and method, candidates are the configuration cells at
the main 1024 forward budget. Each image ranks its candidates on insertion AUC
(high is better) and deletion AUC (low is better); the cell with the lowest
mean rank over images and both metrics is selected, ties broken by the lower
median attribution time. Stability is reported but does not enter the rule.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_m4_exp3 import analyze as analyze_exp3  # noqa: E402

DOMAINS = ("imagenet", "chncxr")
MODELS = ("resnet50", "vgg13")
SEEDS = (0, 1, 2, 3, 4)
BUDGETS = (256, 512, 1024, 2048)
MAIN_BUDGET = 1024
BOOTSTRAP_REPEATS = 10000
BOOTSTRAP_SEED = 20261009
MAIN_METRICS = (
    "insertion_auc", "deletion_auc", "normalized_insertion_auc",
    "normalized_deletion_auc", "attribution_seconds", "forward_samples",
    "actual_regions", "localization_energy", "pointing_game",
)
STABILITY_METRICS = ("max_sensitivity", "cosine_similarity")
SEED_METRICS = ("spearman", "top20_jaccard")
METRICS = MAIN_METRICS + STABILITY_METRICS + SEED_METRICS
DEFAULTS = {"n_segments": 100, "grid_size": 7, "mask_rate": 0.5,
            "num_samples": MAIN_BUDGET}


def cell_info(variant: str) -> dict:
    """Map a cell ID to its method, scanned factor and factor level."""
    if variant == "ablation_default":
        return {"method": "ablation", "factor": "default", "level": None,
                "budget": None}
    if "_budget_" in variant:
        method, _, rest = variant.partition("_budget_")
        budget = int(rest)
        factor = "budget" if budget != MAIN_BUDGET else "default"
        return {"method": method, "factor": factor, "level": budget, "budget": budget}
    for method in ("lime", "rise"):
        for factor, key, cast in (("slic", "n_segments", int), ("grid", "grid_size", int),
                                  ("mask", "mask_rate", float)):
            prefix = f"{method}_{factor}_"
            if variant.startswith(prefix):
                return {"method": method, "factor": key,
                        "level": cast(variant[len(prefix):]), "budget": MAIN_BUDGET}
    raise ValueError(f"Unknown variant: {variant}")


def cell_parameters(variant: str) -> dict:
    """Return the explicit parameter override needed to rerun a cell."""
    info = cell_info(variant)
    if info["method"] == "ablation":
        return {}
    if info["factor"] in ("budget", "default"):
        return {"num_samples": info["level"]}
    return {info["factor"]: info["level"]}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_shards() -> tuple[list[dict], list[dict], list[dict], dict]:
    main, stability, seeds, fingerprints = [], [], [], {}
    for experiment in ("exp1", "exp2"):
        for domain in DOMAINS:
            for shard in (0, 1):
                directory = ROOT / "results/metrics" / f"m4_{domain}_config_{experiment}_shard{shard}"
                for marker in ("complete_main.json", "complete_stability.json"):
                    complete = json.loads((directory / marker).read_text(encoding="utf-8"))
                    if complete["rows"] != complete["expected"]:
                        raise ValueError(f"Incomplete {marker}: {directory}")
                fingerprints[str(directory.relative_to(ROOT))] = json.loads(
                    (directory / "run.json").read_text(encoding="utf-8"))["fingerprint"]
                main.extend(read_jsonl(directory / "metrics.jsonl"))
                stability.extend(read_jsonl(directory / "stability.jsonl"))
                for row in json.loads((directory / "seed_stability.json").read_text()):
                    seeds.append({**row, "domain": domain})
    return main, stability, seeds, fingerprints


def image_level(main: list[dict], stability: list[dict], seeds: list[dict]) -> dict:
    """Average seeds within image; returns {(domain, model, id, variant): metrics}."""
    grouped = defaultdict(list)
    for row in main:
        grouped[row["domain"], row["model"], row["id"], row["variant"]].append(row)
    stable = defaultdict(list)
    for row in stability:
        stable[row["domain"], row["model"], row["id"], row["variant"]].append(row)
    pairs = defaultdict(list)
    for row in seeds:
        pairs[row["domain"], row["model"], row["id"], row["variant"]].append(row)
    result = {}
    for key, rows in grouped.items():
        expected = {None} if key[3] == "ablation_default" else set(SEEDS)
        if {row["seed"] for row in rows} != expected or len(rows) != len(expected):
            raise ValueError(f"Invalid main seed set for {key}")
        if {row["seed"] for row in stable[key]} != expected:
            raise ValueError(f"Invalid stability seed set for {key}")
        values = {}
        for metric in MAIN_METRICS:
            finite = [row[metric] for row in rows if row.get(metric) is not None]
            values[metric] = float(np.mean(finite)) if finite else None
        for metric in STABILITY_METRICS:
            finite = [row[metric] for row in stable[key] if row.get(metric) is not None]
            values[metric] = float(np.mean(finite)) if finite else None
        for metric in SEED_METRICS:
            finite = [row[metric] for row in pairs[key] if row.get(metric) is not None]
            values[metric] = float(np.mean(finite)) if finite else None
        result[key] = values
    return result


def stable_seed(*parts: object) -> int:
    text = ":".join(str(part) for part in parts)
    return BOOTSTRAP_SEED + int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % 1000003


def median_ci(values: np.ndarray, seed: int) -> list[float] | None:
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(BOOTSTRAP_REPEATS, len(values)))
    estimates = np.median(values[index], axis=1)
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]


def describe(values: list[float | None], seed: int) -> dict:
    array = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(array):
        return {"n": 0, "median": None, "iqr": None, "median_ci95": None, "mean": None}
    return {"n": int(len(array)), "median": float(np.median(array)),
            "iqr": [float(np.percentile(array, 25)), float(np.percentile(array, 75))],
            "median_ci95": median_ci(array, seed), "mean": float(np.mean(array))}


def paired(images: dict, domain: str, model: str, left: str, right: str,
           metric: str) -> dict:
    """Describe right - left over images with both values."""
    ids = sorted({key[2] for key in images if key[:2] == (domain, model)})
    deltas = []
    for image_id in ids:
        a = images.get((domain, model, image_id, left), {}).get(metric)
        b = images.get((domain, model, image_id, right), {}).get(metric)
        if a is not None and b is not None:
            deltas.append(b - a)
    return describe(deltas, stable_seed(domain, model, left, right, metric))


def summarize(images: dict) -> list[dict]:
    output = []
    variants = sorted({key[3] for key in images})
    for domain in DOMAINS:
        for model in MODELS:
            for variant in variants:
                selected = [metrics for key, metrics in images.items()
                            if key[:2] == (domain, model) and key[3] == variant]
                if not selected:
                    continue
                output.append({
                    "domain": domain, "model": model, "variant": variant,
                    **cell_info(variant), "images": len(selected),
                    **{metric: describe([item[metric] for item in selected],
                                        stable_seed(domain, model, variant, metric))
                       for metric in METRICS},
                })
    return output


def budget_curves(images: dict) -> list[dict]:
    """Experiment 1: marginal gains between consecutive budgets per method."""
    output = []
    for domain in DOMAINS:
        for model in MODELS:
            for method in ("lime", "rise", "kernel_shap"):
                for low, high in zip(BUDGETS, BUDGETS[1:]):
                    left, right = f"{method}_budget_{low}", f"{method}_budget_{high}"
                    output.append({
                        "domain": domain, "model": model, "method": method,
                        "low_budget": low, "high_budget": high,
                        "delta_high_minus_low": {
                            metric: paired(images, domain, model, left, right, metric)
                            for metric in METRICS},
                    })
    return output


def granularity_contrasts(images: dict) -> list[dict]:
    """Experiment 2: every scanned level versus the method default."""
    output = []
    variants = sorted({key[3] for key in images})
    for domain in DOMAINS:
        for model in MODELS:
            for variant in variants:
                info = cell_info(variant)
                if info["factor"] not in ("n_segments", "grid_size", "mask_rate"):
                    continue
                default = f"{info['method']}_budget_{MAIN_BUDGET}"
                output.append({
                    "domain": domain, "model": model, "method": info["method"],
                    "factor": info["factor"], "level": info["level"],
                    "default_level": DEFAULTS[info["factor"]], "variant": variant,
                    "delta_level_minus_default": {
                        metric: paired(images, domain, model, default, variant, metric)
                        for metric in METRICS},
                })
    return output


def pareto(summary: list[dict]) -> list[dict]:
    """Non-dominated cells by median insertion, deletion and attribution time."""
    output = []
    for domain in DOMAINS:
        for model in MODELS:
            cells = [row for row in summary if (row["domain"], row["model"]) == (domain, model)]
            points = {row["variant"]: (row["insertion_auc"]["median"],
                                       row["deletion_auc"]["median"],
                                       row["attribution_seconds"]["median"])
                      for row in cells}

            def dominates(a, b):
                return (a[0] >= b[0] and a[1] <= b[1] and a[2] <= b[2]
                        and (a[0] > b[0] or a[1] < b[1] or a[2] < b[2]))

            front = sorted(name for name, point in points.items()
                           if not any(dominates(other, point)
                                      for key, other in points.items() if key != name))
            output.append({"domain": domain, "model": model, "nondominated": front})
    return output


def select_cells(images: dict, summary: list[dict]) -> dict:
    """Apply the module-docstring rule; returns per domain/model/method choices."""
    medians = {(row["domain"], row["model"], row["variant"]): row["attribution_seconds"]["median"]
               for row in summary}
    choice = {}
    for domain in DOMAINS:
        for model in MODELS:
            ids = sorted({key[2] for key in images if key[:2] == (domain, model)})
            choice.setdefault(domain, {})[model] = {}
            for method in ("lime", "rise", "kernel_shap", "ablation"):
                candidates = sorted(
                    variant for variant in {key[3] for key in images}
                    if cell_info(variant)["method"] == method
                    and (cell_info(variant)["budget"] in (None, MAIN_BUDGET))
                )
                ranks = defaultdict(list)
                for image_id in ids:
                    for metric, sign in (("insertion_auc", -1.0), ("deletion_auc", 1.0)):
                        values = [sign * images[domain, model, image_id, v][metric]
                                  for v in candidates]
                        for variant, rank in zip(candidates, rankdata(values)):
                            ranks[variant].append(float(rank))
                table = [{"variant": v, "mean_rank": float(np.mean(ranks[v])),
                          "median_seconds": medians[domain, model, v]} for v in candidates]
                table.sort(key=lambda row: (row["mean_rank"], row["median_seconds"]))
                choice[domain][model][method] = {
                    "selected": table[0]["variant"],
                    "parameters": cell_parameters(table[0]["variant"]),
                    "candidates": table, "images": len(ids),
                }
    return choice


def final_cells(choice: dict) -> dict:
    """Default cells, domain-tuned cells and, for CHNCXR, ImageNet-transferred cells."""
    cells = {}
    for domain in DOMAINS:
        cells[domain] = {}
        for model in MODELS:
            roles = {}
            for method in ("lime", "rise", "kernel_shap", "ablation"):
                default = "ablation_default" if method == "ablation" else f"{method}_budget_{MAIN_BUDGET}"
                roles.setdefault(default, set()).add(f"default:{method}")
                roles.setdefault(choice[domain][model][method]["selected"], set()).add(
                    f"tuned:{method}")
                if domain == "chncxr":
                    roles.setdefault(choice["imagenet"][model][method]["selected"], set()).add(
                        f"transfer:{method}")
            cells[domain][model] = [
                {"id": variant, "method": cell_info(variant)["method"],
                 **cell_parameters(variant), "roles": sorted(role_set)}
                for variant, role_set in sorted(roles.items())
            ]
    return cells


def fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def ci(stat: dict, digits: int = 4) -> str:
    if stat["median"] is None:
        return "—"
    low, high = stat["median_ci95"]
    return f"{stat['median']:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def write_report(path: Path, summary: list[dict], curves: list[dict],
                 contrasts: list[dict], front: list[dict], choice: dict,
                 cells: dict, exp3: list[dict]) -> None:
    lines = [
        "# M4 配置选择集汇总（实验 1–3）", "",
        "范围：`data/splits/m4_frozen_20260917.json` 的 `configuration_selection`（ImageNet 75 张，"
        "CHNCXR 86 张）。图像为分析单位：随机方法先在图内对 5 个种子取均值。",
        f"区间为图像级 bootstrap 中位数 95% CI（{BOOTSTRAP_REPEATS} 次，种子 {BOOTSTRAP_SEED}）。"
        "Insertion 越高越好，Deletion 越低越好，Max-Sens 越低越好，Cosine/Spearman/Jaccard 越高越好。",
        "",
        "> 注意：LIME 的 SLIC 粒度扫描同时改变了解释单元和 Insertion/Deletion 的评价划分；"
        "RISE 网格扫描固定评价用 SLIC=100。跨 SLIC 粒度的 AUC 差异包含评价分辨率的影响，"
        "最终评价集额外在统一 SLIC=100 划分上重算 AUC 以分离这一因素。", "",
    ]
    for domain in DOMAINS:
        for model in MODELS:
            lines += [f"## {domain} / {model}", "",
                      "| 配置 | 方法 | 因素 | 水平 | Insertion | Deletion | Max-Sens | Cosine | "
                      "Spearman | Top20 Jaccard | 秒/图 | 前向数 |",
                      "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            rows = [row for row in summary if (row["domain"], row["model"]) == (domain, model)]
            rows.sort(key=lambda row: (row["method"], row["factor"], str(row["level"])))
            for row in rows:
                lines.append(
                    f"| {row['variant']} | {row['method']} | {row['factor']} | {row['level']} | "
                    f"{fmt(row['insertion_auc']['median'])} | {fmt(row['deletion_auc']['median'])} | "
                    f"{fmt(row['max_sensitivity']['median'])} | {fmt(row['cosine_similarity']['median'])} | "
                    f"{fmt(row['spearman']['median'])} | {fmt(row['top20_jaccard']['median'])} | "
                    f"{fmt(row['attribution_seconds']['median'], 3)} | "
                    f"{fmt(row['forward_samples']['median'], 0)} |")
            nondominated = next(item for item in front
                                if (item["domain"], item["model"]) == (domain, model))
            lines += ["", "非支配配置（中位 Insertion↑、Deletion↓、耗时↓）：" +
                      "、".join(f"`{name}`" for name in nondominated["nondominated"]), ""]
    lines += ["## 实验 1：预算翻倍的边际收益", "",
              "差值为高预算减低预算的逐图配对差值中位数及 95% CI。", "",
              "| 域 | 模型 | 方法 | 预算 | ΔInsertion | ΔDeletion | ΔMax-Sens | ΔSpearman | Δ秒 |",
              "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for row in curves:
        d = row["delta_high_minus_low"]
        lines.append(f"| {row['domain']} | {row['model']} | {row['method']} | "
                     f"{row['low_budget']}→{row['high_budget']} | {ci(d['insertion_auc'])} | "
                     f"{ci(d['deletion_auc'])} | {ci(d['max_sensitivity'])} | "
                     f"{ci(d['spearman'])} | {fmt(d['attribution_seconds']['median'], 3)} |")
    lines += ["", "## 实验 2：粒度水平相对默认值", "",
              "差值为该水平减默认水平（SLIC=100、grid=7、mask=0.5）的逐图配对差值中位数及 95% CI。", "",
              "| 域 | 模型 | 方法 | 因素 | 水平 | ΔInsertion | ΔDeletion | ΔMax-Sens | ΔCosine | ΔSpearman |",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in contrasts:
        d = row["delta_level_minus_default"]
        lines.append(f"| {row['domain']} | {row['model']} | {row['method']} | {row['factor']} | "
                     f"{row['level']} | {ci(d['insertion_auc'])} | {ci(d['deletion_auc'])} | "
                     f"{ci(d['max_sensitivity'])} | {ci(d['cosine_similarity'])} | "
                     f"{ci(d['spearman'])} |")
    lines += ["", "## 实验 3：LIME → +RISE → +KernelSHAP 候选集扩展", "",
              "逐图取候选集中最优 Insertion / 最低 Deletion；增益为加入新方法前后差值（正值为改善）。"
              "“新前沿比例”为新加入方法不被已有方法（含 Ablation）支配的图像比例。", "",
              "| 域 | 模型 | 预算 | +RISE ΔIns | +RISE ΔDel | +SHAP ΔIns | +SHAP ΔDel | "
              "RISE 新前沿 | SHAP 新前沿 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in exp3:
        lines.append(f"| {row['domain']} | {row['model']} | {row['budget']} | "
                     f"{fmt(row['median_rise_insertion_gain'])} | {fmt(row['median_rise_deletion_gain'])} | "
                     f"{fmt(row['median_shap_insertion_gain'])} | {fmt(row['median_shap_deletion_gain'])} | "
                     f"{row['rise_new_frontier_fraction']:.3f} | {row['shap_new_frontier_fraction']:.3f} |")
    lines += ["", "## 冻结的最终评价配置", "",
              "规则见本脚本文档字符串：1024 预算内逐图对 Insertion、Deletion 排名，取平均名次最低者。", "",
              "| 域 | 模型 | 方法 | 选中配置 | 平均名次 | 次优配置 | 次优名次 |",
              "|---|---|---|---|---:|---|---:|"]
    for domain in DOMAINS:
        for model in MODELS:
            for method, item in choice[domain][model].items():
                table = item["candidates"]
                second_name, second_rank = "—", "—"
                if len(table) > 1:
                    second_name = f"`{table[1]['variant']}`"
                    second_rank = f"{table[1]['mean_rank']:.3f}"
                lines.append(f"| {domain} | {model} | {method} | `{item['selected']}` | "
                             f"{table[0]['mean_rank']:.3f} | {second_name} | {second_rank} |")
    lines += ["", "最终评价单元（含角色：default 主设置、tuned 域内选择、transfer ImageNet 迁移）：", ""]
    for domain in DOMAINS:
        for model in MODELS:
            joined = "；".join(f"`{cell['id']}`（{', '.join(cell['roles'])}）"
                              for cell in cells[domain][model])
            lines.append(f"- {domain} / {model}：{joined}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/metrics/m4_config_analysis")
    args = parser.parse_args()
    main_rows, stability_rows, seed_rows, fingerprints = load_shards()
    images = image_level(main_rows, stability_rows, seed_rows)
    summary = summarize(images)
    curves = budget_curves(images)
    contrasts = granularity_contrasts(images)
    front = pareto(summary)
    choice = select_cells(images, summary)
    cells = final_cells(choice)

    exp3_rows = analyze_exp3([row for row in main_rows if row["experiment"] == "exp1"], SEEDS)
    grouped = defaultdict(list)
    for row in exp3_rows:
        grouped[row["domain"], row["model"], row["budget"]].append(row)
    exp3 = []
    for (domain, model, budget), selected in sorted(grouped.items()):
        exp3.append({
            "domain": domain, "model": model, "budget": budget, "images": len(selected),
            **{f"median_{metric}": float(np.median([row[metric] for row in selected]))
               for metric in ("rise_insertion_gain", "rise_deletion_gain",
                              "shap_insertion_gain", "shap_deletion_gain")},
            "rise_new_frontier_fraction": float(np.mean(
                [row["stages"][1]["added_method_nondominated"] for row in selected])),
            "shap_new_frontier_fraction": float(np.mean(
                [row["stages"][2]["added_method_nondominated"] for row in selected])),
        })

    args.output.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat()
    with (args.output / "per_image.jsonl").open("w", encoding="utf-8") as stream:
        for (domain, model, image_id, variant), metrics in sorted(images.items()):
            stream.write(json.dumps({"domain": domain, "model": model, "id": image_id,
                                     "variant": variant, **cell_info(variant), **metrics},
                                    allow_nan=False) + "\n")
    common = {"created_utc": created, "scope": "configuration_selection",
              "source_fingerprints": fingerprints,
              "bootstrap": {"repeats": BOOTSTRAP_REPEATS, "seed": BOOTSTRAP_SEED}}
    (args.output / "summary.json").write_text(json.dumps({
        **common, "cells": summary, "budget_marginal_gains": curves,
        "granularity_contrasts": contrasts, "nondominated": front, "exp3": exp3,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "final_selection.json").write_text(json.dumps({
        **common, "rule": __doc__.split("Final-evaluation selection rule")[1].strip(),
        "choice": choice, "final_cells": cells,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.output / "REPORT.md", summary, curves, contrasts, front, choice, cells, exp3)
    print(json.dumps({"images": len(images), "cells": len(summary),
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
