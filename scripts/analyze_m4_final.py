"""Analyze the M4 final-evaluation runs (held-out split) and CHNCXR transfer.

Test families, fixed before any final-evaluation result was inspected:

* Confirmatory (M1): default setting only. For every domain x model x metric,
  all pairs of core methods (LIME, RISE, Ablation, KernelSHAP; seed-stability
  metrics use the three random methods) form one family. Two-sided Wilcoxon
  signed-rank test, paired rank-biserial effect size, Holm correction,
  alpha = 0.05; all-zero differences give p = 1 and effect 0.
* Domain tuning (exploratory): tuned versus default cell per domain x model x
  method x metric; reported with paired CIs and unadjusted p values.
* Optional experiment 7 (CHNCXR): domain-tuned versus ImageNet-transferred
  cells per model x metric; the LIME and RISE comparisons form one Holm family
  (KernelSHAP and Ablation transfer unchanged). Lesion strata are reported
  descriptively: lesion union smaller than one default SLIC region, and single
  versus multiple lesion annotations.

Paired differences are summarized by their median and an image-level
bootstrap 95% CI (10000 resamples). Seeds are averaged within an image first.
Insertion/Deletion are reported both on each cell's own partition and on the
common default SLIC partition (``common_*``).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_m4_config import (  # noqa: E402
    BOOTSTRAP_REPEATS, BOOTSTRAP_SEED, describe, fmt, stable_seed, ci,
)

DOMAINS = ("imagenet", "chncxr")
MODELS = ("resnet50", "vgg13")
SEEDS = (0, 1, 2, 3, 4)
CORE = ("lime", "rise", "ablation", "kernel_shap")
RANDOM = ("lime", "rise", "kernel_shap")
DEFAULT = {"lime": "lime_budget_1024", "rise": "rise_budget_1024",
           "kernel_shap": "kernel_shap_budget_1024", "ablation": "ablation_default"}
MAIN_METRICS = ("insertion_auc", "deletion_auc", "common_insertion_auc",
                "common_deletion_auc", "normalized_insertion_auc",
                "normalized_deletion_auc", "attribution_seconds", "clean_attribution_seconds",
                "forward_samples",
                "actual_regions", "localization_energy", "pointing_game")
STABILITY_METRICS = ("max_sensitivity", "cosine_similarity")
SEED_METRICS = ("spearman", "top20_jaccard")
METRICS = MAIN_METRICS + STABILITY_METRICS + SEED_METRICS
CONFIRMATORY = ("insertion_auc", "deletion_auc", "max_sensitivity", "cosine_similarity",
                "spearman", "top20_jaccard")
SHARDS = 12
# Another tenant's inference job shared the final-evaluation GPUs from about
# this time on; attribution timings written later are excluded from
# ``clean_attribution_seconds`` (artifact modification time marks each row).
SHARED_GPU_FROM = datetime(2026, 10, 9, 20, 0).timestamp()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load(require_stability: bool, allow_partial: bool = False) -> tuple[list, list, list, dict, bool]:
    main, stability, seeds, roles = [], [], [], {}
    stability_complete = True
    for domain in DOMAINS:
        for shard in range(SHARDS):
            directory = ROOT / "results/metrics" / f"m4_{domain}_final_shard{shard}"
            if not (directory / "complete_main.json").is_file() and not allow_partial:
                raise ValueError(f"Main phase incomplete: {directory}")
            if not (directory / "run.json").is_file():
                continue
            run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
            roles[domain] = run["config"]["final_roles"]
            main.extend(read_jsonl(directory / "metrics.jsonl"))
            if (directory / "complete_stability.json").is_file():
                stability.extend(read_jsonl(directory / "stability.jsonl"))
            else:
                stability_complete = False
            if (directory / "seed_stability.json").is_file():
                for row in json.loads((directory / "seed_stability.json").read_text()):
                    seeds.append({**row, "domain": domain})
    if require_stability and not stability_complete:
        raise ValueError("Stability phase incomplete")
    if not stability_complete:
        stability = []
    return main, stability, seeds, roles, stability_complete


def image_level(main, stability, seeds, allow_partial: bool = False) -> dict:
    grouped, stable, pairs = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in main:
        written = (ROOT / row["artifact"]).stat().st_mtime
        row["clean_attribution_seconds"] = (row["attribution_seconds"]
                                            if written < SHARED_GPU_FROM else None)
        grouped[row["domain"], row["model"], row["id"], row["variant"]].append(row)
    for row in stability:
        stable[row["domain"], row["model"], row["id"], row["variant"]].append(row)
    for row in seeds:
        pairs[row["domain"], row["model"], row["id"], row["variant"]].append(row)
    result = {}
    for key, rows in grouped.items():
        expected = {None} if key[3] == "ablation_default" else set(SEEDS)
        if {row["seed"] for row in rows} != expected or len(rows) != len(expected):
            if allow_partial:
                continue
            raise ValueError(f"Invalid seed set for {key}")
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


def lesion_strata(images: dict) -> dict:
    """Per CHNCXR id: lesion area fraction at 224 px and annotation count."""
    mask_root = ROOT / "data/raw/chncxr/mergeMask"
    per_lesion = ROOT / "data/raw/chncxr/Annotations/masks"
    pattern = re.compile(r"^(CHNCXR_\d+_[01])_.+\.png$", re.IGNORECASE)
    counts = defaultdict(int)
    for path in per_lesion.glob("*.png"):
        match = pattern.match(path.name)
        if match:
            counts[match.group(1)] += 1
    strata = {}
    for image_id in sorted({key[2] for key in images if key[0] == "chncxr"}):
        path = mask_root / f"{image_id}.png"
        if not path.is_file():
            continue
        with Image.open(path) as mask:
            area = float((np.asarray(mask.convert("L").resize((224, 224),
                                                               Image.Resampling.NEAREST)) > 127).mean())
        strata[image_id] = {"area_fraction": area, "annotations": counts.get(image_id, 0)}
    return strata


def paired_test(deltas: list[float]) -> dict:
    """Wilcoxon signed-rank p value and paired rank-biserial correlation."""
    array = np.asarray(deltas, dtype=float)
    nonzero = array[array != 0]
    if not len(nonzero):
        return {"p": 1.0, "rank_biserial": 0.0, "all_zero": True}
    ranks = rankdata(np.abs(nonzero))
    positive, negative = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
    p = float(wilcoxon(array, zero_method="wilcox", alternative="two-sided").pvalue)
    return {"p": p, "rank_biserial": float((positive - negative) / (positive + negative)),
            "all_zero": False}


def holm(tests: list[dict]) -> None:
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p"])
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(tests) - rank) * tests[index]["p"]))
        tests[index]["p_holm"] = running
        tests[index]["significant"] = running < 0.05


def compare(images: dict, domain: str, model: str, left: str, right: str, metric: str,
            ids: set | None = None) -> dict:
    """Paired right - left over images where both values exist."""
    deltas = []
    for key, values in images.items():
        if key[:2] != (domain, model) or key[3] != left or (ids is not None and key[2] not in ids):
            continue
        other = images.get((domain, model, key[2], right))
        if other and values.get(metric) is not None and other.get(metric) is not None:
            deltas.append(other[metric] - values[metric])
    out = describe(deltas, stable_seed("final", domain, model, left, right, metric,
                                       len(ids) if ids else 0))
    out.update(paired_test(deltas) if deltas else {"p": None, "rank_biserial": None})
    return out


def summarize(images: dict) -> list[dict]:
    output = []
    for domain, model, variant in sorted({(k[0], k[1], k[3]) for k in images}):
        selected = [v for k, v in images.items() if (k[0], k[1], k[3]) == (domain, model, variant)]
        output.append({"domain": domain, "model": model, "variant": variant,
                       "images": len(selected),
                       **{metric: describe([s[metric] for s in selected],
                                           stable_seed("final", domain, model, variant, metric))
                          for metric in METRICS}})
    return output


def confirmatory(images: dict, stability_complete: bool) -> list[dict]:
    families = []
    for domain in DOMAINS:
        for model in MODELS:
            for metric in CONFIRMATORY:
                if metric in STABILITY_METRICS and not stability_complete:
                    continue
                methods = RANDOM if metric in SEED_METRICS else CORE
                tests = []
                for a, b in combinations(methods, 2):
                    result = compare(images, domain, model, DEFAULT[a], DEFAULT[b], metric)
                    if result["p"] is None:
                        continue
                    tests.append({"left": a, "right": b, **result})
                holm(tests)
                families.append({"domain": domain, "model": model, "metric": metric,
                                 "tests": tests})
    return families


def tuning(images: dict, roles: dict) -> list[dict]:
    output = []
    for domain in (d for d in DOMAINS if d in roles):
        for model in MODELS:
            tuned = {role.split(":")[1]: variant for variant, rs in roles[domain][model].items()
                     for role in rs if role.startswith("tuned:")}
            for method, variant in sorted(tuned.items()):
                if variant == DEFAULT[method]:
                    continue
                output.append({"domain": domain, "model": model, "method": method,
                               "default": DEFAULT[method], "tuned": variant,
                               "delta_tuned_minus_default": {
                                   metric: compare(images, domain, model, DEFAULT[method],
                                                   variant, metric) for metric in METRICS}})
    return output


def transfer(images: dict, roles: dict, strata: dict, stability_complete: bool) -> dict:
    """Optional experiment 7: CHNCXR tuned minus ImageNet-transferred cells."""
    output = {"comparisons": [], "families": []}
    if "chncxr" not in roles:
        return output
    for model in MODELS:
        role_map = defaultdict(dict)
        for variant, rs in roles["chncxr"][model].items():
            for role in rs:
                kind, method = role.split(":")
                role_map[method][kind] = variant
        regions = {k[2]: v["actual_regions"] for k, v in images.items()
                   if k[:2] == ("chncxr", model) and k[3] == DEFAULT["lime"]}
        small = {i for i, s in strata.items() if i in regions
                 and s["area_fraction"] * regions[i] < 1.0}
        groups = {"all": None, "lesion_masked": set(strata),
                  "lesion_below_one_region": small,
                  "lesion_at_least_one_region": set(strata) - small,
                  "single_annotation": {i for i, s in strata.items() if s["annotations"] == 1},
                  "multiple_annotations": {i for i, s in strata.items() if s["annotations"] > 1}}
        for method in RANDOM + ("ablation",):
            left, right = role_map[method]["transfer"], role_map[method]["tuned"]
            entry = {"model": model, "method": method, "transfer": left, "tuned": right,
                     "identical": left == right, "groups": {}}
            for name, ids in groups.items():
                if ids is not None and not ids:
                    continue
                entry["groups"][name] = {
                    "images": None if ids is None else len(ids),
                    "delta_tuned_minus_transfer": {
                        metric: compare(images, "chncxr", model, left, right, metric, ids)
                        for metric in METRICS
                        if stability_complete or metric not in STABILITY_METRICS}}
            output["comparisons"].append(entry)
        for metric in ("insertion_auc", "deletion_auc", "common_insertion_auc",
                       "common_deletion_auc", "max_sensitivity", "cosine_similarity",
                       "localization_energy", "pointing_game"):
            if metric in STABILITY_METRICS and not stability_complete:
                continue
            tests = []
            for entry in output["comparisons"]:
                if entry["model"] != model or entry["identical"]:
                    continue
                result = dict(entry["groups"]["lesion_masked" if metric in
                                              ("localization_energy", "pointing_game")
                                              else "all"]["delta_tuned_minus_transfer"][metric])
                if result.get("p") is None:
                    continue
                tests.append({"method": entry["method"], **result})
            holm(tests)
            output["families"].append({"model": model, "metric": metric, "tests": tests})
    return output


def method_ranks(summary: list[dict]) -> list[dict]:
    """Rank default cells by median per domain x model (1 = best)."""
    better_low = {"deletion_auc", "common_deletion_auc", "max_sensitivity",
                  "attribution_seconds", "forward_samples"}
    output = []
    for domain in DOMAINS:
        for model in MODELS:
            cells = {row["variant"]: row for row in summary
                     if (row["domain"], row["model"]) == (domain, model)}
            for metric in ("insertion_auc", "deletion_auc", "max_sensitivity",
                           "cosine_similarity", "attribution_seconds", "localization_energy"):
                values = {m: cells[DEFAULT[m]][metric]["median"] for m in CORE
                          if DEFAULT[m] in cells and cells[DEFAULT[m]][metric]["median"] is not None}
                if not values:
                    continue
                sign = 1 if metric in better_low else -1
                ordered = sorted(values, key=lambda m: sign * values[m])
                output.append({"domain": domain, "model": model, "metric": metric,
                               "ranking": ordered, "medians": values})
    return output


def baseline_rankings() -> list[dict]:
    """Does the explanation baseline change the method ranking? (M4 control run)."""
    path = ROOT / "results/metrics/m4_baseline_sensitivity/summary.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["domain"], row["model"], row["explanation_baseline"]][row["method"]] = row
    output = []
    for (domain, model, baseline), methods in sorted(grouped.items()):
        for metric, sign in (("normalized_insertion_auc", -1), ("normalized_deletion_auc", 1)):
            ordered = sorted(methods, key=lambda m: sign * methods[m][metric]["median"])
            output.append({"domain": domain, "model": model, "baseline": baseline,
                           "metric": metric, "ranking": ordered})
    return output


def write_report(path: Path, summary, families, tuned, exp7, ranks, baselines,
                 strata, stability_complete) -> None:
    names = {"lime": "LIME", "rise": "RISE", "ablation": "Ablation", "kernel_shap": "KernelSHAP"}
    lines = ["# M4 最终评价集结果", "",
             "范围：`final_evaluation`（ImageNet 299 张，CHNCXR 339 张），配置由 "
             "`results/metrics/m4_config_analysis/final_selection.json` 在调参集上冻结。"
             "图像为分析单位，随机方法先在图内对 5 个种子取均值。",
             f"区间为图像级 bootstrap 中位数 95% CI（{BOOTSTRAP_REPEATS} 次，基种子 {BOOTSTRAP_SEED}）。",
             "`common_*` 为在统一 SLIC=100 划分上重算的 Insertion/Deletion。", ""]
    if not stability_complete:
        lines += ["> 输入扰动稳定性阶段尚未完成，本版本不含 Max-Sensitivity / Cosine。", ""]
    for domain in DOMAINS:
        for model in MODELS:
            lines += [f"## {domain} / {model}", "",
                      "| 配置 | 图像 | Insertion | Deletion | 统一划分 Ins | 统一划分 Del | "
                      "Max-Sens | Cosine | Spearman | 定位能量 | 秒/图（独占 GPU） | 前向数 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for row in [r for r in summary if (r["domain"], r["model"]) == (domain, model)]:
                lines.append(
                    f"| {row['variant']} | {row['images']} | {fmt(row['insertion_auc']['median'])} | "
                    f"{fmt(row['deletion_auc']['median'])} | {fmt(row['common_insertion_auc']['median'])} | "
                    f"{fmt(row['common_deletion_auc']['median'])} | {fmt(row['max_sensitivity']['median'])} | "
                    f"{fmt(row['cosine_similarity']['median'])} | {fmt(row['spearman']['median'])} | "
                    f"{fmt(row['localization_energy']['median'])} | "
                    f"{fmt(row['clean_attribution_seconds']['median'], 3)} | "
                    f"{fmt(row['forward_samples']['median'], 0)} |")
            lines.append("")
    lines += ["## 确认性检验（默认设置，Holm 校正）", "",
              "差值为右方法减左方法；r 为配对秩二列相关。", "",
              "| 域 | 模型 | 指标 | 左 | 右 | 差值中位数 [95% CI] | r | p_holm |",
              "|---|---|---|---|---|---:|---:|---:|"]
    for family in families:
        for test in family["tests"]:
            mark = " *" if test["significant"] else ""
            lines.append(f"| {family['domain']} | {family['model']} | {family['metric']} | "
                         f"{names[test['left']]} | {names[test['right']]} | {ci(test)} | "
                         f"{test['rank_biserial']:.3f} | {test['p_holm']:.2e}{mark} |")
    lines += ["", "## 域内调参配置相对默认（探索性）", "",
              "| 域 | 模型 | 方法 | 调参配置 | ΔIns | ΔDel | Δ统一划分 Ins | Δ统一划分 Del | ΔMax-Sens |",
              "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for row in tuned:
        d = row["delta_tuned_minus_default"]
        lines.append(f"| {row['domain']} | {row['model']} | {names[row['method']]} | `{row['tuned']}` | "
                     f"{ci(d['insertion_auc'])} | {ci(d['deletion_auc'])} | "
                     f"{ci(d['common_insertion_auc'])} | {ci(d['common_deletion_auc'])} | "
                     f"{ci(d['max_sensitivity'])} |")
    lines += ["", "## 选做实验 7：CHNCXR 域内调参 vs ImageNet 参数直接迁移", "",
              f"有病灶并集掩码的最终评价胸片 {len(strata)} 张；差值为域内调参减直接迁移。", "",
              "| 模型 | 方法 | 迁移配置 | 调参配置 | 子集 | 图像 | ΔIns | ΔDel | Δ统一划分 Ins | "
              "Δ统一划分 Del | Δ定位能量 | ΔPointing |",
              "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for entry in exp7["comparisons"]:
        if entry["identical"]:
            continue
        for group, data in entry["groups"].items():
            d = data["delta_tuned_minus_transfer"]
            lines.append(f"| {entry['model']} | {names[entry['method']]} | `{entry['transfer']}` | "
                         f"`{entry['tuned']}` | {group} | {d['insertion_auc']['n']} | "
                         f"{ci(d['insertion_auc'])} | {ci(d['deletion_auc'])} | "
                         f"{ci(d['common_insertion_auc'])} | {ci(d['common_deletion_auc'])} | "
                         f"{ci(d['localization_energy'])} | {ci(d['pointing_game'])} |")
    lines += ["", "实验 7 检验族（每模型 × 指标，LIME 与 RISE 两个比较，Holm）：", "",
              "| 模型 | 指标 | 方法 | 差值中位数 | p_holm |", "|---|---|---|---:|---:|"]
    for family in exp7["families"]:
        for test in family["tests"]:
            lines.append(f"| {family['model']} | {family['metric']} | {names[test['method']]} | "
                         f"{fmt(test['median'])} | {test['p_holm']:.2e}"
                         f"{' *' if test['significant'] else ''} |")
    lines += ["", "## 方法排序（默认设置，按中位数，1 为最优）", "",
              "| 域 | 模型 | 指标 | 排序 |", "|---|---|---|---|"]
    for row in ranks:
        lines.append(f"| {row['domain']} | {row['model']} | {row['metric']} | "
                     + " > ".join(names[m] for m in row["ranking"]) + " |")
    lines += ["", "## 遮蔽基线是否改变方法排序（调参集基线敏感性实验）", "",
              "| 域 | 模型 | 解释基线 | 指标 | 排序 |", "|---|---|---|---|---|"]
    for row in baselines:
        lines.append(f"| {row['domain']} | {row['model']} | {row['baseline']} | {row['metric']} | "
                     + " > ".join(names[m] for m in row["ranking"]) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/metrics/m4_final_analysis")
    parser.add_argument("--require-stability", action="store_true")
    parser.add_argument("--allow-partial", action="store_true",
                        help="development only: analyze completed images of running shards")
    args = parser.parse_args()
    main_rows, stability_rows, seed_rows, roles, stability_complete = load(
        args.require_stability, args.allow_partial)
    images = image_level(main_rows, stability_rows, seed_rows, args.allow_partial)
    strata = lesion_strata(images)
    summary = summarize(images)
    families = confirmatory(images, stability_complete)
    tuned = tuning(images, roles)
    exp7 = transfer(images, roles, strata, stability_complete)
    ranks = method_ranks(summary)
    baselines = baseline_rankings()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per_image.jsonl").open("w", encoding="utf-8") as stream:
        for (domain, model, image_id, variant), values in sorted(images.items()):
            stream.write(json.dumps({"domain": domain, "model": model, "id": image_id,
                                     "variant": variant, **values,
                                     **({"lesion": strata[image_id]} if image_id in strata else {})},
                                    allow_nan=False) + "\n")
    (args.output / "summary.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "scope": "final_evaluation",
        "stability_complete": stability_complete,
        "bootstrap": {"repeats": BOOTSTRAP_REPEATS, "seed": BOOTSTRAP_SEED},
        "cells": summary, "confirmatory": families, "tuning": tuned,
        "exp7_transfer": exp7, "method_ranks": ranks, "baseline_rankings": baselines,
        "lesion_strata": strata,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.output / "REPORT.md", summary, families, tuned, exp7, ranks, baselines,
                 strata, stability_complete)
    print(json.dumps({"images": len(images), "stability_complete": stability_complete,
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
