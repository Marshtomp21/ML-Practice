"""Summarize M4 optional experiment 5 (OrdShap position attribution).

Comparisons, fixed before the formal run:

* Stability: mean pairwise cross-seed Spearman / top-20% Jaccard of OrdShap-VI,
  OrdShap-PI and patch KernelSHAP per image.
* Agreement: Spearman between seed-averaged OrdShap-VI and patch KernelSHAP.
* Position effect size: |PI| times the standard deviation of the centred slot
  index (the spread of a patch's contribution across slots implied by PI),
  relative to |VI|; and the share of patches whose PI sign agrees across all
  seeds (chance level 2 * 0.5**seeds).
* Value attribution: patch Insertion/Deletion of OrdShap-VI versus KernelSHAP.
* Position-move check: mean |target-logit change| per rule; ``ordshap_pi``
  versus each other rule is one Holm family per model (two-sided Wilcoxon).
  Deletion of the same patches is a content-removal control.

Images are the analysis unit; seeds and move repeats are never counted as
samples. Forward counts are the primary cost unit; seconds were measured while
other jobs shared the GPU and are indicative only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_m4_config import describe, fmt, ci, stable_seed  # noqa: E402
from scripts.analyze_m4_final import holm, paired_test  # noqa: E402
from evaluation.perturbation import seed_stability  # noqa: E402

OUTPUT = ROOT / "results/metrics/m4_ordshap"
MODELS = ("resnet50", "vgg13")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def vectors(rows: list[dict]) -> dict:
    """{(model, id, method): {seed: {key: array}}}"""
    out = defaultdict(dict)
    for row in rows:
        with np.load(ROOT / row["artifact"]) as array:
            out[row["model"], row["id"], row["method"]][row["seed"]] = {
                key: array[key] for key in array.files}
    return out


def mean_pairwise(per_seed: dict, key: str) -> dict:
    pairs = [seed_stability(per_seed[a][key], per_seed[b][key])
             for a, b in combinations(sorted(per_seed), 2)]
    rho = [p["spearman"] for p in pairs if p["spearman"] is not None]
    return {"spearman": float(np.mean(rho)) if rho else None,
            "top20_jaccard": float(np.mean([p["top20_jaccard"] for p in pairs]))}


def per_image(rows: list[dict], grid: int) -> list[dict]:
    store = vectors(rows)
    meta = defaultdict(list)
    for row in rows:
        meta[row["model"], row["id"], row["method"]].append(row)
    d = grid * grid
    slot_sd = float(np.std(np.arange(1, d + 1)))
    output = []
    for model, image_id in sorted({(k[0], k[1]) for k in store}):
        ordshap = store[model, image_id, "ordshap"]
        kshap = store[model, image_id, "kernel_shap_patch"]
        vi = np.mean([s["value_importance"] for s in ordshap.values()], axis=0)
        pi = np.mean([s["position_importance"] for s in ordshap.values()], axis=0)
        ks = np.mean([s["value_importance"] for s in kshap.values()], axis=0)
        signs = np.sign(np.stack([s["position_importance"] for s in ordshap.values()]))
        consistent = float(np.mean(np.all(signs == signs[:1], axis=0) & (signs[0] != 0)))
        o_rows, k_rows = meta[model, image_id, "ordshap"], meta[model, image_id, "kernel_shap_patch"]

        def avg(items, field):
            return float(np.mean([r[field] for r in items]))

        output.append({
            "model": model, "id": image_id,
            "vi_stability": mean_pairwise(ordshap, "value_importance"),
            "pi_stability": mean_pairwise(ordshap, "position_importance"),
            "kernel_shap_stability": mean_pairwise(kshap, "value_importance"),
            "vi_kernel_shap_spearman": float(spearmanr(vi, ks).statistic),
            "position_to_value_ratio": float(np.abs(pi).sum() * slot_sd / np.abs(vi).sum())
            if np.abs(vi).sum() > 0 else None,
            "pi_sign_consistent_fraction": consistent,
            "chance_sign_consistent": 2 * 0.5 ** len(ordshap),
            "ordshap_insertion_auc": avg(o_rows, "insertion_auc"),
            "ordshap_deletion_auc": avg(o_rows, "deletion_auc"),
            "kernel_shap_insertion_auc": avg(k_rows, "insertion_auc"),
            "kernel_shap_deletion_auc": avg(k_rows, "deletion_auc"),
            "ordshap_forward_samples": avg(o_rows, "forward_samples"),
            "kernel_shap_forward_samples": avg(k_rows, "forward_samples"),
            "ordshap_seconds": avg(o_rows, "attribution_seconds"),
            "kernel_shap_seconds": avg(k_rows, "attribution_seconds"),
            "original_logit": float(np.mean([r["diagnostics"]["original_logit"] for r in o_rows])),
            "permuted_full_mean": float(np.mean([r["diagnostics"]["permuted_full_mean"]
                                                 for r in o_rows])),
        })
    return output


def summarize(images: list[dict]) -> list[dict]:
    fields = {
        "vi_spearman": lambda r: r["vi_stability"]["spearman"],
        "pi_spearman": lambda r: r["pi_stability"]["spearman"],
        "kernel_shap_spearman": lambda r: r["kernel_shap_stability"]["spearman"],
        "vi_kernel_shap_spearman": lambda r: r["vi_kernel_shap_spearman"],
        "position_to_value_ratio": lambda r: r["position_to_value_ratio"],
        "pi_sign_consistent_fraction": lambda r: r["pi_sign_consistent_fraction"],
        "ordshap_insertion_auc": lambda r: r["ordshap_insertion_auc"],
        "kernel_shap_insertion_auc": lambda r: r["kernel_shap_insertion_auc"],
        "ordshap_deletion_auc": lambda r: r["ordshap_deletion_auc"],
        "kernel_shap_deletion_auc": lambda r: r["kernel_shap_deletion_auc"],
        "insertion_ordshap_minus_kernel_shap":
            lambda r: r["ordshap_insertion_auc"] - r["kernel_shap_insertion_auc"],
        "deletion_ordshap_minus_kernel_shap":
            lambda r: r["ordshap_deletion_auc"] - r["kernel_shap_deletion_auc"],
        "ordshap_forward_samples": lambda r: r["ordshap_forward_samples"],
        "kernel_shap_forward_samples": lambda r: r["kernel_shap_forward_samples"],
        "ordshap_seconds": lambda r: r["ordshap_seconds"],
        "kernel_shap_seconds": lambda r: r["kernel_shap_seconds"],
        "scramble_logit_drop": lambda r: r["original_logit"] - r["permuted_full_mean"],
    }
    output = []
    for model in MODELS:
        selected = [r for r in images if r["model"] == model]
        output.append({"model": model, "images": len(selected),
                       **{name: describe([get(r) for r in selected],
                                         stable_seed("ordshap", model, name))
                          for name, get in fields.items()}})
    return output


def moves(rows: list[dict]) -> list[dict]:
    table = {(r["model"], r["id"], r["rule"]): r for r in rows}
    rules = sorted({r["rule"] for r in rows})
    output = []
    for model in MODELS:
        ids = sorted({k[1] for k in table if k[0] == model})
        per_rule = {rule: {
            "mean_abs_move_change": describe(
                [table[model, i, rule]["mean_abs_move_change"] for i in ids],
                stable_seed("moves", model, rule)),
            "deletion_drop": describe([table[model, i, rule]["deletion_drop"] for i in ids],
                                      stable_seed("delete", model, rule)),
        } for rule in rules}
        tests = []
        for other in (r for r in rules if r != "ordshap_pi"):
            deltas = [table[model, i, "ordshap_pi"]["mean_abs_move_change"]
                      - table[model, i, other]["mean_abs_move_change"] for i in ids]
            tests.append({"left": other, "right": "ordshap_pi",
                          **describe(deltas, stable_seed("moves-test", model, other)),
                          **paired_test(deltas)})
        holm(tests)
        overlap = float(np.mean([len(set(table[model, i, "ordshap_pi"]["patches"])
                                     & set(table[model, i, "kernel_shap"]["patches"])) > 0
                                 for i in ids]))
        output.append({"model": model, "images": len(ids), "rules": per_rule,
                       "tests_pi_minus_other": tests,
                       "pi_kernel_shap_set_overlap_fraction": overlap})
    return output


def write_report(path: Path, pilot: dict, summary: list[dict], move_summary: list[dict]) -> None:
    lines = ["# M4 选做实验 5：OrdShap 位置归因（ImageNet，4×4 图块）", "",
             "按行优先编号的 4×4 等尺寸图块为特征、网格槽位为位置；缺失图块填归一化零基线，"
             "目标为正确类别 logit。实现与官方代码在完全枚举下逐位一致（`results/tables/"
             "ordshap_reference_check.json`）。", "",
             "## 预算预实验（配置选择集）", "",
             f"规则：{pilot['rule']}。是否有预算达标：{'是' if pilot['qualified'] else '否'}；"
             f"选中 `k{pilot['selected']['num_coalitions']}_p{pilot['selected']['num_permutations']}`。", "",
             "| 预算 | 联盟数 | 每联盟置换数 | VI Spearman | PI Spearman | 前向数 | 秒 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in pilot["table"]:
        lines.append(f"| {row['cell']} | {row['num_coalitions']} | {row['num_permutations']} | "
                     f"{fmt(row['median_vi_spearman'], 3)} | {fmt(row['median_pi_spearman'], 3)} | "
                     f"{fmt(row['median_forward_samples'], 0)} | {fmt(row['median_seconds'], 1)} |")
    lines += ["", "## 正式实验（最终评价集种子子集）", "",
              "图像级中位数 [bootstrap 95% CI]。", "",
              "| 指标 | " + " | ".join(r["model"] for r in summary) + " |",
              "|---|" + "---:|" * len(summary)]
    labels = {
        "vi_spearman": "OrdShap-VI 跨种子 Spearman", "pi_spearman": "OrdShap-PI 跨种子 Spearman",
        "kernel_shap_spearman": "图块 KernelSHAP 跨种子 Spearman",
        "vi_kernel_shap_spearman": "OrdShap-VI 与 KernelSHAP 的 Spearman",
        "position_to_value_ratio": "位置效应 / 值效应（Σ|PI|·sd(槽位) / Σ|VI|）",
        "pi_sign_consistent_fraction": "PI 符号在全部种子一致的图块比例",
        "insertion_ordshap_minus_kernel_shap": "图块 Insertion：OrdShap-VI − KernelSHAP",
        "deletion_ordshap_minus_kernel_shap": "图块 Deletion：OrdShap-VI − KernelSHAP",
        "ordshap_forward_samples": "OrdShap 前向数", "kernel_shap_forward_samples": "KernelSHAP 前向数",
        "ordshap_seconds": "OrdShap 秒/次（共享 GPU）", "kernel_shap_seconds": "KernelSHAP 秒/次（共享 GPU）",
        "scramble_logit_drop": "全图随机打乱后 logit 下降",
    }
    for key, label in labels.items():
        lines.append(f"| {label} | " + " | ".join(ci(r[key]) for r in summary) + " |")
    chance = 2 * 0.5 ** 5
    lines += ["", f"PI 符号一致比例的随机水平为 {chance:.4f}（5 个种子）。", "",
              "## 位置移动检验", "",
              "每条规则选 2 个图块，逐个与规则外的伙伴槽位交换（32 次，伙伴顺序在规则间共享），"
              "记录目标 logit 变化绝对值的均值；删除对照把同样的图块替换为基线。", "",
              "| 模型 | 规则 | 移动 |Δlogit| | 删除后 logit 下降 |", "|---|---|---:|---:|"]
    for item in move_summary:
        for rule, data in item["rules"].items():
            lines.append(f"| {item['model']} | {rule} | {ci(data['mean_abs_move_change'])} | "
                         f"{ci(data['deletion_drop'])} |")
    lines += ["", "| 模型 | 比较 | 差值中位数 [95% CI] | r | p_holm |", "|---|---|---:|---:|---:|"]
    for item in move_summary:
        for test in item["tests_pi_minus_other"]:
            lines.append(f"| {item['model']} | ordshap_pi − {test['left']} | {ci(test)} | "
                         f"{test['rank_biserial']:.3f} | {test['p_holm']:.2e}"
                         f"{' *' if test['significant'] else ''} |")
        lines.append(f"| {item['model']} | top-PI 与 top-KernelSHAP 图块集合有交集的图像比例 | "
                     f"{item['pi_kernel_shap_set_overlap_fraction']:.3f} | | |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--pilot", type=Path, default=ROOT / "results/metrics/m4_ordshap_pilot")
    args = parser.parse_args()
    for marker in ("complete_main.json", "complete_moves.json"):
        if not (args.output / marker).is_file():
            raise ValueError(f"Missing {marker} in {args.output}")
    pilot = json.loads((args.pilot / "budget_selection.json").read_text(encoding="utf-8"))
    config = json.loads((args.output / "run.json").read_text(encoding="utf-8"))["config"]
    images = per_image(read_jsonl(args.output / "metrics.jsonl"), config["grid"])
    summary = summarize(images)
    move_summary = moves(read_jsonl(args.output / "moves.jsonl"))
    (args.output / "per_image_summary.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in images), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "pilot": pilot,
        "summary": summary, "moves": move_summary,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.output / "REPORT.md", pilot, summary, move_summary)
    print(json.dumps({"images": len(images), "output": str(args.output)}))


if __name__ == "__main__":
    main()
