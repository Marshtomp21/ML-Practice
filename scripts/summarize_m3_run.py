"""Generate the Markdown M3 result report from a merged default run."""

import argparse
import json
from pathlib import Path

import numpy as np

METHOD_ORDER = ("ablation", "lime", "rise", "kernel_shap")


def format_metric(record, key):
    value = record.get(key)
    return "—" if value is None else f"{value['median']:.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()

    def read(name):
        return json.loads((run / name).read_text(encoding="utf-8"))

    metadata = read("run.json")
    selection = read("selection.json")
    complete = read("complete.json")
    summary = read("summary.json")
    seed_rows = read("seed_stability.json")
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line]
    config = metadata["config"]
    domain_name = "ImageNet" if metadata["domain"] == "imagenet" else "CHNCXR"
    by_key = {(row["model"], row["method"]): row for row in summary}

    lines = [
        f"# M3 默认设置结果：{domain_name}", "",
        f"- 完成时间（UTC）：{complete['completed_utc']}",
        f"- 候选池：{selection['candidate_count']} 张；共同正确：{selection['joint_correct_count']} 张；固定评价样本：{len(selection['selected'])} 张。",
        f"- 方法：Ablation、LIME、RISE、KernelSHAP；模型：{', '.join(config['models'])}。",
        f"- 随机方法种子：{config['seeds']}；归因预算：{config['num_samples']}；输入稳定性：ε={config['stability_epsilon']}、K={config['stability_repeats']}。",
        f"- 完整结果：{complete['rows']}/{complete['expected']} 条。", "",
        "## 忠实性、稳定性、定位性与效率", "",
        "随机方法先在同图内对种子取均值，再对图像取中位数。Insertion 越高、Deletion 越低、Max-Sensitivity 越低、Cosine Similarity 越高。", "",
        "| 模型 | 方法 | 图像 / 运行 | Insertion | Deletion | Max-Sens | Cosine | 掩码内能量 | Pointing | 归因秒数 | 前向数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in config["models"]:
        for method in METHOD_ORDER:
            record = by_key[(model, method)]
            fields = [format_metric(record, key) for key in (
                "insertion_auc", "deletion_auc", "max_sensitivity", "cosine_similarity",
                "localization_energy", "pointing_game", "attribution_seconds", "forward_samples",
            )]
            lines.append(
                f"| {model} | {method} | {record['images']} / {record['runs']} | "
                + " | ".join(fields) + " |"
            )

    lines += ["", "## 跨种子稳定性", "",
              "| 模型 | 方法 | Spearman | Top 20% Jaccard |",
              "|---|---|---:|---:|"]
    for model in config["models"]:
        for method in ("lime", "rise", "kernel_shap"):
            group = [row for row in seed_rows if row["model"] == model and row["method"] == method]
            values = []
            for key in ("spearman", "top20_jaccard"):
                per_image = []
                for sample_id in sorted({row["id"] for row in group}):
                    valid = [row[key] for row in group if row["id"] == sample_id
                             and row[key] is not None]
                    if valid:
                        per_image.append(float(np.mean(valid)))
                values.append(float(np.median(per_image)))
            lines.append(f"| {model} | {method} | {values[0]:.4f} | {values[1]:.4f} |")

    lines += ["", "## KernelSHAP 数值诊断", "",
              "| 模型 | 效率残差绝对值最大值 | 满秩运行 | 加权 R² 中位数 |",
              "|---|---:|---:|---:|"]
    for model in config["models"]:
        selected = [row for row in rows if row["model"] == model and row["method"] == "kernel_shap"]
        residual = max(abs(row["diagnostics"]["efficiency_residual"]) for row in selected)
        full_rank = sum(row["diagnostics"]["design_rank"] == row["actual_regions"] for row in selected)
        r2 = np.median([row["diagnostics"]["weighted_r2"] for row in selected])
        lines.append(f"| {model} | {residual:.3e} | {full_rank}/{len(selected)} | {r2:.4f} |")

    lines += ["", "## 可追溯产物", "",
              "- `metrics.jsonl`：逐图、逐模型、逐方法、逐种子指标和预算。",
              "- `summary.json`：先按图汇总种子，再计算中位数与四分位区间。",
              "- `seed_stability.json`：随机方法全部种子对的 Spearman 与 Top 20% Jaccard。",
              "- `selection.json`：候选池推理、共同正确集合与固定评价样本。",
              "- `run.json` 与 `source/`：运行配置、版本、哈希和源代码快照。",
              "- `complete.json`：结果完整性计数。"]
    (run / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(run / "REPORT.md")


if __name__ == "__main__":
    main()
