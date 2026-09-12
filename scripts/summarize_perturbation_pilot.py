"""Generate a Chinese pilot report from saved records; no model calls or plots."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    directory = args.run_dir

    def read(name):
        return json.loads((directory/name).read_text(encoding="utf-8"))

    complete = read("complete.json")
    metadata, selection, summary = read("run.json"), read("selection.json"), read("summary.json")
    stability = read("seed_stability.json")
    rows = [json.loads(line) for line in (directory/"metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == complete["expected"] == complete["rows"]
    config = metadata["config"]
    is_pilot = metadata.get("stage", "pilot") == "pilot"
    domain = metadata.get("domain", "chncxr")
    domain_name = "ImageNet" if domain == "imagenet" else "CHNCXR"
    title = f"{domain_name} 首轮预实验" if is_pilot else f"{domain_name} 全样本主设置实验"
    joint_targets = {row["target"] for row in selection["joint_correct"]}
    selected_targets = {row["target"] for row in selection["selected"]}
    distribution_line = (
        f"- 共同正确样本类别分布：{selection['joint_correct_by_class']}（0 正常，1 结核阳性）。"
        if domain == "chncxr" else
        f"- 共同正确样本覆盖 {len(joint_targets)} 个类别；固定子集覆盖 {len(selected_targets)} 个类别。")
    lines = [f"# 三个扰动基线：{title}", "",
             ("本轮用于验证真实模型上的完整实验流程，并观察方法差异，不作为全量实验或最终方法排名。"
              if is_pilot else "本轮按 M1 主设置运行全部共同正确样本。"), "",
             f"- 完成时间（UTC）：{complete['completed_utc']}",
             f"- 候选池：{selection['candidate_count']} 张；两个模型共同正确：{selection['joint_correct_count']} 张。",
             distribution_line,
             (f"- 实际抽样：每类 {config['samples_per_class']} 张；抽样种子 {config['selection_seed']}。"
              if config.get("selection", "balanced_subset") == "balanced_subset" else
              f"- 实际抽样：共同正确集合中固定抽取 {config['sample_count']} 张；抽样种子 {config['selection_seed']}。"
              if config.get("selection") == "seeded_subset" else
              f"- 实际样本：全部 {len(selection['selected'])} 张共同正确样本。"),
             f"- LIME/RISE 种子：{config['seeds']}；Ablation 为确定性方法，每张图仅运行一次。",
             f"- 共 {len(rows)} 条方法运行记录；LIME/RISE 单次预算 {config['num_samples']}，Ablation 为实际区域数 + 1。",
             f"- GPU：{metadata['device_name']}；批大小 {config['batch_size']}；每模型预热 {config['warmup_batches']} 批。",
             f"- 预处理：RGB，{metadata['preprocessing']['resize']} 至 224×224；"
             f"mean={metadata['preprocessing']['mean']}，std={metadata['preprocessing']['std']}。",
             "- 目标：正确类别原始 logit；遮蔽：归一化输入空间的零；SLIC 在归一化前 RGB 上生成并共享。",
             "", "## 筛选检查", "",
             "准确率只用于检查模型、预处理与共同正确筛选，不作为本项目的分类性能结论。", ""]
    for model, accuracy in selection["model_accuracy"].items():
        lines.append(f"- {model} 在当前 {selection['candidate_count']} 张候选池上的分类正确率：{accuracy:.2%}。")
    lines += ["", "## 忠实性与效率", "",
              "各图先对随机种子取均值，再对图像取中位数。原始 logit AUC 不限于 [0,1]，",
              "仅在相同模型、相同样本及目标类别下比较；Insertion 越高越好，Deletion 越低越好。",
              "表内耗时不含公共预处理、评价及写盘；前向数不含筛选、预热和评价。", "",
              "| 模型 | 方法 | 图像 / 运行数 | Insertion AUC | Deletion AUC | 归因秒数 | 前向数 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        values = [row[key]["median"] for key in ("insertion_auc", "deletion_auc", "attribution_seconds", "forward_samples")]
        lines.append(f"| {row['model']} | {row['method']} | {row['images']} / {row['runs']} | "
                     + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += ["", "## 同图配对差值", "",
              "先对同图的种子取均值，再逐图计算 A−B，最后取差值中位数。",
              "Insertion 差值为正表示 A 更高，Deletion 差值为负表示 A 更低；",
              "这与分别计算两个方法中位数后相减不同。胜出张数只作描述，不作显著性判断。", "",
              "| 模型 | A − B | ΔInsertion | ΔDeletion | A 插入胜出图数 | A 删除胜出图数 |",
              "|---|---|---:|---:|---:|---:|"]
    for model in config["models"]:
        for a, b in (("lime", "ablation"), ("rise", "ablation"), ("lime", "rise")):
            differences = {"insertion_auc": [], "deletion_auc": []}
            for sid in [sample["id"] for sample in selection["selected"]]:
                for metric in differences:
                    means = [np.mean([r[metric] for r in rows if r["model"] == model
                                      and r["id"] == sid and r["method"] == method]) for method in (a, b)]
                    differences[metric].append(means[0]-means[1])
            insertion, deletion = map(np.asarray, differences.values())
            lines.append(f"| {model} | {a} − {b} | {np.median(insertion):.4f} | {np.median(deletion):.4f} | "
                         f"{int((insertion > 0).sum())}/{len(insertion)} | {int((deletion < 0).sum())}/{len(deletion)} |")
    lines += ["", "## 随机种子稳定性", "",
              "同图、同模型、同方法的全部种子对：先对种子对取均值，再对图像取中位数。",
              "常数分数的 Spearman 记为缺失；Ablation 不重复运行或参与随机稳定性比较。", "",
              "| 模型 | 方法 | Spearman | Top 20% Jaccard |",
              "|---|---|---:|---:|"]
    for model in config["models"]:
        for method in ("lime", "rise"):
            group = [r for r in stability if r["model"] == model and r["method"] == method]
            values = []
            for key in ("spearman", "top20_jaccard"):
                image_values = []
                for sid in sorted({r["id"] for r in group}):
                    valid = [r[key] for r in group if r["id"] == sid and r[key] is not None]
                    if valid:
                        image_values.append(float(np.mean(valid)))
                values.append(f"{np.median(image_values):.4f}" if image_values else "NA")
            lines.append(f"| {model} | {method} | {' | '.join(values)} |")
    lines += ["", "## 拟合诊断与使用限制", ""]
    for model in config["models"]:
        selected = [r for r in rows if r["model"] == model and r["method"] == "lime"]
        r2 = [r["diagnostics"]["weighted_r2"] for r in selected]
        lines.append(f"- {model} 的 LIME 邻域加权 R²：中位数 {np.median(r2):.4f}，"
                     f"范围 {min(r2):.4f}–{max(r2):.4f}。该指标仅衡量局部代理拟合。")
    lines += [f"- 本轮仅 {domain_name} 两个模型，尚未运行参数扫描、输入噪声稳定性或定位性评价。",
              "- RISE 使用原始 logit 替代概率；其像素图按 SLIC 区域求和参与排序。LIME/Ablation 直接使用区域系数，不额外乘区域面积。",
              "- 本轮不做显著性检验，不将种子重复或种子对当作独立图像样本。",
              "- 单次方法计时受 GPU 温度与系统负载影响；方法执行顺序按图像固定随机打乱。",
              "", "## 可追溯产物", "",
              "- `run.json`：完整配置、数据/权重/源代码 SHA256、环境与 Git 状态。",
              "- `selection.json`、`screen_*.json`：候选池推理、共同正确样本及抽样清单。",
              "- `metrics.jsonl`：逐方法、逐图、逐种子的 AUC、完整曲线、计时、预算和诊断。",
              "- `summary.json`：图像层面汇总的中位数和四分位数；`seed_stability.json`：种子对明细。",
              "- `source/`：本轮运行的源代码快照；`complete.json`：完整性计数。",
              "- `results/attributions/<run_name>/`：区域分数、SLIC 划分与 RISE 数值图；"
              "可视化由 `scripts/visualize_perturbation_run.py` 生成到 `results/figures/<run_name>/`。", ""]
    (directory/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(directory/"REPORT.md")


if __name__ == "__main__":
    main()
