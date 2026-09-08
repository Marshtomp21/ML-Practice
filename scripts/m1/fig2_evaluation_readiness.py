"""候选集评价覆盖、类别支持度与逐图病灶标注数量。

运行：python scripts/m1/fig2_evaluation_readiness.py
输出：报告插图、逐样本 JSON、摘要 Markdown。
统计完整候选池，不替代尚未完成的模型共同正确筛选。
"""
from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

from _common import CHNCXR, IMAGENET, ROOT, C_MED, C_NAT, plt, save, chncxr_lesion_masks


def inventory() -> tuple[list[dict], list[dict], dict]:
    nat_dir, med_dir = Path(IMAGENET), Path(CHNCXR)
    nat_images = {p.stem: p for p in (nat_dir / "subsetEBPG").iterdir()
                  if p.suffix.lower() in {".jpeg", ".jpg", ".png"}}
    annotations = {p.stem: p for p in (nat_dir / "val_bbox").glob("*.xml")}
    if set(nat_images) != set(annotations):
        raise ValueError("ImageNet image / XML ID mismatch")
    natural = []
    for sample_id, path in sorted(annotations.items()):
        objects = ET.parse(path).getroot().findall("object")
        if len(objects) != 1 or not objects[0].findtext("name"):
            raise ValueError(f"Expected one labelled object: {path}")
        natural.append({"id": sample_id, "class": objects[0].findtext("name"),
                        "bbox_count": len(objects)})
    med_images = sorted((med_dir / "CXR_png").glob("*.png"))
    lesions = chncxr_lesion_masks()
    med_ids = {p.stem for p in med_images}
    if set(lesions) - med_ids:
        raise ValueError("Lesion masks reference absent images")
    medical = []
    for path in med_images:
        match = re.fullmatch(r"CHNCXR_\d+_([01])", path.stem)
        if match is None:
            raise ValueError(f"Invalid sample ID: {path}")
        positive = bool(int(match.group(1)))
        count = len(lesions.get(path.stem, []))
        merged = (med_dir / "mergeMask" / path.name).exists()
        if (count > 0) != merged or (count and not positive):
            raise ValueError(f"Inconsistent mask availability: {path}")
        medical.append({"id": path.stem, "positive": positive,
                        "lesion_annotation_count": count,
                        "localization_eligible_before_model_filter": positive and count > 0})
    prior_path = Path(ROOT) / "data/splits/chncxr_pred_samples.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    prior_ids = prior["image_ids"]
    if len(prior_ids) != len(set(prior_ids)) or set(prior_ids) - med_ids:
        raise ValueError("Prior cohort has duplicate or absent IDs")
    if prior["n_images"] != len(prior_ids):
        raise ValueError("Prior cohort metadata count mismatch")
    digest = hashlib.sha256()
    for path in sorted([*annotations.values(), prior_path]):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(json.dumps(medical, sort_keys=True).encode())
    digest.update("\n".join(sorted(str(Path(p).relative_to(med_dir))
                                  for paths in lesions.values() for p in paths)).encode())
    return natural, medical, {"prior_ids": prior_ids, "metadata_sha256": digest.hexdigest()}


def analyze(natural: list[dict], medical: list[dict], metadata: dict) -> dict:
    classes = Counter(row["class"] for row in natural)
    positive = [row for row in medical if row["positive"]]
    eligible = [row for row in positive if row["localization_eligible_before_model_filter"]]
    prior = [row for row in medical if row["id"] in set(metadata["prior_ids"])]
    counts = np.array([row["lesion_annotation_count"] for row in eligible])
    if not len(counts):
        raise ValueError("No annotated positive images")
    summary = {
        "imagenet_n": len(natural), "imagenet_classes": len(classes),
        "class_size_frequency": dict(sorted(Counter(classes.values()).items())),
        "classes_singleton": sum(v == 1 for v in classes.values()),
        "images_in_classes_le2": sum(v for v in classes.values() if v <= 2),
        "max_class_size": max(classes.values()),
        "chncxr_n": len(medical), "positive_n": len(positive),
        "negative_n": len(medical) - len(positive), "localization_n": len(eligible),
        "unannotated_positive_ids": [r["id"] for r in positive if not r["lesion_annotation_count"]],
        "prior_n": len(prior), "prior_positive_n": sum(r["positive"] for r in prior),
        "prior_localization_n": sum(r["localization_eligible_before_model_filter"] for r in prior),
        "lesion_annotation_n": int(counts.sum()), "single_annotation_n": int((counts == 1).sum()),
        "multi_annotation_n": int((counts >= 2).sum()),
        "lesion_count_median": float(np.median(counts)),
        "lesion_count_q25": float(np.percentile(counts, 25)),
        "lesion_count_q75": float(np.percentile(counts, 75)),
        "lesion_count_max": int(counts.max()),
        "lesion_count_frequency": dict(sorted(Counter(counts.tolist()).items())),
        "metadata_sha256": metadata["metadata_sha256"],
    }
    assert summary["positive_n"] + summary["negative_n"] == summary["chncxr_n"]
    assert summary["single_annotation_n"] + summary["multi_annotation_n"] == summary["localization_n"]
    return summary


def plot(s: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2), constrained_layout=True)
    values = [s["negative_n"], s["localization_n"], len(s["unannotated_positive_ids"])]
    bars = axes[0].bar(["阴性", "阳性有标注", "阳性无标注"], values,
                       color=[C_NAT, C_MED, "#999999"])
    axes[0].bar_label(bars, padding=3)
    axes[0].set_ylim(0, max(values) * 1.2)
    axes[0].set_ylabel("胸片数")
    axes[0].set_title(f"CHNCXR 候选池（n={s['chncxr_n']}）")
    ratios = [s["positive_n"] / s["chncxr_n"], s["prior_positive_n"] / s["prior_n"]]
    bars = axes[1].bar(["完整候选池", "前届预测样例"], np.array(ratios) * 100,
                       color=[C_NAT, "#E69F00"])
    axes[1].bar_label(bars, labels=[f"{x:.1%}" for x in ratios], padding=3)
    axes[1].set_ylim(0, 120)
    axes[1].set_ylabel("阳性比例（%）")
    axes[1].set_title("参考队列的类别覆盖")
    save(fig, "m1_evaluation_coverage.png")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3), constrained_layout=True)
    freq = s["class_size_frequency"]
    bars = axes[0].bar(list(freq), list(freq.values()), color=C_NAT)
    axes[0].bar_label(bars, padding=3, fontsize=8)
    axes[0].set_xticks(list(freq))
    axes[0].set_ylim(0, max(freq.values()) * 1.2)
    axes[0].set_xlabel("每个类别包含的图像数")
    axes[0].set_ylabel("类别数")
    axes[0].set_title(f"ImageNet 类别支持度（{s['imagenet_classes']} 类）")
    freq = s["lesion_count_frequency"]
    values = [freq.get(i, 0) for i in range(1, 5)]
    values.append(sum(v for k, v in freq.items() if k >= 5))
    bars = axes[1].bar(["1", "2", "3", "4", "≥5"], values, color=C_MED)
    axes[1].bar_label(bars, padding=3)
    axes[1].set_ylim(0, max(values) * 1.2)
    axes[1].set_xlabel("每张胸片的病灶标注数")
    axes[1].set_ylabel("胸片数")
    axes[1].set_title(f"有标注阳性胸片（n={s['localization_n']}）")
    save(fig, "m1_class_and_lesion_support.png")


def export(natural: list[dict], medical: list[dict], s: dict) -> None:
    out = Path(ROOT) / "reports/requirements_modeling/analysis"
    out.mkdir(exist_ok=True)
    (out / "evaluation_readiness.json").write_text(
        json.dumps({"summary": s, "imagenet": natural, "chncxr": medical},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text = (
        "# M1 候选集评价准备度分析\n\n"
        "来源：ImageNet XML 类别标注、CHNCXR 文件 ID / 病灶标注文件及前届样例清单。\n\n"
        "方法：完整候选池枚举与 ID 一致性检查；统计标注文件数量，不以连通域推断临床病灶数。\n"
        "尚未运行模型共同正确筛选，以下数量是筛选前覆盖，不是最终实验规模。\n\n"
        f"元数据指纹：{s['metadata_sha256']}。逐样本记录见同目录 JSON。\n\n"
        "## 统计结果\n\n" + json.dumps(s, ensure_ascii=False, indent=2) + "\n\n"
        "## 对实验设计的影响\n\n"
        "- 分类标签与定位掩码分别控制指标适用范围；缺失标注不等于负样本。\n"
        "- 前届全阳性队列不能替代完整候选池。\n"
        "- 类别样本稀疏时不支持逐类别显著性排名，按图像配对并审计筛选后覆盖。\n"
        "- 单/多病灶标注分层；并集定位结果不能证明所有病灶均被覆盖。\n\n"
        "复现：在 code 目录运行 python scripts/m1/fig2_evaluation_readiness.py。\n"
    )
    (out / "evaluation_readiness.md").write_text(text, encoding="utf-8")


def main() -> None:
    natural, medical, metadata = inventory()
    summary = analyze(natural, medical, metadata)
    plot(summary)
    export(natural, medical, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
