"""生成直接服务于实验参数选择的 M1 数据分析与插图。

分析回答两个问题：
1. 候选 SLIC/RISE 粒度能否解析 ImageNet bbox 与 CHNCXR 病灶并集；
2. 采样预算相对于 SLIC 实际产生的特征维数是否足够。

运行：
    python scripts/m1/experiment_design_analysis.py

输出：
    reports/requirements_modeling/figures/m1_spatial_resolution.png
    reports/requirements_modeling/figures/m1_slic_budget.png
    reports/requirements_modeling/analysis/experiment_design_analysis.json
    reports/requirements_modeling/analysis/experiment_design_analysis.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.patches as patches
import numpy as np
from PIL import Image
from skimage.segmentation import slic

from _common import (CHNCXR, IMAGENET, ROOT, C_AUX, C_MED, C_NAT,
                     chncxr_lesion_masks, ecdf, imagenet_bboxes, plt, save)

INPUT_SIZE = 224
N_SEGMENTS = [50, 100, 200, 300]
RISE_GRIDS = [4, 7, 10, 14]
BUDGETS = [256, 512, 1024, 2048]
SLIC_CONFIG = {
    "compactness": 10.0,
    "sigma": 1.0,
    "start_label": 0,
    "enforce_connectivity": True,
    "convert2lab": True,
}


def _resized_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize(
            (INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR
        )
        return np.asarray(image, dtype=np.float32) / 255.0


def _imagenet_mask(sample_id: str, boxes: dict) -> np.ndarray:
    _, (x0, y0, x1, y1), (width, height) = boxes[sample_id]
    left = int(np.clip(np.floor(x0 / width * INPUT_SIZE), 0, INPUT_SIZE - 1))
    top = int(np.clip(np.floor(y0 / height * INPUT_SIZE), 0, INPUT_SIZE - 1))
    right = int(np.clip(np.ceil(x1 / width * INPUT_SIZE), left + 1, INPUT_SIZE))
    bottom = int(np.clip(np.ceil(y1 / height * INPUT_SIZE), top + 1, INPUT_SIZE))
    mask = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def _chncxr_union(paths: list[str]) -> np.ndarray:
    union = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=bool)
    for path in paths:
        with Image.open(path) as mask:
            resized = mask.convert("L").resize(
                (INPUT_SIZE, INPUT_SIZE), Image.Resampling.NEAREST
            )
            union |= np.asarray(resized) > 127
    if not union.any():
        raise ValueError(f"Empty lesion union: {paths[0]}")
    return union


def _oracle_iou(labels: np.ndarray, target: np.ndarray) -> tuple[float, int]:
    """返回整区域表示目标时可达到的最大 IoU 与目标相交区域数。

    任意候选表示均是若干完整 SLIC 区域的并集。按区域内目标像素占比降序
    累加并枚举截断点，可得到所有最优候选集合中的最大 IoU。
    """
    flat_labels = labels.ravel()
    count = int(flat_labels.max()) + 1
    region_area = np.bincount(flat_labels, minlength=count).astype(np.float64)
    intersection = np.bincount(
        flat_labels[target.ravel()], minlength=count
    ).astype(np.float64)
    purity = np.divide(
        intersection,
        region_area,
        out=np.zeros_like(intersection),
        where=region_area > 0,
    )
    order = np.argsort(-purity, kind="stable")
    cumulative_intersection = np.cumsum(intersection[order])
    cumulative_false_positive = np.cumsum(
        region_area[order] - intersection[order]
    )
    denominator = float(target.sum()) + cumulative_false_positive
    iou = np.divide(
        cumulative_intersection,
        denominator,
        out=np.zeros_like(cumulative_intersection),
        where=denominator > 0,
    )
    return float(iou.max()), int((intersection > 0).sum())


def _analyze_one(task: dict) -> list[dict]:
    image = _resized_rgb(Path(task["image_path"]))
    if task["domain"] == "ImageNet":
        target = _imagenet_mask(task["sample_id"], task["boxes"])
    else:
        target = _chncxr_union(task["mask_paths"])

    target_pixels = int(target.sum())
    diameter = 2.0 * math.sqrt(target_pixels / math.pi)
    rows = []
    for nominal in N_SEGMENTS:
        labels = slic(
            image,
            n_segments=nominal,
            channel_axis=-1,
            **SLIC_CONFIG,
        )
        actual = int(np.unique(labels).size)
        oracle_iou, intersecting = _oracle_iou(labels, target)
        rows.append({
            "domain": task["domain"],
            "sample_id": task["sample_id"],
            "nominal_segments": nominal,
            "actual_segments": actual,
            "target_area_fraction": target_pixels / float(INPUT_SIZE ** 2),
            "target_equivalent_diameter_px": diameter,
            "target_intersecting_segments": intersecting,
            "oracle_region_iou": oracle_iou,
        })
    return rows


def _inventory_and_tasks() -> tuple[dict, list[dict], str]:
    imagenet_dir = Path(IMAGENET)
    chncxr_dir = Path(CHNCXR)
    boxes = imagenet_bboxes()
    natural_images = {
        path.stem: path
        for path in (imagenet_dir / "subsetEBPG").iterdir()
        if path.suffix.lower() in {".jpeg", ".jpg", ".png"}
    }
    if set(natural_images) != set(boxes):
        raise ValueError("ImageNet image / bbox ID mismatch")

    class_counts = Counter()
    xml_paths = sorted((imagenet_dir / "val_bbox").glob("*.xml"))
    for path in xml_paths:
        objects = ET.parse(path).getroot().findall("object")
        if len(objects) != 1 or not objects[0].findtext("name"):
            raise ValueError(f"Expected exactly one labelled object: {path}")
        class_counts[objects[0].findtext("name")] += 1

    lesion_paths = chncxr_lesion_masks()
    medical_images = {
        path.stem: path for path in (chncxr_dir / "CXR_png").glob("*.png")
    }
    positive_ids = {
        sample_id
        for sample_id in medical_images
        if re.fullmatch(r"CHNCXR_\d+_1", sample_id)
    }
    if set(lesion_paths) - set(medical_images):
        raise ValueError("CHNCXR masks reference missing images")
    if set(lesion_paths) - positive_ids:
        raise ValueError("CHNCXR lesion mask attached to a negative image")

    tasks = [
        {
            "domain": "ImageNet",
            "sample_id": sample_id,
            "image_path": str(natural_images[sample_id]),
            "boxes": boxes,
        }
        for sample_id in sorted(natural_images)
    ]
    tasks.extend(
        {
            "domain": "CHNCXR",
            "sample_id": sample_id,
            "image_path": str(medical_images[sample_id]),
            "mask_paths": lesion_paths[sample_id],
        }
        for sample_id in sorted(lesion_paths)
    )

    inventory = {
        "imagenet_candidate_images": len(natural_images),
        "imagenet_classes": len(class_counts),
        "imagenet_singleton_classes": sum(value == 1 for value in class_counts.values()),
        "chncxr_candidate_images": len(medical_images),
        "chncxr_positive_images": len(positive_ids),
        "chncxr_negative_images": len(medical_images) - len(positive_ids),
        "chncxr_localization_eligible_images": len(lesion_paths),
        "chncxr_unannotated_positive_images": len(positive_ids - set(lesion_paths)),
    }
    digest_payload = {
        "config": {
            "input_size": INPUT_SIZE,
            "n_segments": N_SEGMENTS,
            "rise_grids": RISE_GRIDS,
            "budgets": BUDGETS,
            "slic": SLIC_CONFIG,
        },
        "imagenet_xml": [path.name for path in xml_paths],
        "chncxr_images": sorted(medical_images),
        "chncxr_masks": sorted(
            Path(path).name for paths in lesion_paths.values() for path in paths
        ),
    }
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return inventory, tasks, digest


def _percentiles(values: np.ndarray) -> dict:
    return {
        "q25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "q75": float(np.percentile(values, 75)),
    }


def summarize(inventory: dict, rows: list[dict], digest: str) -> dict:
    summary = {
        "inventory": inventory,
        "analysis_sample": {
            "ImageNet": inventory["imagenet_candidate_images"],
            "CHNCXR": inventory["chncxr_localization_eligible_images"],
        },
        "input_size": INPUT_SIZE,
        "nominal_segments": N_SEGMENTS,
        "rise_grids": RISE_GRIDS,
        "budgets": BUDGETS,
        "slic_config": SLIC_CONFIG,
        "source_inventory_sha256": digest,
        "domain": {},
        "pooled_budget_density_q05": {},
    }
    for domain in ("ImageNet", "CHNCXR"):
        domain_rows = [row for row in rows if row["domain"] == domain]
        base_rows = [row for row in domain_rows if row["nominal_segments"] == N_SEGMENTS[0]]
        diameters = np.array(
            [row["target_equivalent_diameter_px"] for row in base_rows]
        )
        domain_summary = {
            "target_equivalent_diameter_px": _percentiles(diameters),
            "fraction_below_rise_cell_width": {
                str(grid): float((diameters < INPUT_SIZE / grid).mean())
                for grid in RISE_GRIDS
            },
            "by_nominal_segments": {},
        }
        for nominal in N_SEGMENTS:
            selected = [
                row for row in domain_rows if row["nominal_segments"] == nominal
            ]
            actual = np.array([row["actual_segments"] for row in selected])
            oracle = np.array([row["oracle_region_iou"] for row in selected])
            intersecting = np.array(
                [row["target_intersecting_segments"] for row in selected]
            )
            domain_summary["by_nominal_segments"][str(nominal)] = {
                "actual_segments": _percentiles(actual),
                "oracle_region_iou": _percentiles(oracle),
                "target_intersecting_segments": _percentiles(intersecting),
                "budget_density_median": {
                    str(budget): float(np.median(budget / (actual + 1.0)))
                    for budget in BUDGETS
                },
                "fraction_budget_below_c_plus_1": {
                    str(budget): float((budget < actual + 1).mean())
                    for budget in BUDGETS
                },
            }
        summary["domain"][domain] = domain_summary

    for nominal in N_SEGMENTS:
        actual = np.array([
            row["actual_segments"]
            for row in rows
            if row["nominal_segments"] == nominal
        ])
        summary["pooled_budget_density_q05"][str(nominal)] = {
            str(budget): float(np.percentile(budget / (actual + 1.0), 5))
            for budget in BUDGETS
        }
    return summary


def _plot_spatial(rows: list[dict], summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)
    colors = {"ImageNet": C_NAT, "CHNCXR": C_MED}
    labels = {"ImageNet": "ImageNet bbox", "CHNCXR": "CHNCXR 病灶并集"}

    for domain in ("ImageNet", "CHNCXR"):
        values = np.array([
            row["target_equivalent_diameter_px"]
            for row in rows
            if row["domain"] == domain and row["nominal_segments"] == N_SEGMENTS[0]
        ])
        x, y = ecdf(values)
        axes[0].step(x, y, where="post", lw=1.7, color=colors[domain],
                     label=f"{labels[domain]}（n={values.size}）")
    for index, grid in enumerate(RISE_GRIDS):
        width = INPUT_SIZE / grid
        axes[0].axvline(width, color=C_AUX, lw=0.8, ls=":" if index else "--")
        axes[0].text(width, 0.03 + 0.055 * index, f"g={grid}", rotation=90,
                     ha="center", va="bottom", fontsize=7, color=C_AUX)
    axes[0].set_xscale("log")
    axes[0].set_xlim(3, INPUT_SIZE)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("等面积圆直径（224 像素输入）")
    axes[0].set_ylabel("经验累积分布")
    axes[0].set_title("目标尺度与 RISE 网格单元宽度")
    axes[0].legend(loc="upper left", fontsize=7.5)

    x_values = np.array(N_SEGMENTS)
    for domain in ("ImageNet", "CHNCXR"):
        medians, q25, q75 = [], [], []
        for nominal in N_SEGMENTS:
            values = np.array([
                row["oracle_region_iou"]
                for row in rows
                if row["domain"] == domain and row["nominal_segments"] == nominal
            ])
            q25.append(np.percentile(values, 25))
            medians.append(np.median(values))
            q75.append(np.percentile(values, 75))
        axes[1].plot(x_values, medians, marker="o", ms=4, lw=1.7,
                     color=colors[domain], label=labels[domain])
        axes[1].fill_between(x_values, q25, q75, color=colors[domain], alpha=0.14)
    axes[1].set_xticks(N_SEGMENTS)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("SLIC 名义区域数")
    axes[1].set_ylabel("区域并集可达到的最大 IoU")
    axes[1].set_title("完整区域表示目标的几何上限")
    axes[1].legend(loc="lower right", fontsize=7.5)
    save(fig, "m1_spatial_resolution.png")


def _plot_budget(rows: list[dict], summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)
    colors = {"ImageNet": C_NAT, "CHNCXR": C_MED}
    for domain in ("ImageNet", "CHNCXR"):
        medians, q25, q75 = [], [], []
        for nominal in N_SEGMENTS:
            values = np.array([
                row["actual_segments"]
                for row in rows
                if row["domain"] == domain and row["nominal_segments"] == nominal
            ])
            q25.append(np.percentile(values, 25))
            medians.append(np.median(values))
            q75.append(np.percentile(values, 75))
        axes[0].plot(N_SEGMENTS, medians, marker="o", ms=4, lw=1.7,
                     color=colors[domain], label=domain)
        axes[0].fill_between(N_SEGMENTS, q25, q75,
                             color=colors[domain], alpha=0.14)
    axes[0].plot(N_SEGMENTS, N_SEGMENTS, color=C_AUX, lw=0.9, ls="--",
                 label="实际值=名义值")
    axes[0].set_xticks(N_SEGMENTS)
    axes[0].set_xlabel("SLIC 名义区域数")
    axes[0].set_ylabel("实际区域数（中位数与四分位区间）")
    axes[0].set_title("实际解释维数")
    axes[0].legend(fontsize=7.5)

    density = np.array([
        [summary["pooled_budget_density_q05"][str(nominal)][str(budget)]
         for budget in BUDGETS]
        for nominal in N_SEGMENTS
    ])
    shown = np.minimum(density, 8.0)
    image = axes[1].imshow(shown, cmap="RdYlGn", vmin=0, vmax=8, aspect="auto")
    axes[1].set_xticks(range(len(BUDGETS)), BUDGETS)
    axes[1].set_yticks(range(len(N_SEGMENTS)), N_SEGMENTS)
    axes[1].set_xlabel("归因阶段前向预算 $B$")
    axes[1].set_ylabel("SLIC 名义区域数")
    axes[1].set_title("保守采样密度：$B/(C+1)$ 的第 5 百分位")
    for row_index in range(len(N_SEGMENTS)):
        for column_index in range(len(BUDGETS)):
            value = density[row_index, column_index]
            axes[1].text(column_index, row_index, f"{value:.1f}",
                         ha="center", va="center", fontsize=8,
                         color="white" if value < 0.8 or value > 6 else "black")
            if value < 1:
                axes[1].add_patch(patches.Rectangle(
                    (column_index - 0.48, row_index - 0.48), 0.96, 0.96,
                    fill=False, edgecolor="#7f0000", linewidth=1.5,
                ))
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.03)
    colorbar.set_label("每个回归系数的样本数（8 以上截断）", fontsize=8)
    save(fig, "m1_slic_budget.png")


def export(rows: list[dict], summary: dict) -> None:
    output_dir = Path(ROOT) / "reports" / "requirements_modeling" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "per_image_by_granularity": rows}
    (output_dir / "experiment_design_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# 面向实验设计的 M1 数据分析",
        "",
        "分析只回答空间粒度与采样预算两个会改变实验设计的问题。",
        "SLIC 在 224x224 的 RGB 输入上运行；CHNCXR 使用病灶并集。",
        "",
        f"数据清单指纹：`{summary['source_inventory_sha256']}`。",
        "",
        "## 关键结果",
        "",
    ]
    for domain in ("ImageNet", "CHNCXR"):
        item = summary["domain"][domain]
        diameter = item["target_equivalent_diameter_px"]["median"]
        oracle_100 = item["by_nominal_segments"]["100"]["oracle_region_iou"]["median"]
        oracle_300 = item["by_nominal_segments"]["300"]["oracle_region_iou"]["median"]
        lines.append(
            f"- {domain}：目标等效直径中位数 {diameter:.1f} px；"
            f"SLIC 几何上限中位数由 n=100 的 {oracle_100:.3f} "
            f"变为 n=300 的 {oracle_300:.3f}。"
        )
    lines.extend([
        "",
        "## 对实验设计的约束",
        "",
        "- 粒度扫描按数据域分别报告，主设置仅作为共同参照。",
        "- SLIC 名义区域数不能替代实际区域数，逐图保存实际值。",
        "- 单因素扫描保持 n=100 时扫描预算、B=1024 时扫描粒度；"
        "不把全部粒度与预算机械地做笛卡尔积。",
        "- B/(C+1)<1 的组合不作为无正则 KernelSHAP 的确认性设置。",
        "",
        "复现：`python scripts/m1/experiment_design_analysis.py`。",
        "",
    ])
    (output_dir / "experiment_design_analysis.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="并行处理图像的线程数",
    )
    args = parser.parse_args()
    inventory, tasks, digest = _inventory_and_tasks()
    print(f"分析 {len(tasks)} 张图像、每张 {len(N_SEGMENTS)} 个粒度...", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(_analyze_one, tasks):
            rows.extend(result)
    summary = summarize(inventory, rows, digest)
    _plot_spatial(rows, summary)
    _plot_budget(rows, summary)
    export(rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
