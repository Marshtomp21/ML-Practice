"""图二：超像素划分示意与实际区域数的多图汇总。

上排自然图像、下排胸片，各在超像素数 n=50/100/200 下画出划分边界，
并叠加两种真值：红色实线为标注真值（ImageNet 为 XML 中的 bbox，
CHNCXR 为逐病灶掩码轮廓），蓝色虚线为资源包/数据集另行提供的掩码轮廓
（ImageNet 的 subsetEBPG_masks、CHNCXR 的 mergeMask）。脚本除绘制代表性样本外，
还会在全数据集上核对前者是否为 bbox 的栅格化、后者是否为逐病灶掩码的并集。

叠加图按确定性规则挑选代表样本：各取该域内目标面积占比最接近中位数者。
实际区域数汇总固定随机种子 42，ImageNet 随机抽 50 张，CHNCXR 分层抽取
阳性、阴性各 25 张，并在 n=50/100/200/300 下报告实得区域数相对名义 n 的比例。

全部结果基于 M1 暂定的 224x224 直接缩放和 SLIC 参数；正式预处理及参数配置
落库后必须重新生成，不能将本图解释为最优粒度结论。

注意：本图内嵌 ILSVRC2012 与 CHNCXR 原图，属再分发，故产物不入库（见 .gitignore），
由各人本地重跑本脚本生成。

用法：
    python scripts/m1/fig2_slic_overlay.py
"""
from __future__ import annotations

import os
import sys
import hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.patheffects as pe
import numpy as np
from PIL import Image
from skimage.measure import find_contours
from skimage.segmentation import mark_boundaries, slic

from _common import (C_MED, C_NAT, CHNCXR, IMAGENET, chncxr_lesion_masks,
                     imagenet_bboxes, plt, save)

SIZE = 224                       # M1 暂定：各向异性直接缩放到 224x224
OVERLAY_N_GRID = [50, 100, 200]
SUMMARY_N_GRID = [50, 100, 200, 300]
SUMMARY_SEED = 42
SUMMARY_N_PER_DOMAIN = 50
C_TRUTH = "#d62728"              # 标注真值
C_PROVIDED = "#1f77b4"           # 数据集另行提供的掩码
C_SEG = (1.0, 1.0, 1.0)          # 超像素边界；白细线在照片与胸片上都最不喧宾夺主


def _resize_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.float64) / 255.0


def _resize_mask(path: str) -> np.ndarray:
    img = Image.open(path).convert("L").resize((SIZE, SIZE), Image.NEAREST)
    return np.asarray(img) > 127


def _segment(image: np.ndarray, n: int) -> np.ndarray:
    """按 M1 暂定参数运行 SLIC；正式配置落库后应由配置层传入。"""
    return slic(image, n_segments=n, compactness=10.0, sigma=1.0,
                start_label=1, enforce_connectivity=True,
                convert2lab=True, channel_axis=-1)


def _region_count(labels: np.ndarray) -> int:
    """使用唯一标签数，避免依赖标签必须连续的隐含假设。"""
    return int(np.unique(labels).size)


def pick_imagenet() -> dict:
    """取 bbox 面积占比最接近中位数的一张。"""
    boxes = imagenet_bboxes()
    fr = np.array([v[0] for v in boxes.values()])
    med = np.median(fr)
    img_id = min(boxes, key=lambda k: abs(boxes[k][0] - med))
    frac, (x0, y0, x1, y1), (w, h) = boxes[img_id]

    # bbox 按缩放比例线性映射到 224x224；不裁剪，故整框完整保留
    box = np.zeros((SIZE, SIZE), dtype=bool)
    c0, c1 = int(round(x0 / w * SIZE)), int(round(x1 / w * SIZE))
    r0, r1 = int(round(y0 / h * SIZE)), int(round(y1 / h * SIZE))
    box[r0:r1, c0:c1] = True
    return {
        "title": f"ImageNet  {img_id.replace('ILSVRC2012_val_', 'val_')}",
        "img": _resize_rgb(os.path.join(IMAGENET, "subsetEBPG", img_id + ".JPEG")),
        "truth": [box],
        "provided": _resize_mask(
            os.path.join(IMAGENET, "subsetEBPG_masks", img_id + ".png")),
        "frac": frac,
        "note": f"bbox 占全图 {frac * 100:.1f}%（中位）",
    }


def pick_chncxr() -> dict:
    """取病灶并集面积占比最接近中位数的一张阳性片。"""
    per_image = chncxr_lesion_masks()
    union_frac = {}
    for img_id, paths in per_image.items():
        acc = None
        for p in paths:
            m = np.asarray(Image.open(p).convert("L")) > 127
            acc = m if acc is None else (acc | m)
        union_frac[img_id] = float(acc.mean())
    med = np.median(list(union_frac.values()))
    img_id = min(union_frac, key=lambda k: abs(union_frac[k] - med))
    return {
        "title": f"CHNCXR  {img_id.replace('CHNCXR_', '')}",
        "img": _resize_rgb(os.path.join(CHNCXR, "CXR_png", img_id + ".png")),
        "truth": [_resize_mask(p) for p in per_image[img_id]],
        "provided": _resize_mask(os.path.join(CHNCXR, "mergeMask", img_id + ".png")),
        "frac": union_frac[img_id],
        "note": (f"{len(per_image[img_id])} 个病灶，并集占全图 "
                 f"{union_frac[img_id] * 100:.2f}%（中位）"),
    }


def _draw_contours(ax, mask: np.ndarray, color: str, ls: str, lw: float) -> None:
    for c in find_contours(mask.astype(float), 0.5):
        ax.plot(c[:, 1], c[:, 0], color=color, ls=ls, lw=lw)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def validate_mask_relationships() -> tuple:
    """全量核对附带掩码与原始标注的关系，返回两域 IoU 数组。"""
    nat_ious = []
    for img_id, (_, (x0, y0, x1, y1), (w, h)) in imagenet_bboxes().items():
        truth = np.zeros((h, w), dtype=bool)
        truth[int(round(y0)):int(round(y1)), int(round(x0)):int(round(x1))] = True
        provided = np.asarray(Image.open(
            os.path.join(IMAGENET, "subsetEBPG_masks", img_id + ".png")
        ).convert("L")) > 127
        if truth.shape != provided.shape:
            raise ValueError(f"ImageNet 掩码尺寸不匹配: {img_id}")
        nat_ious.append(_iou(truth, provided))

    med_ious = []
    for img_id, paths in chncxr_lesion_masks().items():
        truth = np.logical_or.reduce([
            np.asarray(Image.open(path).convert("L")) > 127 for path in paths
        ])
        provided = np.asarray(Image.open(
            os.path.join(CHNCXR, "mergeMask", img_id + ".png")
        ).convert("L")) > 127
        if truth.shape != provided.shape:
            raise ValueError(f"CHNCXR 掩码尺寸不匹配: {img_id}")
        med_ious.append(_iou(truth, provided))

    return np.asarray(nat_ious), np.asarray(med_ious)


def slic_summary_samples() -> dict:
    """返回固定抽样的两域图像路径，用于多图 SLIC 区域数汇总。"""
    rng = np.random.default_rng(SUMMARY_SEED)

    nat_dir = os.path.join(IMAGENET, "subsetEBPG")
    nat_files = sorted(f for f in os.listdir(nat_dir)
                       if f.lower().endswith((".jpeg", ".jpg", ".png")))
    nat_pick = sorted(rng.choice(nat_files, SUMMARY_N_PER_DOMAIN,
                                 replace=False).tolist())

    med_dir = os.path.join(CHNCXR, "CXR_png")
    med_files = sorted(f for f in os.listdir(med_dir)
                       if f.lower().endswith(".png"))
    pos = [f for f in med_files if f.endswith("_1.png")]
    neg = [f for f in med_files if f.endswith("_0.png")]
    n_pos = SUMMARY_N_PER_DOMAIN // 2
    med_pick = sorted(
        rng.choice(pos, n_pos, replace=False).tolist()
        + rng.choice(neg, SUMMARY_N_PER_DOMAIN - n_pos,
                     replace=False).tolist()
    )
    return {
        "ImageNet": [os.path.join(nat_dir, f) for f in nat_pick],
        "CHNCXR": [os.path.join(med_dir, f) for f in med_pick],
    }


def collect_slic_count_summary() -> dict:
    """对固定样本计算各名义 n 下的实际区域数。"""
    out = {}
    for domain, paths in slic_summary_samples().items():
        counts = np.empty((len(paths), len(SUMMARY_N_GRID)), dtype=int)
        for i, path in enumerate(paths):
            image = _resize_rgb(path)
            for j, n in enumerate(SUMMARY_N_GRID):
                counts[i, j] = _region_count(_segment(image, n))
        out[domain] = {"paths": paths, "counts": counts}
    return out


def draw_slic_count_summary(summary: dict) -> None:
    """绘制实际区域数/名义 n 的中位数与四分位区间。"""
    x = np.asarray(SUMMARY_N_GRID, dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for domain, color in (("ImageNet", C_NAT), ("CHNCXR", C_MED)):
        counts = summary[domain]["counts"]
        ratios = counts / x[None, :]
        q25, med, q75 = np.percentile(ratios, [25, 50, 75], axis=0)
        med_counts = np.median(counts, axis=0)
        ax.plot(x, med, color=color, marker="o", lw=1.5,
                label=f"{domain}（n={len(counts)}）")
        ax.fill_between(x, q25, q75, color=color, alpha=0.16)
        for n, ratio, count in zip(x, med, med_counts):
            ax.annotate(f"{count:g}", (n, ratio), xytext=(0, 6),
                        textcoords="offset points", ha="center",
                        color=color, fontsize=7.5)

    ax.axhline(1.0, color="0.45", lw=0.9, ls=":", label="名义值")
    ax.set_xticks(SUMMARY_N_GRID)
    ax.set_xlabel("名义超像素数 $n$")
    ax.set_ylabel("实际区域数 / 名义 $n$")
    ax.set_ylim(0.5, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.01, 0.02, "点：中位数；阴影：四分位区间；数字：实际区域数中位数",
            transform=ax.transAxes, fontsize=7.5, color="0.35")
    fig.tight_layout()
    save(fig, "m1_slic_actual_counts.png")


def print_slic_count_summary(summary: dict) -> None:
    for domain, values in summary.items():
        paths, counts = values["paths"], values["counts"]
        sample_ids = "\n".join(os.path.basename(p) for p in paths)
        digest = hashlib.sha256(sample_ids.encode("utf-8")).hexdigest()[:12]
        print(f"  {domain:8s} 固定样本 n={len(paths)}，ID 摘要 {digest}")
        for j, n in enumerate(SUMMARY_N_GRID):
            q25, med, q75 = np.percentile(counts[:, j], [25, 50, 75])
            print(f"    名义 n={n:3d}：实际区域数中位 {med:g}，"
                  f"IQR {q25:g}-{q75:g}，范围 "
                  f"{counts[:, j].min()}-{counts[:, j].max()}")


def main() -> None:
    rows = [pick_imagenet(), pick_chncxr()]
    fig, axes = plt.subplots(2, len(OVERLAY_N_GRID), figsize=(6.6, 4.9))

    for r, row in enumerate(rows):
        for c, n in enumerate(OVERLAY_N_GRID):
            ax = axes[r][c]
            seg = _segment(row["img"], n)
            n_regions = _region_count(seg)
            ax.imshow(mark_boundaries(row["img"], seg, color=C_SEG, mode="thin"))
            for m in row["truth"]:
                _draw_contours(ax, m, C_TRUTH, "-", 1.3)
            _draw_contours(ax, row["provided"], C_PROVIDED, (0, (3, 3)), 1.0)
            ax.text(0.98, 0.02,
                    f"实得 {n_regions} 区域\n平均区域 {100.0 / n_regions:.2f}%",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
                    color="white",
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="0.2")])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if r == 0:
                ax.set_title(f"名义 n={n}", fontsize=8)
        axes[r][0].set_ylabel(row["title"] + "\n" + row["note"], fontsize=7.5)

    handles = [plt.Line2D([], [], color=C_TRUTH, lw=1.3, label="标注真值（bbox / 逐病灶掩码）"),
               plt.Line2D([], [], color=C_PROVIDED, lw=1.0, ls=(0, (3, 3)),
                          label="数据集提供的掩码（subsetEBPG_masks / mergeMask）"),
               plt.Line2D([], [], color=C_SEG, lw=1.3, label="超像素边界",
                          path_effects=[pe.withStroke(linewidth=2.6, foreground="0.35")])]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    save(fig, "m1_slic_overlay.png")

    for row in rows:
        truth = np.logical_or.reduce(row["truth"])
        print(f"  {row['title']}  代表样本 IoU = {_iou(truth, row['provided']):.4f}")

    nat_ious, med_ious = validate_mask_relationships()
    for name, values in (("ImageNet", nat_ious), ("CHNCXR", med_ious)):
        print(f"  {name:8s} 全量 IoU：中位 {np.median(values):.4f}，"
              f"最小 {values.min():.4f}，完全重合 {(values == 1.0).sum()}/{len(values)}")

    summary = collect_slic_count_summary()
    draw_slic_count_summary(summary)
    print_slic_count_summary(summary)


if __name__ == "__main__":
    main()
