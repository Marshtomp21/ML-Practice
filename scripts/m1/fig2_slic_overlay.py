"""图二：超像素划分与真值形态的定性对照（2 行 x 3 列）。

上排自然图像、下排胸片，各在超像素数 n=50/100/200 下画出划分边界，
并叠加两种真值：红色实线为标注真值（ImageNet 为 XML 中的 bbox，
CHNCXR 为逐病灶掩码轮廓），蓝色虚线为资源包/数据集另行提供的掩码轮廓
（ImageNet 的 subsetEBPG_masks、CHNCXR 的 mergeMask）。两者重合本身就是结论：
前者说明所谓「逐像素掩码」实为 bbox 的栅格化，后者说明 mergeMask 是病灶掩码的并集。

样本按确定性规则挑选，不做主观取舍：各取该域内目标面积占比最接近其中位数者。

注意：本图内嵌 ILSVRC2012 与 CHNCXR 原图，属再分发，故产物不入库（见 .gitignore），
由各人本地重跑本脚本生成。

用法：
    python scripts/m1/fig2_slic_overlay.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.patheffects as pe
import numpy as np
from PIL import Image
from skimage.measure import find_contours
from skimage.segmentation import mark_boundaries, slic

from _common import (CHNCXR, IMAGENET, chncxr_lesion_masks, imagenet_bboxes,
                     plt, save)

SIZE = 224                       # 与固化的预处理一致：各向异性缩放到 224x224
N_GRID = [50, 100, 200]
C_TRUTH = "#d62728"              # 标注真值
C_PROVIDED = "#1f77b4"           # 数据集另行提供的掩码
C_SEG = (1.0, 1.0, 1.0)          # 超像素边界；白细线在照片与胸片上都最不喧宾夺主


def _resize_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.float64) / 255.0


def _resize_mask(path: str) -> np.ndarray:
    img = Image.open(path).convert("L").resize((SIZE, SIZE), Image.NEAREST)
    return np.asarray(img) > 127


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


def main() -> None:
    rows = [pick_imagenet(), pick_chncxr()]
    fig, axes = plt.subplots(2, len(N_GRID), figsize=(6.6, 4.9))

    for r, row in enumerate(rows):
        for c, n in enumerate(N_GRID):
            ax = axes[r][c]
            seg = slic(row["img"], n_segments=n, compactness=10.0, sigma=1.0,
                       start_label=1)
            ax.imshow(mark_boundaries(row["img"], seg, color=C_SEG, mode="thin"))
            for m in row["truth"]:
                _draw_contours(ax, m, C_TRUTH, "-", 1.3)
            _draw_contours(ax, row["provided"], C_PROVIDED, (0, (3, 3)), 1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if r == 0:
                ax.set_title(f"n={n}（实得 {seg.max()} 区域，"
                             f"单区域约 {100.0 / seg.max():.2f}%）", fontsize=7.5)
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
        inter = sum(m for m in row["truth"]).astype(bool) & row["provided"]
        uni = sum(m for m in row["truth"]).astype(bool) | row["provided"]
        print(f"  {row['title']}  真值与提供掩码 IoU = {inter.sum() / uni.sum():.4f}")


if __name__ == "__main__":
    main()
