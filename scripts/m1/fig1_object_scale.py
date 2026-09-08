"""图一：两域解释目标的尺度分布，与超像素粒度的分辨率参考线。

论证 M1 报告「约束一」：两域的粒度网格必须分别设定。
横轴取对数的面积占比，纵轴经验累积分布；竖直参考线为暂定 224x224
直接缩放输入下，名义超像素数 n 对应的平均区域面积占比（1/n）。目标面积
小于该参考值只提示粒度不足风险；SLIC 区域并非等面积，不能据此断言目标无法分辨。

用法：
    python scripts/m1/fig1_object_scale.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from concurrent.futures import ThreadPoolExecutor

import matplotlib.ticker as mticker
import numpy as np
from PIL import Image

from _common import (C_AUX, C_MED, C_NAT, chncxr_lesion_masks, ecdf,
                     imagenet_bboxes, plt, save)

N_GRID = [50, 100, 200, 300]     # M1 暂定候选网格；正式配置落库后须复核


def _one_image(paths: list) -> tuple:
    """返回该影像的（逐病灶面积占比列表, 病灶并集面积占比）。

    病灶之间可能重叠（如结节位于浸润灶内），故并集按逐像素求或，不能简单相加。
    """
    acc = None
    fracs = []
    for p in paths:
        m = np.asarray(Image.open(p).convert("L")) > 127
        fracs.append(float(m.mean()))
        acc = m if acc is None else (acc | m)
    return fracs, float(acc.mean())


def collect() -> tuple:
    bbox = np.array([v[0] for v in imagenet_bboxes().values()])

    per_image = chncxr_lesion_masks()
    lesion, union = [], []
    with ThreadPoolExecutor(8) as ex:
        for fracs, u in ex.map(_one_image, per_image.values()):
            lesion.extend(fracs)
            union.append(u)
    return bbox, np.array(lesion), np.array(union)


def main() -> None:
    bbox, lesion, union = collect()

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for vals, color, style, label in [
        (bbox, C_NAT, "-", f"ImageNet 目标 bbox（n={bbox.size}）"),
        (lesion, C_MED, "-", f"CHNCXR 单个病灶（n={lesion.size}）"),
        (union, C_MED, "--", f"CHNCXR 逐图病灶并集（n={union.size}）"),
    ]:
        x, y = ecdf(vals)
        ax.step(x * 100, y, where="post", color=color, ls=style, lw=1.6, label=label)

    for n in N_GRID:
        frac = 100.0 / n
        ax.axvline(frac, color=C_AUX, lw=0.8, ls=":")
        ax.text(frac, 0.015, f" n={n}", rotation=90, ha="center", va="bottom",
                fontsize=7.5, color=C_AUX)
    ax.text(0.5, 1.01, "竖线：暂定 224$\\times$224 直接缩放下的名义平均区域面积 1/$n$"
                       "（仅作风险参照，不代表最优粒度）",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8, color=C_AUX)

    below = (lesion < 100.0 / N_GRID[-1] / 100.0).mean()
    ax.annotate(f"{below:.0%} 的病灶面积小于\n最细粒度的名义平均区域面积",
                xy=(100.0 / N_GRID[-1], below), xytext=(0.006, 0.62),
                fontsize=8, color=C_MED,
                arrowprops=dict(arrowstyle="->", color=C_MED, lw=0.8))

    ax.set_xscale("log")
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 10, 100])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(1e-3, 100)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("目标面积占全图比例（%，对数轴）")
    ax.set_ylabel("经验累积分布")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "m1_object_scale.png")

    print(f"  ImageNet bbox 中位 {np.median(bbox):.4f}  "
          f"CHNCXR 病灶中位 {np.median(lesion):.5f}  "
          f"倍数 {np.median(bbox) / np.median(lesion):.1f}")
    print(f"  病灶并集中位 {np.median(union):.5f}")


if __name__ == "__main__":
    main()
