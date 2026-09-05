"""图三：两域逐图灰度统计的分布对比。

论证 M1 报告「约束二」：CHNCXR 的逐图亮度与对比度都很集中，
「数据集均值图」基线在该域会退化为接近常数的灰场，与「均值色」基线几乎重合，
因此医学域的基线消融应以模糊与噪声为主。这是一个可证伪的预先判断，M3 直接检验。

左图为逐图灰度均值，右图为逐图灰度标准差，均在 [0,1] 归一化灰度上计算。
CHNCXR 按类别分层抽 120 张（与报告正文所引统计同一口径），ImageNet 用全部 500 张。

用法：
    python scripts/m1/fig3_intensity.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from _common import (C_MED, C_NAT, CHNCXR, IMAGENET, chncxr_intensity_sample,
                     gray01, plt, save)


def _stat(path: str) -> tuple:
    a = gray01(path)
    return float(a.mean()), float(a.std())


def collect(paths: list) -> np.ndarray:
    with ThreadPoolExecutor(8) as ex:
        return np.array(list(ex.map(_stat, paths)))


def main() -> None:
    med = collect([os.path.join(CHNCXR, "CXR_png", f)
                   for f in chncxr_intensity_sample()])
    nat_dir = os.path.join(IMAGENET, "subsetEBPG")
    nat = collect([os.path.join(nat_dir, f) for f in sorted(os.listdir(nat_dir))])

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
    for ax, col, name in zip(axes, (0, 1), ("逐图灰度均值", "逐图灰度标准差")):
        bins = np.linspace(0, 1, 41) if col == 0 else np.linspace(0, 0.5, 41)
        for data, color, label in [(nat[:, col], C_NAT, f"ImageNet（n={len(nat)}）"),
                                   (med[:, col], C_MED, f"CHNCXR（n={len(med)}）")]:
            ax.hist(data, bins=bins, density=True, color=color, alpha=0.45, label=label)
            ax.axvline(np.median(data), color=color, lw=1.2, ls="--")
        ax.set_xlabel(name)
        ax.set_ylabel("概率密度")
    axes[0].legend(loc="upper left", fontsize=8)

    lo, hi = np.percentile(med[:, 0], [5, 95])
    axes[0].annotate(f"5%-95% 分位仅 {lo:.2f}-{hi:.2f}，\n均值图基线在此退化为常数灰场",
                     xy=(hi, 1.0), xytext=(0.02, 0.62),
                     textcoords="axes fraction", fontsize=7.5, color=C_MED,
                     arrowprops=dict(arrowstyle="->", color=C_MED, lw=0.8))

    fig.tight_layout()
    save(fig, "m1_intensity.png")

    for name, arr in (("ImageNet", nat), ("CHNCXR", med)):
        p5m, p95m = np.percentile(arr[:, 0], [5, 95])
        print(f"  {name:9s} 均值中位 {np.median(arr[:, 0]):.3f} "
              f"(5%-95% {p5m:.3f}-{p95m:.3f})  "
              f"标准差中位 {np.median(arr[:, 1]):.3f}  "
              f"均值的跨图标准差 {arr[:, 0].std():.3f}")


if __name__ == "__main__":
    main()
