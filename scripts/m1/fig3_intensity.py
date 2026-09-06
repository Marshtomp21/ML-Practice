"""图三：两域逐图灰度统计的分布对比。

量化 M1 报告「约束二」：在暂定 224x224 直接缩放输入空间中，描述两域的
逐图亮度与全局 RMS 对比度代理（灰度标准差）。这些差异只用于提出遮蔽基线
可能造成不同分布偏移的待验证假设，不直接证明模型输出会受到何种影响。

左图为逐图灰度均值，右图为逐图灰度标准差，均在直接缩放至 224x224 后的
[0,1] 归一化灰度上计算。CHNCXR 使用全部 662 张，ImageNet 使用全部 500 张。

用法：
    python scripts/m1/fig3_intensity.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from _common import C_MED, C_NAT, CHNCXR, IMAGENET, gray01, plt, save

SIZE = 224


def _stat(path: str) -> tuple:
    a = gray01(path, size=(SIZE, SIZE))
    return float(a.mean()), float(a.std())


def collect(paths: list) -> np.ndarray:
    with ThreadPoolExecutor(8) as ex:
        return np.array(list(ex.map(_stat, paths)))


def mean_image_diagnostic(paths: list) -> tuple:
    """返回均值图空间标准差，以及它相对典型单图标准差的比例。"""
    total = np.zeros((SIZE, SIZE), dtype=np.float64)
    image_stds = []
    for path in paths:
        image = gray01(path, size=(SIZE, SIZE))
        total += image
        image_stds.append(float(image.std()))
    mean_image = total / len(paths)
    spatial_std = float(mean_image.std())
    return spatial_std, spatial_std / float(np.median(image_stds))


def main() -> None:
    med_dir = os.path.join(CHNCXR, "CXR_png")
    med_paths = [os.path.join(med_dir, f) for f in sorted(os.listdir(med_dir))
                 if f.lower().endswith(".png")]
    nat_dir = os.path.join(IMAGENET, "subsetEBPG")
    nat_paths = [os.path.join(nat_dir, f) for f in sorted(os.listdir(nat_dir))
                 if f.lower().endswith((".jpeg", ".jpg", ".png"))]
    med = collect(med_paths)
    nat = collect(nat_paths)
    med_mean_std, med_mean_ratio = mean_image_diagnostic(med_paths)
    nat_mean_std, nat_mean_ratio = mean_image_diagnostic(nat_paths)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
    for ax, col, name in zip(
            axes, (0, 1),
            ("逐图灰度均值", "全局 RMS 对比度（灰度标准差）")):
        bins = np.linspace(0, 1, 41) if col == 0 else np.linspace(0, 0.5, 41)
        for data, color, label in [(nat[:, col], C_NAT, f"ImageNet（n={len(nat)}）"),
                                   (med[:, col], C_MED, f"CHNCXR（n={len(med)}）")]:
            ax.hist(data, bins=bins, density=True, color=color, alpha=0.45, label=label)
            ax.axvline(np.median(data), color=color, lw=1.2, ls="--")
        ax.set_xlabel(name)
        ax.set_ylabel("概率密度")
    axes[0].legend(loc="upper left", fontsize=8)

    lo, hi = np.percentile(med[:, 0], [5, 95])
    axes[0].annotate(f"逐图均值 5%-95% 分位为 {lo:.2f}-{hi:.2f}；\n"
                     f"但均值图空间标准差为 {med_mean_std:.3f}，并非恒定灰场",
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
    print(f"  ImageNet  均值图空间标准差 {nat_mean_std:.3f}，"
          f"占典型单图标准差 {nat_mean_ratio:.1%}")
    print(f"  CHNCXR    均值图空间标准差 {med_mean_std:.3f}，"
          f"占典型单图标准差 {med_mean_ratio:.1%}")


if __name__ == "__main__":
    main()
