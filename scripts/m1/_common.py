"""M1 报告插图的公共设置：路径、字体、配色与保存。

三个绘图脚本共用本模块。原始影像不入库，因此插图也由各人本地重跑生成，
脚本本身是唯一入库的产物。
"""
from __future__ import annotations

import io
import os
import xml.etree.ElementTree as ET

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  必须在 use("Agg") 之后

Image.MAX_IMAGE_PIXELS = None

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHNCXR = os.path.join(ROOT, "data", "raw", "chncxr")
IMAGENET = os.path.join(ROOT, "data", "raw", "imagenet")
FIGDIR = os.path.join(ROOT, "reports", "requirements_modeling", "figures")

# 报告用 XeLaTeX + ctexbook 编译，插图需自带中文字形
plt.rcParams.update({
    "font.sans-serif": ["SimHei", "Microsoft YaHei"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "mathtext.fontset": "dejavusans",
})

C_NAT = "#1f77b4"        # 自然图像域
C_MED = "#d62728"        # 医学影像域
C_AUX = "#7f7f7f"        # 参考线


DPI = 300                # 300 dpi 下 6.5 英寸宽的图约 1950 像素，够正文印刷


def save(fig, name: str) -> str:
    """保存为 PNG，供 \\includegraphics 直接引用，也可脱离 TeX 直接预览。"""
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"已生成 {os.path.relpath(path, ROOT)}")
    return path


def mask_area_fraction(path: str) -> float:
    """二值掩码占其画布的面积比例。掩码与原图同尺寸，故等于占全图比例。"""
    return float((np.asarray(Image.open(path).convert("L")) > 127).mean())


def imagenet_bboxes() -> dict:
    """读取 500 个标注，返回 {图像 id: (面积占比, (xmin, ymin, xmax, ymax), (宽, 高))}。

    子集的筛选规则是「仅 1 个 object」，实测 500/500 成立，故每图只取第一个框。
    """
    out = {}
    for name in sorted(os.listdir(os.path.join(IMAGENET, "val_bbox"))):
        root = ET.parse(os.path.join(IMAGENET, "val_bbox", name)).getroot()
        size = root.find("size")
        w, h = int(size.find("width").text), int(size.find("height").text)
        box = root.find("object").find("bndbox")
        x0, y0, x1, y1 = (float(box.find(k).text)
                          for k in ("xmin", "ymin", "xmax", "ymax"))
        out[os.path.splitext(name)[0]] = ((x1 - x0) * (y1 - y0) / (w * h),
                                          (x0, y0, x1, y1), (w, h))
    return out


def chncxr_lesion_masks() -> dict:
    """返回 {阳性影像 id: [逐病灶掩码路径, ...]}，覆盖 330 张影像的 1153 个掩码。"""
    import re
    pat = re.compile(r"^(CHNCXR_\d+_[01])_.+\.png$", re.IGNORECASE)
    d = os.path.join(CHNCXR, "Annotations", "masks")
    out: dict = {}
    for name in sorted(os.listdir(d)):
        m = pat.match(name)
        if m:
            out.setdefault(m.group(1), []).append(os.path.join(d, name))
    return out


def chncxr_intensity_sample(n_per_class: int = 60, seed: int = 42) -> list:
    """按类别分层抽样影像文件名。与 M1 报告正文所引统计使用同一抽样口径。"""
    files = sorted(f for f in os.listdir(os.path.join(CHNCXR, "CXR_png"))
                   if f.endswith(".png"))
    rng = np.random.default_rng(seed)
    pos = [f for f in files if f.endswith("_1.png")]
    neg = [f for f in files if f.endswith("_0.png")]
    return (list(rng.choice(pos, n_per_class, replace=False))
            + list(rng.choice(neg, n_per_class, replace=False)))


def gray01(path: str, size: tuple | None = None) -> np.ndarray:
    """读为 [0,1] 灰度数组，可选先缩放到指定尺寸。"""
    img = Image.open(path).convert("L")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def ecdf(values) -> tuple:
    x = np.sort(np.asarray(values, dtype=float))
    return x, np.arange(1, x.size + 1) / x.size


__all__ = ["ROOT", "CHNCXR", "IMAGENET", "FIGDIR", "C_NAT", "C_MED", "C_AUX",
           "plt", "np", "io", "os", "Image", "save", "mask_area_fraction",
           "imagenet_bboxes", "chncxr_lesion_masks", "chncxr_intensity_sample",
           "gray01", "ecdf"]
