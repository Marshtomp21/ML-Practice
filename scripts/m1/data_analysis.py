"""M1 初步数据分析：为实验变量取值提供依据。

只回答两个直接决定实验设置的问题：

1. 评价目标有多大，候选粒度（SLIC 名义区域数 n、RISE 网格边长 g）能不能把它划出来；
2. 采样预算 B 相对于 SLIC 实际产生的区域数够不够。

运行（数分钟，主要开销是 830 张图 x 4 个粒度的 SLIC）：
    python scripts/m1/data_analysis.py

输出两张插图，正文引用的数字一并打印到标准输出：
    reports/requirements_modeling/figures/m1_target_scale.png
    reports/requirements_modeling/figures/m1_budget.png
"""
from __future__ import annotations

import os
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
from skimage.segmentation import slic

from _common import (ROOT, CHNCXR, IMAGENET, C_AUX, C_MED, C_NAT,
                     chncxr_lesion_masks, imagenet_bboxes, np, os, plt, save)

SIZE = 224
N_GRID = [50, 100, 200, 300]          # SLIC 名义区域数
G_GRID = [4, 7, 10, 14]               # RISE 掩码网格边长
B_GRID = [256, 512, 1024, 2048]       # 模型前向预算
AUDIT_OUT = os.path.join(ROOT, "results", "tables", "m2_postscreen_audit.json")
SLIC_KW = dict(compactness=10.0, sigma=1.0, start_label=0,
               enforce_connectivity=True, convert2lab=True, channel_axis=-1)


def _rgb224(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _imagenet_target(box: tuple) -> float:
    """bbox 整体线性映射到 224x224 后的面积占比。"""
    (x0, y0, x1, y1), (w, h) = box[1], box[2]
    left = int(np.clip(np.floor(x0 / w * SIZE), 0, SIZE - 1))
    top = int(np.clip(np.floor(y0 / h * SIZE), 0, SIZE - 1))
    right = int(np.clip(np.ceil(x1 / w * SIZE), left + 1, SIZE))
    bottom = int(np.clip(np.ceil(y1 / h * SIZE), top + 1, SIZE))
    return (right - left) * (bottom - top) / float(SIZE ** 2)


def _chncxr_target(paths: list) -> float:
    """逐图病灶并集的面积占比。病灶之间可能重叠，故逐像素求或而非相加。"""
    union = np.zeros((SIZE, SIZE), dtype=bool)
    for p in paths:
        with Image.open(p) as m:
            m = m.convert("L").resize((SIZE, SIZE), Image.Resampling.NEAREST)
        union |= np.asarray(m) > 127
    return float(union.mean())


def _one(task: dict) -> dict:
    image = _rgb224(task["path"])
    return {"domain": task["domain"], "id": task["id"],
            "frac": task["frac"],
            "actual": [int(np.unique(slic(image, n_segments=n, **SLIC_KW)).size)
                       for n in N_GRID]}


def collect() -> dict:
    boxes = imagenet_bboxes()
    tasks = [{"domain": "ImageNet",
              "id": sid,
              "path": os.path.join(IMAGENET, "subsetEBPG", f"{sid}.JPEG"),
              "frac": _imagenet_target(boxes[sid])}
             for sid in sorted(boxes)]

    lesions = chncxr_lesion_masks()
    tasks += [{"domain": "CHNCXR",
               "id": sid,
               "path": os.path.join(CHNCXR, "CXR_png", f"{sid}.png"),
               "frac": _chncxr_target(lesions[sid])}
              for sid in sorted(lesions)]

    with ThreadPoolExecutor() as pool:
        rows = list(pool.map(_one, tasks))

    out = {}
    for dom in ("ImageNet", "CHNCXR"):
        sel = [r for r in rows if r["domain"] == dom]
        out[dom] = {"ids": [r["id"] for r in sel],
                    "frac": np.array([r["frac"] for r in sel]),
                    "actual": np.array([r["actual"] for r in sel])}  # (N, 4)
    return out


def _describe(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {"count": int(values.size), "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75))}


def _selection(path: str) -> dict:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _class_audit(labels: dict, selected_ids: set) -> dict:
    before, after = Counter(labels.values()), Counter(
        labels[sid] for sid in selected_ids if sid in labels)
    by_class = {}
    for label in sorted(before, key=str):
        by_class[str(label)] = {"before": before[label], "after": after[label],
                                "retention_rate": after[label] / before[label]}
    return {"classes_before": len(before), "classes_after": len(after),
            "class_coverage_retention_rate": len(after) / len(before),
            "by_class": by_class}


def postscreen_audit(data: dict) -> dict:
    """M2 recheck required by M1: compare candidate and joint-correct pools."""
    selections = {
        "ImageNet": _selection(os.path.join(ROOT, "data", "splits",
                                             "imagenet_m2_20260911", "imagenet_selection.json")),
        "CHNCXR": _selection(os.path.join(ROOT, "data", "splits",
                                           "chncxr_full_main", "chncxr_selection.json")),
    }
    boxes = imagenet_bboxes()
    imagenet_labels = {}
    for sid in boxes:
        xml_path = os.path.join(IMAGENET, "val_bbox", f"{sid}.xml")
        imagenet_labels[sid] = ET.parse(xml_path).getroot().findtext("object/name")
    chncxr_ids = [os.path.splitext(name)[0] for name in
                  os.listdir(os.path.join(CHNCXR, "CXR_png")) if name.endswith(".png")]
    chncxr_labels = {sid: int(sid.rsplit("_", 1)[1]) for sid in chncxr_ids}
    label_maps = {"ImageNet": imagenet_labels, "CHNCXR": chncxr_labels}

    result = {"protocol": {"image_size": [SIZE, SIZE], "slic_nominal_regions": N_GRID,
                            "slic": SLIC_KW,
                            "target_area_scope": "localization-eligible samples"},
              "domains": {}}
    for domain in ("ImageNet", "CHNCXR"):
        selection = selections[domain]
        joint_ids = {row["id"] for row in selection["joint_correct"]}
        domain_data = data[domain]
        eligible_ids = domain_data["ids"]
        keep = np.array([sid in joint_ids for sid in eligible_ids])
        if not keep.any():
            raise RuntimeError(f"No localization-eligible jointly correct samples for {domain}")
        actual = {}
        for index, nominal in enumerate(N_GRID):
            actual[str(nominal)] = {"before": _describe(domain_data["actual"][:, index]),
                                    "after": _describe(domain_data["actual"][keep, index])}
        result["domains"][domain] = {
            "selection_fingerprint": selection["run_fingerprint"],
            "candidate_count": selection["candidate_count"],
            "joint_correct_count": selection["joint_correct_count"],
            "sample_retention_rate": selection["joint_correct_count"] / selection["candidate_count"],
            "class_retention": _class_audit(label_maps[domain], joint_ids),
            "localization_eligible_count": {"before": len(eligible_ids),
                                            "after": int(keep.sum())},
            "target_area_fraction": {"before": _describe(domain_data["frac"]),
                                     "after": _describe(domain_data["frac"][keep])},
            "actual_slic_regions": actual,
        }
    os.makedirs(os.path.dirname(AUDIT_OUT), exist_ok=True)
    temporary = AUDIT_OUT + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, AUDIT_OUT)
    return result


def inventory() -> dict:
    """正文“分析单位”一节引用的样本构成。"""
    cls = Counter()
    d = os.path.join(IMAGENET, "val_bbox")
    for name in sorted(os.listdir(d)):
        root = ET.parse(os.path.join(d, name)).getroot()
        cls[root.find("object").findtext("name")] += 1
    ids = [os.path.splitext(f)[0]
           for f in os.listdir(os.path.join(CHNCXR, "CXR_png"))]
    positive = [i for i in ids if re.fullmatch(r"CHNCXR_\d+_1", i)]
    return {"imagenet_images": sum(cls.values()),
            "imagenet_classes": len(cls),
            "imagenet_singleton_classes": sum(1 for v in cls.values() if v == 1),
            "chncxr_images": len(ids),
            "chncxr_positive": len(positive),
            "chncxr_negative": len(ids) - len(positive),
            "chncxr_with_mask": len(chncxr_lesion_masks())}


def fig_target_scale(data: dict) -> None:
    """左：目标面积占比分布；右：一个目标里装得下几个候选单元。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.8))

    bins = np.logspace(-4, 0, 41)
    for dom, color in (("ImageNet", C_NAT), ("CHNCXR", C_MED)):
        f = data[dom]["frac"]
        ax1.hist(f, bins=bins, color=color, alpha=0.55,
                 label=f"{dom}（中位 {np.median(f) * 100:.2f}%）")
        ax1.axvline(np.median(f), color=color, lw=1.2, ls="--")
    ax1.set_xscale("log")
    ax1.set_xticks([1e-3, 1e-2, 1e-1, 1])
    ax1.set_xticklabels(["0.1%", "1%", "10%", "100%"])
    ax1.set_xlabel("评价目标占整图的面积比例")
    ax1.set_ylabel("图像数")
    ax1.set_title("(a) 目标尺度", fontsize=9)
    ax1.legend(fontsize=7.5, loc="upper left")

    # 目标覆盖的单元数 = 目标面积占比 x 候选单元总数：
    # SLIC 用逐图实测区域数，RISE 的 g x g 网格等分输入，单元总数即 g^2。
    for dom, color in (("ImageNet", C_NAT), ("CHNCXR", C_MED)):
        f, actual = data[dom]["frac"], data[dom]["actual"]
        cov = f[:, None] * actual
        x = np.median(actual, axis=0)
        ax2.fill_between(x, np.percentile(cov, 25, axis=0),
                         np.percentile(cov, 75, axis=0),
                         color=color, alpha=0.15, lw=0)
        ax2.plot(x, np.median(cov, axis=0), "-o", color=color, ms=4,
                 label=f"{dom} · SLIC")
        g2 = np.array(G_GRID, dtype=float) ** 2
        ax2.plot(g2, np.median(f) * g2, "--s", color=color, ms=4, mfc="none",
                 label=f"{dom} · RISE")
    ax2.axhline(1.0, color=C_AUX, lw=1.0)
    ax2.text(0.98, 0.06, "目标小于一个单元", transform=ax2.transAxes,
             ha="right", va="bottom", fontsize=7.5, color=C_AUX)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xticks([16, 49, 100, 196, 300])
    ax2.set_xticklabels(["16", "49", "100", "196", "300"])
    ax2.set_xlabel("候选单元总数")
    ax2.set_ylabel("目标覆盖的单元数")
    ax2.set_title("(b) 粒度能否划出目标", fontsize=9)
    ax2.legend(fontsize=7.5, loc="upper left")

    save(fig, "m1_target_scale.png")


def fig_budget(data: dict) -> None:
    """LIME 与 KernelSHAP 每个待估系数分到的扰动样本数。"""
    actual = np.concatenate([data["ImageNet"]["actual"],
                             data["CHNCXR"]["actual"]])
    c_med = np.median(actual, axis=0)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for b, style in zip(B_GRID, ["-o", "-s", "-^", "-d"]):
        ax.plot(N_GRID, b / (c_med + 1), style, ms=4, label=f"B={b}")
    ax.axhline(1.0, color=C_MED, lw=1.2)
    ax.text(0.03, 0.06, "样本数不足以定出全部系数", transform=ax.transAxes,
            fontsize=7.5, color=C_MED)
    ax.set_yscale("log")
    ax.set_xticks(N_GRID)
    ax.set_xlabel("SLIC 名义区域数 n")
    ax.set_ylabel("每个待估系数分到的扰动样本数")
    ax.legend(fontsize=7.5, ncol=2)
    save(fig, "m1_budget.png")


def main() -> None:
    print("样本构成：", inventory())
    data = collect()
    audit = postscreen_audit(data)
    print("\nM2 筛选后复核：")
    for dom, record in audit["domains"].items():
        before, after = record["target_area_fraction"]["before"], record["target_area_fraction"]["after"]
        regions = record["actual_slic_regions"]["100"]
        print(f"  {dom}: 样本保留 {record['joint_correct_count']}/{record['candidate_count']}；"
              f"目标面积中位 {before['median']*100:.2f}% -> {after['median']*100:.2f}%；"
              f"n=100 实际区域中位 {regions['before']['median']:.0f} -> {regions['after']['median']:.0f}")
    print(f"  已写入 {os.path.relpath(AUDIT_OUT, ROOT)}")
    for dom in ("ImageNet", "CHNCXR"):
        f, actual = data[dom]["frac"], data[dom]["actual"]
        print(f"\n[{dom}] 图像 {f.size} 张")
        print(f"  目标面积占比 中位 {np.median(f) * 100:.2f}%"
              f"  四分位 {np.percentile(f, 25) * 100:.2f}%"
              f"-{np.percentile(f, 75) * 100:.2f}%")
        for j, n in enumerate(N_GRID):
            c, cov = actual[:, j], f * actual[:, j]
            print(f"  n={n:<4d} 实测区域数中位 {np.median(c):3.0f}"
                  f"  目标覆盖单元数中位 {np.median(cov):6.2f}"
                  f"  不足一个单元 {(cov < 1).mean() * 100:5.1f}%")
        for g in G_GRID:
            cov = f * g * g
            print(f"  g={g:<4d} 单元总数     {g * g:3d}"
                  f"  目标覆盖单元数中位 {np.median(cov):6.2f}"
                  f"  不足一个单元 {(cov < 1).mean() * 100:5.1f}%")

    c_all = np.concatenate([data["ImageNet"]["actual"],
                            data["CHNCXR"]["actual"]])
    print("\n每个待估系数分到的扰动样本数 B/(C+1)（两域合并的中位区域数 C）：")
    for j, n in enumerate(N_GRID):
        c = float(np.median(c_all[:, j]))
        print(f"  n={n:<4d} C={c:<4.0f} "
              + "  ".join(f"B={b}: {b / (c + 1):5.1f}" for b in B_GRID))

    fig_target_scale(data)
    fig_budget(data)


if __name__ == "__main__":
    main()
