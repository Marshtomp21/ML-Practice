"""Draw the M4 report figures from the analysis summaries (no model calls).

Figures go to ``reports/m4/figures``. Chinese labels need a CJK font: put
Noto Sans SC (OFL, https://fonts.google.com/noto/specimen/Noto+Sans+SC) in
``reports/fonts/`` or have SimHei installed. Method colours follow a fixed
categorical order validated for colour-vision deficiency; every line also has
a distinct marker and a direct label, so identity never relies on colour alone.

    python scripts/plot_m4.py [config|final|ordshap|baseline|localization|all]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "reports/m4/figures"
DOMAINS = ("imagenet", "chncxr")
MODELS = ("resnet50", "vgg13")
PANEL = {("imagenet", "resnet50"): "ImageNet · ResNet-50", ("imagenet", "vgg13"): "ImageNet · VGG13",
         ("chncxr", "resnet50"): "CHNCXR · ResNet-50", ("chncxr", "vgg13"): "CHNCXR · VGG13"}
# Categorical slots 1-4 of the validated reference palette, fixed per method.
COLOR = {"lime": "#2a78d6", "rise": "#eb6834", "kernel_shap": "#1baf7a", "ablation": "#eda100",
         "ordshap": "#2a78d6"}
MARKER = {"lime": "o", "rise": "s", "kernel_shap": "^", "ablation": "D"}
NAME = {"lime": "LIME", "rise": "RISE", "kernel_shap": "KernelSHAP", "ablation": "Ablation"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
METRIC = {"insertion_auc": "Insertion AUC ↑", "deletion_auc": "Deletion AUC ↓",
          "max_sensitivity": "Max-Sensitivity ↓", "cosine_similarity": "Cosine ↑",
          "spearman": "跨种子 Spearman ↑", "common_insertion_auc": "统一划分 Insertion ↑",
          "common_deletion_auc": "统一划分 Deletion ↓", "attribution_seconds": "秒/图 ↓"}


def setup() -> None:
    for path in sorted((ROOT / "reports/fonts").glob("*.otf")):
        font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({
        "font.sans-serif": ["Noto Sans SC", "SimHei", "Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 8.5, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 5,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    })


def save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / name, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {(FIGDIR / name).relative_to(ROOT)}")


def stat(cell: dict, metric: str) -> tuple[float, float, float]:
    value = cell[metric]
    low, high = value["median_ci95"] or (np.nan, np.nan)
    return value["median"], low, high


PENDING: dict = {}


def end_label(ax, x, y, text, color=None) -> None:
    """Queue a direct label; ``place_labels`` spreads them so they never overlap."""
    PENDING.setdefault(ax, []).append((x, y, text))


def place_labels(fig, gap: float = 0.075) -> None:
    fig.canvas.draw()
    for ax, labels in PENDING.items():
        to_axes = ax.transAxes.inverted()
        items = sorted(((float(to_axes.transform(ax.transData.transform((x, y)))[1]), x, text)
                        for x, y, text in labels), key=lambda item: item[0])
        placed = []
        for y, x, text in items:
            y = max(y, placed[-1] + gap) if placed else y
            placed.append(y)
        overflow = max(0.0, placed[-1] - 1.0) if placed else 0.0
        for (raw, x, text), y in zip(items, placed):
            y -= overflow
            transform = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.annotate(text, (x, y), xycoords=transform, xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=7.5, color=INK,
                        annotation_clip=False)
    PENDING.clear()


def legend_row(fig, methods) -> None:
    handles = [matplotlib.lines.Line2D([], [], color=COLOR[m], marker=MARKER[m],
                                       markeredgecolor="#fcfcfb", label=NAME[m]) for m in methods]
    fig.legend(handles=handles, loc="upper right", ncol=len(methods), bbox_to_anchor=(1.0, 1.005),
               fontsize=8, handlelength=2.2)


def line_with_band(ax, xs, cells, metric, method, label=True) -> None:
    values = np.array([stat(cell, metric) for cell in cells], dtype=float)
    ax.fill_between(xs, values[:, 1], values[:, 2], color=COLOR[method], alpha=0.15, linewidth=0)
    ax.plot(xs, values[:, 0], color=COLOR[method], marker=MARKER[method],
            markeredgecolor="#fcfcfb", markeredgewidth=1.2, markersize=6)
    if label:
        end_label(ax, xs[-1], values[-1, 0], NAME[method], COLOR[method])


def config_figures() -> None:
    summary = json.loads((ROOT / "results/metrics/m4_config_analysis/summary.json").read_text())
    cells = {(c["domain"], c["model"], c["variant"]): c for c in summary["cells"]}

    # Experiment 1: quality versus forward budget.
    metrics = ("insertion_auc", "deletion_auc", "max_sensitivity", "spearman")
    fig, axes = plt.subplots(len(metrics), 4, figsize=(10.5, 9.2), sharex=True)
    budgets = [256, 512, 1024, 2048]
    for col, key in enumerate(PANEL):
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            for method in ("lime", "rise", "kernel_shap"):
                line_with_band(ax, budgets, [cells[(*key, f"{method}_budget_{b}")] for b in budgets],
                               metric, method, label=col == 3)
            if metric != "spearman":
                ablation = cells[(*key, "ablation_default")]
                median, low, high = stat(ablation, metric)
                forward = ablation["forward_samples"]["median"]
                ax.errorbar([forward], [median], yerr=[[median - low], [high - median]],
                            color=COLOR["ablation"], marker=MARKER["ablation"], markersize=6,
                            markeredgecolor="#fcfcfb", capsize=0, linewidth=1.2)
                if col == 3:
                    end_label(ax, forward, median, "Ablation", COLOR["ablation"])
            ax.set_xscale("log", base=2)
            ax.set_xticks([64, 256, 512, 1024, 2048], ["64", "256", "512", "1k", "2k"])
            ax.minorticks_off()
            if row == 0:
                ax.set_title(PANEL[key], fontsize=9)
            if col == 0:
                ax.set_ylabel(METRIC[metric])

    fig.suptitle("实验 1：质量—预算曲线（调参集，图像级中位数与 95% CI）", fontsize=10, x=0.01, ha="left")
    legend_row(fig, ("lime", "rise", "kernel_shap", "ablation"))
    fig.supxlabel("模型前向预算（次/图，log 轴；Ablation 为实际区域数）", fontsize=8.5)
    fig.tight_layout()
    place_labels(fig)
    save(fig, "m4_exp1_budget.png")

    # Experiment 2: granularity as number of explanation units.
    metrics = ("insertion_auc", "deletion_auc", "max_sensitivity", "spearman")
    fig, axes = plt.subplots(len(metrics), 4, figsize=(10.5, 9.2))
    for col, key in enumerate(PANEL):
        lime_ids = ["lime_slic_50", "lime_budget_1024", "lime_slic_200", "lime_slic_300"]
        rise_ids = ["rise_grid_4", "rise_budget_1024", "rise_grid_10", "rise_grid_14"]
        lime_x = [cells[(*key, v)]["actual_regions"]["median"] for v in lime_ids]
        rise_x = [g * g for g in (4, 7, 10, 14)]
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            line_with_band(ax, lime_x, [cells[(*key, v)] for v in lime_ids], metric, "lime",
                           label=col == 3)
            line_with_band(ax, rise_x, [cells[(*key, v)] for v in rise_ids], metric, "rise",
                           label=col == 3)
            ax.set_xscale("log")
            ax.set_xticks([16, 49, 100, 196], ["16", "49", "100", "196"])
            ax.minorticks_off()
            if row == 0:
                ax.set_title(PANEL[key], fontsize=9)
            if col == 0:
                ax.set_ylabel(METRIC[metric])

    fig.suptitle("实验 2：粒度扫描（预算 1024；LIME 的评价划分随 SLIC 变化，RISE 固定 SLIC=100）",
                 fontsize=10, x=0.01, ha="left", y=1.02)
    legend_row(fig, ("lime", "rise"))
    fig.supxlabel("解释单元数（log 轴；LIME 为实际 SLIC 区域数中位数，RISE 为 g²）", fontsize=8.5)
    fig.tight_layout()
    place_labels(fig)
    save(fig, "m4_exp2_granularity.png")

    # Experiment 2: mask rate.
    metrics = ("insertion_auc", "deletion_auc", "max_sensitivity")
    fig, axes = plt.subplots(len(metrics), 4, figsize=(10.5, 6.6))
    rates = [0.25, 0.5, 0.75]
    for col, key in enumerate(PANEL):
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            for method in ("lime", "rise"):
                ids = [f"{method}_mask_0.25", f"{method}_budget_1024", f"{method}_mask_0.75"]
                line_with_band(ax, rates, [cells[(*key, v)] for v in ids], metric, method,
                               label=col == 3)
            ax.set_xticks(rates)
            if row == 0:
                ax.set_title(PANEL[key], fontsize=9)
            if col == 0:
                ax.set_ylabel(METRIC[metric])
            if row == len(metrics) - 1:
                ax.set_xlabel("遮蔽率 / RISE 保留概率")
    fig.suptitle("实验 2：遮蔽率扫描（调参集）", fontsize=10, x=0.01, ha="left")
    legend_row(fig, ("lime", "rise"))
    fig.tight_layout()
    place_labels(fig)
    save(fig, "m4_exp2_mask_rate.png")

    # Experiment 3: share of images where the added method is not dominated.
    exp3 = {(r["domain"], r["model"], r["budget"]): r for r in summary["exp3"]}
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.6), sharey=True)
    width = 0.36
    for col, key in enumerate(PANEL):
        ax = axes[col]
        x = np.arange(len(budgets))
        rise = [exp3[(*key, b)]["rise_new_frontier_fraction"] for b in budgets]
        shap = [exp3[(*key, b)]["shap_new_frontier_fraction"] for b in budgets]
        ax.bar(x - width / 2 - 0.01, rise, width, color=COLOR["rise"], label="+RISE")
        ax.bar(x + width / 2 + 0.01, shap, width, color=COLOR["kernel_shap"], label="+KernelSHAP")
        ax.set_xticks(x, [str(b) for b in budgets])
        ax.set_ylim(0, 1)
        ax.set_title(PANEL[key], fontsize=9)
        ax.set_xlabel("预算")
        ax.grid(axis="x", visible=False)
        if col == 0:
            ax.set_ylabel("新方法不被支配的图像比例")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("实验 3：LIME → +RISE → +KernelSHAP 的边际前沿贡献", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "m4_exp3_frontier.png")


def final_figures() -> None:
    path = ROOT / "results/metrics/m4_final_analysis/summary.json"
    summary = json.loads(path.read_text())
    cells = {(c["domain"], c["model"], c["variant"]): c for c in summary["cells"]}
    defaults = {"lime": "lime_budget_1024", "rise": "rise_budget_1024",
                "kernel_shap": "kernel_shap_budget_1024", "ablation": "ablation_default"}
    metrics = ["insertion_auc", "deletion_auc", "attribution_seconds"]
    if summary["stability_complete"]:
        metrics[2:2] = ["max_sensitivity", "cosine_similarity"]
    methods = ("lime", "rise", "ablation", "kernel_shap")
    fig, axes = plt.subplots(len(metrics), 4, figsize=(10.5, 1.9 * len(metrics) + 0.6))
    for col, key in enumerate(PANEL):
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            for index, method in enumerate(methods):
                cell = cells.get((*key, defaults[method]))
                if cell is None:
                    continue
                median, low, high = stat(cell, metric)
                ax.errorbar([median], [index], xerr=[[median - low], [high - median]],
                            color=COLOR[method], marker=MARKER[method], markersize=6,
                            markeredgecolor="#fcfcfb", linewidth=1.5, capsize=0)
            ax.set_yticks(range(len(methods)), [NAME[m] for m in methods] if col == 0 else [])
            ax.set_ylim(-0.6, len(methods) - 0.4)
            ax.invert_yaxis()
            ax.grid(axis="y", visible=False)
            if metric == "attribution_seconds":
                ax.set_xscale("log")
            ax.set_xlabel(METRIC[metric], fontsize=8)
            if row == 0:
                ax.set_title(PANEL[key], fontsize=9)
    fig.suptitle("最终评价集：默认设置四种方法（中位数与 95% CI）", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "m4_final_methods.png")

    # Optional experiment 7: tuned minus transferred, native vs common partition.
    exp7 = [entry for entry in summary["exp7_transfer"]["comparisons"] if not entry["identical"]]
    if not exp7:
        return
    metrics = ("insertion_auc", "common_insertion_auc", "deletion_auc", "common_deletion_auc",
               "localization_energy")
    fig, axes = plt.subplots(1, len(metrics), figsize=(10.5, 2.4), sharey=True)
    labels = [f"{NAME[e['method']]} · {e['model']}\n{e['transfer']} → {e['tuned']}" for e in exp7]
    for col, metric in enumerate(metrics):
        ax = axes[col]
        for index, entry in enumerate(exp7):
            group = "lesion_masked" if metric == "localization_energy" else "all"
            d = entry["groups"][group]["delta_tuned_minus_transfer"][metric]
            if d["median"] is None:
                continue
            low, high = d["median_ci95"]
            ax.errorbar([d["median"]], [index], xerr=[[d["median"] - low], [high - d["median"]]],
                        color=COLOR[entry["method"]], marker=MARKER[entry["method"]],
                        markersize=6, markeredgecolor="#fcfcfb", linewidth=1.5)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        if col == 0:
            ax.set_yticks(range(len(exp7)), labels, fontsize=7)
            ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Δ " + METRIC.get(metric, "定位能量 ↑"), fontsize=8)
    fig.suptitle("选做实验 7：CHNCXR 域内调参减 ImageNet 参数直接迁移（逐图配对差值中位数与 95% CI）",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "m4_exp7_transfer.png")


def ordshap_figures() -> None:
    path = ROOT / "results/metrics/m4_ordshap/summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    pilot_rows = sorted(summary["pilot"]["table"], key=lambda r: r["median_forward_samples"])
    moves_by_model = {m["model"]: m for m in summary["moves"]}

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})

    # (a) Pilot budget convergence for VI and PI Spearman.
    ax = axes[0]
    xs = np.arange(len(pilot_rows))
    vi_vals = [r["median_vi_spearman"] for r in pilot_rows]
    pi_vals = [r["median_pi_spearman"] for r in pilot_rows]
    xlabels = [f"{r['cell']}\n({r['median_forward_samples'] / 1000:.1f}k)" for r in pilot_rows]

    ax.axhline(0.90, color=COLOR["lime"], linestyle="--", linewidth=0.9, alpha=0.65)
    ax.axhline(0.80, color=COLOR["rise"], linestyle="--", linewidth=0.9, alpha=0.65)
    ax.plot(xs, vi_vals, color=COLOR["lime"], marker="o", markeredgecolor="#fcfcfb",
            markeredgewidth=1.2, markersize=6, label="VI Spearman")
    ax.plot(xs, pi_vals, color=COLOR["rise"], marker="s", markeredgecolor="#fcfcfb",
            markeredgewidth=1.2, markersize=6, label="PI Spearman")
    for i, (x, vi, pi) in enumerate(zip(xs, vi_vals, pi_vals)):
        ax.annotate(f"{vi:.3f}", (x, vi), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=7.2, color=COLOR["lime"])
        pi_offset = (0, -12) if i < len(xs) - 1 else (-14, 7)
        ax.annotate(f"{pi:.3f}", (x, pi), xytext=pi_offset, textcoords="offset points",
                    ha="center", fontsize=7.2, color=COLOR["rise"])
    ax.annotate("VI 阈值 0.90", (0.02, 0.90), xycoords=("axes fraction", "data"),
                xytext=(0, 3), textcoords="offset points", fontsize=6.8, color=COLOR["lime"])
    ax.annotate("PI 阈值 0.80", (0.02, 0.80), xycoords=("axes fraction", "data"),
                xytext=(0, -9), textcoords="offset points", fontsize=6.8, color=COLOR["rise"])
    ax.annotate("选中配置", (xs[-1], pi_vals[-1]), xytext=(-18, -26), textcoords="offset points",
                ha="center", fontsize=7.5, fontweight="bold", color=INK,
                arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.9})
    ax.set_xticks(xs, xlabels, fontsize=7.5)
    ax.set_ylim(0.35, 1.05)
    ax.set_ylabel("跨种子 Spearman 中位数 ↑")
    ax.set_xlabel("预算配置（前向次数/次）")
    ax.set_title("(a) 预算预实验收敛性", fontsize=9)
    ax.legend(loc="lower right", fontsize=7.5)

    # (b)-(c) Patch movement test for ResNet-50 and VGG13.
    rule_order = ("ordshap_pi", "ordshap_vi", "kernel_shap", "random")
    rule_labels = {
        "ordshap_pi": "OrdShap-PI",
        "ordshap_vi": "OrdShap-VI",
        "kernel_shap": "KernelSHAP",
        "random": "Random",
    }
    rule_colors = {
        "ordshap_pi": COLOR["rise"],
        "ordshap_vi": COLOR["lime"],
        "kernel_shap": COLOR["kernel_shap"],
        "random": MUTED,
    }
    rule_markers = {"ordshap_pi": "s", "ordshap_vi": "o", "kernel_shap": "^", "random": "D"}
    model_titles = {"resnet50": "(b) 位置移动检验 · ResNet-50", "vgg13": "(c) 位置移动检验 · VGG13"}

    for idx, model in enumerate(("resnet50", "vgg13"), start=1):
        ax = axes[idx]
        entry = moves_by_model[model]
        for row_i, rule in enumerate(rule_order):
            stats = entry["rules"][rule]["mean_abs_move_change"]
            med = stats["median"]
            low, high = stats["median_ci95"]
            ax.errorbar([med], [row_i], xerr=[[med - low], [high - med]],
                        color=rule_colors[rule], marker=rule_markers[rule], markersize=6,
                        markeredgecolor="#fcfcfb", linewidth=1.6, capsize=2.5)
            ax.annotate(f"{med:.2f}", (high, row_i), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=7.5, color=INK)
        p_rand = next(t["p_holm"] for t in entry["tests_pi_minus_other"] if t["left"] == "random")
        p_ks = next(t["p_holm"] for t in entry["tests_pi_minus_other"] if t["left"] == "kernel_shap")
        ax.text(0.98, 0.03, f"PI vs Random: p={p_rand:.1e}*\nPI vs KS: p={p_ks:.2g}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=MUTED,
                bbox={"facecolor": "#fcfcfb", "edgecolor": GRID, "boxstyle": "round,pad=0.25"})
        ax.set_yticks(range(len(rule_order)), [rule_labels[r] for r in rule_order] if idx == 1 else [])
        ax.set_ylim(-0.55, len(rule_order) + 0.35)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        xlim_right = max(entry["rules"][r]["mean_abs_move_change"]["median_ci95"][1] for r in rule_order)
        ax.set_xlim(0, xlim_right * 1.26)
        ax.set_xlabel("移动 |Δlogit| 中位数与 95% CI ↑")
        ax.set_title(model_titles[model], fontsize=9)

    fig.suptitle("选做实验 5：OrdShap 位置归因（ImageNet 4×4 图块，预算预实验与位置移动检验）",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "m4_ordshap.png")


def baseline_figures() -> None:
    path = ROOT / "results/metrics/m4_baseline_sensitivity/summary.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    lookup = {(r["domain"], r["model"], r["method"], r["explanation_baseline"]): r for r in rows}

    baselines = (
        ("normalized_zero", "零基线"),
        ("dataset_mean_color", "均值色"),
        ("dataset_mean_image", "均值图"),
        ("gaussian_blur_sigma10", "模糊"),
        ("gaussian_noise_dataset_stats", "噪声"),
    )
    methods = ("lime", "kernel_shap", "ablation", "rise")
    metrics = (
        ("normalized_insertion_auc", "归一化 Insertion AUC ↑"),
        ("normalized_deletion_auc", "归一化 Deletion AUC ↓"),
    )

    fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.4), sharex=True)
    xs = np.arange(len(baselines))
    for col, key in enumerate(PANEL):
        for row_i, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_i, col]
            for method in methods:
                meds = [lookup[(*key, method, b)][metric]["median"] for b, _ in baselines]
                q25 = [lookup[(*key, method, b)][metric]["q25"] for b, _ in baselines]
                q75 = [lookup[(*key, method, b)][metric]["q75"] for b, _ in baselines]
                ax.fill_between(xs, q25, q75, color=COLOR[method], alpha=0.10, linewidth=0)
                ax.plot(xs, meds, color=COLOR[method], marker=MARKER[method],
                        markeredgecolor="#fcfcfb", markeredgewidth=1.1, markersize=5.5)
                if col == 3:
                    end_label(ax, xs[-1], meds[-1], NAME[method], COLOR[method])
            ax.set_xticks(xs, [lbl for _, lbl in baselines], fontsize=7.5)
            ax.grid(axis="x", visible=False)
            if row_i == 0:
                ax.set_title(PANEL[key], fontsize=9)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8.5)
            if row_i == 1:
                ax.set_xlabel("解释遮蔽基线", fontsize=8)

    fig.suptitle("M4 遮蔽基线敏感性：ImageNet 排序稳健，CHNCXR 遇均值图/模糊基线出现排序翻转（中位数与 IQR）",
                 fontsize=9.8, x=0.01, ha="left")
    legend_row(fig, methods)
    fig.tight_layout()
    place_labels(fig, gap=0.09)
    save(fig, "m4_baseline_sensitivity.png")


def localization_figures() -> None:
    path = ROOT / "results/metrics/m4_config_analysis/localization_diagnostics.json"
    diag = json.loads(path.read_text(encoding="utf-8"))
    cells = {(c["domain"], c["model"], c["variant"]): c for c in diag["cells"]}

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.8), gridspec_kw={"width_ratios": [1.0, 1.12, 1.25]})

    # (a) ImageNet localization energy vs chance area fraction.
    ax = axes[0]
    variants_im = [
        ("lime_budget_1024", "LIME"),
        ("kernel_shap_budget_1024", "KernelSHAP"),
        ("ablation_default", "Ablation"),
        ("rise_budget_1024", "RISE"),
    ]
    x = np.arange(len(variants_im))
    w = 0.36
    im_r50 = [cells[("imagenet", "resnet50", v)]["energy_median"] for v, _ in variants_im]
    im_vgg = [cells[("imagenet", "vgg13", v)]["energy_median"] for v, _ in variants_im]
    chance_im = diag["chance"]["imagenet"]["median_area_fraction"]
    ax.bar(x - w / 2, im_r50, w, color=COLOR["lime"], label="ResNet-50")
    ax.bar(x + w / 2, im_vgg, w, color=COLOR["kernel_shap"], alpha=0.85, label="VGG13")
    ax.axhline(chance_im, color=COLOR["rise"], linestyle="--", linewidth=1.2,
               label=f"目标面积基线 ({chance_im:.3f})")
    for xi, val in zip(x - w / 2, im_r50):
        ax.text(xi, val + 0.014, f"{val:.2f}", ha="center", va="bottom", fontsize=6.8, color=INK)
    for xi, val in zip(x + w / 2, im_vgg):
        ax.text(xi, val + 0.014, f"{val:.2f}", ha="center", va="bottom", fontsize=6.8, color=INK)
    ax.set_xticks(x, [lbl for _, lbl in variants_im], fontsize=7.5)
    ax.set_ylim(0, 0.90)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("掩码内正能量占比中位数 ↑")
    ax.set_title("(a) ImageNet 定位能量（n=75）", fontsize=9)
    ax.legend(loc="upper right", fontsize=7.0)

    # (b) CHNCXR lesion localization energy vs tiny lesion chance baseline.
    ax = axes[1]
    variants_cxr = [
        ("lime_budget_1024", "LIME\n(100)"),
        ("lime_slic_300", "LIME\n(300)"),
        ("kernel_shap_budget_1024", "Kernel\nSHAP"),
        ("ablation_default", "Ablation"),
        ("rise_budget_1024", "RISE\n(默认)"),
    ]
    x = np.arange(len(variants_cxr))
    cxr_r50 = [cells[("chncxr", "resnet50", v)]["energy_median"] for v, _ in variants_cxr]
    cxr_vgg = [cells[("chncxr", "vgg13", v)]["energy_median"] for v, _ in variants_cxr]
    chance_cxr = diag["chance"]["chncxr"]["median_area_fraction"]
    ax.bar(x - w / 2, cxr_r50, w, color=COLOR["lime"], label="ResNet-50")
    ax.bar(x + w / 2, cxr_vgg, w, color=COLOR["kernel_shap"], alpha=0.85, label="VGG13")
    ax.axhline(chance_cxr, color=COLOR["rise"], linestyle="--", linewidth=1.2)
    ax.text(0.02, chance_cxr + 0.0007, f"病灶面积中位基线 {chance_cxr:.4f}",
            transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData),
            ha="left", va="bottom", fontsize=7.2, color=COLOR["rise"])
    for xi, val in zip(x - w / 2, cxr_r50):
        ax.text(xi, val + 0.0005, f"{val:.3f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    for xi, val in zip(x + w / 2, cxr_vgg):
        ax.text(xi, val + 0.0005, f"{val:.3f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(x, [lbl for _, lbl in variants_cxr], fontsize=7.2)
    ax.set_ylim(0, 0.0225)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("病灶掩码内正能量占比中位数 ↑")
    ax.set_title("(b) CHNCXR 病灶定位能量（n=51）", fontsize=9)

    # (c) RISE on CHNCXR: raw positive-clipped vs centred energy recovery.
    ax = axes[2]
    rise_cases = [
        ("resnet50", "rise_budget_1024", "R50\n默认"),
        ("resnet50", "rise_grid_4", "R50\ngrid4"),
        ("resnet50", "rise_grid_14", "R50\ngrid14"),
        ("resnet50", "rise_mask_0.25", "R50\nm0.25"),
        ("resnet50", "rise_mask_0.75", "R50\nm0.75"),
        ("vgg13", "rise_budget_1024", "VGG13\n默认"),
    ]
    x = np.arange(len(rise_cases))
    raw_e = [cells[("chncxr", m, v)]["energy_median"] for m, v, _ in rise_cases]
    ctr_e = [cells[("chncxr", m, v)]["centred_energy_median"] for m, v, _ in rise_cases]
    ax.bar(x - w / 2, raw_e, w, color=COLOR["rise"], label="原始正截断能量")
    ax.bar(x + w / 2, ctr_e, w, color=COLOR["ablation"], label="去中心化后能量")
    ax.axhline(chance_cxr, color=MUTED, linestyle="--", linewidth=1.0)
    for xi, val in zip(x - w / 2, raw_e):
        ax.text(xi, val + 0.0005, f"{val:.3f}" if val > 0 else "0",
                ha="center", va="bottom", fontsize=6.5, color=COLOR["rise"] if val == 0 else INK)
    for xi, val in zip(x + w / 2, ctr_e):
        ax.text(xi, val + 0.0005, f"{val:.3f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    r50_logit = cells[("chncxr", "resnet50", "rise_budget_1024")]["mean_target_logit_median"]
    vgg_logit = cells[("chncxr", "vgg13", "rise_budget_1024")]["mean_target_logit_median"]
    ax.text(0.03, 0.97,
            f"R50 遮蔽 logit 中位={r50_logit:.2f}（全负→归零）\nVGG13 遮蔽 logit 中位=+{vgg_logit:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.6, color=MUTED,
            bbox={"facecolor": "#fcfcfb", "edgecolor": GRID, "boxstyle": "round,pad=0.22"})
    ax.set_xticks(x, [lbl for _, _, lbl in rise_cases], fontsize=7.0)
    ax.set_ylim(0, 0.0265)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("病灶掩码内能量占比")
    ax.set_title("(c) CHNCXR · RISE 负 logit 归零机制诊断", fontsize=9)
    ax.legend(loc="upper right", fontsize=6.8)

    fig.suptitle("CHNCXR 病灶定位诊断：极小病灶占比（1.57% vs 34.9%）与 RISE 负 logit 截断效应",
                 fontsize=9.8, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "m4_localization_diagnostics.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "which",
        nargs="?",
        default="all",
        choices=("config", "final", "ordshap", "baseline", "localization", "all"),
    )
    args = parser.parse_args()
    setup()
    if args.which in ("config", "all"):
        config_figures()
    if args.which in ("final", "all"):
        if (ROOT / "results/metrics/m4_final_analysis/summary.json").is_file():
            final_figures()
        else:
            print("skip final figures: run scripts/analyze_m4_final.py first")
    if args.which in ("ordshap", "all"):
        if (ROOT / "results/metrics/m4_ordshap/summary.json").is_file():
            ordshap_figures()
        else:
            print("skip ordshap figures: run scripts/analyze_m4_ordshap.py first")
    if args.which in ("baseline", "all"):
        if (ROOT / "results/metrics/m4_baseline_sensitivity/summary.json").is_file():
            baseline_figures()
        else:
            print("skip baseline figures: run scripts/run_m4_baseline_sensitivity.py first")
    if args.which in ("localization", "all"):
        if (ROOT / "results/metrics/m4_config_analysis/localization_diagnostics.json").is_file():
            localization_figures()
        else:
            print("skip localization figures: run scripts/analyze_m4_localization.py first")


if __name__ == "__main__":
    main()
