"""Render four-method M3 attribution panels from a merged run."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.perturbation import region_attribution_map

METHODS = ("ablation", "lime", "rise", "kernel_shap")


def normalized(values):
    scale = np.percentile(np.abs(values), 99)
    return values / scale if scale > 0 else values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    figure_dir = ROOT / "results/figures" / run_dir.name
    figure_dir.mkdir(parents=True, exist_ok=True)
    written = []
    samples = selection["selected"][:args.limit]
    if metadata["domain"] == "chncxr":
        negatives = [row for row in selection["selected"] if row["target"] == 0]
        positives = [row for row in selection["selected"] if row["target"] == 1]
        negative_count = args.limit // 2
        samples = negatives[:negative_count] + positives[:args.limit - negative_count]
    for model in metadata["config"]["models"]:
        for sample in samples:
            image_path = ROOT / Path(sample["path"].replace("\\", "/"))
            with Image.open(image_path) as source:
                rgb = np.asarray(
                    source.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                )
            artifacts = {}
            for method in METHODS:
                seed = None if method == "ablation" else args.seed
                row = next(item for item in rows if item["model"] == model
                           and item["id"] == sample["id"] and item["method"] == method
                           and item["seed"] == seed)
                with np.load(ROOT / row["artifact"], allow_pickle=False) as data:
                    artifacts[method] = {name: data[name].copy() for name in data.files}
            segments = artifacts["ablation"]["segments"]
            fig, axes = plt.subplots(1, 6, figsize=(16.5, 3), constrained_layout=True)
            axes[0].imshow(rgb)
            axes[0].set_title("original")
            axes[1].imshow(mark_boundaries(rgb / 255.0, segments, color=(1, 1, 0)))
            axes[1].set_title(f"SLIC ({len(np.unique(segments))})")
            for axis, method in zip(axes[2:], METHODS):
                data = artifacts[method]
                heatmap = (
                    data["saliency"] if method == "rise"
                    else region_attribution_map(data["segments"], data["region_scores"])
                )
                axis.imshow(rgb, alpha=0.45)
                axis.imshow(normalized(heatmap), cmap="coolwarm", vmin=-1, vmax=1, alpha=0.70)
                axis.set_title(method if method == "ablation" else f"{method} · seed {args.seed}")
            for axis in axes:
                axis.axis("off")
            fig.suptitle(
                f"{metadata['domain']} · {model} · {sample['id']} · target {sample['target']}"
            )
            destination = figure_dir / f"{model}_{sample['id']}.png"
            fig.savefig(destination, dpi=180)
            plt.close(fig)
            written.append(str(destination.relative_to(ROOT)))
    manifest = {"run": run_dir.name, "seed": args.seed, "figures": written}
    (figure_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
