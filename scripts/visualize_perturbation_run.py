"""Render comparable M2 attribution panels from a completed run."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries

ROOT = Path(__file__).resolve().parents[1]


def stored_path(value):
    return ROOT/Path(str(value).replace("\\", "/"))


def normalized_map(segments, region_ids, region_scores):
    lookup = {int(region): float(score)
              for region, score in zip(region_ids, region_scores)}
    heatmap = np.vectorize(lookup.__getitem__)(segments)
    scale = np.percentile(np.abs(heatmap), 99)
    return heatmap / scale if scale > 0 else heatmap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    metadata = json.loads((run_dir/"run.json").read_text(encoding="utf-8"))
    selection = json.loads((run_dir/"selection.json").read_text(encoding="utf-8"))
    array_dir = ROOT/"results/attributions"/run_dir.name
    figure_dir = ROOT/"results/figures"/run_dir.name
    figure_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for model in metadata["config"]["models"]:
        for sample in selection["selected"][:args.limit]:
            with Image.open(stored_path(sample["path"])) as source:
                rgb = np.asarray(source.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
            artifacts = {}
            for method, seed in (("ablation", None), ("lime", args.seed), ("rise", args.seed)):
                path = array_dir/f"{model}_{sample['id']}_{method}_{seed}.npz"
                with np.load(path, allow_pickle=False) as data:
                    artifacts[method] = {name: data[name].copy() for name in data.files}
            segments = artifacts["ablation"]["segments"]
            fig, axes = plt.subplots(1, 5, figsize=(14, 3), constrained_layout=True)
            axes[0].imshow(rgb)
            axes[0].set_title("original")
            axes[1].imshow(mark_boundaries(rgb/255.0, segments, color=(1, 1, 0)))
            axes[1].set_title(f"SLIC ({len(np.unique(segments))} regions)")
            for axis, method in zip(axes[2:], ("ablation", "lime", "rise")):
                data = artifacts[method]
                heatmap = normalized_map(data["segments"], data["region_ids"], data["region_scores"])
                axis.imshow(rgb, alpha=0.45)
                axis.imshow(heatmap, cmap="coolwarm", vmin=-1, vmax=1, alpha=0.70)
                axis.set_title(method if method == "ablation" else f"{method} (seed={args.seed})")
            for axis in axes:
                axis.axis("off")
            fig.suptitle(f"{metadata.get('domain', 'chncxr')} · {model} · {sample['id']} · target {sample['target']}")
            destination = figure_dir/f"{model}_{sample['id']}.png"
            fig.savefig(destination, dpi=180)
            plt.close(fig)
            written.append(str(destination.relative_to(ROOT)))
    manifest = {"run": run_dir.name, "seed": args.seed, "figures": written}
    (figure_dir/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
