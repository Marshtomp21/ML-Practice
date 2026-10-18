"""Diagnose localization on the M4 configuration-selection runs.

The protocol scores localization on the positive part of the attribution map.
This script adds three diagnostics without changing that protocol:

* chance level: a uniformly random pixel hits the target with probability
  equal to its area fraction, and a uniform map has energy equal to it;
* the share of runs whose map has no positive mass (metrics then default to 0);
* for RISE, whose saliency is a mask-weighted logit mean without a baseline
  term, Pointing on the raw map's peak and energy of the mean-centred map.

Image-level values average seeds first; the output feeds the M3 revision.
"""

from __future__ import annotations

from collections import defaultdict
import glob
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocessing.localization import load_localization_mask  # noqa: E402

OUTPUT = ROOT / "results/metrics/m4_config_analysis/localization_diagnostics.json"
MASK_ROOTS = {"imagenet": "data/raw/imagenet/subsetEBPG_masks", "chncxr": "data/raw/chncxr/mergeMask"}


def main() -> None:
    rows = []
    for path in sorted(glob.glob(str(ROOT / "results/metrics/m4_*_config_exp*_shard*/metrics.jsonl"))):
        with open(path, encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    masks, grouped = {}, defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["pointing_game"] is None:
            continue
        key = row["domain"], row["id"]
        if key not in masks:
            masks[key] = load_localization_mask(row["domain"], row["id"],
                                                ROOT / MASK_ROOTS[row["domain"]], (224, 224))
        record = {"pointing": row["pointing_game"], "energy": row["localization_energy"],
                  "zero_positive": bool(row["zero_positive_energy"])}
        if row["method"] == "rise":
            with np.load(ROOT / row["artifact"]) as array:
                saliency = array["saliency"]
            mask = masks[key]
            centred = np.maximum(saliency - saliency.mean(), 0.0)
            record.update({
                "raw_peak_pointing": int(mask.ravel()[int(np.argmax(saliency))]),
                "centred_energy": float(centred[mask].sum() / centred.sum()) if centred.sum() else 0.0,
                "mean_target_logit": row["diagnostics"]["mean_target_logit"],
            })
        grouped[row["domain"], row["model"], row["variant"]][row["id"]].append(record)
    cells = []
    for (domain, model, variant), images in sorted(grouped.items()):
        per_image = {i: {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
                     for i, rs in images.items()}
        values = list(per_image.values())

        def agg(key, how):
            if key not in values[0]:
                return None
            data = [v[key] for v in values]
            return float(np.mean(data) if how == "mean" else np.median(data))

        cells.append({
            "domain": domain, "model": model, "variant": variant, "images": len(values),
            "pointing_hit_rate": agg("pointing", "mean"), "energy_median": agg("energy", "median"),
            "zero_positive_share": agg("zero_positive", "mean"),
            "raw_peak_hit_rate": agg("raw_peak_pointing", "mean"),
            "centred_energy_median": agg("centred_energy", "median"),
            "mean_target_logit_median": agg("mean_target_logit", "median"),
        })
    chance = {}
    for domain in MASK_ROOTS:
        areas = [float(m.mean()) for (d, _), m in masks.items() if d == domain]
        chance[domain] = {"images": len(areas), "mean_area_fraction": float(np.mean(areas)),
                          "median_area_fraction": float(np.median(areas))}
    OUTPUT.write_text(json.dumps({"chance": chance, "cells": cells}, ensure_ascii=False, indent=2)
                      + "\n", encoding="utf-8")
    print(json.dumps(chance, ensure_ascii=False))


if __name__ == "__main__":
    main()
