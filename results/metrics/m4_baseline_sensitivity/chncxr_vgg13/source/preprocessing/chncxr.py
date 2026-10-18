"""Course CHNCXR input protocol, with no center crop or lesion crop.

Source: XAI.zip / XAI/prior/src.zip / src/src/datasets.py,
DatasetLoader._load_ShenzhenPredSample: RGB, bilinear 224x224, mean/std=0.5.
Labels follow CHNCXR filename suffix: 0 normal, 1 tuberculosis.
"""
from pathlib import Path
import re

import numpy as np
from PIL import Image

PROTOCOL = {"color": "RGB", "size": [224, 224], "resize": "PIL bilinear",
            "mean": [0.5]*3, "std": [0.5]*3,
            "source": "XAI.zip/XAI/prior/src.zip/src/src/datasets.py:_load_ShenzhenPredSample"}


def candidates(root):
    rows = []
    for path in sorted(Path(root).glob("*.png")):
        match = re.fullmatch(r"CHNCXR_\d+_([01])", path.stem)
        if match:
            rows.append({"id": path.stem, "path": str(path), "target": int(match[1])})
    if not rows:
        raise ValueError(f"No CHNCXR images in {root}")
    return rows


def load_image(path):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR),
                         dtype=np.float32) / 255.0
    return rgb, (rgb - np.float32(0.5)) / np.float32(0.5)
