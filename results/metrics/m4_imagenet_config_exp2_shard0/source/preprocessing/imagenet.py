"""ImageNet validation-subset protocol used by the M2 experiments.

Images are resized directly to 224x224, without cropping, so that the XML
bounding boxes remain aligned with the model input as required by M1.
"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


PROTOCOL = {
    "color": "RGB",
    "size": [224, 224],
    "resize": "PIL bilinear (direct resize, no crop)",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "source": "M1 direct-resize protocol; torchvision ImageNet normalization",
}


def candidates(image_root, annotation_root, class_index):
    mapping = json.loads(Path(class_index).read_text(encoding="utf-8"))
    wnid_to_index = {value[0]: int(index) for index, value in mapping.items()}
    rows = []
    for path in sorted(Path(image_root).glob("ILSVRC2012_val_*.JPEG")):
        annotation = Path(annotation_root) / f"{path.stem}.xml"
        if not annotation.is_file():
            raise ValueError(f"Missing ImageNet annotation: {annotation}")
        wnid = ET.parse(annotation).getroot().findtext("object/name")
        if wnid not in wnid_to_index:
            raise ValueError(f"Unknown ImageNet synset {wnid!r} in {annotation}")
        rows.append({"id": path.stem, "path": str(path),
                     "target": wnid_to_index[wnid], "wnid": wnid})
    if not rows:
        raise ValueError(f"No ImageNet validation images in {image_root}")
    return rows


def load_image(path):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR),
                         dtype=np.float32) / 255.0
    mean = np.asarray(PROTOCOL["mean"], dtype=np.float32)
    std = np.asarray(PROTOCOL["std"], dtype=np.float32)
    return rgb, (rgb - mean) / std
