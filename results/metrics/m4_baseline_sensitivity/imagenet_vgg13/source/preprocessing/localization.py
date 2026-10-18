"""Load localization targets aligned with the shared 224x224 input."""

from pathlib import Path

import numpy as np
from PIL import Image


def load_localization_mask(domain, sample_id, mask_root, size=(224, 224)):
    """Return a boolean HW mask, or None for CHNCXR rows without a mask."""
    domain = str(domain).lower()
    if domain not in {"imagenet", "chncxr"}:
        raise ValueError("domain must be 'imagenet' or 'chncxr'")
    if (not isinstance(size, (tuple, list)) or len(size) != 2
            or any(type(value) is not int or value < 1 for value in size)):
        raise ValueError("size must contain two positive integers")
    path = Path(mask_root) / f"{sample_id}.png"
    if not path.is_file():
        if domain == "chncxr":
            return None
        raise ValueError(f"Missing ImageNet localization mask: {path}")
    with Image.open(path) as image:
        mask = np.asarray(
            image.convert("L").resize(tuple(size), Image.Resampling.NEAREST)
        ) > 127
    if not mask.any():
        raise ValueError(f"Empty localization mask: {path}")
    return mask
