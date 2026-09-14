import numpy as np
from PIL import Image
import pytest

from preprocessing.localization import load_localization_mask


def test_load_localization_mask_resizes_nearest(tmp_path):
    mask = np.array([[0, 255], [0, 0]], dtype=np.uint8)
    Image.fromarray(mask).save(tmp_path / "sample.png")
    result = load_localization_mask("imagenet", "sample", tmp_path, (4, 4))
    assert result.dtype == bool
    assert result.shape == (4, 4)
    assert result.sum() == 4


def test_missing_mask_contract(tmp_path):
    assert load_localization_mask("chncxr", "negative", tmp_path) is None
    with pytest.raises(ValueError, match="Missing ImageNet"):
        load_localization_mask("imagenet", "missing", tmp_path)
