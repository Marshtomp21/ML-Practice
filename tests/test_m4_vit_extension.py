"""Contracts for the M4 ViT architecture and partition extension."""

from pathlib import Path

import numpy as np
import yaml

from attribution.perturbation.region_rise import RegionRiseConfig, explain_region_rise
from evaluation.perturbation import insertion_deletion
from experiments.m4_vit_extension import (
    adapt_m3_row,
    evaluation_fraction_mode,
    experiment_models,
    summarize_partition_contrasts,
    tasks,
    validate_config,
    variants,
)
from preprocessing.superpixels import regular_grid_segments


ROOT = Path(__file__).resolve().parents[1]


def test_vit_patch_partition_is_exact_and_row_major():
    segments = regular_grid_segments((224, 224), 16)
    assert segments.shape == (224, 224)
    assert len(np.unique(segments)) == 196
    assert segments[0, 0] == 0
    assert segments[15, 15] == 0
    assert segments[0, 16] == 1
    assert segments[16, 0] == 14
    assert segments[-1, -1] == 195


def test_region_rise_is_reproducible_and_counts_the_budget():
    image = np.ones((2, 2, 1), dtype=np.float32)
    segments = np.array([[0, 0], [1, 1]], dtype=np.int64)

    def predict(batch):
        return batch.sum(axis=(1, 2, 3), keepdims=False)[:, None]

    config = RegionRiseConfig(num_masks=32, batch_size=7, mask_rate=0.5, seed=4)
    first = explain_region_rise(image, segments, predict, 0, config)
    second = explain_region_rise(image, segments, predict, 0, config)
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    assert first.forward_samples == 32
    assert first.forward_batches == 5
    assert first.saliency.shape == segments.shape


def test_pixel_fraction_axis_uses_actual_region_areas():
    image = np.ones((1, 4, 1), dtype=np.float32)
    segments = np.array([[0, 1, 1, 1]], dtype=np.int64)

    def predict(batch):
        return batch.sum(axis=(1, 2, 3))[:, None]

    result = insertion_deletion(
        image, segments, np.array([2.0, 1.0]), predict, 0,
        batch_size=4, steps=2, fraction_mode="pixels",
    )
    assert result["fraction_mode"] == "pixels"
    assert result["fractions"] == [0.0, 0.25, 1.0]
    assert result["normalized_insertion"] == [0.0, 0.25, 1.0]
    assert result["normalized_deletion"] == [1.0, 0.75, 0.0]
    assert result["normalized_insertion_auc"] == 0.5
    assert result["normalized_deletion_auc"] == 0.5


def test_partition_contrasts_average_seeds_before_interaction(tmp_path):
    rows = []
    for model, model_offset in (("resnet50", 0.0), ("vit_b_16", 0.2)):
        for partition, partition_gain in (("slic_196", 0.0), ("patch_16", 0.1)):
            for seed in (0, 1):
                rows.append({
                    "model": model,
                    "id": "image",
                    "method": "lime",
                    "partition": partition,
                    "seed": seed,
                    "normalized_insertion_auc": (
                        0.5 + model_offset + partition_gain
                        + (0.1 if model == "vit_b_16" and partition == "patch_16" else 0.0)
                        + seed * 0.01
                    ),
                    "normalized_deletion_auc": 0.5,
                    "localization_energy": 0.5,
                    "pointing_game": 1.0,
                })
    output = tmp_path / "contrasts.json"
    summarize_partition_contrasts(rows, output)
    records = yaml.safe_load(output.read_text(encoding="utf-8"))
    insertion = next(
        row for row in records if row["metric"] == "normalized_insertion_auc"
    )
    assert np.isclose(insertion["resnet_patch_minus_slic"]["median"], 0.1)
    assert np.isclose(insertion["vit_patch_minus_slic"]["median"], 0.2)
    assert np.isclose(insertion["vit_specific_interaction"]["median"], 0.1)


def test_m3_curves_can_be_normalized_without_rerunning_cnn():
    row = {
        "model": "resnet50", "id": "image", "method": "lime", "seed": 0,
        "fractions": [0.0, 0.5, 1.0],
        "insertion": [2.0, 4.0, 6.0],
        "deletion": [6.0, 4.0, 2.0],
    }
    adapted = adapt_m3_row(row)
    assert adapted["partition"] == "slic_100"
    assert adapted["reused_from"] == "m3_default"
    assert adapted["normalized_insertion"] == [0.0, 0.5, 1.0]
    assert adapted["normalized_deletion"] == [1.0, 0.5, 0.0]
    assert adapted["normalized_insertion_auc"] == 0.5
    assert adapted["normalized_deletion_auc"] == 0.5


def test_extension_matrix_and_record_counts():
    config = yaml.safe_load(
        (ROOT / "configs/experiment/m4_vit_extension.yaml").read_text(encoding="utf-8")
    )
    validate_config(config)
    assert experiment_models(config, "architecture") == ["vit_b_16"]
    assert config["experiments"]["architecture"]["reuse_models"] == [
        "resnet50", "vgg13"
    ]
    assert evaluation_fraction_mode(config, "architecture") == "regions"
    assert evaluation_fraction_mode(config, "partition") == "pixels"
    architecture = variants(config, "architecture")
    partition = variants(config, "partition")
    assert len(architecture) == 4
    assert len(partition) == 8
    assert len(tasks(config, "architecture")) == 16
    assert len(tasks(config, "partition")) == 32
    assert {cell["partition"] for cell in partition} == {"slic_196", "patch_16"}
    assert {cell["method"] for cell in partition} == {
        "ablation", "lime", "region_rise", "kernel_shap"
    }
