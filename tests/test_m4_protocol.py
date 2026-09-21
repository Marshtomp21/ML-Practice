"""M4 split and one-factor matrix contracts."""

import json
from pathlib import Path

from experiments.m4_matrix import all_tasks, load_config, variants
from scripts.analyze_m4_exp3 import analyze


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_samples_are_disjoint_and_m3_seen_are_not_final():
    frozen = json.loads((ROOT / "data/splits/m4_frozen_20260917.json").read_text())
    assert frozen["split_fingerprint"]
    for domain, expected in (("imagenet", (75, 299)), ("chncxr", (86, 339))):
        item = frozen["domains"][domain]
        tuning = {row["id"] for row in item["configuration_selection"]}
        final = {row["id"] for row in item["final_evaluation"]}
        excluded = {row["id"] for row in item.get("exploratory_only", [])}
        assert (len(tuning), len(final)) == expected
        assert not (tuning & final or tuning & excluded or final & excluded)
        assert not (set(item["m3_seen_ids"]) & final)


def test_m4_one_factor_cells_and_task_counts():
    config, selection = load_config(
        ROOT / "configs/experiment/m4.yaml", "imagenet", "exp1", "configuration_selection"
    )
    exp1 = variants("exp1")
    exp2 = variants("exp2")
    assert len(selection["selected"]) == 75
    assert len(exp1) == 13 and len(all_tasks(config, "exp1")) == 61
    assert len(exp2) == 10 and len(all_tasks(config, "exp2")) == 50
    assert len({cell["id"] for cell in exp1 + exp2}) == 23
    assert {cell["num_samples"] for cell in exp1 if "num_samples" in cell} == {
        256, 512, 1024, 2048
    }
    assert {cell["n_segments"] for cell in exp2 if "n_segments" in cell} == {
        50, 200, 300
    }
    assert {cell["grid_size"] for cell in exp2 if "grid_size" in cell} == {4, 10, 14}


def test_exp3_gains_use_image_level_seed_means():
    rows = [{"domain": "imagenet", "model": "resnet50", "id": "one",
             "variant": "ablation_default", "method": "ablation", "seed": None,
             "insertion_auc": 0.0, "deletion_auc": 5.0,
             "attribution_seconds": 0.1, "forward_samples": 81}]
    options = {"lime": (1.0, 4.0, 1.0), "rise": (2.0, 4.5, 2.0),
               "kernel_shap": (1.5, 2.0, 1.5)}
    for budget in (256, 512, 1024, 2048):
        for method, (insertion, deletion, seconds) in options.items():
            for seed in range(5):
                rows.append({"domain": "imagenet", "model": "resnet50", "id": "one",
                             "variant": f"{method}_budget_{budget}", "method": method,
                             "seed": seed, "insertion_auc": insertion,
                             "deletion_auc": deletion, "attribution_seconds": seconds,
                             "forward_samples": budget})
    result = analyze(rows, (0, 1, 2, 3, 4))
    assert len(result) == 4
    for row in result:
        assert row["rise_insertion_gain"] == 1.0
        assert row["rise_deletion_gain"] == 0.0
        assert row["shap_insertion_gain"] == 0.0
        assert row["shap_deletion_gain"] == 2.0
        assert row["stages"][2]["total_forward_samples_all_seeds"] == (
            81 + 15 * row["budget"]
        )
