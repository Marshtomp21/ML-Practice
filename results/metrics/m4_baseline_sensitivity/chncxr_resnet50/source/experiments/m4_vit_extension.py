"""Run the M4 ImageNet ViT and SLIC/ViT-patch extension experiments.

The runner reuses the fixed M3 ImageNet sample set and the completed ResNet-50
and VGG13 results. It screens only ViT-B/16, then either adds the missing ViT
cross-architecture cells or executes the new feature-partition matrix. Results
are append-only and resumable.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from attribution.game_theoretic.kernel_shap import KernelShapConfig, fit_kernel_shap
from attribution.perturbation.ablation import AblationConfig, fit_ablation
from attribution.perturbation.lime import LimeConfig, fit_lime
from attribution.perturbation.region_rise import RegionRiseConfig, explain_region_rise
from attribution.perturbation.rise import RiseConfig, explain_rise, region_scores_from_saliency
from evaluation.perturbation import (
    input_stability,
    insertion_deletion,
    localization_metrics,
    region_attribution_map,
    seed_stability,
)
from experiments.m3_default import make_stability_inputs
from experiments.perturbation_pilot import (
    ROOT,
    domain_api,
    freeze_run,
    read_rows,
    write_json,
)
from models.inference import configure_runtime
from preprocessing.localization import load_localization_mask
from preprocessing.superpixels import regular_grid_segments, slic_segments


RANDOM_METHODS = {"lime", "rise", "region_rise", "kernel_shap"}
M3_PROTOCOL_KEYS = (
    "num_samples", "batch_size", "mask_rate", "kernel_width", "ridge_alpha",
    "kernel_shap_ridge_alpha", "kernel_shap_paired_sampling", "grid_size",
    "compactness", "sigma", "evaluation_steps", "stability_epsilon",
    "stability_repeats", "selection_seed", "seeds",
)


def validate_config(config: dict) -> None:
    """Fail early on protocol changes that would invalidate the comparison."""
    if config.get("stage") != "m4_vit_extension" or config.get("domain") != "imagenet":
        raise ValueError("Expected stage=m4_vit_extension and domain=imagenet")
    if set(config.get("models", {})) != {"resnet50", "vit_b_16"}:
        raise ValueError("M4 execution requires resnet50 and vit_b_16 checkpoints")
    for key in ("batch_size", "num_samples", "evaluation_steps", "stability_repeats",
                "threads", "warmup_batches", "sample_count"):
        if isinstance(config.get(key), bool) or not isinstance(config.get(key), int) \
                or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config.get("selection") != "m3_reuse" or not config.get("m3_results_dir"):
        raise ValueError("The extension protocol must reuse the frozen M3 ImageNet run")
    if not config.get("seeds") or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("seeds must be nonempty and unique")
    architecture = config.get("experiments", {}).get("architecture", {})
    partition = config.get("experiments", {}).get("partition", {})
    if architecture.get("models") != ["vit_b_16"]:
        raise ValueError("architecture executes only the missing ViT-B/16 cells")
    if architecture.get("reuse_models") != ["resnet50", "vgg13"]:
        raise ValueError("architecture must reuse the two frozen M3 CNN models")
    if architecture.get("evaluation_fraction_mode") != "regions":
        raise ValueError("architecture must preserve the M3 region-fraction protocol")
    if partition.get("models") != ["resnet50", "vit_b_16"]:
        raise ValueError("partition models must be the CNN control and ViT-B/16")
    if set(partition.get("partitions", [])) != {"slic", "patch"}:
        raise ValueError("partition experiment requires slic and patch")
    if partition.get("patch_size") != 16:
        raise ValueError("ViT-B/16 alignment requires patch_size=16")
    if partition.get("n_segments") != 196:
        raise ValueError("The matched SLIC condition must request 196 regions")
    if partition.get("evaluation_fraction_mode") != "pixels":
        raise ValueError("SLIC/patch comparison requires pixel-fraction evaluation")
    for seed in config["seeds"]:
        LimeConfig(config["num_samples"], config["batch_size"], config["mask_rate"],
                   config["kernel_width"], config["ridge_alpha"], seed)
        RiseConfig(config["num_samples"], config["batch_size"], config["grid_size"],
                   config["mask_rate"], seed)
        RegionRiseConfig(config["num_samples"], config["batch_size"],
                         config["mask_rate"], seed)
        KernelShapConfig(config["num_samples"], config["batch_size"],
                         config["kernel_shap_ridge_alpha"],
                         config["kernel_shap_paired_sampling"], seed)


def experiment_models(config: dict, experiment: str) -> list[str]:
    if experiment not in ("architecture", "partition"):
        raise ValueError(f"Unknown experiment: {experiment}")
    return list(config["experiments"][experiment]["models"])


def evaluation_fraction_mode(config: dict, experiment: str) -> str:
    return config["experiments"][experiment]["evaluation_fraction_mode"]


def m3_results_dir(config: dict) -> Path:
    path = (ROOT / config["m3_results_dir"]).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("m3_results_dir must be inside the repository") from exc
    return path


def load_m3_reuse(config: dict) -> tuple[Path, dict, dict]:
    """Load and validate the frozen M3 run used by the architecture comparison."""
    source = m3_results_dir(config)
    required = ("run.json", "selection.json", "metrics.jsonl", "seed_stability.json",
                "complete.json")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise ValueError(f"M3 reuse source is incomplete; missing: {', '.join(missing)}")
    run = json.loads((source / "run.json").read_text(encoding="utf-8"))
    selection = json.loads((source / "selection.json").read_text(encoding="utf-8"))
    source_config = run.get("config", {})
    if source_config.get("stage") != "m3_default" or source_config.get("domain") != "imagenet":
        raise ValueError("m3_results_dir is not a frozen M3 ImageNet default run")
    mismatches = [
        key for key in M3_PROTOCOL_KEYS if source_config.get(key) != config.get(key)
    ]
    if source_config.get("n_segments") != config["experiments"]["architecture"]["n_segments"]:
        mismatches.append("n_segments")
    if mismatches:
        raise ValueError("M4 architecture protocol differs from M3: "
                         + ", ".join(sorted(set(mismatches))))
    complete = json.loads((source / "complete.json").read_text(encoding="utf-8"))
    if complete.get("rows") != complete.get("expected"):
        raise ValueError("The M3 source run did not pass its completeness check")
    if len(selection.get("selected", [])) < config["sample_count"]:
        raise ValueError("M3 source contains fewer selected samples than requested")
    return source, run, selection


def screen_m3_reuse(config: dict, output: Path, metadata: dict) -> dict:
    """Screen ViT only on the fixed M3 samples; never rerun CNN screening."""
    selection_path = output / "selection.json"
    if selection_path.is_file():
        return json.loads(selection_path.read_text(encoding="utf-8"))
    source, source_run, source_selection = load_m3_reuse(config)
    requested = source_selection["selected"][:config["sample_count"]]
    current_pool = {row["id"]: row for row in metadata["candidate_pool"]}
    for sample in requested:
        current = current_pool.get(sample["id"])
        if current is None or any(current.get(key) != sample.get(key)
                                  for key in ("path", "target", "sha256")):
            raise ValueError(f"M3 sample changed or is missing: {sample['id']}")

    screen_path = output / "screen_vit_b_16.json"
    if screen_path.is_file():
        report = json.loads(screen_path.read_text(encoding="utf-8"))
    else:
        _, _, _, load_image, load_model = domain_api(config)
        predictor = load_model(
            "vit_b_16", ROOT / config["models"]["vit_b_16"], config["device"]
        )
        rows = []
        for start in range(0, len(requested), config["batch_size"]):
            batch = requested[start:start + config["batch_size"]]
            logits = predictor(np.stack([load_image(ROOT / row["path"])[1] for row in batch]))
            rows.extend({**row, "prediction": int(logit.argmax())}
                        for row, logit in zip(batch, logits))
        report = {
            "rows": rows,
            "forward_samples": predictor.samples,
            "accuracy": float(np.mean([
                row["prediction"] == row["target"] for row in rows
            ])),
        }
        write_json(screen_path, report)
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    by_id = {row["id"]: row for row in report["rows"]}
    if set(by_id) != {row["id"] for row in requested}:
        raise ValueError("Existing ViT screening does not match the requested M3 subset")
    selected = [row for row in requested
                if by_id[row["id"]]["prediction"] == row["target"]]
    if not selected:
        raise ValueError("ViT-B/16 misclassified every requested M3 sample")
    excluded = [
        {"id": row["id"], "target": row["target"],
         "prediction": by_id[row["id"]]["prediction"]}
        for row in requested if row["id"] not in {item["id"] for item in selected}
    ]
    selection = {
        "selection_policy": "fixed_m3_subset_intersect_vit_correct",
        "m3_results_dir": str(source.relative_to(ROOT)),
        "m3_run_fingerprint": source_run["fingerprint"],
        "m3_requested_count": len(requested),
        "vit_correct_count": len(selected),
        "excluded_by_vit": excluded,
        "reused_models": config["experiments"]["architecture"]["reuse_models"],
        "screened_models": ["vit_b_16"],
        "selected": selected,
    }
    write_json(selection_path, selection)
    split_dir = ROOT / "data/splits" / output.name
    split_dir.mkdir(parents=True, exist_ok=True)
    write_json(split_dir / "imagenet_selection.json",
               {"run_fingerprint": metadata["fingerprint"], **selection})
    return selection


def variants(config: dict, experiment: str) -> list[dict]:
    """Return the fully named cells for one extension experiment."""
    section = config["experiments"][experiment]
    if experiment == "architecture":
        partition_specs = [{
            "partition": f"slic_{section['n_segments']}",
            "partition_kind": "slic",
            "n_segments": section["n_segments"],
        }]
        methods = ("ablation", "lime", "rise", "kernel_shap")
    elif experiment == "partition":
        partition_specs = [
            {"partition": f"slic_{section['n_segments']}",
             "partition_kind": "slic", "n_segments": section["n_segments"]},
            {"partition": f"patch_{section['patch_size']}",
             "partition_kind": "patch", "patch_size": section["patch_size"]},
        ]
        methods = ("ablation", "lime", "region_rise", "kernel_shap")
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    return [
        {**partition, "method": method, "id": f"{partition['partition']}__{method}"}
        for partition in partition_specs for method in methods
    ]


def tasks(config: dict, experiment: str) -> list[tuple[dict, int | None]]:
    return [
        (cell, seed)
        for cell in variants(config, experiment)
        for seed in (config["seeds"] if cell["method"] in RANDOM_METHODS else [None])
    ]


def method_config(config: dict, method: str, seed: int | None):
    if method == "ablation":
        return AblationConfig(config["batch_size"])
    if seed is None:
        raise ValueError(f"{method} requires an explicit seed")
    if method == "lime":
        return LimeConfig(config["num_samples"], config["batch_size"],
                          config["mask_rate"], config["kernel_width"],
                          config["ridge_alpha"], seed)
    if method == "rise":
        return RiseConfig(config["num_samples"], config["batch_size"],
                          config["grid_size"], config["mask_rate"], seed)
    if method == "region_rise":
        return RegionRiseConfig(config["num_samples"], config["batch_size"],
                                config["mask_rate"], seed)
    if method == "kernel_shap":
        return KernelShapConfig(config["num_samples"], config["batch_size"],
                                config["kernel_shap_ridge_alpha"],
                                config["kernel_shap_paired_sampling"], seed)
    raise ValueError(f"Unknown method: {method}")


def partition_image(rgb: np.ndarray, cell: dict, config: dict) -> np.ndarray:
    if cell["partition_kind"] == "slic":
        return slic_segments(rgb, cell["n_segments"], config["compactness"], config["sigma"])
    if cell["partition_kind"] == "patch":
        return regular_grid_segments(rgb.shape[:2], cell["patch_size"])
    raise ValueError(f"Unknown partition: {cell['partition_kind']}")


def explain(method: str, image: np.ndarray, segments: np.ndarray, predictor,
            target: int, options):
    if method == "ablation":
        result = fit_ablation(image, segments, predictor, target, options)
        ids, scores = result.region_ids, result.coefficients
        pixel_map = region_attribution_map(segments, scores)
    elif method == "lime":
        result = fit_lime(image, segments, predictor, target, options)
        ids, scores = result.region_ids, result.coefficients
        pixel_map = region_attribution_map(segments, scores)
    elif method == "rise":
        result = explain_rise(image, predictor, target, options)
        ids, scores = region_scores_from_saliency(result.saliency, segments)
        pixel_map = result.saliency
    elif method == "region_rise":
        result = explain_region_rise(image, segments, predictor, target, options)
        ids, scores, pixel_map = result.region_ids, result.coefficients, result.saliency
    elif method == "kernel_shap":
        result = fit_kernel_shap(image, segments, predictor, target, options)
        ids, scores = result.region_ids, result.coefficients
        pixel_map = region_attribution_map(segments, scores)
    else:
        raise ValueError(f"Unknown method: {method}")
    return result, ids, scores, pixel_map


def artifact_path(output: Path, experiment: str, model: str, sample_id: str,
                  variant: str, seed: int | None) -> Path:
    seed_text = "none" if seed is None else str(seed)
    filename = f"{experiment}_{model}_{sample_id}_{variant}_{seed_text}.npz"
    return ROOT / "results/attributions" / output.name / filename


def row_key(row: dict) -> tuple:
    return row["model"], row["id"], row["variant"], row["seed"]


def expected_keys(config: dict, selected: list[dict], experiment: str) -> set[tuple]:
    return {
        (model, sample["id"], cell["id"], seed)
        for model in experiment_models(config, experiment)
        for sample in selected
        for cell, seed in tasks(config, experiment)
    }


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()


def describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(array), "median": float(np.median(array)),
            "q25": float(np.percentile(array, 25)),
            "q75": float(np.percentile(array, 75))}


def summarize(rows: list[dict], path: Path, metric_names: tuple[str, ...]) -> None:
    summary = []
    groups = sorted({(row["model"], row["partition"], row["method"]) for row in rows})
    for model, partition, method in groups:
        selected = [row for row in rows if
                    (row["model"], row["partition"], row["method"])
                    == (model, partition, method)]
        ids = sorted({row["id"] for row in selected})
        record = {"model": model, "partition": partition, "method": method,
                  "images": len(ids), "runs": len(selected)}
        for metric in metric_names:
            image_values = []
            for sample_id in ids:
                values = [row.get(metric) for row in selected if row["id"] == sample_id
                          and row.get(metric) is not None]
                if values:
                    image_values.append(float(np.mean(values)))
            record[metric] = describe(image_values) if image_values else None
        summary.append(record)
    write_json(path, summary)


def _trapezoid(values: list[float], fractions: list[float]) -> float:
    y = np.asarray(values, dtype=np.float64)
    x = np.asarray(fractions, dtype=np.float64)
    return float(np.sum((y[:-1] + y[1:]) * np.diff(x) / 2.0))


def adapt_m3_row(row: dict) -> dict:
    """Add M4 comparison fields without changing the frozen M3 result files."""
    adapted = {
        **row,
        "experiment": "architecture",
        "partition": "slic_100",
        "partition_kind": "slic",
        "variant": f"slic_100__{row['method']}",
        "fraction_mode": "regions",
        "reused_from": "m3_default",
    }
    insertion = row.get("insertion")
    deletion = row.get("deletion")
    fractions = row.get("fractions")
    if insertion is None or deletion is None or fractions is None:
        return adapted
    baseline = float((insertion[0] + deletion[-1]) / 2.0)
    full = float((insertion[-1] + deletion[0]) / 2.0)
    delta = full - baseline
    tolerance = 32 * np.finfo(np.float64).eps * max(1.0, abs(baseline), abs(full))
    adapted.update({"baseline_logit": baseline, "full_logit": full,
                    "normalization_delta": delta})
    if abs(delta) <= tolerance:
        adapted.update({"normalized_insertion": None, "normalized_deletion": None,
                        "normalized_insertion_auc": None,
                        "normalized_deletion_auc": None})
        return adapted
    normalized_insertion = ((np.asarray(insertion) - baseline) / delta).tolist()
    normalized_deletion = ((np.asarray(deletion) - baseline) / delta).tolist()
    adapted.update({
        "normalized_insertion": normalized_insertion,
        "normalized_deletion": normalized_deletion,
        "normalized_insertion_auc": _trapezoid(normalized_insertion, fractions),
        "normalized_deletion_auc": _trapezoid(normalized_deletion, fractions),
    })
    return adapted


def reused_m3_rows(config: dict, selected: list[dict]) -> list[dict]:
    source, _, _ = load_m3_reuse(config)
    ids = {sample["id"] for sample in selected}
    models = set(config["experiments"]["architecture"]["reuse_models"])
    rows = [adapt_m3_row(row) for row in read_rows(source / "metrics.jsonl")
            if row["id"] in ids and row["model"] in models]
    expected = len(ids) * len(models) * len(tasks(config, "architecture"))
    if len(rows) != expected:
        raise ValueError(f"M3 reusable rows incomplete: {len(rows)}/{expected}")
    return rows


def reused_m3_seed_stability(config: dict, selected: list[dict]) -> list[dict]:
    source, _, _ = load_m3_reuse(config)
    ids = {sample["id"] for sample in selected}
    models = set(config["experiments"]["architecture"]["reuse_models"])
    rows = json.loads((source / "seed_stability.json").read_text(encoding="utf-8"))
    return [{**row, "experiment": "architecture", "partition": "slic_100",
             "variant": f"slic_100__{row['method']}", "reused_from": "m3_default"}
            for row in rows if row["id"] in ids and row["model"] in models]


def summarize_partition_contrasts(rows: list[dict], path: Path) -> None:
    """Write paired Patch-SLIC effects and the ViT-specific interaction.

    Seeds are averaged within each image before any contrast is calculated.
    Positive interaction means that Patch-SLIC is larger for ViT than for
    ResNet. Metric direction still applies: higher is better for insertion and
    localization, whereas lower is better for deletion.
    """
    models = ("resnet50", "vit_b_16")
    partitions = ("slic_196", "patch_16")
    metrics = (
        "normalized_insertion_auc",
        "normalized_deletion_auc",
        "localization_energy",
        "pointing_game",
    )
    records = []
    methods = sorted({row["method"] for row in rows})
    sample_ids = sorted({row["id"] for row in rows})

    def image_mean(model: str, sample_id: str, method: str,
                   partition: str, metric: str) -> float | None:
        values = [
            row.get(metric) for row in rows
            if (row["model"], row["id"], row["method"], row["partition"])
            == (model, sample_id, method, partition)
            and row.get(metric) is not None
        ]
        return float(np.mean(values)) if values else None

    for method in methods:
        for metric in metrics:
            paired = []
            for sample_id in sample_ids:
                values = {
                    (model, partition): image_mean(
                        model, sample_id, method, partition, metric
                    )
                    for model in models for partition in partitions
                }
                if any(value is None for value in values.values()):
                    continue
                resnet_delta = (
                    values[("resnet50", "patch_16")]
                    - values[("resnet50", "slic_196")]
                )
                vit_delta = (
                    values[("vit_b_16", "patch_16")]
                    - values[("vit_b_16", "slic_196")]
                )
                paired.append({
                    "id": sample_id,
                    "resnet_patch_minus_slic": resnet_delta,
                    "vit_patch_minus_slic": vit_delta,
                    "vit_specific_interaction": vit_delta - resnet_delta,
                })
            if not paired:
                continue
            records.append({
                "method": method,
                "metric": metric,
                "direction": "lower_is_better" if metric == "normalized_deletion_auc"
                             else "higher_is_better",
                "images": len(paired),
                "resnet_patch_minus_slic": describe([
                    row["resnet_patch_minus_slic"] for row in paired
                ]),
                "vit_patch_minus_slic": describe([
                    row["vit_patch_minus_slic"] for row in paired
                ]),
                "vit_specific_interaction": describe([
                    row["vit_specific_interaction"] for row in paired
                ]),
                "per_image": paired,
            })
    write_json(path, records)


def finish_main(config: dict, output: Path, selected: list[dict], experiment: str) -> None:
    rows = read_rows(output / "metrics.jsonl")
    expected = expected_keys(config, selected, experiment)
    actual = [row_key(row) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f"Main records incomplete or duplicated: {len(actual)}/{len(expected)}")
    stability = []
    for model in experiment_models(config, experiment):
        for sample in selected:
            for cell in variants(config, experiment):
                if cell["method"] not in RANDOM_METHODS:
                    continue
                for left, right in combinations(config["seeds"], 2):
                    vectors = []
                    for seed in (left, right):
                        with np.load(artifact_path(
                                output, experiment, model, sample["id"], cell["id"], seed
                        )) as array:
                            vectors.append(array["region_scores"])
                    stability.append({"experiment": experiment, "model": model,
                                      "id": sample["id"], "partition": cell["partition"],
                                      "method": cell["method"], "variant": cell["id"],
                                      "seeds": [left, right], **seed_stability(*vectors)})
    write_json(output / "seed_stability.json", stability)
    summarize(rows, output / "summary.json", (
        "insertion_auc", "deletion_auc", "normalized_insertion_auc",
        "normalized_deletion_auc", "localization_energy", "pointing_game",
        "actual_regions", "attribution_seconds", "forward_samples",
    ))
    if experiment == "partition":
        summarize_partition_contrasts(rows, output / "partition_contrasts.json")
    else:
        reused = reused_m3_rows(config, selected)
        summarize(reused + rows, output / "architecture_comparison_summary.json", (
            "insertion_auc", "deletion_auc", "normalized_insertion_auc",
            "normalized_deletion_auc", "localization_energy", "pointing_game",
            "actual_regions", "attribution_seconds", "forward_samples",
        ))
        write_json(output / "architecture_seed_stability.json",
                   reused_m3_seed_stability(config, selected) + stability)
        write_json(output / "m3_reuse_manifest.json", {
            "source": config["m3_results_dir"],
            "models": config["experiments"]["architecture"]["reuse_models"],
            "sample_ids": [sample["id"] for sample in selected],
            "rows_reused": len(reused),
            "note": "M3 files are read-only; CNN attribution cells were not rerun.",
        })
    write_json(output / "complete_main.json", {
        "experiment": experiment, "rows": len(rows), "expected": len(expected),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    })


def run_main(config: dict, output: Path, metadata: dict, selected: list[dict],
             experiment: str) -> None:
    domain, _, _, load_image, load_model = domain_api(config)
    rows_path = output / "metrics.jsonl"
    existing = read_rows(rows_path)
    done = {row_key(row) for row in existing}
    if len(done) != len(existing):
        raise ValueError("Duplicate main result keys")
    (ROOT / "results/attributions" / output.name).mkdir(parents=True, exist_ok=True)

    for model_name in experiment_models(config, experiment):
        predictor = load_model(model_name, ROOT / config["models"][model_name], config["device"])
        warm = load_image(ROOT / selected[0]["path"])[1]
        for _ in range(config["warmup_batches"]):
            predictor(np.repeat(warm[None], config["batch_size"], axis=0))
        for sample in selected:
            scheduled = tasks(config, experiment)
            digest = hashlib.sha256(f"{experiment}:{model_name}:{sample['id']}".encode()).hexdigest()
            np.random.default_rng(int(digest[:8], 16)).shuffle(scheduled)
            if all((model_name, sample["id"], cell["id"], seed) in done
                   for cell, seed in scheduled):
                continue
            rgb, image = load_image(ROOT / sample["path"])
            target_mask = load_localization_mask(
                domain, sample["id"], ROOT / config["localization_mask_root"], rgb.shape[:2]
            )
            segment_cache = {}
            for cell, seed in scheduled:
                key = model_name, sample["id"], cell["id"], seed
                artifact = artifact_path(output, experiment, *key)
                if key in done:
                    if not artifact.is_file():
                        raise ValueError(f"Missing attribution artifact for {key}")
                    continue
                if cell["partition"] not in segment_cache:
                    segment_cache[cell["partition"]] = partition_image(rgb, cell, config)
                segments = segment_cache[cell["partition"]]
                options = method_config(config, cell["method"], seed)
                before_samples, before_batches = predictor.samples, predictor.batches
                predictor.synchronize()
                started = time.perf_counter()
                result, ids, scores, pixel_map = explain(
                    cell["method"], image, segments, predictor, sample["target"], options
                )
                predictor.synchronize()
                elapsed = time.perf_counter() - started
                forward_samples = predictor.samples - before_samples
                forward_batches = predictor.batches - before_batches
                if (forward_samples, forward_batches) != (
                        result.forward_samples, result.forward_batches):
                    raise AssertionError("Attribution forward count mismatch")

                evaluation_before = predictor.samples
                curves = insertion_deletion(
                    image, segments, scores, predictor, sample["target"],
                    config["batch_size"], config["evaluation_steps"],
                    evaluation_fraction_mode(config, experiment),
                )
                if predictor.samples - evaluation_before != curves["evaluation_forward_samples"]:
                    raise AssertionError("Evaluation forward count mismatch")
                localization = {"localization_energy": None, "pointing_game": None,
                                "zero_positive_energy": None}
                if target_mask is not None:
                    localization = localization_metrics(pixel_map, target_mask)
                np.savez_compressed(artifact, segments=segments, region_ids=ids,
                                    region_scores=scores, saliency=pixel_map)
                diagnostics = {name: value for name, value in asdict(result).items()
                               if not isinstance(value, np.ndarray)}
                row = {
                    "domain": domain, "experiment": experiment, "model": model_name,
                    "id": sample["id"], "target": sample["target"],
                    "method": cell["method"], "method_family": (
                        "rise" if cell["method"] == "region_rise" else cell["method"]
                    ),
                    "variant": cell["id"], "partition": cell["partition"],
                    "partition_kind": cell["partition_kind"], "seed": seed,
                    "config": asdict(options), "actual_regions": len(ids),
                    "attribution_seconds": elapsed, "forward_samples": forward_samples,
                    "forward_batches": forward_batches, "diagnostics": diagnostics,
                    "localization_eligible": target_mask is not None,
                    **curves, **localization, "run_fingerprint": metadata["fingerprint"],
                    "artifact": str(artifact.relative_to(ROOT)),
                }
                append_jsonl(rows_path, row)
                done.add(key)
                print(f"{len(done)} done {experiment} {model_name} {sample['id']} "
                      f"{cell['id']} seed={seed} {elapsed:.2f}s", flush=True)
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finish_main(config, output, selected, experiment)


def run_stability(config: dict, output: Path, metadata: dict, selected: list[dict],
                  experiment: str) -> None:
    if not (output / "complete_main.json").is_file():
        raise ValueError("Complete the main phase before input-stability evaluation")
    domain, preprocessing, _, load_image, load_model = domain_api(config)
    rows_path = output / "stability.jsonl"
    existing = read_rows(rows_path)
    done = {row_key(row) for row in existing}
    if len(done) != len(existing):
        raise ValueError("Duplicate stability result keys")
    for model_name in experiment_models(config, experiment):
        predictor = load_model(model_name, ROOT / config["models"][model_name], config["device"])
        for sample in selected:
            scheduled = tasks(config, experiment)
            if all((model_name, sample["id"], cell["id"], seed) in done
                   for cell, seed in scheduled):
                continue
            rgb, _ = load_image(ROOT / sample["path"])
            inputs, noise_seed = make_stability_inputs(
                rgb, preprocessing, sample["id"], config["selection_seed"],
                config["stability_repeats"], config["stability_epsilon"],
            )
            prediction_before = predictor.samples
            predictions = predictor(inputs).argmax(axis=1).tolist()
            prediction_forward_samples = predictor.samples - prediction_before
            for cell, seed in scheduled:
                key = model_name, sample["id"], cell["id"], seed
                if key in done:
                    continue
                with np.load(artifact_path(output, experiment, *key)) as array:
                    segments = array["segments"]
                    ids = array["region_ids"]
                    reference = array["region_scores"]
                options = method_config(config, cell["method"], seed)
                before_samples, before_batches = predictor.samples, predictor.batches
                predictor.synchronize()
                started = time.perf_counter()
                vectors = []
                expected_samples = expected_batches = 0
                for image in inputs:
                    result, new_ids, scores, _ = explain(
                        cell["method"], image, segments, predictor, sample["target"], options
                    )
                    if not np.array_equal(ids, new_ids):
                        raise AssertionError("Region IDs changed during stability evaluation")
                    vectors.append(scores)
                    expected_samples += result.forward_samples
                    expected_batches += result.forward_batches
                predictor.synchronize()
                actual_samples = predictor.samples - before_samples
                actual_batches = predictor.batches - before_batches
                if (actual_samples, actual_batches) != (expected_samples, expected_batches):
                    raise AssertionError("Stability forward count mismatch")
                row = {
                    "domain": domain, "experiment": experiment, "model": model_name,
                    "id": sample["id"], "target": sample["target"],
                    "method": cell["method"], "method_family": (
                        "rise" if cell["method"] == "region_rise" else cell["method"]
                    ),
                    "variant": cell["id"], "partition": cell["partition"],
                    "partition_kind": cell["partition_kind"], "seed": seed,
                    "stability_seconds": time.perf_counter() - started,
                    "stability_forward_samples": actual_samples,
                    "stability_forward_batches": actual_batches,
                    "stability_prediction_forward_samples": prediction_forward_samples,
                    "noise_seed": noise_seed,
                    "prediction_changes": sum(value != sample["target"] for value in predictions),
                    **input_stability(reference, np.stack(vectors)),
                    "run_fingerprint": metadata["fingerprint"],
                }
                append_jsonl(rows_path, row)
                done.add(key)
                print(f"{len(done)} stability {experiment} {model_name} {sample['id']} "
                      f"{cell['id']} seed={seed}", flush=True)
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rows = read_rows(rows_path)
    expected = expected_keys(config, selected, experiment)
    actual = [row_key(row) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f"Stability records incomplete or duplicated: {len(actual)}/{len(expected)}")
    summarize(rows, output / "stability_summary.json",
              ("max_sensitivity", "cosine_similarity", "stability_seconds",
               "stability_forward_samples", "prediction_changes"))
    if experiment == "architecture":
        reused = reused_m3_rows(config, selected)
        summarize(reused + rows, output / "architecture_stability_comparison_summary.json",
                  ("max_sensitivity", "cosine_similarity", "stability_seconds",
                   "stability_forward_samples", "prediction_changes"))
    write_json(output / "complete_stability.json", {
        "experiment": experiment, "rows": len(rows), "expected": len(expected),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/experiment/m4_vit_extension.yaml")
    parser.add_argument("--experiment", choices=("architecture", "partition"), required=True)
    parser.add_argument("--phase", choices=("screen", "main", "stability"), default="main")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.device:
        config["device"] = args.device
    if args.sample_count is not None:
        config["sample_count"] = args.sample_count
    validate_config(config)
    if args.dry_run:
        per_image = len(experiment_models(config, args.experiment)) * len(
            tasks(config, args.experiment)
        )
        print(json.dumps({"experiment": args.experiment, "phase": args.phase,
                          "sample_count": config["sample_count"],
                          "records_per_image": per_image,
                          "total_records": config["sample_count"] * per_image,
                          "models_screened": ["vit_b_16"],
                          "models_reused": (
                              config["experiments"]["architecture"]["reuse_models"]
                              if args.experiment == "architecture" else []
                          ),
                          "evaluation_fraction_mode": evaluation_fraction_mode(
                              config, args.experiment
                          ),
                          "cells": variants(config, args.experiment)}, indent=2))
        return

    output = args.output.resolve()
    configure_runtime(config["selection_seed"], config["threads"])
    metadata = freeze_run(config, output, args.resume)
    try:
        selection = screen_m3_reuse(config, output, metadata)
        selected = selection["selected"]
        print("Selected: " + ", ".join(row["id"] for row in selected), flush=True)
        if args.phase == "screen":
            return
        if args.phase == "main":
            run_main(config, output, metadata, selected, args.experiment)
        else:
            run_stability(config, output, metadata, selected, args.experiment)
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        append_jsonl(output / "failures.jsonl", {
            "utc": datetime.now(timezone.utc).isoformat(),
            "type": type(exc).__name__, "message": str(exc),
        })
        raise


if __name__ == "__main__":
    main()
