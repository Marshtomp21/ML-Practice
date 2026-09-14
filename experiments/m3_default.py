"""Run the M3 default four-method comparison on a fixed real-data subset."""

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
from attribution.perturbation.rise import (
    RiseConfig,
    explain_rise,
    region_scores_from_saliency,
)
from evaluation.perturbation import (
    input_stability,
    insertion_deletion,
    localization_metrics,
    region_attribution_map,
    seed_stability,
)
from experiments.perturbation_pilot import (
    ROOT,
    domain_api,
    freeze_run,
    read_rows,
    screen,
    validate_config,
    write_json,
)
from models.inference import configure_runtime
from preprocessing.localization import load_localization_mask
from preprocessing.superpixels import slic_segments

RANDOM_METHODS = ("lime", "rise", "kernel_shap")
ALL_METHODS = (*RANDOM_METHODS, "ablation")


def validate_m3_config(config):
    validate_config(config)
    if config.get("stage") != "m3_default":
        raise ValueError("M3 config stage must be 'm3_default'")
    for key in ("stability_repeats",):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    epsilon = config.get("stability_epsilon")
    if not isinstance(epsilon, (int, float)) or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("stability_epsilon must be finite and positive")
    if not isinstance(config.get("localization_mask_root"), str):
        raise ValueError("localization_mask_root must be configured")
    for seed in config["seeds"]:
        KernelShapConfig(
            num_samples=config["num_samples"],
            batch_size=config["batch_size"],
            ridge_alpha=config["kernel_shap_ridge_alpha"],
            paired_sampling=config["kernel_shap_paired_sampling"],
            seed=seed,
        )


def method_config(config, method, seed):
    if method == "lime":
        return LimeConfig(
            config["num_samples"], config["batch_size"], config["mask_rate"],
            config["kernel_width"], config["ridge_alpha"], seed,
        )
    if method == "rise":
        return RiseConfig(
            config["num_samples"], config["batch_size"], config["grid_size"],
            config["mask_rate"], seed,
        )
    if method == "kernel_shap":
        return KernelShapConfig(
            num_samples=config["num_samples"],
            batch_size=config["batch_size"],
            ridge_alpha=config["kernel_shap_ridge_alpha"],
            paired_sampling=config["kernel_shap_paired_sampling"],
            seed=seed,
        )
    if method == "ablation":
        return AblationConfig(config["batch_size"])
    raise ValueError(f"Unknown method: {method}")


def explain(method, image, segments, predictor, target, config):
    if method == "lime":
        result = fit_lime(image, segments, predictor, target, config)
        ids, scores = result.region_ids, result.coefficients
        pixel_map = region_attribution_map(segments, scores)
    elif method == "rise":
        result = explain_rise(image, predictor, target, config)
        ids, scores = region_scores_from_saliency(result.saliency, segments)
        pixel_map = result.saliency
    elif method == "kernel_shap":
        result = fit_kernel_shap(image, segments, predictor, target, config)
        ids, scores = result.region_ids, result.coefficients
        pixel_map = region_attribution_map(segments, scores)
    elif method == "ablation":
        result = fit_ablation(image, segments, predictor, target, config)
        ids, scores = result.region_ids, result.coefficients
        pixel_map = region_attribution_map(segments, scores)
    else:
        raise ValueError(f"Unknown method: {method}")
    return result, ids, scores, pixel_map


def make_stability_inputs(rgb, preprocessing, sample_id, selection_seed, repeats, epsilon):
    digest = hashlib.sha256(f"{selection_seed}:{sample_id}".encode()).hexdigest()
    noise_seed = int(digest[:16], 16) % (2**32)
    rng = np.random.default_rng(noise_seed)
    noise = rng.uniform(-epsilon, epsilon, size=(repeats, *rgb.shape)).astype(np.float32)
    perturbed_rgb = np.clip(rgb[None, ...] + noise, 0.0, 1.0)
    mean = np.asarray(preprocessing["mean"], dtype=np.float32)
    std = np.asarray(preprocessing["std"], dtype=np.float32)
    return (perturbed_rgb - mean) / std, noise_seed


def _describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
    }


def summarize_m3(output):
    rows = read_rows(output / "metrics.jsonl")
    metrics = (
        "insertion_auc", "deletion_auc", "max_sensitivity", "cosine_similarity",
        "localization_energy", "pointing_game", "attribution_seconds",
        "forward_samples", "stability_seconds", "stability_forward_samples",
    )
    summary = []
    for model, method in sorted({(row["model"], row["method"]) for row in rows}):
        selected = [row for row in rows if (row["model"], row["method"]) == (model, method)]
        ids = sorted({row["id"] for row in selected})
        record = {"model": model, "method": method, "images": len(ids), "runs": len(selected)}
        for metric in metrics:
            image_values = []
            for sample_id in ids:
                values = [row[metric] for row in selected
                          if row["id"] == sample_id and row.get(metric) is not None]
                if values:
                    image_values.append(float(np.mean(values)))
            record[metric] = _describe(image_values) if image_values else None
        summary.append(record)
    write_json(output / "summary.json", summary)
    return rows, summary


def run_m3(config, output, metadata, selection):
    domain, preprocessing, _, load_image, load_model = domain_api(config)
    rows_path = output / "metrics.jsonl"
    done = {
        (row["model"], row["id"], row["method"], row["seed"])
        for row in read_rows(rows_path)
    }
    array_dir = ROOT / "results/attributions" / output.name
    array_dir.mkdir(parents=True, exist_ok=True)

    for model_name, checkpoint in config["models"].items():
        predictor = load_model(model_name, ROOT / checkpoint, config["device"])
        warm = load_image(ROOT / selection["selected"][0]["path"])[1]
        for _ in range(config["warmup_batches"]):
            predictor(np.repeat(warm[None], config["batch_size"], axis=0))

        for sample in selection["selected"]:
            rgb, image = load_image(ROOT / sample["path"])
            segments = slic_segments(
                rgb, config["n_segments"], config["compactness"], config["sigma"]
            )
            stability_images, noise_seed = make_stability_inputs(
                rgb, preprocessing, sample["id"], config["selection_seed"],
                config["stability_repeats"], config["stability_epsilon"],
            )
            before_predictions = predictor.samples
            perturbed_predictions = predictor(stability_images).argmax(axis=1).tolist()
            prediction_forward_samples = predictor.samples - before_predictions
            target_mask = load_localization_mask(
                domain,
                sample["id"],
                ROOT / config["localization_mask_root"],
                tuple(preprocessing["size"]),
            )

            tasks = [(method, seed) for method in RANDOM_METHODS for seed in config["seeds"]]
            tasks.append(("ablation", None))
            order_seed = int(hashlib.sha256(sample["id"].encode()).hexdigest()[:8], 16)
            np.random.default_rng(order_seed).shuffle(tasks)

            for method, seed in tasks:
                key = (model_name, sample["id"], method, seed)
                seed_text = "none" if seed is None else str(seed)
                stem = f"{model_name}_{sample['id']}_{method}_{seed_text}"
                artifact = array_dir / f"{stem}.npz"
                if key in done:
                    if not artifact.is_file():
                        raise ValueError(f"Missing attribution artifact for {key}")
                    continue
                current_config = method_config(config, method, seed)

                before_samples, before_batches = predictor.samples, predictor.batches
                predictor.synchronize()
                started = time.perf_counter()
                result, ids, scores, pixel_map = explain(
                    method, image, segments, predictor, sample["target"], current_config
                )
                predictor.synchronize()
                attribution_seconds = time.perf_counter() - started
                actual_samples = predictor.samples - before_samples
                actual_batches = predictor.batches - before_batches
                if (actual_samples, actual_batches) != (
                    result.forward_samples, result.forward_batches
                ):
                    raise AssertionError("Attribution budget disagrees with model adapter")

                evaluation_before = predictor.samples
                evaluation_started = time.perf_counter()
                curves = insertion_deletion(
                    image, segments, scores, predictor, sample["target"],
                    config["batch_size"], config["evaluation_steps"],
                )
                predictor.synchronize()
                evaluation_seconds = time.perf_counter() - evaluation_started
                if predictor.samples - evaluation_before != curves["evaluation_forward_samples"]:
                    raise AssertionError("Evaluation budget disagrees with model adapter")

                stability_before = predictor.samples
                stability_batches_before = predictor.batches
                predictor.synchronize()
                stability_started = time.perf_counter()
                perturbed_scores = []
                expected_stability_samples = 0
                expected_stability_batches = 0
                for perturbed_image in stability_images:
                    perturbed_result, perturbed_ids, vector, _ = explain(
                        method, perturbed_image, segments, predictor, sample["target"], current_config
                    )
                    if not np.array_equal(ids, perturbed_ids):
                        raise AssertionError("Stability attribution regions changed")
                    perturbed_scores.append(vector)
                    expected_stability_samples += perturbed_result.forward_samples
                    expected_stability_batches += perturbed_result.forward_batches
                predictor.synchronize()
                stability_seconds = time.perf_counter() - stability_started
                stability_forward_samples = predictor.samples - stability_before
                stability_forward_batches = predictor.batches - stability_batches_before
                if (stability_forward_samples, stability_forward_batches) != (
                    expected_stability_samples, expected_stability_batches
                ):
                    raise AssertionError("Stability budget disagrees with model adapter")
                stability = input_stability(scores, np.stack(perturbed_scores))

                localization = {
                    "localization_energy": None,
                    "pointing_game": None,
                    "zero_positive_energy": None,
                }
                if target_mask is not None:
                    localization = localization_metrics(pixel_map, target_mask)

                arrays = {"segments": segments, "region_ids": ids, "region_scores": scores}
                if method == "rise":
                    arrays["saliency"] = pixel_map
                np.savez_compressed(artifact, **arrays)
                diagnostics = {
                    field: value for field, value in asdict(result).items()
                    if not isinstance(value, np.ndarray)
                }
                row = {
                    "domain": domain,
                    "model": model_name,
                    "id": sample["id"],
                    "target": sample["target"],
                    "method": method,
                    "seed": seed,
                    "config": asdict(current_config),
                    "actual_regions": len(ids),
                    "attribution_seconds": attribution_seconds,
                    "evaluation_seconds": evaluation_seconds,
                    "stability_seconds": stability_seconds,
                    "forward_samples": actual_samples,
                    "forward_batches": actual_batches,
                    "stability_forward_samples": stability_forward_samples,
                    "stability_forward_batches": stability_forward_batches,
                    "stability_prediction_forward_samples": prediction_forward_samples,
                    "noise_seed": noise_seed,
                    "prediction_changes": int(sum(
                        prediction != sample["target"] for prediction in perturbed_predictions
                    )),
                    "perturbed_predictions": perturbed_predictions,
                    "localization_eligible": target_mask is not None,
                    "diagnostics": diagnostics,
                    **curves,
                    **stability,
                    **localization,
                    "run_fingerprint": metadata["fingerprint"],
                    "artifact": str(artifact.relative_to(ROOT)),
                }
                with rows_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                done.add(key)
                print(
                    f"{len(done)} done: {domain} {model_name} {sample['id']} "
                    f"{method} seed={seed} base={attribution_seconds:.2f}s "
                    f"stability={stability_seconds:.2f}s",
                    flush=True,
                )
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows, summary = summarize_m3(output)
    stability_rows = []
    for model_name in config["models"]:
        for sample in selection["selected"]:
            for method in RANDOM_METHODS:
                for left_seed, right_seed in combinations(config["seeds"], 2):
                    vectors = []
                    for seed in (left_seed, right_seed):
                        path = array_dir / f"{model_name}_{sample['id']}_{method}_{seed}.npz"
                        with np.load(path) as data:
                            vectors.append(data["region_scores"])
                    stability_rows.append({
                        "model": model_name,
                        "id": sample["id"],
                        "method": method,
                        "seeds": [left_seed, right_seed],
                        **seed_stability(*vectors),
                    })
    write_json(output / "seed_stability.json", stability_rows)
    expected = len(config["models"]) * len(selection["selected"]) * (
        len(RANDOM_METHODS) * len(config["seeds"]) + 1
    )
    if len(rows) != expected:
        raise AssertionError(f"Expected {expected} rows, found {len(rows)}")
    write_json(output / "complete.json", {
        "rows": len(rows),
        "expected": expected,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("require shard-count >= 1 and 0 <= shard-index < shard-count")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_m3_config(config)
    output = args.output.resolve()
    configure_runtime(config["selection_seed"], config["threads"])
    metadata = freeze_run(config, output, args.resume)
    try:
        selection = screen(config, output, metadata)
        selection = {
            **selection,
            "selected": selection["selected"][args.shard_index::args.shard_count],
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
        }
        if not selection["selected"]:
            raise ValueError("Selected sample shard is empty")
        write_json(output / "selected_shard.json", selection)
        print("Selected: " + ", ".join(row["id"] for row in selection["selected"]), flush=True)
        run_m3(config, output, metadata, selection)
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        with (output / "failures.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "utc": datetime.now(timezone.utc).isoformat(),
                "type": type(exc).__name__,
                "message": str(exc),
            }) + "\n")
        raise


if __name__ == "__main__":
    main()
