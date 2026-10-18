"""Run frozen M4 experiment-1/2 cells with per-image resumable records."""

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

from evaluation.perturbation import (
    input_stability,
    insertion_deletion,
    localization_metrics,
    seed_stability,
)
from experiments.m3_default import explain, make_stability_inputs, method_config
from experiments.perturbation_pilot import (
    ROOT,
    domain_api,
    freeze_run,
    read_rows,
    write_json,
)
from models.inference import configure_runtime
from preprocessing.localization import load_localization_mask
from preprocessing.superpixels import slic_segments


def variants(experiment: str) -> list[dict]:
    """Return one-factor cells; E2 reuses E1's 1024-budget default cells."""
    cells = []
    if experiment == "exp1":
        for method in ("lime", "rise", "kernel_shap"):
            for budget in (256, 512, 1024, 2048):
                cells.append({"id": f"{method}_budget_{budget}", "method": method,
                              "num_samples": budget})
        cells.append({"id": "ablation_default", "method": "ablation"})
    elif experiment == "exp2":
        for count in (50, 200, 300):
            cells.append({"id": f"lime_slic_{count}", "method": "lime",
                          "n_segments": count})
        for size in (4, 10, 14):
            cells.append({"id": f"rise_grid_{size}", "method": "rise",
                          "grid_size": size})
        for method in ("lime", "rise"):
            for rate in (0.25, 0.75):
                cells.append({"id": f"{method}_mask_{rate:.2f}", "method": method,
                              "mask_rate": rate})
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    return cells


def all_tasks(config: dict, experiment: str) -> list[tuple[dict, int | None]]:
    return [(cell, seed) for cell in variants(experiment)
            for seed in (config["seeds"] if cell["method"] != "ablation" else [None])]


def load_config(path: Path, domain: str, experiment: str, split_name: str) -> tuple[dict, dict]:
    base = yaml.safe_load(path.read_text(encoding="utf-8"))
    if base.get("stage") != "m4" or domain not in base["domains"]:
        raise ValueError("Expected an M4 config with the requested domain")
    frozen_path = ROOT / base["split_file"]
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("stage") != "m4" or split_name not in (
        "configuration_selection", "final_evaluation"
    ):
        raise ValueError("Invalid frozen split")
    selected = frozen["domains"][domain][split_name]
    if not selected:
        raise ValueError("Frozen split is empty")
    cfg = {key: value for key, value in base.items() if key != "domains"}
    cfg.update(base["domains"][domain])
    cfg.update({
        "domain": domain,
        "stage": f"m4_{experiment}_{split_name}",
        "split_name": split_name,
        "split_fingerprint": frozen["split_fingerprint"],
        "split_file_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "matrix": variants(experiment),
    })
    for key in ("batch_size", "evaluation_steps", "stability_repeats", "threads"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be positive")
    if len(set(cfg["seeds"])) != len(cfg["seeds"]) or not cfg["seeds"]:
        raise ValueError("Seeds must be unique and nonempty")
    return cfg, {"selected": selected}


def artifact_path(output: Path, model: str, sample_id: str, cell_id: str,
                  seed: int | None) -> Path:
    stem = f"{model}_{sample_id}_{cell_id}_{seed if seed is not None else 'none'}"
    return ROOT / "results/attributions" / output.name / f"{stem}.npz"


def row_key(row: dict) -> tuple:
    return row["model"], row["id"], row["variant"], row["seed"]


def expected_keys(config: dict, selected: list[dict], experiment: str) -> set[tuple]:
    return {(model, sample["id"], cell["id"], seed)
            for model in config["models"] for sample in selected
            for cell, seed in all_tasks(config, experiment)}


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()


def finish_main(config: dict, output: Path, selection: dict, experiment: str) -> None:
    rows = read_rows(output / "metrics.jsonl")
    expected = expected_keys(config, selection["selected"], experiment)
    actual = [row_key(row) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f"Main records incomplete/duplicate: {len(actual)}/{len(expected)}")
    array_dir = ROOT / "results/attributions" / output.name
    stability = []
    for model in config["models"]:
        for sample in selection["selected"]:
            for cell in variants(experiment):
                if cell["method"] == "ablation":
                    continue
                for left, right in combinations(config["seeds"], 2):
                    vectors = []
                    for seed in (left, right):
                        path = artifact_path(output, model, sample["id"], cell["id"], seed)
                        with np.load(path) as array:
                            vectors.append(array["region_scores"])
                    stability.append({"model": model, "id": sample["id"],
                                      "variant": cell["id"], "method": cell["method"],
                                      "seeds": [left, right], **seed_stability(*vectors)})
    write_json(output / "seed_stability.json", stability)
    write_json(output / "complete_main.json", {
        "rows": len(rows), "expected": len(expected),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    })


def run_main(config: dict, output: Path, metadata: dict, selection: dict,
             experiment: str) -> None:
    domain, _, _, load_image, load_model = domain_api(config)
    rows_path = output / "metrics.jsonl"
    existing = read_rows(rows_path)
    done = {row_key(row) for row in existing}
    if len(done) != len(existing):
        raise ValueError("Duplicate main result keys")
    (ROOT / "results/attributions" / output.name).mkdir(parents=True, exist_ok=True)
    for model_name, checkpoint in config["models"].items():
        predictor = load_model(model_name, ROOT / checkpoint, config["device"])
        warm = load_image(ROOT / selection["selected"][0]["path"])[1]
        for _ in range(config["warmup_batches"]):
            predictor(np.repeat(warm[None], config["batch_size"], axis=0))
        for sample in selection["selected"]:
            tasks = all_tasks(config, experiment)
            order_seed = int(hashlib.sha256(sample["id"].encode()).hexdigest()[:8], 16)
            np.random.default_rng(order_seed).shuffle(tasks)
            if all((model_name, sample["id"], cell["id"], seed) in done
                   for cell, seed in tasks):
                continue
            rgb, image = load_image(ROOT / sample["path"])
            segments_by_count = {}
            target_mask = load_localization_mask(
                domain, sample["id"], ROOT / config["localization_mask_root"], rgb.shape[:2]
            )
            for cell, seed in tasks:
                key = model_name, sample["id"], cell["id"], seed
                if key in done:
                    if not artifact_path(output, *key).is_file():
                        raise ValueError(f"Missing attribution artifact for {key}")
                    continue
                effective = {**config, **cell}
                count = effective["n_segments"]
                if count not in segments_by_count:
                    segments_by_count[count] = slic_segments(
                        rgb, count, config["compactness"], config["sigma"]
                    )
                segments = segments_by_count[count]
                method = cell["method"]
                method_options = method_config(effective, method, seed)
                before_samples, before_batches = predictor.samples, predictor.batches
                predictor.synchronize()
                started = time.perf_counter()
                result, ids, scores, pixel_map = explain(
                    method, image, segments, predictor, sample["target"], method_options
                )
                predictor.synchronize()
                elapsed = time.perf_counter() - started
                forward = predictor.samples - before_samples
                batches = predictor.batches - before_batches
                if (forward, batches) != (result.forward_samples, result.forward_batches):
                    raise AssertionError("Attribution forward count mismatch")
                evaluation_before = predictor.samples
                curves = insertion_deletion(
                    image, segments, scores, predictor, sample["target"],
                    config["batch_size"], config["evaluation_steps"]
                )
                if predictor.samples - evaluation_before != curves["evaluation_forward_samples"]:
                    raise AssertionError("Evaluation forward count mismatch")
                localization = {"localization_energy": None, "pointing_game": None,
                                "zero_positive_energy": None}
                if target_mask is not None:
                    localization = localization_metrics(pixel_map, target_mask)
                artifact = artifact_path(output, *key)
                arrays = {"segments": segments, "region_ids": ids, "region_scores": scores}
                if method == "rise":
                    arrays["saliency"] = pixel_map
                np.savez_compressed(artifact, **arrays)
                diagnostics = {name: value for name, value in asdict(result).items()
                               if not isinstance(value, np.ndarray)}
                row = {
                    "domain": domain, "split": config["split_name"],
                    "experiment": experiment, "model": model_name, "id": sample["id"],
                    "target": sample["target"], "method": method,
                    "variant": cell["id"], "seed": seed,
                    "config": asdict(method_options), "n_segments": count,
                    "actual_regions": len(ids), "attribution_seconds": elapsed,
                    "forward_samples": forward, "forward_batches": batches,
                    "diagnostics": diagnostics, **curves, **localization,
                    "run_fingerprint": metadata["fingerprint"],
                    "artifact": str(artifact.relative_to(ROOT)),
                }
                append_jsonl(rows_path, row)
                done.add(key)
                print(f"{len(done)} done {domain} {model_name} {sample['id']} "
                      f"{cell['id']} seed={seed} {elapsed:.2f}s", flush=True)
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finish_main(config, output, selection, experiment)


def run_stability(config: dict, output: Path, metadata: dict, selection: dict,
                  experiment: str) -> None:
    if not (output / "complete_main.json").is_file():
        raise ValueError("Complete main results before input-stability re-attribution")
    domain, preprocessing, _, load_image, load_model = domain_api(config)
    rows_path = output / "stability.jsonl"
    existing = read_rows(rows_path)
    done = {row_key(row) for row in existing}
    if len(done) != len(existing):
        raise ValueError("Duplicate stability result keys")
    for model_name, checkpoint in config["models"].items():
        predictor = load_model(model_name, ROOT / checkpoint, config["device"])
        for sample in selection["selected"]:
            tasks = all_tasks(config, experiment)
            if all((model_name, sample["id"], cell["id"], seed) in done
                   for cell, seed in tasks):
                continue
            rgb, _ = load_image(ROOT / sample["path"])
            inputs, noise_seed = make_stability_inputs(
                rgb, preprocessing, sample["id"], config["selection_seed"],
                config["stability_repeats"], config["stability_epsilon"]
            )
            predictions = predictor(inputs).argmax(axis=1).tolist()
            for cell, seed in tasks:
                key = model_name, sample["id"], cell["id"], seed
                if key in done:
                    continue
                effective = {**config, **cell}
                options = method_config(effective, cell["method"], seed)
                with np.load(artifact_path(output, *key)) as array:
                    segments = array["segments"]
                    ids = array["region_ids"]
                    scores = array["region_scores"]
                before = predictor.samples
                predictor.synchronize()
                started = time.perf_counter()
                vectors = []
                expected = 0
                for image in inputs:
                    result, new_ids, vector, _ = explain(
                        cell["method"], image, segments, predictor, sample["target"], options
                    )
                    if not np.array_equal(new_ids, ids):
                        raise AssertionError("Region IDs changed during stability evaluation")
                    vectors.append(vector)
                    expected += result.forward_samples
                predictor.synchronize()
                actual = predictor.samples - before
                if actual != expected:
                    raise AssertionError("Stability forward count mismatch")
                row = {
                    "domain": domain, "split": config["split_name"],
                    "experiment": experiment, "model": model_name, "id": sample["id"],
                    "method": cell["method"], "variant": cell["id"], "seed": seed,
                    "stability_seconds": time.perf_counter() - started,
                    "stability_forward_samples": actual,
                    "noise_seed": noise_seed,
                    "prediction_changes": sum(prediction != sample["target"]
                                              for prediction in predictions),
                    **input_stability(scores, np.stack(vectors)),
                    "run_fingerprint": metadata["fingerprint"],
                }
                append_jsonl(rows_path, row)
                done.add(key)
                print(f"{len(done)} stability {domain} {model_name} "
                      f"{sample['id']} {cell['id']} seed={seed}", flush=True)
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    expected = expected_keys(config, selection["selected"], experiment)
    actual = [row_key(row) for row in read_rows(rows_path)]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f"Stability records incomplete: {len(actual)}/{len(expected)}")
    write_json(output / "complete_stability.json", {
        "rows": len(actual), "expected": len(expected),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment/m4.yaml")
    parser.add_argument("--domain", choices=("imagenet", "chncxr"), required=True)
    parser.add_argument("--split", choices=("configuration_selection", "final_evaluation"),
                        default="configuration_selection")
    parser.add_argument("--experiment", choices=("exp1", "exp2"), required=True)
    parser.add_argument("--phase", choices=("main", "stability"), default="main")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Invalid shard index/count")
    config, selection = load_config(args.config, args.domain, args.experiment, args.split)
    selection["selected"] = selection["selected"][args.shard_index::args.shard_count]
    if not selection["selected"]:
        parser.error("Selected shard is empty")
    output = args.output.resolve()
    configure_runtime(config["selection_seed"], config["threads"])
    metadata = freeze_run(config, output, args.resume or args.phase == "stability")
    frozen = output / "selected_shard.json"
    selected_ids = [row["id"] for row in selection["selected"]]
    if frozen.exists():
        prior = json.loads(frozen.read_text(encoding="utf-8"))
        if [row["id"] for row in prior["selected"]] != selected_ids:
            raise ValueError("Selected shard changed")
    else:
        write_json(frozen, {**selection, "shard_index": args.shard_index,
                            "shard_count": args.shard_count})
    if args.phase == "main":
        run_main(config, output, metadata, selection, args.experiment)
    else:
        run_stability(config, output, metadata, selection, args.experiment)


if __name__ == "__main__":
    main()
