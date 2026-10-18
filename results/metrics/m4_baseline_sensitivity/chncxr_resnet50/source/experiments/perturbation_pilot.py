"""Run a reproducible ImageNet or CHNCXR pilot for perturbation methods.

Screen the entire candidate pool with both models, freeze their jointly correct
intersection, and select a seeded balanced pilot before computing explanations.
All paths in the YAML are relative to the repository root. No plots are made.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
import yaml

from attribution.perturbation.ablation import AblationConfig, fit_ablation
from attribution.perturbation.lime import LimeConfig, fit_lime
from attribution.perturbation.rise import RiseConfig, explain_rise, region_scores_from_saliency
from evaluation.perturbation import insertion_deletion, seed_stability
from models.inference import configure_runtime, load_chncxr, load_imagenet
from preprocessing.superpixels import slic_segments

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("attribution", "preprocessing", "models", "evaluation", "experiments")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def balanced_subset(rows, per_class, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for label in (0, 1):
        pool = sorted([r for r in rows if r["target"] == label], key=lambda r: r["id"])
        if len(pool) < per_class:
            raise ValueError(f"Only {len(pool)} jointly correct class-{label} images; need {per_class}")
        indices = rng.choice(len(pool), per_class, replace=False)
        selected.extend(pool[int(i)] for i in indices)
    return sorted(selected, key=lambda r: r["id"])


def seeded_subset(rows, count, seed):
    if len(rows) < count:
        raise ValueError(f"Only {len(rows)} jointly correct images; need {count}")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(rows), count, replace=False)
    return sorted((sorted(rows, key=lambda r: r["id"])[int(i)] for i in indices),
                  key=lambda r: r["id"])


def select_samples(rows, config):
    mode = config.get("selection", "balanced_subset")
    if mode == "all_joint_correct":
        return sorted(rows, key=lambda row: row["id"])
    if mode == "balanced_subset":
        return balanced_subset(rows, config["samples_per_class"], config["selection_seed"])
    if mode == "seeded_subset":
        return seeded_subset(rows, config["sample_count"], config["selection_seed"])
    raise ValueError("selection must be 'all_joint_correct', 'balanced_subset' or 'seeded_subset'")


def domain_api(config):
    domain = config.get("domain", "chncxr").lower()
    if domain == "chncxr":
        from preprocessing.chncxr import PROTOCOL, candidates, load_image
        return domain, PROTOCOL, lambda: candidates(ROOT/config["data_root"]), load_image, load_chncxr
    if domain == "imagenet":
        from preprocessing.imagenet import PROTOCOL, candidates, load_image
        make_candidates = lambda: candidates(ROOT/config["data_root"],
                                              ROOT/config["annotation_root"],
                                              ROOT/config["class_index"])
        return domain, PROTOCOL, make_candidates, load_image, load_imagenet
    raise ValueError("domain must be 'chncxr' or 'imagenet'")


def validate_config(config):
    if set(config["models"]) != {"resnet50", "vgg13"}:
        raise ValueError("Pilot requires both resnet50 and vgg13 for joint-correct screening")
    for key in ("batch_size", "n_segments", "evaluation_steps", "threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config.get("selection", "balanced_subset") == "balanced_subset":
        if type(config.get("samples_per_class")) is not int or config["samples_per_class"] < 1:
            raise ValueError("samples_per_class must be a positive integer for balanced_subset")
    if config.get("selection") == "seeded_subset":
        if type(config.get("sample_count")) is not int or config["sample_count"] < 1:
            raise ValueError("sample_count must be a positive integer for seeded_subset")
    if type(config["warmup_batches"]) is not int or config["warmup_batches"] < 1:
        raise ValueError("warmup_batches must be >= 1")
    seeds = config["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    for seed in seeds:
        LimeConfig(config["num_samples"], config["batch_size"], config["mask_rate"],
                   config["kernel_width"], config["ridge_alpha"], seed)
        RiseConfig(config["num_samples"], config["batch_size"], config["grid_size"],
                   config["mask_rate"], seed)


def freeze_run(config, output, resume):
    domain, protocol, make_candidates, _, _ = domain_api(config)
    source = {str(p.relative_to(ROOT)): sha256(p)
              for folder in SOURCE_DIRS for p in sorted((ROOT/folder).rglob("*.py"))}
    pool = make_candidates()
    for row in pool:
        row["sha256"] = sha256(row["path"])
        row["path"] = str(Path(row["path"]).relative_to(ROOT))
    weights = {name: sha256(ROOT/path) for name, path in config["models"].items()}
    import importlib.metadata
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "torch", "torchvision", "scipy", "scikit-learn", "scikit-image", "Pillow", "PyYAML")}
    protocol = {"config": config, "domain": domain, "preprocessing": protocol,
                "weights_sha256": weights,
                "candidate_pool": pool, "source_sha256": source, "versions": versions,
                "device_name": torch.cuda.get_device_name() if config["device"].startswith("cuda") else "CPU"}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if output.exists():
        if not resume:
            raise ValueError("Output exists; use a new directory or --resume")
        saved = json.loads((output/"run.json").read_text(encoding="utf-8"))
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Configuration, source, data, weights or environment changed; start a new run")
        return saved
    output.mkdir(parents=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True)
    stage = config.get("stage", "pilot")
    saved = {**protocol, "fingerprint": fingerprint, "created_utc": datetime.now(timezone.utc).isoformat(),
             "python": platform.python_version(), "git_commit": commit.stdout.strip(),
             "git_status": status.stdout, "stage": stage, "results_are_exploratory": stage == "pilot"}
    write_json(output/"run.json", saved)
    for relative in source:
        destination = output/"source"/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/relative).read_bytes())
    return saved


def screen(config, output, metadata):
    selection_file = output/"selection.json"
    if selection_file.exists():
        return json.loads(selection_file.read_text(encoding="utf-8"))
    domain, _, _, load_image, load_model = domain_api(config)
    pool = metadata["candidate_pool"]
    for name, checkpoint in config["models"].items():
        path = output/f"screen_{name}.json"
        if path.exists():
            continue
        predictor = load_model(name, ROOT/checkpoint, config["device"])
        rows = []
        for start in range(0, len(pool), config["batch_size"]):
            batch = pool[start:start+config["batch_size"]]
            logits = predictor(np.stack([load_image(ROOT/r["path"])[1] for r in batch]))
            rows.extend({**row, "prediction": int(logit.argmax()), "logits": logit.tolist()}
                        for row, logit in zip(batch, logits))
            print(f"screen {name}: {len(rows)}/{len(pool)}", flush=True)
        write_json(path, {"rows": rows, "forward_samples": predictor.samples,
                          "accuracy": float(np.mean([r["prediction"] == r["target"] for r in rows]))})
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    screenings = {name: json.loads((output/f"screen_{name}.json").read_text(encoding="utf-8"))
                  for name in config["models"]}
    correct = [{r["id"] for r in report["rows"] if r["prediction"] == r["target"]}
               for report in screenings.values()]
    intersection = set.intersection(*correct)
    eligible = [r for r in pool if r["id"] in intersection]
    targets = sorted({r["target"] for r in eligible})
    selection = {"candidate_count": len(pool), "joint_correct_count": len(eligible),
                 "joint_correct_by_class": {
                     str(k): sum(r["target"] == k for r in eligible) for k in targets},
                 "model_accuracy": {name: report["accuracy"] for name, report in screenings.items()},
                 "joint_correct": eligible,
                 "selected": select_samples(eligible, config)}
    write_json(selection_file, selection)
    split_dir = ROOT/"data/splits"/output.name
    split_dir.mkdir(parents=True, exist_ok=True)
    write_json(split_dir/f"{domain}_selection.json",
               {"run_fingerprint": metadata["fingerprint"], **selection})
    return selection


def summarize(output):
    rows = read_rows(output/"metrics.jsonl")
    summary = []
    # Average seeds within image first; images, not repeated seeds, are units.
    metrics = ("insertion_auc", "deletion_auc", "attribution_seconds", "forward_samples")
    for model, method in sorted({(r["model"], r["method"]) for r in rows}):
        selected = [r for r in rows if (r["model"], r["method"]) == (model, method)]
        ids = sorted({r["id"] for r in selected})
        record = {"model": model, "method": method, "images": len(ids), "runs": len(selected)}
        for metric in metrics:
            values = [np.mean([r[metric] for r in selected if r["id"] == sid]) for sid in ids]
            record[metric] = {"median": float(np.median(values)),
                              "q25": float(np.percentile(values, 25)), "q75": float(np.percentile(values, 75))}
        summary.append(record)
    write_json(output/"summary.json", summary)
    return rows, summary


def run_experiments(config, output, metadata, selection):
    _, _, _, load_image, load_model = domain_api(config)
    done = {(r["model"], r["id"], r["method"], r["seed"]) for r in read_rows(output/"metrics.jsonl")}
    array_dir = ROOT/"results/attributions"/output.name
    array_dir.mkdir(parents=True, exist_ok=True)
    for name, checkpoint in config["models"].items():
        predictor = load_model(name, ROOT/checkpoint, config["device"])
        warm = load_image(ROOT/selection["selected"][0]["path"])[1]
        for _ in range(config["warmup_batches"]):
            predictor(np.repeat(warm[None], config["batch_size"], axis=0))
        for sample in selection["selected"]:
            rgb, image = load_image(ROOT/sample["path"])
            segments = slic_segments(rgb, config["n_segments"], config["compactness"], config["sigma"])
            tasks = [(method, seed) for seed in config["seeds"] for method in ("lime", "rise")]
            tasks.append(("ablation", None))
            # Rotate method execution order per image to reduce fixed ordering bias.
            schedule_seed = int(hashlib.sha256(sample["id"].encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(schedule_seed)
            rng.shuffle(tasks)
            for method, seed in tasks:
                key = (name, sample["id"], method, seed)
                stem = f"{name}_{sample['id']}_{method}_{seed}"
                if key in done:
                    if not (array_dir/f"{stem}.npz").exists():
                        raise ValueError(f"Missing attribution artifact for {key}")
                    continue
                before, before_batches = predictor.samples, predictor.batches
                predictor.synchronize()
                started = time.perf_counter()
                if method == "lime":
                    method_config = LimeConfig(config["num_samples"], config["batch_size"], config["mask_rate"],
                                               config["kernel_width"], config["ridge_alpha"], seed)
                    result = fit_lime(image, segments, predictor, sample["target"], method_config)
                elif method == "ablation":
                    method_config = AblationConfig(config["batch_size"])
                    result = fit_ablation(image, segments, predictor, sample["target"], method_config)
                else:
                    method_config = RiseConfig(config["num_samples"], config["batch_size"], config["grid_size"],
                                               config["mask_rate"], seed)
                    result = explain_rise(image, predictor, sample["target"], method_config)
                predictor.synchronize()
                elapsed = time.perf_counter() - started
                actual_samples, actual_batches = predictor.samples-before, predictor.batches-before_batches
                if (actual_samples, actual_batches) != (result.forward_samples, result.forward_batches):
                    raise AssertionError("Attribution budget disagrees with model adapter")
                if method == "rise":
                    ids, scores = region_scores_from_saliency(result.saliency, segments)
                else:
                    ids, scores = result.region_ids, result.coefficients
                evaluation_started = time.perf_counter()
                evaluation_before = predictor.samples
                curves = insertion_deletion(image, segments, scores, predictor, sample["target"],
                                            config["batch_size"], config["evaluation_steps"])
                predictor.synchronize()
                if predictor.samples-evaluation_before != curves["evaluation_forward_samples"]:
                    raise AssertionError("Evaluation budget disagrees with adapter")
                evaluation_seconds = time.perf_counter()-evaluation_started
                arrays = {"segments": segments, "region_ids": ids, "region_scores": scores}
                if method == "rise":
                    arrays["saliency"] = result.saliency
                np.savez_compressed(array_dir/f"{stem}.npz", **arrays)
                diagnostics = {k: v for k, v in asdict(result).items() if not isinstance(v, np.ndarray)}
                row = {"model": name, "id": sample["id"], "target": sample["target"], "method": method,
                       "seed": seed, "config": asdict(method_config), "actual_regions": len(ids),
                       "attribution_seconds": elapsed, "evaluation_seconds": evaluation_seconds,
                       "forward_samples": actual_samples, "forward_batches": actual_batches,
                       "diagnostics": diagnostics, **curves,
                       "run_fingerprint": metadata["fingerprint"],
                       "artifact": str((array_dir/f"{stem}.npz").relative_to(ROOT))}
                with (output/"metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                done.add(key)
                print(f"{len(done)} done: {name} {sample['id']} {method} seed={seed} {elapsed:.2f}s", flush=True)
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rows, summary = summarize(output)
    stability = []
    for name in config["models"]:
        for sample in selection["selected"]:
            for method in ("lime", "rise"):
                for a, b in combinations(config["seeds"], 2):
                    vectors = []
                    for seed in (a, b):
                        with np.load(array_dir/f"{name}_{sample['id']}_{method}_{seed}.npz") as data:
                            vectors.append(data["region_scores"])
                    stability.append({"model": name, "id": sample["id"], "method": method,
                                      "seeds": [a, b], **seed_stability(*vectors)})
    write_json(output/"seed_stability.json", stability)
    expected = len(config["models"])*len(selection["selected"])*(2*len(config["seeds"])+1)
    if len(rows) != expected:
        raise AssertionError(f"Expected {expected} rows, found {len(rows)}")
    write_json(output/"complete.json", {"rows": len(rows), "expected": expected,
                                        "completed_utc": datetime.now(timezone.utc).isoformat()})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/experiment/perturbation_pilot.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    if args.dry_run:
        mode = config.get("selection", "balanced_subset")
        images = (2*config["samples_per_class"] if mode == "balanced_subset" else
                  config["sample_count"] if mode == "seeded_subset" else
                  "determined by joint-correct screening")
        runs = (images*len(config["models"])*(2*len(config["seeds"])+1)
                if isinstance(images, int) else "determined after screening")
        print(json.dumps({"config": config, "images": images, "runs": runs}, indent=2))
        return
    output = args.output.resolve()
    configure_runtime(config["selection_seed"], config["threads"])
    metadata = freeze_run(config, output, args.resume)
    try:
        selection = screen(config, output, metadata)
        print("Selected: " + ", ".join(r["id"] for r in selection["selected"]), flush=True)
        run_experiments(config, output, metadata, selection)
    except Exception as exc:
        with (output/"failures.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(),
                                     "type": type(exc).__name__, "message": str(exc)}) + "\n")
        raise


if __name__ == "__main__":
    main()
