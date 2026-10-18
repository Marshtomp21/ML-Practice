"""Run M4 optional experiment 5: OrdShap position attribution on ImageNet.

Phases (see ``configs/experiment/m4_ordshap.yaml`` for the frozen protocol):

``pilot``  OrdShap budget grid on configuration-selection images; writes the
           budget chosen by the pre-registered stability rule.
``main``   OrdShap at the chosen budget and KernelSHAP on the same patch
           lattice, five seeds each, on a seeded final-evaluation subset.
``moves``  Position-move check: swap the top-k patches of each selection rule
           with partner slots and record the target-logit change, plus a
           deletion control on the same patches.

Rows are append-only and keyed, so every phase resumes after interruption.
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
from attribution.game_theoretic.ordshap import (
    OrdShapConfig,
    compose_patches,
    fit_ordshap,
    patch_labels,
    split_patches,
)
from evaluation.perturbation import insertion_deletion, seed_stability
from experiments.perturbation_pilot import (
    ROOT,
    domain_api,
    freeze_run,
    read_rows,
    seeded_subset,
    write_json,
)
from models.inference import configure_runtime

PHASES = ("pilot", "main", "moves")


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("stage") != "m4_ordshap" or config.get("domain") != "imagenet":
        raise ValueError("Expected stage=m4_ordshap on ImageNet")
    split_path = ROOT / config["split_file"]
    config["split_file_sha256"] = hashlib.sha256(split_path.read_bytes()).hexdigest()
    return config


def selected_images(config: dict, phase: str) -> list[dict]:
    split = json.loads((ROOT / config["split_file"]).read_text(encoding="utf-8"))
    pool_name = "configuration_selection" if phase == "pilot" else "final_evaluation"
    count = config["pilot" if phase == "pilot" else "main"]["images"]
    return seeded_subset(split["domains"]["imagenet"][pool_name], count, config["selection_seed"])


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()


def budget_id(budget: dict) -> str:
    return f"k{budget['num_coalitions']}_p{budget['num_permutations']}"


def ordshap_config(config: dict, budget: dict, seed: int) -> OrdShapConfig:
    return OrdShapConfig(
        num_coalitions=budget["num_coalitions"],
        num_permutations=budget["num_permutations"],
        full_permutations=10 * budget["num_permutations"],
        batch_size=config["batch_size"], paired_sampling=True, seed=seed)


def tasks(config: dict, phase: str, budget: dict | None) -> list[tuple[str, str, int]]:
    """(method, cell id, seed) triples for one image and model."""
    if phase == "pilot":
        return [("ordshap", budget_id(b), seed) for b in config["pilot"]["budgets"]
                for seed in config["pilot"]["seeds"]]
    seeds = config["main"]["seeds"]
    return ([("ordshap", budget_id(budget), seed) for seed in seeds]
            + [("kernel_shap_patch", f"n{config['main']['kernel_shap_num_samples']}", seed)
               for seed in seeds])


def artifact_path(output: Path, model: str, sample_id: str, cell: str, seed: int) -> Path:
    return ROOT / "results/attributions" / output.name / f"{model}_{sample_id}_{cell}_{seed}.npz"


def chosen_budget(pilot_output: Path) -> dict:
    path = pilot_output / "budget_selection.json"
    if not path.is_file():
        raise ValueError("Run the pilot phase first; budget_selection.json is missing")
    return json.loads(path.read_text(encoding="utf-8"))["selected"]


def run_attributions(config: dict, output: Path, metadata: dict, images: list[dict],
                     phase: str, budget: dict | None) -> None:
    _, _, _, load_image, load_model = domain_api(config)
    rows_path = output / "metrics.jsonl"
    done = {(r["model"], r["id"], r["cell"], r["seed"]) for r in read_rows(rows_path)}
    budgets = {budget_id(b): b for b in (config["pilot"]["budgets"] if phase == "pilot"
                                         else [budget])}
    (ROOT / "results/attributions" / output.name).mkdir(parents=True, exist_ok=True)
    grid = config["grid"]
    for model_name, checkpoint in config["models"].items():
        predictor = load_model(model_name, ROOT / checkpoint, config["device"])
        for sample in images:
            pending = [t for t in tasks(config, phase, budget)
                       if (model_name, sample["id"], t[1], t[2]) not in done]
            if not pending:
                continue
            _, image = load_image(ROOT / sample["path"])
            labels = patch_labels(*image.shape[:2], grid)
            for method, cell, seed in pending:
                predictor.synchronize()
                before = predictor.samples
                started = time.perf_counter()
                if method == "ordshap":
                    result = fit_ordshap(image, grid, predictor, sample["target"],
                                         ordshap_config(config, budgets[cell], seed))
                    scores = result.value_importance
                    arrays = {"value_importance": result.value_importance,
                              "position_importance": result.position_importance,
                              "position_importance_2d": result.position_importance_2d}
                else:
                    result = fit_kernel_shap(image, labels, predictor, sample["target"],
                                             KernelShapConfig(
                                                 num_samples=config["main"]["kernel_shap_num_samples"],
                                                 batch_size=config["batch_size"], seed=seed))
                    scores = result.coefficients
                    arrays = {"value_importance": result.coefficients}
                predictor.synchronize()
                elapsed = time.perf_counter() - started
                forward = predictor.samples - before
                if forward != result.forward_samples:
                    raise AssertionError("Attribution forward count mismatch")
                curves = insertion_deletion(image, labels, scores, predictor, sample["target"],
                                            config["batch_size"], config["evaluation_steps"])
                artifact = artifact_path(output, model_name, sample["id"], cell, seed)
                np.savez_compressed(artifact, **arrays)
                diagnostics = {k: v for k, v in asdict(result).items()
                               if not isinstance(v, np.ndarray)}
                append_jsonl(rows_path, {
                    "phase": phase, "model": model_name, "id": sample["id"],
                    "target": sample["target"], "method": method, "cell": cell, "seed": seed,
                    "attribution_seconds": elapsed, "forward_samples": forward,
                    "diagnostics": diagnostics,
                    **{k: curves[k] for k in ("insertion_auc", "deletion_auc",
                                              "normalized_insertion_auc",
                                              "normalized_deletion_auc",
                                              "evaluation_forward_samples")},
                    "run_fingerprint": metadata["fingerprint"],
                    "artifact": str(artifact.relative_to(ROOT)),
                })
                done.add((model_name, sample["id"], cell, seed))
                print(f"{len(done)} {phase} {model_name} {sample['id']} {cell} "
                      f"seed={seed} {elapsed:.1f}s", flush=True)
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    expected = {(m, s["id"], t[1], t[2]) for m in config["models"] for s in images
                for t in tasks(config, phase, budget)}
    if done != expected:
        raise ValueError(f"Incomplete {phase} records: {len(done)}/{len(expected)}")


def pairwise_stability(output: Path, rows: list[dict], key: str) -> dict:
    """Mean pairwise cross-seed Spearman/Jaccard of an artifact array per (model, id, cell)."""
    grouped = {}
    for row in rows:
        grouped.setdefault((row["model"], row["id"], row["cell"]), []).append(row)
    result = {}
    for group, members in grouped.items():
        vectors = {}
        for row in members:
            with np.load(ROOT / row["artifact"]) as array:
                if key in array:
                    vectors[row["seed"]] = array[key]
        if len(vectors) < 2:
            continue
        pairs = [seed_stability(vectors[a], vectors[b])
                 for a, b in combinations(sorted(vectors), 2)]
        result[group] = {
            "spearman": float(np.mean([p["spearman"] for p in pairs
                                       if p["spearman"] is not None]))
            if any(p["spearman"] is not None for p in pairs) else None,
            "top20_jaccard": float(np.mean([p["top20_jaccard"] for p in pairs])),
        }
    return result


def select_budget(config: dict, output: Path) -> None:
    rows = read_rows(output / "metrics.jsonl")
    vi = pairwise_stability(output, rows, "value_importance")
    pi = pairwise_stability(output, rows, "position_importance")
    table = []
    for budget in config["pilot"]["budgets"]:
        cell = budget_id(budget)
        cell_rows = [r for r in rows if r["cell"] == cell]

        def median(values):
            values = [v for v in values if v is not None]
            return float(np.median(values)) if values else None

        table.append({
            "cell": cell, **budget,
            "median_vi_spearman": median([v["spearman"] for k, v in vi.items() if k[2] == cell]),
            "median_pi_spearman": median([v["spearman"] for k, v in pi.items() if k[2] == cell]),
            "median_forward_samples": median([r["forward_samples"] for r in cell_rows]),
            "median_seconds": median([r["attribution_seconds"] for r in cell_rows]),
        })
    qualifying = [row for row in table
                  if (row["median_vi_spearman"] or 0) >= 0.9
                  and (row["median_pi_spearman"] or 0) >= 0.8]
    pool = qualifying or table
    pick = (min(pool, key=lambda r: r["median_forward_samples"]) if qualifying
            else max(pool, key=lambda r: r["median_forward_samples"]))
    write_json(output / "budget_selection.json", {
        "rule": "cheapest budget with median VI Spearman >= 0.9 and PI Spearman >= 0.8; "
                "else the most expensive",
        "qualified": bool(qualifying), "table": table,
        "selected": {"num_coalitions": pick["num_coalitions"],
                     "num_permutations": pick["num_permutations"]},
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    })


def rule_sets(config: dict, output: Path, rows: list[dict], model: str,
              sample_id: str) -> dict[str, list[int]]:
    """Top-k patches by seed-averaged |score| per rule, plus a seeded random set."""
    k = config["moves"]["k"]
    scores = {}
    for rule, method, key in (("ordshap_pi", "ordshap", "position_importance"),
                              ("ordshap_vi", "ordshap", "value_importance"),
                              ("kernel_shap", "kernel_shap_patch", "value_importance")):
        stack = []
        for row in rows:
            if (row["model"], row["id"], row["method"]) == (model, sample_id, method):
                with np.load(ROOT / row["artifact"]) as array:
                    stack.append(array[key])
        scores[rule] = np.abs(np.mean(stack, axis=0))
    d = config["grid"] ** 2
    sets = {rule: sorted(np.argsort(-value, kind="stable")[:k].tolist())
            for rule, value in scores.items()}
    digest = int(hashlib.sha256(f"{config['selection_seed']}:{sample_id}:random".encode())
                 .hexdigest()[:8], 16)
    sets["random"] = sorted(np.random.default_rng(digest).choice(d, k, replace=False).tolist())
    return {rule: sets[rule] for rule in config["moves"]["rules"]}


def move_permutations(config: dict, sample_id: str, chosen: list[int]) -> np.ndarray:
    """Swap each chosen patch with a partner slot; partner order shared across rules."""
    d = config["grid"] ** 2
    perms = []
    for repeat in range(config["moves"]["repeats"]):
        digest = int(hashlib.sha256(
            f"{config['selection_seed']}:{sample_id}:move:{repeat}".encode()).hexdigest()[:8], 16)
        order = np.random.default_rng(digest).permutation(d)
        partners = [int(slot) for slot in order if slot not in chosen][:len(chosen)]
        perm = np.arange(d)
        for patch, partner in zip(chosen, partners):
            perm[patch], perm[partner] = perm[partner], perm[patch]
        perms.append(perm)
    return np.stack(perms)


def run_moves(config: dict, output: Path, metadata: dict, images: list[dict]) -> None:
    _, _, _, load_image, load_model = domain_api(config)
    attribution_rows = read_rows(output / "metrics.jsonl")
    rows_path = output / "moves.jsonl"
    done = {(r["model"], r["id"], r["rule"]) for r in read_rows(rows_path)}
    grid, d = config["grid"], config["grid"] ** 2
    for model_name, checkpoint in config["models"].items():
        predictor = load_model(model_name, ROOT / checkpoint, config["device"])
        for sample in images:
            sets = rule_sets(config, output, attribution_rows, model_name, sample["id"])
            if all((model_name, sample["id"], rule) in done for rule in sets):
                continue
            _, image = load_image(ROOT / sample["path"])
            patches = split_patches(image, grid)
            fill = np.zeros(image.shape[-1], dtype=image.dtype)
            full = np.ones((1, d), dtype=bool)
            original = float(predictor(image[None])[0, sample["target"]])
            for rule, chosen in sets.items():
                if (model_name, sample["id"], rule) in done:
                    continue
                perms = move_permutations(config, sample["id"], chosen)
                moved = []
                for start in range(0, len(perms), config["batch_size"]):
                    batch = perms[start:start + config["batch_size"]]
                    images_batch = compose_patches(patches, fill, np.repeat(full, len(batch), 0),
                                                   batch, grid)
                    moved.extend(predictor(images_batch)[:, sample["target"]].tolist())
                deleted_mask = full.copy()
                deleted_mask[0, chosen] = False
                deleted = float(predictor(compose_patches(
                    patches, fill, deleted_mask, np.arange(d)[None], grid))[0, sample["target"]])
                drops = original - np.asarray(moved)
                affected = sorted({int(s) for perm in perms for s in np.flatnonzero(perm != np.arange(d))})
                append_jsonl(rows_path, {
                    "model": model_name, "id": sample["id"], "target": sample["target"],
                    "rule": rule, "patches": chosen, "original_logit": original,
                    "mean_move_drop": float(drops.mean()),
                    "mean_abs_move_change": float(np.abs(drops).mean()),
                    "deletion_drop": original - deleted,
                    "permutations": perms.tolist(), "affected_slots": affected,
                    "run_fingerprint": metadata["fingerprint"],
                })
                done.add((model_name, sample["id"], rule))
                print(f"moves {model_name} {sample['id']} {rule} "
                      f"|Δ|={np.abs(drops).mean():.3f}", flush=True)
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment/m4_ordshap.yaml")
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=None,
                        help="override the output directory of this phase")
    parser.add_argument("--pilot-output", type=Path,
                        default=ROOT / "results/metrics/m4_ordshap_pilot")
    args = parser.parse_args()
    config = load_config(args.config)
    configure_runtime(config["selection_seed"], config["threads"])
    if args.phase == "pilot":
        output, budget = args.pilot_output, None
    else:
        output = ROOT / "results/metrics/m4_ordshap"
        budget = chosen_budget(args.pilot_output.resolve())
        config["main"]["ordshap_budget"] = budget
    output = (args.output or output).resolve()
    images = selected_images(config, args.phase)
    metadata = freeze_run(config, output, args.resume or args.phase == "moves")
    write_json(output / f"selection_{'pilot' if args.phase == 'pilot' else 'main'}.json",
               {"images": images})
    if args.phase == "moves":
        run_moves(config, output, metadata, images)
    else:
        run_attributions(config, output, metadata, images, args.phase, budget)
        if args.phase == "pilot":
            select_budget(config, output)
    write_json(output / f"complete_{args.phase}.json",
               {"completed_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
