"""Run M4 ViT extension phases on disjoint sample workers and merge records."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from experiments import m4_vit_extension as extension
from experiments.perturbation_pilot import freeze_run, read_rows, write_json
from models.inference import configure_runtime


def append_rows(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("a", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_context(experiment: str):
    config_path = ROOT / "configs/experiment/m4_vit_extension.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    extension.validate_config(config)
    output = ROOT / "results/metrics" / f"m4_vit_{experiment}"
    if not (output / "selection.json").is_file():
        raise RuntimeError(f"Prepare the frozen selection first: {output}")
    metadata = freeze_run(config, output, True)
    selected = read_json(output / "selection.json")["selected"]
    return config, output, metadata, selected


def existing_rows(output: Path, phase: str) -> list[dict]:
    filename = "metrics.jsonl" if phase == "main" else "stability.jsonl"
    return read_rows(output / filename)


def assign(config: dict, experiment: str, selected: list[dict], rows: list[dict],
           phase: str, gpus: list[int]) -> list[list[dict]]:
    done = defaultdict(set)
    for row in rows:
        done[row["id"]].add(extension.row_key(row))
    total_per_image = len(extension.experiment_models(config, experiment)) * len(
        extension.tasks(config, experiment)
    )
    ordered = sorted(selected, key=lambda item: len(done[item["id"]]))
    loads = [0] * len(gpus)
    groups = [[] for _ in gpus]
    for sample in ordered:
        index = min(range(len(gpus)), key=lambda i: loads[i])
        groups[index].append(sample)
        loads[index] += total_per_image - len(done[sample["id"]])
    return groups


def worker(experiment: str, phase: str, gpu: int, samples: list[dict],
           config: dict, canonical: Path, metadata: dict) -> None:
    if not samples:
        print(f"GPU {gpu}: no samples", flush=True)
        return
    name = f"m4_vit_{experiment}_{phase}_aux_gpu{gpu}"
    output = ROOT / "results/.cache" / name
    sample_ids = [sample["id"] for sample in samples]
    if output.exists():
        manifest = read_json(output / "worker_manifest.json")
        if manifest["sample_ids"] != sample_ids or manifest["phase"] != phase:
            raise RuntimeError(f"Worker assignment changed: {output}")
        worker_metadata = freeze_run(config, output, True)
    else:
        worker_metadata = freeze_run(config, output, False)
        write_json(output / "worker_manifest.json", {
            "experiment": experiment, "phase": phase, "gpu": gpu,
            "sample_ids": sample_ids,
            "canonical_output": str(canonical.relative_to(ROOT)),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })
        source_rows = existing_rows(canonical, phase)
        filename = "metrics.jsonl" if phase == "main" else "stability.jsonl"
        append_rows(output / filename, [row for row in source_rows if row["id"] in set(sample_ids)])
        if phase == "stability":
            write_json(output / "complete_main.json", read_json(canonical / "complete_main.json"))

    attr_dir = ROOT / "results/attributions" / canonical.name
    attr_dir.mkdir(parents=True, exist_ok=True)

    def resolve_artifact(_output, exp, model, sample_id, variant, seed):
        seed_text = "none" if seed is None else str(seed)
        filename = f"{exp}_{model}_{sample_id}_{variant}_{seed_text}.npz"
        return attr_dir / filename

    extension.artifact_path = resolve_artifact
    configure_runtime(config["selection_seed"], config["threads"])
    config = {**config, "device": "cuda"}
    print(f"GPU {gpu}: {phase} on {len(samples)} images", flush=True)
    if phase == "main":
        extension.run_main(config, output, worker_metadata, samples, experiment)
    else:
        extension.run_stability(config, output, worker_metadata, samples, experiment)


def merge(experiment: str, phase: str) -> None:
    config, canonical, metadata, selected = load_context(experiment)
    phase_aux = sorted((ROOT / "results/.cache").glob(
        f"m4_vit_{experiment}_{phase}_aux_gpu*"
    ))
    if len(phase_aux) < 1:
        raise RuntimeError("No auxiliary worker results found")
    marker = "complete_main.json" if phase == "main" else "complete_stability.json"
    filename = "metrics.jsonl" if phase == "main" else "stability.jsonl"
    target_path = canonical / filename
    rows = read_rows(target_path)
    keys = {extension.row_key(row) for row in rows}
    appended = 0
    for aux in phase_aux:
        if not (aux / marker).is_file():
            raise RuntimeError(f"Worker is incomplete: {aux}")
        for row in read_rows(aux / filename):
            key = extension.row_key(row)
            if key not in keys:
                append_rows(target_path, [row])
                keys.add(key)
                appended += 1

    rows = read_rows(target_path)
    expected = extension.expected_keys(config, selected, experiment)
    actual = [extension.row_key(row) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError(f"Merged {phase} rows incomplete/duplicate: {len(actual)}/{len(expected)}")

    if phase == "main":
        extension.artifact_path = lambda _output, exp, model, sample_id, variant, seed: (
            ROOT / "results/attributions" / canonical.name /
            f"{exp}_{model}_{sample_id}_{variant}_{'none' if seed is None else seed}.npz"
        )
        extension.finish_main(config, canonical, selected, experiment)
    else:
        extension.summarize(rows, canonical / "stability_summary.json", (
            "max_sensitivity", "cosine_similarity", "stability_seconds",
            "stability_forward_samples", "prediction_changes",
        ))
        if experiment == "architecture":
            reused = extension.reused_m3_rows(config, selected)
            extension.summarize(reused + rows,
                                canonical / "architecture_stability_comparison_summary.json", (
                "max_sensitivity", "cosine_similarity", "stability_seconds",
                "stability_forward_samples", "prediction_changes",
            ))
        write_json(canonical / "complete_stability.json", {
            "experiment": experiment, "rows": len(rows), "expected": len(expected),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "auxiliary_workers": [path.name for path in phase_aux],
        })
    print(f"Merged {appended} {phase} rows into {canonical}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("architecture", "partition"), required=True)
    parser.add_argument("--phase", choices=("main", "stability"), required=True)
    parser.add_argument("--gpus", default="2,3,4,6")
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        merge(args.experiment, args.phase)
        return
    gpus = [int(value) for value in args.gpus.split(",")]
    if args.worker_index is None or not 0 <= args.worker_index < len(gpus):
        raise SystemExit("A valid --worker-index is required")
    config, canonical, metadata, selected = load_context(args.experiment)
    rows = existing_rows(canonical, args.phase)
    groups = assign(config, args.experiment, selected, rows, args.phase, gpus)
    worker(args.experiment, args.phase, gpus[args.worker_index],
           groups[args.worker_index], config, canonical, metadata)


if __name__ == "__main__":
    main()
