"""Resume M4 stability work on disjoint auxiliary workers, then merge it.

The helper keeps the existing frozen shard outputs intact while work is
rebalanced across GPUs. It imports existing per-image records into each worker,
so completed stability cells are skipped by the regular M4 runner.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import m4_matrix
from experiments.perturbation_pilot import freeze_run, read_rows, write_json
from models.inference import configure_runtime


DOMAIN = "chncxr"
EXPERIMENT = "exp2"
SPLIT_NAME = "configuration_selection"
ORIGINAL_NAMES = [f"m4_{DOMAIN}_config_exp2_shard{i}" for i in range(2)]
EXPECTED_TASKS_PER_IMAGE = 2 * len(m4_matrix.all_tasks({"seeds": [0, 1, 2, 3, 4]}, EXPERIMENT))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()


def load_sources():
    config, selection = m4_matrix.load_config(
        ROOT / "configs/experiment/m4.yaml", DOMAIN, EXPERIMENT, SPLIT_NAME
    )
    original_dirs = [ROOT / "results/metrics" / name for name in ORIGINAL_NAMES]
    source_rows = []
    for directory in original_dirs:
        if not (directory / "complete_main.json").is_file():
            raise RuntimeError(f"Main results are incomplete: {directory}")
        source_rows.extend(read_rows(directory / "stability.jsonl"))
    return config, selection, original_dirs, source_rows


def assign_workers(config: dict, selection: dict, rows: list[dict], gpus: list[int]):
    counts = defaultdict(set)
    for row in rows:
        counts[row["id"]].add(m4_matrix.row_key(row))
    pending = sorted(
        selection["selected"],
        key=lambda sample: len(counts[sample["id"]]),
    )
    loads = [0] * len(gpus)
    groups = [[] for _ in gpus]
    for sample in pending:
        index = min(range(len(gpus)), key=lambda i: loads[i])
        groups[index].append(sample)
        loads[index] += EXPECTED_TASKS_PER_IMAGE - len(counts[sample["id"]])
    return groups


def run_worker(gpu: int, samples: list[dict], config: dict, original_dirs: list[Path],
               source_rows: list[dict]) -> None:
    if not samples:
        print(f"GPU {gpu}: no pending images", flush=True)
        return
    output = ROOT / "results/metrics" / f"m4_{DOMAIN}_exp2_stability_aux_gpu{gpu}"
    ids = [sample["id"] for sample in samples]
    if output.exists():
        manifest = read_json(output / "worker_manifest.json")
        if manifest["sample_ids"] != ids:
            raise RuntimeError(f"Worker assignment changed for {output}")
        metadata = freeze_run(config, output, True)
    else:
        metadata = freeze_run(config, output, False)
        write_json(output / "complete_main.json", {"source": "original M4 main shards"})
        write_json(output / "worker_manifest.json", {
            "gpu": gpu, "sample_ids": ids,
            "source_shards": ORIGINAL_NAMES,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })
        append_rows(output / "stability.jsonl", [
            row for row in source_rows if row["id"] in set(ids)
        ])

    selection = {"selected": samples}

    def resolve_artifact(_output, model, sample_id, cell_id, seed):
        stem = f"{model}_{sample_id}_{cell_id}_{seed if seed is not None else 'none'}.npz"
        for source_name in ORIGINAL_NAMES:
            candidate = ROOT / "results/attributions" / source_name / stem
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"No frozen main attribution for {stem}")

    m4_matrix.artifact_path = resolve_artifact
    config = {**config, "device": "cuda"}
    configure_runtime(config["selection_seed"], config["threads"])
    print(f"GPU {gpu}: resuming {len(samples)} images", flush=True)
    m4_matrix.run_stability(config, output, metadata, selection, EXPERIMENT)


def merge_results() -> None:
    config, selection, original_dirs, _ = load_sources()
    original_map = {}
    for directory in original_dirs:
        shard = read_json(directory / "selected_shard.json")
        for sample in shard["selected"]:
            original_map[sample["id"]] = directory

    aux_dirs = sorted((ROOT / "results/metrics").glob(
        f"m4_{DOMAIN}_exp2_stability_aux_gpu*"
    ))
    merged = 0
    existing_by_dir = {
        directory: {m4_matrix.row_key(row) for row in read_rows(directory / "stability.jsonl")}
        for directory in original_dirs
    }
    for aux in aux_dirs:
        if not (aux / "complete_stability.json").is_file():
            raise RuntimeError(f"Auxiliary worker is incomplete: {aux}")
        for row in read_rows(aux / "stability.jsonl"):
            target = original_map[row["id"]]
            key = m4_matrix.row_key(row)
            if key not in existing_by_dir[target]:
                append_rows(target / "stability.jsonl", [row])
                existing_by_dir[target].add(key)
                merged += 1

    for directory in original_dirs:
        shard = read_json(directory / "selected_shard.json")
        expected = m4_matrix.expected_keys(config, shard["selected"], EXPERIMENT)
        rows = read_rows(directory / "stability.jsonl")
        actual = [m4_matrix.row_key(row) for row in rows]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise RuntimeError(f"Merged shard incomplete/duplicate: {directory}: {len(actual)}/{len(expected)}")
        write_json(directory / "complete_stability.json", {
            "rows": len(actual), "expected": len(expected),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "auxiliary_workers": [path.name for path in aux_dirs],
        })
    print(f"Merged {merged} new stability rows into the two frozen shards", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="2,3,4,6")
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        merge_results()
        return
    gpus = [int(value) for value in args.gpus.split(",")]
    config, selection, original_dirs, rows = load_sources()
    groups = assign_workers(config, selection, rows, gpus)
    if args.worker_index is None:
        raise SystemExit("--worker-index is required")
    if not 0 <= args.worker_index < len(gpus):
        raise SystemExit("invalid worker index")
    run_worker(gpus[args.worker_index], groups[args.worker_index], config,
               original_dirs, rows)


if __name__ == "__main__":
    main()
