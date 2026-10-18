"""Run the M4 E3 masking-baseline sensitivity study on frozen config samples."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageFilter
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.perturbation import insertion_deletion, localization_metrics
from experiments.m3_default import explain, method_config
from experiments.m4_matrix import load_config as load_m4_config
from experiments.perturbation_pilot import (
    ROOT as EXPERIMENT_ROOT,
    domain_api,
    freeze_run,
    read_rows,
    write_json,
)
from models.inference import configure_runtime
from preprocessing.localization import load_localization_mask
from preprocessing.superpixels import slic_segments

BASELINE_CONFIG_PATH = ROOT / "configs/experiment/m4_baseline_sensitivity.yaml"
M4_CONFIG_PATH = ROOT / "configs/experiment/m4.yaml"
OUTPUT_ROOT = ROOT / "results/metrics/m4_baseline_sensitivity"
ATTR_ROOT = ROOT / "results/attributions/m4_baseline_sensitivity"
PAIRINGS = [(domain, model) for domain in ("imagenet", "chncxr")
            for model in ("resnet50", "vgg13")]
RANDOM_METHODS = {"lime", "rise", "kernel_shap"}


def append_row(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()


def normalize_rgb(rgb: np.ndarray, preprocessing: dict) -> np.ndarray:
    mean = np.asarray(preprocessing["mean"], dtype=np.float32)
    std = np.asarray(preprocessing["std"], dtype=np.float32)
    return (np.asarray(rgb, dtype=np.float32) - mean) / std


def build_dataset_references(selected: list[dict], load_image,
                             preprocessing: dict) -> tuple[np.ndarray, np.ndarray,
                                                            np.ndarray, np.ndarray]:
    images = [load_image(ROOT / row["path"])[0] for row in selected]
    stacked = np.stack(images).astype(np.float32, copy=False)
    mean_image = stacked.mean(axis=0, dtype=np.float64).astype(np.float32)
    mean_color = stacked.mean(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    pixel_std = stacked.std(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    return (normalize_rgb(mean_image, preprocessing),
            normalize_rgb(np.broadcast_to(mean_color, mean_image.shape), preprocessing),
            mean_color, pixel_std)


def sample_references(rgb: np.ndarray, sample_id: str, shared: tuple,
                      preprocessing: dict, selection_seed: int,
                      blur_sigma: float) -> dict[str, np.ndarray]:
    mean_image, mean_color, color_mean, color_std = shared
    blur = np.asarray(
        Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))
        .filter(ImageFilter.GaussianBlur(radius=blur_sigma)), dtype=np.float32
    ) / 255.0
    digest = hashlib.sha256(
        f"m4-baseline-noise-v1:{selection_seed}:{sample_id}".encode()
    ).hexdigest()
    rng = np.random.default_rng(int(digest[:16], 16) % (2**32))
    noise = np.clip(rng.normal(color_mean, color_std, size=rgb.shape), 0.0, 1.0)
    return {
        "normalized_zero": np.zeros_like(normalize_rgb(rgb, preprocessing)),
        "dataset_mean_image": mean_image,
        "dataset_mean_color": mean_color,
        "gaussian_blur_sigma10": normalize_rgb(blur, preprocessing),
        "gaussian_noise_dataset_stats": normalize_rgb(noise, preprocessing),
    }


def make_config(domain: str, model: str) -> tuple[dict, dict, dict]:
    baseline_cfg = yaml.safe_load(BASELINE_CONFIG_PATH.read_text(encoding="utf-8"))
    config, selection = load_m4_config(
        M4_CONFIG_PATH, domain, "exp1", baseline_cfg["split_name"]
    )
    model_path = config["models"][model]
    config["models"] = {model: model_path}
    config.update({
        "stage": baseline_cfg["stage"],
        "num_samples": baseline_cfg["num_samples"],
        "n_segments": baseline_cfg["n_segments"],
        "mask_rate": baseline_cfg["mask_rate"],
        "grid_size": baseline_cfg["grid_size"],
        "baseline_sensitivity": baseline_cfg,
        "device": "cuda",
    })
    return config, selection, baseline_cfg


def expected_keys(selected: list[dict], baseline_cfg: dict) -> set[tuple]:
    keys = set()
    for sample in selected:
        for baseline in baseline_cfg["baselines"]:
            for method in baseline_cfg["methods"]:
                seeds = baseline_cfg["seeds"] if method in RANDOM_METHODS else [None]
                for seed in seeds:
                    keys.add((sample["id"], baseline, method, seed))
    return keys


def run_worker(worker_index: int, sample_limit: int | None = None) -> None:
    domain, model = PAIRINGS[worker_index]
    config, selection_payload, baseline_cfg = make_config(domain, model)
    selected = selection_payload["selected"]
    output = OUTPUT_ROOT / f"{domain}_{model}"
    metadata = freeze_run(config, output, output.exists())
    selection_path = output / "selection.json"
    if selection_path.exists():
        saved = json.loads(selection_path.read_text(encoding="utf-8"))
        if saved["ids"] != [row["id"] for row in selected]:
            raise RuntimeError(f"Frozen selection changed for {output}")
    else:
        write_json(selection_path, {
            "split_file": config["split_file"], "split_name": config["split_name"],
            "split_fingerprint": config["split_fingerprint"],
            "ids": [row["id"] for row in selected],
        })

    domain_name, preprocessing, _, load_image, load_model = domain_api(config)
    shared = build_dataset_references(selected, load_image, preprocessing)
    predictor = load_model(model, ROOT / config["models"][model], config["device"])
    warm = load_image(ROOT / selected[0]["path"])[1]
    for _ in range(config["warmup_batches"]):
        predictor(np.repeat(warm[None], config["batch_size"], axis=0))

    rows_path = output / "metrics.jsonl"
    existing = read_rows(rows_path)
    done = {(row["id"], row["explanation_baseline"], row["method"], row["seed"])
            for row in existing}
    if len(done) != len(existing):
        raise ValueError(f"Duplicate result keys in {rows_path}")
    artifact_dir = ATTR_ROOT / f"{domain}_{model}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    work_selected = selected if sample_limit is None else selected[:sample_limit]
    if not work_selected:
        raise ValueError("Worker selection is empty")
    for sample in work_selected:
        rgb, image = load_image(ROOT / sample["path"])
        segments = slic_segments(
            rgb, config["n_segments"], config["compactness"], config["sigma"]
        )
        references = sample_references(
            rgb, sample["id"], shared, preprocessing,
            baseline_cfg["selection_seed"], baseline_cfg["blur_sigma_rgb_pixels"],
        )
        target_mask = load_localization_mask(
            domain_name, sample["id"], ROOT / config["localization_mask_root"], rgb.shape[:2]
        )
        for baseline_name in baseline_cfg["baselines"]:
            reference = references[baseline_name]
            for method in baseline_cfg["methods"]:
                seeds = baseline_cfg["seeds"] if method in RANDOM_METHODS else [None]
                for seed in seeds:
                    key = sample["id"], baseline_name, method, seed
                    if key in done:
                        artifact = artifact_dir / (
                            f"{sample['id']}_{baseline_name}_{method}_"
                            f"{'none' if seed is None else seed}.npz"
                        )
                        if not artifact.is_file():
                            raise ValueError(f"Missing attribution artifact: {artifact}")
                        continue
                    options = method_config(config, method, seed)
                    before_samples, before_batches = predictor.samples, predictor.batches
                    predictor.synchronize()
                    started = time.perf_counter()
                    result, ids, scores, pixel_map = explain(
                        method, image, segments, predictor, sample["target"], options,
                        baseline=reference,
                    )
                    predictor.synchronize()
                    attribution_seconds = time.perf_counter() - started
                    forward_samples = predictor.samples - before_samples
                    forward_batches = predictor.batches - before_batches
                    if (forward_samples, forward_batches) != (
                            result.forward_samples, result.forward_batches):
                        raise AssertionError("Attribution forward-count mismatch")

                    # Keep the evaluation baseline at the M4 primary zero baseline.
                    evaluation_before = predictor.samples
                    curves = insertion_deletion(
                        image, segments, scores, predictor, sample["target"],
                        config["batch_size"], config["evaluation_steps"],
                    )
                    if predictor.samples - evaluation_before != curves["evaluation_forward_samples"]:
                        raise AssertionError("Evaluation forward-count mismatch")
                    localization = {"localization_energy": None, "pointing_game": None,
                                    "zero_positive_energy": None}
                    if target_mask is not None:
                        localization = localization_metrics(pixel_map, target_mask)

                    artifact = artifact_dir / (
                        f"{sample['id']}_{baseline_name}_{method}_"
                        f"{'none' if seed is None else seed}.npz"
                    )
                    np.savez_compressed(artifact, segments=segments, region_ids=ids,
                                        region_scores=scores, saliency=pixel_map)
                    diagnostics = {name: value for name, value in asdict(result).items()
                                   if not isinstance(value, np.ndarray)}
                    row = {
                        "domain": domain, "split": config["split_name"], "model": model,
                        "id": sample["id"], "target": sample["target"],
                        "method": method, "explanation_baseline": baseline_name,
                        "evaluation_baseline": baseline_cfg["evaluation_baseline"],
                        "seed": seed, "config": asdict(options),
                        "actual_regions": len(ids),
                        "attribution_seconds": attribution_seconds,
                        "forward_samples": forward_samples,
                        "forward_batches": forward_batches,
                        "diagnostics": diagnostics, **curves, **localization,
                        "run_fingerprint": metadata["fingerprint"],
                        "artifact": str(artifact.relative_to(ROOT)),
                    }
                    append_row(rows_path, row)
                    done.add(key)
                    print(f"{len(done)} {domain} {model} {sample['id']} "
                          f"{baseline_name} {method} seed={seed}", flush=True)

    expected = expected_keys(work_selected, baseline_cfg)
    actual = {(row["id"], row["explanation_baseline"], row["method"], row["seed"])
              for row in read_rows(rows_path)}
    if actual != expected:
        raise ValueError(f"Incomplete worker output {len(actual)}/{len(expected)}")
    write_json(output / "complete_main.json", {
        "domain": domain, "model": model, "rows": len(actual), "expected": len(expected),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(f"Completed {domain}/{model}: {len(actual)} rows", flush=True)


def quantiles(values: list[float]) -> dict:
    data = np.asarray(values, dtype=np.float64)
    return {"count": int(len(data)), "median": float(np.median(data)),
            "q25": float(np.percentile(data, 25)), "q75": float(np.percentile(data, 75))}


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator,
                        replicates: int) -> list[float]:
    if not len(values):
        return [float("nan"), float("nan")]
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    samples = np.median(values[indices], axis=1)
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def merge() -> None:
    baseline_cfg = yaml.safe_load(BASELINE_CONFIG_PATH.read_text(encoding="utf-8"))
    rows = []
    for domain, model in PAIRINGS:
        output = OUTPUT_ROOT / f"{domain}_{model}"
        marker = json.loads((output / "complete_main.json").read_text(encoding="utf-8"))
        worker_rows = read_rows(output / "metrics.jsonl")
        if marker["rows"] != marker["expected"] or len(worker_rows) != marker["expected"]:
            raise ValueError(f"Worker output is incomplete: {output}")
        rows.extend(worker_rows)

    all_rows = OUTPUT_ROOT / "per_run.jsonl"
    all_rows.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
                        encoding="utf-8")
    metric_names = ("insertion_auc", "deletion_auc", "normalized_insertion_auc",
                    "normalized_deletion_auc", "localization_energy", "pointing_game",
                    "attribution_seconds", "forward_samples")
    grouped = {}
    for row in rows:
        key = (row["domain"], row["model"], row["method"],
               row["explanation_baseline"], row["id"])
        grouped.setdefault(key, []).append(row)
    per_image = {}
    for key, values in grouped.items():
        if key[2] in RANDOM_METHODS and {row["seed"] for row in values} != set(baseline_cfg["seeds"]):
            raise ValueError(f"Incomplete seed set for {key}")
        per_image[key] = {
            metric: (float(np.mean([row[metric] for row in values if row[metric] is not None]))
                     if any(row[metric] is not None for row in values) else None)
            for metric in metric_names
        }

    summary, contrasts = [], []
    rng = np.random.default_rng(baseline_cfg["bootstrap_seed"])
    for domain, model in PAIRINGS:
        for method in baseline_cfg["methods"]:
            baseline_names = baseline_cfg["baselines"]
            zero_key_name = "normalized_zero"
            ids = sorted({key[4] for key in per_image
                          if key[:3] == (domain, model, method)})
            for baseline in baseline_names:
                matched = [image_id for image_id in ids
                           if (domain, model, method, baseline, image_id) in per_image]
                record = {"domain": domain, "model": model, "method": method,
                          "explanation_baseline": baseline, "evaluation_baseline": "normalized_zero",
                          "images": len(matched)}
                for metric in metric_names:
                    vals = [per_image[domain, model, method, baseline, image_id][metric]
                            for image_id in matched]
                    vals = [value for value in vals if value is not None]
                    record[metric] = quantiles(vals) if vals else None
                summary.append(record)
                if baseline == zero_key_name:
                    continue
                baseline_ids = [image_id for image_id in matched
                                if per_image[domain, model, method, zero_key_name, image_id]
                                .get("insertion_auc") is not None]
                for metric in metric_names:
                    paired = [(per_image[domain, model, method, baseline, image_id][metric],
                               per_image[domain, model, method, zero_key_name, image_id][metric])
                              for image_id in baseline_ids
                              if per_image[domain, model, method, baseline, image_id][metric] is not None
                              and per_image[domain, model, method, zero_key_name, image_id][metric] is not None]
                    if not paired:
                        continue
                    differences = np.asarray([left - right for left, right in paired], dtype=np.float64)
                    contrasts.append({
                        "domain": domain, "model": model, "method": method,
                        "baseline_vs": zero_key_name, "explanation_baseline": baseline,
                        "metric": metric, "images": len(differences),
                        "difference_direction": "alternative minus zero baseline",
                        "median_difference": float(np.median(differences)),
                        "bootstrap_95ci_median_difference": bootstrap_median_ci(
                            differences, rng, baseline_cfg["bootstrap_replicates"]
                        ),
                    })

    write_json(OUTPUT_ROOT / "summary.json", summary)
    write_json(OUTPUT_ROOT / "paired_contrasts.json", contrasts)
    write_json(OUTPUT_ROOT / "complete_main.json", {
        "rows": len(rows), "expected": len(rows), "completed_utc": datetime.now(
            timezone.utc).isoformat(), "worker_outputs": [f"{d}_{m}" for d, m in PAIRINGS],
    })
    write_protocol_report(OUTPUT_ROOT / "REPORT.md", baseline_cfg, summary, contrasts)
    print(f"Merged {len(rows)} run rows; {len(summary)} baseline summaries; "
          f"{len(contrasts)} paired contrasts")


def write_protocol_report(path: Path, config: dict, summary: list[dict],
                          contrasts: list[dict]) -> None:
    lines = [
        "# M4 实验 3：遮蔽基线敏感性",
        "",
        "## 预先固定的协议",
        "",
        "在 M4 configuration-selection 样本上，对 Ablation、LIME、RISE、KernelSHAP "
        "分别使用同一套默认配置重新生成归因。解释阶段的遮蔽基线为实验自变量；Insertion/Deletion "
        "评价仍固定使用项目主协议的 normalized-zero baseline，因此结果估计的是解释基线改变的影响。",
        "",
        f"- 样本清单：`{config['split_file']}` 的 `{config['split_name']}`；图像为配对分析单位。",
        f"- 默认参数：SLIC={config['n_segments']}、预算={config['num_samples']}、"
        f"mask rate={config['mask_rate']}、RISE grid={config['grid_size']}、随机种子={config['seeds']}。",
        "- normalized zero：模型归一化空间中的全零张量（RGB 空间对应数据预处理均值）。",
        "- dataset mean image：configuration-selection 图像逐像素 RGB 均值图。",
        "- dataset mean color：configuration-selection 图像所有像素的逐通道 RGB 均值常量图。",
        f"- blur：在 RGB [0,1] 空间对每张原图作 Gaussian blur，sigma={config['blur_sigma_rgb_pixels']} 像素。",
        f"- noise：用冻结 configuration-selection 池的逐通道 RGB 均值/标准差采样高斯噪声图，"
        f"裁剪至 [0,1]；种子由样本 ID 哈希固定。",
        "- 对随机方法跨基线复用同一组种子；保留实际归因前向次数、耗时与定位指标。",
        "",
        "## 配对结果",
        "",
        "逐基线中位数、四分位数见 `summary.json`；相对 normalized-zero 的逐图配对差值中位数和 "
        "bootstrap 95% CI 见 `paired_contrasts.json`。Bootstrap 以图像为单位，重复次数 "
        f"{config['bootstrap_replicates']}，种子 {config['bootstrap_seed']}。",
        "",
        "Insertion 越高、Deletion 越低通常越好；定位能量越高越好；Pointing Game 越高越好。",
        "这些基线结果是 M4 的补充控制分析，不改变主实验统一使用 normalized-zero 评价基线的口径。",
        "",
        "## 各解释基线的指标中位数",
        "",
        "| 域 | 模型 | 方法 | 解释基线 | 图像数 | Normalized insertion | Normalized deletion | 归因秒数 | 前向数 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        def med(metric):
            value = row.get(metric)
            return "NA" if not value else f"{value['median']:.4f}"
        lines.append(
            f"| {row['domain']} | {row['model']} | {row['method']} | "
            f"{row['explanation_baseline']} | {row['images']} | "
            f"{med('normalized_insertion_auc')} | {med('normalized_deletion_auc')} | "
            f"{med('attribution_seconds')} | {med('forward_samples')} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--sample-limit", type=int,
                        help="Process the first N frozen IDs for a resumable pilot")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        merge()
        return
    if args.worker_index is None or not 0 <= args.worker_index < len(PAIRINGS):
        raise SystemExit("Pass --worker-index 0..3 or --merge")
    run_worker(args.worker_index, args.sample_limit)


if __name__ == "__main__":
    main()
