"""Validate and preview KernelSHAP configuration; never run attribution."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import yaml

from attribution.game_theoretic.kernel_shap import KernelShapConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/method/kernel_shap.yaml",
    )
    parser.add_argument("--num-regions", type=int, default=100)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        required=True,
        help="Only validate configuration and print per-image budget",
    )
    args = parser.parse_args()
    if args.num_regions < 1:
        parser.error("num-regions must be positive")
    with args.config.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        parser.error("config must contain a mapping of KernelShapConfig fields")
    try:
        config = KernelShapConfig(**payload)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    exact = args.num_regions <= config.num_samples.bit_length() - 1
    forward_samples = (1 << args.num_regions) if exact else config.num_samples
    print(
        json.dumps(
            {
                "method": "KernelSHAP",
                "config": asdict(config),
                "actual_regions": args.num_regions,
                "sampling_mode": (
                    "exact"
                    if exact
                    else ("paired_kernel" if config.paired_sampling else "kernel")
                ),
                "forward_samples_per_image": forward_samples,
                "forward_batches_per_image": (forward_samples + config.batch_size - 1)
                // config.batch_size,
                "anchor_samples": 2,
                "value_function": "target class pre-Softmax logit",
                "baseline": "zero in normalized input space",
                "efficiency_constraint": "sum(phi)=v(full)-v(empty)",
                "executes_attribution": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
