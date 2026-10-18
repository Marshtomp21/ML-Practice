"""Validate and preview the RISE configuration; never run image attribution."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import yaml

from attribution.perturbation.rise import RiseConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/method/rise.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        required=True,
        help="Only validate configuration and print per-image budget",
    )
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        parser.error("config must contain a mapping of RiseConfig fields")
    try:
        config = RiseConfig(**payload)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "method": "RISE",
                "config": asdict(config),
                "keep_probability": config.keep_probability,
                "forward_samples_per_image": config.num_masks,
                "forward_batches_per_image": (config.num_masks + config.batch_size - 1)
                // config.batch_size,
                "value_function": "target class pre-Softmax logit",
                "baseline": "zero in normalized input space",
                "mask_interpolation": "bilinear",
                "output": "signed pixel saliency map",
                "executes_attribution": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
