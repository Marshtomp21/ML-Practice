"""Validate and preview the LIME configuration; never run image attribution."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import yaml

from attribution.perturbation.lime import LimeConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs/method/lime.yaml")
    parser.add_argument("--dry-run", action="store_true", required=True,
                        help="Only validate configuration and print per-image budget")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        parser.error("config must contain a mapping of LimeConfig fields")
    try:
        config = LimeConfig(**payload)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"method": "LIME", "config": asdict(config),
                      "forward_samples_per_image": config.num_samples,
                      "forward_batches_per_image":
                          (config.num_samples + config.batch_size - 1) // config.batch_size,
                      "value_function": "target class pre-Softmax logit",
                      "baseline": "zero in normalized input space",
                      "feature_selection": "none", "executes_attribution": False}, indent=2))


if __name__ == "__main__":
    main()
