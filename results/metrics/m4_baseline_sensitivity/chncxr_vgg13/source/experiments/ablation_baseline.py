"""Validate and preview the Ablation configuration; never run image attribution."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import yaml

from attribution.perturbation.ablation import AblationConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs/method/ablation.yaml")
    parser.add_argument("--dry-run", action="store_true", required=True,
                        help="Only validate configuration and print per-image budget")
    parser.add_argument("--num-regions", type=int, required=True,
                        help="Actual region count for this preview, not nominal SLIC n_segments")
    parser.add_argument("--cached-original", action="store_true",
                        help="Preview reuse of a public cached original logit")
    args = parser.parse_args()
    if args.num_regions < 1:
        parser.error("num-regions must be >= 1")
    with args.config.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        parser.error("config must contain a mapping of AblationConfig fields")
    try:
        config = AblationConfig(**payload)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    forward_samples = args.num_regions + int(not args.cached_original)
    print(json.dumps({"method": "Ablation", "config": asdict(config),
                      "actual_regions": args.num_regions,
                      "used_cached_original": args.cached_original,
                      "forward_samples_per_image": forward_samples,
                      "forward_batches_per_image":
                          (forward_samples + config.batch_size - 1) // config.batch_size,
                      "value_function": "target class pre-Softmax logit",
                      "baseline": "zero in normalized input space",
                      "executes_attribution": False}, indent=2))


if __name__ == "__main__":
    main()
