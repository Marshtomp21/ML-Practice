"""Download and save the exact TorchVision ViT-B/16 ImageNet-1K V1 state dict."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.models import ViT_B_16_Weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/checkpoints/vit_b_16_imagenet1k_v1.pth"),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    state = ViT_B_16_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
    torch.save(state, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
