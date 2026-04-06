from __future__ import annotations

import argparse

from .model import build_feathervit_emotion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Count FeatherViT-Emotion-XXS parameters")
    parser.add_argument("--num-classes", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_feathervit_emotion(num_classes=args.num_classes, dropout=0.0)
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params:,} ({params / 1e6:.3f}M)")


if __name__ == "__main__":
    main()
