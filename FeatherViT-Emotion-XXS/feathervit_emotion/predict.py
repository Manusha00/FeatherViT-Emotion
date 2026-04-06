from __future__ import annotations

import argparse

import torch
from PIL import Image

from .data import build_eval_transforms
from .model import build_feathervit_emotion
from .utils import load_checkpoint, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Predict with FeatherViT-Emotion-XXS")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    payload = torch.load(args.checkpoint, map_location="cpu")
    class_to_idx = payload.get("class_to_idx", None)
    num_classes = len(class_to_idx) if class_to_idx else 1000

    model = build_feathervit_emotion(num_classes=num_classes)
    payload = load_checkpoint(args.checkpoint, model, map_location="cpu")
    idx_to_class = {idx: name for name, idx in class_to_idx.items()} if class_to_idx else None

    model.to(device).eval()
    transform = build_eval_transforms(args.img_size)
    image = Image.open(args.image).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    k = min(args.topk, probs.shape[0])
    values, indices = torch.topk(probs, k=k)

    for rank, (score, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        label = idx_to_class[idx] if idx_to_class else str(idx)
        print(f"{rank}. {label}: {score:.4f}")


if __name__ == "__main__":
    main()
