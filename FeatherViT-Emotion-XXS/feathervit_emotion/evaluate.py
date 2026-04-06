from __future__ import annotations

import argparse
from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from .data import build_eval_transforms
from .model import build_feathervit_emotion
from .utils import load_checkpoint, mean, resolve_device, topk_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate FeatherViT-Emotion-XXS")
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    use_amp = args.amp and device.type == "cuda"
    if args.amp and device.type != "cuda":
        print("AMP requested but only CUDA AMP is enabled in this project. Continuing without AMP.")
    print(f"Using device: {device}")
    amp_ctx = torch.cuda.amp.autocast if use_amp else nullcontext

    ds = datasets.ImageFolder(args.val_dir, transform=build_eval_transforms(args.img_size))
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_feathervit_emotion(num_classes=len(ds.classes))
    payload = load_checkpoint(args.checkpoint, model, map_location="cpu")
    model.to(device).eval()

    criterion = nn.CrossEntropyLoss()
    losses = []
    top1s = []
    top5s = []
    eval_k = 5
    for images, targets in tqdm(loader, desc="eval"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with amp_ctx():
            logits = model(images)
            loss = criterion(logits, targets)
        eval_k = min(5, logits.shape[1])
        t1, t5 = topk_accuracy(logits, targets, topk=(1, eval_k))
        losses.append(loss.item())
        top1s.append(t1)
        top5s.append(t5)

    print(f"Checkpoint epoch: {payload.get('epoch', 'n/a')}")
    print(f"Loss: {mean(losses):.4f}")
    print(f"Top1: {mean(top1s):.2f}")
    print(f"Top{eval_k}: {mean(top5s):.2f}")


if __name__ == "__main__":
    main()
