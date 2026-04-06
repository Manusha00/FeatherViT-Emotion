from __future__ import annotations

import argparse
import time
from contextlib import nullcontext

import torch

from .model import build_feathervit_emotion
from .utils import count_parameters_millions, load_checkpoint, resolve_device, synchronize_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Benchmark FeatherViT-Emotion-XXS")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
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

    model = build_feathervit_emotion(num_classes=args.num_classes).to(device).eval()
    if args.checkpoint:
        payload = load_checkpoint(args.checkpoint, model, map_location="cpu")
        print(f"Loaded checkpoint epoch: {payload.get('epoch', 'n/a')}")
    print(f"Params (M): {count_parameters_millions(model):.2f}")

    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
    autocast_ctx = torch.cuda.amp.autocast if use_amp else nullcontext

    for _ in range(args.warmup):
        with autocast_ctx(enabled=use_amp):
            _ = model(x)
    synchronize_device(device)

    t0 = time.perf_counter()
    for _ in range(args.iters):
        with autocast_ctx(enabled=use_amp):
            _ = model(x)
    synchronize_device(device)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    total_samples = args.iters * args.batch_size
    sps = total_samples / elapsed
    ms_per_batch = (elapsed / args.iters) * 1000.0
    print(f"Device: {device}")
    print(f"Samples/sec: {sps:.2f}")
    print(f"ms/batch: {ms_per_batch:.2f}")


if __name__ == "__main__":
    main()
