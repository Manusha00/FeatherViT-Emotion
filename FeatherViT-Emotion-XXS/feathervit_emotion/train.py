from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from typing import Tuple

import torch
from torch import nn
from tqdm import tqdm

from .data import create_imagefolder_dataloaders
from .model import build_feathervit_emotion
from .utils import (
    count_parameters_millions,
    load_checkpoint,
    mean,
    resolve_device,
    save_checkpoint,
    set_seed,
    topk_accuracy,
    write_json,
)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    model.train()
    losses = []
    top1s = []
    amp_ctx = torch.cuda.amp.autocast if use_amp else nullcontext

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx():
            logits = model(images)
            loss = criterion(logits, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        top1 = topk_accuracy(logits.detach(), targets, topk=(1,))[0]
        losses.append(loss.item())
        top1s.append(top1)

    return mean(losses), mean(top1s)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float, float, int]:
    model.eval()
    losses = []
    top1s = []
    top5s = []
    amp_ctx = torch.cuda.amp.autocast if use_amp else nullcontext

    eval_k = 5
    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with amp_ctx():
            logits = model(images)
            loss = criterion(logits, targets)
        eval_k = min(5, logits.shape[1])
        top1, topk = topk_accuracy(logits, targets, topk=(1, eval_k))
        losses.append(loss.item())
        top1s.append(top1)
        top5s.append(topk)
    return mean(losses), mean(top1s), mean(top5s), eval_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train FeatherViT-Emotion-XXS")
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--val-every", type=int, default=1, help="Run validation every N epochs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = resolve_device(args.device)
    if args.val_every < 1:
        raise ValueError("--val-every must be >= 1")
    use_amp = args.amp and device.type == "cuda"
    if args.amp and device.type != "cuda":
        print("AMP requested but only CUDA AMP is enabled in this project. Continuing without AMP.")
    print(f"Using device: {device}")

    train_loader, val_loader, class_to_idx = create_imagefolder_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        pin_memory=(device.type == "cuda"),
    )
    inferred_num_classes = len(class_to_idx)
    if args.num_classes > 0 and args.num_classes != inferred_num_classes:
        raise ValueError(
            f"--num-classes ({args.num_classes}) does not match dataset classes ({inferred_num_classes})."
        )
    num_classes = inferred_num_classes

    model = build_feathervit_emotion(num_classes=num_classes, dropout=args.dropout).to(device)
    print(f"Model params: {count_parameters_millions(model):.2f}M")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best_top1 = 0.0
    if args.resume:
        payload = load_checkpoint(args.resume, model, map_location="cpu")
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if "scheduler" in payload:
            scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        best_top1 = float(payload.get("best_top1", 0.0))
        if "class_to_idx" in payload and payload["class_to_idx"] != class_to_idx:
            print("Warning: class_to_idx in checkpoint differs from dataset mapping.")

    write_json(
        os.path.join(args.output_dir, "run_config.json"),
        {
            "train_dir": args.train_dir,
            "val_dir": args.val_dir,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "img_size": args.img_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "num_classes": num_classes,
            "seed": args.seed,
            "amp": use_amp,
            "device": str(device),
            "val_every": args.val_every,
            "class_to_idx": class_to_idx,
        },
    )

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, train_top1 = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )
        scheduler.step()

        should_validate = ((epoch + 1) % args.val_every == 0) or ((epoch + 1) == args.epochs)
        if should_validate:
            val_loss, val_top1, val_top5, eval_k = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
            )
            print(
                f"train_loss={train_loss:.4f} train_top1={train_top1:.2f} "
                f"val_loss={val_loss:.4f} val_top1={val_top1:.2f} val_top{eval_k}={val_top5:.2f}"
            )
        else:
            val_top1 = None
            print(
                f"train_loss={train_loss:.4f} train_top1={train_top1:.2f} "
                f"(validation skipped; next at epoch multiple of {args.val_every})"
            )

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                path=os.path.join(args.output_dir, "last.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_top1=best_top1,
                class_to_idx=class_to_idx,
            )
        if val_top1 is not None and val_top1 > best_top1:
            best_top1 = val_top1
            save_checkpoint(
                path=os.path.join(args.output_dir, "best.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_top1=best_top1,
                class_to_idx=class_to_idx,
            )
            print(f"New best checkpoint saved (top1={best_top1:.2f})")

    print(f"\nTraining complete. Best val top1: {best_top1:.2f}")


if __name__ == "__main__":
    main()
