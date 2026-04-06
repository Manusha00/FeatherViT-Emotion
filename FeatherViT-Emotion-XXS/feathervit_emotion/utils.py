from __future__ import annotations

import json
import os
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str = "auto") -> torch.device:
    device_arg = device_arg.lower()
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {device_arg}")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def topk_accuracy(logits: Tensor, targets: Tensor, topk: Tuple[int, ...] = (1, 5)) -> List[float]:
    num_classes = logits.shape[1]
    maxk = min(max(topk), num_classes)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.reshape(1, -1).expand_as(pred))
    res: List[float] = []
    batch_size = targets.size(0)
    for k in topk:
        k_eff = min(k, num_classes)
        correct_k = correct[:k_eff].reshape(-1).float().sum(0)
        res.append((correct_k * (100.0 / batch_size)).item())
    return res


def count_parameters_millions(model: torch.nn.Module) -> float:
    n = sum(p.numel() for p in model.parameters())
    return n / 1e6


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_top1: float,
    class_to_idx: Dict[str, int],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "best_top1": best_top1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "class_to_idx": class_to_idx,
    }
    torch.save(payload, path)


def load_checkpoint(path: str, model: torch.nn.Module, map_location: str = "cpu") -> Dict:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    return payload


def write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / max(1, len(values))
