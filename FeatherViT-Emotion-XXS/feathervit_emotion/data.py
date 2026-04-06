from __future__ import annotations

from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transforms(img_size: int = 256) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25),
        ]
    )


def build_eval_transforms(img_size: int = 256) -> transforms.Compose:
    resize_size = int(round(img_size * 1.125))
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def create_imagefolder_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int,
    num_workers: int,
    img_size: int,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    train_ds = datasets.ImageFolder(train_dir, transform=build_train_transforms(img_size))
    val_ds = datasets.ImageFolder(val_dir, transform=build_eval_transforms(img_size))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, train_ds.class_to_idx
