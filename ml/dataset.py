"""
ImageFolder-based loading from --data_dir with train/val/test subfolders.
Grayscale X-ray/CT images are converted to 3-channel since the pretrained
backbone expects RGB. Light augmentation on train only; val/test just get
resize + normalize.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def build_transforms(train: bool) -> transforms.Compose:
    ops = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def build_datasets(data_dir: str):
    data_dir = Path(data_dir)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=build_transforms(train=True))
    val_ds = datasets.ImageFolder(data_dir / "val", transform=build_transforms(train=False))
    test_ds = datasets.ImageFolder(data_dir / "test", transform=build_transforms(train=False))

    # ImageFolder assigns class indices by sorted folder name independently per
    # split; if a class is missing from one split, indices would silently
    # misalign. Fail loudly instead.
    if not (train_ds.classes == val_ds.classes == test_ds.classes):
        raise ValueError(
            f"Class mismatch across splits: train={train_ds.classes} "
            f"val={val_ds.classes} test={test_ds.classes}"
        )

    return train_ds, val_ds, test_ds


def build_dataloaders(data_dir: str, batch_size: int, num_workers: int = 2):
    train_ds, val_ds, test_ds = build_datasets(data_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.classes


def compute_class_weights(train_dataset: datasets.ImageFolder) -> torch.Tensor:
    """Inverse-frequency class weights, indexed to match ImageFolder's
    class_to_idx ordering, for use in a weighted CrossEntropyLoss."""
    num_classes = len(train_dataset.classes)
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, label in train_dataset.samples:
        counts[label] += 1

    if (counts == 0).any():
        missing = [train_dataset.classes[i] for i in range(num_classes) if counts[i] == 0]
        raise ValueError(f"Classes with zero training samples: {missing}")

    total = counts.sum().item()
    weights = total / (num_classes * counts.float())
    return weights, counts
