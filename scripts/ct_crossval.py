"""
5-fold stratified cross-validation on the CT scan dataset (IQ-OTH/NCCD).
Runs on GPU (CUDA) with mixed precision (torch.cuda.amp) when available.

Why CV instead of the single split:
  Patient leakage cannot be fixed without patient IDs (see prepare_ct_scan.py).
  5-fold CV doesn't eliminate leakage risk, but it shows whether the single-split
  99-100% is a stable pattern across different data partitions or one lucky draw.

Output:
  eval_outputs/ct_crossval_results.json   -- per-fold metrics + mean/std
  ml_logs/ct_crossval_curves.png          -- accuracy curves across folds

Usage:
    python scripts/ct_crossval.py
  or from project root:
    python -m scripts.ct_crossval    (if __init__.py exists, else use cd ml)

  Add --epochs N to override default (10 epochs per fold is enough for stability check).
  Add --backbone resnet18 for a faster run.

Estimated wall-clock (GPU RTX3050, densenet121, AMP, 10 epochs, 5 folds): ~8-15 min.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ── path setup: allow "python scripts/ct_crossval.py" from project root ──────
ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "ml"
sys.path.insert(0, str(ML_DIR))

from model import build_model, SUPPORTED_BACKBONES  # noqa: E402 (after sys.path)

CT_SRC = ROOT / "raw" / "ct_scan" / "The IQ-OTHNCCD lung cancer dataset"

SOURCE_DIRS = [
    ("Bengin cases",    "Benign"),
    ("Malignant cases", "Malignant"),
    ("Normal cases",    "Normal"),
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224
IMAGE_EXTS    = {".png", ".jpg", ".jpeg"}


# ── reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── transforms ────────────────────────────────────────────────────────────────

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


# ── dataset loading (from raw/, not data/ — CV uses its own splits) ───────────

class CTRawDataset(torch.utils.data.Dataset):
    """
    Loads directly from raw/ct_scan/ without assuming train/val/test structure.
    Returns (image_tensor, class_idx), class list is sorted alphabetically.
    Transform is applied dynamically so train vs val augmentation works per-fold.
    """
    def __init__(self, samples: list[tuple[Path, int]], transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path)
        return self.transform(img), label


def load_all_samples() -> tuple[list[tuple[Path, int]], list[str]]:
    """
    Collect (path, label_idx) for all CT images from the raw source.
    Classes sorted alphabetically (Benign=0, Malignant=1, Normal=2).
    """
    classes = sorted([c for _, c in SOURCE_DIRS])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    samples = []
    for dir_name, cls in SOURCE_DIRS:
        src_dir = CT_SRC / dir_name
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Missing CT source dir: {src_dir}")
        for p in sorted(src_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                samples.append((p, class_to_idx[cls]))
    return samples, classes


# ── one-epoch training/eval ───────────────────────────────────────────────────

def run_one_epoch(model, loader, criterion, optimizer, device, train: bool,
                  scaler: GradScaler | None = None):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = (scaler is not None) and (device.type == "cuda")

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            if use_amp:
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                if train:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                if train:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += images.size(0)

    return total_loss / total, correct / total


# ── single fold training ──────────────────────────────────────────────────────

def train_fold(
    fold_idx: int,
    train_indices: list[int],
    val_indices:   list[int],
    all_samples:   list[tuple[Path, int]],
    classes:       list[str],
    args,
    device: torch.device,
) -> dict:
    """Train one fold, return dict with val accuracy history and best val acc."""
    train_ds = CTRawDataset(
        [all_samples[i] for i in train_indices],
        build_transforms(train=True),
    )
    val_ds = CTRawDataset(
        [all_samples[i] for i in val_indices],
        build_transforms(train=False),
    )

    nw = 2 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=nw, pin_memory=(nw > 0))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=nw, pin_memory=(nw > 0))

    # Inverse-frequency class weights from this fold's training set
    num_classes = len(classes)
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, lbl in train_ds.samples:
        counts[lbl] += 1
    total    = counts.sum().item()
    weights  = (total / (num_classes * counts.float())).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model     = build_model(args.backbone, num_classes, pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler    = GradScaler('cuda') if device.type == "cuda" else None

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True,  scaler=scaler)
        va_loss, va_acc = run_one_epoch(model, val_loader,   criterion, optimizer, device, train=False, scaler=scaler)
        dt = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(f"  [fold {fold_idx+1} epoch {epoch}/{args.epochs}] "
              f"({dt:.1f}s) tr_acc={tr_acc:.4f}  val_acc={va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc

    print(f"  [fold {fold_idx+1}] best val_acc = {best_val_acc:.4f}")
    return {"fold": fold_idx + 1, "best_val_acc": best_val_acc, "history": history}


# ── stratified k-fold splits (no sklearn dependency) ─────────────────────────

def stratified_kfold_indices(
    labels: list[int], n_splits: int, seed: int
) -> list[tuple[list[int], list[int]]]:
    """
    Manual stratified k-fold: split each class separately, then interleave.
    Returns list of (train_indices, val_indices) pairs.
    """
    rng = random.Random(seed)
    class_indices: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels):
        class_indices.setdefault(lbl, []).append(i)
    for lbl in class_indices:
        rng.shuffle(class_indices[lbl])

    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for lbl, idxs in class_indices.items():
        for i, idx in enumerate(idxs):
            folds[i % n_splits].append(idx)

    splits = []
    for k in range(n_splits):
        val   = folds[k]
        train = [idx for j in range(n_splits) if j != k for idx in folds[j]]
        splits.append((train, val))
    return splits


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_fold_curves(fold_results: list[dict], out_path: Path):
    n_folds = len(fold_results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for res in fold_results:
        h = res["history"]
        ep = range(1, len(h["val_acc"]) + 1)
        axes[0].plot(ep, h["train_acc"], alpha=0.5, label=f"fold {res['fold']} train")
        axes[0].plot(ep, h["val_acc"],   alpha=0.9, linestyle="--", label=f"fold {res['fold']} val")
        axes[1].plot(ep, h["val_acc"],   label=f"fold {res['fold']} (best={res['best_val_acc']:.3f})")

    axes[0].set_title("CT 5-Fold CV — train vs val accuracy")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=7)

    axes[1].set_title("CT 5-Fold CV — val accuracy per fold")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val accuracy")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Curves saved to {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="5-fold CV on CT scan dataset")
    p.add_argument("--backbone", default="densenet121", choices=SUPPORTED_BACKBONES)
    p.add_argument("--epochs",   type=int, default=10,
                   help="epochs per fold (10 is sufficient for stability check)")
    p.add_argument("--batch_size", type=int, default=16,
                   help="smaller batch for CPU (16 recommended)")
    p.add_argument("--lr",       type=float, default=1e-4)
    p.add_argument("--n_splits", type=int,   default=5)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--device",   default=None,
                   help="'cpu' or 'cuda'. Auto-detected if omitted.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"[ct_crossval] torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[ct_crossval] torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    print(f"[ct_crossval] device={device}  amp={'ON' if device.type == 'cuda' else 'OFF'}")
    print(f"[ct_crossval] backbone={args.backbone}, epochs={args.epochs}, "
          f"batch_size={args.batch_size}, n_splits={args.n_splits}, seed={args.seed}")

    all_samples, classes = load_all_samples()
    labels = [lbl for _, lbl in all_samples]
    print(f"[ct_crossval] total images={len(all_samples)}, classes={classes}")
    from collections import Counter
    counts = Counter(labels)
    for i, c in enumerate(classes):
        print(f"  {c}: {counts[i]}")

    splits = stratified_kfold_indices(labels, n_splits=args.n_splits, seed=args.seed)

    fold_results = []
    t_total = time.time()
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx+1}/{args.n_splits}  "
              f"(train={len(train_idx)}, val={len(val_idx)})")
        print(f"{'='*60}")
        result = train_fold(
            fold_idx, train_idx, val_idx,
            all_samples, classes, args, device
        )
        fold_results.append(result)

    elapsed = time.time() - t_total
    best_accs = [r["best_val_acc"] for r in fold_results]
    mean_acc  = float(np.mean(best_accs))
    std_acc   = float(np.std(best_accs))

    print(f"\n{'='*60}")
    print(f"CT 5-Fold Cross-Validation Results")
    print(f"{'='*60}")
    for r in fold_results:
        print(f"  Fold {r['fold']}: best_val_acc = {r['best_val_acc']:.4f}")
    print(f"  Mean ± Std = {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Total elapsed: {elapsed/60:.1f} min")

    # Save results
    out_dir = ROOT / "eval_outputs"
    out_dir.mkdir(exist_ok=True)
    log_dir = ROOT / "ml_logs"
    log_dir.mkdir(exist_ok=True)

    results = {
        "backbone":     args.backbone,
        "n_splits":     args.n_splits,
        "epochs":       args.epochs,
        "seed":         args.seed,
        "classes":      classes,
        "dataset_size": len(all_samples),
        "fold_results": [
            {"fold": r["fold"], "best_val_acc": r["best_val_acc"]}
            for r in fold_results
        ],
        "mean_val_acc": mean_acc,
        "std_val_acc":  std_acc,
        "elapsed_min":  round(elapsed / 60, 1),
    }

    results_path = out_dir / "ct_crossval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    curves_path = log_dir / "ct_crossval_curves.png"
    plot_fold_curves(fold_results, curves_path)

    return results


if __name__ == "__main__":
    main()
