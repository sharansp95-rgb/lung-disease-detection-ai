"""
Training loop: frozen-backbone warmup epochs, then full fine-tuning,
weighted CrossEntropyLoss for class imbalance, best-val-accuracy checkpoint,
training-curves plot.

    python train.py --data_dir ../data/xray --backbone densenet121 --epochs 15
"""

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import build_dataloaders, compute_class_weights
from model import build_model, SUPPORTED_BACKBONES

ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_one_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    grad_context = torch.enable_grad() if train else torch.no_grad()
    with grad_context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def plot_curves(history: dict, out_path: Path, title: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"{title} — loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title(f"{title} — accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run(args) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[train] device={device}")

    train_loader, val_loader, _, class_names = build_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    num_classes = args.num_classes or len(class_names)
    print(f"[train] classes ({num_classes}): {class_names}")

    class_weights, class_counts = compute_class_weights(train_loader.dataset)
    print("[train] training-set class counts:")
    for name, cnt in zip(class_names, class_counts.tolist()):
        print(f"  {name}: {cnt}")
    print("[train] computed inverse-frequency class weights (for weighted CrossEntropyLoss):")
    for name, w in zip(class_names, class_weights.tolist()):
        print(f"  {name}: {w:.4f}")

    model = build_model(args.backbone, num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_epoch = -1

    run_name = args.run_name or f"{Path(args.data_dir).name}_{args.backbone}"
    ckpt_dir = ROOT / args.output_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{run_name}_best.pt"

    total_epochs = args.epochs
    warmup_epochs = min(args.warmup_epochs, total_epochs)

    for epoch in range(1, total_epochs + 1):
        phase_is_warmup = epoch <= warmup_epochs
        if epoch == 1 and phase_is_warmup:
            model.freeze_backbone()
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
            print(f"[train] epoch {epoch}: backbone FROZEN (warmup), lr={args.lr}")
        elif epoch == warmup_epochs + 1:
            model.unfreeze_backbone()
            optimizer = optim.Adam(model.parameters(), lr=args.finetune_lr)
            print(f"[train] epoch {epoch}: backbone UNFROZEN (fine-tune), lr={args.finetune_lr}")

        t0 = time.time()
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False)
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        phase = "warmup" if phase_is_warmup else "finetune"
        print(f"[train] epoch {epoch}/{total_epochs} ({phase}, {dt:.1f}s) "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "backbone": args.backbone,
                "num_classes": num_classes,
                "class_names": class_names,
                "val_acc": val_acc,
                "epoch": epoch,
            }, ckpt_path)

    print(f"[train] best val_acc={best_val_acc:.4f} at epoch {best_epoch} -> saved to {ckpt_path}")

    curves_path = log_dir / f"{run_name}_curves.png"
    plot_curves(history, curves_path, title=run_name)
    print(f"[train] curves saved to {curves_path}")

    history_path = log_dir / f"{run_name}_history.json"
    with open(history_path, "w") as f:
        json.dump({
            "run_name": run_name,
            "backbone": args.backbone,
            "data_dir": str(args.data_dir),
            "class_names": class_names,
            "class_counts": class_counts.tolist(),
            "class_weights": class_weights.tolist(),
            "epochs": total_epochs,
            "warmup_epochs": warmup_epochs,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "history": history,
        }, f, indent=2)
    print(f"[train] history saved to {history_path}")

    return {"best_val_acc": best_val_acc, "best_epoch": best_epoch, "history": history,
            "class_names": class_names, "class_weights": class_weights.tolist(),
            "checkpoint": str(ckpt_path)}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="folder with train/val/test subfolders")
    p.add_argument("--backbone", default="densenet121", choices=SUPPORTED_BACKBONES)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="lr during frozen-backbone warmup")
    p.add_argument("--finetune_lr", type=float, default=1e-4, help="lr after unfreezing")
    p.add_argument("--num_classes", type=int, default=None, help="inferred from data_dir if omitted")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda / cpu, autodetected if omitted")
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--log_dir", default="ml_logs")
    p.add_argument("--run_name", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
