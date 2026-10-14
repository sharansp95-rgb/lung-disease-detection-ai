"""
Evaluates the new RadImageNet-backbone CT checkpoint
(checkpoints/ct_radimagenet_densenet121_best.pt, once uploaded) against the
existing data/ct_scan/test split, WITHOUT touching the active production
checkpoint (checkpoints/ct_best_model_phase35_v2.pt) or backend/main.py.

Mirrors scripts/test_new_xray_checkpoint.py exactly (same two parts, same
box geometry for the occlusion test), for a directly comparable format:

  1. Standard test-set metrics -- accuracy, confusion matrix, per-class
     precision/recall/f1.

  2. Center-blank occlusion test: blank the central region of every test
     image and re-measure per-class accuracy. Uses the SAME central-box
     geometry (MARGIN_FRAC=0.22, applied at 224x224) as the X-ray test, so
     results are comparable across modalities, not just old-vs-new within
     one modality. Note this project's CT confound (scanner-table
     visibility, see data/ct_scan/'s protocol-mismatch exclusion) is a
     different shortcut signature than X-ray's border/acquisition-style
     confound -- the table sits at the BOTTOM of the frame, not scattered
     across the border generally, so a center-blank test here is a
     complementary probe, not a direct reuse of a table-specific audit.

IMPORTANT: this new checkpoint uses RadImageNet's own preprocessing
(3-channel, flat (x-127.5)*2/255 normalization, plain resize to 224x224) --
NOT ml/dataset.py's ImageNet-normalized pipeline that the old
ct_best_model_phase35_v2.pt uses. The transform and model-loading code here
are imported directly from ml/train_ct_monai.py (the already-smoke-tested
source of truth), not reimplemented.

Occlusion blanking is applied as 0.0 in RadImageNet's OWN normalized space
(0.0 there corresponds to raw pixel value 127.5 -- that space's own neutral
value), the same convention used in test_new_xray_checkpoint.py.

Usage:  python scripts/test_new_ct_checkpoint.py [--checkpoint path/to/file.pt]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from train_ct_monai import (  # noqa: E402
    RadImageNetClassifier, build_radimagenet_transform, radimagenet_loader, IMAGE_SIZE as CT_IMAGE_SIZE,
)

CHECKPOINT = ROOT / "checkpoints" / "ct_radimagenet_densenet121_best.pt"
DATA_DIR = ROOT / "data" / "ct_scan"
OUT_DIR = ROOT / "eval_outputs" / "ct_radimagenet_densenet121"
IMG_EXTS = {".png", ".jpg", ".jpeg"}

MARGIN_FRAC = 0.22  # same geometry as the X-ray occlusion test, for cross-modality comparability


def list_images(d: Path):
    return [p for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXTS]


def central_box_bounds(size: int, margin_frac: float = MARGIN_FRAC):
    lo = int(round(size * margin_frac))
    hi = size - lo
    return lo, hi


def load_model(device, checkpoint_path: Path):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"{checkpoint_path} not found -- upload/train the RadImageNet CT checkpoint there first."
        )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    model = RadImageNetClassifier(len(class_names)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, class_names, ckpt.get("val_acc"), ckpt.get("epoch")


class FileListDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = radimagenet_loader(str(path))
        return self.transform(img), label


def build_test_dataset(class_names, transform):
    samples = []
    for cls in class_names:
        d = DATA_DIR / "test" / cls
        cls_idx = class_names.index(cls)
        for p in list_images(d):
            samples.append((p, cls_idx))
    return FileListDataset(samples, transform)


def run_test_metrics(model, test_ds, class_names, device, batch_size=32):
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    report = classification_report(all_labels, all_preds, labels=list(range(len(class_names))),
                                    target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(all_labels, all_preds, labels=list(range(len(class_names))),
                                         target_names=class_names, zero_division=0)
    return {"accuracy": acc, "confusion_matrix": cm.tolist(),
            "classification_report": report, "classification_report_text": report_text}


def plot_confusion_matrix(cm, class_names, out_path: Path, title: str):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(1.4 * len(class_names) + 2, 1.4 * len(class_names) + 2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def occlude_center(batch: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    out = batch.clone()
    out[:, :, lo:hi, lo:hi] = 0.0
    return out


def run_occlusion_test(model, device, transform, class_names):
    lo, hi = central_box_bounds(CT_IMAGE_SIZE)
    box_side_frac = round((hi - lo) / CT_IMAGE_SIZE, 4)
    print(f"\n=== Center-blank occlusion test (central box [{lo}:{hi}] of {CT_IMAGE_SIZE}, "
          f"box side frac={box_side_frac}, margin={MARGIN_FRAC}) ===")

    modes = ("full", "center_blanked")
    result = {}
    pooled_pred_counts = {m: Counter() for m in modes}
    pooled_total = 0
    n_classes = len(class_names)

    for cls in class_names:
        d = DATA_DIR / "test" / cls
        paths = list_images(d)
        cls_idx = class_names.index(cls)
        ds = FileListDataset([(p, cls_idx) for p in paths], transform)
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        accs, pred_dists = {}, {}
        for mode in modes:
            correct, total = 0, 0
            pred_counts = Counter()
            with torch.no_grad():
                for imgs, labels in loader:
                    if mode == "center_blanked":
                        imgs = occlude_center(imgs, lo, hi)
                    imgs, labels = imgs.to(device), labels.to(device)
                    preds = model(imgs).argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    for pr in preds.cpu().tolist():
                        pred_counts[class_names[pr]] += 1
            accs[mode] = round(correct / total, 4)
            pred_dists[mode] = {c: round(pred_counts.get(c, 0) / total, 4) for c in class_names}
            pooled_pred_counts[mode].update(pred_counts)

        pooled_total += len(paths)
        result[cls] = {"n": len(paths), **accs, f"chance_{n_classes}class": round(1 / n_classes, 4),
                        "pred_distribution": pred_dists}
        print(f"  {cls:12s} n={len(paths):4d}  full={accs['full']:.4f}  "
              f"center_blanked={accs['center_blanked']:.4f}  (chance={1/n_classes:.4f})")

    pooled_dist = {m: {c: round(pooled_pred_counts[m].get(c, 0) / pooled_total, 4) for c in class_names}
                   for m in modes}
    result["pooled_pred_distribution"] = pooled_dist
    result["pooled_n"] = pooled_total
    print(f"\n  pooled across all {pooled_total} test images:")
    for m in modes:
        print(f"      {m:15s} pred_dist={pooled_dist[m]}")

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CHECKPOINT),
                    help="Path to the RadImageNet CT checkpoint to evaluate")
    args = ap.parse_args()
    checkpoint_path = Path(args.checkpoint)
    out_dir = ROOT / "eval_outputs" / checkpoint_path.stem
    tag = checkpoint_path.stem

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test_new_ct_checkpoint] device={device}")
    print(f"[test_new_ct_checkpoint] checkpoint={checkpoint_path} (read-only; active "
          f"checkpoint ct_best_model_phase35_v2.pt and backend/main.py untouched)")

    model, class_names, val_acc, epoch = load_model(device, checkpoint_path)
    print(f"[test_new_ct_checkpoint] loaded: classes={class_names}, "
          f"reported val_acc={val_acc:.4f} (epoch {epoch})")

    eval_transform = build_radimagenet_transform(train=False)
    test_ds = build_test_dataset(class_names, eval_transform)
    print(f"[test_new_ct_checkpoint] test set size: {len(test_ds)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_test_metrics(model, test_ds, class_names, device)
    print(f"\n[test_new_ct_checkpoint] TEST accuracy: {metrics['accuracy']:.4f}")
    print(metrics["classification_report_text"])

    cm_path = out_dir / f"{tag}_confusion_matrix.png"
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, cm_path,
                          title=f"{tag} test confusion matrix")
    print(f"[test_new_ct_checkpoint] confusion matrix saved to {cm_path}")

    occlusion = run_occlusion_test(model, device, eval_transform, class_names)

    out = {
        "checkpoint": str(checkpoint_path),
        "class_names": class_names,
        "reported_val_acc": val_acc,
        "checkpoint_epoch": epoch,
        "test_accuracy": metrics["accuracy"],
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
        "occlusion_test": occlusion,
    }
    metrics_path = out_dir / f"{tag}_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[test_new_ct_checkpoint] full metrics saved to {metrics_path}")
    return out


if __name__ == "__main__":
    main()
