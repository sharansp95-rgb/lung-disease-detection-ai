"""
Evaluates the torchxrayvision-backbone X-ray checkpoint
(checkpoints/xray_txrv_densenet121_best.pt) -- the X-ray model that
backend/main.py serves -- against the data/xray/test split. Read-only: it
doesn't modify the checkpoint or backend/main.py. (Written when this was
the candidate replacing xray_best_model_phase35_v2.pt; "old" below means
that earlier ImageNet-backbone model.)

Two parts:

  1. Standard test-set metrics -- accuracy, confusion matrix, per-class
     precision/recall/f1 -- in the same format as evaluate.py / the paper's
     Table IV, for direct old-vs-new comparison.

  2. Center-blank occlusion test (paper Section V-C): blank the central
     lung region of every test image and re-measure per-class accuracy.
     A model relying on genuine lung pathology should collapse toward
     chance (25%) once the lung field is gone; a model still scoring well
     is leaning on peripheral/border cues instead -- the same shortcut
     check as scripts/xray_source_correlation_audit.py's Q3
     ("border_only" mode there == center-blanked here; renamed for
     clarity). Reuses that script's exact central-box geometry
     (MARGIN_FRAC=0.22) so old-vs-new results are comparable box-for-box.

IMPORTANT: this new checkpoint uses torchxrayvision's own preprocessing
(single-channel, (2*(x/255)-1)*1024 normalization, XRayCenterCrop ->
XRayResizer(224)) -- NOT ml/dataset.py's ImageNet-normalized 3-channel
pipeline that the old xray_source_correlation_audit.py / evaluate.py use.
The transform and model-loading code here are imported directly from
ml/train_xray_txrv.py (the already-smoke-tested source of truth), not
reimplemented, so there's no risk of the two pipelines drifting apart.

Occlusion blanking is applied as 0.0 in torchxrayvision's OWN normalized
space (0.0 there corresponds to raw pixel value 127.5 -- i.e. "neutral
mid-gray" in that space's own convention), matching the ORIGINAL script's
convention of blanking to 0.0 in ITS normalized space (which is also that
space's own neutral/mean-centered value). Same relative operation in each
model's own preprocessing space, not the same literal pixel value -- the
fair way to replicate one methodology across two different normalization
schemes.

Usage:  python scripts/test_new_xray_checkpoint.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from train_xray_txrv import (  # noqa: E402
    XRVClassifier, build_xrv_transform, xrv_loader, IMAGE_SIZE,
)

CHECKPOINT = ROOT / "checkpoints" / "xray_txrv_densenet121_best.pt"
DATA_DIR = ROOT / "data" / "xray"
OUT_DIR = ROOT / "eval_outputs" / "xray_txrv_densenet121"
IMG_EXTS = {".png", ".jpg", ".jpeg"}

MARGIN_FRAC = 0.22  # matches scripts/xray_source_correlation_audit.py exactly


def list_images(d: Path):
    return [p for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXTS]


def central_box_bounds(size: int, margin_frac: float = MARGIN_FRAC):
    lo = int(round(size * margin_frac))
    hi = size - lo
    return lo, hi


def load_model(device):
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    xrv_weights = ckpt["backbone"]
    model = XRVClassifier(xrv_weights, len(class_names)).to(device)
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
        img = xrv_loader(str(path))  # single-channel 'L' load -- matches training
        return self.transform(img), label


def build_test_dataset(class_names, transform):
    samples = []
    for cls in class_names:
        d = DATA_DIR / "test" / cls
        cls_idx = class_names.index(cls)
        for p in list_images(d):
            samples.append((p, cls_idx))
    return FileListDataset(samples, transform)


# ── Part 1: standard test metrics (Table IV format) ─────────────────────────

def run_test_metrics(model, test_ds, class_names, device, batch_size=32):
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    report = classification_report(all_labels, all_preds, labels=list(range(len(class_names))),
                                    target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(all_labels, all_preds, labels=list(range(len(class_names))),
                                         target_names=class_names, zero_division=0)

    return {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "classification_report_text": report_text,
    }


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


# ── Part 2: center-blank occlusion test (paper Section V-C) ─────────────────

def occlude_center(batch: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    out = batch.clone()
    out[:, :, lo:hi, lo:hi] = 0.0  # 0.0 == this space's own neutral value, see module docstring
    return out


def run_occlusion_test(model, device, transform, class_names):
    lo, hi = central_box_bounds(IMAGE_SIZE)
    box_side_frac = round((hi - lo) / IMAGE_SIZE, 4)
    print(f"\n=== Center-blank occlusion test (central box [{lo}:{hi}] of {IMAGE_SIZE}, "
          f"box side frac={box_side_frac}, margin={MARGIN_FRAC}) ===")

    modes = ("full", "center_blanked")
    result = {}
    pooled_pred_counts = {m: Counter() for m in modes}
    pooled_total = 0

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
        result[cls] = {"n": len(paths), **accs, "chance_4class": round(1 / len(class_names), 4),
                        "pred_distribution": pred_dists}
        print(f"  {cls:15s} n={len(paths):4d}  full={accs['full']:.4f}  "
              f"center_blanked={accs['center_blanked']:.4f}  (chance={1/len(class_names):.4f})")

    pooled_dist = {m: {c: round(pooled_pred_counts[m].get(c, 0) / pooled_total, 4) for c in class_names}
                   for m in modes}
    result["pooled_pred_distribution"] = pooled_dist
    result["pooled_n"] = pooled_total
    print(f"\n  pooled across all {pooled_total} test images:")
    for m in modes:
        print(f"      {m:15s} pred_dist={pooled_dist[m]}")

    return result


# ── driver ────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test_new_xray_checkpoint] device={device}")
    print(f"[test_new_xray_checkpoint] checkpoint={CHECKPOINT} (read-only; this is the "
          f"served model, xray_txrv_densenet121_best.pt -- backend/main.py untouched)")

    model, class_names, val_acc, epoch = load_model(device)
    print(f"[test_new_xray_checkpoint] loaded: classes={class_names}, "
          f"reported val_acc={val_acc:.4f} (epoch {epoch})")

    eval_transform = build_xrv_transform(train=False)
    test_ds = build_test_dataset(class_names, eval_transform)
    print(f"[test_new_xray_checkpoint] test set size: {len(test_ds)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Part 1
    metrics = run_test_metrics(model, test_ds, class_names, device)
    print(f"\n[test_new_xray_checkpoint] TEST accuracy: {metrics['accuracy']:.4f}")
    print(metrics["classification_report_text"])

    cm_path = OUT_DIR / "xray_txrv_densenet121_confusion_matrix.png"
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, cm_path,
                          title="xray_txrv_densenet121 test confusion matrix")
    print(f"[test_new_xray_checkpoint] confusion matrix saved to {cm_path}")

    # Part 2
    occlusion = run_occlusion_test(model, device, eval_transform, class_names)

    out = {
        "checkpoint": str(CHECKPOINT),
        "class_names": class_names,
        "reported_val_acc": val_acc,
        "checkpoint_epoch": epoch,
        "test_accuracy": metrics["accuracy"],
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
        "occlusion_test": occlusion,
    }
    metrics_path = OUT_DIR / "xray_txrv_densenet121_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[test_new_xray_checkpoint] full metrics saved to {metrics_path}")

    return out


if __name__ == "__main__":
    main()
