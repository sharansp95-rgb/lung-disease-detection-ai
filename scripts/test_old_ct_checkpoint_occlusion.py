"""
Center-blank occlusion baseline for the CURRENT PRODUCTION CT checkpoint
(checkpoints/ct_best_model_phase35_v2.pt), computed ahead of the new
RadImageNet CT checkpoint's arrival so both sides of the old-vs-new
comparison are ready as soon as scripts/test_new_ct_checkpoint.py can run.

Same box geometry (MARGIN_FRAC=0.22, 224x224) as
scripts/test_new_xray_checkpoint.py and scripts/test_new_ct_checkpoint.py,
but using the OLD model's own preprocessing (ml/model.py's ImageClassifier,
ml/dataset.py's ImageNet-normalized 3-channel pipeline) -- imported directly,
not reimplemented, same principle as the two "new checkpoint" test scripts.

Read-only: does not modify data/ct_scan/ or any checkpoint.

Usage:  python scripts/test_old_ct_checkpoint_occlusion.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from dataset import build_transforms, IMAGE_SIZE  # noqa: E402
from model import build_model  # noqa: E402

CHECKPOINT = ROOT / "checkpoints" / "ct_best_model_phase35_v2.pt"
DATA_DIR = ROOT / "data" / "ct_scan"
OUT_DIR = ROOT / "eval_outputs" / "ct_best_model_phase35_v2"
IMG_EXTS = {".png", ".jpg", ".jpeg"}

MARGIN_FRAC = 0.22


def list_images(d: Path):
    return [p for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXTS]


def central_box_bounds(size: int, margin_frac: float = MARGIN_FRAC):
    lo = int(round(size * margin_frac))
    hi = size - lo
    return lo, hi


def load_model(device):
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    backbone = ckpt["backbone"]
    model = build_model(backbone, len(class_names), pretrained=False).to(device)
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
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def occlude_center(batch: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    out = batch.clone()
    out[:, :, lo:hi, lo:hi] = 0.0
    return out


def run_occlusion_test(model, device, transform, class_names):
    lo, hi = central_box_bounds(IMAGE_SIZE)
    box_side_frac = round((hi - lo) / IMAGE_SIZE, 4)
    print(f"=== Center-blank occlusion test, OLD CT model (central box [{lo}:{hi}] of {IMAGE_SIZE}, "
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test_old_ct_checkpoint_occlusion] device={device}")
    model, class_names, val_acc, epoch = load_model(device)
    print(f"[test_old_ct_checkpoint_occlusion] loaded: classes={class_names}, "
          f"reported val_acc={val_acc:.4f} (epoch {epoch})")

    eval_transform = build_transforms(train=False)
    occlusion = run_occlusion_test(model, device, eval_transform, class_names)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "checkpoint": str(CHECKPOINT),
        "class_names": class_names,
        "reported_val_acc": val_acc,
        "checkpoint_epoch": epoch,
        "occlusion_test": occlusion,
    }
    out_path = OUT_DIR / "ct_best_model_phase35_v2_occlusion_test.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[test_old_ct_checkpoint_occlusion] saved to {out_path}")
    return out


if __name__ == "__main__":
    main()
