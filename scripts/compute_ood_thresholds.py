"""
Computes an out-of-distribution (OOD) detection threshold per modality:
extracts the penultimate-layer embedding (via each model's own
forward_features) for every training-set image using the trained model as a
fixed feature extractor, computes the centroid (mean embedding), then sets
the rejection threshold at the 99th percentile of training-image distances
from that centroid -- so ~1% of genuine in-distribution images would sit
outside it.

Uses the EVAL transform (no augmentation) on the train folder, not the
augmented training transform -- the centroid must be computed the same way
inference images are processed, and augmentation noise would blur it.

Branches on the checkpoint's own `backbone` field: torchxrayvision-backboned
checkpoints (e.g. xray_txrv_densenet121_best.pt) use
ml/train_xray_txrv.py's preprocessing and XRVClassifier; everything else
uses the original ml/dataset.py / ml/model.py pipeline. This matters here
specifically because a checkpoint's embedding space is tied to its own
architecture -- reusing a threshold/centroid computed against a DIFFERENT
model's features would silently produce meaningless distances, not an error.

Output: backend/ood_stats/{modality}_ood.pt containing centroid, threshold,
and the full train-distance array (for diagnostics / re-checking FPR later).

Usage:  python scripts/compute_ood_thresholds.py [--modality {xray,ct,both}]
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from dataset import build_transforms  # noqa: E402
from model import build_model  # noqa: E402

CHECKPOINTS = {
    "xray": ROOT / "checkpoints" / "xray_txrv_densenet121_best.pt",
    "ct": ROOT / "checkpoints" / "ct_best_model_phase35_v2.pt",
}
DATA_DIRS = {
    "xray": ROOT / "data" / "xray",
    "ct": ROOT / "data" / "ct_scan",
}
OUT_DIR = ROOT / "backend" / "ood_stats"
PERCENTILE = 0.99

XRV_BACKBONE_PREFIX = "densenet121-res224"


def _is_xrv_backbone(backbone: str) -> bool:
    return backbone.startswith(XRV_BACKBONE_PREFIX)


def compute_for_modality(name: str, device: torch.device):
    ckpt_path = CHECKPOINTS[name]
    data_dir = DATA_DIRS[name]

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    backbone = ckpt["backbone"]

    if _is_xrv_backbone(backbone):
        # Deferred import: torchxrayvision pollutes sys.modules['model'] as an
        # import-time side effect (see ml/train_xray_txrv.py's own note).
        # `from model import build_model` above has already executed and
        # cached the real ml/model.py, so importing this here keeps that safe.
        from train_xray_txrv import XRVClassifier, build_xrv_transform, xrv_loader
        model = XRVClassifier(backbone, len(class_names)).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        eval_transform = build_xrv_transform(train=False)
        train_ds = datasets.ImageFolder(data_dir / "train", transform=eval_transform, loader=xrv_loader)
    else:
        model = build_model(backbone, len(class_names), pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        eval_transform = build_transforms(train=False)
        train_ds = datasets.ImageFolder(data_dir / "train", transform=eval_transform)

    # num_workers=0: build_xrv_transform() returns a nested closure, which
    # Windows' spawn-based multiprocessing can't pickle to hand to worker
    # processes (confirmed: AttributeError: Can't get local object
    # 'build_xrv_transform.<locals>._transform' with num_workers>0). Single-
    # process is slower but this is a one-time offline script, not a
    # per-request path, so that's an acceptable trade here.
    loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)

    print(f"[{name}] backbone={backbone!r}, extracting embeddings for {len(train_ds)} training images...")
    t0 = time.time()
    embeddings = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            feats = model.forward_features(images)
            embeddings.append(feats.cpu())
    embeddings = torch.cat(embeddings, dim=0)
    print(f"[{name}] extracted {embeddings.shape} in {time.time()-t0:.1f}s")

    centroid = embeddings.mean(dim=0)
    distances = torch.norm(embeddings - centroid.unsqueeze(0), dim=1)

    threshold = torch.quantile(distances, PERCENTILE).item()
    print(f"[{name}] centroid dim={centroid.shape[0]}, "
          f"distance range=[{distances.min():.2f}, {distances.max():.2f}], "
          f"mean={distances.mean():.2f}, "
          f"{PERCENTILE:.0%} threshold={threshold:.2f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name}_ood.pt"
    torch.save({
        "centroid": centroid,
        "threshold": threshold,
        "percentile": PERCENTILE,
        "train_distances": distances,
        "n_train_images": len(train_ds),
    }, out_path)
    print(f"[{name}] saved to {out_path}")
    return centroid, threshold, distances


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["xray", "ct", "both"], default="both")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    modalities = ("xray", "ct") if args.modality == "both" else (args.modality,)
    for name in modalities:
        compute_for_modality(name, device)


if __name__ == "__main__":
    main()
