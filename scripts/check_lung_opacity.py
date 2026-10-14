"""
One-off investigation: is /predict/xray's behavior on Lung_Opacity images
(a class the classifier was never trained on -- see prepare_xray.py's
"Lung_Opacity intentionally excluded") genuine model uncertainty, or a bug
introduced by the segmentation-crop integration?

For each of a fixed set of 10 Lung_Opacity raw images:
  - runs segmentation, records whether the mask was degenerate (fell back
    to border_crop) and the bbox area fraction
  - classifies via the segmentation-crop path (what /predict/xray does now)
  - classifies via a forced border-crop-only path (segmentation disabled),
    to isolate whether segmentation specifically changes the outcome

Also saves a 4-example visual grid: original / predicted mask / resulting
crop, for direct inspection.

Run from backend/ so relative imports resolve the same way main.py does:
    cd backend && python ../scripts/check_lung_opacity.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from gradcam_utils import preprocess_image  # noqa: E402
from models import load_model_bundle  # noqa: E402
from seg_utils import (  # noqa: E402
    SEG_INPUT_SIZE, MASK_PAD_PX, MIN_BBOX_AREA_FRAC, MAX_BBOX_AREA_FRAC,
    _to_seg_tensor, compute_lung_crop, load_seg_bundle, mask_bbox,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_xray import border_crop, TB_BORDER_FRAC  # noqa: E402

CHECKPOINT = ROOT / "checkpoints" / "xray_best_model_phase35_v2.pt"
DATA_DIR = ROOT / "data" / "xray"
SEG_CHECKPOINT = ROOT / "checkpoints" / "xray_seg_model.pt"
SEG_DATA_DIR = ROOT / "seg_data"
SRC_DIR = ROOT / "raw" / "xray_covid" / "COVID-19_Radiography_Dataset" / "Lung_Opacity" / "images"

FILES = [
    "Lung_Opacity-2384", "Lung_Opacity-4164", "Lung_Opacity-3946", "Lung_Opacity-312",
    "Lung_Opacity-1355", "Lung_Opacity-1160", "Lung_Opacity-338", "Lung_Opacity-4497",
    "Lung_Opacity-2360", "Lung_Opacity-5242",
]


def classify(pil_image, bundle):
    x = preprocess_image(pil_image, bundle.device)
    bundle.model.eval()
    with torch.no_grad():
        logits = bundle.model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return bundle.class_names[idx], float(probs[idx]), {n: float(p) for n, p in zip(bundle.class_names, probs)}


def get_seg_mask(pil_image, seg_bundle):
    """Raw thresholded mask at SEG_INPUT_SIZE, plus bbox + degeneracy info."""
    with seg_bundle.lock:
        x = _to_seg_tensor(pil_image, seg_bundle.device)
        with torch.no_grad():
            logits = seg_bundle.model(x)
            probs = torch.sigmoid(logits)
            mask = (probs[0, 0] > 0.5).cpu().numpy().astype(np.uint8)

    if mask.max() == 0:
        return mask, None, 0.0, True

    r0, r1, c0, c1 = mask_bbox(mask, pad=MASK_PAD_PX)
    area_frac = ((r1 - r0) * (c1 - c0)) / (SEG_INPUT_SIZE * SEG_INPUT_SIZE)
    degenerate = area_frac < MIN_BBOX_AREA_FRAC or area_frac > MAX_BBOX_AREA_FRAC
    return mask, (r0, r1, c0, c1), area_frac, degenerate


def main():
    print("Loading xray classifier (with reproduction check)...")
    bundle = load_model_bundle("xray", CHECKPOINT, DATA_DIR)
    print("Loading segmentation model (with reproduction check)...")
    seg_bundle = load_seg_bundle(SEG_CHECKPOINT, SEG_DATA_DIR)

    rows = []
    for name in FILES:
        raw_img = Image.open(SRC_DIR / f"{name}.png")

        mask, bbox, area_frac, degenerate = get_seg_mask(raw_img, seg_bundle)

        seg_cropped, used_seg = compute_lung_crop(raw_img, seg_bundle)
        seg_pred, seg_conf, seg_probs = classify(seg_cropped, bundle)

        arr = np.array(raw_img.convert("L"))
        border_cropped = Image.fromarray(border_crop(arr, TB_BORDER_FRAC))
        border_pred, border_conf, border_probs = classify(border_cropped, bundle)

        rows.append({
            "name": name, "mask": mask, "bbox": bbox, "area_frac": area_frac,
            "degenerate": degenerate, "used_seg": used_seg,
            "seg_pred": seg_pred, "seg_conf": seg_conf, "seg_probs": seg_probs,
            "border_pred": border_pred, "border_conf": border_conf, "border_probs": border_probs,
            "seg_cropped": seg_cropped, "border_cropped": border_cropped, "raw_img": raw_img,
        })

    print()
    print("=== Segmentation mask check ===")
    for r in rows:
        print(f"{r['name']:22s} area_frac={r['area_frac']:.3f}  degenerate={r['degenerate']}  "
              f"used_seg={r['used_seg']}")

    print()
    print("=== seg-crop (live behavior) vs border-crop-only (seg disabled) ===")
    changed = 0
    for r in rows:
        same = r["seg_pred"] == r["border_pred"]
        if not same:
            changed += 1
        print(f"{r['name']:22s} seg: {r['seg_pred']:12s} {r['seg_conf']:.4f}   "
              f"border: {r['border_pred']:12s} {r['border_conf']:.4f}   "
              f"{'SAME' if same else 'DIFFERENT'}")
    print(f"\nPrediction changed by disabling segmentation: {changed}/{len(rows)}")

    # --- visual grid: 4 examples, original / mask / crop ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample = rows[:4]
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for i, r in enumerate(sample):
        axes[0, i].imshow(r["raw_img"].convert("L"), cmap="gray")
        axes[0, i].set_title(f"{r['name']}\noriginal", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(r["mask"], cmap="gray")
        axes[1, i].set_title(f"predicted mask (area={r['area_frac']:.2f})", fontsize=9)
        axes[1, i].axis("off")

        axes[2, i].imshow(r["seg_cropped"], cmap="gray")
        axes[2, i].set_title(f"seg-crop -> {r['seg_pred']} {r['seg_conf']:.0%}", fontsize=9)
        axes[2, i].axis("off")

    plt.suptitle("Lung_Opacity: original / predicted mask / segmentation-crop result", fontsize=12)
    plt.tight_layout()
    plt.savefig(ROOT / "lung_opacity_seg_check.png", dpi=110)
    print(f"\nSaved visual grid: lung_opacity_seg_check.png")


if __name__ == "__main__":
    main()
