"""
One-off comparison: for a stratified sample of raw (never-cropped) X-ray
images with known ground truth, classify each one three ways --
  A. old fix: border_crop() only (the previous train/inference-mismatch fix)
  B. new fix: segmentation-guided crop (compute_lung_crop, with border_crop
     fallback for degenerate masks)
  C. reference: the already-cropped file from data/xray/test/<class>/ that
     scripts/prepare_xray.py built from this exact raw image

Bypasses the FastAPI/Groq layer entirely (goes straight at the classifier
and segmentation models) since only the classification output matters here,
not report generation -- this avoids ~60x3 unnecessary Groq calls.

Run from backend/ so relative imports resolve the same way main.py does:
    cd backend && python ../scripts/compare_crop_fixes.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from gradcam_utils import preprocess_image  # noqa: E402
from models import load_model_bundle  # noqa: E402
from seg_utils import compute_lung_crop, load_seg_bundle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_xray import border_crop, TB_BORDER_FRAC  # noqa: E402

CHECKPOINT = ROOT / "checkpoints" / "xray_best_model_phase35_v2.pt"
DATA_DIR = ROOT / "data" / "xray"
SEG_CHECKPOINT = ROOT / "checkpoints" / "xray_seg_model.pt"
SEG_DATA_DIR = ROOT / "seg_data"


def classify(pil_image, bundle):
    x = preprocess_image(pil_image, bundle.device)
    bundle.model.eval()
    with torch.no_grad():
        logits = bundle.model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return bundle.class_names[idx], float(probs[idx])


def main():
    samples = json.loads(Path(ROOT / "sample_manifest.json").read_text())
    print(f"Loaded {len(samples)} samples")

    print("Loading xray classifier (with reproduction check)...")
    bundle = load_model_bundle("xray", CHECKPOINT, DATA_DIR)

    print("Loading segmentation model (with reproduction check)...")
    seg_bundle = load_seg_bundle(SEG_CHECKPOINT, SEG_DATA_DIR)

    results = []
    for i, (true_class, ref_path, raw_path) in enumerate(samples):
        raw_img = Image.open(ROOT / raw_path)

        # A: old fix -- border crop only
        arr = np.array(raw_img.convert("L"))
        old_cropped = Image.fromarray(border_crop(arr, TB_BORDER_FRAC))
        old_pred, old_conf = classify(old_cropped, bundle)

        # B: new fix -- segmentation-guided crop (with fallback)
        new_cropped, used_seg = compute_lung_crop(raw_img, seg_bundle)
        new_pred, new_conf = classify(new_cropped, bundle)

        # C: reference -- the actual pre-cropped training/test file
        ref_img = Image.open(ROOT / ref_path)
        ref_pred, ref_conf = classify(ref_img, bundle)

        results.append({
            "true_class": true_class,
            "raw_path": raw_path,
            "old_pred": old_pred, "old_conf": old_conf, "old_correct": old_pred == true_class,
            "new_pred": new_pred, "new_conf": new_conf, "new_correct": new_pred == true_class,
            "used_segmentation": used_seg,
            "ref_pred": ref_pred, "ref_conf": ref_conf, "ref_correct": ref_pred == true_class,
        })

        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(samples)}", flush=True)

    Path(ROOT / "crop_fix_comparison.json").write_text(json.dumps(results, indent=1))

    # --- summary ---
    def summarize(key_pred, key_conf, key_correct, label):
        n = len(results)
        acc = sum(r[key_correct] for r in results) / n
        mean_conf = float(np.mean([r[key_conf] for r in results]))
        mean_conf_correct = float(np.mean([r[key_conf] for r in results if r[key_correct]]) or 0)
        print(f"{label:30s} accuracy={acc:.1%}  mean_confidence={mean_conf:.1%}  "
              f"mean_confidence(correct only)={mean_conf_correct:.1%}")

    print()
    print("=== Overall (n=60, all classes) ===")
    summarize("old_pred", "old_conf", "old_correct", "old fix (border-crop only)")
    summarize("new_pred", "new_conf", "new_correct", "new fix (segmentation-crop)")
    summarize("ref_pred", "ref_conf", "ref_correct", "reference (fully-cropped)")

    print()
    print("=== Per-class ===")
    for cls in ["COVID-19", "Normal", "Pneumonia", "Tuberculosis"]:
        subset = [r for r in results if r["true_class"] == cls]
        n = len(subset)
        old_acc = sum(r["old_correct"] for r in subset) / n
        new_acc = sum(r["new_correct"] for r in subset) / n
        ref_acc = sum(r["ref_correct"] for r in subset) / n
        print(f"{cls:14s} n={n:3d}  old_fix={old_acc:.0%}  new_fix={new_acc:.0%}  reference={ref_acc:.0%}")

    seg_used_count = sum(r["used_segmentation"] for r in results)
    print(f"\nSegmentation crop actually used (not degenerate-fallback): {seg_used_count}/{len(results)}")


if __name__ == "__main__":
    main()
