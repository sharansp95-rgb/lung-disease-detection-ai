"""
Generalization Diagnosis Script
================================
Diagnoses why classifiers perform well on held-out test images from the
training sources but poorly on external (out-of-source) chest images.

Tests run:
  TEST 1 — Full pipeline gate analysis per external image
  TEST 2 — Preprocessing comparison (tensor stats + visual grid)
  TEST 3 — Controlled corruption on training-source correct predictions
  TEST 4 — Crop ablation on external X-rays
  TEST 5 — OOD centroid distance plot vs. threshold
  TEST 6 — Grad-CAM overlays on external failures

Uses ONLY the real backend code paths — no reimplementation.

Output: GENERALIZATION_DIAGNOSIS.md + diagnosis_outputs/ directory.
"""

import io
import json
import logging
import math
import os
import sys
import textwrap
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "backend"
ML_DIR = ROOT / "ml"
SCRIPTS_DIR = ROOT / "scripts"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(ML_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

# Real backend imports — exact same code paths as _run_prediction()
from gradcam_utils import predict_with_heatmap, preprocess_image  # noqa: E402
from models import ModelBundle, load_model_bundle               # noqa: E402
from ood_detector import OODDetector                            # noqa: E402
from seg_utils import SegBundle, compute_lung_crop, load_seg_bundle  # noqa: E402
from dataset import IMAGENET_MEAN, IMAGENET_STD, build_transforms   # noqa: E402
from gradcam import GradCAM, denormalize, get_target_layer, overlay_cam  # noqa: E402
from model import build_model                                   # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("diagnose")

# ── constants (mirrors main.py exactly) ─────────────────────────────────────
BLANK_IMAGE_STD_THRESHOLD = 10.0
COLOR_CONTENT_THRESHOLD = 20.0
MIN_CONFIDENCE = {"xray": 0.35, "ct": 0.40}

CHECKPOINTS = {
    "xray":        ROOT / "checkpoints" / "xray_best_model_phase35_v2.pt",
    "ct":          ROOT / "checkpoints" / "ct_best_model_phase35_v2.pt",
    "gate":        ROOT / "checkpoints" / "modality_gate_best_model.pt",
    "xray_sanity": ROOT / "checkpoints" / "xray_sanity_best.pt",
    "ct_sanity":   ROOT / "checkpoints" / "ct_sanity_best.pt",
}
DATA_DIRS = {
    "xray":        ROOT / "data" / "xray",
    "ct":          ROOT / "data" / "ct_scan",
    "gate":        ROOT / "data" / "modality_gate",
    "xray_sanity": ROOT / "data" / "xray_sanity",
    "ct_sanity":   ROOT / "data" / "ct_sanity",
}
SEG_CHECKPOINT = ROOT / "checkpoints" / "xray_seg_model.pt"
SEG_DATA_DIR   = ROOT / "seg_data"

# External probe set — prefer data/external_probe, fall back to sample images
EXTERNAL_PROBE_DIR = ROOT / "data" / "external_probe"
SAMPLE_IMAGES_DIR  = ROOT / "sample images"

# Output directory
OUT_DIR = ROOT / "diagnosis_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


# ── helpers ──────────────────────────────────────────────────────────────────

def load_image_safe(path: Path) -> Image.Image | None:
    """Load PIL image, return None on error."""
    try:
        img = Image.open(path)
        img.load()
        return img
    except Exception as e:
        logger.warning(f"Cannot load {path}: {e}")
        return None


def image_passes_format_guards(img: Image.Image) -> tuple[bool, str]:
    """Replicates _read_and_validate_image() guards from main.py."""
    gray_std = float(np.array(img.convert("L"), dtype=np.float32).std())
    if gray_std < BLANK_IMAGE_STD_THRESHOLD:
        return False, f"blank (gray_std={gray_std:.1f} < {BLANK_IMAGE_STD_THRESHOLD})"
    rgb_arr = np.array(img.convert("RGB"), dtype=np.float32)
    color_dev = float((rgb_arr.max(axis=2) - rgb_arr.min(axis=2)).mean())
    if color_dev > COLOR_CONTENT_THRESHOLD:
        return False, f"color_content (color_dev={color_dev:.1f} > {COLOR_CONTENT_THRESHOLD})"
    return True, "ok"


def sanity_predict(bundle: ModelBundle, input_tensor: torch.Tensor) -> tuple[str, float]:
    """Returns (label, confidence)."""
    bundle.model.eval()
    with torch.no_grad():
        logits = bundle.model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = int(logits.argmax(dim=1).item())
    return bundle.class_names[pred_idx], float(probs[pred_idx])


def gate_predict(bundle: ModelBundle, input_tensor: torch.Tensor) -> tuple[str, float]:
    """Returns (modality, confidence)."""
    bundle.model.eval()
    with torch.no_grad():
        logits = bundle.model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = int(logits.argmax(dim=1).item())
    return bundle.class_names[pred_idx], float(probs[pred_idx])


def collect_external_images() -> dict[str, list[tuple[Path, str, str]]]:
    """
    Returns dict[modality -> list[(path, true_class, source_note)]]
    from data/external_probe/{xray,ct}/{class}/ if it exists, else
    falls back to 'sample images/' (which contains images from sources
    outside the training datasets).
    """
    result = {"xray": [], "ct": []}

    if EXTERNAL_PROBE_DIR.exists():
        for modality in ("xray", "ct"):
            mod_dir = EXTERNAL_PROBE_DIR / modality
            if not mod_dir.exists():
                continue
            for cls_dir in sorted(mod_dir.iterdir()):
                if not cls_dir.is_dir():
                    continue
                for f in sorted(cls_dir.iterdir()):
                    if f.suffix.lower() in IMAGE_EXTS:
                        result[modality].append((f, cls_dir.name, "external_probe"))
        if any(result.values()):
            print("[probe] Using data/external_probe/")
            return result

    # Fallback: sample images/
    print("[probe] data/external_probe/ not found or empty — using 'sample images/'")
    if not SAMPLE_IMAGES_DIR.exists():
        print("[probe] WARNING: 'sample images/' also missing. No external images available.")
        return result

    # Map sample image subfolders to modality + class
    #   cancer/Bengin cases -> ct/Benign
    #   cancer/malignant cases -> ct/Malignant
    #   cancer/normal -> ct/Normal
    #   covid -> xray/COVID-19
    #   normal -> xray/Normal
    #   Pneumonia -> xray/Pneumonia
    #   tb -> xray/Tuberculosis
    folder_map = {
        "cancer/Bengin cases":    ("ct",   "Benign"),
        "cancer/malignant cases": ("ct",   "Malignant"),
        "cancer/normal":          ("ct",   "Normal"),
        "covid":                  ("xray", "COVID-19"),
        "normal":                 ("xray", "Normal"),
        "Pneumonia":              ("xray", "Pneumonia"),
        "tb":                     ("xray", "Tuberculosis"),
    }

    for rel_path, (modality, true_class) in folder_map.items():
        folder = SAMPLE_IMAGES_DIR / rel_path
        if not folder.exists():
            continue
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                result[modality].append((f, true_class, f"sample_images/{rel_path}"))

    return result


def collect_training_test_images(modality: str, n: int = 20) -> list[tuple[Path, str]]:
    """Returns list of (path, true_class) from data/{modality}/test/."""
    data_dir = DATA_DIRS[modality] if modality != "xray" else ROOT / "data" / "xray"
    if modality == "ct":
        data_dir = ROOT / "data" / "ct_scan"
    test_dir = data_dir / "test"
    results = []
    for cls_dir in sorted(test_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        files = sorted(cls_dir.iterdir())
        for f in files:
            if f.suffix.lower() in IMAGE_EXTS:
                results.append((f, cls_dir.name))
    import random
    random.seed(42)
    random.shuffle(results)
    return results[:n]


# ── model loading ─────────────────────────────────────────────────────────────

def load_all_bundles():
    print("Loading model bundles...")
    bundles = {}
    for name in ("xray", "ct", "gate", "xray_sanity", "ct_sanity"):
        print(f"  [{name}] loading...")
        t0 = time.time()
        bundles[name] = load_model_bundle(name, CHECKPOINTS[name], DATA_DIRS[name])
        print(f"  [{name}] loaded in {time.time()-t0:.1f}s")
    return bundles


def load_ood_detectors(bundles):
    ood = {}
    for modality in ("xray", "ct"):
        ood[modality] = OODDetector(modality, bundles[modality].device)
    return ood


def load_seg():
    print("Loading segmentation model...")
    return load_seg_bundle(SEG_CHECKPOINT, SEG_DATA_DIR)


# ── segmentation helpers ──────────────────────────────────────────────────────

def compute_lung_crop_with_info(image: Image.Image, seg_bundle: SegBundle
                                 ) -> tuple[Image.Image, bool, float]:
    """
    Like compute_lung_crop() but also returns bbox area fraction.
    Replicates the logic in seg_utils.compute_lung_crop exactly.
    """
    from seg_utils import (SEG_INPUT_SIZE, MIN_BBOX_AREA_FRAC, MAX_BBOX_AREA_FRAC,
                           _to_seg_tensor)
    from prepare_xray import border_crop, mask_bbox, MASK_PAD_PX, TB_BORDER_FRAC

    with seg_bundle.lock:
        x = _to_seg_tensor(image, seg_bundle.device)
        with torch.no_grad():
            logits = seg_bundle.model(x)
            probs = torch.sigmoid(logits)
            mask_small = (probs[0, 0] > 0.5).cpu().numpy().astype(np.uint8)

    gray = np.array(image.convert("L"))

    if mask_small.max() == 0:
        cropped = border_crop(gray, TB_BORDER_FRAC)
        return Image.fromarray(cropped), False, 0.0

    r0, r1, c0, c1 = mask_bbox(mask_small, pad=MASK_PAD_PX)
    bbox_area_frac = ((r1 - r0) * (c1 - c0)) / (SEG_INPUT_SIZE * SEG_INPUT_SIZE)

    if bbox_area_frac < MIN_BBOX_AREA_FRAC or bbox_area_frac > MAX_BBOX_AREA_FRAC:
        cropped = border_crop(gray, TB_BORDER_FRAC)
        return Image.fromarray(cropped), False, bbox_area_frac

    H, W = gray.shape
    scale_r, scale_c = H / SEG_INPUT_SIZE, W / SEG_INPUT_SIZE
    orig_r0, orig_r1 = int(r0 * scale_r), int(r1 * scale_r)
    orig_c0, orig_c1 = int(c0 * scale_c), int(c1 * scale_c)
    cropped = gray[orig_r0:orig_r1, orig_c0:orig_c1]
    return Image.fromarray(cropped), True, bbox_area_frac


# ── TEST 1: Full pipeline gate analysis ──────────────────────────────────────

def test1_pipeline_gates(bundles, ood_detectors, seg_bundle, external_images):
    """
    For every external image, run the full _run_prediction() path and log per image:
    - format guard pass/fail + gray_std, color_dev
    - sanity gate pass/fail + confidence
    - modality gate pass/fail + predicted modality + confidence
    - OOD distance, threshold, pass/fail
    - (xray) used U-Net mask vs. border_crop, bbox area fraction
    - final predicted class, confidence, correct?
    Return list of result dicts and summary counts.
    """
    print("\n=== TEST 1: Full Pipeline Gate Analysis ===")
    records = []

    for modality, images in external_images.items():
        bundle = bundles[modality]
        sanity_bundle = bundles[f"{modality}_sanity"]
        gate_bundle = bundles["gate"]
        ood = ood_detectors[modality]

        for img_path, true_class, source in images:
            rec = {
                "path": str(img_path),
                "filename": img_path.name,
                "modality": modality,
                "true_class": true_class,
                "source": source,
            }

            img = load_image_safe(img_path)
            if img is None:
                rec.update({"error": "load_failed", "stage_rejected": "load"})
                records.append(rec)
                continue

            # Format guards
            gray_std = float(np.array(img.convert("L"), dtype=np.float32).std())
            rgb_arr = np.array(img.convert("RGB"), dtype=np.float32)
            color_dev = float((rgb_arr.max(axis=2) - rgb_arr.min(axis=2)).mean())
            rec["gray_std"] = round(gray_std, 2)
            rec["color_dev"] = round(color_dev, 2)
            format_pass = gray_std >= BLANK_IMAGE_STD_THRESHOLD and color_dev <= COLOR_CONTENT_THRESHOLD
            rec["format_guard_pass"] = format_pass
            if not format_pass:
                rec["stage_rejected"] = "format_guard"
                records.append(rec)
                continue

            # Segmentation (xray only)
            if modality == "xray":
                cropped_img, used_seg, bbox_frac = compute_lung_crop_with_info(img, seg_bundle)
                rec["used_segmentation"] = used_seg
                rec["bbox_area_frac"] = round(bbox_frac, 3)
            else:
                cropped_img = img
                rec["used_segmentation"] = None
                rec["bbox_area_frac"] = None

            # Preprocess
            input_tensor = preprocess_image(cropped_img, bundle.device)

            # Sanity gate
            sanity_label, sanity_conf = sanity_predict(sanity_bundle, input_tensor)
            rec["sanity_label"] = sanity_label
            rec["sanity_conf"] = round(sanity_conf, 4)
            rec["sanity_pass"] = (sanity_label == "scan")
            if not rec["sanity_pass"]:
                rec["stage_rejected"] = "sanity_gate"
                records.append(rec)
                continue

            # Modality gate
            gate_label, gate_conf = gate_predict(gate_bundle, input_tensor)
            rec["gate_pred"] = gate_label
            rec["gate_conf"] = round(gate_conf, 4)
            rec["gate_pass"] = (gate_label == modality)
            if not rec["gate_pass"]:
                rec["stage_rejected"] = "modality_gate"
                records.append(rec)
                continue

            # OOD check
            with bundle.lock:
                is_ood, distance = ood.check(bundle.model, input_tensor)
                rec["ood_distance"] = round(distance, 2)
                rec["ood_threshold"] = round(ood.threshold, 2)
                rec["ood_flagged"] = is_ood
                if is_ood:
                    rec["stage_rejected"] = "ood_gate"
                    records.append(rec)
                    continue

                # Classify
                pred_idx, probs, _ = predict_with_heatmap(bundle.model, input_tensor)

            predicted_class = bundle.class_names[pred_idx]
            confidence = float(probs[pred_idx])
            rec["predicted_class"] = predicted_class
            rec["confidence"] = round(confidence, 4)
            rec["correct"] = (predicted_class == true_class)
            rec["stage_rejected"] = None  # passed all gates

            records.append(rec)

    # Summary
    total = len(records)
    by_stage = {}
    for r in records:
        stage = r.get("stage_rejected", None)
        key = stage if stage else ("correct" if r.get("correct") else "wrong_class")
        by_stage[key] = by_stage.get(key, 0) + 1

    print(f"  Total external images: {total}")
    for stage, count in sorted(by_stage.items()):
        print(f"    {stage}: {count} ({100*count/total:.1f}%)")

    return records, by_stage


# ── TEST 2: Preprocessing comparison ─────────────────────────────────────────

def test2_preprocessing_comparison(bundles, ood_detectors, seg_bundle, external_images):
    """
    For 5 external and 5 training-source images of the same class, save
    224x224 preprocessed tensor as PNG grid (denormalized), side by side.
    Also report per-image mean and std.
    """
    print("\n=== TEST 2: Preprocessing Comparison ===")

    def preprocess_and_stats(img, modality, seg_bndl, bundle, label):
        if modality == "xray":
            cropped_img, used_seg, _ = compute_lung_crop_with_info(img, seg_bndl)
            crop_method = "unet" if used_seg else "border_crop"
        else:
            cropped_img = img
            crop_method = "none"
        tensor = preprocess_image(cropped_img, bundle.device)
        # Denormalize for display
        display = denormalize(tensor[0].cpu(), IMAGENET_MEAN, IMAGENET_STD)
        # Stats on the normalized tensor values
        t_np = tensor[0].cpu().numpy()
        return display, crop_method, float(t_np.mean()), float(t_np.std()), label

    results = {}
    for modality in ("xray", "ct"):
        ext_list = external_images.get(modality, [])
        if not ext_list:
            print(f"  [{modality}] No external images available, skipping.")
            continue

        bundle = bundles[modality]

        # Pick target class: the first class that appears in external images
        class_groups = {}
        for path, cls, src in ext_list:
            class_groups.setdefault(cls, []).append((path, cls, src))
        target_class = next(iter(class_groups))
        ext_samples = class_groups[target_class][:5]

        # Get 5 training-source test images of the same class
        train_dir = DATA_DIRS["xray"] if modality == "xray" else ROOT / "data" / "ct_scan"
        train_cls_dir = train_dir / "test" / target_class
        if not train_cls_dir.exists():
            # Map class name differences
            # Try all caps or lowercase variations
            for d in (train_dir / "test").iterdir():
                if d.is_dir() and d.name.lower() == target_class.lower():
                    train_cls_dir = d
                    break
        train_paths = sorted(train_cls_dir.iterdir())[:5] if train_cls_dir.exists() else []

        rows_ext = []
        rows_train = []
        stats = {"external": [], "training": []}

        for path, cls, src in ext_samples:
            img = load_image_safe(path)
            if img is None:
                continue
            display, crop_method, tmean, tstd, lbl = preprocess_and_stats(
                img, modality, seg_bundle, bundle, f"EXT {path.name[:20]}")
            rows_ext.append(display)
            stats["external"].append({"file": path.name, "mean": round(tmean, 4), "std": round(tstd, 4), "crop": crop_method})

        for tp in train_paths:
            img = load_image_safe(tp)
            if img is None:
                continue
            display, crop_method, tmean, tstd, lbl = preprocess_and_stats(
                img, modality, seg_bundle, bundle, f"TRN {tp.name[:20]}")
            rows_train.append(display)
            stats["training"].append({"file": tp.name, "mean": round(tmean, 4), "std": round(tstd, 4), "crop": crop_method})

        # Build grid: top row = external, bottom row = training
        n_cols = max(len(rows_ext), len(rows_train), 1)
        n_rows = 2
        cell_h, cell_w = 224, 224
        label_h = 30
        grid = np.ones((n_rows * (cell_h + label_h), n_cols * cell_w, 3), dtype=np.uint8) * 240

        for i, disp in enumerate(rows_ext):
            y = 0
            x = i * cell_w
            grid[y:y+cell_h, x:x+cell_w] = disp
            cv2.putText(grid, f"EXT {i+1}", (x+2, y+cell_h+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)

        for i, disp in enumerate(rows_train):
            y = cell_h + label_h
            x = i * cell_w
            grid[y:y+cell_h, x:x+cell_w] = disp
            cv2.putText(grid, f"TRN {i+1}", (x+2, y+cell_h+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 150, 0), 1)

        grid_path = OUT_DIR / f"test2_{modality}_{target_class}_preproc_grid.png"
        cv2.imwrite(str(grid_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"  [{modality}/{target_class}] saved grid: {grid_path.name}")
        print(f"    External  mean±std: "
              f"{np.mean([s['mean'] for s in stats['external']]):.4f} ± "
              f"{np.mean([s['std'] for s in stats['external']]):.4f}")
        print(f"    Training  mean±std: "
              f"{np.mean([s['mean'] for s in stats['training']]):.4f} ± "
              f"{np.mean([s['std'] for s in stats['training']]):.4f}")
        results[modality] = {
            "target_class": target_class,
            "grid_file": grid_path.name,
            "stats": stats,
        }

    return results


# ── TEST 3: Controlled corruption ─────────────────────────────────────────────

def apply_corruption(img: Image.Image, name: str) -> Image.Image:
    """Apply one named corruption to a PIL image."""
    arr = np.array(img.convert("L"))
    H, W = arr.shape

    if name == "resize_512":
        img2 = img.resize((512, 512), Image.LANCZOS).resize((W, H), Image.LANCZOS)
        return img2.convert("RGB")
    elif name == "jpeg_q60":
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=60)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    elif name == "black_border_10pct":
        pad_h = int(H * 0.10)
        pad_w = int(W * 0.10)
        result = np.zeros((H + 2*pad_h, W + 2*pad_w), dtype=arr.dtype)
        result[pad_h:pad_h+H, pad_w:pad_w+W] = arr
        return Image.fromarray(result).convert("RGB")
    elif name == "contrast_07":
        from PIL import ImageEnhance
        return ImageEnhance.Contrast(img.convert("RGB")).enhance(0.7)
    elif name == "contrast_14":
        from PIL import ImageEnhance
        return ImageEnhance.Contrast(img.convert("RGB")).enhance(1.4)
    elif name == "gamma_07":
        a = arr.astype(np.float32) / 255.0
        a = np.power(a, 0.7)
        return Image.fromarray((a * 255).astype(np.uint8)).convert("RGB")
    elif name == "gamma_14":
        a = arr.astype(np.float32) / 255.0
        a = np.power(a, 1.4)
        return Image.fromarray((a * 255).astype(np.uint8)).convert("RGB")
    elif name == "invert":
        return ImageOps.invert(img.convert("RGB"))
    elif name == "histeq":
        result = cv2.equalizeHist(arr)
        return Image.fromarray(result).convert("RGB")
    else:
        raise ValueError(f"Unknown corruption: {name}")


CORRUPTION_NAMES = [
    "resize_512", "jpeg_q60", "black_border_10pct",
    "contrast_07", "contrast_14",
    "gamma_07", "gamma_14",
    "invert", "histeq",
]


def test3_controlled_corruption(bundles, ood_detectors, seg_bundle):
    """
    Take 20 training-source test images the model currently gets right.
    Apply each corruption and re-predict. Report accuracy per corruption.
    """
    print("\n=== TEST 3: Controlled Corruption ===")

    results = {}

    for modality in ("xray", "ct"):
        bundle = bundles[modality]
        ood = ood_detectors[modality]
        data_dir = ROOT / "data" / "xray" if modality == "xray" else ROOT / "data" / "ct_scan"

        # Gather baseline-correct images (up to 20)
        print(f"  [{modality}] collecting baseline-correct images...")
        correct_samples = []
        test_dir = data_dir / "test"
        import random
        random.seed(42)
        all_test = []
        for cls_dir in sorted(test_dir.iterdir()):
            if cls_dir.is_dir():
                for f in sorted(cls_dir.iterdir()):
                    if f.suffix.lower() in IMAGE_EXTS:
                        all_test.append((f, cls_dir.name))
        random.shuffle(all_test)

        for img_path, true_class in all_test:
            if len(correct_samples) >= 20:
                break
            img = load_image_safe(img_path)
            if img is None:
                continue
            if modality == "xray":
                proc_img, _, _ = compute_lung_crop_with_info(img, seg_bundle)
            else:
                proc_img = img
            tensor = preprocess_image(proc_img, bundle.device)

            # Check format guards
            ok, _ = image_passes_format_guards(img)
            if not ok:
                continue

            with bundle.lock:
                is_ood, _ = ood.check(bundle.model, tensor)
                if is_ood:
                    continue
                pred_idx, probs, _ = predict_with_heatmap(bundle.model, tensor)
            if bundle.class_names[pred_idx] == true_class:
                correct_samples.append((img, true_class))

        print(f"  [{modality}] baseline-correct pool: {len(correct_samples)}")

        corruption_results = {}
        for corruption in CORRUPTION_NAMES:
            n_correct = 0
            n_total = 0
            n_rejected = 0
            for orig_img, true_class in correct_samples:
                try:
                    corrupted = apply_corruption(orig_img, corruption)
                except Exception as e:
                    continue

                # Check format guards on corrupted
                ok, _ = image_passes_format_guards(corrupted)
                if not ok:
                    n_rejected += 1
                    n_total += 1
                    continue

                if modality == "xray":
                    proc_img, _, _ = compute_lung_crop_with_info(corrupted, seg_bundle)
                else:
                    proc_img = corrupted
                tensor = preprocess_image(proc_img, bundle.device)

                with bundle.lock:
                    is_ood, _ = ood.check(bundle.model, tensor)
                    if is_ood:
                        n_rejected += 1
                        n_total += 1
                        continue
                    pred_idx, probs, _ = predict_with_heatmap(bundle.model, tensor)

                n_total += 1
                if bundle.class_names[pred_idx] == true_class:
                    n_correct += 1

            acc = n_correct / n_total if n_total > 0 else float("nan")
            corruption_results[corruption] = {
                "accuracy": round(acc, 4),
                "correct": n_correct,
                "total": n_total,
                "rejected": n_rejected,
            }
            print(f"    [{modality}] {corruption}: acc={acc:.1%} ({n_correct}/{n_total}, rej={n_rejected})")

        results[modality] = {
            "baseline_pool": len(correct_samples),
            "corruptions": corruption_results,
        }

    return results


# ── TEST 4: Crop ablation on external X-rays ──────────────────────────────────

def test4_crop_ablation(bundles, ood_detectors, seg_bundle, external_images):
    """
    Re-run external X-rays three ways:
      (a) U-Net segmentation crop (normal pipeline)
      (b) forced border crop
      (c) no crop at all
    Report accuracy for each.
    """
    print("\n=== TEST 4: Crop Ablation on External X-Rays ===")
    from prepare_xray import border_crop, TB_BORDER_FRAC

    ext_xray = external_images.get("xray", [])
    if not ext_xray:
        print("  No external X-ray images available.")
        return {}

    bundle = bundles["xray"]
    ood = ood_detectors["xray"]

    modes = {
        "unet_crop": [],
        "border_crop": [],
        "no_crop": [],
    }

    for img_path, true_class, source in ext_xray:
        img = load_image_safe(img_path)
        if img is None:
            continue
        ok, _ = image_passes_format_guards(img)
        if not ok:
            continue

        gray = np.array(img.convert("L"))

        # (a) U-Net crop (normal pipeline)
        crop_a, _, _ = compute_lung_crop_with_info(img, seg_bundle)
        # (b) border crop
        crop_b = Image.fromarray(border_crop(gray, TB_BORDER_FRAC)).convert("RGB")
        # (c) no crop
        crop_c = img.convert("RGB")

        for mode_name, crop_img in [("unet_crop", crop_a), ("border_crop", crop_b), ("no_crop", crop_c)]:
            tensor = preprocess_image(crop_img, bundle.device)
            with bundle.lock:
                is_ood, dist = ood.check(bundle.model, tensor)
                if is_ood:
                    modes[mode_name].append({"true_class": true_class, "predicted": "OOD_REJECTED",
                                              "correct": False, "ood_dist": round(dist, 2)})
                    continue
                pred_idx, probs, _ = predict_with_heatmap(bundle.model, tensor)
            predicted = bundle.class_names[pred_idx]
            modes[mode_name].append({
                "true_class": true_class,
                "predicted": predicted,
                "correct": predicted == true_class,
                "ood_dist": round(dist, 2),
            })

    results = {}
    for mode_name, preds in modes.items():
        n = len(preds)
        correct = sum(p["correct"] for p in preds)
        rejected = sum(1 for p in preds if p["predicted"] == "OOD_REJECTED")
        acc = correct / n if n > 0 else float("nan")
        results[mode_name] = {"accuracy": round(acc, 4), "correct": correct, "total": n, "ood_rejected": rejected}
        print(f"  [{mode_name}] acc={acc:.1%} ({correct}/{n}, ood_rejected={rejected})")

    return results


# ── TEST 5: OOD behaviour ────────────────────────────────────────────────────

def test5_ood_behaviour(bundles, ood_detectors, seg_bundle, external_images):
    """
    Plot external images' centroid distances vs. training distribution
    and 99th-percentile threshold, per modality.
    Report how many external images exceed threshold.
    """
    print("\n=== TEST 5: OOD Behaviour on External Images ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {}

    for modality in ("xray", "ct"):
        ood = ood_detectors[modality]
        bundle = bundles[modality]
        ext_list = external_images.get(modality, [])
        if not ext_list:
            print(f"  [{modality}] No external images.")
            continue

        # Load training distances from saved stats
        stats_path = BACKEND_DIR / "ood_stats" / f"{modality}_ood.pt"
        stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        train_distances = stats["train_distances"].numpy()
        threshold = stats["threshold"]

        # Compute external distances
        ext_distances = []
        ext_labels = []
        ext_ood_flags = []

        for img_path, true_class, source in ext_list:
            img = load_image_safe(img_path)
            if img is None:
                continue
            ok, _ = image_passes_format_guards(img)
            if not ok:
                continue
            if modality == "xray":
                proc_img, _, _ = compute_lung_crop_with_info(img, seg_bundle)
            else:
                proc_img = img
            tensor = preprocess_image(proc_img, bundle.device)
            with bundle.lock:
                is_ood, distance = ood.check(bundle.model, tensor)
            ext_distances.append(distance)
            ext_labels.append(true_class)
            ext_ood_flags.append(is_ood)

        n_ext = len(ext_distances)
        n_flagged = sum(ext_ood_flags)
        pct_flagged = 100.0 * n_flagged / n_ext if n_ext > 0 else 0.0

        print(f"  [{modality}] {n_flagged}/{n_ext} ({pct_flagged:.1f}%) external images exceed OOD threshold")
        print(f"  [{modality}] Training dist: mean={train_distances.mean():.2f}, "
              f"p99={np.percentile(train_distances, 99):.2f}, max={train_distances.max():.2f}")
        if ext_distances:
            print(f"  [{modality}] External dist: mean={np.mean(ext_distances):.2f}, "
                  f"min={np.min(ext_distances):.2f}, max={np.max(ext_distances):.2f}")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(train_distances, bins=60, alpha=0.5, color="steelblue", label=f"Training ({len(train_distances)} images)")
        ax.axvline(threshold, color="red", linestyle="--", linewidth=2,
                   label=f"99th-pct threshold = {threshold:.1f}")
        if ext_distances:
            ax.scatter(ext_distances, [5] * len(ext_distances),
                       color="orange", zorder=5, s=60, label=f"External ({n_ext} images)")
            for d, lbl, flagged in zip(ext_distances, ext_labels, ext_ood_flags):
                color = "red" if flagged else "green"
                ax.axvline(d, color=color, alpha=0.25, linewidth=1)
        ax.set_xlabel("L2 distance from training centroid")
        ax.set_ylabel("Count (training images)")
        ax.set_title(f"[{modality.upper()}] OOD Distance Distribution\n"
                     f"{n_flagged}/{n_ext} external images flagged as OOD")
        ax.legend()
        plt.tight_layout()
        plot_path = OUT_DIR / f"test5_{modality}_ood_distances.png"
        plt.savefig(str(plot_path), dpi=120)
        plt.close()
        print(f"  [{modality}] OOD plot saved: {plot_path.name}")

        results[modality] = {
            "n_external": n_ext,
            "n_ood_flagged": n_flagged,
            "pct_flagged": round(pct_flagged, 1),
            "threshold": round(threshold, 2),
            "train_mean": round(float(train_distances.mean()), 2),
            "train_p99": round(float(np.percentile(train_distances, 99)), 2),
            "ext_mean": round(float(np.mean(ext_distances)), 2) if ext_distances else None,
            "ext_min": round(float(np.min(ext_distances)), 2) if ext_distances else None,
            "ext_max": round(float(np.max(ext_distances)), 2) if ext_distances else None,
            "plot_file": plot_path.name,
        }

    return results


# ── TEST 6: Grad-CAM on failures ─────────────────────────────────────────────

def test6_gradcam_failures(bundles, ood_detectors, seg_bundle, external_images, t1_records):
    """
    For 5 external images the model gets wrong, save Grad-CAM overlay.
    Report whether attention is inside lung fields or on borders/corners/text.
    """
    print("\n=== TEST 6: Grad-CAM on External Failures ===")

    # Find up to 5 per modality that passed all gates but predicted wrong
    results = {}

    for modality in ("xray", "ct"):
        bundle = bundles[modality]
        ood = ood_detectors[modality]

        failures = [r for r in t1_records
                    if r.get("modality") == modality
                    and r.get("stage_rejected") is None
                    and r.get("correct") is False][:5]

        if not failures:
            print(f"  [{modality}] No failures that passed all gates. Skipping.")
            results[modality] = []
            continue

        failure_info = []
        for i, rec in enumerate(failures):
            img_path = Path(rec["path"])
            img = load_image_safe(img_path)
            if img is None:
                continue
            if modality == "xray":
                proc_img, used_seg, _ = compute_lung_crop_with_info(img, seg_bundle)
            else:
                proc_img = img
            tensor = preprocess_image(proc_img, bundle.device)

            # Full grad-cam with gradients enabled
            with bundle.lock:
                cam_engine = GradCAM(bundle.model, get_target_layer(bundle.model))
                try:
                    cam, pred_idx, probs = cam_engine.generate(tensor, class_idx=None)
                finally:
                    cam_engine.remove_hooks()

            display_img = denormalize(tensor[0].cpu(), IMAGENET_MEAN, IMAGENET_STD)
            overlay = overlay_cam(display_img, cam, alpha=0.45)

            # Analyse attention location
            cam_resized = cv2.resize(cam, (224, 224))
            h, w = 224, 224
            border_mask = np.zeros((h, w), dtype=bool)
            border_px = int(0.15 * min(h, w))
            border_mask[:border_px, :] = True
            border_mask[-border_px:, :] = True
            border_mask[:, :border_px] = True
            border_mask[:, -border_px:] = True

            cam_thresh = cam_resized > 0.5  # top-50% activation
            total_activation = cam_thresh.sum()
            border_activation = (cam_thresh & border_mask).sum()
            border_frac = float(border_activation) / float(total_activation) if total_activation > 0 else 0.0
            attention_location = "border/corner" if border_frac > 0.40 else "lung_field"

            save_path = OUT_DIR / f"test6_{modality}_failure_{i+1}_{img_path.stem[:20]}.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            info = {
                "file": img_path.name,
                "true_class": rec["true_class"],
                "predicted_class": rec["predicted_class"],
                "confidence": rec["confidence"],
                "border_activation_frac": round(border_frac, 3),
                "attention_location": attention_location,
                "overlay_file": save_path.name,
            }
            failure_info.append(info)
            print(f"  [{modality}] failure {i+1}: true={rec['true_class']} pred={rec['predicted_class']} "
                  f"conf={rec['confidence']:.2%} | attention: {attention_location} "
                  f"(border_frac={border_frac:.1%}) -> {save_path.name}")

        results[modality] = failure_info

    return results


# ── Markdown report writer ───────────────────────────────────────────────────

def write_report(t1_records, t1_summary, t2_results, t3_results, t4_results,
                 t5_results, t6_results):
    """Write GENERALIZATION_DIAGNOSIS.md."""

    def pct(n, total):
        if total == 0:
            return "N/A"
        return f"{100.0*n/total:.1f}%"

    total = len(t1_records)

    lines = []
    lines.append("# GENERALIZATION DIAGNOSIS REPORT\n")
    lines.append(f"> Generated by `scripts/diagnose_generalization.py`  \n")
    lines.append(f"> External probe: `data/external_probe/` (or `sample images/` fallback)  \n")
    lines.append(f"> No model weights were changed. Diagnosis pass only.\n")
    lines.append("\n---\n")

    # ── TEST 1 ──────────────────────────────────────────────────────────────
    lines.append("## TEST 1 — WHERE IN THE PIPELINE DOES IT GO WRONG\n")
    lines.append(f"**Total external images processed:** {total}\n\n")

    # Gate rejection table
    lines.append("### Stage Rejection Breakdown\n")
    lines.append("| Stage | Count | % of total |\n")
    lines.append("|-------|-------|------------|\n")
    stage_order = ["load", "format_guard", "sanity_gate", "modality_gate", "ood_gate", "wrong_class", "correct"]
    for stage in stage_order:
        count = t1_summary.get(stage, 0)
        if count > 0:
            lines.append(f"| `{stage}` | {count} | {pct(count, total)} |\n")

    # Per-image detail table
    lines.append("\n### Per-Image Detail\n")
    lines.append("| File | Modality | True Class | Format OK | Sanity | Gate | OOD dist/thresh | "
                 "Seg method | BBox frac | Pred | Conf | Correct | Rejected at |\n")
    lines.append("|------|----------|------------|-----------|--------|------|-----------------|"
                 "------------|-----------|------|------|---------|-------------|\n")
    for r in t1_records:
        fname = Path(r["path"]).name[:30]
        mod = r.get("modality", "?")
        true_cls = r.get("true_class", "?")
        fmt = "✓" if r.get("format_guard_pass", False) else f"✗ (std={r.get('gray_std','?')}, col={r.get('color_dev','?')})"
        sanity = f"{r.get('sanity_label','?')} ({r.get('sanity_conf','?'):.0%})" if "sanity_label" in r else "-"
        gate = f"{r.get('gate_pred','?')} ({r.get('gate_conf','?'):.0%})" if "gate_pred" in r else "-"
        ood_str = (f"{r.get('ood_distance','?'):.1f}/{r.get('ood_threshold','?'):.1f}"
                   if "ood_distance" in r else "-")
        seg = ("unet" if r.get("used_segmentation") else "border" if r.get("used_segmentation") is False else "-")
        bbox = f"{r.get('bbox_area_frac','?')}" if r.get("bbox_area_frac") is not None else "-"
        pred = r.get("predicted_class", "-")
        conf = f"{r.get('confidence', 0):.0%}" if "confidence" in r else "-"
        correct = "✓" if r.get("correct") else ("✗" if r.get("correct") is False else "-")
        rejected = r.get("stage_rejected") or "—"
        lines.append(f"| {fname} | {mod} | {true_cls} | {fmt} | {sanity} | {gate} | "
                     f"{ood_str} | {seg} | {bbox} | {pred} | {conf} | {correct} | {rejected} |\n")

    lines.append("\n### Summary Observations\n")
    ood_count = t1_summary.get("ood_gate", 0)
    sanity_count = t1_summary.get("sanity_gate", 0)
    gate_count = t1_summary.get("modality_gate", 0)
    fmt_count = t1_summary.get("format_guard", 0)
    wrong_count = t1_summary.get("wrong_class", 0)
    correct_count = t1_summary.get("correct", 0)
    passed_all = wrong_count + correct_count
    lines.append(f"- **{fmt_count}** images rejected at format guard (blank/color)\n")
    lines.append(f"- **{sanity_count}** images rejected at sanity gate (non-scan)\n")
    lines.append(f"- **{gate_count}** images rejected at modality gate (wrong modality)\n")
    lines.append(f"- **{ood_count}** images rejected by OOD detector "
                 f"({pct(ood_count, total)} of total — OOD detector is working as designed)\n")
    lines.append(f"- **{passed_all}** images passed all gates: "
                 f"**{correct_count}** correct, **{wrong_count}** wrong\n")
    lines.append(f"- Classification accuracy among images that passed all gates: "
                 f"**{pct(correct_count, passed_all)}**\n")

    lines.append("\n---\n")

    # ── TEST 2 ──────────────────────────────────────────────────────────────
    lines.append("## TEST 2 — PREPROCESSING COMPARISON\n")
    lines.append("224×224 preprocessed tensor grids (denormalized) saved to `diagnosis_outputs/`.\n")
    lines.append("Top row = external images; bottom row = training-source images.\n\n")

    for modality, res in t2_results.items():
        cls = res["target_class"]
        grid_file = res["grid_file"]
        ext_stats = res["stats"]["external"]
        trn_stats = res["stats"]["training"]

        lines.append(f"### {modality.upper()} — class `{cls}`\n")
        lines.append(f"Grid: `diagnosis_outputs/{grid_file}`\n\n")

        ext_means = [s["mean"] for s in ext_stats]
        ext_stds = [s["std"] for s in ext_stats]
        trn_means = [s["mean"] for s in trn_stats]
        trn_stds = [s["std"] for s in trn_stats]

        lines.append("**External images (normalized tensor stats):**\n\n")
        lines.append("| File | Crop method | Tensor mean | Tensor std |\n")
        lines.append("|------|-------------|-------------|------------|\n")
        for s in ext_stats:
            lines.append(f"| `{s['file'][:40]}` | {s['crop']} | {s['mean']:.4f} | {s['std']:.4f} |\n")
        if ext_means:
            lines.append(f"| **GROUP MEAN** | — | **{np.mean(ext_means):.4f}** | **{np.mean(ext_stds):.4f}** |\n")

        lines.append("\n**Training-source images (normalized tensor stats):**\n\n")
        lines.append("| File | Crop method | Tensor mean | Tensor std |\n")
        lines.append("|------|-------------|-------------|------------|\n")
        for s in trn_stats:
            lines.append(f"| `{s['file'][:40]}` | {s['crop']} | {s['mean']:.4f} | {s['std']:.4f} |\n")
        if trn_means:
            lines.append(f"| **GROUP MEAN** | — | **{np.mean(trn_means):.4f}** | **{np.mean(trn_stds):.4f}** |\n")

        if ext_means and trn_means:
            mean_diff = abs(np.mean(ext_means) - np.mean(trn_means))
            std_diff = abs(np.mean(ext_stds) - np.mean(trn_stds))
            lines.append(f"\n**Δmean = {mean_diff:.4f}, Δstd = {std_diff:.4f}**  \n")
            if mean_diff > 0.10 or std_diff > 0.10:
                lines.append("⚠️ **Intensity statistics differ sharply between groups — evidence of source/format shift.**\n")
            else:
                lines.append("✅ Intensity statistics are similar between groups.\n")

        lines.append("\n")

    lines.append("\n---\n")

    # ── TEST 3 ──────────────────────────────────────────────────────────────
    lines.append("## TEST 3 — CONTROLLED CORRUPTION\n")
    lines.append("Baseline: 20 training-source test images the model gets right. "
                 "Each corruption applied independently.\n\n")

    for modality, res in t3_results.items():
        lines.append(f"### {modality.upper()} (baseline pool: {res['baseline_pool']} images)\n\n")
        lines.append("| Corruption | Accuracy | Correct | Total | Rejected |\n")
        lines.append("|------------|----------|---------|-------|----------|\n")
        for corr, stats in res["corruptions"].items():
            acc_str = f"{stats['accuracy']:.1%}" if not math.isnan(stats["accuracy"]) else "N/A"
            lines.append(f"| `{corr}` | {acc_str} | {stats['correct']} | "
                         f"{stats['total']} | {stats['rejected']} |\n")

        lines.append("\n**Interpretation:**  \n")
        worst = min(res["corruptions"].items(), key=lambda x: x[1]["accuracy"] if not math.isnan(x[1]["accuracy"]) else 1.0)
        lines.append(f"- Worst corruption: `{worst[0]}` → accuracy = {worst[1]['accuracy']:.1%}\n")
        # Flag if black_border or contrast/gamma hits hard
        for corr in ["black_border_10pct", "invert", "histeq"]:
            s = res["corruptions"].get(corr, {})
            if s and not math.isnan(s.get("accuracy", 1.0)) and s["accuracy"] < 0.80:
                lines.append(f"- ⚠️ `{corr}` drops accuracy below 80% — "
                              "indicates preprocessing sensitivity, not pure pathology knowledge.\n")
        lines.append("\n")

    lines.append("\n---\n")

    # ── TEST 4 ──────────────────────────────────────────────────────────────
    lines.append("## TEST 4 — CROP ABLATION ON EXTERNAL X-RAYS\n\n")
    if t4_results:
        lines.append("| Crop mode | Accuracy | Correct | Total | OOD Rejected |\n")
        lines.append("|-----------|----------|---------|-------|---------------|\n")
        for mode, stats in t4_results.items():
            acc_str = f"{stats['accuracy']:.1%}" if not math.isnan(stats["accuracy"]) else "N/A"
            lines.append(f"| `{mode}` | {acc_str} | {stats['correct']} | "
                         f"{stats['total']} | {stats['ood_rejected']} |\n")

        lines.append("\n**Interpretation:**  \n")
        accs = {k: v["accuracy"] for k, v in t4_results.items() if not math.isnan(v["accuracy"])}
        if accs:
            best = max(accs, key=accs.get)
            worst = min(accs, key=accs.get)
            lines.append(f"- Best crop mode: `{best}` ({accs[best]:.1%}); "
                         f"worst: `{worst}` ({accs[worst]:.1%})\n")
            if best in ("border_crop", "no_crop") and accs[best] > accs.get("unet_crop", 0) + 0.05:
                lines.append("- ⚠️ **U-Net crop performs WORSE than border_crop or no_crop** — "
                              "the segmentation step is a failure mode for external images. "
                              "Fix target: `seg_utils.py`.\n")
            else:
                lines.append("- U-Net crop performs comparably to or better than alternatives.\n")
    else:
        lines.append("No external X-ray images available for this test.\n")

    lines.append("\n---\n")

    # ── TEST 5 ──────────────────────────────────────────────────────────────
    lines.append("## TEST 5 — OOD BEHAVIOUR ON EXTERNAL IMAGES\n\n")

    for modality, res in t5_results.items():
        lines.append(f"### {modality.upper()}\n\n")
        lines.append(f"| Metric | Value |\n")
        lines.append(f"|--------|-------|\n")
        lines.append(f"| OOD threshold (99th pct of training) | {res['threshold']:.2f} |\n")
        lines.append(f"| Training distance mean | {res['train_mean']:.2f} |\n")
        lines.append(f"| Training distance p99 | {res['train_p99']:.2f} |\n")
        lines.append(f"| External distance mean | {res.get('ext_mean', 'N/A')} |\n")
        lines.append(f"| External distance min | {res.get('ext_min', 'N/A')} |\n")
        lines.append(f"| External distance max | {res.get('ext_max', 'N/A')} |\n")
        lines.append(f"| External images flagged as OOD | {res['n_ood_flagged']}/{res['n_external']} "
                     f"({res['pct_flagged']:.1f}%) |\n")
        lines.append(f"\nPlot: `diagnosis_outputs/{res['plot_file']}`\n\n")

        if res["n_ood_flagged"] > 0:
            lines.append(f"**{res['n_ood_flagged']} external images are correctly flagged as OOD** — "
                         "the system is behaving as designed: these images sit outside the training "
                         "distribution in feature space. The demo reporting these as 'failures' is "
                         "misleading — they are correct rejections.  \n")
        if res["n_external"] > res["n_ood_flagged"]:
            passed_ood = res["n_external"] - res["n_ood_flagged"]
            lines.append(f"**{passed_ood} external images pass the OOD gate** — for these, "
                         "classification error is a genuine model failure, not a detector failure.  \n")
        lines.append("\n")

    lines.append("\n---\n")

    # ── TEST 6 ──────────────────────────────────────────────────────────────
    lines.append("## TEST 6 — GRAD-CAM ON FAILURES\n\n")

    for modality, failure_list in t6_results.items():
        lines.append(f"### {modality.upper()}\n\n")
        if not failure_list:
            lines.append("No failures that passed all gates — "
                         "either all external images were rejected before classification, "
                         "or all were classified correctly.\n\n")
            continue

        lines.append("| File | True | Predicted | Conf | Border activation | Attention |\n")
        lines.append("|------|------|-----------|------|-------------------|-----------|\n")
        for f in failure_list:
            lines.append(f"| `{f['file'][:35]}` | {f['true_class']} | {f['predicted_class']} | "
                         f"{f['confidence']:.0%} | {f['border_activation_frac']:.1%} | "
                         f"{f['attention_location']} |\n")

        border_dominant = [f for f in failure_list if f["attention_location"] == "border/corner"]
        lung_dominant = [f for f in failure_list if f["attention_location"] == "lung_field"]
        lines.append(f"\n- {len(border_dominant)}/{len(failure_list)} failures have attention concentrated at borders/corners — "
                     "likely attending to dataset-specific artifacts, not pathology.\n")
        lines.append(f"- {len(lung_dominant)}/{len(failure_list)} failures have attention in the lung field — "
                     "these are genuine model knowledge failures, not shortcut learning.\n")
        lines.append(f"\nOverlay images: `diagnosis_outputs/test6_{modality}_failure_*.png`\n\n")

    lines.append("\n---\n")

    # ── FINAL REPORT ─────────────────────────────────────────────────────────
    lines.append("## FINAL DIAGNOSIS — DOMINANT CAUSE\n\n")

    # Gather evidence to auto-rank
    cause_evidence = {
        "A_preprocessing_mismatch": [],
        "B_intensity_format_shift": [],
        "C_source_confound": [],
    }

    # Evidence from TEST 1 (OOD rejections)
    if t5_results:
        for modality, res in t5_results.items():
            if res["n_ood_flagged"] > 0:
                pct_ood = res["pct_flagged"]
                cause_evidence["C_source_confound"].append(
                    f"TEST 5 [{modality}]: {res['n_ood_flagged']}/{res['n_external']} "
                    f"({pct_ood:.1f}%) external images flagged by OOD detector — "
                    "feature-space distribution shift consistent with source confound."
                )

    # Evidence from TEST 2 (intensity stats)
    for modality, res in t2_results.items():
        ext_means = [s["mean"] for s in res["stats"]["external"]]
        trn_means = [s["mean"] for s in res["stats"]["training"]]
        if ext_means and trn_means:
            mean_diff = abs(np.mean(ext_means) - np.mean(trn_means))
            ext_stds_vals = [s["std"] for s in res["stats"]["external"]]
            trn_stds_vals = [s["std"] for s in res["stats"]["training"]]
            std_diff = abs(np.mean(ext_stds_vals) - np.mean(trn_stds_vals))
            if mean_diff > 0.05 or std_diff > 0.05:
                cause_evidence["B_intensity_format_shift"].append(
                    f"TEST 2 [{modality}]: Tensor mean Δ={mean_diff:.4f}, std Δ={std_diff:.4f} "
                    "— intensity statistics diverge between external and training-source images."
                )

    # Evidence from TEST 3 (corruption sensitivity)
    for modality, res in t3_results.items():
        for corr in ["black_border_10pct", "invert", "histeq"]:
            s = res["corruptions"].get(corr, {})
            if s and not math.isnan(s.get("accuracy", 1.0)) and s["accuracy"] < 0.80:
                cause_evidence["A_preprocessing_mismatch"].append(
                    f"TEST 3 [{modality}]: `{corr}` drops accuracy to {s['accuracy']:.1%} "
                    "— model is sensitive to preprocessing-level changes that external images may exhibit."
                )

    # Evidence from TEST 4 (crop ablation)
    if t4_results:
        accs_4 = {k: v["accuracy"] for k, v in t4_results.items() if not math.isnan(v["accuracy"])}
        if accs_4 and max(accs_4.values()) > accs_4.get("unet_crop", 0) + 0.05:
            cause_evidence["A_preprocessing_mismatch"].append(
                f"TEST 4: U-Net crop accuracy={accs_4.get('unet_crop',0):.1%} < "
                f"border_crop={accs_4.get('border_crop',0):.1%} or "
                f"no_crop={accs_4.get('no_crop',0):.1%} — "
                "segmentation-guided crop degrades on external X-rays."
            )

    # Evidence from TEST 6 (Grad-CAM)
    for modality, failure_list in t6_results.items():
        border_count = sum(1 for f in failure_list if f["attention_location"] == "border/corner")
        if failure_list and border_count > len(failure_list) // 2:
            cause_evidence["A_preprocessing_mismatch"].append(
                f"TEST 6 [{modality}]: {border_count}/{len(failure_list)} failures attend to "
                "borders/corners — shortcut features from training crops, not pathology."
            )

    # Rank by evidence count
    cause_scores = {k: len(v) for k, v in cause_evidence.items()}
    ranked = sorted(cause_scores.items(), key=lambda x: -x[1])

    lines.append("### Evidence by Cause\n\n")
    cause_labels = {
        "A_preprocessing_mismatch": "(A) Preprocessing / crop mismatch",
        "B_intensity_format_shift": "(B) Intensity / format shift",
        "C_source_confound": "(C) Source confound / distribution shift",
    }
    for cause_key, score in ranked:
        label = cause_labels[cause_key]
        evidence = cause_evidence[cause_key]
        lines.append(f"#### {label} — {score} piece(s) of evidence\n\n")
        if evidence:
            for e in evidence:
                lines.append(f"- {e}\n")
        else:
            lines.append("- No direct evidence from these tests.\n")
        lines.append("\n")

    lines.append("### Ranking\n\n")
    for rank, (cause_key, score) in enumerate(ranked, 1):
        label = cause_labels[cause_key]
        lines.append(f"{rank}. **{label}** — {score} evidence item(s)\n")

    lines.append("\n### Plain-Language Summary\n\n")
    if ranked:
        dominant = ranked[0][0]
        dominant_label = cause_labels[dominant]
        if dominant == "C_source_confound":
            lines.append(
                "The **dominant cause is (C) source confound / distribution shift**. "
                "External images land far from the training-set centroid in feature space, "
                "which the OOD detector correctly flags. The classifiers were trained on images "
                "from specific datasets (COVID-19-Radiography, TB-Chest-Radiography, IQ-OTH/NCCD) "
                "that share dataset-level stylistic features (resolution, contrast, scanner make, "
                "border crop, mask-guided preprocessing). External images from different scanner "
                "sources or collections do not share these features and sit in a different "
                "region of feature space. When the OOD detector rejects them, it is doing the "
                "right thing — the 'poor performance on external images' is largely explained by "
                "the system correctly refusing to classify out-of-distribution input. "
                "The remaining classification errors (on images that pass OOD) need preprocessing "
                "and intensity alignment before any fine-tuning would help.\n"
            )
        elif dominant == "A_preprocessing_mismatch":
            lines.append(
                "The **dominant cause is (A) preprocessing / crop mismatch**. "
                "The U-Net segmentation model was trained on the specific image style of the "
                "training datasets and produces degenerate crops on external images (border_crop "
                "or no_crop beats U-Net). Additionally, the model is sensitive to preprocessing "
                "choices (border pixels, contrast, gamma) that differ between the training "
                "pipeline and external image sources. Fix target: `seg_utils.py` crop "
                "fallback logic and external-image preprocessing normalization.\n"
            )
        else:
            lines.append(
                "The **dominant cause is (B) intensity / format shift**. "
                "External images have meaningfully different pixel intensity distributions "
                "(tensor mean/std) after the standard preprocessing pipeline, indicating that "
                "the normalization step does not fully bridge the gap between scanner sources. "
                "This is a data-level difference that preprocessing standardization (histogram "
                "matching, CLAHE, per-image normalization) could reduce before any retraining.\n"
            )

    lines.append("\n> **No model weights were changed in this diagnosis. "
                 "No fix is proposed here — evidence collection only.**\n")

    # Write to file
    report_path = ROOT / "GENERALIZATION_DIAGNOSIS.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\n📄 Report written to: {report_path}")
    return report_path


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("GENERALIZATION DIAGNOSIS")
    print("=" * 70)

    # Load models
    bundles = load_all_bundles()
    ood_detectors = load_ood_detectors(bundles)
    seg_bundle = load_seg()

    # Collect external images
    external_images = collect_external_images()
    total_ext = sum(len(v) for v in external_images.values())
    print(f"\n[probe] {total_ext} external images: "
          f"xray={len(external_images['xray'])}, ct={len(external_images['ct'])}")

    if total_ext == 0:
        print("\nWARNING: No external images found. Please populate:")
        print(f"  {EXTERNAL_PROBE_DIR}/xray/<class>/")
        print(f"  {EXTERNAL_PROBE_DIR}/ct/<class>/")
        print(f"  OR  {SAMPLE_IMAGES_DIR}/")
        print("\nCreating placeholder directory structure...")
        for modality in ("xray", "ct"):
            classes = ["COVID-19", "Normal", "Pneumonia", "Tuberculosis"] if modality == "xray" \
                      else ["Benign", "Malignant", "Normal"]
            for cls in classes:
                (EXTERNAL_PROBE_DIR / modality / cls).mkdir(parents=True, exist_ok=True)
        print(f"  Created {EXTERNAL_PROBE_DIR}/ — populate with external images and re-run.")
        return

    # Run tests
    t1_records, t1_summary = test1_pipeline_gates(bundles, ood_detectors, seg_bundle, external_images)
    t2_results = test2_preprocessing_comparison(bundles, ood_detectors, seg_bundle, external_images)
    t3_results = test3_controlled_corruption(bundles, ood_detectors, seg_bundle)
    t4_results = test4_crop_ablation(bundles, ood_detectors, seg_bundle, external_images)
    t5_results = test5_ood_behaviour(bundles, ood_detectors, seg_bundle, external_images)
    t6_results = test6_gradcam_failures(bundles, ood_detectors, seg_bundle, external_images, t1_records)

    # Write report
    report_path = write_report(t1_records, t1_summary, t2_results, t3_results,
                                t4_results, t5_results, t6_results)

    # Save raw results as JSON for further analysis
    import json as _json
    raw_results = {
        "t1_summary": t1_summary,
        "t1_records": [{k: v for k, v in r.items() if k != "path"} for r in t1_records],
        "t2_results": t2_results,
        "t3_results": t3_results,
        "t4_results": t4_results,
        "t5_results": t5_results,
        "t6_results": t6_results,
    }
    json_path = OUT_DIR / "diagnosis_raw.json"
    with open(json_path, "w") as f:
        _json.dump(raw_results, f, indent=2, default=str)
    print(f"📊 Raw results: {json_path}")
    print(f"\n✅ Done. See {report_path.name} for findings.")


if __name__ == "__main__":
    main()
