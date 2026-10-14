#!/usr/bin/env python3
"""
trace_breaking.py
Traces breaking corruption transforms (costing 20+ points) through the real pipeline:
1. X-ray: intensity inverted (-76.5 points)
2. CT: histogram equalized (-35.7 points)
3. CT: intensity inverted (-21.4 points)

Saves 224x224 side-by-side preprocessed tensor images under test/results/.
"""

import os
import sys
import random
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
ML_DIR = ROOT_DIR / "ml"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
DATA_DIR = ROOT_DIR / "data"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(ML_DIR))

from models import build_model
from ood_detector import OODDetector
from gradcam_utils import preprocess_image
from gradcam import denormalize
from dataset import IMAGENET_MEAN, IMAGENET_STD
import seg_utils
from seg_model import UNet

# Load checkpoints helper without slow reproduction checks
def load_raw_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    class_names = ckpt.get("class_names", [])
    backbone = ckpt.get("backbone", "densenet121")
    model = build_model(backbone, len(class_names), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, class_names

def load_seg_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = UNet(in_channels=1, out_channels=1, base=32).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading models on device={device}...")

xray_model, xray_classes = load_raw_model(CHECKPOINTS_DIR / "xray_best_model_phase35_v2.pt", device)
ct_model, ct_classes = load_raw_model(CHECKPOINTS_DIR / "ct_best_model_phase35_v2.pt", device)
gate_model, gate_classes = load_raw_model(CHECKPOINTS_DIR / "modality_gate_best_model.pt", device)
xray_sanity, xray_sanity_classes = load_raw_model(CHECKPOINTS_DIR / "xray_sanity_best.pt", device)
ct_sanity, ct_sanity_classes = load_raw_model(CHECKPOINTS_DIR / "ct_sanity_best.pt", device)
seg_model = load_seg_model(CHECKPOINTS_DIR / "xray_seg_model.pt", device)

xray_ood = OODDetector("xray", device)
ct_ood = OODDetector("ct", device)

# Transforms
def t_invert(im):
    return ImageOps.invert(im.convert("L")).convert(im.mode)

def t_equalize(im):
    return ImageOps.equalize(im.convert("L")).convert(im.mode)

# Detailed lung crop trace
def trace_lung_crop(image: Image.Image):
    x = seg_utils._to_seg_tensor(image, device)
    with torch.no_grad():
        logits = seg_model(x)
        probs = torch.sigmoid(logits)
        mask_small = (probs[0, 0] > 0.5).cpu().numpy().astype(np.uint8)

    gray = np.array(image.convert("L"))
    if mask_small.max() == 0:
        cropped = seg_utils.border_crop(gray, seg_utils.TB_BORDER_FRAC)
        return Image.fromarray(cropped), False, 0.0, "empty mask -> fallback border_crop"

    r0, r1, c0, c1 = seg_utils.mask_bbox(mask_small, pad=seg_utils.MASK_PAD_PX)
    bbox_area_frac = ((r1 - r0) * (c1 - c0)) / (seg_utils.SEG_INPUT_SIZE * seg_utils.SEG_INPUT_SIZE)

    if bbox_area_frac < seg_utils.MIN_BBOX_AREA_FRAC:
        cropped = seg_utils.border_crop(gray, seg_utils.TB_BORDER_FRAC)
        return Image.fromarray(cropped), False, bbox_area_frac, f"bbox_area_frac ({bbox_area_frac:.4f}) < {seg_utils.MIN_BBOX_AREA_FRAC} -> fallback border_crop"
    elif bbox_area_frac > seg_utils.MAX_BBOX_AREA_FRAC:
        cropped = seg_utils.border_crop(gray, seg_utils.TB_BORDER_FRAC)
        return Image.fromarray(cropped), False, bbox_area_frac, f"bbox_area_frac ({bbox_area_frac:.4f}) > {seg_utils.MAX_BBOX_AREA_FRAC} -> fallback border_crop"

    H, W = gray.shape
    scale_r, scale_c = H / seg_utils.SEG_INPUT_SIZE, W / seg_utils.SEG_INPUT_SIZE
    orig_r0, orig_r1 = int(r0 * scale_r), int(r1 * scale_r)
    orig_c0, orig_c1 = int(c0 * scale_c), int(c1 * scale_c)
    cropped = gray[orig_r0:orig_r1, orig_c0:orig_c1]
    return Image.fromarray(cropped), True, bbox_area_frac, f"U-Net crop accepted (bbox_area_frac={bbox_area_frac:.4f})"

def trace_pipeline(modality: str, pil_image: Image.Image):
    trace = {}
    
    # 1. Lung crop (xray only)
    if modality == "xray":
        cropped_img, used_unet, bbox_frac, branch_info = trace_lung_crop(pil_image)
        trace["crop"] = {
            "used_unet": used_unet,
            "bbox_area_frac": bbox_frac,
            "branch": branch_info,
        }
    else:
        cropped_img = pil_image
        trace["crop"] = None

    # 2. Preprocess tensor
    input_tensor = preprocess_image(cropped_img, device)
    trace["tensor"] = input_tensor

    # 3. Sanity gate
    sanity_m = xray_sanity if modality == "xray" else ct_sanity
    sanity_cls = xray_sanity_classes if modality == "xray" else ct_sanity_classes
    with torch.no_grad():
        s_logits = sanity_m(input_tensor)
        s_probs = F.softmax(s_logits, dim=1)[0].cpu().numpy()
        s_pred_idx = int(s_logits.argmax(dim=1).item())
        s_pred = sanity_cls[s_pred_idx]
        s_conf = float(s_probs[s_pred_idx])
    trace["sanity"] = {
        "pred": s_pred,
        "conf": s_conf,
        "passed": (s_pred == "scan"),
        "probs": {c: float(p) for c, p in zip(sanity_cls, s_probs)}
    }

    # 4. Modality gate
    with torch.no_grad():
        g_logits = gate_model(input_tensor)
        g_probs = F.softmax(g_logits, dim=1)[0].cpu().numpy()
        g_pred_idx = int(g_logits.argmax(dim=1).item())
        g_pred = gate_classes[g_pred_idx]
        g_conf = float(g_probs[g_pred_idx])
    trace["modality_gate"] = {
        "pred": g_pred,
        "conf": g_conf,
        "passed": (g_pred == modality),
        "probs": {c: float(p) for c, p in zip(gate_classes, g_probs)}
    }

    # 5. OOD check
    ood_det = xray_ood if modality == "xray" else ct_ood
    disease_m = xray_model if modality == "xray" else ct_model
    is_ood, dist = ood_det.check(disease_m, input_tensor)
    trace["ood"] = {
        "distance": dist,
        "threshold": ood_det.threshold,
        "passed": not is_ood,
    }

    # 6. Disease classifier
    disease_cls = xray_classes if modality == "xray" else ct_classes
    with torch.no_grad():
        d_logits = disease_m(input_tensor)
        d_probs = F.softmax(d_logits, dim=1)[0].cpu().numpy()
        d_pred_idx = int(d_logits.argmax(dim=1).item())
        d_pred = disease_cls[d_pred_idx]
        d_conf = float(d_probs[d_pred_idx])
    trace["disease"] = {
        "pred": d_pred,
        "conf": d_conf,
        "probs": {c: float(p) for c, p in zip(disease_cls, d_probs)}
    }

    return trace

def create_side_by_side(orig_tensor, trans_tensor, title_left, title_right, save_path):
    orig_np = denormalize(orig_tensor[0].cpu(), IMAGENET_MEAN, IMAGENET_STD)
    trans_np = denormalize(trans_tensor[0].cpu(), IMAGENET_MEAN, IMAGENET_STD)

    im_orig = Image.fromarray(orig_np)
    im_trans = Image.fromarray(trans_np)

    # 224x224 each -> side by side with banner
    W, H = 224, 224
    banner_h = 36
    combined = Image.new("RGB", (W * 2 + 20, H + banner_h), color=(240, 240, 240))
    draw = ImageDraw.Draw(combined)

    combined.paste(im_orig, (0, banner_h))
    combined.paste(im_trans, (W + 20, banner_h))

    draw.text((10, 8), title_left, fill=(20, 20, 20))
    draw.text((W + 30, 8), title_right, fill=(20, 20, 20))

    combined.save(save_path)
    print(f"Saved side-by-side image: {save_path}")

def main():
    out_dir = ROOT_DIR / "test" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("TRACING BREAKING TRANSFORMS")
    print("="*70)

    # -------------------------------------------------------------
    # Case 1: X-ray, intensity inverted
    # -------------------------------------------------------------
    # Baseline X-ray images (seed 0)
    random.seed(0)
    from scripts.test_corruption import load_labelled
    xray_samples = load_labelled(str(DATA_DIR / "xray" / "test"), per_class=5)
    
    # Find flipped images under intensity inverted
    flipped_xray_invert = []
    for path, label in xray_samples:
        with Image.open(path) as im:
            tr_orig = trace_pipeline("xray", im)
            if tr_orig["disease"]["pred"].lower() != label.lower():
                continue
            tr_trans = trace_pipeline("xray", t_invert(im))
            if tr_trans["disease"]["pred"].lower() != label.lower():
                flipped_xray_invert.append((path, label, tr_orig, tr_trans))

    print(f"\nX-ray intensity inverted: found {len(flipped_xray_invert)} flipped images.")
    # Pick one specific image to report
    chosen_path, label, tr_orig, tr_trans = flipped_xray_invert[0]
    filename = Path(chosen_path).name
    print(f"\nTracing X-ray image: {filename} (True class: {label})")
    print(f"--- Unaltered Original ---")
    print(f"  Crop: used_unet={tr_orig['crop']['used_unet']}, bbox_frac={tr_orig['crop']['bbox_area_frac']:.4f}, branch={tr_orig['crop']['branch']}")
    print(f"  Sanity gate: passed={tr_orig['sanity']['passed']}, pred={tr_orig['sanity']['pred']}, conf={tr_orig['sanity']['conf']:.4f}")
    print(f"  Modality gate: passed={tr_orig['modality_gate']['passed']}, pred={tr_orig['modality_gate']['pred']}, conf={tr_orig['modality_gate']['conf']:.4f}")
    print(f"  OOD: dist={tr_orig['ood']['distance']:.2f}, thresh={tr_orig['ood']['threshold']:.2f}, passed={tr_orig['ood']['passed']}")
    print(f"  Final disease: pred={tr_orig['disease']['pred']}, conf={tr_orig['disease']['conf']:.4f}")

    print(f"--- Transformed (intensity inverted) ---")
    print(f"  Crop: used_unet={tr_trans['crop']['used_unet']}, bbox_frac={tr_trans['crop']['bbox_area_frac']:.4f}, branch={tr_trans['crop']['branch']}")
    print(f"  Sanity gate: passed={tr_trans['sanity']['passed']}, pred={tr_trans['sanity']['pred']}, conf={tr_trans['sanity']['conf']:.4f}")
    print(f"  Modality gate: passed={tr_trans['modality_gate']['passed']}, pred={tr_trans['modality_gate']['pred']}, conf={tr_trans['modality_gate']['conf']:.4f}")
    print(f"  OOD: dist={tr_trans['ood']['distance']:.2f}, thresh={tr_trans['ood']['threshold']:.2f}, passed={tr_trans['ood']['passed']}")
    print(f"  Final disease: pred={tr_trans['disease']['pred']}, conf={tr_trans['disease']['conf']:.4f}")

    create_side_by_side(
        tr_orig["tensor"], tr_trans["tensor"],
        f"Original ({tr_orig['disease']['pred']} {tr_orig['disease']['conf']:.1%})",
        f"Inverted ({tr_trans['disease']['pred']} {tr_trans['disease']['conf']:.1%})",
        out_dir / "trace_xray_intensity_inverted.png"
    )

    # -------------------------------------------------------------
    # Case 2: CT, histogram equalized
    # -------------------------------------------------------------
    random.seed(0)
    ct_samples = load_labelled(str(DATA_DIR / "ct_scan" / "test"), per_class=5)
    flipped_ct_eq = []
    for path, label in ct_samples:
        with Image.open(path) as im:
            tr_orig = trace_pipeline("ct", im)
            if tr_orig["disease"]["pred"].lower() != label.lower():
                continue
            tr_trans = trace_pipeline("ct", t_equalize(im))
            if tr_trans["disease"]["pred"].lower() != label.lower():
                flipped_ct_eq.append((path, label, tr_orig, tr_trans))

    print(f"\nCT histogram equalized: found {len(flipped_ct_eq)} flipped images.")
    chosen_path, label, tr_orig, tr_trans = flipped_ct_eq[0]
    filename = Path(chosen_path).name
    print(f"\nTracing CT image: {filename} (True class: {label})")
    print(f"--- Unaltered Original ---")
    print(f"  Sanity gate: passed={tr_orig['sanity']['passed']}, pred={tr_orig['sanity']['pred']}, conf={tr_orig['sanity']['conf']:.4f}")
    print(f"  Modality gate: passed={tr_orig['modality_gate']['passed']}, pred={tr_orig['modality_gate']['pred']}, conf={tr_orig['modality_gate']['conf']:.4f}")
    print(f"  OOD: dist={tr_orig['ood']['distance']:.2f}, thresh={tr_orig['ood']['threshold']:.2f}, passed={tr_orig['ood']['passed']}")
    print(f"  Final disease: pred={tr_orig['disease']['pred']}, conf={tr_orig['disease']['conf']:.4f}")

    print(f"--- Transformed (histogram equalized) ---")
    print(f"  Sanity gate: passed={tr_trans['sanity']['passed']}, pred={tr_trans['sanity']['pred']}, conf={tr_trans['sanity']['conf']:.4f}")
    print(f"  Modality gate: passed={tr_trans['modality_gate']['passed']}, pred={tr_trans['modality_gate']['pred']}, conf={tr_trans['modality_gate']['conf']:.4f}")
    print(f"  OOD: dist={tr_trans['ood']['distance']:.2f}, thresh={tr_trans['ood']['threshold']:.2f}, passed={tr_trans['ood']['passed']}")
    print(f"  Final disease: pred={tr_trans['disease']['pred']}, conf={tr_trans['disease']['conf']:.4f}")

    create_side_by_side(
        tr_orig["tensor"], tr_trans["tensor"],
        f"Original ({tr_orig['disease']['pred']} {tr_orig['disease']['conf']:.1%})",
        f"HistEq ({tr_trans['disease']['pred']} {tr_trans['disease']['conf']:.1%})",
        out_dir / "trace_ct_histogram_equalized.png"
    )

    # -------------------------------------------------------------
    # Case 3: CT, intensity inverted
    # -------------------------------------------------------------
    flipped_ct_invert = []
    for path, label in ct_samples:
        with Image.open(path) as im:
            tr_orig = trace_pipeline("ct", im)
            if tr_orig["disease"]["pred"].lower() != label.lower():
                continue
            tr_trans = trace_pipeline("ct", t_invert(im))
            if tr_trans["disease"]["pred"].lower() != label.lower():
                flipped_ct_invert.append((path, label, tr_orig, tr_trans))

    print(f"\nCT intensity inverted: found {len(flipped_ct_invert)} flipped images.")
    chosen_path, label, tr_orig, tr_trans = flipped_ct_invert[0]
    filename = Path(chosen_path).name
    print(f"\nTracing CT image: {filename} (True class: {label})")
    print(f"--- Unaltered Original ---")
    print(f"  Sanity gate: passed={tr_orig['sanity']['passed']}, pred={tr_orig['sanity']['pred']}, conf={tr_orig['sanity']['conf']:.4f}")
    print(f"  Modality gate: passed={tr_orig['modality_gate']['passed']}, pred={tr_orig['modality_gate']['pred']}, conf={tr_orig['modality_gate']['conf']:.4f}")
    print(f"  OOD: dist={tr_orig['ood']['distance']:.2f}, thresh={tr_orig['ood']['threshold']:.2f}, passed={tr_orig['ood']['passed']}")
    print(f"  Final disease: pred={tr_orig['disease']['pred']}, conf={tr_orig['disease']['conf']:.4f}")

    print(f"--- Transformed (intensity inverted) ---")
    print(f"  Sanity gate: passed={tr_trans['sanity']['passed']}, pred={tr_trans['sanity']['pred']}, conf={tr_trans['sanity']['conf']:.4f}")
    print(f"  Modality gate: passed={tr_trans['modality_gate']['passed']}, pred={tr_trans['modality_gate']['pred']}, conf={tr_trans['modality_gate']['conf']:.4f}")
    print(f"  OOD: dist={tr_trans['ood']['distance']:.2f}, thresh={tr_trans['ood']['threshold']:.2f}, passed={tr_trans['ood']['passed']}")
    print(f"  Final disease: pred={tr_trans['disease']['pred']}, conf={tr_trans['disease']['conf']:.4f}")

    create_side_by_side(
        tr_orig["tensor"], tr_trans["tensor"],
        f"Original ({tr_orig['disease']['pred']} {tr_orig['disease']['conf']:.1%})",
        f"Inverted ({tr_trans['disease']['pred']} {tr_trans['disease']['conf']:.1%})",
        out_dir / "trace_ct_intensity_inverted.png"
    )

if __name__ == "__main__":
    main()
