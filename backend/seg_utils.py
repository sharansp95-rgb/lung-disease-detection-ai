"""
Loads the lung-segmentation U-Net (checkpoints/xray_seg_model.pt) and
provides the mask-guided crop that /predict/xray uses in place of (or as an
upgrade over) the border-crop approximation added in the earlier
train/inference-mismatch fix.

Same reproduction-check discipline as models.py: the checkpoint's claimed
val_dice/val_iou are re-verified against a fresh eval pass over
seg_data/val/ at load time. Refuses to start if they don't reproduce --
this is the same guard that has caught real checkpoint/data mismatches
before in this project, and here it also validates an assumption this
module has to make: the checkpoint carries no record of what input
normalization it was trained with, so _to_seg_tensor() assumes plain
[0,1] scaling (no mean/std normalization) based on what empirically
reproduces the claimed metrics. If that assumption were wrong, this check
would fail loudly rather than silently serving a broken segmentation step.
"""

import os
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent
ROOT = BACKEND_DIR.parent
ML_DIR = ROOT / "ml"
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(ML_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from seg_model import UNet  # noqa: E402
from prepare_xray import (  # noqa: E402
    border_crop, crop_and_resize, mask_bbox, MASK_PAD_PX, TARGET_SIZE, TB_BORDER_FRAC,
)

logger = logging.getLogger("ct_xray_api.seg")

SEG_INPUT_SIZE = 256
# Same switch as models.py: strict by default, SKIP_REPRO_CHECK=1 disables it.
SKIP_REPRO_CHECK = os.environ.get("SKIP_REPRO_CHECK", "0") == "1"
DICE_REPRO_TOLERANCE = 1.0 if SKIP_REPRO_CHECK else 0.01
IOU_REPRO_TOLERANCE = 1.0 if SKIP_REPRO_CHECK else 0.02

# A predicted mask is treated as degenerate (fall back to border_crop)
# if its bbox covers an implausible fraction of the frame -- either the
# model found essentially nothing, or it found "the whole image", neither
# of which is a usable lung crop.
MIN_BBOX_AREA_FRAC = 0.05
MAX_BBOX_AREA_FRAC = 0.98


@dataclass
class SegBundle:
    model: torch.nn.Module
    device: torch.device
    checkpoint_val_dice: float
    checkpoint_val_iou: float
    reproduced_val_dice: float
    reproduced_val_iou: float
    lock: threading.Lock = field(default_factory=threading.Lock)


class SegReproductionCheckFailed(RuntimeError):
    pass


def _pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_seg_tensor(pil_image: Image.Image, device: torch.device) -> torch.Tensor:
    img = pil_image.convert("L").resize((SEG_INPUT_SIZE, SEG_INPUT_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def _dice_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> tuple[float, float]:
    inter = (pred * target).sum().item()
    p_sum = pred.sum().item()
    t_sum = target.sum().item()
    union = p_sum + t_sum - inter
    dice = (2 * inter + eps) / (p_sum + t_sum + eps)
    iou = (inter + eps) / (union + eps)
    return dice, iou


def _reproduce_val_metrics(model: torch.nn.Module, seg_val_dir: Path, device: torch.device) -> tuple[float, float]:
    images_dir = seg_val_dir / "images"
    masks_dir = seg_val_dir / "masks"
    img_paths = sorted(images_dir.iterdir())

    dices, ious = [], []
    model.eval()
    with torch.no_grad():
        for img_path in img_paths:
            mask_path = masks_dir / img_path.name
            x = _to_seg_tensor(Image.open(img_path), device)
            target_arr = np.array(Image.open(mask_path), dtype=np.float32) / 255.0
            target = torch.from_numpy((target_arr > 0.5).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

            logits = model(x)
            pred = (torch.sigmoid(logits) > 0.5).float()
            dice, iou = _dice_iou(pred, target)
            dices.append(dice)
            ious.append(iou)

    return float(np.mean(dices)), float(np.mean(ious))


def load_seg_bundle(checkpoint_path: Path, seg_data_dir: Path) -> SegBundle:
    device = _pick_device()
    logger.info(f"[seg] loading checkpoint {checkpoint_path} on device={device}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    checkpoint_val_dice = ckpt["val_dice"]
    checkpoint_val_iou = ckpt["val_iou"]

    model = UNet(in_channels=1, out_channels=1, base=32).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # When tolerances are >= 1.0 the check always passes, so skip the
    # expensive validation pass entirely for faster startup.
    if DICE_REPRO_TOLERANCE >= 1.0 and IOU_REPRO_TOLERANCE >= 1.0:
        logger.info("[seg] reproduction check SKIPPED (tolerances effectively disabled)")
        reproduced_dice = checkpoint_val_dice
        reproduced_iou = checkpoint_val_iou
    else:
        t0 = time.time()
        reproduced_dice, reproduced_iou = _reproduce_val_metrics(model, seg_data_dir / "val", device)
        dt = time.time() - t0

        dice_diff = abs(reproduced_dice - checkpoint_val_dice)
        iou_diff = abs(reproduced_iou - checkpoint_val_iou)
        logger.info(
            f"[seg] reproduction check: checkpoint claims dice={checkpoint_val_dice:.4f} iou={checkpoint_val_iou:.4f}, "
            f"reproduced dice={reproduced_dice:.4f} iou={reproduced_iou:.4f} in {dt:.1f}s "
            f"(diff dice={dice_diff:.4f}, iou={iou_diff:.4f})"
        )

        if dice_diff > DICE_REPRO_TOLERANCE or iou_diff > IOU_REPRO_TOLERANCE:
            raise SegReproductionCheckFailed(
                f"[seg] REPRODUCTION CHECK FAILED: checkpoint reports dice={checkpoint_val_dice:.4f} "
                f"iou={checkpoint_val_iou:.4f} but local val set (seg_data/val/) gives "
                f"dice={reproduced_dice:.4f} iou={reproduced_iou:.4f} "
                f"(diff dice={dice_diff:.4f} > {DICE_REPRO_TOLERANCE} or iou={iou_diff:.4f} > {IOU_REPRO_TOLERANCE}). "
                f"This checkpoint does not match the local data it's being served against, or the input "
                f"preprocessing assumption in seg_utils.py is wrong -- refusing to start."
            )

        logger.info("[seg] reproduction check PASSED")

    return SegBundle(
        model=model, device=device,
        checkpoint_val_dice=checkpoint_val_dice, checkpoint_val_iou=checkpoint_val_iou,
        reproduced_val_dice=reproduced_dice, reproduced_val_iou=reproduced_iou,
    )


def compute_lung_crop(image: Image.Image, seg_bundle: SegBundle) -> tuple[Image.Image, bool]:
    """Runs segmentation, thresholds the mask, computes the lung bbox
    (reusing prepare_xray.py's mask_bbox() -- same +4px pad as training),
    and crops the ORIGINAL full-resolution image to it. The bbox is
    computed at SEG_INPUT_SIZE (the model's fixed input resolution) and
    then scaled up proportionally, so the crop is applied to the upload's
    actual resolution rather than to a downsampled copy.

    Falls back to prepare_xray.py's border_crop() (reused, not
    reimplemented) if the predicted mask is empty or its bbox covers an
    implausible fraction of the image.

    EVERY return path finishes through prepare_xray.py's crop_and_resize()
    -- the same non-aspect-preserving LANCZOS squash to TARGET_SIZE that
    built data/xray/. This is load-bearing, not cosmetic. A lung bbox is
    almost never square (measured: 384x312, 248x185, 222x259 on real
    uploads), and the classifier's XRV transform starts with
    xrv.datasets.XRayCenterCrop(), which crops a non-square input down to
    its shorter side. Returning the raw bbox therefore fed the model a
    centre-cropped lung field with the outer lung margins amputated --
    e.g. 72px of width discarded on a 384x312 crop -- while the training
    images it was fitted on had the *whole* bbox squashed to 224x224 with
    nothing discarded. Squashing here makes XRayCenterCrop a no-op and
    restores the training geometry. See eval_run/xray_cropfix_applied.md
    (crop-diff PNGs in eval_run/xray_cropfix_backups/): this one line
    moved live accuracy on original uploads from 0.883 to 0.950 and on
    the pre-cropped test split from 0.800 to 0.921.

    Returns (cropped_image, used_segmentation), the image always
    TARGET_SIZE x TARGET_SIZE greyscale.
    """
    with seg_bundle.lock:
        x = _to_seg_tensor(image, seg_bundle.device)
        with torch.no_grad():
            logits = seg_bundle.model(x)
            probs = torch.sigmoid(logits)
            mask_small = (probs[0, 0] > 0.5).cpu().numpy().astype(np.uint8)

    gray = np.array(image.convert("L"))

    if mask_small.max() == 0:
        cropped = border_crop(gray, TB_BORDER_FRAC)
        return crop_and_resize(cropped, TARGET_SIZE), False

    r0, r1, c0, c1 = mask_bbox(mask_small, pad=MASK_PAD_PX)
    bbox_area_frac = ((r1 - r0) * (c1 - c0)) / (SEG_INPUT_SIZE * SEG_INPUT_SIZE)

    if bbox_area_frac < MIN_BBOX_AREA_FRAC or bbox_area_frac > MAX_BBOX_AREA_FRAC:
        cropped = border_crop(gray, TB_BORDER_FRAC)
        return crop_and_resize(cropped, TARGET_SIZE), False

    H, W = gray.shape
    scale_r, scale_c = H / SEG_INPUT_SIZE, W / SEG_INPUT_SIZE
    orig_r0, orig_r1 = int(r0 * scale_r), int(r1 * scale_r)
    orig_c0, orig_c1 = int(c0 * scale_c), int(c1 * scale_c)
    cropped = gray[orig_r0:orig_r1, orig_c0:orig_c1]
    return crop_and_resize(cropped, TARGET_SIZE), True
