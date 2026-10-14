"""
Thin per-request wrapper around ml/gradcam.py -- reuses the exact GradCAM
implementation verified in the Phase 3 audits, doesn't reimplement it.

A fresh GradCAM instance (i.e. fresh forward hook) is created per request
rather than reused across requests: the hook stores activations as
instance state, and FastAPI's sync route handlers can run concurrently in
a threadpool, so a shared instance risks one request's activations being
clobbered by another's mid-flight. Registering a hook is cheap; this
trades a negligible per-request cost for correctness under concurrency.
"""

import base64
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent
ML_DIR = BACKEND_DIR.parent / "ml"
sys.path.insert(0, str(ML_DIR))

from dataset import build_transforms, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from gradcam import GradCAM, get_target_layer, denormalize, overlay_cam  # noqa: E402

_transform = build_transforms(train=False)


def preprocess_image(pil_image: Image.Image, device: torch.device) -> torch.Tensor:
    """[1,3,224,224] normalized tensor -- the exact preprocessing used for
    both the OOD centroid-distance check and the classifier/Grad-CAM pass,
    so both operate on identically-processed input."""
    image = pil_image.convert("RGB")
    return _transform(image).unsqueeze(0).to(device)


def predict_with_heatmap(model: torch.nn.Module, input_tensor: torch.Tensor):
    """input_tensor: output of preprocess_image(). Returns (pred_idx, probs_array, heatmap_data_uri_str)."""
    cam_engine = GradCAM(model, get_target_layer(model))
    try:
        cam, pred_idx, probs = cam_engine.generate(input_tensor, class_idx=None)
    finally:
        cam_engine.remove_hooks()

    display_img = denormalize(input_tensor[0].cpu(), IMAGENET_MEAN, IMAGENET_STD)
    overlay = overlay_cam(display_img, cam, alpha=0.45)

    png_bytes = io.BytesIO()
    Image.fromarray(overlay).save(png_bytes, format="PNG")
    b64 = base64.b64encode(png_bytes.getvalue()).decode("ascii")
    data_uri = f"data:image/png;base64,{b64}"

    return pred_idx, probs, data_uri


# --- torchxrayvision-backboned X-ray classifier: separate preprocessing/
# display path. NOT used for CT or either gate model -- those keep using
# preprocess_image()/predict_with_heatmap() above unconditionally, since
# they're unaffected by which disease-classifier backbone is active. ---

def preprocess_image_xrv(pil_image: Image.Image, device: torch.device) -> torch.Tensor:
    """[1,1,224,224] tensor in torchxrayvision's own ~[-1024,1024] space --
    NOT the same preprocessing as preprocess_image() above."""
    from train_xray_txrv import build_xrv_transform  # deferred -- see models.py's note on import order
    image = pil_image.convert("L")
    return build_xrv_transform(train=False)(image).unsqueeze(0).to(device)


def denormalize_xrv(tensor: torch.Tensor) -> np.ndarray:
    """[1,H,W] tensor in ~[-1024,1024] -> uint8 HWC RGB numpy (grayscale
    replicated across channels), for heatmap overlay display. Inverse of
    xrv.utils.normalize(img, maxval=255, reshape=True):
        norm = (2*(img/255) - 1) * 1024   =>   img = (norm/1024 + 1) * 127.5
    Only undoes the normalization -- the crop/resize already baked the
    actual 224x224 image content into the tensor, same as the old
    denormalize() only ever undoing ImageNet mean/std, not any geometric step."""
    arr = tensor.detach().cpu().numpy()[0]  # (H, W)
    arr = (arr / 1024.0 + 1.0) * 127.5
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.stack([arr, arr, arr], axis=-1)  # (H, W, 3)


def predict_with_heatmap_xrv(model: torch.nn.Module, input_tensor: torch.Tensor):
    """XRV counterpart to predict_with_heatmap() above -- same Grad-CAM
    flow, using denormalize_xrv() for display instead of ImageNet
    denormalize (which would silently mis-render a single-channel,
    differently-scaled tensor rather than crash)."""
    cam_engine = GradCAM(model, get_target_layer(model))
    try:
        cam, pred_idx, probs = cam_engine.generate(input_tensor, class_idx=None)
    finally:
        cam_engine.remove_hooks()

    display_img = denormalize_xrv(input_tensor[0].cpu())
    overlay = overlay_cam(display_img, cam, alpha=0.45)

    png_bytes = io.BytesIO()
    Image.fromarray(overlay).save(png_bytes, format="PNG")
    b64 = base64.b64encode(png_bytes.getvalue()).decode("ascii")
    data_uri = f"data:image/png;base64,{b64}"

    return pred_idx, probs, data_uri
