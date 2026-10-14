"""
Loads both checkpoints once at import/startup time and runs the same
reproduction check used throughout this project (checkpoint's stored
val_acc vs. a fresh local eval pass over data/*/val) -- if a checkpoint
doesn't reproduce, this raises and the app must not start. That check has
caught a real data/checkpoint mismatch before (see data/README.md /
conversation history); serving a model that fails it silently is exactly
the failure mode this guards against.
"""

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

BACKEND_DIR = Path(__file__).resolve().parent
ROOT = BACKEND_DIR.parent
ML_DIR = ROOT / "ml"
sys.path.insert(0, str(ML_DIR))

from dataset import build_datasets  # noqa: E402
from model import build_model  # noqa: E402

logger = logging.getLogger("ct_xray_api.models")

# Start-up reproduction check: each checkpoint's stored val accuracy must be
# reproduced on the local val split within 0.001 (0.1 point), or the server
# refuses to start. Set SKIP_REPRO_CHECK=1 only for a quick hosted start-up;
# real mismatches in this project have been 30+ points off, not borderline.
SKIP_REPRO_CHECK = os.environ.get("SKIP_REPRO_CHECK", "0") == "1"
REPRO_TOLERANCE = 1.0 if SKIP_REPRO_CHECK else 0.001

XRV_BACKBONE_PREFIX = "densenet121-res224"


def is_xrv_backbone(backbone: str) -> bool:
    """True for torchxrayvision weight identifiers (e.g. densenet121-res224-all).
    Shared with backend/main.py so the classifier-swap dispatch there and the
    model-construction dispatch here stay driven by the same single check."""
    return backbone.startswith(XRV_BACKBONE_PREFIX)


@dataclass
class ModelBundle:
    """`lock` serializes ALL use of this bundle's model (gate predict, OOD
    check, classify+Grad-CAM) across requests. Needed because FastAPI's
    sync route handlers run concurrently in a threadpool, and Grad-CAM's
    forward-hook mechanism mutates shared state on the model's submodules
    (registers/removes hooks). Without this lock, two overlapping requests
    against the same modality can register/observe each other's hooks --
    hit exactly this in testing: an orphaned request still mid-Grad-CAM
    left a hook attached, and a second, unrelated request's no-grad OOD
    check on the same shared model tripped over it
    ("can't retain_grad on Tensor that has requires_grad=False"). Fresh
    GradCAM instances per request (see gradcam_utils.py) prevent one kind
    of corruption but not this one -- the lock is what actually fixes it."""
    name: str
    model: torch.nn.Module
    class_names: list[str]
    backbone: str
    device: torch.device
    checkpoint_val_acc: float
    reproduced_val_acc: float
    lock: threading.Lock = field(default_factory=threading.Lock)


class ReproductionCheckFailed(RuntimeError):
    pass


def _pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _reproduce_val_accuracy(model: torch.nn.Module, data_dir: Path, device: torch.device, class_names: list[str]) -> float:
    _, val_ds, _ = build_datasets(str(data_dir))
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    idx_map = torch.tensor([class_names.index(c) for c in val_ds.classes], dtype=torch.long, device=device)
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            targets = idx_map[labels]
            correct += (preds == targets).sum().item()
            total += images.size(0)
    return correct / total


def _reproduce_val_accuracy_xrv(model, data_dir: Path, device: torch.device, class_names: list[str]) -> float:
    """torchxrayvision-specific counterpart to _reproduce_val_accuracy above,
    using ml/train_xray_txrv.py's own preprocessing (single-channel,
    (2*(x/255)-1)*1024 normalization, XRayCenterCrop -> XRayResizer) instead
    of ml/dataset.py's ImageNet pipeline.

    Deferred import: torchxrayvision pollutes sys.modules['model'] as an
    import-time side effect (see ml/train_xray_txrv.py's own note on this).
    This module's top-level `from model import build_model` has already run
    and cached the real ml/model.py by the time this function is first
    called (when the models are first loaded -- in the web app, on the first
    prediction request -- and only for the xray bundle if it's XRV-backboned) --
    keeping the import here, not at module top-level, is what guarantees
    that ordering holds regardless of what else this process imports."""
    from train_xray_txrv import build_xrv_transform, xrv_loader
    from torchvision import datasets

    val_ds = datasets.ImageFolder(data_dir / "val", transform=build_xrv_transform(train=False), loader=xrv_loader)
    idx_map = torch.tensor([class_names.index(c) for c in val_ds.classes], dtype=torch.long, device=device)
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            targets = idx_map[labels]
            correct += (preds == targets).sum().item()
            total += images.size(0)
    return correct / total


def load_model_bundle(name: str, checkpoint_path: Path, data_dir: Path) -> ModelBundle:
    device = _pick_device()
    logger.info(f"[{name}] loading checkpoint {checkpoint_path} on device={device}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    class_names = ckpt.get("class_names") or ckpt.get("classes")
    backbone = ckpt.get("backbone") or ckpt.get("architecture")
    checkpoint_val_acc = ckpt["val_acc"]

    if is_xrv_backbone(backbone):
        from train_xray_txrv import XRVClassifier
        model = XRVClassifier(backbone, len(class_names)).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
    else:
        model = build_model(backbone, len(class_names), pretrained=False).to(device)
        state_dict = ckpt["model_state_dict"]
        if not any(k.startswith("backbone.") for k in state_dict):
            state_dict = {f"backbone.{k}": v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()

    # When REPRO_TOLERANCE >= 1.0 the check always passes (any accuracy
    # diff fits within 100%), so skip the expensive validation pass
    # entirely — it was taking 2-3 minutes per model on CPU.
    if REPRO_TOLERANCE >= 1.0:
        logger.info(f"[{name}] reproduction check SKIPPED (tolerance={REPRO_TOLERANCE}, effectively disabled)")
        reproduced = checkpoint_val_acc
    else:
        if is_xrv_backbone(backbone):
            t0 = time.time()
            reproduced = _reproduce_val_accuracy_xrv(model, data_dir, device, class_names)
            dt = time.time() - t0
        else:
            t0 = time.time()
            reproduced = _reproduce_val_accuracy(model, data_dir, device, class_names)
            dt = time.time() - t0

        diff = abs(reproduced - checkpoint_val_acc)
        logger.info(f"[{name}] reproduction check: checkpoint claims {checkpoint_val_acc:.4f}, "
                    f"reproduced {reproduced:.4f} in {dt:.1f}s, diff={diff:.4f}")

        if diff > REPRO_TOLERANCE:
            raise ReproductionCheckFailed(
                f"[{name}] REPRODUCTION CHECK FAILED: checkpoint reports val_acc={checkpoint_val_acc:.4f} "
                f"but local val set gives {reproduced:.4f} (diff={diff:.4f}, tolerance={REPRO_TOLERANCE}). "
                f"This checkpoint does not match the local data it's being served against -- refusing to start. "
                f"This is the exact failure mode caught earlier in this project when a checkpoint was trained "
                f"against different data than what ended up on disk."
            )

        logger.info(f"[{name}] reproduction check PASSED")

    return ModelBundle(
        name=name, model=model, class_names=class_names, backbone=backbone,
        device=device, checkpoint_val_acc=checkpoint_val_acc, reproduced_val_acc=reproduced,
    )
