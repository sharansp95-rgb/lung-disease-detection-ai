"""
X-ray training using a torchxrayvision-pretrained DenseNet121 backbone instead
of plain ImageNet-pretrained DenseNet121 (ml/train.py / ml/model.py), aiming
for better generalization to external images: torchxrayvision's weights are
pretrained on real chest X-rays from multiple hospitals/datasets (NIH, PadChest,
CheXpert, RSNA, MIMIC-CXR), not natural photos, so the backbone starts already
adapted to X-ray-specific texture/contrast statistics rather than needing to
unlearn ImageNet priors.

    python train_xray_txrv.py --data_dir ../data/xray --epochs 15

IMPORTANT — torchxrayvision preprocessing is NOT ml/dataset.py's preprocessing.
Verified directly against the installed torchxrayvision==1.5.4 package before
writing this (do not assume from memory / docs, which drift across versions):

  1. INPUT CHANNELS: the pretrained backbone's first conv is
     `Conv2d(1, 64, ...)` -- single-channel, NOT 3-channel. ml/dataset.py's
     `transforms.Grayscale(num_output_channels=3)` (replicating grayscale to
     3 channels for an RGB-expecting backbone) is wrong here and is not used
     below -- images are kept single-channel throughout.

  2. NORMALIZATION: NOT ImageNet mean/std. torchxrayvision's own
     `xrv.utils.normalize(img, maxval)` maps [0, maxval] -> roughly
     [-1024, 1024] via `(2 * (img / maxval) - 1) * 1024`. Using ImageNet
     normalization here would silently feed the pretrained backbone
     out-of-distribution inputs -- it would still run without erroring, and
     would just quietly perform worse, which is exactly the failure mode
     flagged in the task: "a mismatch here silently produces bad results".
     The model itself has a soft sanity check for this
     (`xrv.utils.warn_normalization`) but it only warns, doesn't raise, so
     it's not a substitute for getting this right.

  3. CROP-THEN-RESIZE: torchxrayvision's own models were trained using
     `XRayCenterCrop()` (crop to the square of the smaller dimension) followed
     by `XRayResizer(224)` (skimage-based resize to 224x224) -- replicated
     exactly below via the same classes from `xrv.datasets`, applied on top
     of this project's own existing lung-crop preprocessing (already baked
     into the files under data/xray/ by ml's prepare_xray.py -- unaffected by
     this script, since that crop happens at data-prep time, not here).

  4. The backbone's `forward()` applies `torch.sigmoid` + threshold
     calibration (`op_threshs`) meant for its original 18-pathology
     multi-label output. Since the classifier head is replaced with our own
     4-class `nn.Linear`, `op_threshs` is explicitly set to `None` below so
     `forward()` returns raw logits (matching `nn.CrossEntropyLoss`'s
     expectation) instead of running sigmoid+threshold math against weights
     that no longer correspond to those pathologies.

Compute: same architecture size as the existing densenet121 pipeline (no
extra parameters beyond the swapped 1024->4 head), so memory needs are
similar to ml/train.py. This machine has a local 4GB-VRAM GPU (RTX 3050
Laptop) -- enough to smoke-test with a small --batch_size (e.g. 8) for a
couple of epochs, but a full 15-epoch run at batch_size=32 should still go to
Colab, consistent with how the existing checkpoints in this repo were
trained.
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# torchxrayvision's own package __init__ eagerly imports baseline_models,
# one of which (jfhealthcare) does an unqualified `import model` / `from
# model.utils import ...` at ITS OWN top level -- clearly meant to be a
# package-relative import, but written as an absolute one. This collides
# with this project's own ml/model.py, which every other training/backend
# module imports via the same bare `from model import build_model` pattern
# (relying on ml/ being on sys.path). Whichever import runs first "wins"
# the global sys.modules['model'] cache slot and breaks the other --
# confirmed both directions: importing torchxrayvision first breaks `from
# model import build_model` (ImportError: cannot import name 'build_model'
# from 'model'); importing ml/model.py first breaks torchxrayvision's own
# import instead (ModuleNotFoundError: No module named 'model.utils';
# 'model' is not a package). Neither project's code is wrong here -- this
# is upstream's bug. Worked around by hiding whatever 'model' module is
# already cached (if any) while torchxrayvision imports, then discarding
# whatever IT cached under that name afterward and restoring the original.
# Safe because this project never uses torchxrayvision's jfhealthcare
# baseline model (only xrv.models.DenseNet / xrv.datasets / xrv.utils), so
# nothing later ever needs that submodule reachable via sys.modules['model'].
_saved_model_module = sys.modules.pop("model", None)
import torchxrayvision as xrv
sys.modules.pop("model", None)
if _saved_model_module is not None:
    sys.modules["model"] = _saved_model_module
del _saved_model_module

from PIL import Image
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import compute_class_weights  # noqa: E402

# NOTE: deliberately NOT `from train import run_one_epoch, plot_curves, set_seed`.
# torchxrayvision (imported above) pollutes sys.modules with its own internal
# submodule registered under the bare name "model" (one of its
# baseline_models packages does a non-relative `import model`), which then
# shadows this project's local ml/model.py. train.py's own
# `from model import build_model, SUPPORTED_BACKBONES` breaks as soon as
# torchxrayvision has been imported anywhere in the same process -- confirmed
# by reproducing the ImportError directly. Inlining the three small functions
# below (unchanged from train.py) avoids the collision entirely rather than
# depending on fragile import ordering.

ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_one_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    grad_context = torch.enable_grad() if train else torch.no_grad()
    with grad_context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def plot_curves(history: dict, out_path: Path, title: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"{title} — loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title(f"{title} — accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

IMAGE_SIZE = 224
XRV_MAXVAL = 255  # 8-bit source images (verified: everything under data/xray/ is 8-bit PNG/JPG)

XRV_DENSENET_WEIGHTS = (
    "densenet121-res224-all",   # trained on a combination of NIH/PC/CheXpert/MIMIC/RSNA -- broadest diversity, default
    "densenet121-res224-nih",
    "densenet121-res224-pc",
    "densenet121-res224-chex",
    "densenet121-res224-rsna",
    "densenet121-res224-mimic_nb",
    "densenet121-res224-mimic_ch",
)

_center_crop = xrv.datasets.XRayCenterCrop()
_resizer = xrv.datasets.XRayResizer(IMAGE_SIZE, engine="skimage")


def xrv_loader(path: str) -> Image.Image:
    """Load as single-channel grayscale -- NOT RGB. ImageFolder's default
    loader converts to RGB; that would silently break the 1-channel-input
    assumption of the torchxrayvision backbone."""
    return Image.open(path).convert("L")


def build_xrv_transform(train: bool):
    # Augmentation happens on the PIL 'L' image, BEFORE the xrv-specific
    # normalize/crop/resize pipeline -- ColorJitter/rotation/flip all operate
    # fine on single-channel PIL images, and doing this first keeps the
    # augmented pixel values in the original [0,255] domain that
    # xrv.utils.normalize expects, rather than jittering already-normalized
    # [-1024,1024] values (which would need different, error-prone parameter
    # ranges to have the same visual effect).
    pil_ops = []
    if train:
        pil_ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    pil_transform = transforms.Compose(pil_ops) if pil_ops else None

    def _transform(img: Image.Image) -> torch.Tensor:
        if pil_transform is not None:
            img = pil_transform(img)
        arr = np.array(img, dtype=np.float32)  # (H, W), 0..255
        arr = xrv.utils.normalize(arr, XRV_MAXVAL, reshape=True)  # -> (1, H, W), ~[-1024, 1024]
        arr = _center_crop(arr)
        arr = _resizer(arr)  # -> (1, 224, 224) float32
        return torch.from_numpy(arr.copy())

    return _transform


def build_dataloaders(data_dir: str, batch_size: int, num_workers: int = 0):
    data_dir = Path(data_dir)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=build_xrv_transform(train=True), loader=xrv_loader)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=build_xrv_transform(train=False), loader=xrv_loader)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=build_xrv_transform(train=False), loader=xrv_loader)

    if not (train_ds.classes == val_ds.classes == test_ds.classes):
        raise ValueError(
            f"Class mismatch across splits: train={train_ds.classes} "
            f"val={val_ds.classes} test={test_ds.classes}"
        )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                                num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds.classes


class XRVClassifier(nn.Module):
    """torchxrayvision-pretrained DenseNet121 backbone with its 18-pathology
    classifier head replaced by a fresh `num_classes` head. Mirrors
    ml/model.py's ImageClassifier interface (freeze_backbone/unfreeze_backbone)
    so ml/train.py's warmup/fine-tune loop logic (imported, not duplicated)
    works unchanged."""

    def __init__(self, xrv_weights: str, num_classes: int):
        super().__init__()
        self.backbone = xrv.models.DenseNet(weights=xrv_weights)
        # Disable the pretrained multi-label sigmoid+threshold postprocessing
        # (op_threshs) -- it's calibrated for the original 18 pathologies and
        # is meaningless (actively wrong) against a freshly-initialized head.
        self.backbone.op_threshs = None
        in_features = self.backbone.classifier.in_features  # 1024, verified against installed package
        self.backbone.classifier = nn.Linear(in_features, num_classes)
        self.backbone_name = "densenet121"  # ml/gradcam.py's get_target_layer() checks this attribute

    def forward(self, x):
        return self.backbone(x)

    def forward_features(self, x) -> torch.Tensor:
        """Penultimate-layer embedding (1024-d), for use as a fixed feature
        extractor -- same role as ml/model.py's ImageClassifier.forward_features
        (backend/ood_detector.py calls this generically on whatever model is
        loaded). torchxrayvision's own features2() already does exactly
        features->relu->global-avg-pool->flatten, so this just delegates."""
        return self.backbone.features2(x)

    def freeze_backbone(self):
        for p in self.backbone.features.parameters():
            p.requires_grad = False
        for p in self.backbone.classifier.parameters():
            p.requires_grad = True

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


def build_model(xrv_weights: str, num_classes: int) -> XRVClassifier:
    return XRVClassifier(xrv_weights, num_classes)


def run(args) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[train_xray_txrv] device={device}")

    train_loader, val_loader, _, class_names = build_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    num_classes = len(class_names)
    print(f"[train_xray_txrv] classes ({num_classes}): {class_names}")
    print(f"[train_xray_txrv] torchxrayvision weights: {args.xrv_weights}")

    class_weights, class_counts = compute_class_weights(train_loader.dataset)
    print("[train_xray_txrv] training-set class counts:")
    for name, cnt in zip(class_names, class_counts.tolist()):
        print(f"  {name}: {cnt}")

    model = build_model(args.xrv_weights, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_epoch = -1

    run_name = args.run_name or f"{Path(args.data_dir).name}_txrv_{args.xrv_weights}"
    ckpt_dir = ROOT / args.output_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{run_name}_best.pt"

    total_epochs = args.epochs
    warmup_epochs = min(args.warmup_epochs, total_epochs)

    for epoch in range(1, total_epochs + 1):
        phase_is_warmup = epoch <= warmup_epochs
        if epoch == 1 and phase_is_warmup:
            model.freeze_backbone()
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
            print(f"[train_xray_txrv] epoch {epoch}: backbone FROZEN (warmup), lr={args.lr}")
        elif epoch == warmup_epochs + 1:
            model.unfreeze_backbone()
            optimizer = optim.Adam(model.parameters(), lr=args.finetune_lr)
            print(f"[train_xray_txrv] epoch {epoch}: backbone UNFROZEN (fine-tune), lr={args.finetune_lr}")

        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        phase = "warmup" if phase_is_warmup else "finetune"
        print(f"[train_xray_txrv] epoch {epoch}/{total_epochs} ({phase}) "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "backbone": f"txrv_{args.xrv_weights}",
                "num_classes": num_classes,
                "class_names": class_names,
                "val_acc": val_acc,
                "epoch": epoch,
            }, ckpt_path)

    print(f"[train_xray_txrv] best val_acc={best_val_acc:.4f} at epoch {best_epoch} -> saved to {ckpt_path}")

    curves_path = log_dir / f"{run_name}_curves.png"
    plot_curves(history, curves_path, title=run_name)
    print(f"[train_xray_txrv] curves saved to {curves_path}")

    return {"best_val_acc": best_val_acc, "best_epoch": best_epoch, "history": history,
            "class_names": class_names, "checkpoint": str(ckpt_path)}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=str(ROOT / "data" / "xray"))
    p.add_argument("--xrv_weights", default="densenet121-res224-all", choices=XRV_DENSENET_WEIGHTS)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--finetune_lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--log_dir", default="ml_logs")
    p.add_argument("--run_name", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
