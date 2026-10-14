"""
CT training using a RadImageNet-pretrained DenseNet121 backbone instead of
plain ImageNet-pretrained DenseNet121 (ml/train.py / ml/model.py), aiming for
better generalization: RadImageNet is pretrained on 1.35M real CT/MRI/
ultrasound images (not natural photos), so the backbone starts already
adapted to medical-image texture/contrast statistics.

Despite the filename (kept for consistency with how this track was scoped),
this does NOT use a MONAI pretrained bundle. MONAI's Model Zoo was checked
directly (`monai.bundle.get_all_bundles_list()`) and its only lung/CT-relevant
entry, `lung_nodule_ct_detection`, is a 3D nodule-*detection* bundle (RetinaNet
on LUNA16 full CT volumes, variable-size 3D patches) -- wrong task (detection,
not classification) and wrong data format (3D volumes vs. this project's 2D
JPG slices with no volume/spacing metadata available from IQ-OTH/NCCD). No
usable pretrained CT classifier exists in MONAI for this project's spec.

    python train_ct_monai.py --data_dir ../data/ct_scan --epochs 15

IMPORTANT -- preprocessing, confirmed against the official RadImageNet repo's
own `pytorch_example.ipynb` (github.com/BMEII-AI/RadImageNet), not assumed:

  1. CHANNELS: 3-channel (their example uses `cv2.imread` with no grayscale
     flag). This project's data/ct_scan/ images are already stored as 3-channel
     RGB JPGs with identical R=G=B values (verified directly, max|R-G|=0,
     max|G-B|=0 on sample images from all 3 classes) -- so channel order
     (their BGR vs. this code's RGB) makes no numerical difference here, and
     is not a silent mismatch risk the way it would be for genuinely colored
     images.

  2. NORMALIZATION: NOT ImageNet mean/std, and not torchxrayvision's
     [-1024,1024] scheme either -- RadImageNet's own example applies a flat
     `(image - 127.5) * 2 / 255`, mapping [0,255] -> [-1, 1] uniformly (same
     scale for every channel, no per-channel mean/std). Verified from their
     source, not assumed from any general "medical pretrained -> use
     ImageNet-style normalization" default.

  3. WEIGHTS FILE STRUCTURE: the downloaded checkpoint
     (`Lab-Rasool/RadImageNet` on Hugging Face, `DenseNet121.pt`, MIT license,
     no gating) only contains the feature-extractor backbone -- 725 tensors,
     keys prefixed `backbone.0.*` (an artifact of whatever wrapper class
     RadImageNet's own training code used), with NO classifier head. Verified
     directly: stripping the `backbone.0.` prefix and loading into a plain
     `torchvision.models.densenet121(weights=None).features` submodule with
     `strict=True` gives "All keys matched successfully" -- confirmed
     tensor-for-tensor identical naming/shapes to torchvision's own
     `densenet121().features`, and confirmed the loaded weights actually
     differ from a fresh random init (not a silent no-op). No classifier
     weights to carry over regardless, since a fresh `num_classes`-way head is
     being trained anyway.

Compute: same parameter count as the existing densenet121 pipeline. Local
4GB-VRAM GPU (RTX 3050 Laptop) is fine for smoke-testing; a full run should go
to Colab, consistent with this repo's existing checkpoints.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import plot_curves, set_seed  # noqa: E402
from dataset import compute_class_weights  # noqa: E402

# NOTE: intentionally NOT importing run_one_epoch from train.py -- this local
# copy adds optional gradient clipping (to tame the large val_loss spikes seen
# in the first local CT run) without touching train.py, which is shared with
# the production X-ray/CT training path.


def run_one_epoch(model, loader, criterion, optimizer, device, train: bool, grad_clip_norm=None):
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
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total

ROOT = Path(__file__).resolve().parent.parent
IMAGE_SIZE = 224

RADIMAGENET_REPO = "Lab-Rasool/RadImageNet"
RADIMAGENET_FILE = "DenseNet121.pt"
STATE_DICT_KEY_PREFIX = "backbone.0."  # verified prefix in the released checkpoint


def load_radimagenet_densenet121_features() -> models.DenseNet:
    """Downloads (or reuses the cached copy of) RadImageNet's DenseNet121
    backbone weights and loads them into a plain torchvision densenet121's
    `.features` submodule. strict=True by design -- any future change in the
    checkpoint's key naming must fail loudly here, not silently drop
    weights."""
    weights_path = hf_hub_download(repo_id=RADIMAGENET_REPO, filename=RADIMAGENET_FILE)
    raw_state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    remapped = {k.replace(STATE_DICT_KEY_PREFIX, "", 1): v for k, v in raw_state_dict.items()}

    backbone = models.densenet121(weights=None)
    result = backbone.features.load_state_dict(remapped, strict=True)
    print(f"[train_ct_monai] RadImageNet DenseNet121 backbone loaded: {result}")
    return backbone


def radimagenet_loader(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def build_radimagenet_transform(train: bool):
    pil_ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        pil_ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]

    pil_transform = transforms.Compose(pil_ops)

    def _transform(img: Image.Image) -> torch.Tensor:
        img = pil_transform(img)
        arr = np.array(img, dtype=np.float32)  # (H, W, 3), 0..255
        arr = (arr - 127.5) * 2 / 255  # -> [-1, 1], RadImageNet's own scheme
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        return torch.from_numpy(arr.copy())

    return _transform


def build_dataloaders(data_dir: str, batch_size: int, num_workers: int = 0, oversample_train: bool = False):
    data_dir = Path(data_dir)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=build_radimagenet_transform(train=True), loader=radimagenet_loader)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=build_radimagenet_transform(train=False), loader=radimagenet_loader)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=build_radimagenet_transform(train=False), loader=radimagenet_loader)

    if not (train_ds.classes == val_ds.classes == test_ds.classes):
        raise ValueError(
            f"Class mismatch across splits: train={train_ds.classes} "
            f"val={val_ds.classes} test={test_ds.classes}"
        )

    if oversample_train:
        # Per-sample weight = 1/class_count, so every class is expected to
        # appear with roughly equal total weight per epoch -- unlike the
        # loss-only class weighting (which reweights the *gradient* but
        # doesn't change how often the minority class is actually seen per
        # batch), this changes what the model is actually shown. With
        # replacement, num_samples=len(train_ds) keeps epoch length the same.
        targets = [label for _, label in train_ds.samples]
        class_counts_list = [targets.count(c) for c in range(len(train_ds.classes))]
        sample_weights = [1.0 / class_counts_list[label] for label in targets]
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(train_ds), replacement=True
        )
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                                    num_workers=num_workers, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                                    num_workers=num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds.classes


class RadImageNetClassifier(nn.Module):
    """RadImageNet-pretrained DenseNet121 backbone with a fresh `num_classes`
    classifier head. Mirrors ml/model.py's ImageClassifier interface
    (freeze_backbone/unfreeze_backbone) so ml/train.py's warmup/fine-tune
    functions (imported, not duplicated) work unchanged."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = load_radimagenet_densenet121_features()
        in_features = self.backbone.classifier.in_features  # 1024
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        for p in self.backbone.features.parameters():
            p.requires_grad = False
        for p in self.backbone.classifier.parameters():
            p.requires_grad = True

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


def build_model(num_classes: int) -> RadImageNetClassifier:
    return RadImageNetClassifier(num_classes)


def per_class_recall(model, loader, device, class_names):
    """Per-class recall, computed fresh each epoch. Overall accuracy alone
    can hide a fully-collapsed minority class (a model that never predicts
    a ~11%-of-data class can still post a deceptively reasonable overall
    accuracy) -- this is the diagnostic that was missing when the first
    RadImageNet CT run silently collapsed on Benign."""
    model.eval()
    correct = {c: 0 for c in class_names}
    total = {c: 0 for c in class_names}
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu()
            for p, l in zip(preds.tolist(), labels.tolist()):
                cname = class_names[l]
                total[cname] += 1
                if p == l:
                    correct[cname] += 1
    return {c: round(correct[c] / total[c], 4) if total[c] else None for c in class_names}


def run(args) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[train_ct_monai] device={device}")

    train_loader, val_loader, _, class_names = build_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
        oversample_train=args.oversample_train,
    )
    num_classes = len(class_names)
    print(f"[train_ct_monai] classes ({num_classes}): {class_names}")

    class_weights, class_counts = compute_class_weights(train_loader.dataset)
    print("[train_ct_monai] training-set class counts:")
    for name, cnt in zip(class_names, class_counts.tolist()):
        print(f"  {name}: {cnt}")

    model = build_model(num_classes).to(device)
    if args.oversample_train:
        # Oversampling already equalizes how often each class is SEEN per
        # epoch; also applying inverse-frequency loss weighting on top would
        # double-compensate (amplifying the exact confidently-wrong-Benign
        # loss spikes that motivated adding gradient clipping) -- use plain
        # unweighted loss when the sampler is already balancing exposure.
        print("[train_ct_monai] oversample_train=True -> using UNWEIGHTED loss "
              "(class-weighting would double-compensate on top of oversampling)")
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "train_recall_per_class": [], "val_recall_per_class": []}
    best_macro_recall = -1.0
    best_val_acc_at_best = -1.0
    best_epoch = -1

    run_name = args.run_name or f"{Path(args.data_dir).name}_radimagenet_densenet121"
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
            print(f"[train_ct_monai] epoch {epoch}: backbone FROZEN (warmup), lr={args.lr}")
        elif epoch == warmup_epochs + 1:
            model.unfreeze_backbone()
            optimizer = optim.Adam(model.parameters(), lr=args.finetune_lr)
            print(f"[train_ct_monai] epoch {epoch}: backbone UNFROZEN (fine-tune), lr={args.finetune_lr}")

        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True,
                                               grad_clip_norm=args.grad_clip_norm)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False)
        train_recall = per_class_recall(model, train_loader, device, class_names)
        val_recall = per_class_recall(model, val_loader, device, class_names)
        macro_val_recall = round(sum(val_recall.values()) / len(val_recall), 4)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["train_recall_per_class"].append(train_recall)
        history["val_recall_per_class"].append(val_recall)

        phase = "warmup" if phase_is_warmup else "finetune"
        print(f"[train_ct_monai] epoch {epoch}/{total_epochs} ({phase}) "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} macro_val_recall={macro_val_recall:.4f}")
        print(f"    train_recall_per_class={train_recall}")
        print(f"    val_recall_per_class={val_recall}")

        if macro_val_recall > best_macro_recall:
            best_macro_recall = macro_val_recall
            best_val_acc_at_best = val_acc
            best_epoch = epoch
            print(f"    -> new best macro_val_recall={macro_val_recall:.4f} (val_acc={val_acc:.4f}), saving checkpoint")
            torch.save({
                "model_state_dict": model.state_dict(),
                "backbone": "radimagenet_densenet121",
                "num_classes": num_classes,
                "class_names": class_names,
                "val_acc": val_acc,
                "macro_val_recall": macro_val_recall,
                "val_recall_per_class": val_recall,
                "checkpoint_selection_criterion": "best macro-averaged val recall across classes",
                "epoch": epoch,
            }, ckpt_path)

    print(f"[train_ct_monai] best macro_val_recall={best_macro_recall:.4f} (val_acc={best_val_acc_at_best:.4f}) "
          f"at epoch {best_epoch} -> saved to {ckpt_path}")

    curves_path = log_dir / f"{run_name}_curves.png"
    plot_curves(history, curves_path, title=run_name)
    print(f"[train_ct_monai] curves saved to {curves_path}")

    history_path = log_dir / f"{run_name}_history.json"
    with open(history_path, "w") as f:
        json.dump({
            "run_name": run_name, "class_names": class_names,
            "epochs": total_epochs, "warmup_epochs": warmup_epochs,
            "batch_size": args.batch_size, "oversample_train": args.oversample_train,
            "grad_clip_norm": args.grad_clip_norm, "finetune_lr": args.finetune_lr,
            "checkpoint_selection_criterion": "best macro-averaged val recall across classes",
            "best_macro_val_recall": best_macro_recall, "best_val_acc_at_best_epoch": best_val_acc_at_best,
            "best_epoch": best_epoch,
            "history": history,
        }, f, indent=2)
    print(f"[train_ct_monai] history saved to {history_path}")

    return {"best_macro_val_recall": best_macro_recall, "best_val_acc_at_best_epoch": best_val_acc_at_best,
            "best_epoch": best_epoch, "history": history,
            "class_names": class_names, "checkpoint": str(ckpt_path)}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=str(ROOT / "data" / "ct_scan"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--finetune_lr", type=float, default=1e-4)
    p.add_argument("--oversample_train", action="store_true",
                    help="Balance class exposure per epoch via WeightedRandomSampler "
                         "instead of (not in addition to) loss class-weighting")
    p.add_argument("--grad_clip_norm", type=float, default=None,
                    help="Max gradient norm for clipping during training; omit for no clipping")
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
