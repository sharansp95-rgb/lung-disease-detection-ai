"""
Full (not sampled) Grad-CAM audit of the exclusion-fixed CT retrain
(ct_best_model_phase35_v2.pt) on its test set. Generates one grid per
class covering every correctly-classified test image (not a subsample),
plus a grid of the misclassified ones -- to directly answer whether the
scanner-table shortcut from the earlier audits is actually gone now that
the protocol-mismatched Malignant subset has been excluded.
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))
from dataset import build_datasets, build_transforms, IMAGENET_MEAN, IMAGENET_STD
from gradcam import GradCAM, get_target_layer, denormalize, overlay_cam
from model import build_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))
from evaluate import build_labeled_grid, gradcam_tile

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "checkpoints" / "ct_best_model_phase35_v2.pt"
DATA_DIR = ROOT / "data" / "ct_scan"
OUT_DIR = ROOT / "eval_outputs" / "ct_scan_phase35_v2"


def main():
    device = torch.device("cpu")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    model = build_model(ckpt["backbone"], len(class_names), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, _, test_ds = build_datasets(str(DATA_DIR))
    transform = build_transforms(train=False)

    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    preds, labels, confs = [], [], []
    with torch.no_grad():
        for images, lbls in loader:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            c, p = probs.max(dim=1)
            preds.extend(p.tolist())
            labels.extend(lbls.tolist())
            confs.extend(c.tolist())

    samples = test_ds.samples
    target_layer = get_target_layer(model)
    cam_engine = GradCAM(model, target_layer)

    correct_by_class = {c: [] for c in range(len(class_names))}
    misclassified = []
    for i, (path, true_label) in enumerate(samples):
        if preds[i] == labels[i]:
            correct_by_class[labels[i]].append(i)
        else:
            misclassified.append(i)

    for cls_idx, cls_name in enumerate(class_names):
        idxs = sorted(correct_by_class[cls_idx], key=lambda i: -confs[i])
        tiles = []
        for i in idxs:
            path, true_label = samples[i]
            tile_img, label_text, ok = gradcam_tile(model, cam_engine, path, transform, true_label, class_names, device)
            tiles.append((tile_img, f"{Path(path).name}\n{label_text}"))
        out_path = OUT_DIR / f"ct_phase35_v2_gradcam_FULL_{cls_name}_correct.png"
        build_labeled_grid(tiles, out_path, cols=5,
                            title=f"CT phase35_v2 FULL correct grid -- {cls_name} (n={len(tiles)}, ALL correct test images)")
        print(f"{cls_name}: {len(tiles)} correct images -> {out_path}")

    mis_tiles = []
    for i in misclassified:
        path, true_label = samples[i]
        tile_img, label_text, ok = gradcam_tile(model, cam_engine, path, transform, true_label, class_names, device)
        mis_tiles.append((tile_img, f"{Path(path).name}\n{label_text}"))
    out_path = OUT_DIR / "ct_phase35_v2_gradcam_FULL_misclassified.png"
    if mis_tiles:
        build_labeled_grid(mis_tiles, out_path, cols=min(5, len(mis_tiles)),
                            title=f"CT phase35_v2 FULL misclassified (n={len(mis_tiles)})")
        print(f"misclassified: {len(mis_tiles)} images -> {out_path}")

    cam_engine.remove_hooks()


if __name__ == "__main__":
    main()
