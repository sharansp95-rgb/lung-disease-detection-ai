"""
Phase 3: inference-only evaluation on the held-out TEST split (never used
for training or checkpoint selection), plus Grad-CAM visualization.

    python evaluate.py --data_dir ../data/xray --checkpoint ../checkpoints/xray_best_model.pt \
        --output_dir ../eval_outputs/xray --run_name xray
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from dataset import build_datasets, IMAGENET_MEAN, IMAGENET_STD
from gradcam import GradCAM, get_target_layer, denormalize, overlay_cam
from model import build_model

ROOT = Path(__file__).resolve().parent.parent


def load_checkpoint(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    backbone = ckpt["backbone"]
    model = build_model(backbone, len(class_names), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, class_names, ckpt.get("val_acc")


# ---------------------------------------------------------------- metrics --

def run_test_metrics(model, test_ds, class_names, device, batch_size=32):
    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds, all_labels, all_confs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_confs.extend(confs.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    report = classification_report(all_labels, all_preds, labels=list(range(len(class_names))),
                                    target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(all_labels, all_preds, labels=list(range(len(class_names))),
                                         target_names=class_names, zero_division=0)

    return {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "classification_report_text": report_text,
        "predictions": all_preds,
        "labels": all_labels,
        "confidences": all_confs,
    }


def plot_confusion_matrix(cm, class_names, out_path: Path, title: str):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(1.4 * len(class_names) + 2, 1.4 * len(class_names) + 2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------- grad-cam --

def gradcam_tile(model, cam_engine, sample_path, transform, true_label, class_names, device):
    img = Image.open(sample_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)
    input_tensor.requires_grad_(False)

    cam, pred_idx, probs = cam_engine.generate(input_tensor, class_idx=None)
    display_img = denormalize(input_tensor[0], IMAGENET_MEAN, IMAGENET_STD)
    overlay = overlay_cam(display_img, cam, alpha=0.45)

    correct = pred_idx == true_label
    label_text = (f"true={class_names[true_label]}\npred={class_names[pred_idx]} "
                  f"({probs[pred_idx]:.2f}){'  OK' if correct else '  WRONG'}")
    return Image.fromarray(overlay), label_text, correct


def build_labeled_grid(tiles, out_path: Path, cols: int, title: str, tile_size=224):
    rows = (len(tiles) + cols - 1) // cols
    pad = 6
    label_h = 34
    grid_w = cols * (tile_size + pad) + pad
    grid_h = rows * (tile_size + pad + label_h) + pad + 30

    canvas = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 4), title, fill="black")
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, (img, label_text) in enumerate(tiles):
        row, col = divmod(i, cols)
        x = pad + col * (tile_size + pad)
        y = 30 + pad + row * (tile_size + pad + label_h)
        img = img.resize((tile_size, tile_size))
        canvas.paste(img, (x, y))
        draw.rectangle([x, y, x + tile_size, y + tile_size], outline="black")
        draw.multiline_text((x, y + tile_size + 2), label_text, fill="black", font=font)

    canvas.save(out_path)
    return out_path


def run_gradcam_analysis(model, test_ds, class_names, device, output_dir: Path, run_name: str,
                          n_correct: int, cols: int, results: dict):
    from dataset import build_transforms
    transform = build_transforms(train=False)

    target_layer = get_target_layer(model)
    cam_engine = GradCAM(model, target_layer)

    samples = test_ds.samples  # list of (path, label)
    preds = results["predictions"]
    confs = results["confidences"]
    labels = results["labels"]

    correct_idx = [i for i in range(len(samples)) if preds[i] == labels[i]]
    incorrect_idx = [i for i in range(len(samples)) if preds[i] != labels[i]]

    # --- sanity check: the 2 highest-confidence correct predictions ---
    sanity_n = min(2, len(correct_idx))
    if sanity_n:
        best_idxs = sorted(correct_idx, key=lambda i: -confs[i])[:sanity_n]
        sanity_tiles = []
        for i in best_idxs:
            path, true_label = samples[i]
            tile_img, label_text, ok = gradcam_tile(model, cam_engine, path, transform, true_label, class_names, device)
            sanity_tiles.append((tile_img, f"conf={confs[i]:.3f}\n{label_text}"))
        sanity_path = output_dir / f"{run_name}_gradcam_sanity_check.png"
        build_labeled_grid(sanity_tiles, sanity_path, cols=sanity_n,
                            title=f"{run_name} — Grad-CAM sanity check (highest-confidence correct predictions)")
    else:
        sanity_path = None

    # --- correctly-classified grid, spread across classes ---
    per_class_budget = max(1, n_correct // len(class_names))
    chosen = []
    by_class = {c: [] for c in range(len(class_names))}
    for i in correct_idx:
        by_class[labels[i]].append(i)
    for c in range(len(class_names)):
        idxs = sorted(by_class[c], key=lambda i: -confs[i])[:per_class_budget]
        chosen.extend(idxs)
    chosen = chosen[:n_correct]

    correct_tiles = []
    for i in chosen:
        path, true_label = samples[i]
        tile_img, label_text, ok = gradcam_tile(model, cam_engine, path, transform, true_label, class_names, device)
        correct_tiles.append((tile_img, label_text))

    correct_grid_path = output_dir / f"{run_name}_gradcam_correct_grid.png"
    if correct_tiles:
        build_labeled_grid(correct_tiles, correct_grid_path, cols=cols,
                            title=f"{run_name} — Grad-CAM, correctly classified (n={len(correct_tiles)})")
    else:
        correct_grid_path = None

    # --- misclassified grid (all of them, if any) ---
    misclassified_tiles = []
    for i in incorrect_idx:
        path, true_label = samples[i]
        tile_img, label_text, ok = gradcam_tile(model, cam_engine, path, transform, true_label, class_names, device)
        misclassified_tiles.append((tile_img, label_text))

    misclassified_grid_path = output_dir / f"{run_name}_gradcam_misclassified_grid.png"
    if misclassified_tiles:
        build_labeled_grid(misclassified_tiles, misclassified_grid_path, cols=min(cols, len(misclassified_tiles)),
                            title=f"{run_name} — Grad-CAM, MISCLASSIFIED (n={len(misclassified_tiles)})")
    else:
        misclassified_grid_path = None

    cam_engine.remove_hooks()

    return {
        "sanity_check_path": str(sanity_path) if sanity_path else None,
        "correct_grid_path": str(correct_grid_path) if correct_grid_path else None,
        "misclassified_grid_path": str(misclassified_grid_path) if misclassified_grid_path else None,
        "n_correct_shown": len(correct_tiles),
        "n_misclassified_total": len(incorrect_idx),
        "n_misclassified_shown": len(misclassified_tiles),
    }


# ---------------------------------------------------------------- driver --

def evaluate(data_dir: str, checkpoint: str, output_dir: str, run_name: str,
             n_correct_gradcam: int = 18, grid_cols: int = 5):
    device = torch.device("cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, class_names, val_acc = load_checkpoint(checkpoint, device)
    print(f"[evaluate:{run_name}] loaded checkpoint, classes={class_names}, reported val_acc={val_acc}")

    _, _, test_ds = build_datasets(data_dir)
    print(f"[evaluate:{run_name}] test set size: {len(test_ds)}")

    results = run_test_metrics(model, test_ds, class_names, device)
    print(f"[evaluate:{run_name}] TEST accuracy: {results['accuracy']:.4f}")
    print(results["classification_report_text"])

    cm_path = output_dir / f"{run_name}_confusion_matrix.png"
    plot_confusion_matrix(results["confusion_matrix"], class_names, cm_path, title=f"{run_name} test confusion matrix")

    metrics_path = output_dir / f"{run_name}_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "run_name": run_name,
            "class_names": class_names,
            "reported_val_acc": val_acc,
            "test_accuracy": results["accuracy"],
            "confusion_matrix": results["confusion_matrix"],
            "classification_report": results["classification_report"],
        }, f, indent=2)
    print(f"[evaluate:{run_name}] metrics saved to {metrics_path}")
    print(f"[evaluate:{run_name}] confusion matrix plot saved to {cm_path}")

    gradcam_results = run_gradcam_analysis(model, test_ds, class_names, device, output_dir, run_name,
                                            n_correct=n_correct_gradcam, cols=grid_cols, results=results)
    print(f"[evaluate:{run_name}] gradcam results: {gradcam_results}")

    return {
        "test_accuracy": results["accuracy"],
        "confusion_matrix": results["confusion_matrix"],
        "classification_report_text": results["classification_report_text"],
        "gradcam": gradcam_results,
        "val_acc": val_acc,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--n_correct_gradcam", type=int, default=18)
    p.add_argument("--grid_cols", type=int, default=5)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    evaluate(args.data_dir, args.checkpoint, args.output_dir, args.run_name,
              n_correct_gradcam=args.n_correct_gradcam, grid_cols=args.grid_cols)


if __name__ == "__main__":
    main()
