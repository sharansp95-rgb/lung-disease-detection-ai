"""
Generates the data-driven figures for the IEEE paper (pulmo_ieee_paper) directly
from real, on-disk evaluation/audit artifacts -- no invented numbers.

Figures produced (PDF, vector, IEEE-column-width friendly):
  fig_xray_confusion_matrix.pdf   -- from eval_outputs/xray_phase35_v2/xray_phase35_v2_test_metrics.json
  fig_ct_confusion_matrix.pdf     -- from eval_outputs/ct_scan_phase35_v2/ct_phase35_v2_test_metrics.json
  fig_prf1_bars.pdf               -- per-class precision/recall/F1 for both modalities, same two JSONs
  fig_ood_distribution.pdf        -- from backend/ood_stats/{xray,ct}_ood.pt (real training-set distances)
                                      plus one real non-scan sample's distance, computed live through the
                                      actual trained classifier's forward_features()

Run from the repo root:
    python scripts/generate_paper_figures.py --out_dir <path to paper's figures/ dir>
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
from dataset import build_transforms  # noqa: E402
from model import build_model  # noqa: E402

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

XRAY_METRICS = ROOT / "eval_outputs" / "xray_phase35_v2" / "xray_phase35_v2_test_metrics.json"
CT_METRICS = ROOT / "eval_outputs" / "ct_scan_phase35_v2" / "ct_phase35_v2_test_metrics.json"


def load_metrics(path: Path):
    with open(path) as f:
        return json.load(f)


def plot_confusion_matrix(cm, class_names, title, out_path: Path, accuracy: float):
    cm = np.array(cm)
    n = len(class_names)
    # Sized to print at true scale in a single IEEE column (~3.3-3.5in wide),
    # not scaled down from a larger canvas -- otherwise the 9pt label text
    # shrinks along with the image and becomes illegible at print size.
    side = 2.55 if n <= 3 else 2.85
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=40, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{title}\n(n = {cm.sum()}, accuracy = {accuracy*100:.2f}%)")
    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("count")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_prf1_bars(xray_report, ct_report, out_path: Path):
    # Sized for a single IEEE column (~3.3-3.5in), not the double-column
    # width -- printing near 1:1 keeps the 9pt labels legible instead of
    # shrinking along with a downscaled larger canvas.
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.05))
    metric_colors = {"precision": "#4477AA", "recall": "#EE6677", "f1-score": "#228833"}

    for ax, report, title in [
        (axes[0], xray_report, "X-ray (4-class)"),
        (axes[1], ct_report, "CT (3-class)"),
    ]:
        classes = [k for k in report.keys() if k not in ("accuracy", "macro avg", "weighted avg")]
        x = np.arange(len(classes))
        width = 0.26
        for i, metric in enumerate(["precision", "recall", "f1-score"]):
            vals = [report[c][metric] for c in classes]
            ax.bar(x + (i - 1) * width, vals, width, label=metric.capitalize(),
                   color=metric_colors[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=30, ha="right")
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def compute_ood_distance_for_sample(modality: str, image_path: Path, device: torch.device) -> float:
    checkpoint_name = {
        "xray": "xray_best_model_phase35_v2.pt",
        "ct": "ct_best_model_phase35_v2.pt",
    }[modality]
    ckpt = torch.load(ROOT / "checkpoints" / checkpoint_name, map_location=device, weights_only=False)
    model = build_model(ckpt["backbone"], len(ckpt["class_names"]), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = build_transforms(train=False)
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.forward_features(x)

    stats = torch.load(ROOT / "backend" / "ood_stats" / f"{modality}_ood.pt", map_location=device, weights_only=True)
    centroid = stats["centroid"].to(device)
    distance = torch.norm(feat[0] - centroid).item()
    return distance


def find_most_ood_sample(modality: str, device: torch.device, n_scan: int = 80):
    """Scans up to n_scan real not_scan images and returns the (path, distance)
    of whichever one sits JUST beyond the rejection threshold -- the smallest
    distance that still exceeds it. This is deliberately not the single most
    extreme outlier in the scanned batch: some non-scan images land orders of
    magnitude past the threshold, which would compress the entire training
    histogram into a few pixels on a shared linear axis. Picking the closest
    genuine rejection instead keeps the plot legible while still showing a
    real, verified rejection (many non-scan images do NOT exceed the OOD
    threshold on their own, which is precisely why the sanity gate, Section
    III-D, exists as a separate check)."""
    candidates_dir = ROOT / "data" / ("xray_sanity" if modality == "xray" else "ct_sanity") / "test" / "not_scan"
    paths = sorted(candidates_dir.glob("*"))[:n_scan]

    checkpoint_name = {
        "xray": "xray_best_model_phase35_v2.pt",
        "ct": "ct_best_model_phase35_v2.pt",
    }[modality]
    ckpt = torch.load(ROOT / "checkpoints" / checkpoint_name, map_location=device, weights_only=False)
    model = build_model(ckpt["backbone"], len(ckpt["class_names"]), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    transform = build_transforms(train=False)
    stats = torch.load(ROOT / "backend" / "ood_stats" / f"{modality}_ood.pt", map_location=device, weights_only=True)
    centroid = stats["centroid"].to(device)
    threshold = stats["threshold"]

    rejected = []
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            feat = model.forward_features(x)
            d = torch.norm(feat[0] - centroid).item()
            if d > threshold:
                rejected.append((p, d))
    if not rejected:
        raise RuntimeError(f"[{modality}] no rejected sample found in the first {n_scan} not_scan images")
    rejected.sort(key=lambda t: t[1])
    return rejected[0]


def plot_ood_distributions(out_path: Path, device: torch.device):
    # Sized for a single IEEE column, matching plot_prf1_bars -- see its
    # comment for why this is not the double-column width. Taller than a
    # naive aspect ratio to leave room for a shared legend + shared x-label
    # instead of two independent long x-labels colliding in the middle.
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.25))

    hist_handle = threshold_handle = rejected_handle = None
    for ax, modality, title in [(axes[0], "xray", "X-ray"), (axes[1], "ct", "CT")]:
        stats = torch.load(ROOT / "backend" / "ood_stats" / f"{modality}_ood.pt", map_location="cpu", weights_only=True)
        distances = stats["train_distances"].numpy()
        threshold = stats["threshold"]

        hist_handle = ax.hist(distances, bins=40, color="#4477AA", alpha=0.85,
                               label="training images")[2][0]
        threshold_handle = ax.axvline(threshold, color="#CC3311", linestyle="--", linewidth=1.4,
                                       label="99th-pct. threshold")

        rejected_path, rejected_distance = find_most_ood_sample(modality, device)
        rejected_handle = ax.axvline(rejected_distance, color="#228833", linestyle="-", linewidth=1.6,
                                      label="rejected non-scan sample")

        ax.set_title(f"{title} ($\\tau$={threshold:.1f}, n={len(distances)})", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        print(f"[{modality}] most-OOD sample {rejected_path.name}: distance={rejected_distance:.2f} "
              f"(threshold={threshold:.2f}, {'REJECTED by OOD detector' if rejected_distance > threshold else 'NOT rejected by OOD detector alone'})")

    axes[0].set_ylabel("Count", fontsize=7.5)
    fig.supxlabel("Euclidean distance from training centroid", fontsize=7.5)
    fig.legend([hist_handle, threshold_handle, rejected_handle],
               ["training images", "99th-pct. threshold", "rejected non-scan sample"],
               loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.14), fontsize=6.5, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xray = load_metrics(XRAY_METRICS)
    ct = load_metrics(CT_METRICS)

    plot_confusion_matrix(xray["confusion_matrix"], xray["class_names"],
                           "X-ray test confusion matrix", out_dir / "fig_xray_confusion_matrix.pdf",
                           xray["test_accuracy"])
    plot_confusion_matrix(ct["confusion_matrix"], ct["class_names"],
                           "CT test confusion matrix", out_dir / "fig_ct_confusion_matrix.pdf",
                           ct["test_accuracy"])
    plot_prf1_bars(xray["classification_report"], ct["classification_report"],
                    out_dir / "fig_prf1_bars.pdf")
    plot_ood_distributions(out_dir / "fig_ood_distribution.pdf", device)

    print("\nDone. X-ray test accuracy:", xray["test_accuracy"], " CT test accuracy:", ct["test_accuracy"])


if __name__ == "__main__":
    main()
