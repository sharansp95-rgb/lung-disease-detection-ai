"""
Phase 3.5 diagnostic: visualize proposed preprocessing fixes for both X-ray sources.

COVID-19 Radiography Dataset  → mask-guided lung bbox crop
TB Chest Radiography Database → uniform border crop at several margins

Saves a before/after grid to scripts/crop_preview/ for user sign-off.
"""

import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SEED = 42
random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
COVID_SRC   = ROOT / "raw" / "xray_covid" / "COVID-19_Radiography_Dataset"
TB_SRC      = ROOT / "raw" / "xray_tb_cleaned"
OUT_DIR     = ROOT / "scripts" / "crop_preview"
OUT_DIR.mkdir(exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))

def load_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return (arr > 127).astype(np.uint8)   # binarise


def mask_bbox(mask: np.ndarray, pad: int = 4):
    """Return (r0, r1, c0, c1) tight bounding box of the mask, with padding."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return 0, mask.shape[0], 0, mask.shape[1]
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    H, W = mask.shape
    r0 = max(0, r0 - pad)
    r1 = min(H, r1 + pad + 1)
    c0 = max(0, c0 - pad)
    c1 = min(W, c1 + pad + 1)
    return int(r0), int(r1), int(c0), int(c1)


def border_crop(img: np.ndarray, frac: float):
    """Remove `frac` fraction from each edge (0.08 = 8% crop per edge)."""
    H, W = img.shape[:2]
    dy = max(1, int(round(H * frac)))
    dx = max(1, int(round(W * frac)))
    return img[dy:H-dy, dx:W-dx]


def pil_resize(arr: np.ndarray, size=224) -> np.ndarray:
    return np.array(Image.fromarray(arr).resize((size, size), Image.LANCZOS))


def add_label(ax, text, fontsize=9):
    ax.set_title(text, fontsize=fontsize, pad=3)
    ax.axis("off")


# ── COVID: mask-guided crop preview ──────────────────────────────────────────

def covid_preview(n_samples=6):
    classes = ["COVID", "Normal", "Viral Pneumonia"]
    samples = []
    for cls in classes:
        img_dir  = COVID_SRC / cls / "images"
        mask_dir = COVID_SRC / cls / "masks"
        files = sorted(img_dir.glob("*.png"))
        rng = random.Random(SEED)
        picked = rng.sample(files, min(n_samples // len(classes) + 1, len(files)))
        for p in picked[:max(1, n_samples // len(classes))]:
            mp = mask_dir / p.name
            if mp.exists():
                samples.append((cls, p, mp))

    ncols = 3   # original | mask overlay | cropped-resized
    nrows = len(samples)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    fig.suptitle("COVID-19 dataset — mask-guided lung bbox crop", fontsize=11, y=1.01)

    for i, (cls, img_path, mask_path) in enumerate(samples):
        img  = load_gray(img_path)
        mask = load_mask(mask_path)
        r0, r1, c0, c1 = mask_bbox(mask, pad=4)

        # col 0: original
        ax = axes[i, 0] if nrows > 1 else axes[0]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        add_label(ax, f"Original  [{cls}]\n{img.shape[1]}×{img.shape[0]}")

        # col 1: mask overlay + bbox
        ax = axes[i, 1] if nrows > 1 else axes[1]
        # Resize mask to match image dimensions if they differ
        if mask.shape != img.shape:
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img = mask_img.resize((img.shape[1], img.shape[0]), Image.NEAREST)
            mask = (np.array(mask_img) > 127).astype(np.uint8)
            # Recompute bbox with correctly-sized mask
            r0, r1, c0, c1 = mask_bbox(mask, pad=4)
        overlay = np.stack([img, img, img], axis=-1)
        overlay[mask == 0] = [40, 40, 40]          # dim background
        rect = plt.Rectangle((c0, r0), c1-c0, r1-r0,
                              edgecolor="lime", facecolor="none", linewidth=1.5)
        ax.imshow(overlay)
        ax.add_patch(rect)
        add_label(ax, f"Mask + bbox\ncrop: [{r0}:{r1}, {c0}:{c1}]")

        # col 2: cropped + resized to 224
        ax = axes[i, 2] if nrows > 1 else axes[2]
        cropped = img[r0:r1, c0:c1]
        resized = pil_resize(cropped, 224)
        ax.imshow(resized, cmap="gray", vmin=0, vmax=255)
        add_label(ax, f"Cropped → 224×224\n(mask-guided)")

    plt.tight_layout()
    out = OUT_DIR / "covid_mask_crop_preview.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── TB: border-crop preview at multiple margins ───────────────────────────────

def tb_preview(n_samples=4, margins=(0.05, 0.08, 0.10, 0.12)):
    tb_files = sorted((TB_SRC / "Tuberculosis").glob("*.png"))
    norm_files = sorted((TB_SRC / "Normal").glob("*.png"))

    rng = random.Random(SEED)
    tb_pick   = rng.sample(tb_files,   min(n_samples // 2, len(tb_files)))
    norm_pick = rng.sample(norm_files, min(n_samples // 2, len(norm_files)))
    samples = [("Tuberculosis", p) for p in tb_pick] + [("Normal", p) for p in norm_pick]

    ncols = 1 + len(margins)   # original + one per margin
    nrows = len(samples)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    fig.suptitle("TB dataset — uniform border crop at candidate margins", fontsize=11, y=1.01)

    for i, (cls, img_path) in enumerate(samples):
        img = load_gray(img_path)

        ax = axes[i, 0] if nrows > 1 else axes[0]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        add_label(ax, f"Original [{cls}]\n{img.shape[1]}×{img.shape[0]}")

        for j, frac in enumerate(margins):
            cropped = border_crop(img, frac)
            resized = pil_resize(cropped, 224)
            ax = axes[i, j+1] if nrows > 1 else axes[j+1]
            ax.imshow(resized, cmap="gray", vmin=0, vmax=255)
            add_label(ax, f"{int(frac*100)}% per edge\n→ 224×224")

    plt.tight_layout()
    out = OUT_DIR / "tb_border_crop_preview.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── TB: corner-region zoom to verify watermark location ──────────────────────

def tb_corner_zoom(n_samples=6, crop_frac=0.15):
    """
    Zoom into the four corners of TB + Normal images to see whether
    watermarks are consistently placed there.
    """
    tb_files   = sorted((TB_SRC / "Tuberculosis").glob("*.png"))
    norm_files = sorted((TB_SRC / "Normal").glob("*.png"))
    rng = random.Random(SEED + 1)
    picks = ([("TB", p) for p in rng.sample(tb_files, min(n_samples//2, len(tb_files)))]
           + [("Normal", p) for p in rng.sample(norm_files, min(n_samples//2, len(norm_files)))])

    ncols = 5   # full image + TL + TR + BL + BR
    nrows = len(picks)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    fig.suptitle("TB dataset — corner zoom (watermark audit)", fontsize=11, y=1.01)

    for i, (cls, p) in enumerate(picks):
        img = load_gray(p)
        H, W = img.shape
        dy, dx = max(1, int(H * crop_frac)), max(1, int(W * crop_frac))

        corners = {
            "TL": img[:dy, :dx],
            "TR": img[:dy, W-dx:],
            "BL": img[H-dy:, :dx],
            "BR": img[H-dy:, W-dx:],
        }

        ax = axes[i, 0] if nrows > 1 else axes[0]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        add_label(ax, f"[{cls}] full")

        for j, (corner, patch) in enumerate(corners.items()):
            ax = axes[i, j+1] if nrows > 1 else axes[j+1]
            ax.imshow(patch, cmap="gray", vmin=0, vmax=255)
            add_label(ax, corner)

    plt.tight_layout()
    out = OUT_DIR / "tb_corner_zoom.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Generating COVID mask-guided crop preview...")
    covid_preview(n_samples=6)

    print("Generating TB corner zoom...")
    tb_corner_zoom(n_samples=6, crop_frac=0.15)

    print("Generating TB border-crop comparison at candidate margins...")
    tb_preview(n_samples=4, margins=(0.05, 0.08, 0.10, 0.12))

    print("\nAll previews written to:", OUT_DIR)
