"""
Generates the segmentation-crop comparison figure (G5) for the IEEE paper by
calling the ACTUAL inference-time code path in backend/seg_utils.py against a
real X-ray sample -- not a hand-drawn illustration.

Panels: (1) raw uploaded image, (2) predicted U-Net lung mask, (3) resulting
segmentation-guided crop, (4) the border-crop fallback the same code path uses
when the mask is degenerate (scripts/prepare_xray.py's border_crop(), reused
directly, exactly as backend/seg_utils.py does).

Run from the repo root:
    python scripts/generate_seg_crop_figure.py --out_dir <path to paper's figures/ dir>
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "ml"))

from seg_utils import load_seg_bundle, _to_seg_tensor, SEG_INPUT_SIZE  # noqa: E402
from prepare_xray import border_crop, mask_bbox, MASK_PAD_PX, TB_BORDER_FRAC  # noqa: E402

SEG_CHECKPOINT = ROOT / "checkpoints" / "xray_seg_model.pt"
SEG_DATA_DIR = ROOT / "seg_data"

plt.rcParams.update({
    "font.size": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sample", default=None, help="path to a real X-ray sample; defaults to a test/Normal image")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_path = Path(args.sample) if args.sample else next(
        (ROOT / "data" / "xray" / "test" / "Tuberculosis").glob("*.png")
    )
    print(f"using sample: {sample_path}")

    print("loading segmentation bundle (runs the real reproduction check)...")
    seg_bundle = load_seg_bundle(SEG_CHECKPOINT, SEG_DATA_DIR)

    image = Image.open(sample_path).convert("RGB")
    gray = np.array(image.convert("L"))

    with seg_bundle.lock:
        x = _to_seg_tensor(image, seg_bundle.device)
        with torch.no_grad():
            logits = seg_bundle.model(x)
            probs = torch.sigmoid(logits)
            mask_small = (probs[0, 0] > 0.5).cpu().numpy().astype(np.uint8)

    r0, r1, c0, c1 = mask_bbox(mask_small, pad=MASK_PAD_PX)
    H, W = gray.shape
    scale_r, scale_c = H / SEG_INPUT_SIZE, W / SEG_INPUT_SIZE
    orig_r0, orig_r1 = int(r0 * scale_r), int(r1 * scale_r)
    orig_c0, orig_c1 = int(c0 * scale_c), int(c1 * scale_c)
    seg_crop = gray[orig_r0:orig_r1, orig_c0:orig_c1]

    border_cropped = border_crop(gray, TB_BORDER_FRAC)

    # Sized for a single IEEE column (~3.3-3.5in) so it prints near 1:1
    # rather than shrinking from a double-column-sized canvas.
    fig, axes = plt.subplots(1, 4, figsize=(3.4, 1.05))
    panels = [
        (gray, "(a) Raw input"),
        (mask_small * 255, "(b) U-Net predicted mask\n(256×256)"),
        (seg_crop, "(c) Segmentation-guided crop\n(applied to full-res input)"),
        (border_cropped, "(d) Border-crop fallback\n(uniform 8% margin)"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=5.2)
        ax.axis("off")

    fig.tight_layout()
    out_path = out_dir / "fig_seg_crop_comparison.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"mask bbox (256-space): rows [{r0},{r1}) cols [{c0},{c1}); "
          f"bbox_area_frac={((r1-r0)*(c1-c0))/(SEG_INPUT_SIZE*SEG_INPUT_SIZE):.4f}")


if __name__ == "__main__":
    main()
