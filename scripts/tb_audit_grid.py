"""
Build side-by-side inspection grids of random Normal vs Tuberculosis images
from raw/xray_tb/, to visually check for burned-in artifacts (text markers,
redaction boxes, L/R letters, annotation overlays) that differ systematically
between the two classes and could let a model cheat instead of learning the
actual pathology.

Does not touch data/xray/ -- this is a read-only inspection script.
"""

import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SEED = 42
N_SAMPLES = 36  # 6x6 grid
THUMB = 220
COLS = 6

ROOT = Path(__file__).resolve().parent.parent
TB_SRC = ROOT / "raw" / "xray_tb" / "TB_Chest_Radiography_Database"
OUT_DIR = ROOT / "audit"


def build_grid(src_dir: Path, label: str, out_path: Path):
    files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    rng = random.Random(SEED)
    sample = rng.sample(files, min(N_SAMPLES, len(files)))
    sample.sort(key=lambda p: p.name)

    rows = (len(sample) + COLS - 1) // COLS
    pad = 4
    label_h = 18
    grid_w = COLS * (THUMB + pad) + pad
    grid_h = rows * (THUMB + pad + label_h) + pad + 30

    canvas = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 4), f"{label}  (n={len(sample)} of {len(files)}, seed={SEED})", fill="black")

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, p in enumerate(sample):
        row, col = divmod(i, COLS)
        x = pad + col * (THUMB + pad)
        y = 30 + pad + row * (THUMB + pad + label_h)
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail((THUMB, THUMB))
            paste_x = x + (THUMB - im.width) // 2
            paste_y = y + (THUMB - im.height) // 2
            canvas.paste(im, (paste_x, paste_y))
        draw.rectangle([x, y, x + THUMB, y + THUMB], outline="lightgray")
        draw.text((x, y + THUMB + 2), p.name[:28], fill="black", font=font)

    canvas.save(out_path)
    print(f"{label}: sampled {len(sample)}/{len(files)} -> {out_path}")
    return sample


def main():
    OUT_DIR.mkdir(exist_ok=True)
    normal_sample = build_grid(TB_SRC / "Normal", "Normal", OUT_DIR / "tb_audit_normal_grid.png")
    tb_sample = build_grid(TB_SRC / "Tuberculosis", "Tuberculosis", OUT_DIR / "tb_audit_tuberculosis_grid.png")
    return normal_sample, tb_sample


if __name__ == "__main__":
    main()
