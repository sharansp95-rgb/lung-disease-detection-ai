"""
Investigates whether the scanner table/couch visible in CT slices
correlates with class (a potential shortcut, found via Grad-CAM on the
leakage-fixed CT model). Does NOT modify data/ct_scan/ -- read-only
investigation, run before deciding whether a crop fix or an exclusion
was the right call.

Method: isolate the body silhouette via an intensity threshold (background
~37, body tissue starts ~150+ in these 8-bit greyscale JPGs -- checked via
histogram inspection, not assumed), then check the region below the
torso's lower edge for a bright structure (the table). Sanity-checked
against known table-visible/table-absent examples before trusting it at
scale.

Also cross-checks image dimensions, since a distinct ~60-image Malignant
subset turned out to be both non-square AND table-absent -- multiple
correlated differences pointing at the same images, consistent with a
different original scan series/protocol rather than a simple crop-able
artifact.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
CT_DIR = ROOT / "data" / "ct_scan"
SPLITS = ["train", "val", "test"]
CLASSES = ["Benign", "Malignant", "Normal"]

BODY_THRESHOLD = 150  # background ~37, body tissue ~180-250 in these images
BRIGHT_PIXEL_MIN = 80  # min bright-pixel count below the body to call it "table detected"


def detect_table(img_path: Path):
    arr = np.array(Image.open(img_path).convert("L"))
    H, W = arr.shape

    body_mask = arr > BODY_THRESHOLD
    labeled, n = ndimage.label(body_mask)
    if n == 0:
        return False, 0.0
    sizes = ndimage.sum(body_mask, labeled, range(1, n + 1))
    largest_label = int(np.argmax(sizes)) + 1
    body = labeled == largest_label
    rows = np.any(body, axis=1)
    if not rows.any():
        return False, 0.0
    body_bottom = int(np.where(rows)[0][-1])

    below = arr[body_bottom + 1:H, :]
    if below.shape[0] < 3:
        return False, 0.0

    bright = below > BODY_THRESHOLD
    frame_frac = bright.sum() / (H * W)
    has_table = bright.sum() > BRIGHT_PIXEL_MIN
    return has_table, frame_frac


def main():
    records = []  # (split, cls, path, size, has_table)
    for split in SPLITS:
        for cls in CLASSES:
            d = CT_DIR / split / cls
            for p in sorted(d.glob("*.jpg")):
                im = Image.open(p)
                has_table, frac = detect_table(p)
                records.append((split, cls, p, im.size, has_table, frac))

    print("=== table visibility per class ===")
    for cls in CLASSES:
        cls_records = [r for r in records if r[1] == cls]
        n_table = sum(1 for r in cls_records if r[4])
        n_total = len(cls_records)
        print(f"{cls}: {n_table}/{n_total} ({100*n_table/n_total:.1f}%) table detected")

    print("\n=== non-square image dimensions per class ===")
    for cls in CLASSES:
        cls_records = [r for r in records if r[1] == cls]
        n_nonsquare = sum(1 for r in cls_records if r[3] != (512, 512))
        n_total = len(cls_records)
        print(f"{cls}: {n_nonsquare}/{n_total} ({100*n_nonsquare/n_total:.1f}%) non-square")

    # exclusion list: union of non-square and table-absent, Malignant only
    # (this is where the correlated anomaly was found -- see data/README.md)
    nonsquare_set = set()
    no_table_set = set()
    for split, cls, p, size, has_table, frac in records:
        if cls != "Malignant":
            continue
        key = f"{split}/{p.name}"
        if size != (512, 512):
            nonsquare_set.add(key)
        if not has_table:
            no_table_set.add(key)

    union = nonsquare_set | no_table_set
    overlap = nonsquare_set & no_table_set
    print(f"\nnon-square: {len(nonsquare_set)}, no-table: {len(no_table_set)}, "
          f"overlap: {len(overlap)}, union (exclusion list): {len(union)}")

    out = {
        "union": sorted(union),
        "nonsquare_only": sorted(nonsquare_set - no_table_set),
        "no_table_only": sorted(no_table_set - nonsquare_set),
        "overlap_count": len(overlap),
    }
    out_path = ROOT / "audit" / "ct_table_check" / "exclusion_list.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"exclusion list saved to {out_path}")

    return out


if __name__ == "__main__":
    main()
