"""
Build seg_data/{train,val,test}/{images,masks}/ from the lung-segmentation
masks that ship with raw/xray_covid/COVID-19_Radiography_Dataset/ across all
4 classes (COVID, Lung_Opacity, Normal, Viral Pneumonia) -- ~21,165
image+mask pairs total, per Phase 3.5's discovery when this dataset's masks
were first used (see scripts/prepare_xray.py).

This is data prep only -- no training, no GPU. Output is meant to later
train a lung-segmentation model, which /predict/xray's border-crop fix
(2026-08-25 investigation) explicitly could not stand in for: that fix only
reproduces the TB-source border crop, not the COVID-source mask-guided
lung-bbox crop, because arbitrary uploads have no mask. A trained
segmentation model would close that gap by generating a mask at inference
time instead of requiring one to already exist.

Unlike prepare_xray.py's per-class-subfolder output (data/xray/<split>/<class>/),
this is a flat images/+masks/ pair per split -- segmentation doesn't have a
per-image class label, just an image and its corresponding mask, matched by
identical filename across the two subfolders.

Split assignment happens on the raw file listing (no per-file validation
needed to decide train/val/test membership -- only the count matters for a
stratified shuffle). Validation, resize, and save all happen together in a
single pass per file so nothing is opened more than once.

Run:
    python scripts/prepare_seg_data.py
"""

import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 42
random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
COVID_SRC = ROOT / "raw" / "xray_covid" / "COVID-19_Radiography_Dataset"
OUT = ROOT / "seg_data"

SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
TARGET_SIZE = 256

# (class dir name, output filename tag)
CLASSES = [
    ("COVID", "covid"),
    ("Lung_Opacity", "lung_opacity"),
    ("Normal", "normal"),
    ("Viral Pneumonia", "viral_pneumonia"),
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

# A mask is "binary-ish" if at least this fraction of pixels sit near pure
# black or pure white -- catches masks that are actually grayscale photos,
# gradients, or otherwise not real segmentation masks.
BINARY_FRAC_THRESHOLD = 0.95
NEAR_BLACK = 15
NEAR_WHITE = 240

PROGRESS_EVERY = 2000


def process_pair(img_path: Path, mask_path: Path, img_dest: Path, mask_dest: Path) -> str | None:
    """Validates + resizes + saves one pair in a single pass. Returns None
    on success, or a skip reason string on failure (nothing is written for
    a failed pair)."""
    if not mask_path.exists():
        return "mask_missing"

    try:
        img = Image.open(img_path).convert("L")
        img.load()
    except Exception:
        return "image_corrupt_or_unreadable"

    try:
        mask = Image.open(mask_path).convert("L")
        mask.load()
    except Exception:
        return "mask_corrupt_or_unreadable"

    mask_arr = np.array(mask)
    if mask_arr.max() == 0:
        return "mask_all_zero"
    frac_extreme = float(np.mean((mask_arr < NEAR_BLACK) | (mask_arr > NEAR_WHITE)))
    if frac_extreme < BINARY_FRAC_THRESHOLD:
        return f"mask_not_binary_frac={frac_extreme:.3f}"

    img_resized = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    img_resized.save(img_dest)

    binary = np.where(mask_arr > 127, 255, 0).astype(np.uint8)
    mask_bin_img = Image.fromarray(binary, mode="L")
    mask_resized = mask_bin_img.resize((TARGET_SIZE, TARGET_SIZE), Image.NEAREST)
    # NEAREST only ever picks existing pixel values, but re-threshold
    # defensively so the saved mask is guaranteed strictly {0,255}.
    mask_resized_arr = np.where(np.array(mask_resized) > 127, 255, 0).astype(np.uint8)
    Image.fromarray(mask_resized_arr, mode="L").save(mask_dest)

    return None


def main():
    # --- enumerate raw pairs (listing only, no file opens yet) ---
    by_class: dict[str, list[tuple[Path, Path, str]]] = {tag: [] for _, tag in CLASSES}
    for class_dir, tag in CLASSES:
        images_dir = COVID_SRC / class_dir / "images"
        masks_dir = COVID_SRC / class_dir / "masks"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Expected images dir missing: {images_dir}")
        if not masks_dir.is_dir():
            raise FileNotFoundError(f"Expected masks dir missing: {masks_dir}")

        img_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        for img_path in img_files:
            mask_path = masks_dir / img_path.name
            by_class[tag].append((img_path, mask_path, img_path.stem))

    total_listed = sum(len(v) for v in by_class.values())
    print(f"Listed {total_listed} candidate pairs across {len(CLASSES)} classes.", flush=True)

    # --- stratified split on the listing (validation happens during save) ---
    split_items: dict[str, list[tuple[Path, Path, str, str]]] = {s: [] for s in SPLITS}
    listed_counts = {split: {tag: 0 for _, tag in CLASSES} for split in SPLITS}

    for _, tag in CLASSES:
        items = by_class[tag]
        rng = random.Random(SEED)
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * SPLITS["train"]))
        n_val = int(round(n * SPLITS["val"]))
        parts = {
            "train": items[:n_train],
            "val": items[n_train:n_train + n_val],
            "test": items[n_train + n_val:],
        }
        for split, part in parts.items():
            for img_path, mask_path, stem in part:
                split_items[split].append((img_path, mask_path, tag, stem))
            listed_counts[split][tag] = len(part)

    # --- clean and recreate output dirs ---
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in SPLITS:
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "masks").mkdir(parents=True, exist_ok=True)

    saved_counts = {split: {tag: 0 for _, tag in CLASSES} for split in SPLITS}
    skipped: list[tuple[str, str]] = []
    saved_samples: list[tuple[Path, Path]] = []
    done = 0

    for split, items in split_items.items():
        for img_path, mask_path, tag, stem in items:
            dest_name = f"{tag}-{stem}.png"
            img_dest = OUT / split / "images" / dest_name
            mask_dest = OUT / split / "masks" / dest_name

            reason = process_pair(img_path, mask_path, img_dest, mask_dest)
            done += 1
            if reason is not None:
                skipped.append((f"{tag}/{img_path.name}", reason))
            else:
                saved_counts[split][tag] += 1
                if split == "test" and len(saved_samples) < 4:
                    saved_samples.append((img_dest, mask_dest))

            if done % PROGRESS_EVERY == 0:
                print(f"  ...{done}/{total_listed} processed ({len(skipped)} skipped so far)", flush=True)

    print(f"\nDone: {done}/{total_listed} processed.", flush=True)
    if skipped:
        print(f"Skipped {len(skipped)} broken/missing pair(s):", flush=True)
        for ident, reason in skipped[:30]:
            print(f"  {reason}: {ident}", flush=True)
        if len(skipped) > 30:
            print(f"  ... and {len(skipped) - 30} more", flush=True)

    print()
    print("=== seg_data final counts ===")
    class_tags = [tag for _, tag in CLASSES]
    header = f"{'split':<8}" + "".join(f"{t:<18}" for t in class_tags) + "total"
    print(header)
    for split in SPLITS:
        row_total = sum(saved_counts[split][t] for t in class_tags)
        row = f"{split:<8}" + "".join(f"{saved_counts[split][t]:<18}" for t in class_tags) + str(row_total)
        print(row)

    grand_total = sum(sum(saved_counts[s].values()) for s in SPLITS)
    print(f"\nGrand total saved: {grand_total}")
    print(f"All images/masks: greyscale PNG, {TARGET_SIZE}x{TARGET_SIZE}px")

    return saved_counts, skipped, saved_samples


if __name__ == "__main__":
    main()
