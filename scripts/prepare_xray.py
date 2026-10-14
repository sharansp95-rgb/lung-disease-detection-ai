"""
Build data/xray/{train,val,test}/{COVID-19,Normal,Pneumonia,Tuberculosis}/
from two raw Kaggle sources:
  - raw/xray_covid/COVID-19_Radiography_Dataset/  (COVID, Lung_Opacity, Normal, Viral Pneumonia)
  - raw/xray_tb_cleaned/                          (Normal, Tuberculosis) -- artifact-cleaned
    mirror of raw/xray_tb/TB_Chest_Radiography_Database/, produced by
    tb_artifact_detect.py + tb_clean.py after a data-quality audit found
    burned-in color tint / redaction boxes correlated heavily with the
    Tuberculosis label (see data/README.md, "TB data-quality audit").

Both sources contain a "Normal" class -> dedup across sources by file content hash
(not filename, since filenames like "Normal-1.png" collide by coincidence, not identity).

Split: 80/10/10 per class, fixed random seed, copy (not move).

Phase 3.5 preprocessing fixes (shortcut-learning mitigation):
  COVID-19 Radiography Dataset:
    Real per-image lung-segmentation masks exist under <class>/masks/ alongside
    <class>/images/. Each mask is a binary silhouette of the lung field.
    Fix: load mask, compute tight bounding box of white pixels (+4 px padding),
    crop the image to that bbox, then resize to 224x224.  Corner text / watermarks
    that live outside the lung field are removed at the source.
    Degenerate masks (all-zero or bbox < MIN_BBOX_PX on either side) fall back to
    a 5% border crop and emit a warning.

  TB Chest Radiography Database (no masks):
    Watermarks are consistently confined to the outer border.  Fix: uniform
    TB_BORDER_FRAC crop from each edge before resize.
    Default: 8% (chosen from visual inspection in inspect_crops.py).
"""

import hashlib
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 42
random.seed(SEED)

ROOT     = Path(__file__).resolve().parent.parent
COVID_SRC = ROOT / "raw" / "xray_covid" / "COVID-19_Radiography_Dataset"
TB_SRC    = ROOT / "raw" / "xray_tb_cleaned"
OUT       = ROOT / "data" / "xray"

SPLITS  = {"train": 0.8, "val": 0.1, "test": 0.1}
CLASSES = ["COVID-19", "Normal", "Pneumonia", "Tuberculosis"]

TARGET_SIZE  = 224          # final square size after crop
MASK_PAD_PX  = 4            # pixels to add around tight lung bbox
MIN_BBOX_PX  = 64           # masks smaller than this -> fallback border crop
FALLBACK_FRAC = 0.05        # fallback border fraction for degenerate masks
TB_BORDER_FRAC = 0.08       # uniform border crop for TB source (8% per edge)

# (source_img_dir, target_class, source_tag, crop_mode, mask_dir_or_None)
# Lung_Opacity intentionally excluded (same decision as Phase 1).
SOURCES = [
    (COVID_SRC / "COVID"           / "images", "COVID-19",     "covid19db", "mask",   COVID_SRC / "COVID"           / "masks"),
    (COVID_SRC / "Normal"          / "images", "Normal",       "covid19db", "mask",   COVID_SRC / "Normal"          / "masks"),
    (COVID_SRC / "Viral Pneumonia" / "images", "Pneumonia",    "covid19db", "mask",   COVID_SRC / "Viral Pneumonia" / "masks"),
    (TB_SRC    / "Normal",                     "Normal",       "tbdb",      "border", None),
    (TB_SRC    / "Tuberculosis",               "Tuberculosis", "tbdb",      "border", None),
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


# ── image utilities ──────────────────────────────────────────────────────────

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def load_gray(path: Path) -> np.ndarray:
    """Load any image as uint8 greyscale numpy array."""
    return np.array(Image.open(path).convert("L"))


def load_mask_binary(mask_path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    """
    Load a mask image, binarise at threshold 127, and resize (nearest-neighbour)
    to match the target image dimensions (H, W).
    """
    mask_arr = np.array(Image.open(mask_path).convert("L"))
    binary = (mask_arr > 127).astype(np.uint8)
    H, W = target_hw
    if binary.shape != (H, W):
        pil = Image.fromarray(binary * 255)
        pil = pil.resize((W, H), Image.NEAREST)
        binary = (np.array(pil) > 127).astype(np.uint8)
    return binary


def mask_bbox(mask: np.ndarray, pad: int = MASK_PAD_PX) -> tuple[int, int, int, int]:
    """Tight bounding box of white pixels in mask, with padding."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return 0, mask.shape[0], 0, mask.shape[1]
    r0, r1 = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
    c0, c1 = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
    H, W = mask.shape
    return (max(0, r0 - pad), min(H, r1 + pad + 1),
            max(0, c0 - pad), min(W, c1 + pad + 1))


def border_crop(arr: np.ndarray, frac: float) -> np.ndarray:
    """Remove `frac` fraction from each edge."""
    H, W = arr.shape[:2]
    dy = max(1, int(round(H * frac)))
    dx = max(1, int(round(W * frac)))
    return arr[dy:H - dy, dx:W - dx]


def crop_and_resize(arr: np.ndarray, size: int = TARGET_SIZE) -> Image.Image:
    """Convert numpy greyscale crop to PIL, resize to size×size."""
    return Image.fromarray(arr).resize((size, size), Image.LANCZOS)


# ── per-source crop logic ────────────────────────────────────────────────────

def process_covid(img_path: Path, mask_dir: Path,
                  degenerate_log: list) -> Image.Image:
    """
    Load image + corresponding mask, crop to lung bbox, resize.
    Falls back to border crop if mask is degenerate.
    Returns a PIL Image (greyscale, 224×224).
    """
    img = load_gray(img_path)
    H, W = img.shape
    mask_path = mask_dir / img_path.name
    if not mask_path.exists():
        # No mask file — fall back to border crop
        degenerate_log.append((img_path, "mask_missing"))
        cropped = border_crop(img, FALLBACK_FRAC)
        return crop_and_resize(cropped)

    mask = load_mask_binary(mask_path, (H, W))
    if mask.sum() == 0:
        degenerate_log.append((img_path, "mask_all_zero"))
        cropped = border_crop(img, FALLBACK_FRAC)
        return crop_and_resize(cropped)

    r0, r1, c0, c1 = mask_bbox(mask)
    if (r1 - r0) < MIN_BBOX_PX or (c1 - c0) < MIN_BBOX_PX:
        degenerate_log.append((img_path, f"bbox_too_small_{r1-r0}x{c1-c0}"))
        cropped = border_crop(img, FALLBACK_FRAC)
        return crop_and_resize(cropped)

    cropped = img[r0:r1, c0:c1]
    return crop_and_resize(cropped)


def process_tb(img_path: Path) -> Image.Image:
    """
    Uniform 8% border crop then resize to 224×224.
    Returns a PIL Image (greyscale, 224×224).
    """
    img = load_gray(img_path)
    cropped = border_crop(img, TB_BORDER_FRAC)
    return crop_and_resize(cropped)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    class_files: dict[str, list[tuple[Path, str, str, Path | None]]] = {c: [] for c in CLASSES}
    seen_hash:   dict[str, set[str]]                                  = {c: set() for c in CLASSES}
    dup_count    = {c: 0 for c in CLASSES}
    corrupt_files: list[Path] = []
    degenerate_masks: list[tuple[Path, str]] = []

    for src_dir, cls, tag, crop_mode, mask_dir in SOURCES:
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Expected source dir missing: {src_dir}")
        files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        for p in files:
            if not is_valid_image(p):
                corrupt_files.append(p)
                continue
            h = file_hash(p)
            if h in seen_hash[cls]:
                dup_count[cls] += 1
                continue
            seen_hash[cls].add(h)
            # Store: (source_path, dest_filename, crop_mode, mask_dir)
            class_files[cls].append((p, f"{tag}_{p.stem}.png", crop_mode, mask_dir))

    if corrupt_files:
        print(f"WARNING: {len(corrupt_files)} corrupted/unreadable image(s) skipped:")
        for p in corrupt_files:
            print(f"  {p}")

    print("Duplicate images removed per class (content-hash dedup):")
    for c in CLASSES:
        print(f"  {c}: {dup_count[c]} duplicates removed")

    # Clean and recreate output dirs.
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in SPLITS:
        for cls in CLASSES:
            (OUT / split / cls).mkdir(parents=True, exist_ok=True)

    counts = {split: {c: 0 for c in CLASSES} for split in SPLITS}

    for cls in CLASSES:
        files = class_files[cls]
        rng = random.Random(SEED)
        rng.shuffle(files)

        n = len(files)
        n_train = int(round(n * SPLITS["train"]))
        n_val   = int(round(n * SPLITS["val"]))
        splits_files = {
            "train": files[:n_train],
            "val":   files[n_train:n_train + n_val],
            "test":  files[n_train + n_val:],
        }

        for split, split_list in splits_files.items():
            for src_path, dest_name, crop_mode, mask_dir in split_list:
                dest = OUT / split / cls / dest_name
                try:
                    if crop_mode == "mask":
                        pil_img = process_covid(src_path, mask_dir, degenerate_masks)
                    else:  # "border"
                        pil_img = process_tb(src_path)
                    pil_img.save(dest)
                    counts[split][cls] += 1
                except Exception as e:
                    print(f"WARNING: failed to process {src_path}: {e}")

    if degenerate_masks:
        print(f"\nWARNING: {len(degenerate_masks)} images used fallback border crop (degenerate mask):")
        for p, reason in degenerate_masks[:20]:
            print(f"  {reason}: {p.name}")
        if len(degenerate_masks) > 20:
            print(f"  ... and {len(degenerate_masks)-20} more")

    print()
    print("=== X-ray final counts ===")
    header = f"{'split':<8}" + "".join(f"{c:<22}" for c in CLASSES) + "total"
    print(header)
    for split in SPLITS:
        row_total = sum(counts[split][c] for c in CLASSES)
        row = f"{split:<8}" + "".join(f"{counts[split][c]:<22}" for c in CLASSES) + str(row_total)
        print(row)

    print()
    print(f"Preprocessing: COVID -> mask-guided lung bbox crop (+{MASK_PAD_PX}px pad)")
    print(f"               TB    -> uniform {int(TB_BORDER_FRAC*100)}% border crop per edge")
    print(f"All output images: greyscale PNG, {TARGET_SIZE}x{TARGET_SIZE}px")

    return counts, dup_count, corrupt_files, degenerate_masks


if __name__ == "__main__":
    main()
