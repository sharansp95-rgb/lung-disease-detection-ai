"""
Build data/ct_scan/{train,val,test}/{Normal,Benign,Malignant}/ from
raw/ct_scan/The IQ-OTHNCCD lung cancer dataset/ (hamdallak/the-iqothnccd-lung-cancer-dataset).

Source structure (confirmed by inspection, not assumed):
  "The IQ-OTHNCCD lung cancer dataset"/
    "Bengin cases"/"Bengin case (N).jpg"       (source spells it "Bengin", not "Benign")
    "Malignant cases"/"Malignant case (N).jpg"
    "Normal cases"/"Normal case (N).jpg"

Patient-level split was NOT possible: filenames are just a class-sequential
slice index (1..120 / 1..561 / 1..416, no gaps, no duplicates) with no
folder-per-patient structure and no accompanying manifest mapping image ->
patient. The dataset's own description.txt confirms 110 total patient cases
(40 malignant, 15 benign, 55 normal) but provides no way to recover which
slice belongs to which case from this specific download. Falling back to a
stratified per-image split with a fixed random seed, as instructed when no
patient identifier is recoverable.

No dedup: single dataset, not merged from multiple sources.
"""

import random
import shutil
from pathlib import Path
from PIL import Image

SEED = 42

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "raw" / "ct_scan" / "The IQ-OTHNCCD lung cancer dataset"
OUT = ROOT / "data" / "ct_scan"

SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}

# (source_dir_name, target_class)
SOURCE_DIRS = [
    ("Bengin cases", "Benign"),
    ("Malignant cases", "Malignant"),
    ("Normal cases", "Normal"),
]
CLASSES = [c for _, c in SOURCE_DIRS]

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def main():
    class_files: dict[str, list[Path]] = {c: [] for c in CLASSES}
    corrupt_files: list[Path] = []

    for dir_name, cls in SOURCE_DIRS:
        src_dir = SRC / dir_name
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Expected source dir missing: {src_dir}")
        files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        for p in files:
            if not is_valid_image(p):
                corrupt_files.append(p)
                continue
            class_files[cls].append(p)

    if corrupt_files:
        print(f"WARNING: {len(corrupt_files)} corrupted/unreadable image(s) skipped:")
        for p in corrupt_files:
            print(f"  {p}")

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
        n_val = int(round(n * SPLITS["val"]))
        splits_files = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        for split, split_list in splits_files.items():
            for src_path in split_list:
                dest = OUT / split / cls / src_path.name
                shutil.copy2(src_path, dest)
                counts[split][cls] += 1

    print()
    print("=== CT scan (IQ-OTH/NCCD) final counts ===")
    header = f"{'split':<8}" + "".join(f"{c:<14}" for c in CLASSES) + "total"
    print(header)
    for split in SPLITS:
        row_total = sum(counts[split][c] for c in CLASSES)
        row = f"{split:<8}" + "".join(f"{counts[split][c]:<14}" for c in CLASSES) + str(row_total)
        print(row)

    return counts, corrupt_files


if __name__ == "__main__":
    main()
