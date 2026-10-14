"""
Builds data/modality_gate/{train,val,test}/{xray,ct}/ -- a binary
X-ray-vs-CT dataset, reusing the existing train/val/test split assignment
from each modality's own pipeline (no new splitting logic, no leakage risk
introduced here since each source image keeps whatever split it was
already assigned).

All data/xray/{split}/<disease_class>/* -> modality_gate/{split}/xray/*
All data/ct_scan/{split}/<disease_class>/* -> modality_gate/{split}/ct/*

Filenames are prefixed with their original disease-class folder name to
avoid any cross-class collision within the flattened xray/ct buckets.
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
XRAY_DIR = ROOT / "data" / "xray"
CT_DIR = ROOT / "data" / "ct_scan"
OUT = ROOT / "data" / "modality_gate"

SPLITS = ["train", "val", "test"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def main():
    if OUT.exists():
        shutil.rmtree(OUT)

    counts = {split: {"xray": 0, "ct": 0} for split in SPLITS}

    for split in SPLITS:
        out_xray = OUT / split / "xray"
        out_ct = OUT / split / "ct"
        out_xray.mkdir(parents=True, exist_ok=True)
        out_ct.mkdir(parents=True, exist_ok=True)

        for cls_dir in sorted((XRAY_DIR / split).iterdir()):
            if not cls_dir.is_dir():
                continue
            for p in cls_dir.iterdir():
                if p.suffix.lower() in IMAGE_EXTS:
                    dest = out_xray / f"{cls_dir.name}_{p.name}"
                    shutil.copy2(p, dest)
                    counts[split]["xray"] += 1

        for cls_dir in sorted((CT_DIR / split).iterdir()):
            if not cls_dir.is_dir():
                continue
            for p in cls_dir.iterdir():
                if p.suffix.lower() in IMAGE_EXTS:
                    dest = out_ct / f"{cls_dir.name}_{p.name}"
                    shutil.copy2(p, dest)
                    counts[split]["ct"] += 1

    print("=== modality_gate counts ===")
    header = f"{'split':<8}{'xray':<10}{'ct':<10}total"
    print(header)
    for split in SPLITS:
        total = counts[split]["xray"] + counts[split]["ct"]
        print(f"{split:<8}{counts[split]['xray']:<10}{counts[split]['ct']:<10}{total}")

    return counts


if __name__ == "__main__":
    main()
