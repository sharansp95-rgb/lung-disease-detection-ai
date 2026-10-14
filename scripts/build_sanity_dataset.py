"""
Assembles the ImageFolder-shaped dataset(s) for training the sanity gate
(train_sanity_gate.py uses ml/train.py's standard --data_dir train/val/test
+ class-subfolder convention, so this script's only job is to produce that
shape from the real modality images and the negative pool built by
prepare_sanity_negatives.py).

For each modality (xray, ct):
    data/{modality}_sanity/{train,val,test}/scan/*      <- hardlinked from
                                                            data/{modality dir}/{split}/*/*
    data/{modality}_sanity/{train,val,test}/not_scan/*  <- hardlinked from a
                                                            (possibly subsampled)
                                                            selection of
                                                            data/sanity_negatives/{split}/*/*

Hardlinks (not copies) since source and destination are on the same NTFS
volume -- instant, zero extra disk space, and safe here because nothing
downstream ever writes into the sanity dataset folders.

Negative counts per split are capped (see NEG_CAP_TRAIN) rather than always
using the full negative pool: CT's positive class is much smaller (829
train images) than X-ray's (15373), so uncapped negatives would make CT
training overwhelmingly negative-heavy despite the weighted loss already
handling the class imbalance ml/train.py computes automatically.
"""

import os
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NEG_DIR = ROOT / "data" / "sanity_negatives"
SEED = 4321

MODALITY_DIRS = {
    "xray": ROOT / "data" / "xray",
    "ct": ROOT / "data" / "ct_scan",
}

# train negatives capped at max(3x positive count, 3000), val/test use the
# full negative pool for an accurate false-positive-rate read.
NEG_CAP_TRAIN_FLOOR = 3000
NEG_CAP_TRAIN_MULT = 3


def link_all(src_paths, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(src_paths):
        dst = dst_dir / f"{i:06d}_{src.name}"
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)


def build_for_modality(modality: str):
    src_dir = MODALITY_DIRS[modality]
    out_dir = ROOT / "data" / f"{modality}_sanity"
    rng = random.Random(SEED + hash(modality) % 1000)

    for split in ("train", "val", "test"):
        positives = [f for f in (src_dir / split).rglob("*") if f.is_file()]
        negatives_all = [f for f in (NEG_DIR / split).rglob("*") if f.is_file()]
        rng.shuffle(negatives_all)

        if split == "train":
            cap = max(len(positives) * NEG_CAP_TRAIN_MULT, NEG_CAP_TRAIN_FLOOR)
            negatives = negatives_all[:cap]
        else:
            negatives = negatives_all

        link_all(positives, out_dir / split / "scan")
        link_all(negatives, out_dir / split / "not_scan")
        print(f"[{modality}/{split}] scan={len(positives)} not_scan={len(negatives)} "
              f"(pool={len(negatives_all)})")


def main():
    for modality in ("xray", "ct"):
        build_for_modality(modality)


if __name__ == "__main__":
    main()
