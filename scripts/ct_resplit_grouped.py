"""
Cluster-aware (group-aware) resplit for data/ct_scan/, using the
near-duplicate clusters found by ct_leakage_audit.py as a patient proxy.

Every image in a given cluster is treated as one indivisible group and
lands entirely in one split -- never split across train/val/test. Images
not in any cluster (singletons) are free to be split individually.

Per class, groups are assigned via a largest-first greedy bin-balancing
heuristic: sort groups by size descending (seeded shuffle for tie-breaks),
and repeatedly hand the next group to whichever split is currently furthest
below its 80/10/10 target for that class. This gets close to 80/10/10
where cluster sizes allow, but the actual achieved ratio is what's reported
-- forcing an exact split would violate the no-split-clusters constraint.

Copies from the raw source (same source prepare_ct_scan.py uses), not from
the old data/ct_scan/, so this is a clean rebuild.
"""

import json
import random
import shutil
from pathlib import Path
from PIL import Image

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
RAW_SRC = ROOT / "raw" / "ct_scan" / "The IQ-OTHNCCD lung cancer dataset"
OLD_DATA_DIR = ROOT / "data" / "ct_scan"
AUDIT_JSON = ROOT / "audit" / "ct_leakage_audit.json"
OUT = ROOT / "data" / "ct_scan"

SPLIT_TARGETS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLITS = list(SPLIT_TARGETS.keys())

# (raw source dir name, target class) -- same mapping as prepare_ct_scan.py
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


def build_full_roster():
    """(class, filename) -> raw source Path, for every image in the dataset."""
    roster = {}
    for dir_name, cls in SOURCE_DIRS:
        src_dir = RAW_SRC / dir_name
        for p in sorted(src_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                roster[(cls, p.name)] = p
    return roster


def parse_old_path(old_path_str: str):
    """'data\\ct_scan\\train\\Malignant\\Malignant case (6).jpg' -> (class, filename)"""
    parts = old_path_str.replace("\\", "/").split("/")
    cls = parts[-2]
    filename = parts[-1]
    return cls, filename


def build_groups(roster: dict):
    with open(AUDIT_JSON) as f:
        audit = json.load(f)

    clustered_keys = set()
    groups = []  # list of (class, [ (class,filename), ... ])
    for c in audit["clusters"]:
        keys = [parse_old_path(f) for f in c["files"]]
        classes_here = set(k[0] for k in keys)
        assert len(classes_here) == 1, f"unexpected mixed-class cluster: {classes_here}"
        cls = next(iter(classes_here))
        groups.append((cls, keys))
        clustered_keys.update(keys)

    # singletons: every roster image not already in a cluster
    for key, path in roster.items():
        if key not in clustered_keys:
            groups.append((key[0], [key]))

    return groups


def assign_groups(groups: list):
    rng = random.Random(SEED)

    class_totals = {c: 0 for c in CLASSES}
    for cls, keys in groups:
        class_totals[cls] += len(keys)

    targets = {
        cls: {s: class_totals[cls] * frac for s, frac in SPLIT_TARGETS.items()}
        for cls in CLASSES
    }
    current = {cls: {s: 0 for s in SPLITS} for cls in CLASSES}

    assignment = {}  # (class, filename) -> split

    for cls in CLASSES:
        cls_groups = [g for g in groups if g[0] == cls]
        rng.shuffle(cls_groups)
        cls_groups.sort(key=lambda g: -len(g[1]))  # largest first, stable after shuffle

        for _, keys in cls_groups:
            deficits = {s: targets[cls][s] - current[cls][s] for s in SPLITS}
            best_split = max(SPLITS, key=lambda s: deficits[s])
            for key in keys:
                assignment[key] = best_split
            current[cls][best_split] += len(keys)

    return assignment, current, targets, class_totals


def rebuild_data_dir(roster: dict, assignment: dict):
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in SPLITS:
        for cls in CLASSES:
            (OUT / split / cls).mkdir(parents=True, exist_ok=True)

    counts = {split: {c: 0 for c in CLASSES} for split in SPLITS}
    corrupt_files = []

    for key, path in roster.items():
        cls, filename = key
        split = assignment[key]
        if not is_valid_image(path):
            corrupt_files.append(path)
            continue
        dest = OUT / split / cls / filename
        shutil.copy2(path, dest)
        counts[split][cls] += 1

    return counts, corrupt_files


def verify_no_cross_split_clusters(assignment: dict):
    with open(AUDIT_JSON) as f:
        audit = json.load(f)
    violations = []
    for c in audit["clusters"]:
        keys = [parse_old_path(f) for f in c["files"]]
        splits_in_cluster = set(assignment[k] for k in keys)
        if len(splits_in_cluster) > 1:
            violations.append((c, splits_in_cluster))
    return violations


def main():
    roster = build_full_roster()
    print(f"[ct_resplit] full roster: {len(roster)} images")

    groups = build_groups(roster)
    n_singletons = sum(1 for _, keys in groups if len(keys) == 1)
    n_clusters = sum(1 for _, keys in groups if len(keys) > 1)
    print(f"[ct_resplit] groups: {len(groups)} total ({n_clusters} clusters, {n_singletons} singletons)")

    assignment, current, targets, class_totals = assign_groups(groups)

    print()
    print("=== Per-class split counts (achieved vs 80/10/10 target) ===")
    for cls in CLASSES:
        total = class_totals[cls]
        row = f"{cls:<12} total={total:<5}"
        for s in SPLITS:
            achieved = current[cls][s]
            pct = 100 * achieved / total if total else 0
            row += f" {s}={achieved:<4}({pct:4.1f}%)"
        print(row)

    overall_total = sum(class_totals.values())
    overall = {s: sum(current[cls][s] for cls in CLASSES) for s in SPLITS}
    print()
    print("=== Overall split ratios ===")
    for s in SPLITS:
        pct = 100 * overall[s] / overall_total
        print(f"  {s}: {overall[s]} images ({pct:.1f}%)")

    violations = verify_no_cross_split_clusters(assignment)
    print()
    if violations:
        print(f"[ct_resplit] FAILED: {len(violations)} clusters still span multiple splits!")
        for c, splits_in in violations:
            print("  ", splits_in, c["files"][:3], "...")
        raise SystemExit(1)
    else:
        print("[ct_resplit] VERIFIED: 0 clusters span multiple splits (checked all 111 clusters directly)")

    counts, corrupt_files = rebuild_data_dir(roster, assignment)
    if corrupt_files:
        print(f"WARNING: {len(corrupt_files)} corrupted image(s) skipped: {corrupt_files}")

    print()
    print("=== Final on-disk counts ===")
    header = f"{'split':<8}" + "".join(f"{c:<14}" for c in CLASSES) + "total"
    print(header)
    for split in SPLITS:
        row_total = sum(counts[split][c] for c in CLASSES)
        row = f"{split:<8}" + "".join(f"{counts[split][c]:<14}" for c in CLASSES) + str(row_total)
        print(row)

    return {
        "per_class": current,
        "targets": targets,
        "overall": overall,
        "overall_total": overall_total,
        "counts_on_disk": counts,
        "n_clusters": n_clusters,
        "n_singletons": n_singletons,
    }


if __name__ == "__main__":
    main()
