"""Step 3 of the CT-Normal-augmentation experiment: build
data/ct_scan_normal_augmented/ = the existing data/ct_scan/ with the Normal class
augmented by the Group A external Normal pool built by ct_normal_pool_split.py.

Adapted from scripts/prepare_ct_scan.py + scripts/ct_resplit_grouped.py (neither
of which is modified). Two deliberate differences from ct_resplit_grouped.py:

 1. **Existing images keep their existing split assignment, verbatim.**
    data/ct_scan/ is already the product of the patient-leakage-aware resplit
    (PROJECT_STATUS.md §6: 111 near-duplicate clusters, zero left spanning
    splits) *and* the protocol-mismatch exclusion of 61 Malignant images. Both
    would be lost by rebuilding from raw/. Preserving the assignment also keeps
    the new test split byte-identical to the original one, which is what makes
    the "no regression on data/ct_scan/test" check in step 5 apples-to-apples
    rather than a comparison against a reshuffled test set.

 2. **Group A images go to train and val only, never test.** The experiment's
    honest external test set is Group B (held out entirely,
    eval_run/ct_normal_heldout/); putting Group A images in the new test split
    would inflate test accuracy with the very distribution we just trained on
    and would break difference (1).

Group A is still assigned *group-aware*: the patient-proxy groups from
ct_normal_pool_split.py (phash+SSIM clusters for mohamedhanyyy, study ID for
maftouni) are indivisible -- every image of a group lands in one split. This is
the same constraint ct_resplit_grouped.py enforces, and the same bug it exists to
prevent (35/109 test images once had a near-duplicate sibling in train).

Benign and Malignant are copied across unchanged. data/ct_scan/ is never written.
"""
import argparse, json, random, re, shutil
from pathlib import Path

from PIL import Image

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
SRC_DATA = ROOT / "data" / "ct_scan"
GROUP_A = ROOT / "datasets" / "ct_normal_extra" / "groupA"
MANIFEST = ROOT / "eval_run" / "ct_normal_pool_manifest.json"
OUT = ROOT / "data" / "ct_scan_normal_augmented"
REPORT = ROOT / "eval_run" / "ct_normal_augmented_build.json"

SPLITS = ["train", "val", "test"]
CLASSES = ["Benign", "Malignant", "Normal"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
# Group A goes to train/val only (see module docstring)
A_TARGETS = {"train": 0.85, "val": 0.15}


def group_key(path: Path) -> str:
    """Patient-proxy group for a Group A file, from the naming laid down by
    ct_normal_pool_split.py. Must stay in step with that script's
    mf_study_id() -- the three maftouni sub-sources name slices differently:
        mh_g07_5.png                        -> 'mh_g07'  (phash+SSIM cluster)
        mf_16_Morozov_study_0001_12.png     -> 'mf_16_Morozov_study_0001'
        mf_N001_121.png                     -> 'mf_N001'
        mf_6_Rahimzadeh_normal10_patient151_SR_4_IM00012.png
                                            -> '6_Rahimzadeh_patient151'
    ('normalN' there is a batch number, not part of the patient identity -- see
    ct_normal_pool_split.py::mf_study_id.)
    """
    stem = path.stem
    if stem.startswith("mh_"):
        return "_".join(stem.split("_")[:2])
    m = re.search(r"patient\d+", stem)
    if m:
        return f"6_Rahimzadeh_{m.group(0)}"
    return re.sub(r"_\d+$", "", stem)


def copy_existing():
    counts = {s: {c: 0 for c in CLASSES} for s in SPLITS}
    for s in SPLITS:
        for c in CLASSES:
            src = SRC_DATA / s / c
            dst = OUT / s / c
            dst.mkdir(parents=True, exist_ok=True)
            for p in sorted(src.iterdir()):
                if p.suffix.lower() in IMAGE_EXTS:
                    shutil.copy2(p, dst / p.name)
                    counts[s][c] += 1
    return counts


def assign_group_a(sources):
    files = [p for p in sorted(GROUP_A.rglob("*"))
             if p.suffix.lower() in IMAGE_EXTS and p.parent.name in sources]
    groups = {}
    for p in files:
        groups.setdefault(group_key(p), []).append(p)

    total = len(files)
    targets = {s: total * f for s, f in A_TARGETS.items()}
    current = {s: 0 for s in A_TARGETS}

    rng = random.Random(SEED)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: -len(kv[1]))  # largest group first

    assignment = {}
    for gkey, paths in items:
        split = max(A_TARGETS, key=lambda s: targets[s] - current[s])
        assignment[gkey] = split
        current[split] += len(paths)
    return groups, assignment, current, total


def main():
    global OUT, REPORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(OUT))
    ap.add_argument("--report", default=str(REPORT))
    ap.add_argument("--sources", default="mohamedhanyyy,maftouni",
                    help="which Group A sub-source folders to include")
    args = ap.parse_args()

    OUT = Path(args.out_dir)
    REPORT = Path(args.report)
    sources = set(args.sources.split(","))
    print(f"[prepare] out={OUT}  Group A sources={sorted(sources)}")

    if OUT.exists():
        shutil.rmtree(OUT)

    counts = copy_existing()
    print("[prepare] copied existing data/ct_scan/ (split assignments preserved):")
    for s in SPLITS:
        print(f"   {s:<6} " + " ".join(f"{c}={counts[s][c]}" for c in CLASSES))

    groups, assignment, current, total = assign_group_a(sources)
    print(f"[prepare] Group A: {total} images in {len(groups)} patient-proxy groups")

    added = {s: 0 for s in A_TARGETS}
    per_group_split = {}
    for gkey, paths in groups.items():
        split = assignment[gkey]
        per_group_split[gkey] = split
        for p in paths:
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                print(f"  skipping unreadable {p}")
                continue
            shutil.copy2(p, OUT / split / "Normal" / p.name)
            added[split] += 1
            counts[split]["Normal"] += 1

    # a group must never straddle splits
    straddle = [g for g, s in per_group_split.items() if not isinstance(s, str)]
    assert not straddle

    print(f"[prepare] Group A added: train={added['train']} val={added['val']} test=0")
    print()
    print("=== data/ct_scan_normal_augmented final counts ===")
    hdr = f"{'split':<8}" + "".join(f"{c:<12}" for c in CLASSES) + "total"
    print(hdr)
    for s in SPLITS:
        t = sum(counts[s][c] for c in CLASSES)
        print(f"{s:<8}" + "".join(f"{counts[s][c]:<12}" for c in CLASSES) + str(t))

    on_disk = {s: {c: len(list((OUT / s / c).glob('*'))) for c in CLASSES} for s in SPLITS}
    REPORT.write_text(json.dumps({
        "out_dir": str(OUT),
        "counts": counts,
        "counts_on_disk": on_disk,
        "group_a_total": total,
        "group_a_n_groups": len(groups),
        "group_a_split_assignment": per_group_split,
        "group_a_added_per_split": added,
        "note": "existing data/ct_scan split assignments preserved verbatim; "
                "Group A assigned group-aware to train/val only",
    }, indent=2), encoding="utf-8")
    print(f"[prepare] wrote {REPORT}")


if __name__ == "__main__":
    main()
