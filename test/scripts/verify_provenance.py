#!/usr/bin/env python3
"""
verify_provenance.py — is this probe set actually external?

Before any generalization claim can be made, we have to know that the probe
images are not the training images wearing a different filename. This script
hashes every probe image and every image under the project's data directories,
then reports exact and near-duplicate matches with the split each twin lives in.

An image that is a near-duplicate of a training image cannot test
generalization. If most of the probe set matches, the diagnostic that used it
is invalid and must be re-run on a real external set.

Usage
-----
    python scripts/verify_provenance.py --probe "../sample images"
    python scripts/verify_provenance.py --probe external_probe --json out.json

Run from the `test/` directory placed at the project root.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required:  pip install pillow")

try:
    import imagehash
    HAVE_IMAGEHASH = True
except ImportError:
    HAVE_IMAGEHASH = False

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Directories searched for the "known" corpus, relative to the project root.
CORPUS_DIRS = ["data", "raw", "datasets", "seg_data"]


def iter_images(root):
    for dirpath, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                yield os.path.join(dirpath, f)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def phash(path):
    try:
        with Image.open(path) as im:
            return imagehash.phash(im.convert("L"), hash_size=8)
    except Exception:
        return None


def split_of(path):
    """Infer train/val/test from the path, if the layout encodes it."""
    p = path.replace("\\", "/").lower()
    for name in ("train", "val", "valid", "validation", "test"):
        if f"/{name}/" in p:
            return {"valid": "val", "validation": "val"}.get(name, name)
    return "unknown"


def ssim_ok(a_path, b_path, threshold=0.90):
    """Confirm a perceptual-hash candidate with SSIM. Returns (passed, score)."""
    try:
        import numpy as np
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return None, None
    try:
        with Image.open(a_path) as ia, Image.open(b_path) as ib:
            a = np.array(ia.convert("L").resize((256, 256)))
            b = np.array(ib.convert("L").resize((256, 256)))
        score = float(ssim(a, b))
        return score >= threshold, score
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True,
                    help="directory of probe images to check")
    ap.add_argument("--root", default="..",
                    help="project root containing data/, raw/, datasets/")
    ap.add_argument("--hamming", type=int, default=8,
                    help="max perceptual-hash distance for a near-duplicate")
    ap.add_argument("--json", default="results/provenance.json")
    args = ap.parse_args()

    probe_dir = os.path.abspath(args.probe)
    root = os.path.abspath(args.root)

    if not os.path.isdir(probe_dir):
        sys.exit(f"Probe directory not found: {probe_dir}")

    probe = sorted(iter_images(probe_dir))
    if not probe:
        sys.exit(f"No images found under {probe_dir}")

    corpus = []
    for d in CORPUS_DIRS:
        full = os.path.join(root, d)
        if os.path.isdir(full):
            found = list(iter_images(full))
            corpus.extend(found)
            print(f"  corpus: {d}/  ({len(found)} images)")
        else:
            print(f"  corpus: {d}/  -- not found, skipping")

    print(f"\nProbe images: {len(probe)}")
    print(f"Corpus images: {len(corpus)}")
    if not corpus:
        sys.exit("\nNo corpus images found. Check --root points at the project "
                 "root (the directory containing data/, raw/, datasets/).")

    # ---- exact matches -------------------------------------------------
    print("\nHashing corpus (SHA-256)... this can take a few minutes.")
    corpus_sha = defaultdict(list)
    for i, p in enumerate(corpus, 1):
        if i % 2000 == 0:
            print(f"  {i}/{len(corpus)}")
        try:
            corpus_sha[sha256(p)].append(p)
        except Exception:
            pass

    verdicts = {}
    unmatched = []
    for p in probe:
        try:
            h = sha256(p)
        except Exception:
            verdicts[p] = {"verdict": "UNREADABLE"}
            continue
        if h in corpus_sha:
            twin = corpus_sha[h][0]
            verdicts[p] = {
                "verdict": f"EXACT_MATCH_IN_{split_of(twin).upper()}",
                "twin": os.path.relpath(twin, root),
                "split": split_of(twin),
            }
        else:
            unmatched.append(p)

    # ---- near-duplicates -----------------------------------------------
    if unmatched and HAVE_IMAGEHASH:
        print(f"\n{len(unmatched)} probe images had no exact match. "
              f"Checking perceptual hashes...")
        corpus_ph = []
        for i, p in enumerate(corpus, 1):
            if i % 2000 == 0:
                print(f"  {i}/{len(corpus)}")
            ph = phash(p)
            if ph is not None:
                corpus_ph.append((ph, p))

        for p in unmatched:
            ph = phash(p)
            if ph is None:
                verdicts[p] = {"verdict": "UNREADABLE"}
                continue
            best = None
            for cph, cp in corpus_ph:
                d = ph - cph
                if d <= args.hamming and (best is None or d < best[0]):
                    best = (d, cp)
            if best is None:
                verdicts[p] = {"verdict": "NO_MATCH"}
            else:
                d, cp = best
                passed, score = ssim_ok(p, cp)
                if passed is False:
                    verdicts[p] = {"verdict": "NO_MATCH",
                                   "note": f"phash d={d} but SSIM={score:.3f} "
                                           f"below 0.90"}
                else:
                    verdicts[p] = {
                        "verdict": f"NEAR_DUPLICATE_OF_{split_of(cp).upper()}",
                        "twin": os.path.relpath(cp, root),
                        "split": split_of(cp),
                        "hamming": d,
                        "ssim": None if score is None else round(score, 4),
                    }
    else:
        if unmatched and not HAVE_IMAGEHASH:
            print("\nImageHash not installed; skipping near-duplicate check.")
            print("  pip install imagehash scikit-image")
        for p in unmatched:
            verdicts.setdefault(p, {"verdict": "NO_MATCH_EXACT_ONLY"})

    # ---- report --------------------------------------------------------
    print("\n" + "=" * 74)
    print("PER-IMAGE VERDICT")
    print("=" * 74)
    for p in probe:
        v = verdicts.get(p, {"verdict": "?"})
        rel = os.path.relpath(p, probe_dir)
        line = f"  {rel:<44s} {v['verdict']}"
        if "twin" in v:
            line += f"\n      twin: {v['twin']}"
        print(line)

    counts = defaultdict(int)
    for v in verdicts.values():
        counts[v["verdict"]] += 1

    external = sum(n for k, n in counts.items() if k.startswith("NO_MATCH"))
    total = len(probe)

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for k in sorted(counts):
        print(f"  {k:<36s} {counts[k]}")
    print(f"\n  Genuinely external (NO_MATCH): {external}/{total} "
          f"({100.0*external/total:.0f}%)")

    print("\nVERDICT")
    if external == 0:
        print("  INVALID PROBE SET. Every probe image traces back to the")
        print("  training corpus. Any generalization result computed on this")
        print("  set is meaningless. Build a real external set before")
        print("  re-running the diagnostic.")
    elif external < total * 0.5:
        print(f"  MOSTLY INVALID. Only {external} of {total} images are")
        print("  external. Too few to support a generalization claim, and the")
        print("  matched images will inflate accuracy. Add more external data.")
    elif external < 20:
        print(f"  VALID BUT SMALL. {external} external images is enough to")
        print("  spot a large effect, not to quantify one. Treat any number")
        print("  from this set as indicative, not as a reported result.")
    else:
        print(f"  VALID. {external} genuinely external images. Results from")
        print("  this set can be reported, with the sample size stated.")

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump({"probe_dir": probe_dir, "counts": dict(counts),
                   "external": external, "total": total,
                   "verdicts": {os.path.relpath(k, probe_dir): v
                                for k, v in verdicts.items()}}, fh, indent=2)
    print(f"\nWritten: {args.json}")


if __name__ == "__main__":
    main()
