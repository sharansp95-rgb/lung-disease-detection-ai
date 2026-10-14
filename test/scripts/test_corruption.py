#!/usr/bin/env python3
"""
test_corruption.py — does the model break on its own images, lightly altered?

This is the test to run first, because it needs no external data at all.

Take images the classifier already gets RIGHT, apply one realistic transform at
a time (resize, JPEG, border, gamma, contrast, invert, equalize), and re-predict
through the real API. Any transform that flips a correct prediction is a
preprocessing sensitivity, not a failure to understand pathology.

That distinction decides the fix. If a training image breaks under a 10% black
border, the problem is the crop/preprocessing path and no retraining is needed.
If every transform is survived and only genuinely external images fail, the
problem is the source confound and retraining is required.

Usage
-----
    # start the backend first:  cd backend && uvicorn main:app --port 8000
    python scripts/test_corruption.py --modality xray --images ../data/xray/test
    python scripts/test_corruption.py --modality ct   --images ../data/ct_scan/test

Expects <images>/<ClassName>/*.png layout. Class folder names are the labels.
"""

import argparse
import io
import json
import os
import random
import sys
from collections import defaultdict

try:
    import requests
except ImportError:
    sys.exit("requests is required:  pip install requests")

try:
    import numpy as np
    from PIL import Image, ImageOps
except ImportError:
    sys.exit("Pillow and numpy are required:  pip install pillow numpy")

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------
# Transforms. Each takes a PIL image and returns a PIL image.
# --------------------------------------------------------------------------

def t_identity(im):
    return im


def t_resize_roundtrip(im):
    w, h = im.size
    return im.resize((512, 512), Image.BICUBIC).resize((w, h), Image.BICUBIC)


def t_jpeg60(im):
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=60)
    buf.seek(0)
    return Image.open(buf).copy()


def t_border10(im):
    w, h = im.size
    bw, bh = int(w * 0.10), int(h * 0.10)
    return ImageOps.expand(im, border=(bw, bh, bw, bh), fill=0)


def _gamma(im, g):
    a = np.asarray(im.convert("L")).astype(np.float32) / 255.0
    a = np.power(np.clip(a, 0, 1), g)
    return Image.fromarray((a * 255).astype(np.uint8)).convert(im.mode)


def t_gamma07(im):
    return _gamma(im, 0.7)


def t_gamma14(im):
    return _gamma(im, 1.4)


def _contrast(im, f):
    a = np.asarray(im.convert("L")).astype(np.float32)
    a = (a - a.mean()) * f + a.mean()
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert(im.mode)


def t_contrast07(im):
    return _contrast(im, 0.7)


def t_contrast14(im):
    return _contrast(im, 1.4)


def t_invert(im):
    return ImageOps.invert(im.convert("L")).convert(im.mode)


def t_equalize(im):
    return ImageOps.equalize(im.convert("L")).convert(im.mode)


TRANSFORMS = [
    ("baseline (unaltered)", t_identity),
    ("resize 512 roundtrip", t_resize_roundtrip),
    ("JPEG quality 60", t_jpeg60),
    ("black border 10%", t_border10),
    ("gamma 0.7", t_gamma07),
    ("gamma 1.4", t_gamma14),
    ("contrast x0.7", t_contrast07),
    ("contrast x1.4", t_contrast14),
    ("intensity inverted", t_invert),
    ("histogram equalized", t_equalize),
]


# --------------------------------------------------------------------------

def classify_rejection(detail):
    """Map a 422 detail string onto the gate that most likely produced it."""
    d = (detail or "").lower()
    if "out of distribution" in d or "out-of-distribution" in d:
        return "OOD"
    if "looks like" in d and "not a" in d:
        return "MODALITY_GATE"
    if "does not look like" in d or "actual scan" in d or "not a scan" in d:
        return "SANITY_GATE"
    if "blank" in d or "colour" in d or "color" in d or "image" in d:
        return "INPUT_VALIDATION"
    return "OTHER"


def predict(api, modality, im):
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    try:
        r = requests.post(f"{api}/predict/{modality}",
                          files={"file": ("probe.png", buf, "image/png")},
                          timeout=180)
    except requests.RequestException as e:
        return {"status": "ERROR", "detail": str(e)}
    if r.status_code == 200:
        j = r.json()
        return {"status": "OK",
                "pred": j.get("predicted_class"),
                "conf": j.get("confidence")}
    try:
        detail = r.json().get("detail", "")
    except Exception:
        detail = r.text[:200]
    return {"status": "REJECTED", "code": r.status_code,
            "detail": detail, "gate": classify_rejection(detail)}


def load_labelled(root, per_class):
    out = []
    if not os.path.isdir(root):
        sys.exit(f"Image directory not found: {root}")
    for cls in sorted(os.listdir(root)):
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            continue
        files = [os.path.join(d, f) for f in sorted(os.listdir(d))
                 if os.path.splitext(f)[1].lower() in IMG_EXT]
        random.shuffle(files)
        for f in files[:per_class]:
            out.append((f, cls))
    if not out:
        sys.exit(f"No class subfolders with images under {root}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=["xray", "ct"])
    ap.add_argument("--images", required=True,
                    help="directory laid out as <dir>/<ClassName>/*.png")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--per-class", type=int, default=5,
                    help="images sampled per class (default 5)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    random.seed(args.seed)

    try:
        h = requests.get(f"{args.api}/health", timeout=15)
        h.raise_for_status()
    except Exception as e:
        sys.exit(f"Backend not reachable at {args.api} ({e}).\n"
                 f"Start it with:  cd backend && uvicorn main:app --port 8000")

    samples = load_labelled(args.images, args.per_class)
    print(f"Loaded {len(samples)} images from {args.images}")

    # Keep only the ones the model already gets right unaltered — the whole
    # point is to see what breaks something that currently works.
    print("\nEstablishing baseline (keeping only correct predictions)...")
    correct = []
    for path, label in samples:
        with Image.open(path) as im:
            r = predict(args.api, args.modality, im.copy())
        ok = r["status"] == "OK" and r["pred"].lower() == label.lower()
        mark = "ok " if ok else "no "
        got = r.get("pred") or f"{r['status']}:{r.get('gate', '')}"
        print(f"  {mark} {os.path.basename(path):<38s} true={label:<14s} got={got}")
        if ok:
            correct.append((path, label))

    if not correct:
        sys.exit("\nNo image was classified correctly unaltered. Either the "
                 "labels do not match the class folder names, or --images "
                 "points at the wrong split. Fix that before continuing.")

    print(f"\n{len(correct)} images form the baseline set.\n")

    results = defaultdict(lambda: {"kept": 0, "flipped": 0, "rejected": 0,
                                   "gates": defaultdict(int), "n": 0})

    for tname, fn in TRANSFORMS:
        for path, label in correct:
            with Image.open(path) as im:
                try:
                    altered = fn(im.copy())
                except Exception:
                    continue
            r = predict(args.api, args.modality, altered)
            s = results[tname]
            s["n"] += 1
            if r["status"] == "OK":
                if r["pred"].lower() == label.lower():
                    s["kept"] += 1
                else:
                    s["flipped"] += 1
            elif r["status"] == "REJECTED":
                s["rejected"] += 1
                s["gates"][r["gate"]] += 1
        s = results[tname]
        acc = 100.0 * s["kept"] / max(s["n"], 1)
        print(f"  {tname:<24s} kept {s['kept']:>3d}/{s['n']:<3d} "
              f"({acc:5.1f}%)  flipped {s['flipped']:>3d}  "
              f"rejected {s['rejected']:>3d}")

    print("\n" + "=" * 74)
    print(f"CORRUPTION ROBUSTNESS — {args.modality.upper()}")
    print("=" * 74)
    print(f"{'transform':<24s} {'accuracy':>9s} {'flipped':>8s} {'rejected':>9s}")
    print("-" * 74)
    base = results["baseline (unaltered)"]
    base_acc = 100.0 * base["kept"] / max(base["n"], 1)
    for tname, _ in TRANSFORMS:
        s = results[tname]
        acc = 100.0 * s["kept"] / max(s["n"], 1)
        delta = "" if tname.startswith("baseline") else f"  ({acc - base_acc:+.1f})"
        print(f"{tname:<24s} {acc:8.1f}% {s['flipped']:8d} "
              f"{s['rejected']:9d}{delta}")

    fragile = [(t, 100.0 * results[t]["kept"] / max(results[t]["n"], 1))
               for t, _ in TRANSFORMS if not t.startswith("baseline")]
    fragile = [(t, a) for t, a in fragile if base_acc - a >= 20.0]

    print("\nINTERPRETATION")
    if fragile:
        print("  PREPROCESSING SENSITIVITY FOUND. These transforms cost 20+")
        print("  points on images the model otherwise gets right:")
        for t, a in sorted(fragile, key=lambda x: x[1]):
            print(f"    - {t}: {a:.1f}% (from {base_acc:.1f}%)")
        print("\n  This is a preprocessing problem, not a pathology-understanding")
        print("  problem. Internet images differ from training images in exactly")
        print("  these ways. Fix the preprocessing path before retraining --")
        print("  retraining will not help if the input never reaches the model")
        print("  in the form it expects.")
        if any("border" in t for t, _ in fragile):
            print("\n  The black-border result specifically implicates the lung-crop")
            print("  step. Check whether compute_lung_crop() is falling back to")
            print("  border_crop() on these inputs.")
    else:
        print("  ROBUST to all transforms tested. The model survives resizing,")
        print("  compression, borders, and photometric shifts on its own images.")
        print("  That points away from preprocessing and toward the")
        print("  source-database confound as the cause of external failures.")
        print("  Confirm with a genuinely external probe set, then plan a")
        print("  retrain with border-occlusion and photometric augmentation.")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({t: {k: (dict(v) if isinstance(v, defaultdict) else v)
                           for k, v in results[t].items()}
                       for t, _ in TRANSFORMS}, fh, indent=2)
        print(f"\nWritten: {args.json}")


if __name__ == "__main__":
    main()
