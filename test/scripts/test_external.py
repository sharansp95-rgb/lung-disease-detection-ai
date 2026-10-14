#!/usr/bin/env python3
"""
test_external.py — how does the pipeline handle images it has never seen?

Runs every image in an external probe set through the real API and records
which stage handled it: input validation, sanity gate, modality gate, OOD
detector, or the classifier. Then reports accuracy on the ones that made it
through.

The headline number is not accuracy. It is the split between REJECTED and
MISCLASSIFIED, because those mean opposite things:

  - Rejected at OOD  -> the system is working. It recognised the input as
                        unlike its training data and refused. What looks like
                        "wrong results" is a correct refusal.
  - Misclassified    -> the guards passed it and the classifier got it wrong.
                        That is the real generalization gap.

Run verify_provenance.py first. If the probe set is not genuinely external,
this script measures nothing.

Usage
-----
    # start the backend first
    python scripts/test_external.py --modality xray --images external_probe/xray
    python scripts/test_external.py --modality ct   --images external_probe/ct
"""

import argparse
import io
import json
import os
import sys
from collections import defaultdict

try:
    import requests
except ImportError:
    sys.exit("requests is required:  pip install requests")

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required:  pip install pillow")

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def classify_rejection(detail):
    d = (detail or "").lower()
    if "out of distribution" in d or "out-of-distribution" in d:
        return "OOD"
    if "looks like" in d and "not a" in d:
        return "MODALITY_GATE"
    if "does not look like" in d or "actual scan" in d or "not a scan" in d:
        return "SANITY_GATE"
    if "blank" in d or "colour" in d or "color" in d or "too large" in d:
        return "INPUT_VALIDATION"
    return "OTHER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=["xray", "ct"])
    ap.add_argument("--images", required=True,
                    help="directory laid out as <dir>/<classname>/*.png")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    try:
        requests.get(f"{args.api}/health", timeout=15).raise_for_status()
    except Exception as e:
        sys.exit(f"Backend not reachable at {args.api} ({e}).\n"
                 f"Start it with:  cd backend && uvicorn main:app --port 8000")

    samples = []
    for cls in sorted(os.listdir(args.images)):
        d = os.path.join(args.images, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                samples.append((os.path.join(d, f), cls))

    if not samples:
        sys.exit(f"No images found under {args.images}.\n"
                 f"Expected layout: {args.images}/<classname>/*.png\n"
                 f"See SOURCING.md for where to obtain external images.")

    print(f"Probing {len(samples)} images through {args.api}\n")

    rows, stages = [], defaultdict(int)
    correct = wrong = 0

    for path, label in samples:
        with Image.open(path) as im:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="PNG")
        buf.seek(0)
        try:
            r = requests.post(f"{args.api}/predict/{args.modality}",
                              files={"file": (os.path.basename(path), buf,
                                              "image/png")},
                              timeout=180)
        except requests.RequestException as e:
            rows.append({"file": path, "true": label, "stage": "ERROR",
                         "detail": str(e)})
            stages["ERROR"] += 1
            continue

        if r.status_code == 200:
            j = r.json()
            pred, conf = j.get("predicted_class"), j.get("confidence", 0.0)
            ok = pred.lower() == label.lower()
            correct += ok
            wrong += (not ok)
            stage = "CLASSIFIED_CORRECT" if ok else "CLASSIFIED_WRONG"
            stages[stage] += 1
            rows.append({"file": path, "true": label, "stage": stage,
                         "pred": pred, "conf": conf,
                         "probs": j.get("class_probabilities")})
            print(f"  {'ok ' if ok else 'NO '} {os.path.basename(path):<34s} "
                  f"true={label:<13s} pred={pred:<13s} conf={conf:.3f}")
        else:
            try:
                detail = r.json().get("detail", "")
            except Exception:
                detail = r.text[:200]
            gate = classify_rejection(detail)
            stages[f"REJECTED_{gate}"] += 1
            rows.append({"file": path, "true": label,
                         "stage": f"REJECTED_{gate}", "detail": detail})
            print(f"  --  {os.path.basename(path):<34s} "
                  f"true={label:<13s} REJECTED by {gate}")
            print(f"      \"{detail[:90]}\"")

    total = len(samples)
    rejected = sum(v for k, v in stages.items() if k.startswith("REJECTED"))
    reached = correct + wrong

    print("\n" + "=" * 74)
    print(f"EXTERNAL PROBE — {args.modality.upper()}  (n = {total})")
    print("=" * 74)
    for k in sorted(stages):
        print(f"  {k:<28s} {stages[k]:>4d}  ({100.0*stages[k]/total:5.1f}%)")

    print(f"\n  Reached the classifier: {reached}/{total}")
    if reached:
        print(f"  Accuracy once reached:  {100.0*correct/reached:.1f}% "
              f"({correct}/{reached})")

    print("\nINTERPRETATION")
    if total and rejected / total >= 0.5:
        print(f"  Most external images ({rejected}/{total}) were REJECTED, not")
        print("  misclassified. The guards are doing their job -- these inputs")
        print("  genuinely fall outside the training distribution and the")
        print("  system said so. What looks like 'wrong results' in the UI is")
        print("  largely correct refusal.")
        print("\n  Report this as a strength with an honest caveat: the system")
        print("  refuses out-of-source input rather than guessing, but that")
        print("  also means it cannot yet serve images from other institutions.")
        gate_counts = {k: v for k, v in stages.items() if k.startswith("REJECTED")}
        top = max(gate_counts, key=gate_counts.get)
        print(f"\n  Dominant rejecting stage: {top}")
        if top == "REJECTED_SANITY_GATE":
            print("  NOTE: the sanity gate is meant to catch photographs and")
            print("  screenshots, not real scans. If real external scans are")
            print("  being rejected here, the sanity gate is over-triggering")
            print("  and its negative pool likely lacks acquisition diversity.")
    elif reached and correct / reached < 0.6:
        print(f"  The guards passed {reached} images and the classifier got")
        print(f"  {wrong} of them wrong. This is a genuine generalization gap,")
        print("  not a guard problem. It is consistent with the source-database")
        print("  confound already documented in the paper: the model has partly")
        print("  learned acquisition characteristics that external images lack.")
        print("\n  Fix: retrain with border-occlusion, random-crop, gamma and")
        print("  contrast jitter, and JPEG noise. Run test_corruption.py first")
        print("  to confirm preprocessing is not also contributing.")
    else:
        print("  The pipeline handles external images reasonably. If the UI")
        print("  still looks wrong, capture the exact failing upload and run it")
        print("  through this script directly -- the failure may be specific to")
        print("  certain sources rather than external data in general.")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({"modality": args.modality, "n": total,
                       "stages": dict(stages), "rows": rows}, fh, indent=2)
        print(f"\nWritten: {args.json}")


if __name__ == "__main__":
    main()
