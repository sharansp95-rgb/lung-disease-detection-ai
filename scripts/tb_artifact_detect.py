"""
Scan every image in raw/xray_tb/{Normal,Tuberculosis}/ for two burned-in
artifact types spotted by visual inspection of the sample grids:

  1. color tint  -- image is not neutral grayscale (a genuine chest X-ray
     should be near-grayscale; a cyan/teal or other color cast is an
     artifact of how the source scan was digitized/stored, not pathology).
  2. white redaction/blank box -- a large, near-uniform, near-white
     rectangular region (patient-info redaction box, or a corrupted/mostly
     blank scan) that isn't real anatomy.

Reports per-class counts (this is the key evidence for whether the artifact
correlates with class label -- i.e. whether a model could learn the artifact
as a shortcut instead of the disease).
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
TB_SRC = ROOT / "raw" / "xray_tb" / "TB_Chest_Radiography_Database"
OUT_JSON = ROOT / "audit" / "tb_artifact_scan.json"

SAT_THRESHOLD = 12          # mean saturation (0-255) above this => tinted
MIN_BOX_AREA_FRAC = 0.005   # smallest box we bother flagging (0.5% of image)
BOX_FILL_MIN = 0.5          # rectangularity: box pixels / bbox area (0.85 was
                             # too strict -- missed a 6.9%-area box with fill 0.59
                             # because it wasn't a perfect solid rectangle)
MAX_ASPECT_RATIO = 7.5      # reject thin edge-hugging strips (bright diaphragm/
                             # sub-abdominal overexposure at the bottom border,
                             # seen equally in Normal images, aspect ratio 9-53)
                             # vs genuine redaction boxes (aspect ratio 1.3-6.1);
                             # 7.5 sits in the gap between the two populations
BOX_WHITE_VALUE = 248       # near-pure-white threshold


def load_bgr(path: Path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def detect_tint(bgr) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    mask = (v > 15) & (v < 240)  # ignore near-black/near-white, they're not informative for hue
    if mask.sum() == 0:
        return 0.0
    return float(hsv[:, :, 1][mask].mean())


def detect_white_boxes(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    total_area = h * w
    mask = (gray >= BOX_WHITE_VALUE).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    boxes = []
    for i in range(1, n_labels):  # skip background label 0
        x, y, bw, bh, area = stats[i]
        bbox_area = bw * bh
        if bbox_area == 0:
            continue
        area_frac = area / total_area
        fill_ratio = area / bbox_area
        aspect_ratio = max(bw, bh) / max(1, min(bw, bh))
        if (area_frac >= MIN_BOX_AREA_FRAC and fill_ratio >= BOX_FILL_MIN
                and aspect_ratio <= MAX_ASPECT_RATIO):
            boxes.append({
                "x": int(x), "y": int(y), "w": int(bw), "h": int(bh),
                "area_frac": round(float(area_frac), 4),
                "fill_ratio": round(float(fill_ratio), 3),
            })
    return boxes


def main():
    results = {}
    for cls in ["Normal", "Tuberculosis"]:
        src_dir = TB_SRC / cls
        files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        cls_results = []
        for i, p in enumerate(files):
            bgr = load_bgr(p)
            sat = detect_tint(bgr)
            boxes = detect_white_boxes(bgr)
            if sat > SAT_THRESHOLD or boxes:
                cls_results.append({
                    "file": p.name,
                    "saturation": round(sat, 2),
                    "tinted": sat > SAT_THRESHOLD,
                    "boxes": boxes,
                    "max_box_area_frac": max((b["area_frac"] for b in boxes), default=0.0),
                })
            if (i + 1) % 500 == 0:
                print(f"  {cls}: scanned {i+1}/{len(files)}")
        results[cls] = {"total": len(files), "flagged": cls_results}
        print(f"{cls}: {len(cls_results)}/{len(files)} flagged for tint and/or white box")

    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to {OUT_JSON}")

    # summary
    for cls in ["Normal", "Tuberculosis"]:
        flagged = results[cls]["flagged"]
        n_tinted = sum(1 for r in flagged if r["tinted"])
        n_boxed = sum(1 for r in flagged if r["boxes"])
        n_large_box = sum(1 for r in flagged if r["max_box_area_frac"] >= 0.05)
        print(f"{cls}: total={results[cls]['total']}  tinted={n_tinted}  "
              f"has_box={n_boxed}  large_box(>=5%)={n_large_box}")

    return results


if __name__ == "__main__":
    main()
