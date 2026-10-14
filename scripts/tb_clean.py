"""
Apply fixes for the burned-in artifacts found by tb_artifact_detect.py and
write a cleaned mirror of raw/xray_tb/ to raw/xray_tb_cleaned/.

Policy (per image, based on audit/tb_artifact_scan.json):
  - not flagged at all                       -> copy unchanged
  - tinted (color cast)                      -> desaturate to true grayscale
  - has a white box with area_frac < 5%      -> inpaint that box region
  - has any white box with area_frac >= 5%   -> EXCLUDE (not copied at all;
                                                 too much of the image is
                                                 destroyed to trust the label)

Both fixes can apply to the same image (desaturate, then inpaint). This is
a read-only-on-source operation: raw/xray_tb/ is never modified, only read.
raw/xray_tb_cleaned/ is a brand-new, complete mirror (unflagged + fixed
images), ready to be pointed at by prepare_xray.py.
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
TB_SRC = ROOT / "raw" / "xray_tb" / "TB_Chest_Radiography_Database"
TB_CLEANED = ROOT / "raw" / "xray_tb_cleaned"
SCAN_JSON = ROOT / "audit" / "tb_artifact_scan.json"
LARGE_BOX_THRESHOLD = 0.05

BOX_WHITE_VALUE = 248


def load_bgr(path: Path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def desaturate(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def inpaint_boxes(bgr, boxes, pad=3):
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in boxes:
        x0 = max(0, b["x"] - pad)
        y0 = max(0, b["y"] - pad)
        x1 = min(w, b["x"] + b["w"] + pad)
        y1 = min(h, b["y"] + b["h"] + pad)
        mask[y0:y1, x0:x1] = 255
    return cv2.inpaint(bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


def main():
    with open(SCAN_JSON) as f:
        scan = json.load(f)

    if TB_CLEANED.exists():
        shutil.rmtree(TB_CLEANED)

    summary = {}

    for cls in ["Normal", "Tuberculosis"]:
        src_dir = TB_SRC / cls
        out_dir = TB_CLEANED / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        flagged_by_name = {r["file"]: r for r in scan[cls]["flagged"]}
        all_files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})

        n_copied_unchanged = 0
        n_desaturated = 0
        n_inpainted = 0
        n_excluded = 0
        excluded_names = []

        for p in all_files:
            rec = flagged_by_name.get(p.name)
            if rec is None:
                shutil.copy2(p, out_dir / p.name)
                n_copied_unchanged += 1
                continue

            if any(b["area_frac"] >= LARGE_BOX_THRESHOLD for b in rec["boxes"]):
                n_excluded += 1
                excluded_names.append(p.name)
                continue

            bgr = load_bgr(p)
            if rec["tinted"]:
                bgr = desaturate(bgr)
                n_desaturated += 1
            if rec["boxes"]:
                bgr = inpaint_boxes(bgr, rec["boxes"])
                n_inpainted += 1

            out_path = out_dir / p.name
            cv2.imwrite(str(out_path), bgr)

        summary[cls] = {
            "source_total": len(all_files),
            "copied_unchanged": n_copied_unchanged,
            "desaturated": n_desaturated,
            "inpainted": n_inpainted,
            "excluded": n_excluded,
            "cleaned_total": n_copied_unchanged + n_desaturated + n_inpainted - len(
                set()  # desaturated/inpainted counts overlap (same image can be both); compute properly below
            ),
            "excluded_files": excluded_names,
        }

    # cleaned_total should just be: files actually written to out_dir
    for cls in ["Normal", "Tuberculosis"]:
        out_dir = TB_CLEANED / cls
        actual = len(list(out_dir.iterdir()))
        summary[cls]["cleaned_total"] = actual
        print(f"{cls}: source={summary[cls]['source_total']}  "
              f"unchanged={summary[cls]['copied_unchanged']}  "
              f"desaturated={summary[cls]['desaturated']}  "
              f"inpainted={summary[cls]['inpainted']}  "
              f"excluded={summary[cls]['excluded']}  "
              f"-> cleaned_total={actual}")

    with open(ROOT / "audit" / "tb_clean_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    main()
