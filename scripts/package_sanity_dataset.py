"""
Repacks data/xray_sanity and data/ct_sanity (currently hardlinked to
full-resolution source images -- 1.5GB+147MB combined) into a much smaller
export for upload to a remote training environment: every image resized to
the model's actual input size (224x224, matching ml/dataset.py's
build_transforms Resize) and saved as JPEG. Training only ever sees images
at that resolution, so nothing is lost for this purpose -- this purely
avoids uploading resolution the model never uses.
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC_DIRS = [ROOT / "data" / "xray_sanity", ROOT / "data" / "ct_sanity"]
OUT_ROOT = ROOT / "data" / "sanity_export"
SIZE = (224, 224)
JPEG_QUALITY = 85


def main():
    total = 0
    for src_root in SRC_DIRS:
        for f in src_root.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(ROOT / "data")
            out_path = (OUT_ROOT / rel).with_suffix(".jpg")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                img = Image.open(f).convert("RGB").resize(SIZE, Image.BILINEAR)
                img.save(out_path, "JPEG", quality=JPEG_QUALITY)
            except Exception as e:
                print(f"skip {f}: {e}")
                continue
            total += 1
            if total % 5000 == 0:
                print(f"...{total} images processed")

    print(f"done: {total} images written to {OUT_ROOT}")


if __name__ == "__main__":
    main()
