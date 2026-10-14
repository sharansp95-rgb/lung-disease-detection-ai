"""
One continuous run: rebuild data/xray/, verify it, zip it, verify the zip.
No gaps between steps -- run as a single process so nothing external can
interleave with this sequence.
"""

import random
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "xray"
ZIP_PATH = ROOT / "data" / "xray.zip"
EXPECTED_TOTAL = 19217
CLASSES = ["COVID-19", "Normal", "Pneumonia", "Tuberculosis"]
SPLITS = ["train", "val", "test"]

t_start = time.time()

# ── 1. Rebuild ──────────────────────────────────────────────────────────
print("=== STEP 1: rebuild data/xray/ via prepare_xray.py ===", flush=True)
result = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "prepare_xray.py")],
    cwd=str(ROOT), capture_output=True, text=True,
)
print(result.stdout)
if result.returncode != 0:
    print("PREPARE_XRAY.PY FAILED", file=sys.stderr)
    print(result.stderr, file=sys.stderr)
    sys.exit(1)
print(f"[{time.time()-t_start:.1f}s elapsed] rebuild complete", flush=True)

# ── 2. Verify rebuild ───────────────────────────────────────────────────
print("\n=== STEP 2: verify file count + spot-check 5 random images ===", flush=True)
all_files = []
for split in SPLITS:
    for cls in CLASSES:
        d = DATA_DIR / split / cls
        files = list(d.glob("*.png"))
        all_files.extend(files)

total_count = len(all_files)
print(f"total file count: {total_count} (expected {EXPECTED_TOTAL})")
if total_count != EXPECTED_TOTAL:
    print(f"COUNT MISMATCH -- STOPPING", file=sys.stderr)
    sys.exit(1)

rng = random.Random()  # unseeded -- genuine random spot-check, not reproducible-by-design
sample = rng.sample(all_files, 5)
all_ok = True
for p in sample:
    im = Image.open(p)
    im.load()
    size = im.size
    extrema = im.convert("L").getextrema()
    is_blank = extrema[0] == extrema[1]
    ok = (size == (224, 224)) and not is_blank
    all_ok = all_ok and ok
    print(f"  {p.relative_to(ROOT)} | size={size} | pixel_range={extrema} | "
          f"{'OK' if ok else 'FAIL'}")

if not all_ok:
    print("SPOT-CHECK FAILED -- STOPPING", file=sys.stderr)
    sys.exit(1)
print(f"[{time.time()-t_start:.1f}s elapsed] verification passed", flush=True)

# ── 3. Zip ──────────────────────────────────────────────────────────────
print("\n=== STEP 3: zip data/xray/ -> data/xray.zip ===", flush=True)
if ZIP_PATH.exists():
    ZIP_PATH.unlink()

zip_count = 0
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_STORED) as zf:
    for p in sorted(DATA_DIR.rglob("*")):
        if p.is_file():
            zf.write(p, p.relative_to(DATA_DIR.parent))
            zip_count += 1

print(f"zipped {zip_count} files")
print(f"[{time.time()-t_start:.1f}s elapsed] zip complete", flush=True)

# ── 4. Verify zip ───────────────────────────────────────────────────────
print("\n=== STEP 4: verify the zip ===", flush=True)
zf = zipfile.ZipFile(ZIP_PATH)
entries = zf.namelist()
bad = zf.testzip()
size_bytes = ZIP_PATH.stat().st_size

print(f"entry count: {len(entries)}")
print(f"testzip() result (None=clean): {bad}")
print(f"file size on disk: {size_bytes} bytes ({size_bytes/1024/1024:.1f} MB)")

if bad is not None:
    print("ZIP CORRUPTION DETECTED -- STOPPING", file=sys.stderr)
    sys.exit(1)
if len(entries) != EXPECTED_TOTAL:
    print(f"ZIP ENTRY COUNT MISMATCH ({len(entries)} != {EXPECTED_TOTAL}) -- STOPPING", file=sys.stderr)
    sys.exit(1)

print(f"\n[{time.time()-t_start:.1f}s elapsed] ALL STEPS PASSED, NO GAPS")
print(f"FINAL RESULT: {ZIP_PATH} | {size_bytes} bytes | {len(entries)} entries")
