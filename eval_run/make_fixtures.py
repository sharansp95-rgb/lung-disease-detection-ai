"""Builds non-scan / edge-case upload fixtures under eval_run/fixtures/."""
import pathlib, shutil, random
import numpy as np
from PIL import Image, ImageDraw

ROOT = pathlib.Path(r"d:/7th sem/sdp")
FIX = ROOT / "eval_run" / "fixtures"
FIX.mkdir(parents=True, exist_ok=True)
random.seed(42); np.random.seed(42)

# 1. real photo (Windows wallpaper -- genuinely outside any training pool here)
photo_src = pathlib.Path("C:/Windows/Web/Wallpaper/Spotlight/img14.jpg")
if photo_src.exists():
    Image.open(photo_src).convert("RGB").resize((800, 600)).save(FIX / "photo_landscape.png")
photo2 = pathlib.Path("C:/Windows/Web/Screen/img100.jpg")
if photo2.exists():
    Image.open(photo2).convert("RGB").resize((800, 600)).save(FIX / "photo_object.png")

# 1b. grayscale version of a real photo -- defeats the colour heuristic
if (FIX / "photo_landscape.png").exists():
    Image.open(FIX / "photo_landscape.png").convert("L").save(FIX / "photo_grayscale.png")

# 2. screenshot (a real app screenshot from the repo's frontend/ folder)
shot = ROOT / "frontend" / "screenshot_xray.png"
if shot.exists():
    shutil.copy(shot, FIX / "screenshot_app.png")

# 3. scanned text document
img = Image.new("RGB", (850, 1100), (250, 249, 244))
d = ImageDraw.Draw(img)
y = 60
for i in range(45):
    w = random.randint(300, 720)
    d.rectangle([70, y, 70 + w, y + 9], fill=(35, 35, 40))
    y += 22
img.convert("L").save(FIX / "scanned_document.png")

# 4. solid colour
Image.new("RGB", (512, 512), (128, 128, 128)).save(FIX / "solid_gray.png")
Image.new("RGB", (512, 512), (10, 90, 200)).save(FIX / "solid_blue.png")

# 5. corrupted / truncated real x-ray
src = next((ROOT / "data" / "xray" / "test" / "Normal").glob("*.png"))
raw = src.read_bytes()
(FIX / "truncated_xray.png").write_bytes(raw[: len(raw) // 3])

# 6. non-image renamed with an image extension
(FIX / "not_an_image.png").write_bytes((ROOT / "requirements.txt").read_bytes())

# 7. zero-byte file
(FIX / "empty.png").write_bytes(b"")

# 8. random noise (high std, grayscale -- passes the blank/colour heuristics)
Image.fromarray(np.random.randint(0, 256, (512, 512), dtype=np.uint8)).save(FIX / "random_noise.png")

# 9. near-blank with faint noise (blank guard test)
Image.fromarray((np.random.randn(512, 512) * 2 + 128).clip(0, 255).astype(np.uint8)).save(FIX / "near_blank.png")

# --- Section E: segmentation-crop stressors, derived from one real X-ray ---
xr = Image.open(src).convert("L")
w, h = xr.size
xr.crop((int(w * .35), int(h * .35), int(w * .65), int(h * .65))).resize((512, 512)).save(FIX / "seg_zoomed_center.png")
xr.resize((1024, 200)).save(FIX / "seg_wide_aspect.png")
xr.resize((200, 1024)).save(FIX / "seg_tall_aspect.png")
canvas = Image.new("L", (1024, 1024), 8)
canvas.paste(xr.resize((160, 160)), (430, 430))
canvas.save(FIX / "seg_mostly_blank.png")
print("fixtures:", sorted(p.name for p in FIX.iterdir()))
