"""
Builds a "not a medical scan" negative image pool for training the sanity
gate (sanity_utils.py / train_sanity_gate.py) -- a detector that runs
before the modality gate to catch uploads that aren't chest X-rays or CT
scans at all (photos, screenshots, scanned documents), which the existing
modality gate and OOD detector were never calibrated to catch (both are
closed-set classifiers with no "none of the above").

Four negative sources, saved as PNGs under data/sanity_negatives/{split}/:
  1. cifar   - real photos (CIFAR-10, auto-downloaded via torchvision) --
               broad, realistic "this is just a photo of something" signal.
  2. doc     - synthetic scanned-document-style images (text-block/table
               layouts on a paper background) -- the receipt/report case
               that a plain color-channel check can't catch (documents are
               near-grayscale too).
  3. texture - synthetic gradients/noise/checkerboards/shapes -- abstract
               non-photographic, non-document images.
  4. repo    - real non-medical images already in this repo (confusion
               matrices, training curves, architecture/class diagrams, app
               screenshots, logo/hero art), lightly augmented for volume --
               covers the specific "screenshot" failure case called out in
               main.py's docstring.

Split sizes are chosen so each split's negative pool is large enough to
comfortably outnumber even the smaller CT positive set (829 train images)
without being wastefully large for training time.
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "sanity_negatives"
SEED = 1234

# (split, cifar_count, doc_count, texture_count, repo_augments_per_source)
# cifar left at 0 -- CIFAR-10 download was too slow on this network (~50KB/s,
# would've taken ~45min); doc+texture+repo need no network access at all.
SPLITS = {
    "train": dict(cifar=0, doc=3500, texture=3000, repo_mult=60),
    "val":   dict(cifar=0, doc=350,  texture=300,  repo_mult=8),
    "test":  dict(cifar=0, doc=350,  texture=300,  repo_mult=8),
}

REPO_SOURCE_IMAGES = [
    "eval_outputs/xray/xray_confusion_matrix.png",
    "eval_outputs/xray_phase35/xray_phase35_confusion_matrix.png",
    "eval_outputs/ct_scan/ct_confusion_matrix.png",
    "eval_outputs/ct_scan_phase35_v2/ct_phase35_v2_confusion_matrix.png",
    "ml_logs/ct_crossval_curves.png",
    "ml_logs/ct_phase35_v2_curves.png",
    "ml_logs/modality_gate_curves.png",
    "report/images/architecture.png",
    "report/images/class_diagram.png",
    "report/images/er_diagram.png",
    "report/images/gantt_timeline.png",
    "report/images/logo.jpg",
    "report/images/ScreenShots/01_landing_dark.png",
    "report/images/ScreenShots/02_signup_page.png",
    "report/images/ScreenShots/03_signup_verify_prompt.png",
    "report/images/ScreenShots/04_main_app_upload_dark.png",
    "report/images/ScreenShots/06_chat_widget_context.png",
    "report/images/ScreenShots/08_main_app_light.png",
    "report/images/ScreenShots/09_scan_history_light.png",
    "report/images/ScreenShots/10_scan_history_modal.png",
    "report/images/ScreenShots/11_profile_page.png",
    "report/images/ScreenShots/12_landing_after_logout.png",
    "frontend/src/assets/hero.png",
]


def gen_texture(rng: random.Random, size=(256, 256)):
    kind = rng.choice(["gradient", "noise", "checker", "shapes", "stripes"])
    w, h = size
    if kind == "gradient":
        arr = np.zeros((h, w, 3), dtype=np.float32)
        c1 = np.array([rng.randint(0, 255) for _ in range(3)])
        c2 = np.array([rng.randint(0, 255) for _ in range(3)])
        for y in range(h):
            t = y / h
            arr[y, :, :] = c1 * (1 - t) + c2 * t
        arr += np.random.RandomState(rng.randint(0, 1 << 30)).randint(-10, 10, size=arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    elif kind == "noise":
        r = np.random.RandomState(rng.randint(0, 1 << 30))
        arr = r.randint(0, 255, size=(h, w, 3)).astype(np.uint8)
        img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(rng.uniform(0, 3)))
    elif kind == "checker":
        cell = rng.randint(8, 40)
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        c1 = [rng.randint(0, 255) for _ in range(3)]
        c2 = [rng.randint(0, 255) for _ in range(3)]
        for y in range(h):
            for x in range(0, w, cell):
                color = c1 if ((x // cell) + (y // cell)) % 2 == 0 else c2
                arr[y, x:x + cell] = color
        img = Image.fromarray(arr)
    elif kind == "stripes":
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        stripe = rng.randint(4, 30)
        c1 = [rng.randint(0, 255) for _ in range(3)]
        c2 = [rng.randint(0, 255) for _ in range(3)]
        for x in range(0, w, stripe):
            color = c1 if (x // stripe) % 2 == 0 else c2
            arr[:, x:x + stripe] = color
        img = Image.fromarray(arr)
    else:  # shapes
        bg = tuple(rng.randint(0, 255) for _ in range(3))
        img = Image.new("RGB", size, bg)
        draw = ImageDraw.Draw(img)
        for _ in range(rng.randint(3, 10)):
            shape = rng.choice(["ellipse", "rectangle", "line"])
            x0, y0 = rng.randint(0, w), rng.randint(0, h)
            x1, y1 = rng.randint(0, w), rng.randint(0, h)
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])
            color = tuple(rng.randint(0, 255) for _ in range(3))
            if shape == "ellipse":
                draw.ellipse([xmin, ymin, xmax, ymax], fill=color)
            elif shape == "rectangle":
                draw.rectangle([xmin, ymin, xmax, ymax], fill=color)
            else:
                draw.line([x0, y0, x1, y1], fill=color, width=rng.randint(1, 8))
    return img.convert("RGB")


def gen_document(rng: random.Random, size=(300, 400)):
    w, h = size
    grayscale = rng.random() < 0.6
    bg = rng.randint(235, 255)
    paper = (bg, bg, bg) if grayscale else (
        bg, bg - rng.randint(0, 8), bg - rng.randint(0, 15)
    )
    img = Image.new("RGB", size, paper)
    draw = ImageDraw.Draw(img)

    layout = rng.choice(["text_block", "table", "form", "letterhead"])
    ink = (rng.randint(0, 40),) * 3

    if layout == "text_block":
        y = rng.randint(10, 30)
        while y < h - 20:
            x0 = rng.randint(15, 40)
            x1 = rng.randint(x0 + 40, w - 15)
            draw.rectangle([x0, y, x1, y + rng.randint(6, 12)], fill=ink)
            y += rng.randint(14, 26)
    elif layout == "table":
        rows = rng.randint(5, 12)
        cols = rng.randint(2, 4)
        margin = 15
        row_h = (h - 2 * margin) // rows
        col_w = (w - 2 * margin) // cols
        for r in range(rows + 1):
            y = margin + r * row_h
            draw.line([margin, y, w - margin, y], fill=(120, 120, 120), width=1)
        for c in range(cols + 1):
            x = margin + c * col_w
            draw.line([x, margin, x, h - margin], fill=(120, 120, 120), width=1)
        for r in range(rows):
            for c in range(cols):
                if rng.random() < 0.7:
                    x0 = margin + c * col_w + 6
                    y0 = margin + r * row_h + row_h // 3
                    draw.rectangle([x0, y0, x0 + rng.randint(20, col_w - 12), y0 + 6], fill=ink)
    elif layout == "form":
        y = 20
        for _ in range(rng.randint(6, 10)):
            label_w = rng.randint(50, 100)
            draw.rectangle([20, y, 20 + label_w, y + 8], fill=ink)
            draw.line([30 + label_w, y + 8, w - 20, y + 8], fill=(80, 80, 80), width=1)
            y += rng.randint(24, 40)
    else:  # letterhead
        accent = tuple(rng.randint(0, 255) for _ in range(3)) if rng.random() < 0.5 else ink
        draw.rectangle([20, 15, w - 20, 45], fill=accent)
        y = 65
        while y < h - 20:
            x1 = rng.randint(80, w - 20)
            draw.rectangle([20, y, x1, y + rng.randint(6, 10)], fill=ink)
            y += rng.randint(16, 28)

    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    arr = np.array(img, dtype=np.float32)
    arr += np.random.RandomState(rng.randint(0, 1 << 30)).normal(0, rng.uniform(1, 6), size=arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def augment_repo_image(img: Image.Image, rng: random.Random, size=(256, 256)):
    img = img.convert("RGB")
    w, h = img.size
    # random crop (zoom in on a sub-region so augmented copies differ)
    scale = rng.uniform(0.5, 1.0)
    cw, ch = int(w * scale), int(h * scale)
    if cw > 0 and ch > 0 and cw <= w and ch <= h:
        x0 = rng.randint(0, w - cw) if w > cw else 0
        y0 = rng.randint(0, h - ch) if h > ch else 0
        img = img.crop((x0, y0, x0 + cw, y0 + ch))
    img = img.resize(size, Image.BILINEAR)
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    arr = np.array(img, dtype=np.float32)
    arr *= rng.uniform(0.85, 1.15)
    arr = np.clip(arr, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def prepare_cifar(n_needed: dict, out_root: Path):
    from torchvision.datasets import CIFAR10

    cache_dir = ROOT / "data" / "_cifar10_cache"
    print("Downloading/loading CIFAR-10 (cached after first run)...")
    train_set = CIFAR10(root=str(cache_dir), train=True, download=True)
    test_set = CIFAR10(root=str(cache_dir), train=False, download=True)

    all_images = [train_set[i][0] for i in range(len(train_set))] + \
                 [test_set[i][0] for i in range(len(test_set))]
    rng = random.Random(SEED)
    rng.shuffle(all_images)

    idx = 0
    for split, count in n_needed.items():
        out_dir = out_root / split / "cifar"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img = all_images[idx % len(all_images)]
            idx += 1
            img.resize((256, 256), Image.BILINEAR).save(out_dir / f"cifar_{split}_{i:05d}.png")
    print(f"cifar: wrote {sum(n_needed.values())} images")


def prepare_synthetic(kind: str, gen_fn, n_needed: dict, out_root: Path, seed_offset: int):
    for split, count in n_needed.items():
        out_dir = out_root / split / kind
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(SEED + seed_offset + hash(split) % 1000)
        for i in range(count):
            img = gen_fn(rng)
            img.save(out_dir / f"{kind}_{split}_{i:05d}.png")
        print(f"{kind} [{split}]: wrote {count} images")


def prepare_repo(splits: dict, out_root: Path):
    sources = [ROOT / p for p in REPO_SOURCE_IMAGES if (ROOT / p).exists()]
    missing = [p for p in REPO_SOURCE_IMAGES if not (ROOT / p).exists()]
    if missing:
        print(f"repo: skipping {len(missing)} missing source files: {missing}")
    print(f"repo: {len(sources)} source images found")

    rng = random.Random(SEED + 777)
    # split source files themselves across train/val/test so val/test never
    # see augmented copies of a train source image (avoid leakage)
    rng.shuffle(sources)
    n = len(sources)
    cut1, cut2 = int(n * 0.7), int(n * 0.85)
    source_splits = {
        "train": sources[:cut1],
        "val": sources[cut1:cut2],
        "test": sources[cut2:],
    }

    for split, cfg in SPLITS.items():
        out_dir = out_root / split / "repo"
        out_dir.mkdir(parents=True, exist_ok=True)
        mult = cfg["repo_mult"]
        split_sources = source_splits[split] or sources  # fall back if a split got none
        count = 0
        for src in split_sources:
            try:
                img = Image.open(src)
                img.load()
            except Exception as e:
                print(f"repo: skipping unreadable {src}: {e}")
                continue
            for i in range(mult):
                aug = augment_repo_image(img, rng)
                aug.save(out_dir / f"repo_{split}_{src.stem}_{i:03d}.png")
                count += 1
        print(f"repo [{split}]: wrote {count} images from {len(split_sources)} sources")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cifar_counts = {s: cfg["cifar"] for s, cfg in SPLITS.items()}
    if any(cifar_counts.values()):
        prepare_cifar(cifar_counts, OUT_DIR)
    else:
        print("cifar: skipped (all counts are 0)")
    prepare_synthetic("doc", gen_document, {s: cfg["doc"] for s, cfg in SPLITS.items()}, OUT_DIR, seed_offset=1)
    prepare_synthetic("texture", gen_texture, {s: cfg["texture"] for s, cfg in SPLITS.items()}, OUT_DIR, seed_offset=2)
    prepare_repo(SPLITS, OUT_DIR)

    print("\nDone. Counts per split:")
    for split in SPLITS:
        split_dir = OUT_DIR / split
        n = sum(1 for f in split_dir.rglob("*.png"))
        print(f"  {split}: {n} images")


if __name__ == "__main__":
    main()
