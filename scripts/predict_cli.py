"""
Terminal-only image inference (one image or several per run): runs the exact same pipeline as
backend/main.py's /predict/xray and /predict/ct routes (blank-image check ->
color-content check -> sanity gate -> modality gate -> [X-ray only: lung
crop] -> OOD check -> classifier + Grad-CAM), but as a standalone CLI script
with no web server involved. Built for demoing the model itself, gate by
gate, without the website in the loop.

    python predict_cli.py --image "../test images/x ray/normal/img_189af4.png" --modality xray
    python predict_cli.py --image "../test images/ct scan/benign/img_159819.jpg" --modality ct --save-heatmap out.png

Several images in ONE run (models load and run their reproduction checks
once, then every image goes through the full pipeline with the same
printouts, followed by a summary table):

    # repeat --image/--modality pairs (in order); --save-heatmap is optional,
    # but if given it must be given once per image ("-" = don't save one)
    python predict_cli.py \
        --image "../test images/x ray/normal/img_189af4.png" --modality xray --save-heatmap ../demo/heatmap_normal.png \
        --image "../test images/ct scan/benign/img_159819.jpg" --modality ct --save-heatmap -

    # or a batch file: one "path,modality[,heatmap_path]" per line, '#'
    # comments and blank lines ignored, relative paths resolved against the
    # batch file's own folder
    python predict_cli.py --batch ../demo/demo_batch.txt

In multi-image mode a rejected image is reported and the run continues with
the next one; the exit code is 0 unless an image could not be processed at
all (missing/unreadable file). Single-image mode behaves exactly as before
(a rejection exits with code 1).

This script does NOT re-implement the pipeline: it imports backend/main.py
and calls its own functions (checkpoint table, model loading, image
validation, sanity gate, modality gate, lung crop, OOD check, classifier +
Grad-CAM), so it always serves the same checkpoints, with the same
thresholds, in the same order, as the website. Only the printouts are new.

All models are loaded at startup and the same reproduction checks the web
app runs run first (watch the log lines -- each one reproduces the
checkpoint's claimed accuracy against a fresh eval pass before anything is
allowed to run).
"""

import argparse
import base64
import io
import logging
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import main as backend  # noqa: E402  (backend/main.py: the web app's own pipeline)
import config  # noqa: E402
from gradcam_utils import (  # noqa: E402
    predict_with_heatmap, predict_with_heatmap_xrv, preprocess_image, preprocess_image_xrv,
)
from models import ReproductionCheckFailed, is_xrv_backbone  # noqa: E402
from seg_utils import SegReproductionCheckFailed, compute_lung_crop  # noqa: E402
from report import generate_report  # noqa: E402

logger = logging.getLogger("predict_cli")


class Rejected(Exception):
    """An image was turned away by one of the pipeline's checks (a normal
    outcome, same as the web app's 4xx responses), not a crash."""


class _UploadShim:
    """Just enough of fastapi.UploadFile for backend._read_and_validate_image."""

    def __init__(self, path: Path):
        self.filename = path.name
        self.file = io.BufferedReader(io.FileIO(path, "rb"))


def load_everything():
    print("\n=== loading models + running reproduction checks (same as backend startup) ===")
    t0 = time.perf_counter()
    try:
        backend._ensure_models_loaded()  # the web app's own loader: 5 classifiers + OOD stats + seg model
    except (ReproductionCheckFailed, SegReproductionCheckFailed) as e:
        logger.critical(str(e))
        raise
    dt = time.perf_counter() - t0
    for name in backend.MODEL_NAMES:
        b = backend._bundles[name]
        print(f"[{name}] checkpoint={backend.CHECKPOINTS[name].name} backbone={b.backbone} "
              f"claims val_acc={b.checkpoint_val_acc:.4f} reproduced={b.reproduced_val_acc:.4f}")
    seg = backend._seg_bundle
    print(f"[seg] checkpoint={backend.SEG_CHECKPOINT.name} claims dice={seg.checkpoint_val_dice:.4f} "
          f"iou={seg.checkpoint_val_iou:.4f}, reproduced dice={seg.reproduced_val_dice:.4f} "
          f"iou={seg.reproduced_val_iou:.4f}")
    print(f"=== all models loaded and verified in {dt:.1f}s ===\n")


def read_and_validate_image(path: Path) -> Image.Image:
    # Printouts (display only); the accept/reject decision is the backend's.
    probe = Image.open(path)
    probe.load()
    gray_std = float(np.array(probe.convert("L"), dtype=np.float32).std())
    rgb = np.array(probe.convert("RGB"), dtype=np.float32)
    color_dev = float((rgb.max(axis=2) - rgb.min(axis=2)).mean())
    print(f"[blank-image check] grayscale std = {gray_std:.1f} (reject if < {backend.BLANK_IMAGE_STD_THRESHOLD})")
    print(f"[color-content check] channel spread = {color_dev:.1f} (reject if > {backend.COLOR_CONTENT_THRESHOLD})")

    shim = _UploadShim(path)
    try:
        return backend._read_and_validate_image(shim)
    except HTTPException as e:
        raise Rejected(e.detail)
    finally:
        shim.file.close()


def run_pipeline(image_path: Path, modality: str, save_heatmap) -> tuple[str, float]:
    """Runs one image through the web app's pipeline, printing each step.
    Returns (predicted_class, confidence); raises Rejected if a check fails."""
    label = backend.MODALITY_LABEL[modality]

    print(f"--- running pipeline on {image_path.name} as '{label}' ---\n")

    bundle = backend._bundles[modality]
    ood_detector = backend._ood_detectors[modality]

    image = read_and_validate_image(image_path)
    input_tensor = preprocess_image(image, bundle.device)

    # Same order as backend._run_prediction: sanity gate, then modality gate,
    # both on the full (uncropped) image; lung crop only after both pass.
    sanity = backend._sanity_predict(modality, input_tensor)
    print(f"[sanity gate] verdict: {sanity} (must be 'scan' to continue)")
    if sanity != "scan":
        raise Rejected(f"does not look like a {label}.")

    gate, gate_conf, _, _ = backend._gate_predict(input_tensor)
    print(f"[modality gate] predicted modality: {gate} (confidence {gate_conf:.1%}; "
          f"must be '{modality}' with >= {config.MODALITY_GATE_CONFIDENCE_THRESHOLD:.0%} to continue)")
    if gate_conf < config.MODALITY_GATE_CONFIDENCE_THRESHOLD:
        raise Rejected(f"modality uncertain -- please confirm this is a {label} "
                       f"(confidence: {gate_conf:.1%}).")
    if gate != modality:
        raise Rejected(f"looks like a {backend.MODALITY_LABEL.get(gate, gate)}, not a {label}.")

    if modality == "xray":
        image, used_seg = compute_lung_crop(image, backend._seg_bundle)
        print(f"[lung crop] used segmentation model: {used_seg}")
        input_tensor = (
            preprocess_image_xrv(image, bundle.device) if is_xrv_backbone(bundle.backbone)
            else preprocess_image(image, bundle.device)
        )

    with bundle.lock:
        is_ood, distance = ood_detector.check(bundle.model, input_tensor)
        print(f"[OOD check] feature-space distance = {distance:.1f} (threshold {ood_detector.threshold:.1f})")
        if is_ood:
            raise Rejected(f"too far from training distribution "
                           f"(distance {distance:.1f} > {ood_detector.threshold:.1f}).")

        pred_idx, probs, heatmap_uri = (
            predict_with_heatmap_xrv(bundle.model, input_tensor) if is_xrv_backbone(bundle.backbone)
            else predict_with_heatmap(bundle.model, input_tensor)
        )

    predicted_class = bundle.class_names[pred_idx]
    confidence = float(probs[pred_idx])

    print("\n=== RESULT ===")
    print(f"predicted class : {predicted_class}")
    print(f"confidence      : {confidence:.1%}")
    print("class probabilities:")
    for name, p in zip(bundle.class_names, probs):
        print(f"  {name:<20s} {p:.1%}")
    if confidence < backend.MIN_CONFIDENCE[modality] + 0.15:
        print("[warning] confidence is low -- treat this prediction as uncertain.")

    report_text, report_source = generate_report(modality, predicted_class, confidence,
                                                   {n: float(p) for n, p in zip(bundle.class_names, probs)})
    print(f"\n[report ({report_source})]\n{report_text}")

    if save_heatmap:
        b64_data = heatmap_uri.split(",", 1)[1]
        Path(save_heatmap).write_bytes(base64.b64decode(b64_data))
        print(f"\n[grad-cam] heatmap saved to {save_heatmap}")

    return predicted_class, confidence


def read_batch_file(batch_path: Path) -> list[tuple[Path, str, str | None]]:
    if not batch_path.is_file():
        raise SystemExit(f"batch file not found: {batch_path}")
    jobs = []
    for lineno, line in enumerate(batch_path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [s.strip() for s in line.split(",")]
        if len(parts) not in (2, 3) or parts[1] not in ("xray", "ct"):
            raise SystemExit(f"{batch_path}:{lineno}: expected 'path,xray|ct[,heatmap_path]', got: {line}")
        image = Path(parts[0])
        if not image.is_absolute():
            image = batch_path.parent / image
        heatmap = parts[2] if len(parts) == 3 and parts[2] not in ("", "-") else None
        if heatmap and not Path(heatmap).is_absolute():
            heatmap = str(batch_path.parent / heatmap)
        jobs.append((image, parts[1], heatmap))
    if not jobs:
        raise SystemExit(f"{batch_path}: no images listed")
    return jobs


def build_jobs(parser: argparse.ArgumentParser, args) -> list[tuple[Path, str, str | None]]:
    if args.batch:
        if args.image or args.modality or args.save_heatmap:
            parser.error("use either --batch or --image/--modality, not both")
        return read_batch_file(Path(args.batch))

    if not args.image or not args.modality:
        parser.error("--image and --modality are required (or use --batch)")
    if len(args.image) != len(args.modality):
        parser.error(f"got {len(args.image)} --image but {len(args.modality)} --modality; "
                     f"give one --modality per --image, in the same order")
    heatmaps = args.save_heatmap or [None] * len(args.image)
    if len(heatmaps) != len(args.image):
        parser.error(f"got {len(args.image)} --image but {len(heatmaps)} --save-heatmap; give one "
                     f"--save-heatmap per --image (use '-' for no heatmap) or none at all")
    heatmaps = [None if h in (None, "", "-") else h for h in heatmaps]
    return [(Path(i), m, h) for i, m, h in zip(args.image, args.modality, heatmaps)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", action="append", help="path to the image file (repeatable)")
    parser.add_argument("--modality", action="append", choices=["xray", "ct"],
                        help="modality of the matching --image (repeatable, same order)")
    parser.add_argument("--save-heatmap", action="append", default=None,
                        help="path to save the Grad-CAM overlay PNG for the matching --image ('-' = none)")
    parser.add_argument("--batch", default=None,
                        help="text file with one 'path,modality[,heatmap_path]' per line")
    args = parser.parse_args()

    jobs = build_jobs(parser, args)
    # Fail fast on a typo'd path instead of after the multi-minute model load.
    missing = [str(p) for p, _, _ in jobs if not p.is_file()]
    if missing:
        raise SystemExit("image file(s) not found:\n  " + "\n  ".join(missing))

    load_everything()

    if len(jobs) == 1:
        image_path, modality, save_heatmap = jobs[0]
        try:
            run_pipeline(image_path, modality, save_heatmap)
        except Rejected as e:
            raise SystemExit(f"REJECTED: {e}")
        return

    summary = []
    for i, (image_path, modality, save_heatmap) in enumerate(jobs, 1):
        print(f"\n#################### image {i}/{len(jobs)} ####################")
        t0 = time.perf_counter()
        try:
            predicted_class, confidence = run_pipeline(image_path, modality, save_heatmap)
            outcome = f"{predicted_class} {confidence:.1%}"
        except Rejected as e:
            print(f"REJECTED: {e}")
            outcome = "REJECTED"
        except Exception as e:  # keep going so one bad file doesn't sink the batch
            logger.exception(f"failed on {image_path}")
            outcome = f"ERROR ({type(e).__name__})"
        summary.append((image_path.name, modality, outcome, time.perf_counter() - t0))

    print(f"\n=== SUMMARY ({len(jobs)} images, models loaded once) ===")
    for name, modality, outcome, dt in summary:
        print(f"  {name:<22s} {modality:<5s} {outcome:<28s} {dt:5.1f}s")
    if any(outcome.startswith("ERROR") for _, _, outcome, _ in summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
