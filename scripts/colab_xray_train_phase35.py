"""
Colab training cell — X-ray model (Phase 3.5 retrain, fresh from scratch).

Paste this into a Colab code cell. It assumes:
  - The repo is cloned at /content/sdp/ (or adjust ROOT below)
  - data/xray.zip has been rebuilt locally with the Phase 3.5 preprocessing
    (mask-guided lung crop for COVID source, 8% border crop for TB source)
    and re-uploaded to Drive / made available to Colab.

Run order:
  [1] Mount Drive + clone/pull repo
  [2] Unzip data/xray.zip  (this cell)
  [3] Run training         (this cell)
  [4] Download checkpoint + ml_logs/
"""

import subprocess, sys, os
from pathlib import Path

# ── 0. Config ─────────────────────────────────────────────────────────────────
ROOT       = Path("/content/sdp")           # adjust if your repo lives elsewhere
DATA_ZIP   = ROOT / "data" / "xray.zip"    # upload the new zip here
DATA_DIR   = ROOT / "data" / "xray"
BACKBONE   = "densenet121"
EPOCHS     = 20                             # fresh retrain: a few more epochs than fine-tune
WARMUP     = 3
BATCH_SIZE = 32
LR         = 1e-3
FINETUNE_LR = 1e-4
RUN_NAME   = "xray_densenet121_phase35"    # distinguishes from old checkpoint

# ── 1. Unzip data (idempotent — safe to re-run) ───────────────────────────────
if DATA_ZIP.exists():
    print("Unzipping xray.zip ...")
    subprocess.run(
        ["unzip", "-q", "-o", str(DATA_ZIP), "-d", str(ROOT / "data")],
        check=True,
    )
    print("Done.")
else:
    print(f"WARNING: {DATA_ZIP} not found — assuming {DATA_DIR} already exists")

assert DATA_DIR.exists(), f"data/xray/ not found at {DATA_DIR}"

# ── 2. Quick sanity check: count images per split ─────────────────────────────
for split in ("train", "val", "test"):
    n = sum(1 for _ in (DATA_DIR / split).rglob("*.png"))
    classes = sorted(p.name for p in (DATA_DIR / split).iterdir() if p.is_dir())
    print(f"  {split}: {n} images, classes={classes}")

# ── 3. Train from SCRATCH (no --resume, no pretrained checkpoint) ─────────────
os.chdir(ROOT / "ml")   # train.py does relative imports from ml/

result = subprocess.run(
    [
        sys.executable, "train.py",
        "--data_dir",    str(DATA_DIR),
        "--backbone",    BACKBONE,
        "--epochs",      str(EPOCHS),
        "--warmup_epochs", str(WARMUP),
        "--batch_size",  str(BATCH_SIZE),
        "--lr",          str(LR),
        "--finetune_lr", str(FINETUNE_LR),
        "--run_name",    RUN_NAME,
        "--num_workers", "2",
        # no --device → auto-detects CUDA on Colab T4/A100
    ],
    check=True,
)

print("\n=== Training complete ===")
print(f"Checkpoint: checkpoints/{RUN_NAME}_best.pt")
print(f"Curves:     ml_logs/{RUN_NAME}_curves.png")
print(f"History:    ml_logs/{RUN_NAME}_history.json")
