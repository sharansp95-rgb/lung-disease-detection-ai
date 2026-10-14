"""
Generates a tiny synthetic image-classification dataset and runs one full
training pass through the real pipeline (dataset.py + model.py + train.py),
to catch pipeline bugs cheaply before touching real data.

    python smoke_test.py
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import train as train_module

CLASSES = ["classA", "classB", "classC"]
# deliberately imbalanced train split so the smoke test also exercises
# compute_class_weights() producing genuinely unequal weights, not just
# confirming the balanced-weights case
TRAIN_COUNTS = {"classA": 10, "classB": 5, "classC": 2}
VAL_TEST_COUNTS = {"classA": 4, "classB": 4, "classC": 4}
SYNTH_IMAGE_SIZE = 64


def make_synthetic_dataset(root: Path, seed: int = 0):
    rng = np.random.default_rng(seed)
    per_split = {"train": TRAIN_COUNTS, "val": VAL_TEST_COUNTS, "test": VAL_TEST_COUNTS}
    for split, counts in per_split.items():
        for cls in CLASSES:
            cls_dir = root / split / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(counts[cls]):
                arr = rng.integers(0, 256, size=(SYNTH_IMAGE_SIZE, SYNTH_IMAGE_SIZE), dtype=np.uint8)
                Image.fromarray(arr, mode="L").save(cls_dir / f"{cls}_{i}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="keep the temp synthetic dataset dir for inspection")
    args = parser.parse_args()

    tmp_dir = Path(tempfile.mkdtemp(prefix="smoke_test_data_"))
    print(f"[smoke_test] synthetic dataset -> {tmp_dir}")
    make_synthetic_dataset(tmp_dir)

    tmp_out = Path(tempfile.mkdtemp(prefix="smoke_test_out_"))
    ckpt_dir = tmp_out / "checkpoints"
    log_dir = tmp_out / "logs"

    train_args = argparse.Namespace(
        data_dir=str(tmp_dir),
        backbone="resnet18",       # smallest/fastest supported backbone
        epochs=2,
        warmup_epochs=1,
        batch_size=4,
        lr=1e-3,
        finetune_lr=1e-4,
        num_classes=None,
        num_workers=0,
        device=None,               # autodetect: exercises the real GPU path if available
        output_dir=str(ckpt_dir),
        log_dir=str(log_dir),
        run_name="smoke_test",
        seed=42,
    )
    # run() resolves output_dir/log_dir relative to ROOT (repo root); override
    # by monkeypatching ROOT so checkpoints land in our temp dir regardless.
    train_module.ROOT = Path(".")

    try:
        result = train_module.run(train_args)
    except Exception:
        print("[smoke_test] FAILED -- exception during training", file=sys.stderr)
        raise
    finally:
        if not args.keep:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    assert Path(result["checkpoint"]).exists(), "checkpoint file was not created"
    assert len(result["history"]["train_loss"]) == 2, "expected 2 epochs of history"
    assert result["class_names"] == CLASSES, f"unexpected class names: {result['class_names']}"
    assert len(result["class_weights"]) == len(CLASSES)
    assert 0.0 <= result["best_val_acc"] <= 1.0

    print("[smoke_test] checkpoint:", result["checkpoint"])
    print("[smoke_test] class_names:", result["class_names"])
    print("[smoke_test] class_weights:", result["class_weights"])
    print("[smoke_test] best_val_acc:", result["best_val_acc"])
    print("[smoke_test] SMOKE TEST PASSED")

    if not args.keep:
        shutil.rmtree(tmp_out, ignore_errors=True)


if __name__ == "__main__":
    main()
