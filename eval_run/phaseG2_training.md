# Phase G.2: 3D-aware CT classifier training

**Scope: training only. No validation sweep, no Grad-CAM, no pipeline test.**
The exact command from Phase G.1 §6 was run unmodified. One new checkpoint was written
(`ct_lidc_nodulecrop_v4_stack3.pt`). Nothing else was created, modified, or deployed.

---

## 0. Context files read before starting

- **`eval_run/RESUME_STATE.md`** — read in full (1,310 lines). Most recent entry: Phase G.1
  (2026-09-10), documenting 3,461/3,461 stacks extracted, architecture verified as needing no
  change, smoke test passed, training script written but NOT run.
- **`eval_run/phaseG1_setup.md`** — read in full (558 lines). Confirmed: the exact command is
  `python -u train_stacks.py --min_real_slices 3 --tag v4_stack3` from
  `D:/7th sem/sdp/eval_run/phaseG1`, producing
  `checkpoints/ct_lidc_nodulecrop_v4_stack3.pt` (best val balanced accuracy).

**No conflicts found** between this task's instructions and anything documented in RESUME_STATE.md
as a known risk or already-tried-and-failed approach. The `--min_real_slices 3` flag is the
recommended primary configuration per Phase G.1 §4.3 (removes the class-correlated padding
shortcut).

---

## 1. Command and parameters — zero deviations

```
cd "D:/7th sem/sdp/eval_run/phaseG1"
python -u train_stacks.py --min_real_slices 3 --tag v4_stack3
```

**Exactly the command from Phase G.1 §6, with no modifications.** All parameters at their
Phase G.1 defaults:

| parameter | value | matches Phase G.1? |
|---|---|---|
| epochs | 15 | ✔ |
| warmup_epochs | 2 | ✔ |
| batch_size | **8** | ✔ — no OOM, no reduction needed |
| lr (warmup) | 1e-3 | ✔ |
| finetune_lr | 1e-4 | ✔ |
| seed | 42 | ✔ |
| amp | 1 (True) | ✔ |
| num_workers | 0 | ✔ |
| min_real_slices | **3** | ✔ |
| manifest | `phaseG1/stacks_manifest.csv` | ✔ |
| ckpt | `checkpoints/ct_lidc_nodulecrop_v4_stack3.pt` | ✔ |
| tag | v4_stack3 | ✔ |

**No CUDA out-of-memory occurred. Batch size was not changed.**

---

## 2. Dataset confirmation

| | train | val |
|---|---|---|
| Benign | 512 | 97 |
| Malignant | 390 | 73 |
| Normal | 1,156 | 237 |
| **total** | **2,058** | **407** |

Matches Phase G.1 §4.3's `min_real_slices=3` table exactly. The channel guard
(`assert_channels_differ`) passed on 64 sampled 3-slice train stacks before the first epoch —
confirming no grayscale collapse.

Inverse-frequency class weights: Benign=1.3398, Malignant=1.7590, Normal=0.5934.

---

## 3. Per-epoch training results

| epoch | phase | train_loss | train_acc | train_bal | val_loss | val_acc | val_bal | B recall | M recall | N recall | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | warmup (frozen) | 1.0446 | 0.4820 | 0.4653 | 0.9509 | 0.6216 | 0.5262 | 0.175 | 0.589 | 0.814 | 18.4 |
| 2 | warmup (frozen) | 0.9280 | 0.5928 | 0.5695 | 0.9413 | 0.6216 | 0.5776 | 0.629 | 0.425 | 0.679 | 16.6 |
| 3 | fine-tune | 0.6358 | 0.7663 | 0.7317 | 0.5435 | 0.8403 | 0.7976 | 0.629 | 0.836 | 0.928 | 35.0 |
| 4 | fine-tune | 0.4303 | 0.8654 | 0.8274 | 0.4619 | 0.8649 | 0.8245 | 0.742 | 0.795 | 0.937 | 34.6 |
| 5 | fine-tune | 0.3277 | 0.9014 | 0.8726 | 0.4029 | 0.8624 | 0.8351 | 0.691 | 0.890 | 0.924 | 35.9 |
| 6 | fine-tune | 0.2850 | 0.9198 | 0.8934 | 0.3990 | **0.8968** | 0.8518 | 0.804 | 0.781 | 0.970 | 37.2 |
| 7 | fine-tune | 0.2730 | 0.9227 | 0.9007 | 0.4275 | 0.8747 | 0.8173 | 0.773 | 0.712 | 0.966 | 37.4 |
| 8 | fine-tune | 0.2197 | 0.9368 | 0.9173 | 0.4254 | 0.8845 | 0.8358 | 0.742 | 0.795 | 0.970 | 36.1 |
| 9 | fine-tune | 0.1966 | 0.9388 | 0.9190 | 0.4741 | 0.8821 | 0.8276 | 0.804 | 0.712 | 0.966 | 37.0 |
| 10 | fine-tune | 0.1931 | 0.9456 | 0.9290 | 0.4805 | 0.8943 | 0.8429 | 0.814 | 0.740 | 0.975 | 36.5 |
| 11 | fine-tune | 0.1448 | 0.9553 | 0.9390 | 0.4925 | 0.8919 | 0.8503 | 0.763 | 0.822 | 0.966 | 36.5 |
| 12 | fine-tune | 0.1489 | 0.9582 | 0.9428 | 0.4493 | 0.8919 | 0.8474 | 0.732 | 0.836 | 0.975 | 36.7 |
| **13** | fine-tune | 0.1439 | 0.9558 | 0.9442 | 0.4671 | **0.8943** | **0.8637** | **0.856** | **0.795** | **0.941** | 37.1 |
| 14 | fine-tune | 0.1154 | 0.9670 | 0.9604 | 0.4986 | 0.8821 | 0.8323 | 0.732 | 0.795 | 0.970 | 38.0 |
| 15 | fine-tune | 0.1235 | 0.9694 | 0.9616 | 0.5223 | 0.8894 | 0.8404 | 0.784 | 0.767 | 0.970 | 37.4 |

---

## 4. Best checkpoints

| criterion | epoch | val_acc | val_bal | path |
|---|---|---|---|---|
| **best val balanced accuracy** | **13** | **0.8943** | **0.8637** | **`D:\7th sem\sdp\checkpoints\ct_lidc_nodulecrop_v4_stack3.pt`** (28,459,122 bytes) |
| best val accuracy | 6 | 0.8968 | 0.8518 | `D:\7th sem\sdp\eval_run\phaseG1\ct_lidc_nodulecrop_v4_stack3_valacc.pt` (28,464,253 bytes) |

The primary checkpoint (`ct_lidc_nodulecrop_v4_stack3.pt`) is the best-by-balanced-accuracy one,
consistent with Phase C.1's checkpointing strategy.

**Epoch 13's per-class recalls at the best checkpoint: Benign 0.856, Malignant 0.795, Normal 0.941.**
This is the most balanced recall profile across the entire training run — Benign and Malignant are
within 6.1 points of each other, while Normal stays above 0.94.

---

## 5. Comparison with prior checkpoints (val set, same-recipe)

| checkpoint | input | val_acc | val_bal | B recall | M recall | N recall |
|---|---|---|---|---|---|---|
| v1 (Phase C) | single slice ×3 | 0.9030 | 0.8591 | — | — | — |
| **v2_jitter** (Phase C.1) | single slice ×3 + jitter | **0.8693** | **0.8292** | — | — | — |
| **v4_stack3** (this phase) | **3 adjacent slices** | **0.8943** | **0.8637** | **0.856** | **0.795** | **0.941** |

**v4_stack3 outperforms v2_jitter by +2.5 points on val accuracy and +3.5 points on balanced
accuracy**, and is within 1 point of v1 on accuracy while exceeding it on balanced accuracy by
+0.5 points. However, the comparison is not straightforward because:
- v4_stack3 trains on the `min_real_slices=3` subset (2,058 rows), not the full 2,450
- v1/v2 use Phase B's single-slice crops with `Grayscale(3)` duplication
- v2_jitter includes in-plane translation jitter, v4_stack3 does not

**No claims are made about through-plane tolerance or end-to-end pipeline accuracy from these
numbers alone.** That is Phase G.3's scope.

---

## 6. Resources

| metric | value | Phase C.1 reference |
|---|---|---|
| total wall time | **8.5 min** | 9.6 min |
| peak CUDA memory | **0.62 GB of 4.00** | 0.62 GB |
| mean epoch (warmup) | 17.5 s | — |
| mean epoch (fine-tune) | 36.3 s | 38.3 s |
| device | RTX 3050 Laptop GPU, 4 GB | same |
| AMP | enabled | same |
| batch size | 8 | same |

**8.5 min is within Phase G.1's 8–9 min estimate.** Faster than Phase C.1's 9.6 min, as predicted
(2,058 vs 2,450 train rows = −16%, and no jitter source loading). No OOM, no retry needed.

---

## 7. Training artifacts produced

| path | what |
|---|---|
| `checkpoints/ct_lidc_nodulecrop_v4_stack3.pt` | **NEW** — best val balanced accuracy (epoch 13) |
| `eval_run/phaseG1/ct_lidc_nodulecrop_v4_stack3_valacc.pt` | best val accuracy (epoch 6) |
| `eval_run/phaseG1/training_history_v4_stack3.json` | full per-epoch history, args, class weights, peak VRAM |
| `eval_run/phaseG1/training_curves_v4_stack3.png` | loss / accuracy / balanced accuracy curves |

---

## 8. Files NOT touched — confirmed

All five protected files re-hashed after training. sha256, first 16 hex characters:

```
checkpoints/ct_best_model_phase35_v2.pt                sha=93b3a979884edbc4  MATCHES Phase G.1 §7
checkpoints/ct_lidc_nodulecrop_v1.pt                   sha=bdcbd6b2e5312435  MATCHES
checkpoints/ct_lidc_nodulecrop_v2_jitter.pt            sha=5d60055e02974e91  MATCHES
checkpoints/ct_lidc_nodulecrop_v3_jitter_loosebenign.pt sha=9d1e9d7bf1ff5e16  MATCHES
backend/main.py                                        sha=780120553178c0a5  MATCHES
```

**ALL MATCH Phase G.1 §7's recorded digests exactly.** None was modified.

Additionally confirmed untouched:
- `backend/seg_utils.py` — not opened or referenced
- All files under `eval_run/phaseG1/` that existed before training — the training script writes
  only `training_history_v4_stack3.json`, `training_curves_v4_stack3.png`, and
  `ct_lidc_nodulecrop_v4_stack3_valacc.pt` into that directory; the stacks, manifest, and all
  Phase G.1 scripts are unchanged
- `eval_run/lidc_full/labels_manifest.csv` — not opened by the training script

---

## 9. Deviations from Phase G.1's specification

**None.** The command, parameters, working directory, and expected outputs all matched Phase G.1 §6
exactly. No batch size reduction was needed. No OOM occurred. Training completed in 8.5 min,
within the 8–9 min estimate.

---

## 10. What this phase does NOT show

- **No tolerance sweep was run.** Whether v4_stack3 is more robust to through-plane displacement
  than v2_jitter is unknown — that is Phase G.3's primary question.
- **No end-to-end pipeline accuracy exists for this checkpoint.** The 0.8943/0.8637 numbers are
  val-set accuracy on annotation-centred crops, not pipeline performance.
- **No Grad-CAM analysis was performed.** Whether the model attends to through-plane features is
  unmeasured.
- **The rising val_loss after epoch 6 (0.399 → 0.522) while val_acc stays flat suggests early
  overfitting** — the model is becoming more confident on its errors. Phase G.3 should note whether
  this matters on the test set.
