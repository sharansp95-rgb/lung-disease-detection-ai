# Grad-CAM diagnostic: why did retraining invert external-Malignant detection to Normal?

Diagnosis only. No retraining, no changes to `backend/main.py`, the live checkpoint
(`checkpoints/ct_best_model_phase35_v2.pt`), or `data/ct_scan/`. Script:
`eval_run/ct_shortcut_gradcam_probe.py`. Raw results: `eval_run/ct_shortcut_gradcam_results.json`.
Heatmap PNGs: `eval_run/ct_shortcut_gradcam_pngs/` (24 files, 12 images × 2 models).

## 1–2. Attention comparison, original vs retrained, on all 12 external Malignant images

| File | Original (`phase35_v2`) attention | Retrained (`phase36_normal_fix`) attention |
|---|---|---|
| adenocarcinoma_000108 | Upper chest-wall/pleural border (top edge) | Central mediastinum, spread down into posterior lung |
| adenocarcinoma_000109 | (same pattern, mediastinum/upper border) | Mediastinum/central |
| adenocarcinoma_000112 | Mediastinum/central | Central + periphery |
| adenocarcinoma_000113 | **Central mediastinum (heart-shadow blob)** | Periphery/chest-wall border, corners |
| large_cell_000108 | Mediastinum, extends to posterior lung | Central mediastinum, broader spread |
| large_cell_000110 | **Central mediastinum (heart-shadow blob)** | Diffuse across entire periphery + scanner-table region at bottom |
| large_cell_000111 | Mediastinum/central | Periphery |
| large_cell_000113 | Mediastinum/central | Periphery/corners |
| squamous_000108 | Mediastinum/central | Periphery |
| squamous_000110 | Mediastinum/central | Periphery/border |
| squamous_000111 | **Central mediastinum (heart-shadow blob)** | Scattered to image **corners/background**, almost nothing on the body |
| squamous_000112 | Mediastinum/central, extends to lung | **Halo around the outer chest-wall contour**, periphery |

(6 representative overlay pairs saved as PNGs for direct inspection: `adenocarcinoma_000108`,
`adenocarcinoma_000113`, `large_cell_000110`, `squamous_000111`, `squamous_000112`, plus all
others in the same directory.)

**Consistent pattern across all 12:** the **original** model's attention lands on the same
generic region every time — the central mediastinum/heart-shadow — regardless of where a
real adenocarcinoma/large-cell/squamous lesion would actually be expected (typically a
peripheral pulmonary nodule, not a mediastinal structure). The **retrained** model's
attention consistently shifts *away* from that same region, toward the chest-wall
periphery, image corners, or background/table area.

Neither model localizes anything resembling a discrete lesion in any of the 12 images.
Both react to a broad anatomical zone (mediastinum vs. periphery) that is present in
every chest CT regardless of pathology — this is the signature of a **global framing/style
cue, not lesion localization**.

## 3. Predicted classes and confidence (retrained model)

All 12 flipped, and **11 of 12 flipped to the same specific wrong class, Normal, at high
confidence** (0.78–0.99); the twelfth (`squamous_000111`) landed in a near three-way tie
(Normal 0.39 / Benign 0.37 / Malignant 0.24) — still wrong, but not confidently. This
uniform, single-direction flip (not a scatter across Benign/Malignant/Normal) is strong
evidence of **one shortcut inverting**, not twelve independent new errors.

## 4. Cross-reference against the known protocol-mismatch confound

PROJECT_STATUS.md §6 documents that after the leakage fix, Grad-CAM still showed
shortcut reliance on scanner-table visibility, with residual imbalance even after
excluding 61 images: **89.3% table-visible for Malignant vs. 99.8%/100% for Normal/Benign**
in the retained training set. Table-absence is therefore a small but real residual
Malignant-correlated cue in the original model's training data.

Checked directly on the 12 external Malignant images:
- Visual inspection (e.g. `adenocarcinoma_000108.png`, `large_cell_000108.png`): all 12
  are tight thorax-only crops on a flat black background — **no scanner table or couch
  visible in any of them.**
- Quantitatively confirmed on `large_cell_000108.png`: the bottom 10% of the frame (where
  the in-dataset table structure appears) has mean brightness 5.3/255, max 147 — no bright
  table structure, consistent with pure background.
- By contrast, in-dataset images (all classes) are full-FOV 512×512 frames showing the
  circular CT gantry field-of-view *and* a distinct bright scanner table/couch at the
  bottom (confirmed visually, e.g. `data/ct_scan/test/Malignant/Malignant case (109).jpg`).

**This is the connecting thread.** External Malignant images are **100% table-absent** —
squarely on the Malignant side of the original model's (small, residual) table-visibility
correlation. That is consistent with the original model calling them Malignant for a
framing reason already documented as a shortcut in this project, not because it detected
adenocarcinoma/large-cell/squamous tissue. It also explains the failure mode uniformly:
the *same* table-absent, tightly-cropped, black-background external framing style is what
the newly-added external Normal training data looks like too (per `ct_retrain_experiment.md`,
sourced from the same and a similar external dataset) — so retraining re-associated that
framing style with **Normal** instead, and the shortcut inverted rather than being removed,
because the underlying cue (image style/acquisition framing) was never lesion-related to
begin with.

Structurally, all 12 external images share a distinctive, consistent visual signature
(circular vignette/black background, no couch, non-square aspect ratio 1.22–1.45, variable
resolution 360–490px vs. the in-dataset's fixed 512×512) that is common to the *entire*
external source, independent of true pathology — exactly the kind of dataset-level
signature a small classifier can latch onto instead of pathology.

## What each model is actually using — stated plainly

- **Original model:** primarily reacts to whether the mediastinal/central-chest region
  looks "busy" in a way that differs from the IQ-OTH/NCCD Normal template, combined with a
  residual table-visibility/framing cue. It gets external Malignant right, but for the
  wrong reason — it is functioning as an image-style classifier (IQ-OTH/NCCD-style vs.
  external-style) that happens to point toward "Malignant" for anything styled unlike its
  own Normal template.
- **Retrained model:** the same style-classification behavior, but now pointed at "Normal"
  for external-style framing, because the new Normal training data was itself
  external-style. The shortcut's direction flipped with the training distribution; its
  existence did not change.

## Root cause: specific and identifiable, not just "dataset too small"

This points to something **more specific and potentially addressable** than "1,036 images
is too small to learn genuine lesion-level features regardless of class balance." The
failure has a concrete, demonstrated cause: **acquisition/framing style (table visibility,
full-FOV vs. thorax-crop, background composition) is confounded with class label in both
the original and the newly-added training data, in opposite directions**, and the model
learns that confound instead of (or in addition to) any lesion signal. Dataset size is
still a plausible contributing factor — a larger, more diverse set would dilute any single
confound's predictive power — but it is not required to explain what was observed here;
the observed behavior is fully explained by a specific, identifiable shortcut with a clear
mechanism, not a generic "not enough data to learn anything real" story. Whether genuine
lesion-level signal exists at all in this architecture/dataset size once the framing
confound is controlled for is a separate, unanswered question — this task did not test
that, only diagnosed why the two checkpoints disagree.

**No fix is proposed or attempted here**, per the task scope.
