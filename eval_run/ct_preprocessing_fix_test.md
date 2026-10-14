# Preprocessing-fix experiment: does it recover external CT Normal misclassifications?

Standalone script, does not touch `backend/`, checkpoints, or the live pipeline:
`eval_run/ct_preprocessing_fix_test.py`. Loads `checkpoints/ct_best_model_phase35_v2.pt`
directly and runs the classifier head-to-head under two preprocessing pipelines on the
same 9 external Normal + 12 external Malignant images used in the root-cause investigation.

**Baseline** = exactly `ml/dataset.py::build_transforms(train=False)` (`Grayscale(3ch) →
Resize((224,224)) → ToTensor → ImageNet-normalize`) — same as the live API. Baseline
predictions here reproduce `results_external.json` exactly (e.g. `normal_10` 0.9306
Malignant, `normal_22` 0.984 Malignant), confirming the harness is faithful.

**Fixed** = percentile contrast normalization (clip to 1st/99th percentile, rescale to
0–255, replacing reliance on IQ-OTH/NCCD's fixed `min=37` floor) + aspect-preserving
resize with zero-padding to a square (instead of squashing) before the same 224×224
tensor pipeline.

## Result: the fix does not help, and mildly hurts

| Group | Baseline correct | Fixed correct |
|---|---|---|
| External Normal (n=9) | 0/9 | **1/9** |
| External Malignant (n=12) | 12/12 | 12/12 |

Malignant holds perfectly under the new preprocessing — no regression there. But only
**one** of the nine Normal misclassifications flips to correct (`normal_11`: Benign 0.56 →
**Normal 0.60**). The other eight remain wrong, and on most of them **confidence in the
wrong class increased**, not decreased:

| File | Baseline (wrong) | Fixed (wrong) |
|---|---|---|
| normal_10 | Malignant 0.93 | Malignant **0.98** |
| normal_21 | Malignant 0.54 | Malignant **0.85** |
| normal_22 | Malignant 0.98 | Malignant **1.00** |
| normal_23 | Malignant 0.69 | Malignant **0.82** |
| normal_24 | Benign 0.70 | Malignant 0.62 (still wrong, different wrong class) |
| normal_6 | Malignant 0.53 | Malignant **0.88** |
| normal_7 | Malignant 0.42 | Malignant **0.82** |
| normal_8 | Malignant 0.47 | Malignant **0.74** |

Six of the eight remaining failures got *more* confidently wrong after the fix, some
sharply so (`normal_6`: 0.53→0.88, `normal_7`: 0.42→0.82). The percentile contrast
stretch appears to amplify whatever central-mediastinum structure the model was already
keying on (per the Grad-CAM findings in `ct_normal_root_cause.md`) rather than neutralize
it — increasing local contrast makes those structures more salient, not less.

## Interpretation

This is consistent with, and reinforces, the root-cause conclusion: the failure isn't a
raw pixel-intensity/windowing mismatch that contrast normalization can paper over. It's
that the model has no learned concept of "Normal" at the slice levels/framings present in
these external images. Sharpening the contrast on an image the model was never trained to
recognize as Normal just makes it a more confident wrong answer, not a right one.

**Net assessment: not a net win, and not worth integrating as-is.** One recovered case
out of nine, against a general increase in confident-wrongness on the rest, plus the
added complexity and runtime cost of percentile computation per image. Malignant not
regressing is the only clean positive, but that alone doesn't justify the change.

## No fix has been integrated

`backend/main.py`, `ml/dataset.py`, and the checkpoint are all untouched. This was a
read-only experiment against a copy of the preprocessing logic in
`eval_run/ct_preprocessing_fix_test.py`. Per the root-cause writeup, the path forward
most likely to actually help is retraining with a more anatomically diverse Normal class
(multiple slice levels/scanners) rather than any further preprocessing-only tweak —
happy to try a different preprocessing variant first if you'd rather rule that out more
thoroughly before concluding retraining is needed, but awaiting direction before doing
either.
