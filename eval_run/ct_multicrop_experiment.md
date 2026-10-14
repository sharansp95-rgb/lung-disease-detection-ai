# Sliding-window multi-crop inference — no retraining

Bounded, single-session experiment. Live checkpoint (`ct_best_model_phase35_v2.pt`),
`backend/main.py`, and `data/ct_scan/` all untouched — standalone script only:
`eval_run/ct_multicrop_inference.py`. Raw per-image results:
`eval_run/results_ct_multicrop.json`; summary: `eval_run/results_ct_multicrop_summary.json`.

## Method

For each image: run the whole-image baseline (identical to every prior measurement in this
investigation), plus a 3×3 grid of overlapping sub-crops (each 50% of image width/height,
stepped so the 9 crops tile the image edge-to-edge with 50% overlap between neighbours — a
standard sliding-window layout). Each crop goes through the unmodified classifier
independently. Three aggregation strategies compared against the whole-image baseline:

- **maxpool** — per class, take the max probability that class received across any crop, then
  argmax over those maxes (a standard multiple-instance-learning pooling rule).
- **vote** — majority vote over each crop's own top-1 prediction, ties broken by mean
  crop probability.
- **avg** — mean probability across all 9 crops, then argmax.

## Results

| Source | Class | n | whole (baseline) | maxpool | vote | avg |
|---|---|---|---|---|---|---|
| external | Malignant | 12 | 12/12 | 12/12 | 12/12 | 12/12 |
| external | **Normal** | 9 | **0/9** | **0/9** | **0/9** | **0/9** |
| in-dataset | Malignant | 50 | 50/50 | 42/50 | 41/50 | 41/50 |
| in-dataset | Normal | 41 | 41/41 | 26/41 | 20/41 | 22/41 |
| in-dataset | Benign | 12 | 10/12 | 8/12 | 11/12 | 10/12 |

## Verdict: no-go, clean negative

**None of the three aggregation strategies improved external Normal detection at all —
still 0/9 across the board.** Meanwhile every strategy **regressed in-dataset accuracy**,
substantially so for Malignant (50/50 → 41-42/50) and Normal (41/41 → 20-26/41). This is a
straightforward net loss on every axis measured: the one thing this was meant to fix (0/9
external Normal) didn't move, and the one thing that had to be protected (in-dataset
accuracy, external Malignant) got worse on two of three classes.

**Why this doesn't work, given what's already known:** the shortcut the model relies on
(per `ct_shortcut_inversion_analysis.md` and `ct_framing_normalization_experiment.md`) is a
*global* style/framing signal — background composition, scanner-table visibility, overall
contrast/HU-floor characteristic of the source. A 50%-width sub-crop still contains a large
share of that same global signal (corner crops still see background/border; center crops
still see the same mediastinum region the model already over-attends to). Sliding-window
cropping changes *what portion of the image* the model looks at, but does nothing to remove
the style cues that portion still carries — unlike the framing-normalization experiment,
which explicitly masked out background/table pixels. External Malignant holding at 12/12
across every strategy is consistent with this: the same "call it Malignant" shortcut fires
on most crops of most external images regardless of which crop, so pooling doesn't change
the outcome. And forcing that same shortcut to fire crop-by-crop instead of once per image
is what erodes in-dataset accuracy — more chances per image for some crop to trip the
"external-style-ish" shortcut on Normal images the model previously got right at full-image
scale.

**Step 4 (Grad-CAM sanity check) not run** — its trigger condition ("any aggregation
approach meaningfully improves external Normal") was not met by any strategy, so there is no
improved strategy to sanity-check.

## Standing recommendation unchanged

Nothing here changes what should be deployed: keep `ct_best_model_phase35_v2.pt` live as-is.
Multi-crop pooling over the existing model is not a viable path to better external Normal
detection without addressing the underlying global-style shortcut first (which whole-image
crop-pooling structurally cannot do).
