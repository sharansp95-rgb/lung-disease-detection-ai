# Demo Rehearsal Report (macOS, Apple Silicon)

Rehearsal run: Tue 2026-10-13, 19:55–21:50, for Review 2 on Thursday 15 Oct.
**Status: ✅ COMPLETE.** All three terminal demos run on this Mac and reproduce the expected numbers.

## TL;DR
- **All 12 demo commands work on the Mac.** Terminal 1 and Terminal 2 reproduce the 2026-10-04 numbers exactly (X-ray **0.9657**, CT **0.9806**).
- **The Terminal 3 "KNOWN ISSUE" is already fixed.** I verified it by running the CLI, not just by reading the code. `predict_cli.py` now uses backend/main.py's own loader and pipeline, so both X-ray runs pass the OOD check (distance 3.0 and 4.6, threshold 8.7) and predict correctly. **I made no code changes.**
- **The new risk is time.** On this Mac (CPU only) every `predict_cli.py` run spends **9–13 min** re-verifying all models before it predicts. Plan the demo around that (see "Fallback plan").
- **Backend tests: 59 / 61 pass.** The 2 failures (`test_health.py`) come from the tests running before any model is loaded, now that models load lazily. They are not caused by this Mac (details below).

## Thursday checklist (5 lines)
1. Plug the Mac in. In each terminal: `cd ~/Documents/sdp && source demo/run_demo.sh <1|2|3>` (activates `backend/.venv` and prints the commands).
2. **~15 min before your slot:** start Terminal 3 run 1 with `caffeinate -ims python predict_cli.py ...` so the 9-min model verification is finished (or nearly) when you present. Start Terminal 2's X-ray script too, so the accuracy line is already on screen.
3. Open the 4 eval PNGs listed in `demo/terminal2_commands.txt` and the heatmaps `demo/heatmap_*.png` in Preview before the slot.
4. Run Terminal 1 (2 min) and the CT eval (12 s) live. They're fast and safe.
5. For Terminal 3 runs 2–6, show `demo/rehearsal_outputs/t3_*.txt` (real output from today) or use the web app. Don't start a fresh 9–13 min run per image in front of the panel.

## Estimated vs actual time
| Step | Estimate | Actual | Status |
|---|---|---|---|
| 0. Inspect env + write estimate | – | ~4 min | ✅ |
| 1. Environment | 3 min | ~3 min (reused existing venv) | ✅ |
| 2. Backup | 1 min | <1 min | ✅ |
| 3. Terminal 1 | 5 min | ~2.5 min (1 min 47 s + 20 s) | ✅ |
| 4. Terminal 2 | 15 min | ~8.5 min (7 min 50 s + 12 s) | ✅ |
| 5. Terminal 3 (6 CLI runs) | 45 min | **~69 min** (9–15 min per run, incl. ~6 min the Mac slept) | ✅ |
| 6. pytest | 10 min | ~22 min (clean run 10 min 39 s, plus a false start: I accidentally ran two pytest sessions at once, killed both and reran once) | ⚠️ 59/61 |
| 7. Update demo/*.txt | 10 min | ~10 min (done while runs were going) | ✅ |
| 8. run_demo.sh + report | 10 min | ~15 min | ✅ |
| **Total** | **~1 h 40 min** | **~1 h 55 min** | |

Main miss: I estimated 4–6 min of CLI startup per run, and it was 9–13 min. That startup also explains the slow pytest run.

## Every demo command
Times are wall-clock on the Mac (Apple M2, CPU).

| # | Terminal | Command | Result | Time |
|---|---|---|---|---|
| 1 | T1 | `python ct_leakage_audit.py` | ✅ 0 / 111 clusters split, 287 pairs (identical to expected) | 1 min 47 s |
| 2 | T1 | `python tb_artifact_detect.py` | ✅ Normal 13/3500, TB 372/700 (identical) | 20 s |
| 3 | T2 | `python test_new_xray_checkpoint.py` | ✅ **TEST accuracy 0.9657** (1922 imgs), per-class identical | 7 min 50 s (accuracy at ~1.5 min) |
| 4 | T2 | `python evaluate.py ... ct_phase35_v2` | ✅ **TEST accuracy 0.9806** (103 imgs), per-class identical | 12 s |
| 5 | T3 | run 1: normal X-ray `img_189af4.png` | ✅ **Normal 99.8%** (OOD 3.0 / 8.7), heatmap saved | 8 min 54 s |
| 6 | T3 | run 2: pneumonia X-ray `img_0058a2.png` | ✅ **Pneumonia 99.9%** (OOD 4.6 / 8.7), heatmap saved | 9 min 0 s |
| 7 | T3 | run 3: benign CT `img_159819.jpg` as ct | ✅ **Benign 90.3%** (OOD 18.3 / 25.6), same as expected | 15 min 20 s (Mac slept ~6 min; normal ≈ 9 min) |
| 8 | T3 | run 4: same CT slice as xray | ✅ **REJECTED** at sanity gate ("does not look like a chest X-ray"), as intended; exit code 1 is expected | 9 min 29 s |
| 9 | T3 | extra: COVID X-ray `img_09dafc.png` | ✅ **COVID-19 58.6%** (Normal 41.2%; OOD 3.9 / 8.7), heatmap saved; correct but borderline | 12 min 12 s |
| 10 | T3 | extra: malignant CT `img_07ec48.jpg` | ✅ **Malignant 98.1%** (OOD 15.6 / 25.6), heatmap saved | 13 min 6 s |
| 11 | – | `pytest` in backend/ | ⚠️ **59 passed, 2 failed** (61 collected) | 10 min 39 s |
| 12 | – | `source demo/run_demo.sh [1/2/3]` | ✅ tested in zsh and bash; prints commands only | <1 s |

Runs 9–10 were slower than runs 5–6 (~12–13 min vs ~9). The likely reason is thermal throttling after ~1 h of continuous CPU load. Plan on 9–13 min.

## Key numbers
- X-ray (served `xray_txrv_densenet121_best.pt`) test accuracy **0.9657**. Startup reproduction check: val 0.9646 = claimed.
- CT (served `ct_best_model_phase35_v2.pt`) test accuracy **0.9806**. Val 0.9904 = claimed.
- Gate 1.0000, X-ray sanity 0.9996, CT sanity 1.0000, seg dice 0.9830 / IoU 0.9671 (claimed 0.9890 / 0.9786, within tolerance). All reproduction checks PASSED in all six CLI runs.
- CT leakage: **0 / 111** clusters split. TB artifacts: **372 / 700** TB vs **13 / 3500** Normal.
- pytest: **59 / 61 pass**.

## The KNOWN ISSUE: verified, not assumed
1. **Code:** `scripts/predict_cli.py` is identical to the copy I backed up before starting (`cmp` confirms). It has no checkpoint table of its own. It calls `backend._ensure_models_loaded()`, `_read_and_validate_image`, `_sanity_predict`, `_gate_predict` (incl. the 90% confidence floor), `compute_lung_crop` + `preprocess_image_xrv`, `OODDetector.check` and `predict_with_heatmap_xrv`, in the same order as `backend/main.py::_run_prediction`. The only CLI-specific code is the printing.
2. **Runtime:** the log shows `[xray] checkpoint=xray_txrv_densenet121_best.pt backbone=densenet121-res224-all`.
3. **OOD stats:** `backend/ood_stats/xray_ood.pt` (Sep 6 14:21) was computed after the txrv checkpoint (Sep 6 13:12). It is 1024-d, with threshold 8.72 over 15,373 train images. Genuine X-rays now score 3.0–4.6, versus 19–28 when the CLI used the old model.
4. **Result:** all 3 X-ray runs pass and predict the correct class. No change to predict_cli.py was needed, so per your rules I made none. Backend thresholds, model files, OOD stats and the web app are untouched.

## Files I changed or created
| File | What / why |
|---|---|
| `docs/reports/DEMO_REHEARSAL_REPORT.md` | **new**: this report |
| `demo/terminal1_commands.txt` | macOS activation line; "re-verified on Mac 2026-10-13"; measured times (1 min 47 s, 20 s). Outputs unchanged because they matched exactly |
| `demo/terminal2_commands.txt` | macOS activation; measured times (7 min 50 s / 12 s); note on the stale "active checkpoint xray_best_model_phase35_v2.pt" line the script prints; tip that you can Ctrl+C during the occlusion test |
| `demo/terminal3_commands.txt` | **KNOWN ISSUE replaced** by a "fixed + verified" status and a warning about the 9–13 min startup; "BROKEN" labels removed; added the 2 extra runs; EXPECTED OUTPUT replaced with today's real output (new modality-gate confidence line, no lung-crop line on the rejected run, `[report (groq)]` instead of the GROQ warning); fixed the run-4 talking point (the web app runs the same sanity gate first) and the closing "do NOT claim same models" note |
| `demo/run_demo.sh` | **new**: `source demo/run_demo.sh [1|2|3]` activates `backend/.venv` and prints the exact commands. Runs nothing heavy |
| `demo/heatmap_normal.png`, `heatmap_pneumonia.png`, `heatmap_covid.png`, `heatmap_malignant_ct.png` | **new**: Grad-CAM overlays from the CLI runs |
| `demo/rehearsal_outputs/*.txt` (10 files) | **new**: full captured output of every demo command (fallback for the live demo). Checked: no secrets in them |
| `backup_before_rehearsal/` | **new**: copies of `audit/`, `eval_outputs/`, original `predict_cli.py`, original three `terminal*_commands.txt` |
| `audit/*.json`, `eval_outputs/xray_txrv_densenet121/*`, `eval_outputs/ct_scan_phase35_v2/*` | rewritten by the demo scripts themselves. `diff -rq` against the backup: **byte-identical**, except `xray_txrv_densenet121_test_metrics.json`, where only the stored checkpoint path changed (the old one contained a stray space, `.../sdp /checkpoints/...`); all metrics identical |
| `backend/tests/_test.db` | recreated by the test suite's own `conftest.py` (throwaway test DB, not `app.db`) |

**Not changed:** `scripts/predict_cli.py`, anything in `backend/` (code, thresholds, `.env`, `ood_stats/`), `checkpoints/`, `data/`, `raw/`. No git commits. **Nothing installed** (no pip/brew). I used the built-in macOS `caffeinate` to stop the Mac sleeping during long runs.

## Still broken / caveats
1. **CLI startup takes 9–13 min per run on this Mac (CPU).** This isn't a bug: the CLI re-runs the backend's reproduction checks (val sets of ~7,000 images plus the seg set) on every invocation, and the code only uses CUDA or CPU, never Apple MPS. I did not change this, because it would mean editing backend code or changing how predict_cli.py matches the web app. Post-review options (not done): let the device picker use `mps`, or give predict_cli.py a mode that takes several `--image` arguments in one process so the checks run once.
2. **2 failing backend tests (`tests/test_health.py::test_root`, `::test_health`).** main.py now lazy-loads models on the first prediction request ("models will be loaded on first request"), but the tests (and `conftest.py`'s comment) still expect the models loaded at startup. pytest runs `test_health.py` before `test_predict.py`, so `/` reports `"degraded"` and `/health` reports `models_loaded: []`. This is a test/code mismatch, not a Mac problem. I didn't fix it (backend change). If asked: "59 of 61 pass; the 2 health tests assume eager model loading, which we switched to lazy loading for faster server start." Untested idea for you: running `pytest tests/test_predict.py tests/test_health.py` (predict first) should let them pass.
3. **The COVID example (`img_09dafc.png`) is only 58.6% confident** (Normal 41.2%). It is correct, and above the 50% low-confidence warning line, but don't use it as the "confident" showcase.
4. **Report text needs internet.** The CLI loads `backend/.env`, so it calls Groq (`[report (groq)]`). Offline it falls back to the template; the prediction is unaffected.
5. ~~Cosmetic: `test_new_xray_checkpoint.py` prints a stale line calling `xray_best_model_phase35_v2.pt` the "active checkpoint".~~ **Fixed later; see "Small fixes" at the end.**
6. ~~`requirements.txt` doesn't list `torchxrayvision`.~~ **Fixed later; see "Small fixes" at the end.**

## Fallback plan for the live demo
- **Terminal 1 and the CT half of Terminal 2:** run live (2 min and 12 s), low risk.
- **X-ray eval (Terminal 2):** start it before your slot. The accuracy and confusion-matrix output appears after ~1.5 min; you can Ctrl+C during the occlusion test. Fallback: `demo/rehearsal_outputs/t2_xray_test.txt` plus the PNG.
- **Terminal 3:** start run 1 ~15 min ahead with `caffeinate -ims`, and narrate the startup verification lines. Show runs 2–6 from `demo/rehearsal_outputs/t3_run*.txt` and the heatmaps, or do them live in the web app. The web app loads models once on its first request, so warm it up with one upload before your slot.
- **If the venv ever breaks:** `python3 -m venv .venv-mac && .venv-mac/bin/pip install -r requirements.txt torchxrayvision==1.5.4`. This needs internet and is untested here. The torchxrayvision weights are already cached in `~/.torchxrayvision`.

## Appendix A. Environment inspection and original estimate (written before any heavy step)

**What I found (before running anything heavy):**

| Item | Finding |
|---|---|
| Machine | Apple M2, 8 cores, 16 GB RAM, macOS (Darwin 27.0, arm64) |
| System Python | 3.12.10 (python.org framework build) |
| `backend/.venv` | **Not a Windows venv.** It is a macOS arm64 venv created by `uv` 0.12.19 on Sep 27, Python 3.12.14 (`bin/activate` exists, no `Scripts/`). Interpreter starts fine. |
| Packages in `backend/.venv` | Every pin in `requirements.txt` imports at the pinned version (torch 2.11.0, torchvision 0.26.0, numpy 2.5.2, fastapi 0.141.1, pytest 9.1.1, opencv 5.0.0, ImageHash 4.3.2, ...). It also has `torchxrayvision` 1.5.4, which the served X-ray model needs but which is **not in requirements.txt**. `monai` is absent (only used by a training script, not by the demos). |
| torch device | `torch.backends.mps.is_available() == True`, CUDA not available. The project code picks `cuda if available else cpu` (backend/models.py, seg_utils.py, test_new_xray_checkpoint.py), so **all demos run on CPU**. I did not change this (changing it would alter backend code). |
| Data sizes | X-ray val/test 1922/1922, CT val/test 104/103, modality gate val 2026, X-ray sanity val 2596, CT sanity val 778 |

**Estimates (made before running; CPU-only, M2):**

| Step | Estimate |
|---|---|
| 1. Environment (reuse existing venv, verify) | 3 min |
| 2. Back up audit/ and eval_outputs/ | 1 min |
| 3. Terminal 1: CT leakage audit (~2 min on Windows) + TB artifact scan (~20 s) | 5 min |
| 4. Terminal 2: X-ray test eval incl. occlusion test (1922 imgs x several passes, CPU) + CT eval | 15 min |
| 5. Terminal 3: 6 CLI runs; each run re-verifies ~7,000 val images at startup on CPU (~4-6 min each) + any debugging/fix of the OOD issue | 45 min |
| 6. Backend pytest suite (also loads all models) | 10 min |
| 7. Update the three demo command files | 10 min |
| 8. Write demo/run_demo.sh + final report | 10 min |
| **Total** | **~1 h 40 min** |

## Appendix B. Step-by-step progress log

### Step 1. Environment: ✅ (actual ~3 min, estimate 3 min)
- **Decision:** reuse `backend/.venv`. It is not a Windows venv and not broken: it is a uv-created macOS arm64 venv (Python 3.12.14) with every requirements.txt pin installed. No `.venv-mac` was needed and **nothing was installed** (no pip, no brew).
- `torchxrayvision` 1.5.4 is present in the venv and its DenseNet weights are already cached in `~/.torchxrayvision/models_data/`, so the X-ray model loads offline.
- Activation on macOS: `source backend/.venv/bin/activate`.
- MPS is available but the code only picks CUDA or CPU, so everything runs on **CPU**.

### Step 2. Backup: ✅ (actual <1 min)
- `audit/` and `eval_outputs/` copied to `backup_before_rehearsal/audit` and `backup_before_rehearsal/eval_outputs` (`diff -rq` showed them identical). I also put the original `scripts/predict_cli.py` (`predict_cli.py.orig`) and the three original `demo/terminal*_commands.txt` files there.
- Note: `audit_backup/`, `eval_outputs_backup/` and `scripts/predict_cli.py.bak` already existed before this run. I left them alone.

### Step 3. Terminal 1, data audits: ✅
| Command | Expected (Windows, 10-04) | Actual (Mac, today) | Time |
|---|---|---|---|
| `python ct_leakage_audit.py` | 1036 imgs, 17224 candidates, 287 pairs, 111 clusters, **0 / 111** split | identical: 1036 / 17224 / 287 / 111 / **0 / 111**, 0 / 1036 images | **1 min 47 s** (real 106.7 s) |
| `python tb_artifact_detect.py` | Normal 13/3500, TB 372/700 (370 tinted, 218 box, 72 large) | identical | **20 s** |

The scalar fields in the new `audit/ct_leakage_audit.json` match the backup exactly.

### Step 4. Terminal 2, test-set evaluation: ✅
| Command | Expected | Actual | Time |
|---|---|---|---|
| `python test_new_xray_checkpoint.py` | TEST accuracy 0.9657, 1922 images | **TEST accuracy: 0.9657**, per-class report identical (COVID 0.93/0.94, Normal 0.98/0.97, Pneumonia 0.91/0.94, TB 0.95/1.00) | **7 min 50 s** in total (the accuracy line appears after ~1.5 min; the rest is the occlusion test) |
| `python evaluate.py ... ct_phase35_v2` (CT) | TEST accuracy 0.9806, 103 images | **TEST accuracy: 0.9806**, identical per-class report, 2 misclassified (both Benign→Normal) | **12 s** |

Cosmetic: the X-ray script's 2nd line still says "active checkpoint xray_best_model_phase35_v2.pt and backend/main.py untouched". That text dates from before the txrv model was promoted to served model and is now stale. Harmless, but don't read it out. I did not edit the script.

### Step 5. Terminal 3, CLI inference: ✅ (actual ~69 min, estimate 45)
- KNOWN ISSUE verified fixed. No code change (see section above).
- Per-run startup breakdown, run 1: xray 165 s, ct 9 s, gate 42 s, xray_sanity 55 s, ct_sanity 17 s, seg 241 s → `all models loaded and verified in 531.4s`.
- Run 3 wall time was 920 s, but its internal timer read 549 s. macOS `perf_counter` doesn't count sleep, so the Mac slept ~6 min mid-run. After that I wrapped the remaining runs with `caffeinate -ims`.
- Results: see the command table above. Full logs: `demo/rehearsal_outputs/`.

### Step 6. Backend tests: ⚠️ 59 passed, 2 failed (10 min 39 s)
```
============= 2 failed, 59 passed, 2 warnings in 639.15s (0:10:39) =============
FAILED tests/test_health.py::test_root - AssertionError: assert 'degraded' == 'ok'
FAILED tests/test_health.py::test_health - AssertionError: assert set() == {'ct', 'ct_sanity', 'gate', 'xray', 'xray_sanity'}
```
Cause: lazy model loading vs. tests that expect eager loading (see "Still broken" #2). The first attempt accidentally ran two pytest sessions at the same time (shared `_test.db`). I killed both, and the numbers above come from one clean run.

### Step 7. Demo command files: ✅ (see "Files I changed")
### Step 8. demo/run_demo.sh: ✅ tested with `zsh -c 'source demo/run_demo.sh 1'` and `bash demo/run_demo.sh 2`. It activates the venv and only prints commands.

---

# Follow-up (2026-10-13 evening)

Status: ✅ COMPLETE (22:33–23:25). The checklist below replaces the original "Thursday checklist" at the top of this file.

## Estimate (written before starting)
Every model load on this Mac takes 9–13 min (CPU), and loads can't run in parallel without slowing each other down, so the heavy steps run one after another.

| Task | Estimate |
|---|---|
| 1. Fix test_health.py + full pytest run | 5 min edit + 11 min run |
| 2. Multi-image predict_cli.py (code + quick checks) | 15 min |
| 3. Find most confident COVID X-ray: one batch run over all 20 COVID images (also tests the new batch mode) | 14 min |
| 4. demo/warmup_backend.sh: write + run once (the first request loads all models) | 5 min + 13 min |
| 5. Update terminal3/run_demo.sh + one end-to-end multi-image run | 5 min + 14 min |
| Report + summary | 8 min |
| **Total** | **~1 h 30 min** |

## Progress

### 1. test_health.py: ✅ 62 / 62 pass (8 min 14 s)
```
================== 62 passed, 2 warnings in 494.75s (0:08:14) ==================
```
- `test_root` / `test_health` now check the lazy-load contract instead of assuming eager loading: before any prediction, `/` says `degraded` and `/health` says `models_loaded: []`; afterwards, `ok` and all 5 models. They read `main._models_loaded`, so they pass in any test order.
- New `test_first_prediction_lazy_loads_all_models`: sends a real X-ray to `/predict/xray`, then asserts all 5 models show as loaded and `/` says `ok`. That's why the count is **62, not 61**. All 61 original tests pass, plus this one.
- Fixed the stale comment in `tests/conftest.py`, which said startup loads the models. `backend/main.py` is untouched. Original test file: `backup_before_rehearsal/test_health.py.orig`.

### 2. predict_cli.py multi-image: ✅ code done (end-to-end run below)
- `--image/--modality` can be repeated (pairs, in order). `--save-heatmap` is optional, but if used, give one per image (`-` = none). Or use `--batch FILE` with `path,modality[,heatmap]` lines (relative paths resolve against the batch file's own folder).
- Models load and run their reproduction checks **once**. Each image then gets exactly the same printouts as before, followed by a `=== SUMMARY ===` table with per-image time.
- The pipeline code is unchanged: it was moved into `run_pipeline()`, and its `SystemExit` rejections became a `Rejected` exception. **Single-image mode behaves as before:** same output, and a rejection still prints `REJECTED: ...` and exits 1. In multi-image mode a rejection is a normal outcome, so the run continues (exit 0; exit 1 only if a file couldn't be processed).
- New: missing image or batch files are reported **before** the 9-min model load, instead of crashing afterwards.
- Argument errors tested: wrong pair counts, `--batch` combined with `--image`, a bad modality in the batch file, missing files. Each gives a clear message and exits without loading models.
- Previous version: `backup_before_rehearsal/predict_cli.py.before_batch`.

### 3. Most confident COVID X-ray: ✅ `img_19570d.png`, COVID-19 100.0%
One batch run over all 20 images in `test images/x ray/covid` (this was also the first real test of batch mode): models loaded once in **530 s**, then 20 images in ~29 s (0.9–1.9 s each). **All 20 were predicted COVID-19**; none were rejected.

| Rank | Image | COVID-19 conf. | OOD distance |
|---|---|---|---|
| 1 | **img_19570d.png** (chosen) | **100.0%** | **4.3** |
| 2 | img_ba4aa8.png | 100.0% | 4.8 |
| 3 | img_26a57f.png | 100.0% | 5.5 |
| 4 | img_7d0731.png | 100.0% | 6.5 |
| 5 | img_d3030f.png | 100.0% | 6.8 |
| … | img_09dafc.png (old demo image) | 58.6% (lowest of the 20) | 3.9 |

Five images round to 100.0%, so I broke the tie by **lowest OOD distance** (the most "typical" image). The old heatmap was saved as `backup_before_rehearsal/heatmap_covid_img_09dafc_58.6pct.png` before being replaced.

### 4. demo/warmup_backend.sh: ✅ (after one fix)
- It starts uvicorn on 127.0.0.1:8000 (same as `start_public.sh`), or reuses a backend already answering `/health`. It refuses to start if port 8000 is taken by something else. It runs `caffeinate` tied to the server PID so the Mac stays awake, POSTs a real X-ray and CT, prints the results and timings, and leaves the server running. The PID is written to `logs/warmup_backend.pid` and the server log to `logs/warmup_backend.log`.
- The first test exposed a bug in my own script: an inline Python one-liner with escaped quotes inside an f-string, which is a SyntaxError on Python 3.12. The prediction itself succeeded: the server log shows the cold first request took **9 min 7 s** (22:52:25 → 23:01:32, HTTP 200). I fixed it with a heredoc.
- Rerun on the already-warm server (reuse path): X-ray **Normal 99.8%** in 1 s, CT **Benign 90.3%** in 1 s, `/health` lists all 5 models. These are the same answers the CLI gives for the same images.
- Note: the backend logs a masked prefix and suffix of the Groq key at startup (existing behaviour in main.py), so `logs/warmup_backend.log` shouldn't be shared publicly.

### 4b. Warmup, clean cold run of the fixed script: ✅
```
[warmup] starting backend on http://127.0.0.1:8000 (log: logs/warmup_backend.log)
[warmup] POST /predict/xray  img_189af4.png  ...
[warmup]   -> Normal 99.8% (report: groq)
[warmup]   took 544 s
[warmup] POST /predict/ct  img_159819.jpg  ...
[warmup]   -> Benign 90.3% (report: groq)
[warmup]   took 0 s
[warmup] done in 545 s total -- models are loaded; the website will answer instantly.
[warmup] health: {"status":"ok","models_loaded":["xray","ct","gate","xray_sanity","ct_sanity"],...}
```
**Cold: ~9 min 5 s** (first request loads and verifies all models). **Warm: ~1 s per request.** After testing I stopped the server and its caffeinate process (see "Cleanup" below).

### 5. Command files + end-to-end multi-image run: ✅ 9 min 5 s
- New `demo/demo_batch.txt` holds the 6 demo images, in order: normal X-ray, pneumonia X-ray, **COVID img_19570d**, benign CT, malignant CT, and benign CT sent as X-ray. I moved the deliberate rejection to the end as a finale.
- `demo/terminal3_commands.txt`: the batch command is now the recommended one (plus the equivalent repeated-pairs form). The single-image commands are kept, with the new COVID image. The status block, expected outputs (batch summary + new run 5) and talking points are updated.
- `demo/run_demo.sh`: Terminal 3 now prints the batch command. There's a new `web` section (`source demo/run_demo.sh web`) for the warmup step. Tested in zsh and bash.
- End-to-end run of `caffeinate -ims python predict_cli.py --batch ../demo/demo_batch.txt`:

```
=== all models loaded and verified in 537.1s ===
=== SUMMARY (6 images, models loaded once) ===
  img_189af4.png         xray  Normal 99.8%                   1.0s
  img_0058a2.png         xray  Pneumonia 99.9%                1.3s
  img_19570d.png         xray  COVID-19 100.0%                1.0s
  img_159819.jpg         ct    Benign 90.3%                   1.0s
  img_07ec48.jpg         ct    Malignant 98.1%                1.1s
  img_159819.jpg         xray  REJECTED                       0.0s
real 544.54   (exit 0)
```
Per-image results are identical to the six separate runs from earlier. Total demo time drops from ~55–70 min (six startups) to **~9 min (one startup)**. Full logs: `demo/rehearsal_outputs/t3_batch_all6_demo.txt` and `t3_batch_all20_covid.txt`. The old COVID run is kept as `t3_run5_covid_xray_OLD_img_09dafc.txt`.

## Follow-up: files changed
| File | Change |
|---|---|
| `backend/tests/test_health.py` | Tests check the lazy-load contract; added `test_first_prediction_lazy_loads_all_models` |
| `backend/tests/conftest.py` | Comment only: models lazy-load on first prediction |
| `scripts/predict_cli.py` | Multi-image support (`--batch`, repeated pairs), summary table, missing files checked before the model load. Pipeline logic unchanged |
| `demo/demo_batch.txt` | **new**: the 6-image demo list |
| `demo/warmup_backend.sh` | **new**: start the backend + 1 X-ray + 1 CT, left running |
| `demo/terminal3_commands.txt`, `demo/run_demo.sh` | batch command, new COVID image, warmup step |
| `demo/heatmap_covid.png` | now from img_19570d (old one in `backup_before_rehearsal/`) |
| `demo/rehearsal_outputs/` | +2 batch logs; old COVID log renamed `..._OLD_img_09dafc.txt` |
| `backup_before_rehearsal/` | + `test_health.py.orig`, `predict_cli.py.before_batch`, `run_demo.sh.v1`, `terminal3_commands.txt.v1`, old COVID heatmap |

**Not touched:** `backend/main.py` or any other backend code, thresholds, `checkpoints/`, `data/`, `raw/`, `backend/ood_stats/`, `.env`. No retraining, no installs, no git commits.

## Follow-up: estimate vs actual
| Task | Estimate | Actual |
|---|---|---|
| 1. Tests + full run | 16 min | ~10 min (run 8 min 14 s) |
| 2. Multi-image CLI | 15 min | ~12 min (written while pytest ran) |
| 3. COVID search (20-image batch) | 14 min | 9 min 19 s |
| 4. Warmup script + run | 18 min | ~30 min (cold run 9 min 7 s, failed on my own one-liner bug, fixed, reuse test, clean cold rerun 9 min 8 s) |
| 5. Docs + end-to-end batch | 19 min | ~15 min (batch 9 min 5 s) |
| Report | 8 min | ~8 min, plus a pause while I waited for your go-ahead |
| **Total** | **~1 h 30 min** | **~1 h 25 min of work** (22:33–23:25 wall clock, including the pause) |

## Closing summary
- **pytest: 62 / 62 pass.** The health tests now match the lazy-loading backend (`main.py` unchanged).
- **One CLI run now covers the whole Terminal 3 demo:** 6 images in **9 min 5 s** total (models verified once, ~1 s per image), with results identical to the single runs. Single-image mode is unchanged.
- **COVID demo image is now `img_19570d.png` at 100.0%.** All 20 COVID test images are classified correctly.
- **`demo/warmup_backend.sh` works:** the first request takes ~9 min cold, then ~1 s per request, and the web answers match the CLI.
- Remaining limitation: everything still runs on CPU (no MPS), so any cold start is ~9 min. Plan around it as below.

## Updated Thursday checklist (supersedes the one at the top)
1. **~20 min before your slot**, plugged in: `bash demo/warmup_backend.sh` (website backend, ~9 min), then start the frontend as usual.
2. **In parallel, Terminal 3:** `source demo/run_demo.sh 3`, then `cd scripts && caffeinate -ims python predict_cli.py --batch ../demo/demo_batch.txt`. Talk through the verification lines; the 6 results and the SUMMARY appear about 9 min in. (Running it alongside the warmup will slow both, so start ~25 min ahead if you do both.)
3. **Terminal 2:** start `python test_new_xray_checkpoint.py` early (accuracy line at ~1.5 min); run the CT eval live (12 s). Open the eval PNGs and `demo/heatmap_*.png` in Preview beforehand.
4. **Terminal 1:** run both audits live (~2 min).
5. **Fallback:** `demo/rehearsal_outputs/*.txt` hold real outputs for every command. If asked about tests: "62 of 62 pass." Afterwards stop the backend: `kill "$(cat logs/warmup_backend.pid)"`.

## Cleanup
At 23:24 I stopped the warmup backend (pid 19856) and its caffeinate process. Verified with `lsof -Pi :8000 -sTCP:LISTEN`: nothing is listening on port 8000, and no uvicorn or caffeinate processes are left running.

## Small fixes (after the follow-up)
| File | Change | Check |
|---|---|---|
| `requirements.txt` | Added `torchxrayvision==1.5.4` after `torchvision`. The served X-ray model needs it, and 1.5.4 is the version installed in `backend/.venv` | `backend/.venv` reports torchxrayvision 1.5.4. No install was run, since the venv already had it |
| `scripts/test_new_xray_checkpoint.py` (lines 233–234) | **Print text only.** The startup line now reads `(read-only; this is the served model, xray_txrv_densenet121_best.pt -- backend/main.py untouched)` instead of calling `xray_best_model_phase35_v2.pt` the "active checkpoint". No logic changed | `py_compile` passes. Not rerun (nothing heavy, as asked), so the new text hasn't been seen in a live run |
| `demo/terminal2_commands.txt` | Expected-output line updated to the new text; removed the "stale wording, don't read this line out" warning | – |

The module docstring of `test_new_xray_checkpoint.py` (line 5) still mentions `xray_best_model_phase35_v2.pt` as the active checkpoint. It's a comment, not printed output, so I left it alone.
