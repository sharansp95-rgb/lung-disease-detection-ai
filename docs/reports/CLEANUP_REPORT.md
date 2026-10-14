# Project Cleanup Report

Started 2026-10-13 23:35. Goal: a clean, professional project folder for Review 2 (Thu 15 Oct) without breaking the demo or the website.
Status: ✅ COMPLETE (23:35 → 00:03). All 5 verification checks passed; nothing had to be moved back.

## Git snapshot
`git rev-parse` → "not a git repository". **No snapshot commit was made.** I did not run `git init`: that would mean committing ~7 GB of data/raw/checkpoints, and you only asked for a commit if it was already a repo. Rollback instead comes from the archive: nothing is deleted, and every move is reversible with one `mv` (command at the end).

## Time estimate (written before any change)
| Step | Estimate |
|---|---|
| 1. Inventory + reference checks + docs/reports/CLEANUP_PLAN.md | 25 min |
| 2. Execute moves | 5 min |
| 3. README.md, stale comments/docstrings, .gitignore | 20 min |
| 4. Verify: pytest (~9 min) → CLI batch (~9 min) → CT eval (12 s) → `npm run build` (~2 min) → demo path check (1 min), run one after another | 23 min |
| Final report | 10 min |
| **Total** | **~1 h 25 min** |

## Progress log

### Step 1. Plan: ✅ (23:35–23:38) → `docs/reports/CLEANUP_PLAN.md`
- Reference scan: every text file in the project, by relative path and file name, plus `import`/`from` for Python modules.
- **13 items pass** (61 files, ~600 KB). **8 proposed items were held back because references were found**, including `audit_backup/`, `eval_outputs_backup/`, `scripts/predict_cli.py.bak` (named in the protected `docs/reports/DEMO_REHEARSAL_REPORT.md`), `docs/archive/` (`scripts/diagnose_generalization.py` writes into it) and `eval_run/` (cited by `backend/seg_utils.py` and others).
- **No checkpoint moved: all 8 are referenced.** That includes `modality_gate_best_model.pt` v1 (referenced by `scripts/diagnose_generalization.py`) and the old `xray_best_model_phase35_v2.pt` (`scripts/compare_crop_fixes.py`).

### Step 2. Execute: ✅ (23:38)
Moved exactly the 13 plan-A items to `_archive_2026-10-13/` with their relative paths kept. Nothing was deleted, including caches.

### Step 3. Tidy: ✅
- **`README.md`** (new): title, team + USNs and guide (taken from the report's own LaTeX source, `documents/Lung_Disease_Detection_Report.zip` → `details.tex` / `titlepage.tex`), overview, key results, folder structure, macOS run commands (from `main.py`'s docstring, `frontend/package.json`, `vite.config.js`, `frontend/src/api.js`, `backend/config.py`), tests, pointer to `demo/`.
- **Stale comments/docstrings (comments only, no logic):**

| File | Was | Now |
|---|---|---|
| `backend/main.py` (module docstring, lines 2–3) | "Phase 4: FastAPI backend serving the two verified checkpoints (xray_best_model_phase35_v2.pt, ct_best_model_phase35_v2.pt)." | Names the served `xray_txrv_densenet121_best.pt` + CT model, mentions the gates/segmentation models and that models lazy-load on the first prediction request |
| `scripts/test_new_xray_checkpoint.py` (module docstring, lines 2–5) | "…WITHOUT touching the active production checkpoint (checkpoints/xray_best_model_phase35_v2.pt)…" | Says this is the model `backend/main.py` serves; explains that "old" in the text means the earlier ImageNet-backbone model |
| `backend/models.py` (`_reproduce_val_accuracy_xrv` docstring, line 107) | "(at startup, …)" | "(when the models are first loaded -- in the web app, on the first prediction request -- …)" |

  Left as is (historical, not misleading): "Phase 3"/"Phase 4" labels in `backend/gradcam_utils.py`, `backend/report.py`, `ml/evaluate.py`.
- **`.gitignore`:** added `.env`, `.env.*` (with `!.env*.example` so the example files stay committable), `.venv-*/`, `_archive_*/`. It already covered `*.env`, `backend/.env`, `*.db`, `__pycache__/`, `.DS_Store`, `logs/`, `.venv/`, `venv/`. `frontend/.gitignore` already covers `*.local` (so `.env.public.local`) and `.DS_Store`.

### Step 4. Verify
- ✅ **4. `npm run build`**: `✓ 54 modules transformed … ✓ built in 145ms`, exit 0 (0.8 s). Before building I copied `frontend/dist/` (the public-mode build from `start_public.sh`), and afterwards I put it back byte-identical, so the build check changed nothing. The test build is kept in the scratchpad, not the project.
- ✅ **5. Demo paths**: 51 paths extracted from `demo/*.txt`, `demo/*.sh`, `demo/demo_batch.txt`, and all exist. The checker's other flags were prose, output text, whole command lines, the optional `tee` target `../demo/out_batch.txt` and the hypothetical `.venv-mac`; I checked their real path parts by hand and all exist.
- ✅ **1. pytest**: `62 passed, 2 warnings in 525.27s (0:08:45)` (23:40 → 23:49).
- ✅ **2. CLI batch** (`cd scripts && caffeinate -ims python predict_cli.py --batch ../demo/demo_batch.txt`): models verified in 627 s, total 10 min 36 s, exit 0. I compared it with `demo/rehearsal_outputs/t3_batch_all6_demo.txt` on every gate line, OOD distance, class probability, verdict and summary row (timings excluded): **82 of 82 result lines identical.**
  ```
  img_189af4.png  xray  Normal 99.8%       img_159819.jpg  ct    Benign 90.3%
  img_0058a2.png  xray  Pneumonia 99.9%    img_07ec48.jpg  ct    Malignant 98.1%
  img_19570d.png  xray  COVID-19 100.0%    img_159819.jpg  xray  REJECTED
  ```
  The run rewrites `demo/heatmap_*.png`. I copied them beforehand, and all 4 came out **byte-identical**, so `demo/` is unchanged.
- ✅ **3. CT eval**: `[evaluate:ct_phase35_v2] TEST accuracy: 0.9806` (17 s, exit 0).


## Estimated vs actual time
| Step | Estimate | Actual |
|---|---|---|
| 1. Plan | 25 min | ~4 min |
| 2. Execute | 5 min | <1 min |
| 3. README, comments, .gitignore | 20 min | ~4 min |
| 4. Verify | 23 min | ~20 min (pytest 8 min 45 s, CLI batch 10 min 36 s, CT 17 s, build 1 s, path check <1 min) |
| Report | 10 min | ~3 min |
| **Total** | **~1 h 25 min** | **~30 min** (23:35 → 00:03) |

The verification runs dominated, as expected. The rest went faster than estimated because only 13 items passed the reference check.

## What was moved, and space freed
13 items, **61 files, ~600 KB**, now in `_archive_2026-10-13/` (list and reasons in `docs/reports/CLEANUP_PLAN.md`, section A): root `.DS_Store`, `Claude outputs/`, 4 `__pycache__/` folders, `docs/ui_backup_2026-10-06/`, and 6 old logs in `logs/`.

**The space freed is small, because the big items were held back by the reference rule.** The largest candidates are `eval_outputs_backup/` (61 MB) and `audit_backup/` (9.4 MB). Both are named in the protected `docs/reports/DEMO_REHEARSAL_REPORT.md`, so they stayed (see `docs/reports/CLEANUP_PLAN.md`, section B, for the one-line `mv` if you want them archived after the review, ~71 MB). No checkpoints moved; all 8 are referenced by code.

Note: the verification runs recreated `backend/`, `backend/tests/`, `ml/` and `scripts/` `__pycache__/`, as any Python run does. They're in `.gitignore` and harmless. The archived copies are the old ones.

## Files changed (text only)
| File | Change |
|---|---|
| `README.md` | **new** (see Step 3) |
| `backend/main.py` | module docstring only (lines 2–6) |
| `backend/models.py` | one docstring line (107–108) |
| `scripts/test_new_xray_checkpoint.py` | module docstring only (lines 2–7) |
| `.gitignore` | +`.env`, `.env.*`, `!.env*.example`, `.venv-*/`, `_archive_*/`; venv comment updated |
| `docs/reports/CLEANUP_PLAN.md`, `docs/reports/CLEANUP_REPORT.md` | **new** |

pytest (62/62) and the CLI batch both ran after these edits, so the docstring changes are verified harmless. Not touched: `backend/.env`, `frontend/.env*`, `backend/app.db`, `data/`, `raw/`, `seg_data/`, `datasets/`, `backend/ood_stats/`, `checkpoints/`, `demo/`, `docs/reports/DEMO_REHEARSAL_REPORT.md`, `documents/` (only read the report source zip for names/USNs), `docs/uml/`, `screenshots/`, `test images/`, `backup_before_rehearsal/`. `frontend/dist/` was restored byte-identical after the build check. No git commit (not a repo), nothing deleted, nothing installed.

## Restore everything from the archive (one command, from the project root)
```
rsync -a --remove-source-files _archive_2026-10-13/ ./ && find _archive_2026-10-13 -type d -empty -delete
```
This moves all 61 files back to their original paths and then removes the emptied archive folders. A dry run (`rsync -a --dry-run --itemize-changes …`) confirmed every file maps back to its original location. To restore a single item: `mv "_archive_2026-10-13/<path>" "<path>"`.

## Summary
- Moved 13 unreferenced items (61 files, ~600 KB) to `_archive_2026-10-13/`. Nothing deleted. The big backups and every checkpoint stayed because references were found (see `docs/reports/CLEANUP_PLAN.md`).
- Added `README.md` (team, guide, real run/test commands, results), fixed 3 stale docstrings, tightened `.gitignore`.
- All checks pass: pytest 62/62, CLI batch identical to rehearsal (82/82 lines, heatmaps byte-identical), CT 0.9806, `npm run build` OK, all demo paths exist.

**Follow-up (2026-10-14):** moved `CLASSIFIER_EVALUATION_REPORT.md`, `CLEANUP_PLAN.md`, `CLEANUP_REPORT.md`, `DEMO_REHEARSAL_REPORT.md` → `docs/reports/` and `DEPLOYMENT_PUBLIC.md` → `docs/` (mv, nothing deleted). Updated references in `README.md` (link + folder line), `start_public.ps1` (3), `scripts/export_graph_metrics.py` (docstrings, source label, and the read path `ROOT / "docs/reports/CLASSIFIER_EVALUATION_REPORT.md"`), comments in `backend/config.py`, `backend/rate_limit.py`, `backend/tests/test_public_proxy.py`, `backend/README_AUTH.md`, `frontend/vite.config.js`, `frontend/src/api.js`, `frontend/.env.public.example`, and mentions inside the moved reports. Left unchanged as historical/generated: `eval_outputs/*.json`, the backup folders, `docs/archive/`, `docs/CLEANUP_LOG.md`. Checks: no bare old-path references remain in scope; `bash -n` passes on `start_public.sh`, `demo/run_demo.sh`, `demo/warmup_backend.sh`; `source demo/run_demo.sh 3` prints correctly.
