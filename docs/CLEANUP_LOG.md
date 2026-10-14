# Cleanup Log

Pre-evaluation cleanup, 2026-10-04. Everything listed here was deleted or moved on purpose, after the plan was approved.
**The full pre-cleanup project is backed up on the college server.** To restore an item, copy that path back from the backup.

Nothing protected was touched: the 6 served checkpoints, `data/` runtime folders, `seg_data/`, `raw/`, `documents/`, `backend/.env`, `backend/app.db` and `backend/.venv` are all intact.

## Deleted: DELETE (unreferenced junk)

| path | size | files |
|---|---|---|
| `.DS_Store` | 12.0 KB | 1 |
| `documents/.DS_Store` | 6.0 KB | 1 |
| `test images/.DS_Store` | 10.0 KB | 1 |
| `test images/ct scan/.DS_Store` | 8.0 KB | 1 |
| `test images/x ray/.DS_Store` | 8.0 KB | 1 |
| `backend/__pycache__` | 165.6 KB | 19 |
| `backend/tests/__pycache__` | 106.5 KB | 6 |
| `backend/.pytest_cache` | 5.7 KB | 5 |
| `ml/__pycache__` | 42.7 KB | 5 |
| `scripts/__pycache__` | 15.4 KB | 1 |
| `eval_run/__pycache__` | 3.6 KB | 1 |
| `eval_run/phaseE/__pycache__` | 51.8 KB | 3 |
| `backend/app.db.bak-20260927` | 640.0 KB | 1 |
| `backend/app.db.bak-20260927-231927` | 640.0 KB | 1 |
| `data/ct_scan.zip` | 141.9 MB | 1 |
| `logs/local_backend.log` | 8.7 KB | 1 |
| `logs/local_frontend.log` | 0.2 KB | 1 |
| `logs/public_start.log` | 1.1 KB | 1 |
| `frontend/dev_frontend.log` | 0.3 KB | 1 |
| `frontend/dev_frontend_run.log` | 0.6 KB | 1 |
| `eval_run/frontend_build.log` | 0.4 KB | 1 |
| `eval_run/frontend_dev.log` | 0.4 KB | 1 |
| `eval_run/bulk_A.out` | 1.5 KB | 1 |
| `eval_run/bulk_B.out` | 0.7 KB | 1 |
| `eval_run/concurrency.out` | 1.0 KB | 1 |
| `eval_run/cropfix_live.out` | 2.8 KB | 1 |
| `eval_run/gates2.out` | 0.3 KB | 1 |
| `eval_run/run_gates.out` | 3.2 KB | 1 |
| `frontend/shot_chat_no_dividers.png` | 42.9 KB | 1 |
| `frontend/shot_design_chat_chips.png` | 46.9 KB | 1 |
| `frontend/shot_design_results.png` | 203.3 KB | 1 |
| `frontend/shot_new_landing_dark.png` | 63.1 KB | 1 |
| `frontend/shot_new_landing_light.png` | 74.8 KB | 1 |
| `frontend/shot_profile_redesigned.png` | 113.3 KB | 1 |
| `seg_data_sample_grid.png` | 327.0 KB | 1 |
| `tb85_regression_check.png` | 244.6 KB | 1 |

Subtotal: 144.6 MB

## Deleted: ASK-A (approved batch)

| path | size | files |
|---|---|---|
| `backend/dev_backend.log` | 98.9 KB | 1 |
| `backend/dev_backend_run.log` | 8.5 KB | 1 |
| `frontend/dist` | 299.0 KB | 5 |
| `backend/tests/_test.db` | 32.0 KB | 1 |
| `eval_run/qa.db` | 196.0 KB | 1 |
| `logs/modality_gate_debug.log` | 1.2 MB | 1 |
| `logs/public_backend.err.log` | 4.9 KB | 1 |
| `logs/public_backend.out.log` | 1.8 KB | 1 |
| `logs/public_preview.err.log` | 0.0 KB | 1 |
| `logs/public_preview.out.log` | 0.0 KB | 1 |
| `test.zip` | 20.5 KB | 1 |
| `data/xray.zip` | 461.5 MB | 1 |
| `frontend/screenshot_backend_down.png` | 70.6 KB | 1 |
| `frontend/screenshot_borderline_warning.png` | 147.1 KB | 1 |
| `frontend/screenshot_chat_closed.png` | 29.2 KB | 1 |
| `frontend/screenshot_chat_declined_treatment.png` | 222.9 KB | 1 |
| `frontend/screenshot_chat_no_context.png` | 64.9 KB | 1 |
| `frontend/screenshot_chat_open_conversation.png` | 229.9 KB | 1 |
| `frontend/screenshot_ct.png` | 211.6 KB | 1 |
| `frontend/screenshot_error_wrongtype.png` | 28.5 KB | 1 |
| `frontend/screenshot_mismatched_modality.png` | 210.8 KB | 1 |
| `frontend/screenshot_mismatched_modality_reverse.png` | 189.9 KB | 1 |
| `frontend/screenshot_original_toggle.png` | 153.6 KB | 1 |
| `frontend/screenshot_oversized.png` | 28.9 KB | 1 |
| `frontend/screenshot_xray.png` | 193.4 KB | 1 |
| `frontend/change_password_success.png` | 29.5 KB | 1 |
| `demo_heatmap_normal.png` | 59.5 KB | 1 |
| `demo_heatmap_pneumonia.png` | 59.1 KB | 1 |
| `checkpoints/ct_best_model_phase36_normal_fix.pt` | 27.1 MB | 1 |
| `checkpoints/ct_best_model_phase36_normal_fix_mfonly.pt` | 27.1 MB | 1 |
| `checkpoints/ct_best_model_phase37_framing_normalized.pt` | 27.1 MB | 1 |
| `checkpoints/ct_phase37_framing_normalized_best.pt` | 27.1 MB | 1 |
| `checkpoints/ct_lidc_nodulecrop_v4_stack3.pt` | 27.1 MB | 1 |
| `PROJECT_CONTEXT.md` | 0.0 KB | 1 |
| `backend/groq_report_text.txt` | 0.5 KB | 1 |
| `backend/ct_fp_test_list.txt` | 1.0 KB | 1 |
| `backend/xray_fp_test_list.txt` | 1.0 KB | 1 |
| `.claude` | 0.0 KB | 0 |
| `logs/ (empty dir)` | 0.0 KB | 0 |

Subtotal: 601.0 MB

## Deleted: ASK-B (approved decisions)

| path | size | files |
|---|---|---|
| `datasets/01_raw_covid19_radiography_database` | 559.4 MB | 30311 |
| `datasets/02_raw_tb_chest_radiography_database` | 662.5 MB | 4203 |
| `datasets/03_raw_iqothnccd_lung_cancer_dataset` | 150.3 MB | 1098 |
| `datasets/04_processed_xray_pipeline` | 458.5 MB | 19217 |
| `datasets/05_processed_ct_pipeline` | 142.8 MB | 1036 |
| `data/sanity_export.zip` | 243.2 MB | 1 |
| `frontend/test_auth_flow.cjs` | 5.1 KB | 1 |
| `frontend/test_change_password.cjs` | 3.2 KB | 1 |
| `frontend/test_chatbot.cjs` | 3.8 KB | 1 |
| `frontend/test_edge_cases.cjs` | 1.9 KB | 1 |
| `frontend/test_error_flow.cjs` | 1.7 KB | 1 |
| `frontend/test_flow.cjs` | 3.1 KB | 1 |
| `frontend/test_mismatch.cjs` | 1.8 KB | 1 |

Subtotal: 2.16 GB

## Moved

| from | to | files |
|---|---|---|
| `PROJECT_BRIEF.md` | `docs/archive/PROJECT_BRIEF.md` | 1 |
| `PROJECT_BRIEFING.md` | `docs/archive/PROJECT_BRIEFING.md` | 1 |
| `PROJECT_HANDOFF.md` | `docs/archive/PROJECT_HANDOFF.md` | 1 |
| `PROJECT_STATUS.md` | `docs/archive/PROJECT_STATUS.md` | 1 |
| `PROJECT_UNDERSTANDING.md` | `docs/archive/PROJECT_UNDERSTANDING.md` | 1 |
| `VERIFICATION_REPORT.md` | `docs/archive/VERIFICATION_REPORT.md` | 1 |
| `VERIFICATION_REPORT_2.md` | `docs/archive/VERIFICATION_REPORT_2.md` | 1 |
| `GENERALIZATION_DIAGNOSIS.md` | `docs/archive/GENERALIZATION_DIAGNOSIS.md` | 1 |
| `diagnosis_outputs` | `docs/archive/diagnosis_outputs` | 10 |
| `lung_opacity_seg_check.png` | `docs/figures/lung_opacity_seg_check.png` | 1 |

## Files edited or created

- `datasets/README.md`: rewritten to point to `raw/` and `data/` now that the duplicate copies are gone.
- `docs/archive/README.md`: new; marks the archived notes as superseded and lists outdated numbers.

## Totals

- Deleted: 88 paths, 55,983 files, **2.89 GB**
- Moved: 10 paths, 19 files


## Other changes made during the cleanup session

- `backend/app.db`: reset on request ("test the website from scratch"). It already held 0 users, 0 tokens and 0 scans; all three tables were emptied again, ID counters reset, and the file compacted from 640 KB to 32 KB. Schema unchanged. Any browser that used the app before should log out, because user IDs restart at 1.
- `backend/tests/conftest.py`: added a `verified_user` fixture that marks the signed-up test user as verified in the test DB.
- `backend/tests/test_predict.py`: the scan-history test now uses `verified_user` (it was failing with 403 because its user was never verified); added `test_predict_unverified_user_gets_403_and_nothing_saved` and `test_chat_unverified_user_gets_403` so the 403 rule stays covered. Suite: 61 passed.
- `CLEANUP_PLAN.md`: deleted after execution (Phase 5).
- Regenerated by the verification runs and removed again at the end: `__pycache__/` (backend, backend/tests, ml, scripts), `backend/tests/_test.db`, `logs/` (main.py recreates it on startup).

## Demo scripts updated (Phase 6)

- `demo/terminal1_commands.txt`, `demo/terminal2_commands.txt`, `demo/terminal3_commands.txt`: `D:\` paths and `sample images/` replaced with relative paths and `test images/`; venv activation added; expected output replaced with real dry-run output from 2026-10-04.
- Terminal 2 now evaluates the **served** models: `scripts/test_new_xray_checkpoint.py` (96.57%) and `ml/evaluate.py` on the CT checkpoint (98.06%). The old command evaluated the retired ImageNet X-ray model (97.87%).
- Terminal 3 is marked with a KNOWN ISSUE: `scripts/predict_cli.py` still uses the retired X-ray model, so its X-ray runs are rejected by the OOD check. Not caused by the cleanup; `predict_cli.py` was not changed.
- The dry runs rewrote 7 evidence files in `audit/` and `eval_outputs/`; all 13 backed-up evidence files were restored byte-identical afterwards (the regenerated versions differed only in file paths, 0.01-level rounding and PNG rendering, with identical metrics).
