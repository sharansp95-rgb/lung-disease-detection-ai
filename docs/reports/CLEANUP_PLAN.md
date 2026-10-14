# Cleanup Plan (2026-10-13)

Destination: `_archive_2026-10-13/<original relative path>`. Nothing is deleted, and everything can be restored with one command (see docs/reports/CLEANUP_REPORT.md).

**How references were checked.** I scanned every text file in the project (`.py .js .jsx .json .md .txt .sh .ps1 .ipynb .html .css .log .gitignore .env*` …, including `frontend/`, `demo/`, `backend/`, configs). The scan excluded `backend/.venv`, `frontend/node_modules`, `frontend/dist`, `data/`, `raw/` and `seg_data/`, which contain no project-authored text. It searched for each candidate's relative path and file name. For `.py` files I also searched for `import <module>` / `from <module> import`. A reference from inside the candidate itself doesn't count.

**Exception: regenerable caches** (`.DS_Store`, `__pycache__`). Step 2 of your instructions allows these regardless. Their only "references" are ignore patterns in `.gitignore` / `frontend/.gitignore` and entries in the old `docs/CLEANUP_LOG.md` saying they were deleted on 10-04. Neither is a dependency. I'm moving them, not deleting them.

## A. To move (reference check passed)
| # | Path | Size | Files | Reason | References found |
|---|---|---|---|---|---|
| 1 | `.DS_Store` (root) | 20 KB | 1 | macOS Finder metadata | none (only ignore patterns in .gitignore files + old cleanup log; regenerable cache) |
| 2 | `Claude outputs/` | 8 KB | 1 | empty folder apart from a `.DS_Store` | none |
| 3 | `backend/__pycache__/` | 148 KB | 17 | Python bytecode cache | none (old cleanup log only; regenerable) |
| 4 | `backend/tests/__pycache__/` | 128 KB | 6 | Python bytecode cache | none (old cleanup log only; regenerable) |
| 5 | `ml/__pycache__/` | 56 KB | 5 | Python bytecode cache | none (old cleanup log only; regenerable) |
| 6 | `scripts/__pycache__/` | 52 KB | 3 | Python bytecode cache | none (old cleanup log only; regenerable) |
| 7 | `docs/ui_backup_2026-10-06/` | 132 KB | 22 | old copy of the frontend `index.html` + `src/` from before the 10-06 UI change | none |
| 8 | `logs/local_backend.err.log` | 8 KB | 1 | old local-run log (Oct 6) | none |
| 9 | `logs/local_backend.out.log` | 8 KB | 1 | old local-run log (Oct 6) | none |
| 10 | `logs/local_frontend.err.log` | 28 KB | 1 | old local-run log (Oct 6) | none |
| 11 | `logs/local_frontend.out.log` | 4 KB | 1 | old local-run log (Oct 6) | none |
| 12 | `logs/start_public.out` | 4 KB | 1 | old console capture of `start_public.sh` (Oct 7) | none |
| 13 | `logs/start_public.out.log` | 4 KB | 1 | old console capture of `start_public.sh` (Oct 7) | none |
| | **Total** | **~600 KB** | **61** | | |

## B. Held back: proposed as clutter, but references were found (left in place)
| Path | Size | References found | Note |
|---|---|---|---|
| `audit_backup/` | 9.4 MB | `docs/reports/DEMO_REHEARSAL_REPORT.md` | Not identical to `backup_before_rehearsal/audit` (an earlier snapshot). The report (protected) names it. |
| `eval_outputs_backup/` | 61 MB | `docs/reports/DEMO_REHEARSAL_REPORT.md` | Same situation. |
| `scripts/predict_cli.py.bak` | 8 KB | `docs/reports/DEMO_REHEARSAL_REPORT.md` | Old pre-10-04 CLI. |
| `docs/archive/` | 2.3 MB | `scripts/diagnose_generalization.py` (writes to `docs/archive/diagnosis_outputs`, reads `GENERALIZATION_DIAGNOSIS.md`), `test/README.md`, `test/results/DIAGNOSIS.md`, `docs/CLEANUP_LOG.md` | Code writes into it, so it must stay. |
| `eval_run/` (whole folder, phase notes + experiment scripts) | 5.3 MB | `backend/seg_utils.py` (cites `eval_run/xray_cropfix_applied.md`), `scripts/export_graph_metrics.py`, `scripts/prepare_ct_scan_normal_augmented.py`, `scripts/ct_normal_pool_split.py`, provenance fields in `eval_outputs/*.json`, `docs/archive/*` | 12 files had no external reference (`chart_*.png` ×4, `run_auth/bulk/concurrency/gates.py`, `make_charts.py`, `phaseJ_runner.py`, 3 small JSONs). But `harness.py` is imported by `cropfix_live_run.py`, and the run scripts produced `results_*.json` files that are cited. Pulling them out would separate experiments from their own results, so I kept the folder whole. |
| `logs/public_backend.*.log`, `logs/public_preview.*.log` | <1 KB | `start_public.sh`, `start_public.ps1`, `docs/DEPLOYMENT_PUBLIC.md` | Written by the start script. |
| `logs/modality_gate_debug.log` | 36 KB | `backend/main.py` | Live log. |
| `logs/warmup_backend.log/.pid` | 6 KB | `demo/warmup_backend.sh`, `demo/run_demo.sh` | Demo. |

If you agree the three backup items are clutter, this moves them after the review, without touching anything else:
```
mkdir -p _archive_2026-10-13/scripts && mv audit_backup eval_outputs_backup _archive_2026-10-13/ && mv scripts/predict_cli.py.bak _archive_2026-10-13/scripts/
```

## C. Checkpoints: all referenced, none moved
| Checkpoint | Size | Referenced by (code/config) |
|---|---|---|
| `xray_txrv_densenet121_best.pt` | 27 MB | `backend/main.py` CHECKPOINTS, `scripts/test_new_xray_checkpoint.py` |
| `ct_best_model_phase35_v2.pt` | 27 MB | `backend/main.py`, `demo/terminal2_commands.txt` |
| `modality_gate_best_model_v2.pt` | 43 MB | `backend/main.py` |
| `xray_sanity_best.pt`, `ct_sanity_best.pt` | 43 MB each | `backend/main.py` |
| `xray_seg_model.pt` | 30 MB | `backend/main.py` (SEG_CHECKPOINT) |
| `modality_gate_best_model.pt` (v1) | 43 MB | **`scripts/diagnose_generalization.py`**, `test/results/trace_breaking.py`: code references, so it stays |
| `xray_best_model_phase35_v2.pt` (old X-ray) | 27 MB | **`scripts/compare_crop_fixes.py`**, `test/results/trace_breaking.py`, `backend/main.py` docstring: code references, so it stays |

## D. Not considered (protected or in use)
`backend/.env`, `frontend/.env*`, `backend/app.db`, `backend/ood_stats/`, `data/`, `raw/`, `seg_data/`, `datasets/`, `demo/`, `docs/reports/DEMO_REHEARSAL_REPORT.md`, `documents/`, `docs/uml/`, `screenshots/`, `test images/` (incl. the `.DS_Store` files inside protected folders), `backup_before_rehearsal/`, `backend/tests/_test.db` (recreated by `conftest.py`), `test/`, `ml_logs/`, `audit/`, `eval_outputs/`, `checkpoints/`, root `*.md`, `start_public.*`.
