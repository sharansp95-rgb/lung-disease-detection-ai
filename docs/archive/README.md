# Archived working notes

These files are **superseded working notes** kept for traceability. They were
written during development as status snapshots and hand-off notes, and they
were moved here from the repository root in the pre-evaluation cleanup (see
`../CLEANUP_LOG.md`). Do not cite them.

## Source of truth

The final IEEE paper and project report in `documents/` are the source of
truth for every number:

| Measurement | Value |
|---|---|
| X-ray classifier, held-out test split (offline) | **96.57%** |
| X-ray, full test split through the live API, all guards active | **91.05%** (1,750 / 1,922) |
| CT classifier, held-out test split | **98.06%** (101 / 103) |

Where a file below disagrees with these, the paper and report win.

## Files with outdated numbers or conclusions

| File | Snapshot | What is outdated |
|---|---|---|
| `PROJECT_STATUS.md` | 2026-08-25 to 09-06 | Quotes 97.87% / 98.23% X-ray test accuracy. Both belong to the earlier ImageNet-backbone model, which is no longer served |
| `VERIFICATION_REPORT.md` | 2026-08-28 | Fact-check of the Review-1 report and slides; describes the project before training and the final pipeline |
| `VERIFICATION_REPORT_2.md` | 2026-09-06 | QA of the running system **before** the X-ray crop fix; its live-accuracy findings predate the 91.05% figure, and it quotes 97.87% for the old model |
| `PROJECT_HANDOFF.md` | 2026-09-05 / 09-10 | Self-described as partly stale; uses the 90.9% (176 / 183) live figure from the slides instead of 91.05% |
| `PROJECT_BRIEFING.md` | 2026-09-15 | Mostly current, but also repeats the 90.9% slide figure |
| `GENERALIZATION_DIAGNOSIS.md` (+ `diagnosis_outputs/`) | 2026-09-05 | Its "external" probe images were never provenance-checked; `test/README.md` warns they may come from the training databases, in which case its conclusions should be discarded |

## Other notes (superseded, numbers broadly current)

| File | Snapshot |
|---|---|
| `PROJECT_BRIEF.md` | 2026-09-15 |
| `PROJECT_UNDERSTANDING.md` | 2026-09-10 |

## Known side effects of the move

- Some notes link to paths relative to the repository root (for example
  `backend/main.py`); those links no longer resolve from this folder.
- `scripts/diagnose_generalization.py` still writes its outputs to the
  repository root (`GENERALIZATION_DIAGNOSIS.md`, `diagnosis_outputs/`). If it
  is re-run, move the new outputs here or delete them.
- `CLASSIFIER_EVALUATION_REPORT.md` was deliberately **not** archived: it stays
  at the repository root because `scripts/export_graph_metrics.py` reads it
  from there.
