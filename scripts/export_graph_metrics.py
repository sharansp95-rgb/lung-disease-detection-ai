"""
Exports bar-graph-ready evaluation metrics for the two SERVED disease
classifiers only:

    X-ray: checkpoints/xray_txrv_densenet121_best.pt
    CT:    checkpoints/ct_best_model_phase35_v2.pt

    python export_graph_metrics.py            # writes eval_outputs/graph_metrics.json

Read-only inference (torch.no_grad, eval mode). No training, no web server, no
Groq call, and no existing project file is modified -- the only file written
is eval_outputs/graph_metrics.json.

Why recompute anything: the only recorded offline per-class X-ray metrics
(eval_outputs/xray_txrv_densenet121/, docs/reports/CLASSIFIER_EVALUATION_REPORT.md
Condition A) apply the XRV transform directly to data/xray/test and skip the
U-Net lung crop that /predict/xray runs on every upload -- and they predate the
2026-09-08 crop_and_resize fix in seg_utils.py. So they are not the served
preprocessing path. This script reuses the backend's own functions rather than
reimplementing them, so there is one definition of "served preprocessing":

  * models.load_model_bundle / seg_utils.load_seg_bundle -- the same loaders,
    including the startup reproduction checks (a checkpoint/data mismatch
    aborts this script exactly as it would abort the app)
  * X-ray: seg_utils.compute_lung_crop -> gradcam_utils.preprocess_image_xrv
  * CT:    gradcam_utils.preprocess_image (ImageNet path, no crop)
  * guardrails (full-pipeline simulation only): the same sanity gate ->
    modality gate -> OOD order and thresholds as main.py::_run_prediction,
    with gates on the uncropped ImageNet tensor and OOD on the classifier
    tensor, exactly as main.py does

main.py itself is NOT imported: importing it builds the FastAPI app and
attaches a FileHandler to logs/modality_gate_debug.log. Its thresholds are
parsed out of the source text instead, so they cannot drift from the app.

Two recomputed views per modality:
  test_offline              classifier only, backend preprocessing, every test image
  test_full_pipeline_sim    + blank/colour checks, sanity gate, modality gate, OOD
                            (an offline re-creation of the live route, used to
                            cross-check the recorded live-HTTP numbers)

Recorded (not recomputed) numbers are pulled from the files named in each
block's "source" field. External images for most recorded external probes were
deleted with eval_run/external_samples/ and eval_run/ct_normal_heldout/, so those
rows can only come from the recorded results. The one external pool still on
disk (datasets/ct_normal_extra/groupA) is recomputed here.
"""

import datetime
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from gradcam_utils import preprocess_image, preprocess_image_xrv  # noqa: E402
from models import is_xrv_backbone, load_model_bundle  # noqa: E402
from ood_detector import OODDetector  # noqa: E402
from seg_utils import compute_lung_crop, load_seg_bundle  # noqa: E402

OUT_PATH = ROOT / "eval_outputs" / "graph_metrics.json"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

CHECKPOINTS = {
    "xray": ROOT / "checkpoints" / "xray_txrv_densenet121_best.pt",
    "ct": ROOT / "checkpoints" / "ct_best_model_phase35_v2.pt",
    "gate": ROOT / "checkpoints" / "modality_gate_best_model_v2.pt",
    "xray_sanity": ROOT / "checkpoints" / "xray_sanity_best.pt",
    "ct_sanity": ROOT / "checkpoints" / "ct_sanity_best.pt",
}
DATA_DIRS = {
    "xray": ROOT / "data" / "xray",
    "ct": ROOT / "data" / "ct_scan",
    "gate": ROOT / "data" / "modality_gate",
    "xray_sanity": ROOT / "data" / "xray_sanity",
    "ct_sanity": ROOT / "data" / "ct_sanity",
}


# ------------------------------------------------------------- app settings --

def _main_py_constant(name: str) -> float:
    src = (BACKEND_DIR / "main.py").read_text(encoding="utf-8")
    m = re.search(rf"^{name}\s*=\s*([0-9.]+)", src, re.M)
    if not m:
        raise RuntimeError(f"could not find {name} in backend/main.py")
    return float(m.group(1))


def _main_py_min_confidence() -> dict:
    src = (BACKEND_DIR / "main.py").read_text(encoding="utf-8")
    m = re.search(r'^MIN_CONFIDENCE\s*=\s*\{"xray":\s*([0-9.]+),\s*"ct":\s*([0-9.]+)\}', src, re.M)
    return {"xray": float(m.group(1)), "ct": float(m.group(2))}


def _gate_threshold() -> float:
    """config.py default, unless backend/.env overrides it. Reads only that
    one key from .env -- no other value in the file is touched."""
    default = float(re.search(
        r'MODALITY_GATE_CONFIDENCE_THRESHOLD",\s*"([0-9.]+)"',
        (BACKEND_DIR / "config.py").read_text(encoding="utf-8"),
    ).group(1))
    env_path = BACKEND_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("MODALITY_GATE_CONFIDENCE_THRESHOLD="):
                return float(line.split("=", 1)[1].strip())
    return float(os.environ.get("MODALITY_GATE_CONFIDENCE_THRESHOLD", default))


BLANK_STD = _main_py_constant("BLANK_IMAGE_STD_THRESHOLD")
COLOR_DEV = _main_py_constant("COLOR_CONTENT_THRESHOLD")
GATE_THRESHOLD = _gate_threshold()


# ------------------------------------------------------------------ helpers --

def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load_image(path: Path) -> Image.Image:
    img = Image.open(path)
    img.load()
    return img


def classifier_tensor(modality: str, image: Image.Image, bundles, seg_bundle):
    """The exact tensor /predict/<modality> feeds the OOD check + classifier."""
    bundle = bundles[modality]
    if modality == "xray":
        image, _ = compute_lung_crop(image, seg_bundle)
        return (preprocess_image_xrv(image, bundle.device) if is_xrv_backbone(bundle.backbone)
                else preprocess_image(image, bundle.device))
    return preprocess_image(image, bundle.device)


@torch.no_grad()
def softmax_of(model, x) -> np.ndarray:
    model.eval()
    return torch.softmax(model(x), dim=1)[0].cpu().numpy()


@torch.no_grad()
def guardrail_verdict(modality: str, image: Image.Image, bundles, ood: OODDetector, x_cls) -> str | None:
    """Mirrors main.py::_read_and_validate_image + _run_prediction ordering.
    Returns None if the image would be served, else the rejection type."""
    if float(np.array(image.convert("L"), dtype=np.float32).std()) < BLANK_STD:
        return "blank"
    rgb = np.array(image.convert("RGB"), dtype=np.float32)
    if float((rgb.max(axis=2) - rgb.min(axis=2)).mean()) > COLOR_DEV:
        return "color"

    x_gate = preprocess_image(image, bundles[modality].device)  # uncropped, ImageNet -- as main.py
    sanity = bundles[f"{modality}_sanity"]
    if sanity.class_names[int(sanity.model(x_gate).argmax(dim=1))] != "scan":
        return "sanity"

    gate = bundles["gate"]
    gprobs = torch.softmax(gate.model(x_gate), dim=1)[0]
    gidx = int(gprobs.argmax())
    if float(gprobs[gidx]) < GATE_THRESHOLD or gate.class_names[gidx] != modality:
        return "gate"

    is_ood, _ = ood.check(bundles[modality].model, x_cls)
    return "ood" if is_ood else None


def metrics_block(y_true: list[str], y_pred: list[str], classes: list[str]) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes, average="macro", zero_division=0)
    _, _, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes, average="weighted", zero_division=0)
    n = len(y_true)
    return {
        "n": n,
        "accuracy": round(sum(t == q for t, q in zip(y_true, y_pred)) / n, 6),
        "macro_precision": round(float(pm), 6),
        "macro_recall": round(float(rm), 6),
        "macro_f1": round(float(fm), 6),
        "weighted_f1": round(float(fw), 6),
        "per_class": {
            c: {"precision": round(float(p[i]), 6), "recall": round(float(r[i]), 6),
                "f1": round(float(f[i]), 6), "support": int(s[i])}
            for i, c in enumerate(classes)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "confusion_matrix_order": "rows = true class, columns = predicted class, both in `classes` order",
    }


def eval_split(modality: str, split_dir: Path, bundles, seg_bundle, ood) -> dict:
    classes = bundles[modality].class_names
    y_true, y_pred = [], []
    live_true, live_pred = [], []          # REJECTED counted as its own column
    rejected = Counter()
    rejected_by_class: dict[str, Counter] = {c: Counter() for c in classes}
    low_conf = 0
    floor = _main_py_min_confidence()[modality] + 0.15

    for cls in classes:
        files = list_images(split_dir / cls)
        print(f"[{modality}] {split_dir.name}/{cls}: {len(files)} images")
        for path in files:
            image = load_image(path)
            x = classifier_tensor(modality, image, bundles, seg_bundle)
            probs = softmax_of(bundles[modality].model, x)
            pred = classes[int(probs.argmax())]
            y_true.append(cls)
            y_pred.append(pred)

            verdict = guardrail_verdict(modality, image, bundles, ood, x)
            live_true.append(cls)
            if verdict is None:
                live_pred.append(pred)
                low_conf += float(probs.max()) < floor
            else:
                live_pred.append("REJECTED")
                rejected[verdict] += 1
                rejected_by_class[cls][verdict] += 1

    offline = metrics_block(y_true, y_pred, classes)

    served = [(t, q) for t, q in zip(live_true, live_pred) if q != "REJECTED"]
    labels_live = classes + ["REJECTED"]
    p, r, f, _ = precision_recall_fscore_support(live_true, live_pred, labels=classes, zero_division=0)
    sim = {
        "n": len(live_true),
        "served": len(served),
        "accuracy_counting_rejections_as_wrong": round(sum(t == q for t, q in zip(live_true, live_pred)) / len(live_true), 6),
        "accuracy_over_served_only": round(sum(t == q for t, q in served) / max(len(served), 1), 6),
        "macro_f1_counting_rejections_as_wrong": round(float(np.mean(f)), 6),
        "per_class_counting_rejections_as_wrong": {
            c: {"precision": round(float(p[i]), 6), "recall": round(float(r[i]), 6), "f1": round(float(f[i]), 6)}
            for i, c in enumerate(classes)
        },
        "rejected": {k: rejected.get(k, 0) for k in ("blank", "color", "sanity", "gate", "ood")},
        "rejected_by_true_class": {c: {k: v for k, v in cnt.items()} for c, cnt in rejected_by_class.items()},
        "low_confidence_warnings_on_served": int(low_conf),
        "confusion_matrix": confusion_matrix(live_true, live_pred, labels=labels_live).tolist()[:len(classes)],
        "confusion_matrix_order": f"rows = true class ({classes}), columns = {labels_live}",
    }
    return offline, sim


def eval_external_pool(modality: str, folders: dict[str, Path], true_class: str, bundles, seg_bundle, ood) -> dict:
    """One true class, several sub-source folders. Classifier-only counts plus
    guardrail-simulated counts."""
    classes = bundles[modality].class_names
    out = {}
    for name, folder in folders.items():
        files = list_images(folder)
        pred_counts, served_pred_counts, rejected = Counter(), Counter(), Counter()
        for path in files:
            image = load_image(path)
            x = classifier_tensor(modality, image, bundles, seg_bundle)
            pred = classes[int(softmax_of(bundles[modality].model, x).argmax())]
            pred_counts[pred] += 1
            verdict = guardrail_verdict(modality, image, bundles, ood, x)
            if verdict is None:
                served_pred_counts[pred] += 1
            else:
                rejected[verdict] += 1
        n = len(files)
        served = sum(served_pred_counts.values())
        out[name] = {
            "n": n,
            "classifier_only": {
                "correct": pred_counts.get(true_class, 0),
                "accuracy": round(pred_counts.get(true_class, 0) / n, 6),
                "predicted_counts": {true_class: {c: pred_counts.get(c, 0) for c in classes}},
            },
            "full_pipeline_sim": {
                "served": served,
                "correct": served_pred_counts.get(true_class, 0),
                "accuracy_over_served": round(served_pred_counts.get(true_class, 0) / max(served, 1), 6),
                "accuracy_counting_rejections_as_wrong": round(served_pred_counts.get(true_class, 0) / n, 6),
                "predicted_counts": {true_class: {c: served_pred_counts.get(c, 0) for c in classes}},
                "rejected": {k: rejected.get(k, 0) for k in ("blank", "color", "sanity", "gate", "ood")},
            },
        }
    return out


def sha256_set(paths) -> set[str]:
    return {hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


# ------------------------------------------------------- recorded results --

def recorded_live_api() -> dict:
    """docs/reports/CLASSIFIER_EVALUATION_REPORT.md Condition B JSON blocks (live HTTP,
    2026-09-10). Rejection types are classified from each rejection_log reason
    using main.py's actual message wording."""
    src = "docs/reports/CLASSIFIER_EVALUATION_REPORT.md (Condition B, live FastAPI HTTP, 2026-09-10)"
    text = (ROOT / "docs/reports/CLASSIFIER_EVALUATION_REPORT.md").read_text(encoding="utf-8")
    out = {}
    for block in re.findall(r"```json\n(.*?)```", text, re.S):
        d = json.loads(block)
        if d["condition"] != "B_live":
            continue
        rej = Counter()
        for e in d["rejection_log"]:
            reason = e["reason"]
            if "feature-space distance" in reason:
                rej["ood"] += 1
            elif "Modality uncertain" in reason or "looks like a" in reason:
                rej["gate"] += 1
            elif "does not look like" in reason:
                rej["sanity"] += 1
            elif "blank" in reason or "color" in reason:
                rej["input_validation"] += 1
            else:
                rej["other"] += 1
        per_class = d["per_class_metrics"]
        supports = d["per_class_counts"]
        f1s = [v["f1"] for v in per_class.values()]
        out[d["modality"]] = {
            "source": src,
            "n": d["sample_size"]["total"],
            "served": d["sample_size"]["classified"],
            "accuracy": d["accuracy"],
            "accuracy_convention": "rejections counted as incorrect",
            "macro_f1": round(float(np.mean(f1s)), 6),
            "weighted_f1": round(sum(per_class[c]["f1"] * supports[c] for c in per_class) / sum(supports.values()), 6),
            "macro_and_weighted_f1_note": "derived here from the recorded per-class values (not stored in the report)",
            "per_class": {c: {**v, "support": supports[c]} for c, v in per_class.items()},
            "confusion_matrix": d["confusion_matrix"]["matrix"][:len(per_class)],
            "confusion_matrix_order": f"rows = true, columns = {d['confusion_matrix']['labels']}",
            "rejected": {"sanity": rej.get("sanity", 0), "gate": rej.get("gate", 0), "ood": rej.get("ood", 0),
                         "input_validation": rej.get("input_validation", 0), "other": rej.get("other", 0)},
            "rejected_by_true_class": dict(Counter((e["true"]) for e in d["rejection_log"])),
        }
    return out


def recorded_ct_external() -> list[dict]:
    """eval_run/results_ct_retrain_before.json -- per-image predictions of the
    served ct_best_model_phase35_v2.pt, classifier only, ImageNet eval
    transform (eval_run/ct_ckpt_eval.py). Images themselves are deleted."""
    path = "eval_run/results_ct_retrain_before.json"
    d = json.loads((ROOT / path).read_text(encoding="utf-8"))
    classes = d["class_names"]
    meta = {
        "ext_malignant_12": ("External Malignant (Kaggle mohamedhanyyy/chest-ctscan-images: adenocarcinoma / large-cell / squamous)",
                             "images deleted (was eval_run/external_samples/ct/malignant)"),
        "ext_normal_9": ("External Normal, original 9-image failure set (Kaggle mohamedhanyyy/chest-ctscan-images)",
                         "images deleted (was eval_run/external_samples/ct/normal); these 9 are ALSO inside groupB_mohamedhanyyy -- do not add them to it"),
        "groupB_mohamedhanyyy": ("External Normal, held-out Group B, mohamedhanyyy (9 patients)",
                                 "images deleted (was eval_run/ct_normal_heldout/groupB_mohamedhanyyy)"),
        "groupB_maftouni": ("External Normal, held-out Group B, Kaggle maedemaftouni/large-covid19-ct-slice-dataset 1NonCOVID (44 patients)",
                            "images deleted (was eval_run/ct_normal_heldout/groupB_maftouni)"),
    }
    rows = []
    for key, g in d["groups"].items():
        name, where = meta[key]
        counts = Counter(r["pred"] for r in g["rows"])
        true = g["true_class"]
        rows.append({
            "name": name,
            "source": f"{path} (groups.{key}); classifier only, no guardrails",
            "images_on_disk": where,
            "n": g["n"],
            "accuracy": round(g["correct"] / g["n"], 6),
            "per_class": {true: {"correct": g["correct"], "total": g["n"]}},
            "predicted_counts": {true: {c: counts.get(c, 0) for c in classes}},
            "rejected": None,
            "rejected_note": "not measured (offline classifier-only run)",
        })
    return rows


def recorded_phase_i() -> dict:
    """eval_run/phaseI_final_numbers.md -- live HTTP, 2026-09-10, after the
    X-ray crop fix. Markdown tables, transcribed here verbatim."""
    src = "eval_run/phaseI_final_numbers.md (live /predict HTTP, 2026-09-10)"
    xray = {
        "name": "External X-ray probe, Kaggle jtiptj/chest-xray-pneumoniacovid19tuberculosis, 10 per class",
        "source": src + " -- confusion breakdown §2; dataset name from eval_outputs/extracted_results_for_plotting.json and eval_run/make_charts_with_external.py",
        "images_on_disk": "images deleted (was eval_run/external_samples/xray/)",
        "n": 40,
        "served": 33,
        "accuracy": round(31 / 40, 6),
        "accuracy_convention": "rejections counted as incorrect (31/40); over served only = 31/33 = 0.939394",
        "accuracy_over_served": round(31 / 33, 6),
        "per_class": {"COVID-19": {"correct": 5, "total": 10}, "Normal": {"correct": 10, "total": 10},
                      "Pneumonia": {"correct": 10, "total": 10}, "Tuberculosis": {"correct": 6, "total": 10}},
        "predicted_counts": {
            "COVID-19": {"COVID-19": 5, "Normal": 2, "Pneumonia": 0, "Tuberculosis": 0, "REJECTED": 3},
            "Normal": {"COVID-19": 0, "Normal": 10, "Pneumonia": 0, "Tuberculosis": 0, "REJECTED": 0},
            "Pneumonia": {"COVID-19": 0, "Normal": 0, "Pneumonia": 10, "Tuberculosis": 0, "REJECTED": 0},
            "Tuberculosis": {"COVID-19": 0, "Normal": 0, "Pneumonia": 0, "Tuberculosis": 6, "REJECTED": 4},
        },
        "rejected": 7,
        "rejected_by_type": {"ood": 3, "color": 4, "sanity": 0, "gate": 0},
    }
    ct = {
        "name": "External CT probe (Kaggle mohamedhanyyy/chest-ctscan-images): 12 Malignant + 9 Normal; no external Benign source exists",
        "source": src + " -- master table",
        "images_on_disk": "images deleted (was eval_run/external_samples/ct/)",
        "n": 21,
        "served": 21,
        "accuracy": round(12 / 21, 6),
        "per_class": {"Benign": {"correct": 0, "total": 0}, "Malignant": {"correct": 12, "total": 12},
                      "Normal": {"correct": 0, "total": 9}},
        "predicted_counts": {"Benign": {}, "Malignant": {"Benign": 0, "Malignant": 12, "Normal": 0},
                             "Normal": {"Benign": 2, "Malignant": 7, "Normal": 0}},
        "predicted_counts_note": "Normal row split (7 Malignant / 2 Benign) from phaseI_final_numbers.md §'Numbers that differ' item 5",
        "rejected": 0,
    }
    return {"xray": xray, "ct": ct}


def recorded_no_crop_xray() -> dict:
    path = "eval_outputs/xray_txrv_densenet121/xray_txrv_densenet121_test_metrics.json"
    d = json.loads((ROOT / path).read_text(encoding="utf-8"))
    cr = d["classification_report"]
    return {
        "source": f"{path} (2026-09-06; XRV transform on data/xray/test WITHOUT the backend U-Net lung crop -- not the served path)",
        "n": int(cr["macro avg"]["support"]),
        "accuracy": round(cr["accuracy"], 6),
        "macro_f1": round(cr["macro avg"]["f1-score"], 6),
        "weighted_f1": round(cr["weighted avg"]["f1-score"], 6),
        "per_class": {c: {"precision": round(cr[c]["precision"], 6), "recall": round(cr[c]["recall"], 6),
                          "f1": round(cr[c]["f1-score"], 6), "support": int(cr[c]["support"])}
                      for c in d["class_names"]},
        "confusion_matrix": d["confusion_matrix"],
    }


# ---------------------------------------------------------------------- main --

def main():
    today = datetime.date.today().isoformat()
    print(f"thresholds from app source: blank<{BLANK_STD} colour>{COLOR_DEV} gate<{GATE_THRESHOLD}")

    bundles = {name: load_model_bundle(name, CHECKPOINTS[name], DATA_DIRS[name]) for name in CHECKPOINTS}
    seg_bundle = load_seg_bundle(ROOT / "checkpoints" / "xray_seg_model.pt", ROOT / "seg_data")
    oods = {m: OODDetector(m, bundles[m].device) for m in ("xray", "ct")}

    recomputed = f"recomputed today ({today}) by scripts/export_graph_metrics.py"
    result = {"generated_on": today}
    live = recorded_live_api()
    phase_i = recorded_phase_i()

    for modality in ("xray", "ct"):
        b = bundles[modality]
        offline, sim = eval_split(modality, DATA_DIRS[modality] / "test", bundles, seg_bundle, oods[modality])
        prep = ("U-Net lung crop (seg_utils.compute_lung_crop) -> preprocess_image_xrv" if modality == "xray"
                else "preprocess_image (ImageNet, no crop)")
        result[modality] = {
            "checkpoint": str(CHECKPOINTS[modality].relative_to(ROOT)).replace("\\", "/"),
            "classes": b.class_names,
            "startup_reproduction_check": {"checkpoint_val_acc": round(b.checkpoint_val_acc, 6),
                                           "reproduced_val_acc": round(b.reproduced_val_acc, 6)},
            "test_offline": {"source": f"{recomputed}; classifier only; {prep}", **offline},
            "test_full_pipeline_sim": {
                "source": f"{recomputed}; offline re-creation of /predict/{modality} guardrails "
                          f"(blank/colour -> sanity -> modality gate@{GATE_THRESHOLD} -> OOD), no HTTP",
                **sim,
            },
            "test_live_api": live[modality],
            "external": [],
        }

    result["xray"]["test_offline_without_lung_crop_superseded"] = recorded_no_crop_xray()
    result["xray"]["external"].append(phase_i["xray"])

    # CT external: recorded probes (images gone) + the one pool still on disk.
    result["ct"]["external"].append(phase_i["ct"])
    result["ct"]["external"].extend(recorded_ct_external())

    pool_root = ROOT / "datasets" / "ct_normal_extra" / "groupA"
    pool = eval_external_pool("ct", {"maftouni": pool_root / "maftouni", "mohamedhanyyy": pool_root / "mohamedhanyyy"},
                              "Normal", bundles, seg_bundle, oods["ct"])
    ct_train_hashes = sha256_set(p for p in DATA_DIRS["ct"].rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    names = {
        "maftouni": "External Normal, Group A pool, Kaggle maedemaftouni/large-covid19-ct-slice-dataset 1NonCOVID (110 patients)",
        "mohamedhanyyy": "External Normal, Group A pool, Kaggle mohamedhanyyy/chest-ctscan-images (18 patient groups)",
    }
    for key, r in pool.items():
        folder = pool_root / key
        overlap = len(sha256_set(list_images(folder)) & ct_train_hashes)
        result["ct"]["external"].append({
            "name": names[key],
            "source": f"{recomputed}; images at datasets/ct_normal_extra/groupA/{key}; classifier only (full-pipeline counts in 'full_pipeline_sim')",
            "images_on_disk": f"yes: datasets/ct_normal_extra/groupA/{key}",
            "n": r["n"],
            "accuracy": r["classifier_only"]["accuracy"],
            "per_class": {"Normal": {"correct": r["classifier_only"]["correct"], "total": r["n"]}},
            "predicted_counts": r["classifier_only"]["predicted_counts"],
            "rejected": sum(r["full_pipeline_sim"]["rejected"].values()),
            "rejected_note": "from full_pipeline_sim (offline guardrail simulation)",
            "full_pipeline_sim": r["full_pipeline_sim"],
            "sha256_overlap_with_data_ct_scan": overlap,
            "note": "Group A was training data for the unserved phase36/phase37 retrains, never for the served phase35_v2 -- external for the served model",
        })

    result["ct_volume_pipeline_removed"] = {
        "source": "eval_run/phaseJ_external_validation.md (live /predict/ct/volume, 2026-09-10)",
        "model": "MONAI 3D RetinaNet nodule detector + checkpoints/ct_lidc_nodulecrop_v4_stack3.pt nodule-crop classifier -- NOT the served whole-slice CT classifier",
        "external_dataset": "SPIE-AAPM-NCI Lung CT Challenge (LUNGx), 14 nodules / 11 patients",
        "n": 14,
        "accuracy": round(4 / 14, 6),
        "wilson_95ci": [0.117, 0.546],
        "per_class": {"Malignant": {"correct": 4, "total": 5}, "Benign": {"correct": 0, "total": 9}},
        "internal_lidc_idri_for_comparison": {"source": "eval_run/phaseI_final_numbers.md", "n": 84, "accuracy": round(53 / 84, 6)},
        "notes": "Verdict NO-GO; route, pipeline and frontend page were subsequently removed from the codebase. Do not plot alongside the served CT classifier as if it were the same model.",
    }

    # Overlap markers so no image is counted twice on a graph.
    for e in result["ct"]["external"]:
        if e["name"].startswith("External Malignant (Kaggle"):
            e["overlaps_with"] = "same 12 images as the Malignant row of the Phase I live CT probe"
        elif e["name"].startswith("External Normal, original 9"):
            e["overlaps_with"] = "same 9 images as the Normal row of the Phase I live CT probe; also a subset of Group B mohamedhanyyy (99)"
        elif e["name"].startswith("External CT probe"):
            e["overlaps_with"] = "its 9 Normal images are a subset of Group B mohamedhanyyy (99)"

    sim_matches_live = {
        m: result[m]["test_full_pipeline_sim"]["confusion_matrix"] == [
            row + ([0] if len(row) == len(result[m]["classes"]) else [])
            for row in result[m]["test_live_api"]["confusion_matrix"]
        ]
        for m in ("xray", "ct")
    }
    x_off, x_nocrop = result["xray"]["test_offline"], result["xray"]["test_offline_without_lung_crop_superseded"]
    result["cross_checks"] = {
        "full_pipeline_sim_confusion_matrix_equals_recorded_live_http": sim_matches_live,
        "xray_offline_with_lung_crop_vs_without": {"with_crop_served_path": x_off["accuracy"],
                                                   "without_crop_superseded": x_nocrop["accuracy"]},
    }
    result["caveats"] = [
        "X-ray 'test_offline' (served preprocessing, with U-Net lung crop) is the number to plot for the served model. The older 96.57% (eval_outputs/xray_txrv_densenet121, docs/reports/CLASSIFIER_EVALUATION_REPORT.md Condition A) skips the lung crop the backend applies to every upload.",
        "data/xray/test images were already lung-cropped at dataset-prep time; the backend crops again on upload. The served-path number therefore includes this double-crop, which is exactly what happens when a test-split file is uploaded to the app.",
        "Internal CT test split is small: 103 images, only 12 Benign (1 image = 8.3 points of Benign recall).",
        "All external X-ray and CT probe rows are tiny (n = 9 to 12 per class) except the CT Normal pools; the Phase I probes should be plotted with their denominators.",
        "No external CT Benign data exists for any source; the Benign external bar must be shown as n=0, not 0%.",
        "External CT is effectively two sources (Kaggle mohamedhanyyy/chest-ctscan-images and maedemaftouni/large-covid19-ct-slice-dataset); several recorded rows reuse the same images (see 'overlaps_with'). Non-overlapping set: Malignant 12, Group B mohamedhanyyy 99, Group B maftouni 165, Group A maftouni 405, Group A mohamedhanyyy 18.",
        "mohamedhanyyy/chest-ctscan-images was the project's originally-chosen CT dataset before it was replaced by IQ-OTH/NCCD (data/README.md); it was not used to train the served CT model. Group A/B were never in the served model's training data; Group A has 0 SHA-256 overlap with data/ct_scan (recomputed today).",
        "External X-ray source (Kaggle jtiptj/chest-xray-pneumoniacovid19tuberculosis) was NEVER provenance-checked against the COVID-19 Radiography / TB Chest Radiography training sources; such Kaggle aggregates often re-host those databases, so overlap is possible and unverified. The images are deleted, so this cannot be checked now.",
        "Recorded external probe images (eval_run/external_samples/, eval_run/ct_normal_heldout/) were deleted in a later cleanup; those rows are transcribed from recorded results and cannot be re-run.",
        "The Group B and Phase I CT results were run offline (classifier only) or live on 2026-09-10; no guardrail rejection data exists for the Group B rows.",
        "'Recomputed today' predictions use torch.no_grad forward passes; the live route computes the same logits inside Grad-CAM with gradients enabled. The confusion matrices are identical to the recorded live run, so this makes no measurable difference.",
        "Live-API macro/weighted F1 are derived from the recorded per-class values; the report itself stores only accuracy and per-class metrics.",
        "The CT volume pipeline's 28.6% (Phase J) belongs to a different, removed model (MONAI detector + LIDC nodule-crop classifier) and must not be plotted as the served CT classifier.",
    ]

    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
