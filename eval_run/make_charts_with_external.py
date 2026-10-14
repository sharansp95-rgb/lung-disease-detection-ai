"""Per-class correct-classification %: dataset (data/ test split, live pipeline) vs
external (images sourced from outside data/ and raw/, see external_samples/manifest.csv).
Source: results_bulk_A.json + results_bulk_B.json (dataset), external_samples/results_external.json (external).
CT external Benign has n=0 -- no independent (non-IQ-OTH/NCCD) source of benign nodule CTs was found;
CT external Malignant/Normal come from mohamedhanyyy/chest-ctscan-images (LIDC-IDRI derived, independent
of IQ-OTH/NCCD which the project's own CT training data uses). X-ray external comes from
jtiptj/chest-xray-pneumoniacovid19tuberculosis, independent of the project's Kaggle sources."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ds_rows = json.loads(open("results_bulk_A.json", encoding="utf-8").read())
ds_rows += json.loads(open("results_bulk_B.json", encoding="utf-8").read())
ext_rows = json.loads(open("external_samples/results_external.json", encoding="utf-8").read())

CLASSES = {"X-ray": ["COVID-19", "Normal", "Pneumonia", "Tuberculosis"],
           "CT": ["Benign", "Malignant", "Normal"]}
MOD_KEY = {"X-ray": "xray", "CT": "ct"}


def per_class(rows, mod, c, pred_key):
    sub = [r for r in rows if r["modality"] == mod and r["true"] == c]
    n = len(sub)
    tp = sum(1 for r in sub if r["status"] == 200 and r[pred_key] == c)
    return tp, n


dataset_n, external_n = {}, {}
for label, classes in CLASSES.items():
    mod = MOD_KEY[label]
    for c in classes:
        dataset_n[(label, c)] = per_class(ds_rows, mod, c, "pred")
        etp = sum(1 for r in ext_rows if r["modality"] == mod and r["true"] == c and r["status"] == 200
                  and r["body"].get("predicted_class") == c)
        en = sum(1 for r in ext_rows if r["modality"] == mod and r["true"] == c)
        external_n[(label, c)] = (etp, en)

for label, classes in CLASSES.items():
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = range(len(classes))
    width = 0.35

    ds_vals, ext_vals = [], []
    for c in classes:
        tp, n = dataset_n[(label, c)]
        ds_vals.append(100 * tp / n if n else 0)
        etp, en = external_n[(label, c)]
        ext_vals.append(100 * etp / en if en else 0)

    ax.bar([i - width/2 for i in x], ds_vals, width, label="Dataset (data/ test split)", color="#4C72B0")
    ax.bar([i + width/2 for i in x], ext_vals, width, label="External (outside data/, raw/)", color="#DD8452")

    for i, c in enumerate(classes):
        tp, n = dataset_n[(label, c)]
        ax.text(i - width/2, ds_vals[i] + 2, f"{tp}/{n}", ha="center", fontsize=8)
        etp, en = external_n[(label, c)]
        label_txt = f"{etp}/{en}" if en else "n=0"
        ax.text(i + width/2, (ext_vals[i] + 2) if en else 2, label_txt, ha="center", fontsize=8,
                color="#000" if en else "#555")

    ax.set_ylim(0, 115)
    ax.set_ylabel("Correct classification %")
    ax.set_xticks(list(x))
    ax.set_xticklabels(classes)
    title_suffix = " (live pipeline incl. lung-crop bug — NOT the offline 96.57% figure)" if label == "X-ray" else ""
    ax.set_title(f"{label} per-class accuracy: dataset vs external{title_suffix}", fontsize=9.5, wrap=True)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    ax.axhline(100, color="gray", linewidth=0.5)
    fig.tight_layout()
    out = f"chart_{label.lower().replace('-', '')}_accuracy_with_external.png"
    fig.savefig(out, dpi=150)
    print("saved", out)

summary = {}
for label, classes in CLASSES.items():
    for c in classes:
        summary[f"{label}/{c}"] = {"dataset_tp_n": dataset_n[(label, c)], "external_tp_n": external_n[(label, c)]}
print(json.dumps(summary, indent=2))
