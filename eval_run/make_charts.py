"""Per-class correct-classification % charts: dataset (live, buggy-crop pipeline) vs
non-dataset (no labeled per-class set exists -- see caption). Source: results_bulk_A.json
+ results_bulk_B.json (section 1A sweep), results_gates2.json (section 1D Lung_Opacity probe)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = json.loads(open("results_bulk_A.json", encoding="utf-8").read())
rows += json.loads(open("results_bulk_B.json", encoding="utf-8").read())

CLASSES = {"X-ray": ["COVID-19", "Normal", "Pneumonia", "Tuberculosis"],
           "CT": ["Benign", "Malignant", "Normal"]}
MOD_KEY = {"X-ray": "xray", "CT": "ct"}

dataset_pct = {}
dataset_n = {}
for label, classes in CLASSES.items():
    mod = MOD_KEY[label]
    for c in classes:
        sub = [r for r in rows if r["modality"] == mod and r["true"] == c]
        n = len(sub)
        tp = sum(1 for r in sub if r["status"] == 200 and r["pred"] == c)
        dataset_pct[(label, c)] = 100 * tp / n if n else None
        dataset_n[(label, c)] = (tp, n)

# Non-dataset: no per-class labeled set exists for any of these 7 classes.
# (Lung_Opacity, n=25, was tested but is a 5th excluded class with no ground truth
# among these 7 -- reported separately, not plotted as a class bar.)
non_dataset_pct = {k: None for k in dataset_pct}
non_dataset_n = {k: (0, 0) for k in dataset_pct}

for label, classes in CLASSES.items():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(classes))
    width = 0.35
    ds_vals = [dataset_pct[(label, c)] for c in classes]
    nd_vals = [non_dataset_pct[(label, c)] or 0 for c in classes]

    bars1 = ax.bar([i - width/2 for i in x], ds_vals, width, label="Dataset (data/ test split)", color="#4C72B0")
    bars2 = ax.bar([i + width/2 for i in x], nd_vals, width, label="Non-dataset (no labeled set — n=0)", color="#DD8452", alpha=0.35, hatch="//")

    for i, c in enumerate(classes):
        tp, n = dataset_n[(label, c)]
        ax.text(i - width/2, ds_vals[i] + 2, f"{tp}/{n}", ha="center", fontsize=8)
        ax.text(i + width/2, 6, "n/a", ha="center", fontsize=8, color="#555")

    ax.set_ylim(0, 115)
    ax.set_ylabel("Correct classification %")
    ax.set_xticks(list(x))
    ax.set_xticklabels(classes)
    title_suffix = " (live pipeline incl. lung-crop bug — NOT the offline 96.57% figure)" if label == "X-ray" else ""
    ax.set_title(f"{label} per-class accuracy{title_suffix}", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.axhline(100, color="gray", linewidth=0.5)
    fig.tight_layout()
    out = f"chart_{label.lower().replace('-', '')}_accuracy.png"
    fig.savefig(out, dpi=150)
    print("saved", out)

print(json.dumps({f"{l}/{c}": {"dataset_tp_n": dataset_n[(l, c)]} for l in CLASSES for c in CLASSES[l]}, indent=2))
