# Where to get the external probe images

`test_external.py` needs chest images from sources **outside** the three
datasets this project trained on:

- COVID-19 Radiography Database (Kaggle)
- Tuberculosis (TB) Chest X-ray Database (Kaggle)
- IQ-OTH/NCCD Lung Cancer Dataset (Kaggle / Mendeley)

I have deliberately not shipped images in this folder. Chest radiographs found
online are frequently mislabelled, and a probe set with wrong labels produces a
confidently wrong diagnosis — worse than no diagnosis. You need to source them
yourself and be sure of the label.

## Target

| Modality | Class | Aim for |
|---|---|---|
| X-ray | Normal | 5 |
| X-ray | Pneumonia | 5 |
| X-ray | COVID-19 | 5 |
| X-ray | Tuberculosis | 5 |
| CT | Normal | 5 |
| CT | Benign | 5 |
| CT | Malignant | 5 |

Twenty X-rays and fifteen CT slices is enough to see a large effect. It is not
enough to quote a percentage as a result — say "on a 20-image external probe"
whenever you report it.

## Sources worth using

**NIH ChestX-ray14** — 112k frontal radiographs from a US hospital, public,
with labels. Different institution and equipment from your COVID/TB sets, which
is exactly the point. Good for Normal and Pneumonia. Search for the NIH Clinical
Center chest X-ray release.

**CheXpert (Stanford)** — registration required, but a genuinely different
institution. Good for Normal and Pneumonia.

**PadChest (Univ. of Alicante)** — Spanish hospital, very different acquisition
characteristics. Strong contrast with your training sources.

**LIDC-IDRI** — the standard public lung-nodule CT collection, via TCIA. Best
option for Benign and Malignant CT. Slices need extracting from DICOM; note the
window/level you use.

**Radiopaedia** — individual teaching cases with radiologist-written findings,
so labels are trustworthy. CC BY-NC-SA: fine for a course project with
attribution, but record the case ID and licence for each image you take. Best
option for Tuberculosis and COVID-19, which are hard to find elsewhere.

**Wikimedia Commons** — a small number of CC-licensed chest radiographs with
described findings. Check the description page for the actual diagnosis rather
than trusting the filename.

## Sourcing rules

1. **Confirm the label from the source's own text**, not from the filename or
   the page title. If the source does not state a diagnosis, do not use it.
2. **Frontal chest radiographs only** for X-ray. No lateral views, no abdominal
   or limb films — the classifier was never trained on those and rejecting them
   proves nothing.
3. **Axial lung-window CT slices only** for CT. A mediastinal-window slice is a
   different input distribution again.
4. **No screenshots of images.** Save the image file itself. A screenshot adds
   UI chrome and recompression that will confound the result.
5. **Record provenance.** Fill in `PROVENANCE.csv` as you go — filename, source
   URL, stated diagnosis, licence. You will need this if anyone asks where the
   evaluation images came from, and it belongs in the report.
6. **Run `verify_provenance.py` afterwards** even so. If any image happens to
   also live in one of the Kaggle sets, this catches it.

## Layout

```
test/external_probe/
├── xray/
│   ├── covid/         COVID-19.png ...
│   ├── normal/
│   ├── pneumonia/
│   └── tuberculosis/
└── ct/
    ├── benign/
    ├── malignant/
    └── normal/
```

Folder names are used as the ground-truth labels and are matched
case-insensitively against the model's class names. If your class names differ
(check `/health` or a prediction response), rename the folders to match.

## If you have no time for this

Run `test_corruption.py` on its own. It uses your existing test split, needs no
external data, and will tell you whether the failure is a preprocessing
sensitivity — which is the cheapest thing to fix and the most likely cause.
