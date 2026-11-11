# Diabetic Retinopathy Data Preparation

A Python pipeline for auditing diabetic retinopathy fundus images, creating patient-isolated train/validation/test splits, preprocessing images, and generating validation and exploratory reports.

## Pipeline

Run the scripts in order:

1. `scripts/01_audit.py` checks the labels and image files, then writes audit reports.
2. `scripts/02_split.py` creates stratified splits while keeping each patient's images in a single split.
3. `scripts/03_preprocess.py` corrects image orientation, converts images to RGB, resizes with aspect ratio preserved, and saves padded JPEGs.
4. `scripts/04_validate.py` checks processed images, labels, splits, and patient isolation.
5. `scripts/05_eda.py` generates class-distribution and image-size plots.

## Data

The dataset is not included in this repository. Place the source files in this layout:

```text
data/
  raw/trainLabels.csv
  extracted/train/<image files>
```

The label CSV must contain `image` and `level` columns. Image levels must be integers from 0 to 4. Supported image extensions are `.jpeg`, `.jpg`, and `.png`.

The scripts write split CSVs and processed images under `data/`, and audit, preprocessing, and EDA outputs under `reports/`. These generated data files are excluded from Git.

## Setup and Usage

Install Python dependencies:

```bash
python -m pip install pandas Pillow tqdm scikit-learn matplotlib
```

The scripts currently set the project directory to `D:\DiabeticRetinopathy`. If you clone the repository elsewhere, update the `PROJECT` path in each script before running it. Then run each script from the repository root in the order listed above, for example:

```bash
python scripts/01_audit.py
```

## Known Validation Mismatch

`scripts/03_preprocess.py` generates 1024×1024 images, but `scripts/04_validate.py` currently checks for 512×512 images. Align these dimensions before running the validation script.