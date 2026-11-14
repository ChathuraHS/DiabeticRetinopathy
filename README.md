# Diabetic Retinopathy Grading

An end-to-end research project for grading diabetic retinopathy from retinal fundus photographs. It includes image auditing, patient-isolated dataset splitting, image preprocessing, exploratory analysis, transfer-learning experiments, and test-set evaluation for five ordinal grades.

> **Research project only.** This model is not a medical device and must not be used to diagnose patients or guide treatment. Results from one dataset are not evidence of clinical readiness.

## Results

The evaluation summaries currently present in the local, Git-ignored `reports/` folder contain the following test-set results. Evaluation uses four-flip test-time augmentation by default; the tuned-threshold QWK is shown alongside QWK with the default grade thresholds.

| Model | Accuracy | Macro F1 | QWK (default thresholds) | QWK (validation-tuned thresholds) |
|---|---:|---:|---:|---:|
| ResNet18 | 0.7687 | 0.5080 | 0.7991 | 0.8114 |
| EfficientNet-B0 | 0.8144 | 0.5767 | 0.8215 | 0.8239 |

These are the values currently recorded in the local `reports/*_metrics.csv` and `reports/model_comparison.csv`; they have not been independently reproduced from a clean checkout. The dataset is imbalanced, so accuracy alone is not sufficient to characterize performance. See the evaluation script for balanced accuracy, per-class precision/recall/F1, and confusion matrices.

## Project workflow

1. **Audit:** `scripts/01_audit.py` checks the label CSV, image presence, image readability, and dimensions. It writes an audited-image manifest and failure report.
2. **Split:** `scripts/02_split.py` creates stratified train/validation/test splits with `StratifiedGroupKFold`, grouping left and right eye images by patient to prevent patient overlap between splits.
3. **Preprocess:** `scripts/03_preprocess.py` applies EXIF orientation, RGB conversion, black-border cropping, aspect-ratio-preserving resize and black padding to **640×640** JPEGs (quality 95). It writes metadata and failure reports.
4. **Validate:** `scripts/04_validate.py` checks processed images, labels, split coverage, and patient isolation. **Known issue:** this script currently expects 512×512 images, while preprocessing produces 640×640, so it will fail on correctly processed images until those dimensions are aligned.
5. **Explore and compare:** `notebooks/01_eda.ipynb` contains exploratory analysis, and `notebooks/02_model_comparison.ipynb` summarizes model results.
6. **Train and evaluate:** `src/train.py` trains either a pretrained ResNet18 or EfficientNet-B0; `src/evaluate.py` evaluates a saved checkpoint and writes metrics, predictions, and confusion-matrix plots.

The training pipeline treats grades 0–4 as an ordinal regression target, uses a Smooth L1 loss and softened class-balanced sampling, tunes hyperparameters with Optuna, and selects grade thresholds on the validation set using quadratic weighted kappa (QWK). CUDA mixed precision is used when available. Evaluation uses four horizontal/vertical flip views by default; pass `--no-tta` to disable test-time augmentation.

## Data

The dataset and downloaded archives are not included. Obtain the fundus images and labels from an authorized source and follow its access and redistribution terms. Do not commit the source images, labels, or downloaded archives.

Expected layout:

```text
data/
  raw/
    trainLabels.csv
  extracted/
    train/
      <image-id>.jpeg
```

`trainLabels.csv` must contain `image` and `level` columns; `level` is an integer from 0 to 4. The audit script supports `.jpeg`, `.jpg`, and `.png` images. Image IDs must end in `_left` or `_right` for the split script to derive patient IDs.

The scripts create split manifests under `data/splits/` and processed images under `data/processed/images/`. Audit, preprocessing, evaluation, and comparison outputs are written to `reports/`. These are generated/local artifacts and are ignored by Git.

## Setup and running

Use Python 3 and install PyTorch and torchvision for your hardware using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Install the remaining packages:

```bash
python -m pip install pandas Pillow tqdm scikit-learn matplotlib opencv-python numpy optuna seaborn
```

There is no dependency lockfile yet. From the repository root, run the scripts in order:

```bash
python scripts/01_audit.py
python scripts/02_split.py
python scripts/03_preprocess.py
python scripts/04_validate.py
```

After preparing the data, train and evaluate a model:

```bash
python src/train.py --model efficientnet_b0
python src/evaluate.py --model efficientnet_b0
```

Use `resnet18` instead of `efficientnet_b0` to run the other model. Training checkpoints and Optuna results are written under `models/<model-name>/`.

**Before running from a clone:** the scripts currently hard-code `D:\DiabeticRetinopathy` as the project directory. Update the `PROJECT` value in the scripts and in `src/train.py` and `src/evaluate.py` to your checkout location. The manifests generated by the audit and preprocessing stages also contain local paths; regenerate them on your machine rather than reusing someone else's manifests.

## Repository contents

```text
preprocessing_config.json  Image preprocessing settings
notebooks/                 EDA and model-comparison notebooks
scripts/                   Audit, split, preprocess, and validation stages
src/                       Dataset loading, training, and evaluation
data/                      Local dataset, manifests, and processed images (ignored)
models/                    Local checkpoints and tuning outputs (ignored)
reports/                   Generated reports, metrics, and figures (ignored)
downloads/                 Local source archives (ignored)
```

## Model files and GitHub

Model checkpoints (`.pth`) are not committed: they are generated by the training script, can be tens of megabytes, and are better distributed separately from source history. To share a trained model, publish a versioned GitHub Release asset or a model-hosting page, include the evaluation details and a checksum, and link to it from this README. Git LFS is another option if the repository explicitly needs versioned model files and its storage/bandwidth limits are understood. Do not upload private, restricted, or patient-identifiable data.

For a recruiter-facing project, keep the repository reproducible and easy to scan: explain the problem and your approach, show honest held-out results, include a few useful visualizations or a demo, document setup and known limitations, and link to any hosted model or short walkthrough. Avoid checking in large raw/generated artifacts; make the important result summaries visible in the README.
